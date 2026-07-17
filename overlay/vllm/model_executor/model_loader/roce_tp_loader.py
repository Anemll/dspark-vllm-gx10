# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: 2026 Anemll contributors
"""Two-rank, rank-0-only checkpoint loading over the TP NCCL group.

The reader rank opens the checkpoint normally.  The receiver rank executes its
existing model-specific weight loaders against a storage-free tensor subclass.
Those loaders therefore describe the exact source views needed by TP rank 1,
including expert ownership, packed dtypes, padding, and fused parameters.  Rank
0 evaluates the descriptions against the current checkpoint tensor and packs
only the resulting bytes for transport over the already-initialized TP group.

This is deliberately a startup model loader, not a storage or checkpoint
format.  Rank 1 still needs model metadata, tokenizer files, and configuration;
it does not open checkpoint payload files while loading model weights.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import time
import uuid
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.distributed import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.tracing import instrument

logger = init_logger(__name__)

_READER_RANK = 0
_RECEIVER_RANK = 1
_REMOTE_EXPR_TAG = "__vllm_roce_remote_expr__"
_SUPPORTED_SOURCE_FORMATS = {"auto", "hf", "pt", "safetensors"}
_LOAD_SEQUENCE = itertools.count()
_LOAD_RUN_ID = uuid.uuid4().hex


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Unknown torch dtype in RoCE weight stream: {name!r}")
    return dtype


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _iter_remote_tensors(value: Any) -> Generator["_RemoteTensor", None, None]:
    if isinstance(value, _RemoteTensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_remote_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_remote_tensors(item)


def _tree_map(value: Any, fn) -> Any:
    if isinstance(value, tuple):
        return tuple(_tree_map(item, fn) for item in value)
    if isinstance(value, list):
        return [_tree_map(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _tree_map(item, fn) for key, item in value.items()}
    return fn(value)


def _encode_operation_arg(value: Any) -> Any:
    if isinstance(value, _RemoteTensor):
        return (_REMOTE_EXPR_TAG, value.expression)
    if isinstance(value, torch.Tensor):
        raise NotImplementedError(
            "RoCE TP startup loading cannot defer an operation that combines "
            "checkpoint data with a materialized tensor. The model's weight "
            "loader must slice/reshape the checkpoint tensor before copy_."
        )
    if isinstance(value, tuple):
        return tuple(_encode_operation_arg(item) for item in value)
    if isinstance(value, list):
        return [_encode_operation_arg(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode_operation_arg(item) for key, item in value.items()}
    return value


def _resolve_torch_op(name: str):
    parts = name.split(".")
    if len(parts) != 3:
        raise ValueError(f"Unexpected torch operation name in RoCE recipe: {name!r}")
    namespace, packet_name, overload_name = parts
    packet = getattr(getattr(torch.ops, namespace), packet_name)
    return getattr(packet, overload_name)


def _evaluate_expression(expression: tuple, source: torch.Tensor) -> Any:
    kind = expression[0]
    if kind == "input":
        return source
    if kind == "getitem":
        value = _evaluate_expression(expression[1], source)
        for index in expression[2]:
            value = value[index]
        if not isinstance(value, torch.Tensor):
            raise TypeError("RoCE recipe selected a non-tensor operation result")
        return value
    if kind != "op":
        raise ValueError(f"Unknown RoCE recipe expression: {kind!r}")

    def decode(value: Any) -> Any:
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and value[0] == _REMOTE_EXPR_TAG
        ):
            return _evaluate_expression(value[1], source)
        if isinstance(value, tuple):
            return tuple(decode(item) for item in value)
        if isinstance(value, list):
            return [decode(item) for item in value]
        if isinstance(value, dict):
            return {key: decode(item) for key, item in value.items()}
        return value

    args = decode(expression[2])
    kwargs = decode(expression[3])
    result = _resolve_torch_op(expression[1])(*args, **kwargs)
    return result


@dataclass
class _CapturedWrite:
    destination: torch.Tensor
    expression: tuple
    source_shape: tuple[int, ...]
    source_dtype: torch.dtype
    nbytes: int

    def specification(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "shape": self.source_shape,
            "dtype": _dtype_name(self.source_dtype),
            "nbytes": self.nbytes,
        }


class _WriteRecorder:
    def __init__(self) -> None:
        self.writes: list[_CapturedWrite] = []

    def capture(self, destination: torch.Tensor, source: "_RemoteTensor") -> None:
        nbytes = int(source.numel()) * int(source.element_size())
        if nbytes == 0:
            return
        self.writes.append(
            _CapturedWrite(
                destination=destination,
                expression=source.expression,
                source_shape=tuple(source.shape),
                source_dtype=source.dtype,
                nbytes=nbytes,
            )
        )


class _RemoteTensor(torch.Tensor):
    """A tensor with metadata and an operation recipe, but no payload storage."""

    @staticmethod
    def __new__(
        cls,
        meta_tensor: torch.Tensor,
        expression: tuple,
        recorder: _WriteRecorder,
        logical_device: torch.device,
    ) -> "_RemoteTensor":
        result = torch.Tensor._make_wrapper_subclass(
            cls,
            meta_tensor.shape,
            strides=meta_tensor.stride(),
            storage_offset=meta_tensor.storage_offset(),
            dtype=meta_tensor.dtype,
            layout=meta_tensor.layout,
            device=logical_device,
            requires_grad=False,
        )
        result.meta_tensor = meta_tensor
        result.expression = expression
        result.recorder = recorder
        result.logical_device = logical_device
        return result

    def __repr__(self) -> str:
        return (
            "RemoteCheckpointTensor("
            f"shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.logical_device})"
        )

    @classmethod
    def __torch_dispatch__(
        cls,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        del types
        kwargs = {} if kwargs is None else kwargs
        remotes = list(_iter_remote_tensors((args, kwargs)))
        if not remotes:
            return func(*args, **kwargs)

        recorder = remotes[0].recorder
        if any(remote.recorder is not recorder for remote in remotes):
            raise RuntimeError("Cannot combine tensors from different RoCE recipes")

        if func is torch.ops.aten.copy_.default:
            destination, source = args[:2]
            if isinstance(source, _RemoteTensor) and not isinstance(
                destination, _RemoteTensor
            ):
                recorder.capture(destination, source)
                return destination

        if func is torch.ops.aten._local_scalar_dense.default:
            raise NotImplementedError(
                "RoCE TP startup loading cannot inspect checkpoint values on "
                "rank 1 before copy_."
            )

        encoded_args = _encode_operation_arg(args)
        encoded_kwargs = _encode_operation_arg(kwargs)
        expression = ("op", str(func), encoded_args, encoded_kwargs)

        def unwrap(value: Any) -> Any:
            if isinstance(value, _RemoteTensor):
                return value.meta_tensor
            if isinstance(value, torch.Tensor):
                raise NotImplementedError(
                    "RoCE TP startup loading cannot combine remote checkpoint "
                    "data with a materialized tensor before copy_."
                )
            return value

        meta_args = _tree_map(args, unwrap)
        meta_kwargs = _tree_map(kwargs, unwrap)

        logical_device = remotes[0].logical_device
        if func is torch.ops.aten._to_copy.default:
            source_meta = remotes[0].meta_tensor
            dtype = meta_kwargs.get("dtype") or source_meta.dtype
            meta_result = torch.empty_strided(
                tuple(source_meta.shape),
                tuple(source_meta.stride()),
                dtype=dtype,
                device="meta",
            )
            requested_device = kwargs.get("device")
            if requested_device is not None:
                logical_device = torch.device(requested_device)
        else:
            meta_result = func(*meta_args, **meta_kwargs)

        def wrap_result(value: Any, path: tuple[int, ...] = ()) -> Any:
            if isinstance(value, torch.Tensor):
                value_expression = (
                    expression if not path else ("getitem", expression, path)
                )
                return _RemoteTensor(
                    value,
                    value_expression,
                    recorder,
                    logical_device,
                )
            if isinstance(value, tuple):
                return tuple(
                    wrap_result(item, (*path, index))
                    for index, item in enumerate(value)
                )
            if isinstance(value, list):
                return [
                    wrap_result(item, (*path, index))
                    for index, item in enumerate(value)
                ]
            return value

        return wrap_result(meta_result)


class _RoCEWeightSender:
    def __init__(self, model: nn.Module, buffer_size_bytes: int) -> None:
        self.group = get_tp_group()
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise RuntimeError(
                "RoCE TP weight loading requires CUDA model parameters, "
                f"got {self.device}"
            )
        self.buffer_size_bytes = buffer_size_bytes
        self.pending: list[torch.Tensor] = []
        self.pending_bytes = 0
        self.source_bytes = 0
        self.sent_bytes = 0
        self.batch_count = 0
        self.tensor_count = 0

    def _flush(self) -> None:
        if not self.pending:
            return
        count = len(self.pending)
        total_bytes = self.pending_bytes
        packed = torch.empty(total_bytes, dtype=torch.uint8, device=self.device)
        offset = 0
        for payload in self.pending:
            raw = payload.contiguous().view(torch.uint8).reshape(-1)
            nbytes = int(raw.numel())
            packed[offset : offset + nbytes].copy_(raw)
            offset += nbytes
        if offset != total_bytes:
            raise RuntimeError(
                f"RoCE packed-byte mismatch: packed={offset}, expected={total_bytes}"
            )

        # Pack before announcing the receive. If allocation or copy fails, the
        # peer is still waiting for a control object instead of a tensor that
        # will never be sent.
        self.group.send_object(("flush", count, total_bytes), dst=_RECEIVER_RANK)
        self.group.send(packed, dst=_RECEIVER_RANK)
        self.sent_bytes += total_bytes
        self.batch_count += 1
        self.pending.clear()
        self.pending_bytes = 0

    def _queue_writes(
        self,
        source: torch.Tensor,
        specifications: list[dict[str, Any]],
    ) -> None:
        current: list[torch.Tensor] = []
        current_bytes = 0
        for spec in specifications:
            payload = _evaluate_expression(spec["expression"], source)
            if not isinstance(payload, torch.Tensor):
                raise TypeError(
                    "RoCE recipe produced a non-tensor payload: "
                    f"{type(payload).__name__}"
                )
            expected_shape = tuple(spec["shape"])
            expected_dtype = _dtype_from_name(spec["dtype"])
            expected_nbytes = int(spec["nbytes"])
            if tuple(payload.shape) != expected_shape:
                raise RuntimeError(
                    "RoCE recipe shape mismatch: "
                    f"evaluated={tuple(payload.shape)}, expected={expected_shape}"
                )
            if payload.dtype != expected_dtype:
                raise RuntimeError(
                    "RoCE recipe dtype mismatch: "
                    f"evaluated={payload.dtype}, expected={expected_dtype}"
                )
            if _tensor_nbytes(payload) != expected_nbytes:
                raise RuntimeError(
                    "RoCE recipe byte-count mismatch: "
                    f"evaluated={_tensor_nbytes(payload)}, expected={expected_nbytes}"
                )
            current.append(payload)
            current_bytes += expected_nbytes

        if self.pending and self.pending_bytes + current_bytes > self.buffer_size_bytes:
            self._flush()
        self.pending.extend(current)
        self.pending_bytes += current_bytes
        if self.pending_bytes >= self.buffer_size_bytes:
            self._flush()

    def iter_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        for name, source in weights:
            self.source_bytes += _tensor_nbytes(source)
            self.group.send_object(
                (
                    "tensor",
                    name,
                    tuple(source.shape),
                    _dtype_name(source.dtype),
                    str(source.device),
                ),
                dst=_RECEIVER_RANK,
            )
            yield name, source

            message = self.group.recv_object(src=_RECEIVER_RANK)
            if not isinstance(message, tuple) or not message:
                raise RuntimeError(f"Invalid RoCE receiver message: {message!r}")
            if message[0] == "error":
                raise RuntimeError(f"RoCE receiver failed: {message[1]}")
            if message[0] != "writes":
                raise RuntimeError(f"Unexpected RoCE receiver message: {message!r}")
            self._queue_writes(source, message[1])
            self.tensor_count += 1

        self._flush()
        self.group.send_object(("end", self.source_bytes), dst=_RECEIVER_RANK)


class _RoCEWeightReceiver:
    def __init__(self) -> None:
        self.group = get_tp_group()
        self.pending: list[_CapturedWrite] = []
        self.source_bytes = 0
        self.received_bytes = 0
        self.batch_count = 0
        self.tensor_count = 0

    def _receive_flush(self, count: int, total_bytes: int) -> None:
        if count <= 0 or count > len(self.pending):
            raise RuntimeError(
                f"Invalid RoCE flush count {count}; pending writes={len(self.pending)}"
            )
        writes = self.pending[:count]
        del self.pending[:count]
        expected_bytes = sum(write.nbytes for write in writes)
        if expected_bytes != total_bytes:
            raise RuntimeError(
                "RoCE flush byte-count mismatch: "
                f"receiver={expected_bytes}, sender={total_bytes}"
            )

        packed = self.group.recv(
            torch.Size((total_bytes,)), torch.uint8, src=_READER_RANK
        )
        offset = 0
        for write in writes:
            raw = packed[offset : offset + write.nbytes]
            payload = raw.view(write.source_dtype).view(write.source_shape)
            write.destination.copy_(payload)
            offset += write.nbytes
        if offset != total_bytes:
            raise RuntimeError(
                "RoCE unpacked-byte mismatch: "
                f"unpacked={offset}, expected={total_bytes}"
            )
        self.received_bytes += total_bytes
        self.batch_count += 1

    def iter_weights(self) -> Generator[tuple[str, torch.Tensor], None, None]:
        while True:
            message = self.group.recv_object(src=_READER_RANK)
            if not isinstance(message, tuple) or not message:
                raise RuntimeError(f"Invalid RoCE sender message: {message!r}")
            kind = message[0]
            if kind == "tensor":
                _, name, shape, dtype_name, device_name = message
                dtype = _dtype_from_name(dtype_name)
                recorder = _WriteRecorder()
                meta_tensor = torch.empty(tuple(shape), dtype=dtype, device="meta")
                remote = _RemoteTensor(
                    meta_tensor,
                    ("input",),
                    recorder,
                    torch.device(device_name),
                )
                yield name, remote
                self.pending.extend(recorder.writes)
                self.group.send_object(
                    (
                        "writes",
                        [write.specification() for write in recorder.writes],
                    ),
                    dst=_READER_RANK,
                )
                self.tensor_count += 1
            elif kind == "flush":
                _, count, total_bytes = message
                self._receive_flush(int(count), int(total_bytes))
            elif kind == "end":
                if self.pending:
                    raise RuntimeError(
                        f"RoCE sender ended with {len(self.pending)} unapplied writes"
                    )
                if len(message) == 1:
                    # Accept the original protocol during reversible image
                    # transitions; exact source-byte telemetry is unavailable.
                    self.source_bytes = 0
                    return
                if len(message) != 2 or (
                    isinstance(message[1], bool)
                    or not isinstance(message[1], int)
                    or message[1] < 0
                ):
                    raise RuntimeError(f"Invalid RoCE end message: {message!r}")
                self.source_bytes = message[1]
                return
            elif kind == "abort":
                raise RuntimeError(f"RoCE sender aborted weight loading: {message[1]}")
            else:
                raise RuntimeError(f"Unexpected RoCE sender message: {message!r}")


class RoCETPModelLoader(DefaultModelLoader):
    """Load a TP=2 model with one checkpoint reader and one RoCE receiver."""

    def __init__(self, load_config: LoadConfig):
        # DefaultModelLoader rejects extension-specific keys. Initialize the
        # common base directly, then retain the DefaultModelLoader iterator and
        # source handling methods through inheritance.
        BaseModelLoader.__init__(self, load_config)
        self.local_expert_ids: set[int] | None = None

        extra_config = load_config.model_loader_extra_config
        if not isinstance(extra_config, dict):
            raise ValueError(
                "model_loader_extra_config must be a dict for load format roce_tp"
            )
        allowed_keys = {
            "buffer_size_mb",
            "enable_multithread_load",
            "enable_weights_track",
            "num_threads",
            "source_load_format",
        }
        unexpected_keys = set(extra_config) - allowed_keys
        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format roce_tp: "
                f"{unexpected_keys}"
            )

        buffer_size_mb = extra_config.get("buffer_size_mb", 256)
        if (
            isinstance(buffer_size_mb, bool)
            or not isinstance(buffer_size_mb, int)
            or buffer_size_mb <= 0
        ):
            raise ValueError(
                f"buffer_size_mb must be a positive integer, got {buffer_size_mb!r}"
            )
        self.buffer_size_bytes = buffer_size_mb * 1024 * 1024

        self.source_load_format = str(
            extra_config.get("source_load_format", "auto")
        ).lower()
        if self.source_load_format not in _SUPPORTED_SOURCE_FORMATS:
            raise ValueError(
                "source_load_format for roce_tp must be one of "
                f"{sorted(_SUPPORTED_SOURCE_FORMATS)}, got "
                f"{self.source_load_format!r}"
            )

        enable_multithread = extra_config.get("enable_multithread_load", False)
        if not isinstance(enable_multithread, bool):
            raise ValueError(
                "enable_multithread_load must be a bool, got "
                f"{type(enable_multithread).__name__}"
            )
        num_threads = extra_config.get("num_threads", self.DEFAULT_NUM_THREADS)
        if (
            isinstance(num_threads, bool)
            or not isinstance(num_threads, int)
            or num_threads <= 0
        ):
            raise ValueError(
                f"num_threads must be a positive integer, got {num_threads!r}"
            )
        self.enable_weights_track = extra_config.get("enable_weights_track")
        if self.enable_weights_track is not None and not isinstance(
            self.enable_weights_track, bool
        ):
            raise ValueError(
                "enable_weights_track must be a bool or null, got "
                f"{type(self.enable_weights_track).__name__}"
            )

    @contextlib.contextmanager
    def _using_source_load_format(self):
        original = self.load_config.load_format
        self.load_config.load_format = self.source_load_format
        try:
            yield
        finally:
            self.load_config.load_format = original

    def _prepare_weights(self, *args, **kwargs):
        with self._using_source_load_format():
            return super()._prepare_weights(*args, **kwargs)

    def _get_weights_iterator(
        self, source: DefaultModelLoader.Source
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        with self._using_source_load_format():
            yield from super()._get_weights_iterator(source)

    def download_model(self, model_config: ModelConfig) -> None:
        node_rank = int(os.getenv("NODE_RANK", "0"))
        if node_rank != _READER_RANK:
            logger.info(
                "RoCE TP loader: NODE_RANK=%d skips checkpoint payload preparation",
                node_rank,
            )
            return
        super().download_model(model_config)

    def _validate_topology(self) -> int:
        if not torch.distributed.is_initialized():
            raise RuntimeError("RoCE TP loader requires initialized distributed groups")
        tp_group = get_tp_group()
        if tp_group.world_size != 2:
            raise ValueError(
                f"RoCE TP loader currently requires tensor_parallel_size=2, "
                f"got {tp_group.world_size}"
            )
        if get_pp_group().world_size != 1:
            raise ValueError(
                "RoCE TP loader currently requires pipeline_parallel_size=1"
            )
        return int(tp_group.rank_in_group)

    @instrument(span_name="Load weights via RoCE TP")
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        if model_config.quantization == "torchao":
            raise ValueError("RoCE TP startup loading does not support torchao loading")

        tp_rank = self._validate_topology()
        get_tp_group().barrier()
        started = time.perf_counter()
        load_id = next(_LOAD_SEQUENCE)
        phase = type(model).__name__
        role = "reader" if tp_rank == _READER_RANK else "receiver"
        logger.info(
            "DSPARK_WEIGHT_LOAD mode=roce_tp event=start run=%s pid=%d id=%d "
            "rank=%d role=%s phase=%s buffer_bytes=%d",
            _LOAD_RUN_ID,
            os.getpid(),
            load_id,
            tp_rank,
            role,
            phase,
            self.buffer_size_bytes,
        )
        if tp_rank == _READER_RANK:
            logger.info(
                "RoCE TP loader enabled: rank 0 is the sole checkpoint reader; "
                "rank-1 writes target %d MiB packed batches (one source "
                "tensor's writes may exceed the target)",
                self.buffer_size_bytes // (1024 * 1024),
            )
            try:
                sender = _RoCEWeightSender(model, self.buffer_size_bytes)
                loaded_weights = model.load_weights(
                    sender.iter_weights(self.get_all_weights(model_config, model))
                )
            except Exception as exc:
                logger.error(
                    "DSPARK_WEIGHT_LOAD mode=roce_tp event=failed run=%s "
                    "pid=%d id=%d rank=%d role=%s phase=%s error_type=%s",
                    _LOAD_RUN_ID,
                    os.getpid(),
                    load_id,
                    tp_rank,
                    role,
                    phase,
                    type(exc).__name__,
                )
                raise
            transferred_bytes = sender.sent_bytes
            source_bytes = sender.source_bytes
            batch_count = sender.batch_count
            tensor_count = sender.tensor_count
        else:
            logger.info_once(
                "RoCE TP loader enabled: rank 1 will not open checkpoint payload files"
            )
            receiver = _RoCEWeightReceiver()
            try:
                loaded_weights = model.load_weights(receiver.iter_weights())
            except Exception as exc:
                logger.error(
                    "DSPARK_WEIGHT_LOAD mode=roce_tp event=failed run=%s "
                    "pid=%d id=%d rank=%d role=%s phase=%s error_type=%s",
                    _LOAD_RUN_ID,
                    os.getpid(),
                    load_id,
                    tp_rank,
                    role,
                    phase,
                    type(exc).__name__,
                )
                # Let rank 0 fail with the receiver's original error instead of
                # waiting forever for a writes message.
                get_tp_group().send_object(("error", repr(exc)), dst=_READER_RANK)
                raise
            transferred_bytes = receiver.received_bytes
            source_bytes = receiver.source_bytes
            batch_count = receiver.batch_count
            tensor_count = receiver.tensor_count

        default_enable_weights_track = (
            model_config.quantization is None and loaded_weights is not None
        )
        enable_weights_track = (
            self.enable_weights_track
            if self.enable_weights_track is not None
            else default_enable_weights_track
        )

        # PyNccl enqueues send/receive and destination copies on the current
        # CUDA stream. Synchronize once per model phase so completion covers
        # tracking and actual RAM residency without serializing batches.
        try:
            if enable_weights_track:
                self.track_weights_loading(model, loaded_weights)
            torch.cuda.current_stream().synchronize()
        except Exception as exc:
            logger.error(
                "DSPARK_WEIGHT_LOAD mode=roce_tp event=failed run=%s pid=%d id=%d "
                "rank=%d role=%s phase=%s error_type=%s",
                _LOAD_RUN_ID,
                os.getpid(),
                load_id,
                tp_rank,
                role,
                phase,
                type(exc).__name__,
            )
            raise
        elapsed = time.perf_counter() - started
        logger.info(
            "DSPARK_WEIGHT_LOAD mode=roce_tp event=complete run=%s pid=%d id=%d "
            "rank=%d role=%s phase=%s tensors=%d batches=%d "
            "source_bytes=%d traffic_bytes=%d elapsed_s=%.6f",
            _LOAD_RUN_ID,
            os.getpid(),
            load_id,
            tp_rank,
            role,
            phase,
            tensor_count,
            batch_count,
            source_bytes,
            transferred_bytes,
            elapsed,
        )
        logger.info(
            "RoCE TP RAM weight load rank=%d complete: tensors=%d, batches=%d, "
            "traffic=%.2f GiB, elapsed=%.2f seconds",
            tp_rank,
            tensor_count,
            batch_count,
            transferred_bytes / (1024**3),
            elapsed,
        )



class TimedDefaultModelLoader(DefaultModelLoader):
    """Default local checkpoint loader with A/B-comparable RAM timing."""

    @contextlib.contextmanager
    def _using_default_source_format(self):
        # `direct_timed` is a wrapper format, not a checkpoint format understood
        # by DefaultModelLoader._prepare_weights(). Present the normal `auto`
        # source format only while the unchanged default loader runs.
        original = self.load_config.load_format
        self.load_config.load_format = "auto"
        try:
            yield
        finally:
            self.load_config.load_format = original

    def download_model(self, model_config: ModelConfig) -> None:
        with self._using_default_source_format():
            super().download_model(model_config)

    @instrument(span_name="Load weights locally with synchronized timing")
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        tp_group = get_tp_group()
        tp_rank = int(tp_group.rank_in_group)
        load_id = next(_LOAD_SEQUENCE)
        phase = type(model).__name__
        tp_group.barrier()
        started = time.perf_counter()
        logger.info(
            "DSPARK_WEIGHT_LOAD mode=direct event=start run=%s pid=%d id=%d "
            "rank=%d role=local_reader phase=%s",
            _LOAD_RUN_ID,
            os.getpid(),
            load_id,
            tp_rank,
            phase,
        )
        try:
            with self._using_default_source_format():
                super().load_weights(model, model_config)
            torch.cuda.current_stream().synchronize()
        except Exception as exc:
            logger.error(
                "DSPARK_WEIGHT_LOAD mode=direct event=failed run=%s pid=%d id=%d "
                "rank=%d role=local_reader phase=%s error_type=%s",
                _LOAD_RUN_ID,
                os.getpid(),
                load_id,
                tp_rank,
                phase,
                type(exc).__name__,
            )
            raise
        elapsed = time.perf_counter() - started
        logger.info(
            "DSPARK_WEIGHT_LOAD mode=direct event=complete run=%s pid=%d id=%d "
            "rank=%d role=local_reader phase=%s elapsed_s=%.6f",
            _LOAD_RUN_ID,
            os.getpid(),
            load_id,
            tp_rank,
            phase,
            elapsed,
        )
        logger.info(
            "Synchronized direct RAM weight load rank=%d complete: elapsed=%.2f seconds",
            tp_rank,
            elapsed,
        )


__all__ = ["RoCETPModelLoader", "TimedDefaultModelLoader"]
