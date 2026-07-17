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
import math
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
_PROTOCOL_VERSION = 2
_REMOTE_EXPR_TAG = "__vllm_roce_remote_expr__"
_SUPPORTED_SOURCE_FORMATS = {"auto", "hf", "pt", "safetensors"}
_LOAD_SEQUENCE = itertools.count()
_LOAD_RUN_ID = uuid.uuid4().hex


class _RoCEPeerAbortedError(RuntimeError):
    """The other TP rank already reported the failure; do not echo it back."""


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Unknown torch dtype in RoCE weight stream: {name!r}")
    return dtype


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _chunk_indices(
    shape: tuple[int, ...], max_elements: int
) -> Generator[tuple[slice, ...], None, None]:
    """Yield row-major logical slices whose element counts fit the limit.

    Slicing before copying is important for TP views that are not contiguous:
    calling ``contiguous()`` on the complete view would defeat the transport
    memory bound.  Every yielded index retains all dimensions so the same
    schedule can be used to scatter into a non-contiguous destination.
    """
    if max_elements <= 0:
        raise ValueError(f"max_elements must be positive, got {max_elements}")
    if any(dimension < 0 for dimension in shape):
        raise ValueError(f"Invalid negative tensor shape: {shape}")
    if math.prod(shape) == 0:
        return
    if not shape:
        yield ()
        return

    def visit(
        dimension: int, prefix: tuple[slice, ...]
    ) -> Generator[tuple[slice, ...], None, None]:
        remaining = math.prod(shape[dimension:])
        if remaining <= max_elements:
            yield prefix + (slice(None),) * (len(shape) - dimension)
            return

        tail = math.prod(shape[dimension + 1 :])
        if tail <= max_elements:
            step = max_elements // tail
            for start in range(0, shape[dimension], step):
                stop = min(start + step, shape[dimension])
                yield prefix + (slice(start, stop),) + (slice(None),) * (
                    len(shape) - dimension - 1
                )
            return

        for index in range(shape[dimension]):
            yield from visit(dimension + 1, prefix + (slice(index, index + 1),))

    yield from visit(0, ())


def _shape_after_index(
    shape: tuple[int, ...], index: tuple[slice, ...]
) -> tuple[int, ...]:
    if len(shape) != len(index):
        raise ValueError(f"Shape/index rank mismatch: shape={shape}, index={index}")
    result: list[int] = []
    for dimension, item in zip(shape, index, strict=True):
        start, stop, step = item.indices(dimension)
        if step != 1:
            raise ValueError(f"RoCE chunk slices must have unit stride, got {item}")
        result.append(max(0, stop - start))
    return tuple(result)


def _can_broadcast_to(
    source_shape: tuple[int, ...], destination_shape: tuple[int, ...]
) -> bool:
    if len(source_shape) > len(destination_shape):
        return False
    return all(
        source == 1 or source == destination
        for source, destination in zip(
            reversed(source_shape), reversed(destination_shape), strict=False
        )
    )


def _recv_into(group: Any, tensor: torch.Tensor, src: int) -> None:
    """Receive NCCL payload directly into caller-owned CUDA storage.

    GroupCoordinator.recv() allocates a fresh tensor.  The pinned vLLM CUDA
    communicator also exposes its grouped PyNccl path, which accepts a P2POp
    backed by an existing tensor.  Using it here lets rank 1 receive matching,
    contiguous writes directly into final model parameters.
    """
    if tensor.device.type != "cuda":
        raise ValueError(f"RoCE NCCL destination must be CUDA, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError("RoCE NCCL destination must be contiguous")
    device_communicator = group.device_communicator
    transport_mode = _device_transport_mode(group)
    if transport_mode == "unavailable":
        raise ValueError("RoCE TP group has no device communicator")
    if transport_mode == "pynccl":
        operation = object.__new__(torch.distributed.P2POp)
        operation.op = torch.distributed.irecv
        operation.tensor = tensor
        operation.group_peer = src
        device_communicator.batch_isend_irecv([operation])
    else:
        # Match CudaCommunicator.send's ProcessGroupNCCL fallback exactly.
        torch.distributed.recv(
            tensor,
            src=group.ranks[src],
            group=group.device_group,
        )


def _device_transport_mode(group: Any) -> str:
    device_communicator = getattr(group, "device_communicator", None)
    if device_communicator is None:
        return "unavailable"
    pynccl_communicator = getattr(device_communicator, "pynccl_comm", None)
    if pynccl_communicator is not None and not pynccl_communicator.disabled:
        return "pynccl"
    return "process_group"


def _negotiate_protocol(group: Any, rank: int, buffer_size_bytes: int) -> str:
    """Fail safely before data traffic if image/config/transport differ."""
    transport_mode = _device_transport_mode(group)
    local = (_PROTOCOL_VERSION, transport_mode, buffer_size_bytes)
    if rank == _READER_RANK:
        group.send_object(("hello", *local), dst=_RECEIVER_RANK)
        response = group.recv_object(src=_RECEIVER_RANK)
        if response != ("hello_ack", *local):
            raise RuntimeError(
                "RoCE TP protocol negotiation failed: "
                f"local={local!r}, receiver={response!r}"
            )
    else:
        request = group.recv_object(src=_READER_RANK)
        expected = ("hello", *local)
        if request != expected or transport_mode == "unavailable":
            error = (
                "hello_error",
                f"local={local!r}, reader={request!r}",
            )
            group.send_object(error, dst=_READER_RANK)
            raise RuntimeError(f"RoCE TP protocol negotiation failed: {error[1]}")
        group.send_object(("hello_ack", *local), dst=_READER_RANK)
    if transport_mode == "unavailable":
        raise RuntimeError("RoCE TP group has no usable device transport")
    return transport_mode


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

    def direct_eligible(self) -> bool:
        return (
            self.destination.layout == torch.strided
            and self.destination.device.type == "cuda"
            and self.destination.is_contiguous()
            and tuple(self.destination.shape) == self.source_shape
            and self.destination.dtype == self.source_dtype
        )

    def specification(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "shape": self.source_shape,
            "dtype": _dtype_name(self.source_dtype),
            "nbytes": self.nbytes,
            "direct": self.direct_eligible(),
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
    def __init__(
        self,
        model: nn.Module,
        buffer_size_bytes: int,
        release_watermark_bytes: int,
    ) -> None:
        self.group = get_tp_group()
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise RuntimeError(
                "RoCE TP weight loading requires CUDA model parameters, "
                f"got {self.device}"
            )
        self.buffer_size_bytes = buffer_size_bytes
        self.staging = torch.empty(
            buffer_size_bytes, dtype=torch.uint8, device=self.device
        )
        self.source_bytes = 0
        self.sent_bytes = 0
        self.batch_count = 0
        self.tensor_count = 0
        self.direct_bytes = 0
        self.staged_bytes = 0
        self.max_frame_bytes = 0
        self.max_write_bytes = 0
        self.release_watermark_bytes = release_watermark_bytes
        self.pending_release_bytes = 0
        self.max_pending_release_bytes = 0
        self.release_count = 0
        self.released_reserved_bytes = 0
        self.control_state = "receiver_control"

    def _release_completed(self, *, force: bool = False) -> None:
        """Drain queued sends and return completed recipe storage to CUDA.

        The fixed staging tensor bounds each transport frame, but rank-1
        recipes can still materialize a complete CUDA payload before it is
        framed.  PyNccl and the staging copies use the current stream, so a
        periodic stream drain makes those payloads reclaimable without
        serializing every tensor or frame.
        """
        if self.pending_release_bytes == 0:
            return
        if (
            not force
            and self.pending_release_bytes < self.release_watermark_bytes
        ):
            return
        torch.cuda.current_stream(self.device).synchronize()
        reserved_before = int(torch.cuda.memory_reserved(self.device))
        torch.cuda.empty_cache()
        reserved_after = int(torch.cuda.memory_reserved(self.device))
        self.release_count += 1
        self.released_reserved_bytes += max(0, reserved_before - reserved_after)
        logger.info(
            "RoCE TP sender reclaim: count=%d pending_bytes=%d "
            "watermark_bytes=%d reserved_before_bytes=%d "
            "reserved_after_bytes=%d released_reserved_total_bytes=%d",
            self.release_count,
            self.pending_release_bytes,
            self.release_watermark_bytes,
            reserved_before,
            reserved_after,
            self.released_reserved_bytes,
        )
        self.pending_release_bytes = 0

    def _validate_payload(
        self,
        payload: Any,
        spec: dict[str, Any],
    ) -> torch.Tensor:
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
        if payload.layout != torch.strided:
            raise NotImplementedError(
                "RoCE TP loading requires strided payload tensors, got "
                f"{payload.layout}"
            )
        return payload

    def _send_payload(
        self,
        payload: torch.Tensor,
        direct: bool,
        write_index: int,
    ) -> None:
        element_size = int(payload.element_size())
        max_elements = self.buffer_size_bytes // element_size
        if max_elements <= 0:
            raise ValueError(
                "RoCE buffer is smaller than one payload element: "
                f"buffer={self.buffer_size_bytes}, element={element_size}"
            )
        write_bytes = _tensor_nbytes(payload)
        self.max_write_bytes = max(self.max_write_bytes, write_bytes)
        for frame_index, index in enumerate(
            _chunk_indices(tuple(payload.shape), max_elements)
        ):
            chunk = payload[index]
            chunk_bytes = _tensor_nbytes(chunk)
            staging_bytes = self.staging[:chunk_bytes]
            staging_bytes.view(payload.dtype).view(chunk.shape).copy_(chunk)
            # Rank 1 posts ncclRecv only after the frame is prepared. Thus an
            # expression or staging-copy error can still be reported over the
            # Gloo control plane without stranding the peer in NCCL.
            self.group.send_object(
                ("frame", write_index, frame_index, chunk_bytes),
                dst=_RECEIVER_RANK,
            )
            self.control_state = "device_receive"
            # Transport raw bytes for every source dtype. PyNccl supports
            # uint8 universally, and this also preserves packed/FP8 bit
            # patterns without depending on its typed NCCL mapping.
            self.group.send(staging_bytes, dst=_RECEIVER_RANK)
            self.control_state = "frame_control"
            self.sent_bytes += chunk_bytes
            self.batch_count += 1
            self.max_frame_bytes = max(self.max_frame_bytes, chunk_bytes)
            if direct:
                self.direct_bytes += chunk_bytes
            else:
                self.staged_bytes += chunk_bytes

    def _send_writes(
        self,
        source: torch.Tensor,
        specifications: list[dict[str, Any]],
    ) -> None:
        for position, spec in enumerate(specifications):
            payload_bytes = int(spec["nbytes"])
            if (
                self.pending_release_bytes
                and self.pending_release_bytes + payload_bytes
                > self.release_watermark_bytes
            ):
                self._release_completed(force=True)
            payload = self._validate_payload(
                _evaluate_expression(spec["expression"], source), spec
            )
            try:
                self._send_payload(
                    payload,
                    bool(spec.get("direct", False)),
                    position,
                )
            finally:
                del payload
            self.pending_release_bytes += payload_bytes
            self.max_pending_release_bytes = max(
                self.max_pending_release_bytes,
                self.pending_release_bytes,
            )
            self._release_completed()

    def abort(self, error: BaseException) -> None:
        """Bring rank 1 back to the control plane, then report rank-0 failure."""
        if self.control_state in {"aborted", "ended", "peer_failed"}:
            return
        if self.control_state == "device_receive":
            # A device-transport failure after the frame marker cannot be
            # repaired with Gloo: rank 1 is already inside ncclRecv and should
            # observe the same communicator failure.
            logger.error(
                "RoCE sender cannot issue control abort while a device receive "
                "is outstanding"
            )
            return
        if self.control_state == "writes":
            message = self.group.recv_object(src=_RECEIVER_RANK)
            if isinstance(message, tuple) and message and message[0] == "error":
                self.control_state = "peer_failed"
                return
            self.control_state = "frame_control"
            if not isinstance(message, tuple) or not message or message[0] != "writes":
                logger.error(
                    "Unexpected RoCE receiver response while aborting: %r", message
                )
        self.group.send_object(("abort", repr(error)), dst=_RECEIVER_RANK)
        self.control_state = "aborted"

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
            self.control_state = "writes"
            yield name, source

            message = self.group.recv_object(src=_RECEIVER_RANK)
            self.control_state = "frame_control"
            if not isinstance(message, tuple) or not message:
                raise RuntimeError(f"Invalid RoCE receiver message: {message!r}")
            if message[0] == "error":
                self.control_state = "peer_failed"
                raise _RoCEPeerAbortedError(
                    f"RoCE receiver failed: {message[1]}"
                )
            if message[0] != "writes":
                raise RuntimeError(f"Unexpected RoCE receiver message: {message!r}")
            self.control_state = "frame_control"
            self._send_writes(source, message[1])
            self.control_state = "receiver_control"
            self.tensor_count += 1

        self._release_completed(force=True)
        self.group.send_object(("end", self.source_bytes), dst=_RECEIVER_RANK)
        self.control_state = "ended"


class _RoCEWeightReceiver:
    def __init__(self, model: nn.Module, buffer_size_bytes: int) -> None:
        self.group = get_tp_group()
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise RuntimeError(
                "RoCE TP weight loading requires CUDA model parameters, "
                f"got {self.device}"
            )
        self.buffer_size_bytes = buffer_size_bytes
        self.staging: torch.Tensor | None = None
        self.source_bytes = 0
        self.received_bytes = 0
        self.batch_count = 0
        self.tensor_count = 0
        self.direct_bytes = 0
        self.staged_bytes = 0
        self.max_frame_bytes = 0
        self.max_write_bytes = 0

    def _validate_write(self, write: _CapturedWrite) -> None:
        if write.destination.layout != torch.strided:
            raise NotImplementedError(
                "RoCE TP loading requires strided destinations, got "
                f"{write.destination.layout}"
            )
        if not _can_broadcast_to(
            write.source_shape, tuple(write.destination.shape)
        ):
            raise ValueError(
                "RoCE source shape cannot be copied into destination: "
                f"source={write.source_shape}, destination="
                f"{tuple(write.destination.shape)}"
            )
        if (
            tuple(write.destination.shape) != write.source_shape
            and write.nbytes > self.buffer_size_bytes
        ):
            raise NotImplementedError(
                "A shape-changing or broadcast RoCE write must fit one bounded "
                f"frame: write={write.nbytes}, frame={self.buffer_size_bytes}, "
                f"source={write.source_shape}, destination="
                f"{tuple(write.destination.shape)}"
            )
        if write.source_dtype.itemsize > self.buffer_size_bytes:
            raise ValueError(
                "RoCE buffer is smaller than one payload element: "
                f"buffer={self.buffer_size_bytes}, dtype={write.source_dtype}"
            )

    def _staging_bytes(self, byte_count: int) -> torch.Tensor:
        if byte_count > self.buffer_size_bytes:
            raise RuntimeError(
                f"RoCE frame exceeds hard bound: {byte_count} > "
                f"{self.buffer_size_bytes}"
            )
        if self.staging is None:
            self.staging = torch.empty(
                self.buffer_size_bytes, dtype=torch.uint8, device=self.device
            )
        return self.staging[:byte_count]

    def _receive_write(self, write: _CapturedWrite, write_index: int) -> None:
        element_size = write.source_dtype.itemsize
        max_elements = self.buffer_size_bytes // element_size
        direct = write.direct_eligible()
        same_shape = tuple(write.destination.shape) == write.source_shape
        direct_offset_bytes = 0
        self.max_write_bytes = max(self.max_write_bytes, write.nbytes)
        for frame_index, index in enumerate(
            _chunk_indices(write.source_shape, max_elements)
        ):
            chunk_shape = _shape_after_index(write.source_shape, index)
            elements = math.prod(chunk_shape)
            chunk_bytes = elements * element_size
            ready = self.group.recv_object(src=_READER_RANK)
            if not isinstance(ready, tuple) or not ready:
                raise RuntimeError(f"Invalid RoCE frame message: {ready!r}")
            if ready[0] == "abort":
                raise _RoCEPeerAbortedError(
                    f"RoCE sender aborted weight loading: {ready[1]}"
                )
            expected = ("frame", write_index, frame_index, chunk_bytes)
            if ready != expected:
                raise RuntimeError(
                    f"Unexpected RoCE frame message: {ready!r}, expected={expected!r}"
                )
            if direct:
                destination = write.destination.reshape(-1).view(torch.uint8)[
                    direct_offset_bytes : direct_offset_bytes + chunk_bytes
                ]
                direct_offset_bytes += chunk_bytes
                _recv_into(self.group, destination, src=_READER_RANK)
                self.direct_bytes += chunk_bytes
            else:
                staging_bytes = self._staging_bytes(chunk_bytes)
                _recv_into(self.group, staging_bytes, src=_READER_RANK)
                payload = staging_bytes.view(write.source_dtype).view(chunk_shape)
                if same_shape:
                    write.destination[index].copy_(payload)
                else:
                    write.destination.copy_(payload.view(write.source_shape))
                self.staged_bytes += chunk_bytes
            self.received_bytes += chunk_bytes
            self.batch_count += 1
            self.max_frame_bytes = max(self.max_frame_bytes, chunk_bytes)

        if direct and direct_offset_bytes != write.nbytes:
            raise RuntimeError(
                "RoCE direct receive byte mismatch: "
                f"received={direct_offset_bytes}, expected={write.nbytes}"
            )

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
                for write in recorder.writes:
                    self._validate_write(write)
                self.group.send_object(
                    (
                        "writes",
                        [write.specification() for write in recorder.writes],
                    ),
                    dst=_READER_RANK,
                )
                for position, write in enumerate(recorder.writes):
                    self._receive_write(write, position)
                self.tensor_count += 1
            elif kind == "end":
                if len(message) != 2 or (
                    isinstance(message[1], bool)
                    or not isinstance(message[1], int)
                    or message[1] < 0
                ):
                    raise RuntimeError(f"Invalid RoCE end message: {message!r}")
                self.source_bytes = message[1]
                return
            elif kind == "abort":
                raise _RoCEPeerAbortedError(
                    f"RoCE sender aborted weight loading: {message[1]}"
                )
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
            "release_watermark_mb",
            "source_load_format",
        }
        unexpected_keys = set(extra_config) - allowed_keys
        if unexpected_keys:
            raise ValueError(
                f"Unexpected extra config keys for load format roce_tp: "
                f"{unexpected_keys}"
            )

        buffer_size_mb = extra_config.get("buffer_size_mb", 64)
        if (
            isinstance(buffer_size_mb, bool)
            or not isinstance(buffer_size_mb, int)
            or buffer_size_mb <= 0
        ):
            raise ValueError(
                f"buffer_size_mb must be a positive integer, got {buffer_size_mb!r}"
            )
        self.buffer_size_bytes = buffer_size_mb * 1024 * 1024

        release_watermark_mb = extra_config.get("release_watermark_mb", 1024)
        if (
            isinstance(release_watermark_mb, bool)
            or not isinstance(release_watermark_mb, int)
            or release_watermark_mb < buffer_size_mb
        ):
            raise ValueError(
                "release_watermark_mb must be an integer greater than or equal "
                f"to buffer_size_mb ({buffer_size_mb}), got "
                f"{release_watermark_mb!r}"
            )
        self.release_watermark_bytes = release_watermark_mb * 1024 * 1024

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
        tp_group = get_tp_group()
        tp_group.barrier()
        transport_mode = _negotiate_protocol(
            tp_group, tp_rank, self.buffer_size_bytes
        )
        started = time.perf_counter()
        load_id = next(_LOAD_SEQUENCE)
        phase = type(model).__name__
        role = "reader" if tp_rank == _READER_RANK else "receiver"
        logger.info(
            "DSPARK_WEIGHT_LOAD mode=roce_tp event=start run=%s pid=%d id=%d "
            "rank=%d role=%s phase=%s buffer_bytes=%d "
            "release_watermark_bytes=%d protocol=%d transport=%s",
            _LOAD_RUN_ID,
            os.getpid(),
            load_id,
            tp_rank,
            role,
            phase,
            self.buffer_size_bytes,
            self.release_watermark_bytes,
            _PROTOCOL_VERSION,
            transport_mode,
        )
        if tp_rank == _READER_RANK:
            logger.info(
                "RoCE TP loader enabled: rank 0 is the sole checkpoint reader; "
                "rank-1 writes use a hard %d MiB frame cap and direct NCCL "
                "receive into eligible final parameters",
                self.buffer_size_bytes // (1024 * 1024),
            )
            sender: _RoCEWeightSender | None = None
            try:
                sender = _RoCEWeightSender(
                    model,
                    self.buffer_size_bytes,
                    self.release_watermark_bytes,
                )
                loaded_weights = model.load_weights(
                    sender.iter_weights(self.get_all_weights(model_config, model))
                )
            except Exception as exc:
                try:
                    if sender is None:
                        get_tp_group().send_object(
                            ("abort", repr(exc)), dst=_RECEIVER_RANK
                        )
                    else:
                        sender.abort(exc)
                except Exception as abort_exc:
                    logger.error(
                        "RoCE rank-0 abort handshake failed: %s",
                        type(abort_exc).__name__,
                    )
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
            assert sender is not None
            transferred_bytes = sender.sent_bytes
            source_bytes = sender.source_bytes
            batch_count = sender.batch_count
            tensor_count = sender.tensor_count
            direct_bytes = sender.direct_bytes
            staged_bytes = sender.staged_bytes
            max_frame_bytes = sender.max_frame_bytes
            max_write_bytes = sender.max_write_bytes
            release_count = sender.release_count
            max_pending_release_bytes = sender.max_pending_release_bytes
            released_reserved_bytes = sender.released_reserved_bytes
        else:
            logger.info_once(
                "RoCE TP loader enabled: rank 1 will not open checkpoint payload files"
            )
            receiver: _RoCEWeightReceiver | None = None
            try:
                receiver = _RoCEWeightReceiver(model, self.buffer_size_bytes)
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
                # Let rank 0 fail with a receiver-originated error instead of
                # waiting forever for a writes message. A sender-originated
                # abort has already crossed the control plane, so echoing it
                # would block after rank 0 has begun unwinding.
                if not isinstance(exc, _RoCEPeerAbortedError):
                    get_tp_group().send_object(
                        ("error", repr(exc)), dst=_READER_RANK
                    )
                raise
            assert receiver is not None
            transferred_bytes = receiver.received_bytes
            source_bytes = receiver.source_bytes
            batch_count = receiver.batch_count
            tensor_count = receiver.tensor_count
            direct_bytes = receiver.direct_bytes
            staged_bytes = receiver.staged_bytes
            max_frame_bytes = receiver.max_frame_bytes
            max_write_bytes = receiver.max_write_bytes
            release_count = 0
            max_pending_release_bytes = 0
            released_reserved_bytes = 0

        default_enable_weights_track = (
            model_config.quantization is None and loaded_weights is not None
        )
        enable_weights_track = (
            self.enable_weights_track
            if self.enable_weights_track is not None
            else default_enable_weights_track
        )

        # Periodic sender drains bound temporary recipe lifetime. Keep this
        # final synchronization on both ranks so the timer covers actual RAM
        # residency, including destination copies after the last payload.
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
            "source_bytes=%d traffic_bytes=%d direct_bytes=%d staged_bytes=%d "
            "max_frame_bytes=%d max_write_bytes=%d releases=%d "
            "max_pending_release_bytes=%d released_reserved_bytes=%d "
            "buffer_bytes=%d release_watermark_bytes=%d protocol=%d "
            "transport=%s elapsed_s=%.6f",
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
            direct_bytes,
            staged_bytes,
            max_frame_bytes,
            max_write_bytes,
            release_count,
            max_pending_release_bytes,
            released_reserved_bytes,
            self.buffer_size_bytes,
            self.release_watermark_bytes,
            _PROTOCOL_VERSION,
            transport_mode,
            elapsed,
        )
        logger.info(
            "RoCE TP RAM weight load rank=%d complete: tensors=%d, frames=%d, "
            "traffic=%.2f GiB (direct %.2f GiB, staged %.2f GiB), "
            "elapsed=%.2f seconds",
            tp_rank,
            tensor_count,
            batch_count,
            transferred_bytes / (1024**3),
            direct_bytes / (1024**3),
            staged_bytes / (1024**3),
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
