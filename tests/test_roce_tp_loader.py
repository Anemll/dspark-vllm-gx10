# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.model_loader.roce_tp_loader as roce_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.roce_tp_loader import (
    _CapturedWrite,
    _RoCEPeerAbortedError,
    _RoCEWeightReceiver,
    _RoCEWeightSender,
    _RemoteTensor,
    _WriteRecorder,
    _can_broadcast_to,
    _chunk_indices,
    _evaluate_expression,
    _negotiate_protocol,
    _shape_after_index,
    TimedDefaultModelLoader,
)


class _FakeGroup:
    def __init__(self, received: list[tuple]) -> None:
        self.received = list(received)
        self.sent: list[tuple[tuple, int]] = []
        self.tensor_sends: list[tuple[torch.Tensor, int]] = []

    def send_object(self, message: tuple, dst: int) -> None:
        self.sent.append((message, dst))

    def recv_object(self, src: int) -> tuple:
        del src
        return self.received.pop(0)

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        self.tensor_sends.append((tensor.detach().clone(), dst))


def _replay(source: torch.Tensor, recorder: _WriteRecorder) -> None:
    for write in recorder.writes:
        payload = _evaluate_expression(write.expression, source)
        assert isinstance(payload, torch.Tensor)
        write.destination.copy_(payload)


def _remote(source: torch.Tensor, recorder: _WriteRecorder) -> _RemoteTensor:
    return _RemoteTensor(
        torch.empty(source.shape, dtype=source.dtype, device="meta"),
        ("input",),
        recorder,
        source.device,
    )


def test_remote_recipe_narrows_without_materializing_source() -> None:
    source = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    recorder = _WriteRecorder()
    remote = _remote(source, recorder)
    destination = torch.empty((4, 4))

    destination.copy_(remote.narrow(0, 4, 4))

    assert len(recorder.writes) == 1
    _replay(source, recorder)
    torch.testing.assert_close(destination, source[4:])


def test_remote_recipe_replays_padding_and_dtype_views() -> None:
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    recorder = _WriteRecorder()
    remote = _remote(source, recorder)
    padded_destination = torch.empty((3, 4))
    byte_destination = torch.empty(
        source.numel() * source.element_size(), dtype=torch.uint8
    )

    padded_destination.copy_(
        torch.cat((remote, remote.new_zeros((1, 4))), dim=0)
    )
    byte_destination.copy_(remote.view(torch.uint8).reshape(-1))

    _replay(source, recorder)
    torch.testing.assert_close(padded_destination[:2], source)
    torch.testing.assert_close(padded_destination[2], torch.zeros(4))
    torch.testing.assert_close(byte_destination, source.view(torch.uint8).reshape(-1))


def test_sender_reports_source_bytes_in_end_message() -> None:
    group = _FakeGroup([("writes", [])])
    sender = _RoCEWeightSender.__new__(_RoCEWeightSender)
    sender.group = group
    sender.source_bytes = 0
    sender.tensor_count = 0
    sender._send_writes = lambda source, specifications: None
    source = torch.empty((3, 5), dtype=torch.float32)

    yielded = list(sender.iter_weights([("weight", source)]))

    assert yielded == [("weight", source)]
    assert sender.source_bytes == source.numel() * source.element_size()
    assert group.sent[-1] == (("end", sender.source_bytes), 1)


def test_chunk_schedule_preserves_order_and_hard_bound() -> None:
    shape = (2, 3, 11)
    indices = list(_chunk_indices(shape, max_elements=8))

    assert indices
    assert all(
        torch.tensor(_shape_after_index(shape, index)).prod().item() <= 8
        for index in indices
    )
    source = torch.arange(66).reshape(shape)
    replayed = torch.cat([source[index].reshape(-1) for index in indices])
    torch.testing.assert_close(replayed, source.reshape(-1))


def test_copy_broadcast_validation_matches_supported_shapes() -> None:
    assert _can_broadcast_to((4,), (3, 4))
    assert _can_broadcast_to((1, 4), (3, 4))
    assert _can_broadcast_to((), (3, 4))
    assert not _can_broadcast_to((2, 4), (3, 4))
    assert not _can_broadcast_to((1, 3, 4), (3, 4))


def test_protocol_negotiates_version_frame_and_transport() -> None:
    expected = ("hello_ack", 2, "pynccl", 64)
    group = _FakeGroup([expected])
    group.device_communicator = SimpleNamespace(
        pynccl_comm=SimpleNamespace(disabled=False)
    )

    assert _negotiate_protocol(group, rank=0, buffer_size_bytes=64) == "pynccl"
    assert group.sent == [(('hello', 2, 'pynccl', 64), 1)]


def test_protocol_rejects_mixed_frame_configuration() -> None:
    group = _FakeGroup([("hello_ack", 2, "pynccl", 128)])
    group.device_communicator = SimpleNamespace(
        pynccl_comm=SimpleNamespace(disabled=False)
    )

    with pytest.raises(RuntimeError, match="protocol negotiation failed"):
        _negotiate_protocol(group, rank=0, buffer_size_bytes=64)


def test_receiver_protocol_role_rejects_transport_mismatch() -> None:
    group = _FakeGroup([("hello", 2, "pynccl", 64)])
    group.device_communicator = SimpleNamespace(pynccl_comm=None)

    with pytest.raises(RuntimeError, match="protocol negotiation failed"):
        _negotiate_protocol(group, rank=1, buffer_size_bytes=64)
    assert group.sent[0][0][0] == "hello_error"


def test_sender_abort_drains_pending_writes_before_notifying_peer() -> None:
    group = _FakeGroup([("writes", [])])
    sender = _RoCEWeightSender.__new__(_RoCEWeightSender)
    sender.group = group
    sender.control_state = "writes"

    sender.abort(RuntimeError("synthetic local loader failure"))

    assert sender.control_state == "aborted"
    assert group.sent[-1][0][0] == "abort"


def test_sender_abort_does_not_echo_receiver_error() -> None:
    group = _FakeGroup([("error", "receiver failed")])
    sender = _RoCEWeightSender.__new__(_RoCEWeightSender)
    sender.group = group
    sender.control_state = "writes"

    sender.abort(RuntimeError("simultaneous local failure"))

    assert sender.control_state == "peer_failed"
    assert group.sent == []


def test_sender_marks_receiver_error_as_peer_abort() -> None:
    group = _FakeGroup([("error", "receiver failed")])
    sender = _RoCEWeightSender.__new__(_RoCEWeightSender)
    sender.group = group
    sender.source_bytes = 0
    sender.tensor_count = 0
    sender.control_state = "receiver_control"

    with pytest.raises(_RoCEPeerAbortedError, match="receiver failed"):
        list(sender.iter_weights([("weight", torch.empty(1))]))
    assert sender.control_state == "peer_failed"


def test_sender_chunks_noncontiguous_write_without_oversized_frame() -> None:
    group = _FakeGroup([])
    sender = _RoCEWeightSender.__new__(_RoCEWeightSender)
    sender.group = group
    sender.device = torch.device("cpu")
    sender.buffer_size_bytes = 16
    sender.staging = torch.empty(16, dtype=torch.uint8)
    sender.sent_bytes = 0
    sender.batch_count = 0
    sender.direct_bytes = 0
    sender.staged_bytes = 0
    sender.max_frame_bytes = 0
    sender.max_write_bytes = 0
    payload = torch.arange(48, dtype=torch.float32).reshape(6, 8)[:, ::2]

    sender.control_state = "frame_control"
    sender._send_payload(payload, direct=True, write_index=0)

    frames = [tensor for tensor, _ in group.tensor_sends]
    assert frames
    assert all(frame.numel() * frame.element_size() <= 16 for frame in frames)
    torch.testing.assert_close(
        torch.cat(frames).view(torch.float32), payload.reshape(-1)
    )
    assert sender.sent_bytes == payload.numel() * payload.element_size()
    assert sender.direct_bytes == sender.sent_bytes
    assert sender.staged_bytes == 0
    assert sender.max_frame_bytes == 16
    assert sender.max_write_bytes == sender.sent_bytes


def test_receiver_scatter_uses_one_bounded_staging_window(monkeypatch) -> None:
    payload = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    indices = list(_chunk_indices((6, 4), 4))
    incoming = [
        payload[index].contiguous().view(torch.uint8).reshape(-1)
        for index in indices
    ]
    base = torch.full((6, 8), -1.0)
    destination = base[:, ::2]
    write = _CapturedWrite(
        destination=destination,
        expression=("input",),
        source_shape=(6, 4),
        source_dtype=torch.float32,
        nbytes=payload.numel() * payload.element_size(),
    )
    receiver = _RoCEWeightReceiver.__new__(_RoCEWeightReceiver)
    receiver.group = _FakeGroup(
        [("frame", 0, frame, tensor.numel()) for frame, tensor in enumerate(incoming)]
    )
    receiver.group.incoming = list(incoming)
    receiver.device = torch.device("cpu")
    receiver.buffer_size_bytes = 16
    receiver.staging = None
    receiver.received_bytes = 0
    receiver.batch_count = 0
    receiver.direct_bytes = 0
    receiver.staged_bytes = 0
    receiver.max_frame_bytes = 0
    receiver.max_write_bytes = 0

    def fake_recv_into(group, tensor, src) -> None:
        del src
        tensor.copy_(group.incoming.pop(0))

    monkeypatch.setattr(roce_loader, "_recv_into", fake_recv_into)
    receiver._receive_write(write, 0)

    torch.testing.assert_close(destination, payload)
    assert receiver.staging is not None
    assert receiver.staging.numel() == 16
    assert receiver.received_bytes == payload.numel() * payload.element_size()
    assert receiver.direct_bytes == 0
    assert receiver.staged_bytes == receiver.received_bytes
    assert receiver.max_frame_bytes == 16


def test_receiver_direct_path_writes_raw_bytes_in_place(monkeypatch) -> None:
    payload = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    raw = payload.view(torch.uint8).reshape(-1)
    frames = [raw[:16], raw[16:32], raw[32:]]
    destination = torch.empty_like(payload)
    write = _CapturedWrite(
        destination=destination,
        expression=("input",),
        source_shape=tuple(payload.shape),
        source_dtype=payload.dtype,
        nbytes=payload.numel() * payload.element_size(),
    )
    write.direct_eligible = lambda: True
    receiver = _RoCEWeightReceiver.__new__(_RoCEWeightReceiver)
    receiver.group = _FakeGroup(
        [("frame", 0, index, frame.numel()) for index, frame in enumerate(frames)]
    )
    receiver.group.incoming = [frame.clone() for frame in frames]
    receiver.device = torch.device("cpu")
    receiver.buffer_size_bytes = 16
    receiver.staging = None
    receiver.received_bytes = 0
    receiver.batch_count = 0
    receiver.direct_bytes = 0
    receiver.staged_bytes = 0
    receiver.max_frame_bytes = 0
    receiver.max_write_bytes = 0

    def fake_recv_into(group, tensor, src) -> None:
        del src
        tensor.copy_(group.incoming.pop(0))

    monkeypatch.setattr(roce_loader, "_recv_into", fake_recv_into)
    receiver._receive_write(write, 0)

    torch.testing.assert_close(destination, payload)
    assert receiver.staging is None
    assert receiver.direct_bytes == raw.numel()
    assert receiver.staged_bytes == 0


def test_receiver_captures_source_bytes_from_end_message() -> None:
    receiver = _RoCEWeightReceiver.__new__(_RoCEWeightReceiver)
    receiver.group = _FakeGroup([("end", 12345)])
    receiver.source_bytes = 0

    assert list(receiver.iter_weights()) == []
    assert receiver.source_bytes == 12345


def test_receiver_rejects_mixed_protocol_legacy_end_message() -> None:
    receiver = _RoCEWeightReceiver.__new__(_RoCEWeightReceiver)
    receiver.group = _FakeGroup([("end",)])
    receiver.source_bytes = 99

    with pytest.raises(RuntimeError, match="Invalid RoCE end message"):
        list(receiver.iter_weights())


def test_timed_default_loader_presents_auto_to_wrapped_loader(monkeypatch) -> None:
    loader = TimedDefaultModelLoader.__new__(TimedDefaultModelLoader)
    loader.load_config = SimpleNamespace(load_format="direct_timed")
    observed_formats: list[str] = []
    barriers: list[bool] = []

    def fake_load_weights(self, model, model_config) -> None:
        del model, model_config
        observed_formats.append(self.load_config.load_format)

    class _FakeStream:
        def synchronize(self) -> None:
            pass

    monkeypatch.setattr(DefaultModelLoader, "load_weights", fake_load_weights)
    monkeypatch.setattr(
        roce_loader,
        "get_tp_group",
        lambda: SimpleNamespace(
            rank_in_group=0,
            barrier=lambda: barriers.append(True),
        ),
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: _FakeStream())

    loader.load_weights(object(), SimpleNamespace())

    assert observed_formats == ["auto"]
    assert barriers == [True]
    assert loader.load_config.load_format == "direct_timed"
