# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.model_loader.roce_tp_loader as roce_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.roce_tp_loader import (
    _RoCEWeightReceiver,
    _RoCEWeightSender,
    _RemoteTensor,
    _WriteRecorder,
    _evaluate_expression,
    TimedDefaultModelLoader,
)


class _FakeGroup:
    def __init__(self, received: list[tuple]) -> None:
        self.received = list(received)
        self.sent: list[tuple[tuple, int]] = []

    def send_object(self, message: tuple, dst: int) -> None:
        self.sent.append((message, dst))

    def recv_object(self, src: int) -> tuple:
        del src
        return self.received.pop(0)


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
    sender._flush = lambda: None
    sender._queue_writes = lambda source, specifications: None
    source = torch.empty((3, 5), dtype=torch.float32)

    yielded = list(sender.iter_weights([("weight", source)]))

    assert yielded == [("weight", source)]
    assert sender.source_bytes == source.numel() * source.element_size()
    assert group.sent[-1] == (("end", sender.source_bytes), 1)


def test_flush_allocation_failure_sends_no_header(monkeypatch) -> None:
    group = _FakeGroup([])
    sender = _RoCEWeightSender.__new__(_RoCEWeightSender)
    sender.group = group
    sender.device = torch.device("cpu")
    sender.pending = [torch.empty(1, dtype=torch.float32)]
    sender.pending_bytes = 4

    def fail_allocation(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic allocation failure")

    monkeypatch.setattr(torch, "empty", fail_allocation)

    with pytest.raises(RuntimeError, match="synthetic allocation failure"):
        sender._flush()

    assert group.sent == []


def test_receiver_captures_source_bytes_from_end_message() -> None:
    receiver = _RoCEWeightReceiver.__new__(_RoCEWeightReceiver)
    receiver.group = _FakeGroup([("end", 12345)])
    receiver.pending = []
    receiver.source_bytes = 0

    assert list(receiver.iter_weights()) == []
    assert receiver.source_bytes == 12345


def test_receiver_accepts_legacy_end_message_for_rollback() -> None:
    receiver = _RoCEWeightReceiver.__new__(_RoCEWeightReceiver)
    receiver.group = _FakeGroup([("end",)])
    receiver.pending = []
    receiver.source_bytes = 99

    assert list(receiver.iter_weights()) == []
    assert receiver.source_bytes == 0


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
