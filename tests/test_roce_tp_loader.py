# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Anemll contributors

import torch

from vllm.model_executor.model_loader.roce_tp_loader import (
    _RemoteTensor,
    _WriteRecorder,
    _evaluate_expression,
)


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
