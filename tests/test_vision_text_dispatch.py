# SPDX-License-Identifier: MIT
"""Execute real SWA metadata control flow with recording CPU fakes.

This checks allocation/dispatch contracts, not CUDA numerics or throughput.
Only the method is compiled from the overlay; the GPU runtime is not imported.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "overlay/vllm/v1/attention/backends/mla/sparse_swa.py"


def load_build(namespace, source=None):
    tree = ast.parse(SOURCE.read_text() if source is None else source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "DeepseekSparseSWAMetadataBuilder")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "build")
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace["build"]


class Tensor:
    def __init__(self, name, shape, events):
        self.name, self.shape, self.events = name, shape, events

    def __getitem__(self, index):
        if not isinstance(index, slice):
            raise AssertionError("Unexpected scalar tensor read")
        start, stop, step = index.indices(self.shape[0])
        return Tensor(f"{self.name}[{start}:{stop}:{step}]",
                      (len(range(start, stop, step)), *self.shape[1:]), self.events)

    def __setitem__(self, index, value):
        self.events.append(("fill", self[index].name, value))

    def __ge__(self, value):
        return Tensor(f"({self.name}>={value})", self.shape, self.events)

    def copy_(self, value):
        self.events.append(("copy", self.name, value.name))

    def stride(self, axis):
        result = 1
        for size in self.shape[axis + 1:]:
            result *= size
        return result


def normalize(value):
    if isinstance(value, Tensor):
        return (value.name, value.shape)
    if isinstance(value, tuple):
        return tuple(normalize(v) for v in value)
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items()}
    return value


class Kernel:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.events.append(("kernel", self.name, grid,
                                normalize(args), normalize(kwargs)))
        return launch


def run_build(split, causal, *, vision_text=False, source=None, repeat=False,
              image_prefill=False):
    events = []
    total = split[2] + split[3]
    def tensor(name, shape):
        return Tensor(name, shape, events)
    def zeros(*shape, **kwargs):
        events.append(("allocate", shape, kwargs))
        return tensor("noncausal", shape)
    namespace = {
        "torch": SimpleNamespace(zeros=zeros, int32="int32"),
        "split_decodes_and_prefills": lambda metadata, **kwargs: split,
        "DeepseekSparseSWAMetadata": lambda **kwargs: normalize(kwargs),
        "_LAYER_TYPE_SWAONLY": 0, "_LAYER_TYPE_C4A": 1, "_LAYER_TYPE_C128A": 2,
        "_compute_swa_indices_and_lens_kernel": Kernel("text", events),
        "_compute_dspark_noncausal_swa_indices_kernel": Kernel("dspark", events),
    }
    calls = []
    def no_image(*args):
        calls.append(True)
        if image_prefill:
            return (tensor("image_indices", (split[3], 1, 512)),
                    tensor("image_lens", (split[3],)))
        return None
    builder = SimpleNamespace(
        decode_threshold=6, window_size=128, noncausal_index_width=256,
        block_size=256, device="recording-only", is_dspark=True, _max_tokens=2048,
        image_visibility=SimpleNamespace(build=no_image) if vision_text else None,
        decode_swa_indices_noncausal=None,
        _build_deepseek_v4_metadata=lambda *args: {},
        build_tile_scheduler=lambda count: {0: None, 1: None, 2: None},
    )
    for name, shape in {
        "token_to_req_indices": (2048,), "is_valid_token": (2048,),
        "decode_swa_indices": (2048, 1, 128), "decode_swa_lens": (2048,),
        "prefill_swa_indices": (2048, 1, 128), "prefill_swa_lens": (2048,),
    }.items():
        setattr(builder, name, tensor(name, shape))
    common = SimpleNamespace(
        seq_lens=tensor("seq_lens", (12,)), seq_lens_cpu_upper_bound="seq_cpu",
        query_start_loc=tensor("qsl", (13,)), query_start_loc_cpu="qsl_cpu",
        block_table_tensor=tensor("block_table", (12, 32)),
        slot_mapping=tensor("slot_mapping", (total,)), causal=causal,
        token_to_req_indices=lambda out: out[:total],
    )
    build = load_build(namespace, source)
    metadata = build(builder, 0, common)
    if repeat:
        events.clear()
        metadata = build(builder, 0, common)
    return metadata, events, calls


class TextDispatchTests(unittest.TestCase):
    def test_text_and_dspark_launch_only_original_kernels(self):
        # Target decode, DSpark draft, text prefill, and mixed target batches.
        for split, causal in [((1, 0, 6, 0), True), ((4, 0, 24, 0), True),
                              ((1, 0, 5, 0), False), ((4, 0, 20, 0), False),
                              ((0, 1, 0, 35), True), ((1, 1, 6, 387), True),
                              ((0, 1, 0, 2000), True), ((0, 0, 0, 0), True)]:
            with self.subTest(split=split, causal=causal):
                text, events, _ = run_build(split, causal)
                vision, vision_events, calls = run_build(split, causal, vision_text=True)
                self.assertEqual(text, vision)
                self.assertEqual(events, vision_events)
                self.assertEqual(len(calls), int(split[3] > 0))
                kernels = [event for event in events if event[0] == "kernel"]
                expected = (["text" if causal else "dspark"] if split[2] else [])
                expected += ["text"] if split[3] else []
                self.assertEqual([event[1] for event in kernels], expected)
                if split[3]:
                    self.assertEqual(text["prefill_swa_indices"][1], (split[3], 1, 128))
                    self.assertEqual(kernels[-1][-1]["token_offset"], split[2])
                else:
                    self.assertIsNone(text["prefill_swa_indices"])
                    self.assertIsNone(text["prefill_swa_lens"])

    def test_draft_metadata_reuses_buffer_after_first_call(self):
        for vision_text in (False, True):
            _, cold, _ = run_build((1, 0, 5, 0), False, vision_text=vision_text)
            _, warm, _ = run_build((1, 0, 5, 0), False, vision_text=vision_text, repeat=True)
            self.assertEqual(sum(e[0] == "allocate" for e in cold), 1)
            self.assertFalse(any(e[0] == "allocate" for e in warm))

    def test_mixed_image_prefill_still_uses_image_metadata(self):
        metadata, events, calls = run_build(
            (1, 1, 6, 387), True, vision_text=True, image_prefill=True,
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(metadata["prefill_swa_indices"], ("image_indices", (387, 1, 512)))
        self.assertEqual(metadata["prefill_swa_lens"], ("image_lens", (387,)))
        # The image callback supplied prefill metadata; only the six decode
        # tokens use the original text kernel. This fake does not test the
        # internals/numerics of the image callback itself.
        kernels = [event for event in events if event[0] == "kernel"]
        self.assertEqual([(event[1], event[2]) for event in kernels], [("text", (6,))])


if __name__ == "__main__":
    unittest.main()
