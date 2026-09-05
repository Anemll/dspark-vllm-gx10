# SPDX-License-Identifier: MIT
"""Dependency-free tests for the real streaming-loader implementation."""
import gc
import importlib.util
from pathlib import Path
import unittest
import weakref

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "vision_loading", ROOT / "overlay/vllm/models/deepseek_v4/nvidia/vision_loading.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TensorStandIn:
    pass


class StreamingLoaderTests(unittest.TestCase):
    def test_interleaved_weights_keep_raw_names_and_single_pass(self):
        seen = []
        source = iter((name, name) for name in (
            "layers.0.ffn.w1", "image_start", "layers.1.ffn.w2",
            "vision.blocks.0.wqkv.weight", "aligner.w1.weight", "mtp.0.weight",
            "head.weight", "image_end",
        ))
        text = list(module.stream_language_weights(source, lambda n, t: seen.append(n)))
        self.assertEqual([n for n, _ in text], [
            "layers.0.ffn.w1", "layers.1.ffn.w2", "mtp.0.weight", "head.weight"
        ])
        self.assertEqual(seen, ["image_start", "vision.blocks.0.wqkv.weight",
                                "aligner.w1.weight", "image_end"])
        self.assertEqual(list(source), [])

    def test_does_not_retain_checkpoint_tensors(self):
        refs = []
        def source():
            for i in range(300):
                tensor = TensorStandIn()
                refs.append(weakref.ref(tensor))
                yield ("vision.weight" if i % 2 else "layers.0.weight"), tensor
                # One producer local plus one consumer local is bounded.
                self.assertLessEqual(sum(ref() is not None for ref in refs), 3)
        for _, tensor in module.stream_language_weights(source(), lambda n, t: None):
            pass
        del tensor
        gc.collect()
        self.assertEqual(sum(ref() is not None for ref in refs), 0)

    def test_errors_stop_without_consuming_rest_of_checkpoint(self):
        visited = []
        def source():
            for name in ("layers.0.weight", "vision.bad", "layers.1.weight"):
                visited.append(name)
                yield name, TensorStandIn()
        def fail(n, t):
            raise KeyError(n)
        stream = module.stream_language_weights(source(), fail)
        self.assertEqual(next(stream)[0], "layers.0.weight")
        with self.assertRaises(KeyError):
            next(stream)
        self.assertEqual(visited, ["layers.0.weight", "vision.bad"])


if __name__ == "__main__":
    unittest.main()
