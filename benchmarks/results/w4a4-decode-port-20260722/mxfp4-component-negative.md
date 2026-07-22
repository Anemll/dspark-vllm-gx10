# Native MXFP4 component gate (negative)

The existing FlashInfer `grouped_mm_fp4(block_size=32)` primitive was tested
on GB10/SM121 as a possible true-MXFP4 W4A4 decode path.  The production image
ships cuDNN 9.19; 9.21 has no SM121 execution plan, so the bounded probe used an
isolated cuDNN 9.25.0.15 runtime without changing the image or service.

The one-expert real-shape controls passed for M=1 and M=4 with both
FlashInfer-quantized weights and the checkpoint-native packed weights plus
128x4-interleaved UE8M0 scales.  This proves the basic cuDNN primitive and the
native checkpoint byte/scale layout.  The complete FC1 -> OAI SwiGLU(limit=10)
-> MXFP4 requantize -> FC2 component chain was not numerically valid, so it is
not eligible for integration.

The invalid full-chain M=4 timing was 0.780688 ms versus the accepted CUTLASS
W4A4 reference of 0.7820 ms.  Even before integration overhead it missed the
0.7429 ms (>=5% speedup) screen.  It is therefore a closed negative path; the
timing must not be presented as a valid kernel result.

Evidence:

- `mxfp4-w4a4-component-rank0-isolation.json`
- `mxfp4-w4a4-component-rank0-isolation.log`
- probe source: `benchmarks/probe_mxfp4_w4a4_component_sm121.py`
- CPU contract: `tests/test_probe_mxfp4_w4a4_component_sm121.py` (10 tests)
