# DSpark target-backend A/B

This is an end-to-end TP=2 serving comparison on the same two GX10 nodes. All
arms used the canonical 35-token prompt, 512 output tokens, temperature zero,
MTP=5, probabilistic draft sampling, confidence scheduling off, and the native
MXFP4 DSpark draft on `DEEPGEMM_MXFP4`. Only the target checkpoint/backend
changed.

The prompt SHA-256 is
`652af3aabacfd4360432d28e0c237e9e445f938d032a604d3a4f7f42a2a7ed38`.
Values are best aggregate throughput with the median in parentheses. C=4 uses
the separate fully warmed three-trial recheck for every arm.

| Target | C=1 | C=4 |
|---|---:|---:|
| FP8/B12X + DSpark | **47.94 (47.80) tok/s** | **103.98 (101.07) tok/s** |
| W4A4/CUTLASS + DSpark | 47.67 (46.95) tok/s | 101.12 (98.08) tok/s |
| W4A4/B12X + DSpark | **49.12 (47.53) tok/s** | 91.97 (90.40) tok/s |

Relative results:

| Comparison | C=1 best / median | C=4 best / median |
|---|---:|---:|
| W4A4/CUTLASS vs FP8/B12X | -0.6% / -1.8% | -2.8% / -3.0% |
| W4A4/B12X vs FP8/B12X | +2.5% / -0.6% | **-11.6% / -10.6%** |
| W4A4/B12X vs W4A4/CUTLASS | +3.0% / +1.2% | **-9.1% / -7.8%** |

The B12X C=1 best trial also had the highest accepted length (3.384 versus
3.140 for the deterministic FP8 arm), so its isolated best-result lead is not
evidence of a faster target. At C=4 the accepted lengths are comparable while
B12X remains materially slower. DSpark therefore does not rescue the W4A4
B12X path at batched decode; W4A4/CUTLASS remains the serving default.

`DSPARK_B12X_MICRO_MAX_ACTIVE_CLUSTERS=40` was configured identically on both
ranks. It applies only when B12X selects its micro kernel at most 40 routed
rows. With MTP=5, C=1 verifies up to six target rows and therefore produces 36
routed rows (`6 * top-k 6`), so MAC40 is active. C=4 verifies up to 24 target
rows and produces 144 routed rows, selecting `MoEStaticKernel`; the MAC40
micro-kernel optimization is bypassed. Inference logs independently reported
`MoEStaticKernel` compilation. The C=4 result therefore tests the configured
B12X arm, but not the target-only MAC40 execution path that won at M=4.

The later collision-safe C2--C4 activation-pack-sharing experiment was not
baked into the immutable control image. It improved one real-layer gate by
only 0.65% and was not promoted. It also gates on two to four input tokens, so
it would not activate for the 24-row C=4/MTP=5 verifier shape.

The B12X arm proved the intended split dispatch in logs:

- prepared W4A4 target: `FLASHINFER_B12X`;
- DSpark draft: `DEEPGEMM_MXFP4`;
- prepared target load: 43 layers, 344 reads, 344 copies;
- model loading: 109.60 seconds;
- candidate image:
  `sha256:681feffd95b0e0e95c1864ea6a76c93444e907eca532e583150ab9469817f676`;
- source revision:
  `c2f93ff8ff2a10d257022c510487da4fdfb0f980`.

The common W4A4 control image was
`sha256:45cc3a5f9bc6b2ed8ce39d242971ae0c258162a788076474f8ad2d5703e5c2b8`.
The FP8 reference used the pinned production image
`sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8`.

Several first trials incurred lazy CuTeDSL/Triton compilation and high TTFT.
Those trials remain in the raw JSON. The table uses separate warmed C=4
rechecks and reports both best and median rather than deleting evidence.
