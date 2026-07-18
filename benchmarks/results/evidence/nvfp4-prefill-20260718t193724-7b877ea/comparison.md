Before: `v0251-candidate-steady` / `0.25.2.dev0+g752a3a504.d20260714`  
After: `nvfp4-a4w4-7b877ea` / `0.25.2.dev0+g752a3a504.d20260714`

Comparison caveat: prompt fingerprints and/or trial counts differ; this is a same-size aggregate comparison, not a paired prompt-matched A/B.

Model: [drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored](https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored), DeepSeek V4 Flash, 284B MoE / approximately 13B active parameters. The checkpoint contains 48 FP8 Safetensors shards totaling 155.43 GiB on each node; the serving KV cache uses `nvfp4_ds_mla`.

| Input tokens | Before server tok/s | After server tok/s | Gain | Before TTFT | After TTFT |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 2,033.0 | 2,242.5 | +10.3% | 0.512s | 0.463s |
| 2,048 | 2,252.0 | 2,473.2 | +9.8% | 0.920s | 0.835s |
| 4,096 | 2,320.7 | 2,659.3 | +14.6% | 1.776s | 1.552s |
| 8,192 | 2,184.2 | 2,593.5 | +18.7% | 3.765s | 3.173s |
| 16,384 | 2,203.8 | 2,501.7 | +13.5% | 7.455s | 6.573s |
| 32,768 | 2,176.1 | 2,477.3 | +13.8% | 15.119s | 13.264s |

Same-size aggregate comparison on two GX10 nodes (TP=2). Each report retains its own seed, trial count, cache-isolation, and exact-token validity checks; see the caveat above.

Candidate scope: base NVIDIA NVFP4 W4A4 prepared-weight loading with
`FLASHINFER_CUTLASS`; DSpark/MTP/speculation was off and RoCE remained paused.
This is therefore a base-W4A4 prefill result, not a DSpark-accelerated decode
comparison.
