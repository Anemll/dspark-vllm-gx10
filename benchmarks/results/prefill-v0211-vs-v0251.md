Before: `v0211-baseline-steady` / `0.21.1rc1.dev339+g1967a5627bc3`  
After: `v0251-candidate-steady` / `0.25.2.dev0+g752a3a504.d20260714`

Model: [drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored](https://huggingface.co/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored), DeepSeek V4 Flash, 284B MoE / approximately 13B active parameters. The checkpoint contains 48 FP8 Safetensors shards totaling 155.43 GiB on each node; the serving KV cache uses `nvfp4_ds_mla`.

Single-node reference: The unchanged 155.43 GiB checkpoint exceeds one GX10's approximately 121 GiB of usable unified memory. See the [TP=1 fit check](prefill-v0251-single-node-fit.md); no single-node throughput samples are valid.

| Input tokens | Before server tok/s | After server tok/s | Gain | Before TTFT | After TTFT |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 1,778.7 | 2,033.0 | +14.3% | 0.585s | 0.512s |
| 2,048 | 1,990.5 | 2,252.0 | +13.1% | 1.037s | 0.920s |
| 4,096 | 2,083.1 | 2,320.7 | +11.4% | 1.977s | 1.776s |
| 8,192 | 2,049.8 | 2,184.2 | +6.6% | 4.015s | 3.765s |
| 16,384 | 2,052.6 | 2,203.8 | +7.4% | 8.010s | 7.455s |
| 32,768 | 1,901.1 | 2,176.1 | +14.5% | 17.284s | 15.119s |

Warmed steady-state comparison on two GX10 nodes (TP=2): 3 trials per size, seed 4106, one output token, zero prefix-cache hits, no overlapping requests, and matching prompt hashes across versions.
