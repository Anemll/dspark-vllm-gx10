#!/usr/bin/env bash
set -euo pipefail

ROLE=${1:?role required}
ART=${2:?artifact directory required}
case "$ROLE" in
  head|worker) ;;
  *) echo "invalid role: $ROLE" >&2; exit 64 ;;
esac

PROD_ID=sha256:3430d6614a8e2925f34d059af6caf05aff42387326db4d05639a60f10f2654d8
CANDIDATE=dspark-vllm-gx10:dev-7b877eaae2a8-aot0615-prepared
CANDIDATE_ID=sha256:222c3295b804664f19442a953143fef45a7fdc3ed278ae5e82eab546f7519b99
DEV=/home/anemll/dspark-vllm-gx10-dev
REV=7b877eaae2a8e2b5800e84b585d7f14fb90f5294
BENCH_SHA=c9563a339b3c8fd82adfa284a3a8e00106bd8a7443d0741b53879533ea65a121
COMPOSE_SHA=f61b672ce263cba9d145d75d8ef81085cd97c72a1cc20ddb21cfd39fdb8df869
TARGET_MANIFEST_SHA=972ba797456da80e586324a5a8c29af42bac86510ceff983e674de41d31e6f26
TARGET_CONFIG_SHA=f5d2f027ba158b88707c6715d7c1080f119b6fb244cfa4162d883e8935f72e1d
TARGET_INDEX_SHA=9f768c86d1eb7e09c9dff4fb9f87b20cf59df17923f415f697f9ee6caecf328f
DRAFT_CONFIG_SHA=6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023
DRAFT_INDEX_SHA=98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b
KV_BYTES=32212254720
NAME=dspark-a0-cutlass-smallm
export PROD_ID CANDIDATE_ID

mkdir -p "$ART"
exec > >(tee "$ART/$ROLE.log") 2>&1

cleanup() {
  if [[ "$ROLE" == worker ]]; then
    sudo -n docker rm -f "$NAME" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

sudo -v
sudo -n true

mapfile -t CIDS < <(sudo -n docker ps -q --filter label=com.docker.compose.service=vllm-dspark)
test "${#CIDS[@]}" -eq 1
CID=${CIDS[0]}
sudo -n docker inspect "$CID" >"$ART/production-inspect.json"
sudo -n docker image inspect "$CANDIDATE" >"$ART/candidate-inspect.json"

readarray -t STATE < <(python3 - "$ART/production-inspect.json" "$ART/candidate-inspect.json" <<'PY'
import json, os, sys
prod=json.load(open(sys.argv[1]))[0]
cand=json.load(open(sys.argv[2]))[0]
assert prod["Image"] == os.environ["PROD_ID"]
assert prod["State"]["Running"] is True
assert prod["State"]["OOMKilled"] is False
assert cand["Id"] == os.environ["CANDIDATE_ID"]
mounts={m["Destination"]:m for m in prod["Mounts"]}
target=mounts["/models/dsv4-abliterated"]
assert target["RW"] is False
print("PROD_SOURCE="+target["Source"])
PY
)
eval "${STATE[0]}"

if [[ "$ROLE" == head ]]; then
  TARGET=/home/anemll/models/DeepSeek-V4-Flash-NVFP4-TP2-CUTLASS-Prepared-v1
  curl -fsS --max-time 5 http://127.0.0.1:8888/health >/dev/null
else
  TARGET=/mnt/xfsflash/models/DeepSeek-V4-Flash-NVFP4-TP2-CUTLASS-Prepared-v1
fi
test -d "$TARGET"
test -d "$PROD_SOURCE"
test "$(sha256sum "$TARGET/dspark-nvfp4-tp2-repack.json" | awk '{print $1}')" = "$TARGET_MANIFEST_SHA"
test "$(sha256sum "$TARGET/config.json" | awk '{print $1}')" = "$TARGET_CONFIG_SHA"
test "$(sha256sum "$TARGET/model.safetensors.index.json" | awk '{print $1}')" = "$TARGET_INDEX_SHA"
test "$(sha256sum "$PROD_SOURCE/config.json" | awk '{print $1}')" = "$DRAFT_CONFIG_SHA"
test "$(sha256sum "$PROD_SOURCE/model.safetensors.index.json" | awk '{print $1}')" = "$DRAFT_INDEX_SHA"
test "$(git -C "$DEV" rev-parse HEAD)" = "$REV"
test -z "$(git -C "$DEV" status --porcelain --untracked-files=no)"
test "$(sha256sum "$DEV/benchmarks/benchmark_nvfp4_a4w4_sm121.py" | awk '{print $1}')" = "$BENCH_SHA"
test "$(sha256sum "$ART/docker-compose.yml" | awk '{print $1}')" = "$COMPOSE_SHA"

export DSPARK_DRAFT_MODEL_HOST="$PROD_SOURCE"
export DSPARK_SPECULATION_MODE=dspark
export MTP_NUM_TOKENS=5
export KV_CACHE_MEMORY_BYTES="$KV_BYTES"
sudo -n --preserve-env=DSPARK_DRAFT_MODEL_HOST,DSPARK_SPECULATION_MODE,MTP_NUM_TOKENS,KV_CACHE_MEMORY_BYTES \
  docker compose --env-file "$ART/role.env" -f "$ART/docker-compose.yml" \
  config --format json >"$ART/compose.json"
python3 - "$ART/compose.json" "$TARGET" "$PROD_SOURCE" "$KV_BYTES" <<'PY'
import json, re, sys
d=json.load(open(sys.argv[1]))
s=d["services"]["vllm-dspark"]
vols={v["target"]:v for v in s["volumes"]}
assert vols["/models/dsv4-abliterated"]["source"] == sys.argv[2]
assert vols["/models/dsv4-abliterated"].get("read_only") is True
assert vols["/models/dspark-draft"]["source"] == sys.argv[3]
assert vols["/models/dspark-draft"].get("read_only") is True
env=s["environment"]
assert str(env["DSPARK_SPECULATION_MODE"]) == "dspark"
assert str(env["MTP_NUM_TOKENS"]) == "5"
assert str(env["KV_CACHE_MEMORY_BYTES"]) == sys.argv[4]
command=" ".join(s["command"]) if isinstance(s["command"], list) else s["command"]
match=re.search(r'SPECULATIVE_CONFIG="(\{.*?\})";', command)
assert match is not None
spec=json.loads(bytes(match.group(1), "utf-8").decode("unicode_escape"))
assert spec == {
    "method":"dspark",
    "model":"/models/dspark-draft",
    "num_speculative_tokens":5,
    "draft_sample_method":"probabilistic",
}
assert '--kv-cache-memory-bytes' in command
assert '--moe-backend flashinfer_cutlass' in command
print("COMPOSE_SPLIT_PASS=true")
PY

echo "ROLE=$ROLE"
echo "CID=$CID"
echo "PROD_SOURCE=$PROD_SOURCE"
echo "TARGET=$TARGET"
echo "COMPOSE_VERSION=$(sudo -n docker compose version --short)"

if [[ "$ROLE" == worker ]]; then
  test -z "$(sudo -n docker ps -aq --filter name=^/${NAME}$)"
  MEM_BEFORE=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  SWAP_BEFORE=$(awk '/SwapFree:/ {print $2}' /proc/meminfo)
  test "$MEM_BEFORE" -ge 8388608
  test "$SWAP_BEFORE" -ge 2097152
  set +e
  timeout --signal=TERM --kill-after=5s 180s sudo -n docker run \
    --name "$NAME" --gpus all --network none --ipc host \
    --ulimit memlock=-1:-1 \
    -e FLASHINFER_DISABLE_VERSION_CHECK= \
    -e FLASHINFER_WORKSPACE_BASE=/cache/huggingface/flashinfer-0.6.15-0472b9b3 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -v "$DEV:/workspace:ro" -v "$ART:/artifacts" -w /workspace \
    --entrypoint python3 "$CANDIDATE" \
    /workspace/benchmarks/benchmark_nvfp4_a4w4_sm121.py \
    --synthetic --synthetic-experts 8 --tp-size 2 --tp-rank 1 \
    --backend flashinfer_cutlass --m 4,8,12,16,20,24,32,64 \
    --correctness-m 4,24,64 --routing balanced --seed 4104 \
    --warmup 3 --iters 10 --repeats 3 --cuda-graph --require-graphs \
    --fail-fast --output /artifacts/worker-smallm.json \
    </dev/null >"$ART/worker-smallm.log" 2>&1
  RUN_RC=$?
  set -e
  sudo -n docker rm -f "$NAME" >/dev/null 2>&1 || true
  test -z "$(sudo -n docker ps -aq --filter name=^/${NAME}$)"
  MEM_AFTER=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  SWAP_AFTER=$(awk '/SwapFree:/ {print $2}' /proc/meminfo)
  echo "RUN_RC=$RUN_RC"
  echo "MEM_BEFORE_KB=$MEM_BEFORE"
  echo "MEM_AFTER_KB=$MEM_AFTER"
  echo "SWAP_BEFORE_KB=$SWAP_BEFORE"
  echo "SWAP_AFTER_KB=$SWAP_AFTER"
  test "$RUN_RC" -eq 0
  test "$MEM_AFTER" -ge 8388608
  test "$SWAP_AFTER" -ge 2097152
  python3 - "$ART/worker-smallm.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
assert d["failures"] == []
assert d["settings"]["m"] == [4,8,12,16,20,24,32,64]
assert d["settings"]["correctness_m"] == [4,24,64]
assert d["settings"]["backend_selection"] == "flashinfer_cutlass"
assert d["backend_proof"]["flashinfer_cutlass"]["activation_precision"] == "nvfp4"
assert [r["m"] for r in d["results"]] == [4,8,12,16,20,24,32,64]
for row in d["results"]:
    mode=row["modes"]["flashinfer_cutlass"]
    assert mode["cuda_graph_status"] == "captured"
    assert mode["eager"]["median_ms"] > 0
    assert mode["cuda_graph"]["median_ms"] > 0
for row in d["results"]:
    if row["m"] in {4,24,64}:
        assert row["eager_output_activity"]["flashinfer_cutlass"]["passed"] is True
        assert row["modes"]["flashinfer_cutlass"]["graph_output_activity"]["passed"] is True
        assert row["modes"]["flashinfer_cutlass"]["graph_numeric_gate_passed"] is True
print("WORKER_SMALLM_PASS=true")
PY
  sha256sum "$ART/worker-smallm.json" "$ART/worker-smallm.log"
fi

sudo -n docker inspect "$CID" >"$ART/production-post.json"
python3 - "$ART/production-post.json" "$CID" <<'PY'
import json, os, sys
d=json.load(open(sys.argv[1]))[0]
assert d["Id"].split(":")[-1].startswith(sys.argv[2])
assert d["Image"] == os.environ["PROD_ID"]
assert d["State"]["Running"] is True
assert d["State"]["OOMKilled"] is False
print("PRODUCTION_POST_PASS=true")
PY
if [[ "$ROLE" == head ]]; then
  test "$(curl -fsS -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:8888/health)" = 200
fi
