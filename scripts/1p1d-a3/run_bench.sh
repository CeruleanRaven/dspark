#!/usr/bin/env bash
# 四个性能用例。服务按社区文档 A3 1P1D 拉起之后，在 Prefill 节点执行。
# 权重与 start_service.sh 相同：DeepSeek-V4.1-Flash-1017-w4a8-03-mtp。请求打到 proxy。
#
#   bash run_bench.sh 19
#   bash run_bench.sh 21
#   bash run_bench.sh 23
#   bash run_bench.sh 28
#
# 平均输入 131072 = random-prefix-len + random-input-len。
# vLLM 里 random-input-len 是前缀之外的长度。
# 表里请求频率 0 表示不限制到达速率，用最大并发做闭环，对应 --request-rate inf。
# 系统并发数是预期在途数，脚本只打印，不作为参数。
# 卡 TTFT 表内未写单位，按秒比较。卡 TPOT 按表内的毫秒比较。比较 mean。
set -euo pipefail

CASE=${1:-}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

read_var() {
    sed -n "s/^$1=//p" "$ROOT/start_service.sh" | head -n 1 | tr -d '\r'
}

PREFILL_NODE_IP=$(read_var PREFILL_NODE_IP)
CONTAINER_NAME=$(read_var CONTAINER_NAME)
PROXY_PORT=$(read_var PROXY_PORT)
MODEL_PATH=$(read_var MODEL_PATH)

: "${PREFILL_NODE_IP:?请先在 start_service.sh 顶部填写 PREFILL_NODE_IP}"
: "${CONTAINER_NAME:?start_service.sh 缺少 CONTAINER_NAME}"
: "${PROXY_PORT:?start_service.sh 缺少 PROXY_PORT}"
: "${MODEL_PATH:?start_service.sh 缺少 MODEL_PATH}"

# prefix unique output prompts concurrency ttft_s tpot_ms inflight label
case "$CASE" in
    19)
        # 纯P，128k/1k（单前缀），PrefixCache 90，只考核 TTFT
        PREFIX=117965
        UNIQUE=13107
        OUTPUT=1
        PROMPTS=168
        CONCURRENCY=42
        TTFT_S=10
        TPOT_MS=
        INFLIGHT=37.0384
        LABEL="19 纯P 128k/1k（单前缀）"
        ;;
    21)
        # 纯D高吞吐，128k/1k（稳态），PrefixCache 100，只考核 TPOT
        PREFIX=131071
        UNIQUE=1
        OUTPUT=1024
        PROMPTS=4608
        CONCURRENCY=1152
        TTFT_S=
        TPOT_MS=50
        INFLIGHT=1112.64
        LABEL="21 纯D高吞吐 128k/1k（稳态）"
        ;;
    23)
        # 纯D低时延，128k/1k（稳态），PrefixCache 100，只考核 TPOT
        PREFIX=131071
        UNIQUE=1
        OUTPUT=1024
        PROMPTS=576
        CONCURRENCY=144
        TTFT_S=
        TPOT_MS=20
        INFLIGHT=144
        LABEL="23 纯D低时延 128k/1k（稳态）"
        ;;
    28)
        # 完整体验（1P1D），128k/1k（单前缀）（稳态），PrefixCache 90
        PREFIX=117965
        UNIQUE=13107
        OUTPUT=1024
        PROMPTS=960
        CONCURRENCY=96
        TTFT_S=10
        TPOT_MS=20
        INFLIGHT=92.07
        LABEL="28 完整体验 128k/1k（单前缀）（稳态）"
        ;;
    *)
        echo "用法: bash run_bench.sh 19|21|23|28" >&2
        exit 1
        ;;
esac

if [[ -z "${INSIDE_CONTAINER:-}" && ! -f /.dockerenv ]]; then
    exec docker exec -it "$CONTAINER_NAME" bash -lc "cd '$ROOT' && INSIDE_CONTAINER=1 bash run_bench.sh '$CASE'"
fi

mkdir -p "$ROOT/results"
OUT="$ROOT/results/row-${CASE}.json"
rm -f "$OUT"

echo "$LABEL"
echo "总输入 $((PREFIX + UNIQUE))，前缀 $PREFIX，输出 $OUTPUT，请求 $PROMPTS，最大并发 $CONCURRENCY"
echo "系统并发数对照 $INFLIGHT"
echo "结果 $OUT"

cmd=(
    vllm bench serve
    --backend openai-chat
    --base-url "http://${PREFILL_NODE_IP}:${PROXY_PORT}"
    --endpoint /v1/chat/completions
    --model "$MODEL_PATH"
    --served-model-name dsv41
    --dataset-name random
    --num-prompts "$PROMPTS"
    --random-input-len "$UNIQUE"
    --random-output-len "$OUTPUT"
    --random-prefix-len "$PREFIX"
    --random-range-ratio 0
    --request-rate inf
    --max-concurrency "$CONCURRENCY"
    --num-warmups 1
    --ignore-eos
    --temperature 0
    --seed 1024
    --percentile-metrics ttft,tpot,itl
    --metric-percentiles 50,99
    --save-result
    --result-dir "$ROOT/results"
    --result-filename "row-${CASE}.json"
)

if [[ -n "$TTFT_S" && -n "$TPOT_MS" ]]; then
    cmd+=(--goodput "ttft:$((TTFT_S * 1000))" "tpot:${TPOT_MS}")
elif [[ -n "$TTFT_S" ]]; then
    cmd+=(--goodput "ttft:$((TTFT_S * 1000))")
elif [[ -n "$TPOT_MS" ]]; then
    cmd+=(--goodput "tpot:${TPOT_MS}")
fi

"${cmd[@]}"

python3 - "$OUT" "$TTFT_S" "$TPOT_MS" "$INFLIGHT" <<'PY'
import json
import sys

path, ttft_s, tpot_ms, inflight = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

ok = True
completed = data.get("completed")
failed = data.get("failed") or 0
print(f"完成 {completed}，失败 {failed}")
if failed:
    ok = False

seen = data.get("max_concurrent_requests")
if seen is not None:
    print(f"观测最大并发 {seen}，表内系统并发数 {inflight}")

if ttft_s:
    got = float(data["mean_ttft_ms"])
    limit = float(ttft_s) * 1000
    flag = "通过" if got <= limit else "不通过"
    print(f"mean TTFT {got:.1f} ms，要求 ≤ {limit:.0f} ms，{flag}")
    ok = ok and got <= limit
else:
    got = data.get("mean_ttft_ms")
    print(f"mean TTFT {got} ms，本行不考核")

if tpot_ms:
    got = float(data["mean_tpot_ms"])
    limit = float(tpot_ms)
    flag = "通过" if got <= limit else "不通过"
    print(f"mean TPOT {got:.2f} ms，要求 ≤ {limit:.0f} ms，{flag}")
    ok = ok and got <= limit
else:
    print(f"mean TPOT {data.get('mean_tpot_ms')} ms，本行不考核")

sys.exit(0 if ok else 1)
PY
