#!/usr/bin/env bash
# A3 1P1D 服务。社区文档 5.2.1。
# 用法:
#   Prefill 节点:  bash start_service.sh prefill
#   Decode 节点:   bash start_service.sh decode
#   两边都起来后，在 Prefill 节点:  bash start_service.sh proxy
set -euo pipefail

MODEL_PATH=/mnt/share/h00990513/models/DeepSeek-V4.1-Flash-1017-w4a8-03-mtp
PREFILL_NODE_IP=
DECODE_NODE_IP=
NIC_NAME=
CONTAINER_NAME=deepseek-v41
PROXY_PORT=2999

ROLE=${1:-}
if [[ "$ROLE" != prefill && "$ROLE" != decode && "$ROLE" != proxy ]]; then
    echo "用法: bash start_service.sh prefill|decode|proxy" >&2
    exit 1
fi

: "${PREFILL_NODE_IP:?请在脚本顶部设置 PREFILL_NODE_IP}"
: "${DECODE_NODE_IP:?请在脚本顶部设置 DECODE_NODE_IP}"
if [[ "$ROLE" != proxy ]]; then
    : "${NIC_NAME:?请在脚本顶部设置 NIC_NAME}"
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -z "${INSIDE_CONTAINER:-}" && ! -f /.dockerenv ]]; then
    exec docker exec -it "$CONTAINER_NAME" bash -lc "cd '$ROOT' && INSIDE_CONTAINER=1 bash start_service.sh '$ROLE'"
fi

prepare_runtime() {
    local local_ip=$1
    unset https_proxy
    unset http_proxy
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/lib/"
    export HCCL_IF_IP="$local_ip"
    export GLOO_SOCKET_IFNAME="$NIC_NAME"
    export TP_SOCKET_IFNAME="$NIC_NAME"
    export HCCL_SOCKET_IFNAME="$NIC_NAME"
    export VLLM_USE_V2_MODEL_RUNNER=1
    export VLLM_ENGINE_READY_TIMEOUT_S=36000
    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
}

serve_prefill() {
    prepare_runtime "$PREFILL_NODE_IP"
    export HCCL_BUFFSIZE=1024
    export HCCL_OP_EXPANSION_MODE="AIV"
    export ASCEND_RT_VISIBLE_DEVICES=$1
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$2" \
        --data-parallel-size "$3" \
        --data-parallel-rank "$4" \
        --data-parallel-address "$5" \
        --data-parallel-rpc-port "$6" \
        --tensor-parallel-size "$7" \
        --prefill-context-parallel-size "$8" \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name dsv41 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 8192 \
        --max-num-seqs 16 \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
        --trust-remote-code \
        --block-size 128 \
        --tokenizer-mode deepseek_v41 \
        --reasoning-parser deepseek_v41 \
        --tool-call-parser deepseek_v41 \
        --enable-auto-tool-choice \
        --safetensors-load-strategy lazy \
        --gpu-memory-utilization 0.9 \
        --quantization ascend \
        --enforce-eager \
        --swa-bounded-replay \
        --enable-prefix-caching \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --additional-config '{
            "enable_cpu_binding":true,
            "enable_fused_mc2":2,
            "enable_flashcomm1":false,
            "multistream_overlap_shared_expert":false
        }' \
        --kv-transfer-config '{
            "kv_connector":"MooncakeHybridConnector",
            "kv_role":"kv_producer",
            "kv_port":"30000",
            "kv_connector_extra_config":{
                "prefill":{"dp_size":4,"tp_size":1},
                "decode":{"dp_size":16,"tp_size":1}
            }
        }'
}

serve_decode() {
    prepare_runtime "$DECODE_NODE_IP"
    export HCCL_BUFFSIZE=4096
    export ASCEND_RT_VISIBLE_DEVICES=$1
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port "$2" \
        --data-parallel-size "$3" \
        --data-parallel-rank "$4" \
        --data-parallel-address "$5" \
        --data-parallel-rpc-port "$6" \
        --tensor-parallel-size "$7" \
        --prefill-context-parallel-size "$8" \
        --enable-expert-parallel \
        --seed 1024 \
        --served-model-name dsv41 \
        --max-model-len 1048576 \
        --max-num-batched-tokens 512 \
        --max-num-seqs 72 \
        --async-scheduling \
        --block-size 128 \
        --no-enable-prefix-caching \
        --no-swa-bounded-replay \
        --trust-remote-code \
        --tokenizer-mode deepseek_v41 \
        --reasoning-parser deepseek_v41 \
        --tool-call-parser deepseek_v41 \
        --enable-auto-tool-choice \
        --safetensors-load-strategy lazy \
        --gpu-memory-utilization 0.95 \
        --quantization ascend \
        --speculative-config '{"num_speculative_tokens":5,"method":"dspark"}' \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
        --kv-transfer-config '{
            "kv_connector":"MooncakeHybridConnector",
            "kv_role":"kv_consumer",
            "kv_port":"30100",
            "kv_connector_extra_config":{
                "prefill":{"dp_size":4,"tp_size":1},
                "decode":{"dp_size":16,"tp_size":1}
            }
        }' \
        --engram-config '{"cpu_offload":true,"dp_shared_memory":true}' \
        --additional-config '{
            "ascend_compilation_config":{
                "enable_npugraph_ex":true,
                "enable_static_kernel":false
            },
            "enable_cpu_binding":true,
            "multistream_overlap_shared_expert":true,
            "recompute_scheduler_enable":true
        }'
}

launch_dp() {
    local dp_size=$1
    local dp_local=$2
    local tp_size=$3
    local pcp_size=$4
    local dp_address=$5
    local devices_per_rank=$((tp_size * pcp_size))
    local i dp_rank port first last devices
    for ((i = 0; i < dp_local; i++)); do
        dp_rank=$i
        port=$((7100 + i))
        first=$((i * devices_per_rank))
        last=$((first + devices_per_rank - 1))
        devices=$(seq -s, "$first" "$last")
        if [[ "$ROLE" == prefill ]]; then
            serve_prefill "$devices" "$port" "$dp_size" "$dp_rank" "$dp_address" 12321 "$tp_size" "$pcp_size" &
        else
            serve_decode "$devices" "$port" "$dp_size" "$dp_rank" "$dp_address" 12321 "$tp_size" "$pcp_size" &
        fi
    done
    wait
}

start_proxy() {
    local proxy_script=${PROXY_SCRIPT:-}
    if [[ -z "$proxy_script" || ! -f "$proxy_script" ]]; then
        proxy_script=$(find /vllm-workspace /workspace /opt /usr/local -name load_balance_proxy_server_example.py 2>/dev/null | head -n 1 || true)
    fi
    if [[ -z "$proxy_script" || ! -f "$proxy_script" ]]; then
        echo "找不到 load_balance_proxy_server_example.py" >&2
        exit 1
    fi

    local prefill_hosts=() prefill_ports=() decode_hosts=() decode_ports=() port
    for port in 7100 7101 7102 7103; do
        prefill_hosts+=("$PREFILL_NODE_IP")
        prefill_ports+=("$port")
    done
    for port in $(seq 7100 7115); do
        decode_hosts+=("$DECODE_NODE_IP")
        decode_ports+=("$port")
    done

    python3 "$proxy_script" \
        --host 0.0.0.0 --port "$PROXY_PORT" \
        --prefiller-hosts "${prefill_hosts[@]}" \
        --prefiller-ports "${prefill_ports[@]}" \
        --decoder-hosts "${decode_hosts[@]}" \
        --decoder-ports "${decode_ports[@]}"
}

case "$ROLE" in
    prefill)
        launch_dp 4 4 1 4 "$PREFILL_NODE_IP"
        ;;
    decode)
        launch_dp 16 16 1 1 "$DECODE_NODE_IP"
        ;;
    proxy)
        start_proxy
        ;;
esac
