#!/usr/bin/env bash
# A3 容器。Prefill、Decode 两台都执行。
# 社区文档 4.1 的设备、网络、ipc 和挂载保持不变；容器改为常驻，便于随后启动服务。
set -euo pipefail

MODEL_ROOT=/mnt/share/h00990513/models
IMAGE=quay.io/ascend/vllm-ascend:v0.30.0-a3
CONTAINER_NAME=deepseek-v41

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -d "$MODEL_ROOT" ]]; then
    echo "MODEL_ROOT 不存在: $MODEL_ROOT" >&2
    exit 1
fi

docker pull "$IMAGE"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker rm -f "$CONTAINER_NAME"
fi

docker run -d \
    --name "$CONTAINER_NAME" \
    --net=host \
    --ipc=host \
    --privileged=true \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci8 \
    --device /dev/davinci9 \
    --device /dev/davinci10 \
    --device /dev/davinci11 \
    --device /dev/davinci12 \
    --device /dev/davinci13 \
    --device /dev/davinci14 \
    --device /dev/davinci15 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /etc/hccn.conf:/etc/hccn.conf \
    -v "$MODEL_ROOT:$MODEL_ROOT" \
    -v "$ROOT:$ROOT" \
    "$IMAGE" \
    bash -lc "sleep infinity"

echo "容器已启动: $CONTAINER_NAME"
docker exec "$CONTAINER_NAME" npu-smi info
docker exec "$CONTAINER_NAME" vllm --version
