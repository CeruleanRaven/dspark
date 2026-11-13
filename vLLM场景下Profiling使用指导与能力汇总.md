# vLLM 场景下 Profiling 使用指导与能力汇总

- 来源：<https://wiki.huawei.com/domains/52290/wiki/212179/WIKI202508047742544>
- 路径：MindStudio 开发部知识体系 / 【业务】性能调优 / 能力提升专区 / 赋能归档
- 创建：2025-08-04 11:57，方岚玥 00837000
- 最后修改：2026-04-15 10:55，方岚玥 00837000

原文中的目录树、时间线界面、代码截图没有随文本导出，下面在对应位置标注为「原 Wiki 配图」。

## 0. 须知

本文档针对 vLLM v0.9.0 介绍 Profiling 性能采集，可以参考其中的使能方式，但不建议直接复制命令。vLLM 社区每两周发布一个新版本，profiling 相关功能会随之更新。建议阅读 vLLM 社区对应版本中关于 profile 的说明。

## 1. 使用指导

vLLM-Ascend 推理框架通过集成 `torch_npu.profiler` 接口实现性能数据采集，使用参数与 Ascend PyTorch Profiler 的参数完全一致。

Ascend PyTorch Profiler 对标 PyTorch-GPU 场景下的使用方式，支持采集 PyTorch 层算子信息、CANN 层算子信息、底层 NPU 算子信息，以及算子内存占用信息，可以全方位分析 PyTorch AI 任务的性能状态。

### 1.1 Profiling 数据采集

通过环境变量 `VLLM_TORCH_PROFILER_DIR` 决定是否启用 Profiling 采集能力。

```bash
export VLLM_TORCH_PROFILER_DIR="./vllm_profile"
```

#### 脚本使能方式

对于离线推理，参照 vLLM 中的 `simple_profiling.py` 样例，通过 `start_profile()`、`stop_profile()` 启用 profiling：

```python
import os
import time
from vllm import LLM, SamplingParams

# enable torch profiler, can also be set on cmd line
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
# Create an LLM.
llm = LLM(model="Qwen2.5-0.5B-Instruct")
# 开启性能采集
llm.start_profile()
# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)
# 结束性能采集
llm.stop_profile()

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

time.sleep(10)
```

#### vLLM-benchmark 方式

Step 1：拉起在线服务。

```bash
VLLM_PROMPT_SEQ_BUCKET_MAX=128 VLLM_PROMPT_SEQ_BUCKET_MIN=128 \
python3 -m vllm.entrypoints.openai.api_server --port 8080 \
    --model "facebook/opt-125m" --tensor-parallel-size 1 \
    --max-num-seqs 128 --dtype bfloat16 \
    --max-model-len 256
```

Step 2：通过 benchmark 向服务发送固定长度请求，并采集 profiling。

```bash
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model "facebook/opt-125m" \
    --port 8080 \
    --dataset-name "random" --random-input-len 128 --random-output-len 4 \
    --random-range-ratio 1.0 \
    --ignore-eos \
    --profile \
    --max-concurrency "4" \
    --request-rate "inf" \
    --num-prompts 4 \
    --percentile-metrics "ttft,tpot,itl,e2el"
```

Step 3：解析 profiling。

```python
from torch_npu.profiler.profiler import analyse

if __name__ == "__main__":
    analyse(profiler_path="path/to/profiling")
```

#### curl 命令触发方式

Step 1：拉起在线服务。

```bash
VLLM_PROMPT_SEQ_BUCKET_MAX=128 VLLM_PROMPT_SEQ_BUCKET_MIN=128 \
python3 -m vllm.entrypoints.openai.api_server --port 8080 \
    --model "facebook/opt-125m" --tensor-parallel-size 1 \
    --max-num-seqs 128 --dtype bfloat16 \
    --max-model-len 256
```

Step 2：服务拉起后，发送开始性能采集请求。

```bash
curl -X POST http://localhost:8080/start_profile
```

Step 3：发送请求。

```bash
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-125m",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
}'
```

Step 4：发送停止性能采集请求。

```bash
curl -X POST http://localhost:8080/stop_profile
```

### 1.2 修改 profiling 配置参数

vLLM-Ascend 在初始化 NPUWorker 时创建 profiler 对象，具体代码在 `vllm-ascend/vllm_ascend/worker.py`（单算子下发）或 `worker_v1.py`（开启图模式）。如果默认配置不满足要求，比如 level、activity 等，可以手动修改 profiling 配置参数。

原 Wiki 配图：worker 中创建 profiler 对象的代码位置。

#### Ascend PyTorch Profiler 性能采集配置参数说明

**1. 基础参数配置**

`torch_npu.profiler.profile` 的参数体系对标原生 PyTorch Profiler，主要包含：

- `activities`：CPU、NPU 事件采集列表。
- `record_shapes`：算子的 InputShapes 和 InputTypes，Bool 类型。
- `profile_memory`：算子的内存占用情况，Bool 类型，默认为 False。
- `with_stack`：算子调用栈，Bool 类型，默认为 False。
- `schedule`：设置不同 step 的行为，Callable 类型。通常使用 `torch_npu.profiler.schedule` 作为参数，与 `profiler.step()` 共同起作用。
- `on_trace_ready`：设置采集结束时执行的操作，Callable 类型。通常选择 `tensorboard_trace_handler`，该接口用于解析采集的 Profiling 数据。

**2. NPU 专属扩展配置**

通过 `experimental_config` 提供昇腾 NPU 特有的性能采集项：

- `export_type`：导出的性能数据结果文件格式，List 类型，例如 `["db", "text"]`。
- `msprof_tx`：打点控制开关。开启后可以使用自定义打点，bool 类型。
- `mstx_domain_include`：输出需要的 domain 数据。调用 `torch_npu.npu.mstx` 系列打点接口时，可只输出本参数配置的 domain 数据。
- `mstx_domain_exclude`：过滤不需要的 domain 数据。
- `profiler_level`：设置 NPU 采集的 Level。

| ProfilerLevel | Profiling 数据 |
| --- | --- |
| Level0 | 框架侧及 Device 侧算子执行耗时，为默认值 |
| Level1 | 在 Level0 基础上增加 CANN 软件栈中 AscendCL 接口、HCCL 通信及 AI Core 的性能指标数据 |
| Level2 | 在 Level1 基础上增加 CANN 软件栈的 Runtime 数据及 AI CPU 数据 |

- `aic_metrics`：AI Core 的性能指标采集项。Level0 时该配置不生效。

| AicMetrics | Metrics 数据 |
| --- | --- |
| AiCoreNone | 关闭 AI Core 的性能指标采集，默认值 |
| PipeUtilization | 计算单元和搬运单元耗时占比 |
| ArithmeticUtilization | 各种计算类指标占比统计 |
| Memory | 外部内存读写类指令占比 |
| MemoryL0 | L0 内存读写类指令占比 |
| ResourceConflictRatio | 流水线队列类指令占比 |
| MemoryUB | 内部内存 |
| L2Cache | 读写 cache 命中次数和缺失后重新分配次数 |

- `l2_cache`：L2 cache 数据采集开关。

### 1.3 Profiling 采集结果

#### 交付件介绍

Profiling 结果输出到 `VLLM_TORCH_PROFILER_DIR` 指定的目录。每张 NPU 对应生成一个以 `ascend_pt` 结尾的文件夹。

原 Wiki 配图：`ascend_pt` 目录，以及各子目录的结构。

`ASCEND_PROFILER_OUTPUT` 文件夹存放解析后的性能数据，分两大类：

1. Timeline 文件（`trace_view.json`）
   - 使用 MindStudio Insight 打开，可视化展示 AI 任务运行时各层级算子的调用关系与执行时序。
2. Summary 文件
   - 多维度的统计摘要信息。
   - 典型文件：`kernel_details.csv`（kernel 详细信息）、`communication.json`（通信算子信息）等。

原 Wiki 配图：`ASCEND_PROFILER_OUTPUT` 目录中的文件列表。

#### 数据可视化

使用 MindStudio Insight 打开 timeline 文件，显示框架侧调用栈。采集时将 `with_module` 参数设为 True。

原 Wiki 配图：MindStudio Insight 中的框架侧调用栈。

从 Python 调用栈可以分析代码执行顺序和调用过程。Ascend Hardware 显示算子执行层，点击对应算子可显示算子执行时间和下发连线。通过 Overlap Analysis 分析通信、计算、空闲的比例，用于定位算子执行过程中是否存在空泡。

#### MSTX 性能自定义打点

mstx 接口是 MindStudio 提供的性能打点接口，允许在应用程序中插入标记，以便在性能分析时更精确地定位关键代码区域。

原 Wiki 配图：mstx 打点在时间线上的示意。

`torch_npu` 中封装的 mstx 接口：

- `torch_npu.npu.mstx.mark` 标识瞬时事件，例如「训练开始」。
- `range_start` / `range_end` 标识时间段事件，例如「前向」「反向」「allreduce」。
- 不传 `stream` 时只记录 CPU 侧时间，适用于纯 CPU 侧操作。传入 `stream` 时记录 CPU 侧时间及对应 NPU 侧时间，例如算子执行情况。
- 采集打点性能数据需要在 `experimental_config` 中设置 `msprof_tx=True`。当前已内置通信算子、dataloader 耗时、检查点接口耗时的性能数据采集。

样例 1：用 mark 打点迭代开始时间。

```python
class TrainModel:
    def __init__(self):
        self.input_shape = (4, 3, 24, 24)
        self.out_shape = (4, 12, 24, 24)
        self.device = torch.device("npu:0")
        self.model = SmallModel(self.input_shape[1], self.out_shape[1]).to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001)

    def train_one_step(self):
        stream = torch_npu.npu.current_stream()
        torch_npu.npu.mstx.mark("mark for train_one_step begin", stream)
        inputs = torch.rand(self.input_shape).to(self.device)
        target = torch.rand(self.out_shape).reshape(self.out_shape[0], -1).to(self.device)
        output = self.model(inputs)
        loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

样例 2：在计算流上打点，记录 Host 侧 range 耗时和 Device 侧对应的 range 耗时。

```python
class SmallModel(torch.nn.Module):
    def __init__(self, in_channel=3, out_channel=12):
        super(SmallModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, in_channel, 3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channel, out_channel, 3, padding=1)

    def forward(self, input_1):
        stream = torch_npu.npu.current_stream()
        mstx_conv1 = torch_npu.npu.mstx.range_start("start/end range for conv1", stream)
        input_1 = self.conv1(input_1)
        torch_npu.npu.mstx.range_end(mstx_conv1)

        mstx_relu1 = torch_npu.npu.mstx.range_start("start/end range for relu", stream)
        input_1 = self.relu1(input_1)
        torch_npu.npu.mstx.range_end(mstx_relu1)

        mstx_conv2 = torch_npu.npu.mstx.range_start("start/end range for conv2", stream)
        input_1 = self.conv2(input_1)
        torch_npu.npu.mstx.range_end(mstx_conv2)

        return input_1.reshape(input_1.shape[0], -1)
```

#### 模型结构拆解与算子关联分析能力

`msprof-analyze` 提供针对 PyTorch 模型自动解析模型层级结构的分析能力（`module_analysis`），用于定位性能瓶颈。该能力提供：

- 自动提取并展示模型的层次化结构，以及模型中的算子调用顺序。
- 建立框架层算子与 NPU 上执行 Kernel 的映射关系。
- 统计并输出 Device 侧 Kernel 的执行耗时。

原 Wiki 配图：模型层级拆解结果。

使用方式：

1. 修改 `vllm_ascend/worker/worker.py` 中的 profile 配置。
   - 将 `experimental_config` 中的 `msprof_tx` 设为 True，开启自定义打点。
   - 数据导出类型 `export_type` 增加 `db`。

   原 Wiki 配图：上述两处配置的修改位置。

2. 在脚本中添加模型层级打点。通过修改 PyTorch 的 `nn.Module.__call__`，在模型前向传播时插入 mstx range 打点。

```python
import os
import time
import torch
import torch_npu
import torch.nn as nn
from vllm import LLM, SamplingParams

original_call = nn.Module.__call__

def custom_call(self, *args, **kwargs):
    """自定义调用方法，添加 MSTX 打点"""
    module_name = self.__class__.__name__
    mstx_id = torch_npu.npu.mstx.range_start(module_name, domain="Module")
    tmp = original_call(self, *args, **kwargs)
    torch_npu.npu.mstx.range_end(mstx_id, domain="Module")
    return tmp

# 替换默认调用方法
nn.Module.__call__ = custom_call

# 通过 VLLM_TORCH_PROFILER_DIR 开启性能采集，也可以在终端设置该环境变量
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./result"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")

llm.start_profile()
outputs = llm.generate(prompts, sampling_params)
llm.stop_profile()

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

原 Wiki 配图：mstx 打点在时间线上的可视化效果。

3. 执行分析命令。

```bash
msprof-analyze cluster -m module_statistic -d ./result --export_type excel
```

演进计划：

- 增加 device 侧 PMU、详细计算耗时等细粒度展示。
- 增加 device 侧通信等待和传输耗时的拆解。
- 增加 profiling 的对比能力。
- 增加昇腾与 NVIDIA 的对比能力。

工具链接：mstt（pre-research）。

#### vLLM 图模式 Profiling 采集

问：图模式如何采集到下发连线、算子 shape 信息？Profiling 解析时报错 `Fail to get acl to npu flow events`？

答：当前 capture 阶段在 LLM 实例初始化环节，而 `start_profile` 在初始化之后，所以采集不到 capture，没有下发连线与算子 shape 等信息。临时解决方式是手动修改 vLLM 的 `vllm/v1/engine/core.py`，同时不再调用 `llm.start_profile()`。

原 Wiki 配图：`core.py` 的修改位置。要点是：

1. 在 EngineCore 初始化阶段开启 profile 采集。
2. `_initialize_kv_caches` 中执行 capture。
