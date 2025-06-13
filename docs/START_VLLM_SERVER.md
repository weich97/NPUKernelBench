# Start vLLM Server

本文档介绍如何启动 vLLM Server 并使用其进行代码生成任务。

## 1. 参数配置

### 1.1 修改配置文件

1. **修改 `base_config.yaml` 文件**：
   - `chat: model_path`：设置为代码生成模型的文件路径

2. **修改 `start_vllm_server.sh` 文件**中的关键参数：

   - `--model $model_name`：模型名称或路径（在 base_config.yaml 中配置）
   - `--port 5600`：服务端口，建议保持默认值，如需修改请同步更新 llm_api.py 中 call_api_vllm 函数的 base_url 参数
   - `--gpu-memory-utilization 0.7`：NPU 内存使用率，避免过小或过大
   - `--tensor-parallel-size 8`：在 8 个 NPU 上并行处理
   - `--max-num-batched-tokens 32768`：每个 batch 的最大 token 数
   - `--max-num-seqs 16`：每批次最大序列数
   - `--max-model-len 32768`：单序列最大 token 长度
   - `--enable-chunked-prefill`：启用分块预填充，提高效率
   - `--enable-prefix-caching`：启用前缀缓存，加速推理
   - `--enable-reasoning`：启用推理功能
   - `--reasoning-parser deepseek_r1`：指定推理解析器

## 2. 启动 vLLM 服务

在后台启动 vLLM 服务：

```bash
nohup bash start_vllm_server.sh > vllm_server.log 2>&1 &
```

## 3. 执行代码生成

完成配置和服务启动后，即可开始代码生成任务。