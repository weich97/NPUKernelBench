# Start the vLLM Service

This guide describes how to launch a vLLM-compatible service for LLM-based kernel generation in NPUKernelBench.

## 1. Configure the Model Path

Set the model checkpoint path in `base_config.yaml`:

```yaml
chat:
  model_path: "/path/to/model/checkpoint"
```

The same value is consumed by `start_vllm_server.sh` through the framework configuration.

## 2. Review Server Parameters

The launch script exposes the main serving parameters:

- `--model`: model name or checkpoint path;
- `--port 5600`: serving port used by `kernel_generator/llm_api.py`;
- `--gpu-memory-utilization 0.7`: target device-memory utilization;
- `--tensor-parallel-size 8`: tensor parallelism across eight Ascend devices;
- `--max-num-batched-tokens 32768`: maximum batched tokens;
- `--max-num-seqs 16`: maximum number of concurrent sequences;
- `--max-model-len 32768`: maximum sequence length;
- `--enable-chunked-prefill`: enable chunked prefill for improved throughput;
- `--enable-prefix-caching`: enable prefix caching;
- `--enable-reasoning`: enable reasoning output support;
- `--reasoning-parser deepseek_r1`: parse reasoning traces with the DeepSeek-R1 parser.

If the port is changed, update the corresponding `base_url` in the vLLM API call path.

## 3. Launch the Service

Start the service in the background:

```bash
nohup bash start_vllm_server.sh > vllm_server.log 2>&1 &
```

Monitor `vllm_server.log` to confirm that the service has loaded the model and is accepting requests.

## 4. Run Code Generation

After the service is available, run a generation stage:

```bash
python run_multi_test.py -chat -task_name Sqrt -stages code_gen
```

Generated code and logs are written under the configured `run_dir`.
