import os
import time
import traceback
from openai import OpenAI
from typing import Dict
from dataclasses import dataclass
import sglang as sgl
from functools import wraps

from framework.logger import init_logger
from framework.kernel_gen_config import config

logger = init_logger(__name__)


def retry_on_failure(max_retries=None, delay=None):
    """Decorator for retrying API calls on failure."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = max_retries or config.chat.api_max_retries
            retry_delay = delay or config.chat.api_retry_delay

            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < retries - 1:
                        logger.warning(f"API call failed (attempt {attempt + 1}/{retries}): {e}")
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"API call failed after {retries} attempts: {e}")
                        raise
            return None

        return wrapper

    return decorator


@dataclass
class APIConfig:
    """Configuration for LLM API calls."""

    def __init__(self, model_name: str, api_key: str, base_url: str, is_stream_api: bool):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.is_stream_api = is_stream_api

    def setup_default_sglang_backend(self):
        """Setup default sglang backend for API calls."""
        sgl.set_default_backend(
            sgl.OpenAI(
                self.model_name,
                base_url=self.base_url,
                api_key=self.api_key,
                is_chat_model=True,
            )
        )
        logger.info(f"SGLang default backend configured: {sgl.global_config.default_backend}")

    @retry_on_failure()
    def call_api(self, sys_prompt: str, user_prompt: str, temperature=0.7, max_tokens=2048):
        """Make a single API call with system and user prompts."""

        @sgl.function
        def api_call(s):
            s += sgl.system(sys_prompt)
            s += sgl.user(user_prompt)
            s += sgl.assistant(sgl.gen(
                name="result",
                max_tokens=max_tokens,
                temperature=temperature
            ))

        response = api_call.run(stream=self.is_stream_api)["result"]
        return response

    def call_api_vllm(self, user_prompt: str, sys_prompt: str = "", temperature=0.7, max_tokens=16384):
        """Call vLLM API with DeepSeek reasoning format."""
        client = OpenAI(
            api_key="YOUR_API_KEY",
            base_url="http://0.0.0.0:5600/v1/"
        )

        for attempt in range(config.chat.api_max_retries):
            try:
                # Format prompt for DeepSeek reasoning
                formatted_prompt = "<|im_start|>user\n\(%s)<|im_end|>\n<|im_start|>assistant\n<think>" % user_prompt

                completion = client.completions.create(
                    model=self.model_name,
                    prompt=formatted_prompt,
                    stream=False,
                    n=1,
                    top_p=0.9,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                content = completion.choices[0].text
                return "", content

            except Exception as e:
                error_details = traceback.format_exc()
                logger.error(f"vLLM API call failed (attempt {attempt + 1}): {error_details}")
                if attempt < config.chat.api_max_retries - 1:
                    time.sleep(config.chat.api_retry_delay)

        return "Generation Failed", "Generation Failed"

    @retry_on_failure()
    def call_api_batch(self, sys_prompt_list: list[str], user_prompt_list: list[str],
                       temperature=0.7, max_tokens=2048):
        """Make batch API calls for multiple prompts."""

        @sgl.function
        def batch_api_call(s, sys_prompt, user_prompt):
            s += sgl.system(sys_prompt)
            s += sgl.user(user_prompt)
            s += sgl.assistant(sgl.gen(
                name="result",
                max_tokens=max_tokens,
                temperature=temperature
            ))

        # Prepare batch inputs
        batched_inputs = []
        for idx in range(len(sys_prompt_list)):
            batched_inputs.append({
                "sys_prompt": sys_prompt_list[idx],
                "user_prompt": user_prompt_list[idx] if len(user_prompt_list) > idx else ""
            })

        response_list = batch_api_call.run_batch(batched_inputs, stream=self.is_stream_api)

        results = []
        for idx, current_response in enumerate(response_list):
            results.append(current_response["result"])
            logger.debug(f"Batch response {idx}: {current_response.messages()}")

        return results

    @retry_on_failure()
    def call_multi_turn_api(self, initial_sys_prompt: str, initial_user_prompt: str,
                            follow_up_prompts: list = None, temperature=0.7, max_tokens=2048):
        """
        Multi-turn conversation API call with retry mechanism.

        Args:
            initial_sys_prompt: Initial system prompt
            initial_user_prompt: Initial user prompt
            follow_up_prompts: List of follow-up prompts, each is a dict with 'role' and 'content'
                              Example: [{'role': 'system', 'content': 'New system prompt'},
                                       {'role': 'user', 'content': 'Follow-up question'}]
            temperature: Model temperature parameter
            max_tokens: Maximum tokens to generate

        Returns:
            List containing assistant replies for each turn
        """
        follow_up_prompts = follow_up_prompts or []

        @sgl.function
        def multi_turn_api_call(s):
            # Initial system and user prompts
            s += sgl.system(initial_sys_prompt)
            s += sgl.user(initial_user_prompt)

            # First assistant reply
            s += sgl.assistant(sgl.gen(
                name="response_0",
                max_tokens=max_tokens,
                temperature=temperature
            ))

            # Process follow-up turns
            for i, prompt in enumerate(follow_up_prompts):
                if prompt['role'].lower() == 'system':
                    s += sgl.system(prompt['content'])
                elif prompt['role'].lower() == 'user':
                    s += sgl.user(prompt['content'])

                # Generate assistant reply after user prompts
                if prompt['role'].lower() == 'user':
                    s += sgl.assistant(sgl.gen(
                        name=f"response_{i + 1}",
                        max_tokens=max_tokens,
                        temperature=temperature
                    ))

        # Execute and extract results
        response = multi_turn_api_call.run(stream=self.is_stream_api)

        results = []
        i = 0
        while f"response_{i}" in response:
            results.append(response[f"response_{i}"])
            i += 1

        return results


class APIConfigFactory:
    """Factory for creating and managing API configurations."""

    _configurations: Dict[str, APIConfig] = {
        "deepseek-r1": APIConfig(
            model_name="deepseek-reasoner",
            api_key="sk-c80e688b2d544f16825ceca256d3c6e2",
            base_url="https://api.deepseek.com",
            is_stream_api=False
        ),
        "huoshan-deepseek-r1": APIConfig(
            model_name="deepseek-r1-250120",
            api_key="9b1afd3f-bbe8-493f-9ba4-c17784556f0c",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            is_stream_api=False
        ),
        "huoshan-deepseek-r1-distill-qwen-7b": APIConfig(
            model_name="deepseek-r1-distill-qwen-7b-250120",
            api_key="9b1afd3f-bbe8-493f-9ba4-c17784556f0c",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            is_stream_api=False
        ),
        "qwq-official": APIConfig(
            model_name="qwq-32b",
            api_key="sk-8db036d3267f461bb042c9c9b14d8143",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            is_stream_api=True
        ),
        "vllm": APIConfig(
            model_name=config.chat.model_path,
            api_key="YOUR_API_KEY",
            base_url="http://0.0.0.0:5600/v1/",
            is_stream_api=False
        )
    }

    @classmethod
    def list_available_configs(cls):
        """List all available API configuration names."""
        return list(cls._configurations.keys())

    @classmethod
    def get_llm_api(cls, api_name: str) -> APIConfig:
        """Get API configuration by name, fallback to default if not found."""
        return cls._configurations.get(api_name, cls._configurations["deepseek-r1"])

    @classmethod
    def register_config(cls, name: str, api_config: APIConfig):
        """Register a new API configuration."""
        cls._configurations[name] = api_config
        logger.info(f"Registered new API configuration: {name}")


# Initialize default LLM API
llm_api = APIConfigFactory.get_llm_api(config.chat.api_model)
llm_api.setup_default_sglang_backend()

logger.info(f"LLM API initialized with model: {llm_api.model_name}")