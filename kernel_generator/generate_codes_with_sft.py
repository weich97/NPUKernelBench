import os
import textwrap
import json
import re
from typing import Optional, List
from framework.utils import string_with_emphasize, save_json_file, safe_read_file, find_case_data
from framework.task_object import TaskObject
from framework.kernel_gen_config import config, TestStage
from kernel_generator.full_sft_sample import FULL_EXAMPLE_CODE, FULL_CODE_REQUIREMENT
from kernel_generator.llm_api import llm_api
from kernel_generator.kernel_only_sft_sample import KERNEL_ONLY_EXAMPLE_CODE, KERNEL_ONLY_CODE_REQUIREMENT
from kernel_generator.code_gen import AscendCodeGen


class AscendCodeGenWithSft(AscendCodeGen):
    """
    Ascend code generator based on SFT (Supervised Fine-Tuning) examples.
    """

    def __init__(self, task_obj: TaskObject, logger=None):
        super().__init__(task_obj, logger)
        self.gen_codes_sys_prompt: Optional[str] = None
        self.gen_codes_user_prompt: Optional[str] = None

        # Define required file paths
        self.test_case_data = find_case_data(
            self.task_obj.get_file_path_from_definition("validation/test_cases.csv"),
            self.task_obj.case_id if config.static_shape_mode else None,
        )
        if config.kernel_only_mode:
            self.required_files: List[str] = [
                self.task_obj.get_file_path_from_definition(f"question/op_kernel/{self.task_obj.kernel_name}.cpp"),
            ]
        else:
            self.required_files: List[str] = [
                self.task_obj.get_file_path_from_definition(f"question/op_kernel/{self.task_obj.kernel_name}.cpp"),
                self.task_obj.get_file_path_from_definition(f"question/op_host/{self.task_obj.kernel_name}.cpp"),
            ]

        # Load API description and test case data
        self.api_description = safe_read_file(
            self.task_obj.get_file_path_from_definition("api_desc.md")
        )

    def generate_answer_template(self,
                                 kernel_impl: str = "// Insert kernel_impl.cpp content here",
                                 host_impl: str = "// Insert host_impl.cpp content here"
                                 ) -> str:
        """
        Generate the template for code output.
        """
        if config.kernel_only_mode:
            cpp_code = re.sub(r"(workspace[^)]*\))", "workspace)", kernel_impl)
            return textwrap.dedent(f"""
<kernel_impl>
```cpp
{cpp_code}
```
</kernel_impl>
""")
        else:
            host_impl = host_impl.replace("context->SetBlockDim(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());", "")
            host_impl = host_impl.replace("context->SetBlockDim(platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic());", "")
            return textwrap.dedent(f"""
================== op_host/{self.task_obj.kernel_name}_tiling.h ==================
```cpp
#include "register/tilingdata_base.h"

namespace optiling {{
BEGIN_TILING_DATA_DEF(TilingData)

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS({self.task_obj.op_name}, TilingData)
}}
```
================== op_host/{self.task_obj.kernel_name}.cpp ==================
```cpp
#include "{self.task_obj.kernel_name}_tiling.h"
{host_impl}
```
================== op_kernel/{self.task_obj.kernel_name}.cpp ==================
```cpp
{kernel_impl}
```

""")
    def enhance_api_description(self) -> str:
        """
        Enhance API description by adding data type support information.

        Returns:
            Enhanced API description string
        """
        if self.test_case_data is None or self.test_case_data.empty or 'dtype' not in self.test_case_data or not config.static_shape_mode:
            return self.api_description

        dtype_info = f"\n## 支持的数据类型\n- {self.test_case_data['dtype']}\n\n"
        enhanced_description = self.api_description.replace(
            "## 功能描述", dtype_info + "## 功能描述"
        )
        return enhanced_description

    def build_code_generation_prompts(self) -> None:
        """
        Build system and user prompts for Ascend C code generation.
        """
        # Load template files
        template_contents = [safe_read_file(file_path) for file_path in self.required_files]
        if config.kernel_only_mode:
            example_content = KERNEL_ONLY_EXAMPLE_CODE
            code_requirement = KERNEL_ONLY_CODE_REQUIREMENT
        else:
            example_content = FULL_EXAMPLE_CODE
            code_requirement = FULL_CODE_REQUIREMENT

        user_prompt = textwrap.dedent(f"""
你是一个精通Ascend C编程的专家。你的任务是根据提供的文档内容生成符合特定任务需求的Ascend C代码。

【任务背景】
我将提供3个关键文件，描述了待实现算子的信息，请仔细阅读每一个：
1. api_desc.md - 待实现算子的接口描述和公式定义相关内容，包含了算子的功能要求和数学表达式
2. test_cases.csv - 其设定了待实现算子需要重点考虑的datatype和shape信息，用于更合适的tiling设计和性能优化
3. hardware.txt - 其给出了运行待实现算子的硬件型号规格信息

【输出格式要求】
- 同时输出思维过程和答案，思维链进行合理的分析推导，答案部分必须严格按照提供的模板格式输出
- 不要修改已有的函数名、类定义、模板参数、命名空间等
- 确保答案部分的代码在XML标签内，每个文件对应一个标签块
- 确保检查算子参考示例写法正确注册了tiling结构，并根据输入shape信息，设计tiling参数及数值

【代码实现要点】
实现代码时，请确保：
{code_requirement}

{example_content}

接下来我将提供待实现算子所需文件信息，请参考这些信息按照要求完成所有指定文件的实现。
【输入文件】
下面是第一个文件："api_desc.md"，是待实现算子的接口描述和公式定义：
{self.enhance_api_description()}

下面是第二个文件："test_cases.csv"，其设定了待实现算子需要重点考虑的典型shape以及DataType信息，用于设计最优tiling策略：
{self.test_case_data}

下面是第三个文件："hardware.txt"，其给出的是硬件信息，待实现算子将要在该硬件上运行：
【
# [Platform Info configuration begin]
#**************************************************************************************
#
[version]
SoC_version=Ascend910_9392
Short_SoC_version=Ascend910_93   [注：Ascend910_93为NPU型号名]

[SoCInfo]
ai_core_cnt=24
cube_core_cnt=24
vector_core_cnt=48

[AICoreSpec]
ub_size=196608
】

【输出模板】
你需要输出的文件模板如下，请对其进行填补：
{self.generate_answer_template(*template_contents)}

【最终任务】
请分析所有提供的信息，特别关注：
 - 算子描述中的功能要求、公式定义和实现细节
 - 测试用例中提供的datatype和shape等信息
 - 输入准备代码中的参数构造和使用方法

请提供完整的思维过程和符合要求的代码答案。"
        """)

        self.gen_codes_user_prompt = user_prompt

        if self.logger:
            self.logger.debug(f"System prompt for code generation:{string_with_emphasize(self.gen_codes_sys_prompt)}")
            self.logger.debug(f"User prompt for code generation:{string_with_emphasize(self.gen_codes_user_prompt)}")

    def process(self):
        """
        Process code generation based on configuration.

        Returns:
            Generated or loaded content
        """
        try:
            if TestStage.CODE_GEN in config.active_stages:
                self.build_code_generation_prompts()
                reasoning_content, generated_content = llm_api.call_api_vllm(
                    sys_prompt=self.gen_codes_sys_prompt or "",
                    user_prompt=self.gen_codes_user_prompt,
                    temperature=config.chat.temperature,
                    max_tokens=config.chat.max_tokens,
                )
                result = f"\n{'-' * 108}\n"
                result += "=" * 50 + "[COT]" + "=" * 50 + "\n" + reasoning_content + "\n"
                result += "=" * 50 + "[Reply]" + "=" * 50 + "\n" + generated_content + "\n"
                result += f"{'-' * 52}END {'-' * 52}\n"
                # self.logger.info(result)

                if not generated_content or generated_content == "Generation Failed":
                    error_msg = f"Empty or failed generation for task {self.task_obj.short_id}"
                    if self.logger:
                        self.logger.error(error_msg)
                    raise RuntimeError(error_msg)

                # Combine reasoning and generated content
                final_content = f"{reasoning_content}\n{generated_content}"
                self.task_obj.gen_content = final_content
                self.task_obj.parse_gen_content()

                # Save generation data
                generation_data = {
                    "gen_codes_sys_prompt": self.gen_codes_sys_prompt,
                    "gen_codes_user_prompt": self.gen_codes_user_prompt,
                    "gen_content": generated_content,
                    "reasoning_content": reasoning_content,
                }
                self.task_obj.save_code_gen_content(generation_data, self.logger)
                return final_content

            else:
                return self.task_obj.load_code_gen_content(self.logger)

        except Exception as e:
            error_msg = f"Processing failed for task {self.task_obj.short_id}: {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
