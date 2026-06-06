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

        # Define required file paths.
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

        # Load API description and test case data.
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
            Enhanced API description string.
        """
        if self.test_case_data is None or self.test_case_data.empty or 'dtype' not in self.test_case_data or not config.static_shape_mode:
            return self.api_description

        dtype_info = f"\n## Supported Data Types\n- {self.test_case_data['dtype']}\n\n"
        markers = ["## Functional Description", "## Function Description", "## Operator Function"]
        for marker in markers:
            if marker in self.api_description:
                return self.api_description.replace(marker, dtype_info + marker, 1)
        return dtype_info + self.api_description

    def build_code_generation_prompts(self) -> None:
        """
        Build system and user prompts for Ascend C code generation.
        """
        template_contents = [safe_read_file(file_path) for file_path in self.required_files]
        if config.kernel_only_mode:
            example_content = KERNEL_ONLY_EXAMPLE_CODE
            code_requirement = KERNEL_ONLY_CODE_REQUIREMENT
        else:
            example_content = FULL_EXAMPLE_CODE
            code_requirement = FULL_CODE_REQUIREMENT

        user_prompt = textwrap.dedent(f"""
You are an expert in Ascend C programming. Your task is to generate Ascend C code that satisfies the provided operator specification and task-specific constraints.

[Task Context]
You will receive three key inputs that describe the operator to be implemented. Read each input carefully:
1. api_desc.md: the operator interface description, formula definition, functional requirements, and mathematical semantics.
2. test_cases.csv: the data type and shape information that must be prioritized when designing tiling parameters and optimizing performance.
3. hardware.txt: the target hardware specification for the generated operator.

[Output Format Requirements]
- Provide both reasoning and the final answer. The reasoning should explain the implementation strategy, and the answer must strictly follow the output template.
- Do not modify existing function names, class definitions, template parameters, namespaces, or protected template interfaces.
- Ensure that the answer code is enclosed in the required XML-style tags; each file must correspond to its own tag block or template section.
- Verify that the implementation follows the reference example for tiling-data registration and uses the input shape information to determine concrete tiling parameters.

[Implementation Requirements]
When implementing the code, ensure that:
{code_requirement}

{example_content}

The following sections provide the files required for the target operator. Use them to complete every requested output file.

[Input File 1: api_desc.md]
This file defines the operator interface, formula, semantics, and constraints:
{self.enhance_api_description()}

[Input File 2: test_cases.csv]
This file specifies the representative shape and data type that should guide the tiling strategy:
{self.test_case_data}

[Input File 3: hardware.txt]
The target operator will run on the following hardware:
# [Platform Info configuration begin]
#**************************************************************************************
#
[version]
SoC_version=Ascend910_9392
Short_SoC_version=Ascend910_93   [Note: Ascend910_93 is the NPU model name]

[SoCInfo]
ai_core_cnt=24
cube_core_cnt=24
vector_core_cnt=48

[AICoreSpec]
ub_size=196608

[Output Template]
Fill in the following template exactly:
{self.generate_answer_template(*template_contents)}

[Final Task]
Analyze all provided information, with particular attention to:
 - functional requirements, formulas, and implementation details in the operator description;
 - data type and shape information in the test cases;
 - parameter construction and usage patterns in the input-preparation code.

Provide a complete reasoning process and a code answer that satisfies the required format.
        """)

        self.gen_codes_user_prompt = user_prompt

        if self.logger:
            self.logger.debug(f"System prompt for code generation:{string_with_emphasize(self.gen_codes_sys_prompt)}")
            self.logger.debug(f"User prompt for code generation:{string_with_emphasize(self.gen_codes_user_prompt)}")

    def process(self):
        """
        Process code generation based on configuration.

        Returns:
            Generated or loaded content.
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

                final_content = f"{reasoning_content}\n{generated_content}"
                self.task_obj.gen_content = final_content
                self.task_obj.parse_gen_content()

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
