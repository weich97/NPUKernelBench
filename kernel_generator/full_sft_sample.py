FULL_EXAMPLE_CODE = '''
'''

FULL_CODE_REQUIREMENT = '''
1. The implementation only needs to satisfy the input information required by the current `test_cases.csv`; functionality described in `api_desc.md` that is irrelevant to the current inputs may be omitted.
2. Design tiling and data movement according to the hardware specification, and avoid out-of-bounds accesses or invalid internal addresses.
3. Keep the syntax rigorous and correct. Re-check the generated code to avoid undefined variables, undefined classes, or non-executable code paths.
4. Do not use placeholder names such as `DTYPE_X` or `DTYPE_Y` to represent concrete data types.
5. The official Ascend C type name for bfloat16 is `bfloat16_t`, not `bfloat16`.
6. The official Ascend C type name for float16 is `float16_t`, not `float16`.
7. The host side may choose the number of active cores according to task requirements.
8. When using `DataCopy`, both the transfer length and the operand start address in UB must satisfy 32-byte alignment.
'''
