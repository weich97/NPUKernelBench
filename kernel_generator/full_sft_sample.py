FULL_EXAMPLE_CODE = '''
'''

FULL_CODE_REQUIREMENT = '''
1. 代码实现能满足当前test_cases.csv要求的输入信息即可，api_desc.md描述中与当前输入信息无关的功能可以不实现
2. 注意参考硬件规格信息来进行分块及搬运设计，不要出现内部地址越界等问题
3. 注意语法严谨和正确，生成的过程中反复检查，不要出现任何未定义的变量和类，保证代码可执行和功能正确
4. 注意代码中不要使用DTYPE_X 和 DTYPE_Y等包含DTYPE的命名来指代数据类型
5. 注意Ascend C官方定义的bfloat16类型名是bfloat16_t，而不是bfloat16
6. 注意Ascend C官方定义的float16类型名是float16_t，而不是float16
7. 注意host侧可根据实际需求自定义核数量
8. 使用DataCopy接口进行数据搬运，搬运的数据长度和操作数的起始地址（UB上）必须保证32字节对齐 
'''