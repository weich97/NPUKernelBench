import argparse


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive kernel tests including compilation, precision, and performance evaluation.')

    parser.add_argument('-chat', '--chat', action='store_true',
                        help='Enable chat-based code generation')
    parser.add_argument('-run_dir', '--run-dir', type=str, default=None, 
                        help='Specifies the directory for the experiment run')    
    parser.add_argument('-n_sample', '--n-sample', type=int, default=None,
                        help='number of samples in chat mode')
    parser.add_argument('-n_case', '--n-case', type=int, default=None,
                        help='number of cases in chat mode')
    parser.add_argument('-test_mode', '--test-mode', type=str, default=None,
                        choices=['static', 'dynamic'], help='Test shape mode')
    parser.add_argument('-template_mode', '--template-mode', type=str, default=None,
                        choices=['kernel_only', 'full'], help='Template mode')
    parser.add_argument('-stages', '--stages', nargs='+',
                        choices=['perf', 'precision', 'code_gen', 'compile'],
                        default=['code_gen', 'compile', 'precision'],
                        help='stages need to perform (choose from: perf, precision, code_gen, compile)')

    parser.add_argument('-task_name', '--task-name', nargs='+',
                        help='list of task names to test (e.g., Matmul Add Conv2D)')
    parser.add_argument('-config', '--config-path', type=str, default="base_config.yaml",
                        help='config path, reletive to project')

    return parser.parse_args()