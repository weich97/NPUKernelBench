import os
import shutil
import argparse
import pandas as pd
from enum import Enum
import re
from typing import List
from rich.console import Console

from framework.arg_parser import parse_arguments
from framework.kernel_gen_config import config, TestStage
from framework.batch_code_gen import group_task_objects_by_problem, batch_code_gen
from kernel_generator.generate_codes_with_sft import AscendCodeGenWithSft
from framework.task_object import TaskObject, CompileResult, CodeGenResult, PrecisionResult
from framework.batch_compile import batch_compile
from framework.batch_precision_eval import batch_precision_test
from framework.batch_performance_eval import batch_performance_test


class TaskDiscovery:
    """Helper class for discovering and filtering tasks."""

    @staticmethod
    def get_all_task_paths(base_dir="tasks"):
        """Get all task paths from the tasks directory structure."""
        task_paths = []

        if not os.path.exists(base_dir):
            print(f"Warning: Directory {base_dir} does not exist")
            return task_paths

        # Traverse directory structure: tasks/level*/category/operator
        for level_dir in os.listdir(base_dir):
            level_path = os.path.join(base_dir, level_dir)
            if not os.path.isdir(level_path) or not level_dir.startswith('level'):
                continue

            for category_dir in os.listdir(level_path):
                category_path = os.path.join(level_path, category_dir)
                if not os.path.isdir(category_path):
                    continue

                for operator_dir in os.listdir(category_path):
                    operator_path = os.path.join(category_path, operator_dir)
                    if os.path.isdir(operator_path):
                        # Use forward slashes for consistency
                        task_path = f"{base_dir}/{level_dir}/{category_dir}/{operator_dir}"
                        task_paths.append(task_path)

        return sorted(task_paths)

    @staticmethod
    def filter_tasks_by_names(all_task_paths, task_names):
        """Filter task paths by specified task names."""
        if not task_names:
            return all_task_paths

        filtered_paths = []
        task_name_set = set(name.strip() for name in task_names)

        for path in all_task_paths:
            # Extract operator name (last directory level)
            operator_name = path.split('/')[-1]
            if operator_name in task_name_set:
                filtered_paths.append(path)

        # Check for missing tasks
        found_names = set(path.split('/')[-1] for path in filtered_paths)
        missing_names = task_name_set - found_names
        if missing_names:
            print(f"Warning: The following tasks were not found: {', '.join(missing_names)}")

        return filtered_paths


class TestRunner:
    """Main test runner class."""

    @staticmethod
    def check_stage_pass(task: TaskObject, check_stages: List[TestStage]):
        """Check if task passes all specified stages."""
        for stage in check_stages:
            result = getattr(task, stage.value)
            if result is None or not result.success:
                return False, stage
        return True, TestStage.FINAL

    @staticmethod
    def remove_previous_results(task_obj_list):
        """Remove previous generation results for clean start."""
        for task_obj in task_obj_list:
            if os.path.exists(task_obj.work_path):
                shutil.rmtree(task_obj.work_path)

    def run_testing_pipeline(self, task_objs: List[TaskObject], args):
        """Execute the complete testing pipeline."""
        # Code generation stage
        if config.chat.active:
            if TestStage.CODE_GEN in config.active_stages:
                self.remove_previous_results(task_objs)
                batch_code_gen(
                    task_objs,
                    num_processes=config.chat.num_processes,
                    timeout=config.chat.timeout,
                    code_gen_args=dict(),
                    code_gen_class=AscendCodeGenWithSft,
                    postfix="sft"
                )

            else:
                for task in task_objs:
                    task.code_gen_result = CodeGenResult(os.path.exists(task.prompt_save_file_path),
                                                         log_file=task.code_gen_log_path, response=None)

        else:
            for task in task_objs:
                task.code_gen_result = CodeGenResult(True, log_file=task.code_gen_log_path, response=None)

        # Generate statistics
        stats_generator = StatisticsGenerator()

        # Compilation stage
        compile_tasks = [
            task for task in task_objs
            if self.check_stage_pass(task, [TestStage.CODE_GEN])[0]
        ]
        if TestStage.COMPILE in config.active_stages:
            batch_compile(compile_tasks, num_processes=config.compile.num_processes,
                          timeout=config.chat.timeout, verbose=True)
        else:
            for task in task_objs:
                task.compile_result = CompileResult(task.check_compile_success()[0], log_file=task.compile_log_path)

        # Precision evaluation stage
        if TestStage.PRECISION in config.active_stages:
            precision_tasks = [
                task for task in task_objs
                if self.check_stage_pass(task, [TestStage.CODE_GEN, TestStage.COMPILE])[0]
            ]

            def update_callback(progress: int):
                if progress % 5 == 0:
                    print(f"{'=' * 100}\nprint current completed task {progress}/{len(precision_tasks)}"
                          f" success_statistics info...")
                    stats_generator.generate_success_statistics(task_objs)
                    print(f"{'=' * 100}")

            batch_precision_test(precision_tasks, num_processes=len(config.eval.gpu_devices),
                                 timeout=config.eval.timeout_precision,
                                 update_callback=update_callback)
            for task in precision_tasks:
                print(f"[INFO] Task {task.short_id} precision result:\n{task.precision_result}")

        # Performance evaluation stage
        if TestStage.PERFORMANCE in config.active_stages:
            performance_tasks = [
                task for task in task_objs
                if self.check_stage_pass(task, [TestStage.CODE_GEN, TestStage.COMPILE, TestStage.PRECISION])[0]
            ]
            batch_performance_test(performance_tasks, num_processes=len(config.eval.gpu_devices),
                                   timeout=config.eval.timeout_perf)
            for task in performance_tasks:
                print(f"[INFO] Task {task.short_id} performance result:\n{task.perf_result}")

        stats_generator.generate_success_statistics(task_objs)


class StatisticsGenerator:
    """Statistics generation and reporting."""

    @staticmethod
    def generate_success_statistics(task_objs: List[TaskObject]):
        """Generate and display success statistics."""
        grouped_tasks = group_task_objects_by_problem(task_objs)
        check_stages = [TestStage.CODE_GEN, TestStage.COMPILE, TestStage.PRECISION, TestStage.PERFORMANCE]
        results_by_group = []

        def count_stage_failures(stage_results, condition):
            """Count tasks that meet the specified condition."""
            return sum(1 for result in stage_results if condition(result))

        for key, group in grouped_tasks.items():
            stage_results = [TestRunner.check_stage_pass(task, check_stages) for task in group]
            group_stats = {}

            # Extract group information
            match = re.search(r'level(\d+)/(.*)', key[0])
            if match:
                group_stats['op_definition'] = match.group(2)
                group_stats['op_level'] = int(match.group(1))
            else:
                group_stats['op_definition'] = key[0]
                group_stats['op_level'] = 'unknown'

            group_stats['case_id'] = key[1]
            group_stats['total_samples'] = len(group)
            # group_stats['all_success'] = count_stage_failures(stage_results, lambda x: x[0])
            group_stats['format_parse_pass'] = (
                    group_stats['total_samples'] -
                    count_stage_failures(stage_results, lambda x: x[1] == TestStage.CODE_GEN)
            )
            group_stats['compile_pass'] = (
                    group_stats['format_parse_pass'] -
                    count_stage_failures(stage_results, lambda x: x[1] == TestStage.COMPILE)
            )
            group_stats['precision_pass'] = (
                    group_stats['compile_pass'] -
                    count_stage_failures(stage_results, lambda x: x[1] == TestStage.PRECISION)
            )
            results_by_group.append(group_stats)

        StatisticsGenerator._display_statistics(results_by_group)

    @staticmethod
    def _display_statistics(results_by_group):
        """Display formatted statistics."""

        from tabulate import tabulate
        with pd.option_context('display.max_rows', None, 'display.max_columns', None,
                               'display.width', None, 'display.max_colwidth', None):
            print("\n\nDetailed Statistics by Operator Level:\n")

            df_results = pd.DataFrame(results_by_group).sort_values(by=['op_level', 'op_definition'])

            # Display by level
            for level in df_results['op_level'].unique():
                print(f"[Statistics for Level {level}]\n")
                level_data = df_results[df_results['op_level'] == level]
                print(tabulate(level_data.reset_index(drop=True),
                               headers='keys',
                               tablefmt='grid',
                               showindex=False))
                print("\n")

            # Overall statistics
            print("Overall Statistics Summary:\n")
            summary_stats = df_results.groupby('op_level').sum(numeric_only=True)

            weight = {1: 0.2, 2: 0.3, 3: 0.5}
            model_score = 0.
            for op_level, row in summary_stats.iterrows():
                model_score += weight[op_level] * row['precision_pass'] / row['total_samples']

            for new_key, old_key in [('parse_pass_rate', 'format_parse_pass'),
                                     ('compile_pass_rate', 'compile_pass'),
                                     ('precision_pass_rate', 'precision_pass')]:
                summary_stats[new_key] = (
                        (summary_stats[old_key] / summary_stats['total_samples'] * 100).round(2).astype(str) + '%'
                )

            final_summary = summary_stats[
                ['total_samples', 'parse_pass_rate', 'compile_pass_rate', 'precision_pass_rate']]

            print(tabulate(final_summary,
                           headers='keys',
                           tablefmt='grid',
                           showindex=True))
            print("\n")

            console = Console()
            console.print(f"[bold green][The NPUKernelBench benchmark achieved a total score of: {model_score.round(2)}][/bold green]", style="bold cyan",
                          justify="left")


def create_task_objects(task_paths):
    """Create task objects based on configuration."""
    task_objects = []
    for op_def_path in task_paths:
        if config.static_shape_mode:
            task_objects.extend([
                TaskObject(op_def_path, i, j)
                for i in range(config.n_sample)
                for j in range(config.n_case)
            ])
        else:
            task_objects.extend([
                TaskObject(op_def_path, i)
                for i in range(config.n_sample)
            ])
    return task_objects


def main():
    """Main execution function."""
    args = parse_arguments()

    # Discover tasks
    discovery = TaskDiscovery()
    all_task_paths = discovery.get_all_task_paths()
    print(f"Discovered {len(all_task_paths)} task paths")

    # Filter tasks if specified
    if args.task_name:
        task_names = args.task_name
        selected_paths = discovery.filter_tasks_by_names(all_task_paths, task_names)
        print(f"Selected {len(selected_paths)} tasks for testing:")
        for path in selected_paths:
            print(f"  - {path}")
    else:
        selected_paths = all_task_paths
        print(f"Testing all {len(selected_paths)} tasks")

    if not selected_paths:
        print("No tasks found for testing. Exiting.")
        exit(1)

    # Configure testing mode
    print(config)
    print(f"[DYNAMIC SHAPE MODE] enabled: {not config.static_shape_mode}")

    # Create task objects
    test_objects = create_task_objects(selected_paths)

    # Run testing pipeline
    runner = TestRunner()
    runner.run_testing_pipeline(test_objects, args)


if __name__ == "__main__":
    main()
