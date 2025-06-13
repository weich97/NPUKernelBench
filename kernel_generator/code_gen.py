import os
from framework.task_object import TaskObject


class AscendCodeGen:
    """Base class for Ascend code generation."""

    method = ""

    def __init__(self, task_obj: TaskObject, logger=None):
        """
        Initialize the code generator.

        Args:
            task_obj: Task object containing task information
            logger: Logger instance for logging
        """
        self.task_obj = task_obj
        self.task_desc_dict = {}
        self.gen_contents_dict = {}
        self.logger = logger

        if self.logger:
            self.logger.info(f"Starting code generation for task {task_obj.short_id}")

    def process(self):
        """
        Process the code generation task.

        This method should be implemented by subclasses to provide
        specific code generation functionality.

        Returns:
            Generated code content
        """
        raise NotImplementedError("Subclasses must implement the process method")