import os
import re
from framework.kernel_gen_config import config
from framework.utils import safe_read_file


class KernelParser:
    """Parser for extracting kernel implementation code from generated content."""

    def __init__(self, kernel_name):
        self.kernel_name = kernel_name

    def parse(self, gen_content: str = None, fake_path: str = None):
        """
        Parse kernel content from either generated content or fake path.

        Args:
            gen_content: Generated content string to parse
            fake_path: Path to existing kernel files

        Returns:
            Tuple of (tiling_content, kernel_content)
        """
        if gen_content is None and fake_path is not None:
            return self._parse_from_files(fake_path)
        else:
            return self._parse_from_content(gen_content)

    def _parse_from_content(self, gen_content: str = None):
        """Parse kernel content from generated text using XML-like tags."""
        if gen_content is None:
            return None, None

        contents = []

        # Define tags to extract based on mode
        if config.static_shape_mode:
            tags = [('kernel_impl', 'kernel_impl')]
        else:
            tags = [('tiling_cpp', 'tiling_cpp'), ('kernel_impl', 'kernel_impl')]

        for start_tag, end_tag in tags:
            # Match content within XML-like tags
            pattern = rf'<{start_tag}>(.*?)</({end_tag})>'
            match = re.search(pattern, gen_content, re.DOTALL)

            if match:
                tag_content = match.group(1).strip()

                # Extract code from ```cpp blocks if present
                cpp_match = re.search(r'```cpp\s*(.*?)\s*```', tag_content, re.DOTALL)

                if cpp_match:
                    content = cpp_match.group(1).strip()
                else:
                    content = tag_content.strip()

                contents.append(content)
            else:
                contents.append(None)

        # Validate that all required content was found
        for (start_tag, end_tag), content in zip(tags, contents):
            if content is None or len(content) == 0:
                raise RuntimeError(f"Empty or missing content for tag <{start_tag}></{end_tag}>")

        # Adjust return format for dynamic shape mode
        if config.static_shape_mode:
            contents = [None] + contents

        return tuple(contents)

    def _parse_from_files(self, fake_path: str = None):
        """Parse kernel content from existing files."""
        file_names = [f'op_host/{self.kernel_name}.cpp', f'op_kernel/{self.kernel_name}.cpp']
        contents = []

        for file_name in file_names:
            file_path = f'{fake_path}/{file_name}'
            content = safe_read_file(file_path)
            contents.append(content)

        return tuple(contents)