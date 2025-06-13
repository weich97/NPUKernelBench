import importlib
import json
import os
import pandas as pd
import importlib.util

import torch
from typing import Union
from framework.kernel_gen_config import config


def string_with_emphasize(input_str: str, tag='=', num=100):
    """Create emphasized string with decorative tags."""
    return f"\n{tag * num}\n{input_str}\n{tag * num}\n"


def save_json_file(data: dict, file_path: str) -> None:
    """Save dictionary data to JSON file with proper encoding."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding=config.encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_write_file(file_path: str, content: Union[str, bytes], create_dir: bool = True, mode: str = None) -> None:
    """
    Safely write content to file with automatic encoding handling.

    Args:
        file_path: Target file path
        content: Content to write (string or bytes)
        create_dir: Whether to create parent directories
        mode: File open mode (auto-detected if None)
    """
    encoding = config.encoding

    if create_dir:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    # Determine file mode based on content type
    if mode is None:
        mode = 'wb' if isinstance(content, bytes) else 'w'

    # Handle content encoding
    if mode == 'w' and isinstance(content, bytes):
        content = content.decode(encoding, errors='replace')
    elif mode == 'wb' and isinstance(content, str):
        content = content.encode(encoding, errors='replace')

    # Write file
    with open(file_path, mode=mode, encoding=encoding if 'b' not in mode else None) as f:
        f.write(content)


def safe_read_file(file_path: str, mode: str = None) -> Union[str, bytes, None]:
    """Safely read file content with proper encoding."""
    if mode is None:
        mode = 'r'
    try:
        with open(file_path, mode=mode, encoding=config.encoding if 'b' not in mode else None) as f:
            return f.read()
    except FileNotFoundError:
        return None


def preprocess_dataframe(df):
    """
    Preprocess DataFrame by converting numeric columns to integers when appropriate.
    Preserves all original data including null values.
    """
    df_copy = df.copy()

    for col in df_copy.columns:
        if pd.api.types.is_numeric_dtype(df_copy[col]) and not df_copy[col].isna().any():
            # Check if all values can be represented as integers
            if df_copy[col].apply(lambda x: float(x) == int(float(x))).all():
                df_copy[col] = df_copy[col].astype(int)

    return df_copy


def load_module_from_path(module_name, module_path):
    """Load Python module from specified file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_case_data(csv_file, case_id=None):
    """Find specific case data from CSV file by case_id."""
    csv_content = pd.read_csv(csv_file, delimiter=';')
    if case_id is None:
        return csv_content

    # Find row with specified case_id
    for index, row in csv_content.iterrows():
        if int(row['case_id']) == case_id:
            return row

    print(f"Error: case_id {case_id} not found in CSV data.")
    return None


def check_precision(outputs, outputs_new, max_abs_error, max_rel_error):
    """Check precision differences between two sets of outputs."""
    # Ensure inputs are in list format
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new

    all_abs_diff, all_rel_diff = [], []
    is_accurate = True

    # Process each output pair
    for out, out_new in zip(outputs, outputs_new):
        abs_diff = torch.abs(out - out_new)
        rel_diff = abs_diff / (torch.abs(out) + 1e-7)
        all_abs_diff.append(abs_diff.view(-1))
        all_rel_diff.append(rel_diff.view(-1))

        # Check if within precision requirements
        if ((abs_diff > max_abs_error) & (rel_diff > max_rel_error)).any():
            is_accurate = False

    # Combine all differences
    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)

    return (1 if is_accurate else 0), all_abs_diff, all_rel_diff

