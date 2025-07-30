import pandas as pd
import os

# 1. Read Excel, sheet 2
excel_file = "./Copy of torchview_ops.xlsx"
sheet_index = 1  # Second sheet (0-based index)
df = pd.read_excel(excel_file, sheet_name=sheet_index)
search_dirs = [
    "./tests/ttnn/unit_tests",
    "./tests/ttnn/unit_tests/operations",
    "./tests/ttnn/unit_tests/operations/ccl",
    "./tests/ttnn/unit_tests/operations/conv",
    "./tests/ttnn/unit_tests/operations/data_movement",
    "./tests/ttnn/unit_tests/operations/eltwise",
    "./tests/ttnn/unit_tests/operations/fused",
    "./tests/ttnn/unit_tests/operations/matmul",
    "./tests/ttnn/unit_tests/operations/pool",
    "./tests/ttnn/unit_tests/operations/rand",
    "./tests/ttnn/unit_tests/operations/reduce",
]
operation_names = df.iloc[:, 0].astype(str).tolist()
for operation in operation_names:
    operation = operation.strip()
    if operation.startswith("_") or not operation:
        continue
    found = False

    # Build operation variants: the original, and the one with "_" removed
    variants = [operation]
    if "_" in operation:
        variants.append(operation.replace("_", ""))

    for search_path in search_dirs:
        if not os.path.exists(search_path):
            continue  # Skip if directory doesn't exist
        for fname in os.listdir(search_path):
            if fname.startswith("test_") and fname.endswith(".py"):
                # Remove "test_" and ".py"
                core = fname[len("test_") : -len(".py")]
                for op_variant in variants:
                    if core.startswith(op_variant):
                        next_char_index = len(op_variant)
                        if next_char_index == len(core) or core[next_char_index] == "_":
                            print(
                                f"found {operation} in file: {os.path.join(search_path, fname)} (matched '{op_variant}')"
                            )
                            found = True
                            break
                if found:
                    break
        if found:
            break
    if not found:
        print(f"not found {operation}")
