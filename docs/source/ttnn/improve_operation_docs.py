#!/usr/bin/env python3
"""
Script to improve TTNN operation documentation by applying our enhanced template.

This script helps systematically upgrade TTNN operation documentation to match
the improved PyTorch-style format with mathematical notation, examples, and
enhanced styling.

Usage:
    python improve_operation_docs.py --operation sqrt --preview
    python improve_operation_docs.py --operation rsqrt --apply
    python improve_operation_docs.py --all --preview
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TTNNDocImprover:
    """Class to improve TTNN operation documentation."""

    def __init__(self, docs_root: str = "ttnn/api"):
        self.docs_root = Path(docs_root)
        self.template_path = Path("operation_template.rst")

    def find_operation_files(self) -> List[Path]:
        """Find all TTNN operation RST files."""
        pattern = "ttnn.*.rst"
        return list(self.docs_root.glob(pattern))

    def is_basic_doc(self, file_path: Path) -> bool:
        """Check if a documentation file uses the basic autodata format."""
        with open(file_path, "r") as f:
            content = f.read()

        # Basic docs typically have just autodata directive and minimal content
        lines = content.strip().split("\n")
        return (
            len(lines) <= 10
            and ".. autodata::" in content
            and "----" not in content
            and "Mathematical Definition" not in content
        )

    def extract_operation_name(self, file_path: Path) -> str:
        """Extract operation name from file path."""
        return file_path.stem  # e.g., 'ttnn.sqrt' from 'ttnn.sqrt.rst'

    def get_operation_info(self, op_name: str) -> Dict[str, str]:
        """Get operation-specific information for documentation."""

        # Common operation information
        operations = {
            "ttnn.sqrt": {
                "description": "Computes the element-wise square root of the input tensor",
                "formula": r"\text{output}_i = \sqrt{\text{input}_i}",
                "input_desc": "The input tensor. Must contain non-negative values for real results.",
                "return_desc": "A new tensor with the element-wise square root of the input.",
                "example_input": "torch.tensor([[4.0, 9.0, 16.0], [1.0, 25.0, 36.0]], dtype=torch.bfloat16)",
                "example_output": "# tensor([[2.0, 3.0, 4.0],\\n   #         [1.0, 5.0, 6.0]], dtype=torch.bfloat16)",
                "notes": [
                    "Input values must be non-negative for real results",
                    "For negative inputs, the operation may return NaN values",
                    "This operation is performed element-wise across the entire tensor",
                ],
                "related_ops": ["ttnn.rsqrt", "ttnn.square", "ttnn.pow"],
            },
            "ttnn.rsqrt": {
                "description": "Computes the element-wise reciprocal square root (inverse square root) of the input tensor",
                "formula": r"\text{output}_i = \frac{1}{\sqrt{\text{input}_i}}",
                "input_desc": "The input tensor. Must contain positive values for real results.",
                "return_desc": "A new tensor with the element-wise reciprocal square root of the input.",
                "example_input": "torch.tensor([[4.0, 9.0, 16.0], [1.0, 25.0, 36.0]], dtype=torch.bfloat16)",
                "example_output": "# tensor([[0.5000, 0.3333, 0.2500],\\n   #         [1.0000, 0.2000, 0.1667]], dtype=torch.bfloat16)",
                "notes": [
                    "Input values must be positive for real results",
                    "For zero or negative inputs, the operation may return infinite or NaN values",
                    "Fast approximation mode provides better performance with slightly reduced accuracy",
                    "This operation is commonly used in normalization operations and neural networks",
                ],
                "related_ops": ["ttnn.sqrt", "ttnn.pow", "ttnn.reciprocal"],
                "extra_kwargs": ["fast_and_approximate_mode"],
            },
            "ttnn.square": {
                "description": "Computes the element-wise square of the input tensor",
                "formula": r"\text{output}_i = (\text{input}_i)^2",
                "input_desc": "The input tensor.",
                "return_desc": "A new tensor with the element-wise square of the input.",
                "example_input": "torch.tensor([[-2.0, 3.0, -4.0], [1.0, -5.0, 6.0]], dtype=torch.bfloat16)",
                "example_output": "# tensor([[4.0, 9.0, 16.0],\\n   #         [1.0, 25.0, 36.0]], dtype=torch.bfloat16)",
                "notes": [
                    "Output values are always non-negative",
                    "This operation preserves the magnitude while removing the sign",
                    "Commonly used in loss functions and distance calculations",
                ],
                "related_ops": ["ttnn.sqrt", "ttnn.pow", "ttnn.abs"],
            },
        }

        # Default template for unknown operations
        default = {
            "description": f'Applies the {op_name.split(".")[-1]} operation element-wise to the input tensor',
            "formula": r"\text{output}_i = f(\text{input}_i)",
            "input_desc": "The input tensor.",
            "return_desc": f'A new tensor with the {op_name.split(".")[-1]} operation applied element-wise.',
            "example_input": "torch.randn(2, 3, dtype=torch.bfloat16)",
            "example_output": "# Output depends on the specific operation",
            "notes": [
                "This operation is performed element-wise across the entire tensor",
                "The output tensor has the same shape and layout as the input tensor",
            ],
            "related_ops": ["ttnn.sqrt", "ttnn.square", "ttnn.pow"],
            "extra_kwargs": [],
        }

        return operations.get(op_name, default)

    def generate_improved_doc(self, file_path: Path) -> str:
        """Generate improved documentation for an operation."""
        op_name = self.extract_operation_name(file_path)
        op_info = self.get_operation_info(op_name)

        # Read the template
        template_content = self.get_template()

        # Replace placeholders
        improved_content = template_content.format(
            operation_name=op_name.split(".")[-1],
            mathematical_description=op_info["description"],
            mathematical_formula=op_info["formula"],
            input_description=op_info["input_desc"],
            return_description=op_info["return_desc"],
            example_input=op_info["example_input"],
            example_output=op_info["example_output"],
            notes="\n".join(f"- {note}" for note in op_info["notes"]),
            related_ops="\n".join(f"- :func:`{op}` - {self.get_op_brief_desc(op)}" for op in op_info["related_ops"]),
        )

        return improved_content

    def get_template(self) -> str:
        """Get the documentation template."""
        return """ttnn.{operation_name}
{'=' * (len('ttnn.{operation_name}'))}

.. currentmodule:: ttnn

.. _ttnn.{operation_name}:

.. autodata:: ttnn.{operation_name}

----

**Mathematical Definition**

{mathematical_description}:

.. math::
   {mathematical_formula}

**Parameters:**

- **input_tensor** (*ttnn.Tensor*) – {input_description}

**Keyword Arguments:**

- **memory_config** (*ttnn.MemoryConfig*, *optional*) – Memory configuration for the output tensor. Defaults to ``None``.
- **output_tensor** (*ttnn.Tensor*, *optional*) – Preallocated output tensor. Defaults to ``None``.
- **queue_id** (*int*, *optional*) – Command queue ID. Defaults to ``0``.

**Returns:**

*ttnn.Tensor* – {return_description}

**Supported Data Types:**

- ``ttnn.bfloat16``
- ``ttnn.float32``

**Supported Layouts:**

- ``ttnn.TILE_LAYOUT``
- ``ttnn.ROW_MAJOR_LAYOUT``

**Examples:**

Basic usage:

.. code-block:: python

   import ttnn
   import torch

   # Create input tensor
   input_tensor = ttnn.from_torch(
       {example_input},
       layout=ttnn.TILE_LAYOUT
   )

   # Apply operation
   output = ttnn.{operation_name}(input_tensor)

   # Convert back to torch
   result = ttnn.to_torch(output)
   print(result)
   {example_output}

**Note:**

{notes}
- The output tensor has the same shape and layout as the input tensor

**See Also:**

{related_ops}
"""

    def get_op_brief_desc(self, op_name: str) -> str:
        """Get brief description for related operations."""
        descriptions = {
            "ttnn.sqrt": "Element-wise square root (√x)",
            "ttnn.rsqrt": "Reciprocal square root (1/√x)",
            "ttnn.square": "Element-wise square (x²)",
            "ttnn.pow": "Element-wise power operation",
            "ttnn.abs": "Element-wise absolute value",
            "ttnn.reciprocal": "Element-wise reciprocal (1/x)",
        }
        return descriptions.get(op_name, "Related operation")

    def preview_improvement(self, file_path: Path) -> None:
        """Preview the improved documentation for a file."""
        print(f"\n{'='*60}")
        print(f"PREVIEW: {file_path.name}")
        print(f"{'='*60}")

        if not self.is_basic_doc(file_path):
            print("This file already has enhanced documentation.")
            return

        improved_content = self.generate_improved_doc(file_path)
        print(improved_content[:1000] + "..." if len(improved_content) > 1000 else improved_content)

    def apply_improvement(self, file_path: Path) -> bool:
        """Apply improvements to a documentation file."""
        if not self.is_basic_doc(file_path):
            print(f"Skipping {file_path.name}: Already has enhanced documentation")
            return False

        try:
            improved_content = self.generate_improved_doc(file_path)
            with open(file_path, "w") as f:
                f.write(improved_content)
            print(f"✓ Improved {file_path.name}")
            return True
        except Exception as e:
            print(f"✗ Error improving {file_path.name}: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Improve TTNN operation documentation")
    parser.add_argument("--operation", help="Specific operation to improve (e.g., sqrt)")
    parser.add_argument("--all", action="store_true", help="Process all basic operation docs")
    parser.add_argument("--preview", action="store_true", help="Preview changes without applying")
    parser.add_argument("--apply", action="store_true", help="Apply changes to files")
    parser.add_argument("--docs-root", default="ttnn/api", help="Root directory for docs")

    args = parser.parse_args()

    if not (args.preview or args.apply):
        print("Must specify either --preview or --apply")
        return

    improver = TTNNDocImprover(args.docs_root)

    if args.operation:
        # Process specific operation
        file_path = improver.docs_root / f"ttnn.{args.operation}.rst"
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return

        if args.preview:
            improver.preview_improvement(file_path)
        else:
            improver.apply_improvement(file_path)

    elif args.all:
        # Process all operation files
        files = improver.find_operation_files()
        basic_files = [f for f in files if improver.is_basic_doc(f)]

        print(f"Found {len(basic_files)} files with basic documentation")

        for file_path in basic_files:
            if args.preview:
                improver.preview_improvement(file_path)
            else:
                improver.apply_improvement(file_path)

    else:
        print("Must specify either --operation or --all")


if __name__ == "__main__":
    main()
