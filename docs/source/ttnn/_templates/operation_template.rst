ttnn.{operation_name}
{'=' * (len('ttnn.{operation_name}') + 6)}

.. currentmodule:: ttnn

.. _ttnn.{operation_name}:

.. autodata:: ttnn.{operation_name}

----

**Mathematical Definition**

{mathematical_description}

.. math::
   {mathematical_formula}

**Parameters**

- **input_tensor** (*ttnn.Tensor*) – {input_description}

**Keyword Arguments**

- **memory_config** (*ttnn.MemoryConfig*, *optional*) – Memory configuration for the output tensor. Defaults to ``None``.
- **output_tensor** (*ttnn.Tensor*, *optional*) – Preallocated output tensor. Defaults to ``None``.
- **queue_id** (*int*, *optional*) – Command queue ID. Defaults to ``0``.

**Returns**

*ttnn.Tensor* – {return_description}

**Supported Data Types**

- ``ttnn.bfloat16``
- ``ttnn.float32``

**Supported Layouts**

- ``ttnn.TILE_LAYOUT``
- ``ttnn.ROW_MAJOR_LAYOUT``

**Example:**

.. code-block:: python

   import ttnn
   import torch

   # {example_description}
   {example_code}

**See Also**

{related_operations}
