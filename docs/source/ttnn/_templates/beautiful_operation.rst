{{ fullname | escape | underline}}

{% if objname == "sqrt" -%}
**Function Signature:**

.. code-block::

   ttnn.sqrt(input_tensor, *, memory_config=None, output_tensor=None, queue_id=0) → Tensor
{% elif objname == "rsqrt" -%}
**Function Signature:**

.. code-block::

   ttnn.rsqrt(input_tensor, *, memory_config=None, output_tensor=None, queue_id=0, fast_and_approximate_mode=False) → Tensor
{% else -%}
.. currentmodule:: {{ module }}

.. _{{ fullname }}:

.. auto{{ objtype }}:: {{ fullname }}
{% endif %}

{% if objname == "sqrt" -%}
Computes the element-wise square root of the input tensor.

**Mathematical Definition:**

.. math::
   \text{output}_i = \sqrt{\text{input}_i}

**Parameters:**

- **input_tensor** (*Tensor*) – the input tensor. Must contain non-negative values for real results.

**Keyword Arguments:**

- **memory_config** (*MemoryConfig*, *optional*) – memory configuration for the output tensor.
- **output_tensor** (*Tensor*, *optional*) – preallocated output tensor.
- **queue_id** (*int*, *optional*) – command queue ID.

**Returns:**

*Tensor* – A tensor with the element-wise square root of the input.

**Supported Data Types:**

- ``ttnn.bfloat16``
- ``ttnn.float32``

**Supported Layouts:**

- ``ttnn.TILE_LAYOUT``
- ``ttnn.ROW_MAJOR_LAYOUT``

**Example:**

.. code-block:: python

   >>> import ttnn
   >>> import torch
   >>> input_tensor = ttnn.from_torch(torch.tensor([4.0, 9.0, 16.0], dtype=torch.bfloat16))
   >>> ttnn.sqrt(input_tensor)
   ttnn.Tensor([2.0, 3.0, 4.0], dtype=ttnn.bfloat16)

**Note:**

This operation is equivalent to ``input_tensor ** 0.5`` but optimized for Tenstorrent hardware.

{% elif objname == "rsqrt" -%}
Computes the element-wise reciprocal square root (inverse square root) of the input tensor.

**Mathematical Definition:**

.. math::
   \text{output}_i = \frac{1}{\sqrt{\text{input}_i}}

**Parameters:**

- **input_tensor** (*Tensor*) – the input tensor. Must contain positive values for real results.

**Keyword Arguments:**

- **memory_config** (*MemoryConfig*, *optional*) – memory configuration for the output tensor.
- **output_tensor** (*Tensor*, *optional*) – preallocated output tensor.
- **queue_id** (*int*, *optional*) – command queue ID.
- **fast_and_approximate_mode** (*bool*, *optional*) – enable fast approximation mode for better performance.

**Returns:**

*Tensor* – A tensor with the element-wise reciprocal square root of the input.

**Supported Data Types:**

- ``ttnn.bfloat16``
- ``ttnn.float32``

**Supported Layouts:**

- ``ttnn.TILE_LAYOUT``
- ``ttnn.ROW_MAJOR_LAYOUT``

**Example:**

.. code-block:: python

   >>> import ttnn
   >>> import torch
   >>> input_tensor = ttnn.from_torch(torch.tensor([4.0, 9.0, 16.0], dtype=torch.bfloat16))
   >>> ttnn.rsqrt(input_tensor)
   ttnn.Tensor([0.5, 0.3333, 0.25], dtype=ttnn.bfloat16)

**Note:**

This operation is equivalent to ``1.0 / ttnn.sqrt(input_tensor)`` but optimized for Tenstorrent hardware. When ``fast_and_approximate_mode=True``, uses hardware-specific approximations for improved performance.

{% else -%}
.. currentmodule:: {{ module }}

.. _{{ fullname }}:

.. auto{{ objtype }}:: {{ fullname }}
{% endif %}
