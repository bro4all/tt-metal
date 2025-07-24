## This commit contains test for ttnn.from_torch and ttnn.to_torch with mesh_mappers.


1. `test_single_device_to_and_from_torch` method in `tests/ttnn/unit_tests/test_to_from_torch_with_mapper.py` file will contain unit test for `single device`.
    -  To run the test for Single-Device, use the following command : `pytest tests/ttnn/unit_tests/test_to_from_torch_with_mapper.py::test_single_device_to_and_from_torch`

2. `test_multi_device_to_and_from_torch` method in `tests/ttnn/unit_tests/test_to_from_torch_with_mapper.py` file will contain unit test for `multi device`.
    - To run the test for Multi-Device, use the following command : `pytest tests/ttnn/unit_tests/test_to_from_torch_with_mapper.py::test_multi_device_to_and_from_torch`

## Details:

#### On WH(n300):

1. [Single Device] Input tensor with shape `[2, 3, 640, 640]`:
   - Time taken for from_torch op - `0.000803` secs.
   - Time taken for to_torch op - `0.000787` secs.

2. [Multi Device] Input tensor with shape `[1, 3, 640, 640]`:
   - Time taken for from_torch op - `0.002397` secs.
   - Time taken for to_torch op - `0.003391` secs.

## Observations:
- `from_torch` on multi-device is 2.98x slower than single device.
- `to_torch` on multi-device is 4.96x slower than single device.

This additional overhead in multi-device directly impacts the end-to-end inference FPS.
