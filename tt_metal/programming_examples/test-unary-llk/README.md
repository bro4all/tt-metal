# Test Unary LLK Example

This example demonstrates a simple unary operation using Low-Level Kernels (LLK) in TT-Metal. It implements an identity operation where input data is copied to output without modification.

## Structure

The example follows the standard TT-Metal programming pattern with three kernel types:

### Files

- `main.cpp` - Host code that sets up the program, buffers, and kernels
- `kernels/compute/compute.cpp` - Compute kernel that applies the identity operation
- `kernels/dataflow/reader.cpp` - Reader kernel that reads data from DRAM to circular buffers
- `kernels/dataflow/writer.cpp` - Writer kernel that writes data from circular buffers to DRAM

### Kernels

1. **Reader Kernel** (`reader.cpp`):
   - Reads tiles from DRAM input buffer
   - Writes tiles to the `input_cb` circular buffer
   - Compile-time args: `Wt` (number of tiles), `input_cb` (circular buffer index)
   - Runtime args: input buffer address

2. **Compute Kernel** (`compute.cpp`):
   - Reads tiles from `input_cb` circular buffer
   - Applies identity operation (copy with no modification)
   - Writes tiles to `output_cb` circular buffer
   - Compile-time args: `Wt` (number of tiles), `input_cb`, `output_cb`

3. **Writer Kernel** (`writer.cpp`):
   - Reads tiles from `output_cb` circular buffer
   - Writes tiles to DRAM output buffer
   - Compile-time args: `Wt` (number of tiles), `output_cb`
   - Runtime args: output buffer address

## Usage

This example serves as a skeleton for implementing other unary operations. To modify it for different operations:

1. Replace the identity operation in `compute.cpp` with your desired LLK operation
2. Add appropriate initialization calls (e.g., `exp_tile_init()` for exponential)
3. Add the actual operation call (e.g., `exp_tile(0)` for exponential)

## Build and Run

The example can be built using the standard TT-Metal build system and run on TT hardware or simulator.

## Example Output

The program processes 4 tiles of random data and verifies that the output matches the input (since it's an identity operation), then prints:

```
Test Unary LLK Example Results:
Number of tiles processed: 4
Elements per tile: 1024

First 10 elements:
Input[0]: 0.1234 -> Output[0]: 0.1234
Input[1]: -0.5678 -> Output[1]: -0.5678
...

Test PASSED: Identity operation successful!
```
