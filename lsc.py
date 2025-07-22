import subprocess
from collections import defaultdict

N_RUNS = 100
failures = []
exit = False

for i in range(N_RUNS):
    print(f"== Test Run {i+1} ==")
    # result = subprocess.run(
    #     ["pytest", "-k", "dims2-layout0-shape5-bfloat16", "tests/tt_eager/python_api_testing/unit_testing/misc/test_transpose.py::test_transpose_3D"],
    #     capture_output=True, text=True
    # )
    # print(result.stdout)
    # result = subprocess.run(
    #     ["pytest", "-k", "dims2-layout0-shape5-float", "tests/tt_eager/python_api_testing/unit_testing/misc/test_transpose.py::test_transpose_3D"],
    #     capture_output=True, text=True
    # )
    # print(result.stdout)
    # result = subprocess.run(
    #     ["pytest", "-k", "dims2-layout0-shape5-int32", "tests/tt_eager/python_api_testing/unit_testing/misc/test_transpose.py::test_transpose_3D"],
    #     capture_output=True, text=True
    # )
    # print(result.stdout)
    result = subprocess.run(
        [
            "pytest",
            "-k",
            "dims2-layout0-shape5-",
            "tests/tt_eager/python_api_testing/unit_testing/misc/test_transpose.py::test_transpose_3D",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    for line in result.stdout.splitlines():
        if line.startswith("FAILED"):
            failures.append(str(i) + " " + line)
            exit = True
            break
    if exit:
        break

print("\n=== Failure Summary ===")
for line in failures:
    print(line)
