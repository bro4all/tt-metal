import sys
import pandas as pd
import re


def extract_sharding(attributes: str) -> str:
    if "HEIGHT_SHARDED" in attributes:
        return "HEIGHT_SHARDED"
    elif "WIDTH_SHARDED" in attributes:
        return "WIDTH_SHARDED"
    elif "BLOCK_SHARDED" in attributes:
        return "BLOCK_SHARDED"
    return "UNKNOWN"


def filter_and_extract(file_path: str):
    df = pd.read_csv(file_path)

    # Filter DEVICE ID == 0
    df = df[df["DEVICE ID"] == 0]

    # Remove PermuteDeviceOperation entries
    df = df[df["OP CODE"] != "PermuteDeviceOperation"]

    # Extract OptimizedConvNew entries
    convs = df[df["OP CODE"] == "OptimizedConvNew"].copy()

    # Extract sharding
    convs["SHARDING"] = convs["ATTRIBUTES"].apply(extract_sharding)
    convs = convs[["DEVICE KERNEL DURATION [ns]", "CORE COUNT", "SHARDING"]]
    convs.reset_index(drop=True, inplace=True)

    # Total duration of all entries
    total_duration = df["DEVICE KERNEL DURATION [ns]"].sum()

    return df, convs, total_duration


def print_comparison(conv1, conv2):
    print(f"{'Single-chip':<40} | {'Tensor-parallel-2':<40} | Speedup (%)")
    print("-" * 100)
    for i in range(min(len(conv1), len(conv2))):
        dur1 = conv1.iloc[i]["DEVICE KERNEL DURATION [ns]"]
        dur2 = conv2.iloc[i]["DEVICE KERNEL DURATION [ns]"]
        speedup = ((dur1 - dur2) / dur1) * 100 if dur1 != 0 else 0

        print(
            f"DUR: {dur1:<12} CORE: {conv1.iloc[i]['CORE COUNT']:<2} SHARD: {conv1.iloc[i]['SHARDING']:<15} | "
            f"DUR: {dur2:<12} CORE: {conv2.iloc[i]['CORE COUNT']:<2} SHARD: {conv2.iloc[i]['SHARDING']:<15} | "
            f"{speedup:>6.2f}% faster"
        )


def analyze_allgather(df, num_eth_links=1):
    allgather_df = df[df["OP CODE"].str.contains("AllGather", na=False)].copy()

    if allgather_df.empty:
        print("\n--- AllGather Overhead ---")
        print("No AllGather operations found.")
        return

    print("\n--- AllGather Overhead ---")
    print(f"{'Duration [ns]':>15} | {'Ideal [ns]':>12} | {'# Links':>8} | {'Eth Utilization (%)':>22}")
    print("-" * 70)

    total_duration = 0
    for _, row in allgather_df.iterrows():
        duration_ns = row["DEVICE KERNEL DURATION [ns]"]
        total_duration += duration_ns

        # Compute tensor size
        try:
            W = int(row["INPUT_0_W"])
            Z = int(row["INPUT_0_Z"])
            Y = int(row["INPUT_0_Y"])
            X = int(row["INPUT_0_X"])
        except (KeyError, ValueError):
            print("Skipping row due to missing or malformed tensor shape.")
            continue

        total_bytes = W * Z * Y * X * 2  # bfloat16: 2 bytes
        total_gb = total_bytes / 1e9

        ideal_sec = total_gb / (12.5 * num_eth_links)
        ideal_ns = ideal_sec * 1e9

        utilization = (ideal_ns / duration_ns) * 100 if duration_ns != 0 else 0

        print(f"{duration_ns:>15,.0f} | {ideal_ns:>12,.0f} | {num_eth_links:>8} | {utilization:>20.2f}")

    print(f"\nTotal AllGather Duration: {total_duration:,.0f} ns")


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_convs.py <file1.csv> <file2.csv> [--eth-links N]")
        return

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    num_eth_links = 1

    if "--eth-links" in sys.argv:
        try:
            idx = sys.argv.index("--eth-links")
            num_eth_links = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            print("Invalid --eth-links argument. Using default: 1")

    df1, convs1, total1 = filter_and_extract(file1)
    df2, convs2, total2 = filter_and_extract(file2)

    print("\n--- Per-layer OptimizedConvNew Comparison ---\n")
    print_comparison(convs1, convs2)

    conv_total1 = convs1["DEVICE KERNEL DURATION [ns]"].sum()
    conv_total2 = convs2["DEVICE KERNEL DURATION [ns]"].sum()
    conv_savings = conv_total1 - conv_total2
    conv_savings_pct = (conv_savings / conv_total1) * 100 if conv_total1 != 0 else 0

    print("\n--- OptimizedConvNew Total Duration ---")
    print(f"Single-chip:        {conv_total1}")
    print(f"Tensor-parallel-2:  {conv_total2}")
    print(f"Conv savings:       {conv_savings} ns ({conv_savings_pct:.2f}%) faster")

    analyze_allgather(df2, num_eth_links)

    print("\n--- Total DEVICE KERNEL DURATION [ns] ---")
    print(f"Single-chip:        {total1}")
    print(f"Tensor-parallel-2:  {total2}")
    overall_speedup = ((total1 - total2) / total1) * 100 if total1 != 0 else 0
    print(f"Overall speedup:    {overall_speedup:.2f}%")


if __name__ == "__main__":
    main()
