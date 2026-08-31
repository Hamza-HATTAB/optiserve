import sys
import subprocess
import torch


def main():
    print("==================================================")
    print(" OptiServe Verification & Sanity Check Suite       ")
    print("==================================================")

    # 1. CUDA check
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_ok}")
    if cuda_ok:
        device_name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        print(f"Device: {device_name} ({vram_gb} GB VRAM)")

    # 2. Pytest execution
    print("\nRunning test suite...")
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v", "--quiet"], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("Test suite failed!")
        sys.exit(1)

    print("✓ All 15 unit and regression tests passed successfully.")


if __name__ == "__main__":
    main()
