import os
import time
import tempfile
import torch
import psutil
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class HardwareProfile:
    gpu_name: str
    vram_total_mb: float
    vram_free_mb: float
    nvme_read_mb_per_sec: float
    pcie_h2d_bandwidth_gb_per_sec: float
    cuda_alloc_latency_us: float


class LatencyProfiler:
    """Profiles local NVMe I/O throughput and PCIe host-to-device streaming speed."""

    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def profile_nvme_throughput(self, block_size_mb: int = 128, num_blocks: int = 4) -> float:
        # write temp chunk to NVMe disk and measure sequential read speed
        total_bytes = block_size_mb * num_blocks * 1024 * 1024
        chunk = os.urandom(block_size_mb * 1024 * 1024)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            for _ in range(num_blocks):
                f.write(chunk)
            f.flush()
            os.fsync(f.fileno())

        try:
            start = time.perf_counter()
            with open(temp_path, "rb") as f:
                while f.read(1024 * 1024 * 32):
                    pass
            elapsed = time.perf_counter() - start
            mb_per_sec = (total_bytes / (1024 * 1024)) / max(elapsed, 1e-6)
            return round(mb_per_sec, 2)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def profile_pcie_bandwidth(self, size_mb: int = 256) -> float:
        if not torch.cuda.is_available():
            return 0.0

        # allocate page-locked (pinned) host memory for clean PCIe throughput
        num_elements = (size_mb * 1024 * 1024) // 4
        host_tensor = torch.empty(num_elements, dtype=torch.float32, pin_memory=True)
        device_tensor = torch.empty(num_elements, dtype=torch.float32, device=self.device)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # warm up transfer channel
        device_tensor.copy_(host_tensor, non_blocking=True)
        torch.cuda.synchronize()

        start_event.record()
        device_tensor.copy_(host_tensor, non_blocking=True)
        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)
        gb_per_sec = (size_mb / 1024.0) / (elapsed_ms / 1000.0)
        return round(gb_per_sec, 2)

    def profile_cuda_alloc_latency(self) -> float:
        if not torch.cuda.is_available():
            return 0.0

        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            t = torch.empty((1024, 1024), device=self.device, dtype=torch.float16)
            torch.cuda.synchronize()
            elapsed_us = (time.perf_counter() - start) * 1e6
            latencies.append(elapsed_us)
            del t

        torch.cuda.empty_cache()
        # drop warmups and return median
        return round(sorted(latencies)[len(latencies) // 2], 2)

    def full_profile(self) -> HardwareProfile:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device)
            gpu_name = props.name
            total_vram = round(props.total_memory / (1024 * 1024), 1)
            free_vram = round((props.total_memory - torch.cuda.memory_allocated(self.device)) / (1024 * 1024), 1)
        else:
            gpu_name = "CPU"
            total_vram = 0.0
            free_vram = 0.0

        nvme_speed = self.profile_nvme_throughput()
        pcie_bw = self.profile_pcie_bandwidth()
        alloc_lat = self.profile_cuda_alloc_latency()

        return HardwareProfile(
            gpu_name=gpu_name,
            vram_total_mb=total_vram,
            vram_free_mb=free_vram,
            nvme_read_mb_per_sec=nvme_speed,
            pcie_h2d_bandwidth_gb_per_sec=pcie_bw,
            cuda_alloc_latency_us=alloc_lat,
        )


if __name__ == "__main__":
    profiler = LatencyProfiler()
    profile = profiler.full_profile()
    print("=== OptiServe Hardware Baseline ===")
    print(f"GPU: {profile.gpu_name}")
    print(f"Total VRAM: {profile.vram_total_mb} MB | Free: {profile.vram_free_mb} MB")
    print(f"NVMe Read Speed: {profile.nvme_read_mb_per_sec} MB/s")
    print(f"PCIe H2D Bandwidth: {profile.pcie_h2d_bandwidth_gb_per_sec} GB/s")
    print(f"CUDA Allocation Latency: {profile.cuda_alloc_latency_us} us")
