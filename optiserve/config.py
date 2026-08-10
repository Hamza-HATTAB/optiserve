from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# project root and asset directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BENCHMARK_DIR = DATA_DIR / "benchmarks"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"


class EngineSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPTISERVE_",
        env_file=".env",
        extra="ignore",
    )

    # hardware limits tailored for local RTX 4060 8GB VRAM
    total_vram_mb: int = 8188
    max_active_vram_mb: int = 6963  # 6.8 GB live ceiling
    max_static_model_vram_mb: int = 5324  # 5.2 GB static model limit
    reserved_kv_cache_vram_mb: int = 1639  # reserve 1.6 GB for dynamic KV growth

    # speculative decoding parameters (Leviathan et al.)
    speculative_k: int = 3
    speculative_temperature: float = 0.7
    speculative_top_p: float = 0.9

    # model identifiers
    student_model_id: str = "Qwen/Qwen2.5-1.5B"
    target_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    teacher_model_id: str = "Qwen/Qwen2.5-72B-Instruct"

    # api and serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    max_batch_size: int = 16
    batch_timeout_ms: float = 20.0

    # airllm streaming settings
    airllm_device: str = "cuda:0"
    layer_preload_buffer_size: int = 2


settings = EngineSettings()
