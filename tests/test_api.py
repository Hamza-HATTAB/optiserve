import pytest
from httpx import ASGITransport, AsyncClient
from optiserve.api.server import app, batcher


@pytest.fixture(autouse=True)
async def setup_batcher():
    await batcher.start()
    yield
    await batcher.stop()


@pytest.mark.asyncio
async def test_api_root_and_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/")
        assert res.status_code == 200
        data = res.json()
        assert data["service"] == "OptiServe ML Inference Engine"

        res_health = await ac.get("/health")
        assert res_health.status_code == 200
        health_data = res_health.json()
        assert health_data["status"] == "healthy"
        assert "gpu_name" in health_data


@pytest.mark.asyncio
async def test_api_metrics_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/metrics")
        assert res.status_code == 200
        assert b"optiserve_" in res.content


@pytest.mark.asyncio
async def test_api_generate_and_benchmarks():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # test synchronous generate
        gen_res = await ac.post(
            "/api/v1/generate",
            json={
                "prompt": "Calculate 15 * 4",
                "max_tokens": 4,
                "speculative": True,
            },
        )
        assert gen_res.status_code == 200
        gen_data = gen_res.json()
        assert gen_data["total_tokens"] == 4
        assert gen_data["tokens_per_second"] > 0
        assert gen_data["speedup_ratio"] is not None

        # test benchmarks retrieval
        bench_res = await ac.get("/api/v1/benchmarks")
        assert bench_res.status_code == 200
        bench_data = bench_res.json()
        assert len(bench_data["benchmarks"]) >= 5

        # test simulator trajectories retrieval
        traj_res = await ac.get("/api/v1/simulator/trajectories")
        assert traj_res.status_code == 200
        traj_data = traj_res.json()
        assert len(traj_data["trajectories"]) >= 3
