from fedadaptops.config.routing_schema import ResourceSimulationConfig
from fedadaptops.resources.simulation import simulate_resource_profiles


def test_resource_profile_simulation_is_reproducible():
    cfg = ResourceSimulationConfig()
    a = simulate_resource_profiles(client_ids=[0, 1, 2], config=cfg, seed=123)
    b = simulate_resource_profiles(client_ids=[0, 1, 2], config=cfg, seed=123)

    assert a.equals(b)
    assert set(a["resource_tier"]).issubset({"low", "medium", "high"})
