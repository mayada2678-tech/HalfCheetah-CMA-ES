from __future__ import annotations

import threading

import numpy as np
from gymnasium import spaces

from halfcheetah_logic import (
    HALF_CHEETAH_ENV_ID,
    HALF_CHEETAH_SOLVED_REWARD,
    OnPolicyConfig,
    SyncVectorEnvAdapter,
    assess_training_quality,
    build_policy_kwargs,
    get_default_parameters_for_method,
    parse_architecture_variants,
    parse_hidden_layers,
)


def test_supported_methods_are_sac_td3_ppo_and_cma_es() -> None:
    assert get_default_parameters_for_method("SAC")["method"] == "SAC"
    assert get_default_parameters_for_method("TD3")["method"] == "TD3"
    assert get_default_parameters_for_method("PPO")["method"] == "PPO"
    assert get_default_parameters_for_method("CMA-ES")["method"] == "CMA-ES"
    assert HALF_CHEETAH_ENV_ID == "HalfCheetah-v5"


def test_default_parameters_include_expected_values() -> None:
    config = OnPolicyConfig()
    assert config.env_id == HALF_CHEETAH_ENV_ID
    assert config.reward_mode == "standard"
    assert config.total_timesteps == 1_000_000
    assert config.seed == 123


def test_parse_hidden_layers_rejects_invalid_values() -> None:
    assert parse_hidden_layers("64,128") == [64, 128]
    try:
        parse_hidden_layers("0,32")
        assert False
    except ValueError:
        pass


def test_parse_architecture_variants_supports_semicolon_lists() -> None:
    variants = parse_architecture_variants("64,64;128,128")
    assert variants == ["64,64", "128,128"]


def test_policy_kwargs_are_valid_for_ppo_and_off_policy_methods() -> None:
    ppo = OnPolicyConfig(method="PPO", net_arch_pi="128,64", net_arch_vf="128,64", activation_fn="relu")
    sac = OnPolicyConfig(method="SAC", net_arch_pi="64,64", net_arch_vf="64,64", activation_fn="tanh")
    td3 = OnPolicyConfig(method="TD3", net_arch_pi="64,64", net_arch_vf="64,64", activation_fn="elu")

    ppo_kwargs = build_policy_kwargs(ppo)
    sac_kwargs = build_policy_kwargs(sac)
    td3_kwargs = build_policy_kwargs(td3)

    assert ppo_kwargs["net_arch"]["pi"] == [128, 64]
    assert sac_kwargs["net_arch"]["pi"] == [64, 64]
    assert td3_kwargs["net_arch"]["pi"] == [64, 64]


def test_assess_quality_messages_match_expected_ranges() -> None:
    strong = assess_training_quality([7000.0, 8000.0, 7500.0], "SAC")
    weak = assess_training_quality([-1.0, -2.0, -3.0], "PPO")

    assert strong["is_well_trained"] is True
    assert strong["mean_reward"] >= HALF_CHEETAH_SOLVED_REWARD
    assert weak["is_well_trained"] is False
    assert weak["message"]


def test_cma_es_policy_roundtrip_and_flattening() -> None:
    from halfcheetah_logic import CMAESPolicyNetwork, flatten_policy_weights, restore_policy_weights

    model = CMAESPolicyNetwork(input_dim=5, hidden_layers=(4, 3), output_dim=2)
    flat = flatten_policy_weights(model)
    restored = model.clone()
    restore_policy_weights(restored, flat)
    assert flat.shape[0] > 0
    assert restored is not model


def test_cma_es_training_emits_episode_events(monkeypatch) -> None:
    import halfcheetah_logic as logic

    class FakeEnv:
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        def reset(self, seed=None):
            return np.zeros(3, dtype=np.float32), {}

        def step(self, action):
            return np.zeros(3, dtype=np.float32), 2.0, True, False, {}

        def close(self):
            return None

    class FakeOptimizer:
        def __init__(self, x0, sigma0, options):
            self.x0 = np.asarray(x0, dtype=np.float64)
            self.iter = 0

        def ask(self):
            self.iter += 1
            return [self.x0.copy()]

        def tell(self, population, fitness_values):
            return None

    monkeypatch.setattr(logic, "cma", type("CMAStub", (), {"CMAEvolutionStrategy": FakeOptimizer}), raising=False)
    monkeypatch.setattr(logic, "make_halfcheetah_env", lambda render_mode=None: FakeEnv())

    events: list[dict[str, object]] = []
    trainer = logic.HalfCheetahTrainer()
    result = trainer._run_cma_es_training(
        logic.OnPolicyConfig(method="CMA-ES", total_timesteps=10_000, seed=123),
        threading.Event(),
        event_callback=events.append,
    )

    assert result["method"] == "CMA-ES"
    assert any(event.get("type") == "episode" for event in events)


def test_sync_vector_env_adapter_split_infos_handles_nested_observations() -> None:
    class DummyEnv:
        def __init__(self):
            self.num_envs = 2
            self.single_observation_space = None
            self.single_action_space = None
            self.envs = [type("E", (), {"name": "a"})(), type("E", (), {"name": "b"})()]

    adapter = object.__new__(SyncVectorEnvAdapter)
    adapter.num_envs = 2
    infos = {"reward": np.array([1.0, 2.0]), "_hidden": np.array([3.0, 4.0])}
    result = SyncVectorEnvAdapter._split_infos(adapter, infos)
    assert result[0]["reward"] == 1.0
    assert result[1]["reward"] == 2.0
    assert all("_hidden" not in entry for entry in result)
