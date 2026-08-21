from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Optional, Union
import threading
import time

import copy

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3

try:
    import cma
except ImportError:  # pragma: no cover - optional dependency for CMA-ES
    cma = None
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

EventCallback = Callable[[dict[str, Any]], None]
SUPPORTED_METHODS = ("SAC", "TD3", "PPO", "CMA-ES")
SUPPORTED_REWARD_MODES = ("standard",)
HALF_CHEETAH_ENV_ID = "HalfCheetah-v5"
HALF_CHEETAH_SOLVED_REWARD = 6000.0
QUALITY_EVALUATION_EPISODES = 10


@lru_cache(maxsize=1)
def _verify_mujoco_runtime() -> None:
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import mujoco; print(mujoco.__version__)"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Der Python-Interpreter fuer MuJoCo wurde nicht gefunden.") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("MuJoCo konnte innerhalb von 15 Sekunden nicht gestartet werden.") from error

    if probe.returncode == 0:
        return

    details = (probe.stderr or probe.stdout).strip()
    if probe.returncode in {-1073741795, 3221225501}:
        raise RuntimeError(
            "MuJoCo kann auf dieser CPU oder virtuellen Maschine nicht ausgefuehrt werden.\n"
            "HalfCheetah-v5 benoetigt einen Rechner oder eine VM mit den erforderlichen CPU-Instruktionen.\n"
            "Eine Paketinstallation kann diese Hardware-Anforderung nicht beheben."
        )
    installation_hint = 'Installiere die Abhaengigkeiten mit: pip install -r requirements.txt'
    raise RuntimeError(f"MuJoCo ist fuer HalfCheetah-v5 nicht verfuegbar. {details}\n{installation_hint}")


def make_halfcheetah_env(render_mode: Optional[str] = None) -> gym.Env:
    _verify_mujoco_runtime()
    try:
        return gym.make(HALF_CHEETAH_ENV_ID, render_mode=render_mode)
    except Exception as exc:  # pragma: no cover - depends on OS/display backend
        message = (
            "MuJoCo konnte keinen gültigen Render-Kontext erstellen. "
            "Das passiert typischerweise in headless-/Remote-Umgebungen, "
            "Windows-VMs oder ohne echte GUI-/OpenGL-Session. "
            "Starte die App auf einem lokalen Desktop oder deaktiviere die Live-Animation."
        )
        raise RuntimeError(message) from exc


def assess_training_quality(rewards: list[float], method: str) -> dict[str, Any]:
    finite = [float(r) for r in rewards if np.isfinite(r)]
    if not finite:
        return {
            "is_well_trained": False,
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "message": "Keine gültigen Evaluations-Rewards vorhanden.",
            "recommendations": ["Training erneut starten und Umgebung / Installation prüfen."],
        }

    mean_reward = float(np.mean(finite))
    std_reward = float(np.std(finite))
    if mean_reward >= HALF_CHEETAH_SOLVED_REWARD:
        return {
            "is_well_trained": True,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "message": "Das Modell ist erfolgreich trainiert.",
            "recommendations": [],
        }

    recommendations = ["total_timesteps erhöhen und danach erneut evaluieren."]
    if std_reward >= 1500.0:
        recommendations.extend([
            "learning_rate reduzieren und mehrere Seeds vergleichen.",
            "batch_size oder n_envs erhöhen, damit die Updates stabiler werden.",
        ])
        message = "Das Modell lernt, ist aber noch instabil."
    elif mean_reward < 0.0:
        recommendations.append("learning_rate und Netzwerkarchitektur im Parametervergleich prüfen.")
        if method == "PPO":
            recommendations.append("n_steps erhöhen und gae_lambda zwischen 0.90 und 0.98 vergleichen.")
        elif method == "SAC":
            recommendations.append("ent_coef='auto' verwenden und learning_starts erhöhen.")
        else:
            recommendations.append("learning_starts erhöhen und tau zwischen 0.003 und 0.01 vergleichen.")
        message = "Das Modell hat deutliche Lernprobleme."
    else:
        recommendations.append("learning_rate, Netzwerkarchitektur und batch_size im Parametervergleich testen.")
        if method == "PPO":
            recommendations.append("n_steps und gae_lambda gemeinsam vergleichen.")
        elif method == "SAC":
            recommendations.append("batch_size und ent_coef im Parametervergleich prüfen.")
        else:
            recommendations.append("buffer_size, learning_starts und policy_delay vergleichen.")
        message = "Das Modell verbessert sich, braucht aber weitere Trainingszeit."
    return {
        "is_well_trained": False,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "message": message,
        "recommendations": recommendations,
    }


@dataclass
class OnPolicyConfig:
    method: str = "PPO"
    env_id: str = HALF_CHEETAH_ENV_ID
    reward_mode: str = "standard"
    total_timesteps: int = 1_000_000
    training_stop_mode: str = "timesteps"
    target_episodes: Optional[int] = 5_000
    seed: Optional[int] = 123
    n_envs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: Union[float, str] = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    clip_range: float = 0.2
    buffer_size: int = 300_000
    learning_starts: int = 10_000
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    policy_delay: int = 2
    net_arch_pi: str = "256,256"
    net_arch_vf: str = "256,256"
    activation_fn: str = "relu"
    ortho_init: bool = True
    device: str = "auto"
    verbose: int = 0

    def validate(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"Unterstützte Methoden: {', '.join(SUPPORTED_METHODS)}")
        if self.n_envs < 1:
            raise ValueError("n_envs muss mindestens 1 sein.")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda muss zwischen 0 und 1 liegen.")
        if self.total_timesteps < 1:
            raise ValueError("total_timesteps muss positiv sein.")
        if self.training_stop_mode not in {"timesteps", "episodes"}:
            raise ValueError("training_stop_mode muss 'timesteps' oder 'episodes' sein.")
        if self.training_stop_mode == "episodes" and (self.target_episodes is None or self.target_episodes < 1):
            raise ValueError("target_episodes muss im Episodenmodus positiv sein.")


def get_default_parameters_for_method(method: str) -> dict[str, Any]:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unbekannte Methode: {method}")
    values = {
        "method": method,
        "env_id": HALF_CHEETAH_ENV_ID,
        "reward_mode": "standard",
        "total_timesteps": 1_000_000,
        "training_stop_mode": "timesteps",
        "target_episodes": 5_000,
        "seed": 123,
        "n_envs": 4,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "clip_range": 0.2,
        "buffer_size": 300_000,
        "learning_starts": 10_000,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_delay": 2,
        "net_arch_pi": "256,256",
        "net_arch_vf": "256,256",
        "activation_fn": "relu",
        "ortho_init": True,
        "device": "auto",
        "verbose": 0,
    }
    if method == "SAC":
        values.update(batch_size=256, ent_coef="auto", net_arch_pi="256,256", net_arch_vf="256,256")
    elif method == "TD3":
        values.update(learning_rate=1e-3, buffer_size=1_000_000, learning_starts=100, batch_size=100, gradient_steps=-1, net_arch_pi="400,300", net_arch_vf="400,300")
    elif method == "CMA-ES":
        values.update(
            learning_rate=1e-4,
            n_envs=1,
            batch_size=128,
            gamma=0.995,
            buffer_size=200_000,
            learning_starts=5000,
            net_arch_pi="128,128",
            net_arch_vf="128,128",
        )
    else:
        values.update(net_arch_pi="256,256", net_arch_vf="256,256")
    return values


def flatten_policy_weights(model: torch.nn.Module) -> np.ndarray:
    vector = torch.nn.utils.parameters_to_vector(model.parameters())
    return vector.detach().cpu().numpy().astype(np.float64).copy()


def restore_policy_weights(model: torch.nn.Module, weights: np.ndarray) -> None:
    tensor = torch.as_tensor(weights, dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
    torch.nn.utils.vector_to_parameters(tensor, model.parameters())


class CMAESPolicyNetwork(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...] = (32, 16), output_dim: int = 6) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(torch.nn.Linear(last_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            last_dim = hidden_dim
        layers.append(torch.nn.Linear(last_dim, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def clone(self) -> "CMAESPolicyNetwork":
        return copy.deepcopy(self)


def parse_hidden_layers(value: str) -> list[int]:
    layers = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not layers or any(layer < 1 for layer in layers):
        raise ValueError("Netzwerkarchitektur muss positive Layergrößen enthalten.")
    return layers


def parse_architecture_variants(value: str) -> list[str]:
    variants = [item.strip() for item in value.replace("\n", ";").split(";") if item.strip()]
    for variant in variants:
        parse_hidden_layers(variant)
    return variants


def build_policy_kwargs(config: OnPolicyConfig) -> dict[str, Any]:
    activation_map = {
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "leakyrelu": torch.nn.LeakyReLU,
    }
    activation_name = config.activation_fn.strip().lower()
    if activation_name not in activation_map:
        allowed = ", ".join(sorted(activation_map))
        raise ValueError(f"Unbekannte Aktivierungsfunktion: {config.activation_fn}. Erlaubt: {allowed}")

    actor_layers = parse_hidden_layers(config.net_arch_pi)
    critic_layers = parse_hidden_layers(config.net_arch_vf)
    if config.method == "PPO":
        return {
            "net_arch": {"pi": actor_layers, "vf": critic_layers},
            "activation_fn": activation_map[activation_name],
            "ortho_init": config.ortho_init,
        }
    return {
        "net_arch": {"pi": actor_layers, "qf": critic_layers},
        "activation_fn": activation_map[activation_name],
    }


class SyncVectorEnvAdapter(VecEnv):
    def __init__(self, vector_env: gym.vector.SyncVectorEnv) -> None:
        self.vector_env = vector_env
        super().__init__(vector_env.num_envs, vector_env.single_observation_space, vector_env.single_action_space)
        self._actions: Any = None
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]

    def reset(self) -> np.ndarray:
        observations, infos = self.vector_env.reset(seed=self._seeds)
        self._seeds = [None for _ in range(self.num_envs)]
        self.reset_infos = self._split_infos(infos)
        return observations

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observations, rewards, terminated, truncated, infos = self.vector_env.step(self._actions)
        dones = np.logical_or(terminated, truncated)
        split_infos = self._split_infos(infos)
        for index, done in enumerate(dones):
            if done:
                split_infos[index]["TimeLimit.truncated"] = bool(truncated[index] and not terminated[index])
                split_infos[index]["terminal_observation"] = observations[index]
        return observations, rewards, dones, split_infos

    def close(self) -> None:
        self.vector_env.close()

    def get_attr(self, attr_name: str, indices: Any = None) -> list[Any]:
        target_indices = self._get_indices(indices)
        return [getattr(self.vector_env.envs[index], attr_name) for index in target_indices]

    def set_attr(self, attr_name: str, value: Any, indices: Any = None) -> None:
        for index in self._get_indices(indices):
            setattr(self.vector_env.envs[index], attr_name, value)

    def env_method(self, method_name: str, *method_args: Any, indices: Any = None, **method_kwargs: Any) -> list[Any]:
        return [getattr(self.vector_env.envs[index], method_name)(*method_args, **method_kwargs) for index in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices: Any = None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

    def render(self, mode: Optional[str] = None) -> Any:
        return self.vector_env.render()

    def _split_infos(self, infos: dict[str, Any]) -> list[dict[str, Any]]:
        result = [{} for _ in range(self.num_envs)]
        for key, value in infos.items():
            if key.startswith("_"):
                continue
            for index in range(self.num_envs):
                try:
                    result[index][key] = value[index]
                except (IndexError, TypeError):
                    pass
        return result


class TrainingCallback(BaseCallback):
    def __init__(
        self,
        stop_event: threading.Event,
        event_callback: Optional[EventCallback] = None,
        *,
        total_timesteps: int = 1,
        method: str = "PPO",
        event_interval: int = 64,
        training_stop_mode: str = "timesteps",
        target_episodes: Optional[int] = None,
        pause_event: Optional[threading.Event] = None,
        render_during_training: bool = False,
        frame_capture_steps: int = 64,
    ) -> None:
        super().__init__(verbose=0)
        self.stop_event = stop_event
        self.event_callback = event_callback
        self.total_timesteps = max(1, int(total_timesteps))
        self.method = method
        self.event_interval = max(1, int(event_interval))
        self.training_stop_mode = training_stop_mode
        self.target_episodes = target_episodes
        self.pause_event = pause_event
        self.render_during_training = bool(render_during_training)
        self.frame_capture_steps = max(1, int(frame_capture_steps))
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self._running_rewards: dict[int, float] = {}
        self._running_lengths: dict[int, int] = {}

    def _emit(self, event: dict[str, Any]) -> None:
        if self.event_callback is not None:
            self.event_callback(event)

    def _render_frame(self) -> Any:
        if not self.render_during_training:
            return None
        try:
            frame = self.training_env.render()
        except Exception:
            return None
        if isinstance(frame, (list, tuple)):
            return frame[0] if frame else None
        if isinstance(frame, np.ndarray) and frame.ndim == 4:
            return frame[0]
        return frame

    def _on_step(self) -> bool:
        while self.pause_event is not None and self.pause_event.is_set():
            if self.stop_event.is_set():
                self._emit({"type": "cancelled", "method": self.method})
                return False
            time.sleep(0.05)
        if self.stop_event.is_set():
            self._emit({"type": "cancelled", "method": self.method})
            return False

        rewards = np.asarray(self.locals.get("rewards", []), dtype=float)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        infos = self.locals.get("infos", [])
        for index, reward in enumerate(rewards):
            self._running_rewards[index] = self._running_rewards.get(index, 0.0) + float(reward)
            self._running_lengths[index] = self._running_lengths.get(index, 0) + 1
            if index >= len(dones) or not dones[index]:
                continue
            info = infos[index] if index < len(infos) and isinstance(infos[index], dict) else {}
            episode = info.get("episode", {})
            episode_reward = float(episode.get("r", self._running_rewards[index]))
            episode_length = int(episode.get("l", self._running_lengths[index]))
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self._running_rewards[index] = 0.0
            self._running_lengths[index] = 0
            self._emit({
                "type": "episode",
                "method": self.method,
                "reward": episode_reward,
                "length": episode_length,
                "episodes": len(self.episode_rewards),
            })
            if self.training_stop_mode == "episodes" and self.target_episodes is not None and len(self.episode_rewards) >= self.target_episodes:
                self._emit({
                    "type": "progress",
                    "method": self.method,
                    "timesteps": int(self.num_timesteps),
                    "total_timesteps": self.total_timesteps,
                    "ratio": 1.0,
                })
                return False

        if self.num_timesteps % self.event_interval == 0:
            self._emit({
                "type": "progress",
                "method": self.method,
                "timesteps": int(self.num_timesteps),
                "total_timesteps": self.total_timesteps,
                "ratio": min(1.0, float(self.num_timesteps) / self.total_timesteps),
            })
        if self.num_timesteps % self.frame_capture_steps == 0:
            frame = self._render_frame()
            if frame is not None:
                self._emit({"type": "frame", "method": self.method, "frame": frame})
        return True


class HalfCheetahTrainer:
    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.configs: dict[str, OnPolicyConfig] = {}

    @staticmethod
    def _evaluate_cma_es_candidate(config: OnPolicyConfig, weights: np.ndarray, episodes: int = 2) -> float:
        if cma is None:
            raise RuntimeError("Das Paket 'cma' fehlt. Installiere es mit: pip install cma")

        env = make_halfcheetah_env(render_mode=None)
        input_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        model = CMAESPolicyNetwork(input_dim=input_dim, hidden_layers=(32, 16), output_dim=action_dim)
        restore_policy_weights(model, weights)

        rewards: list[float] = []
        for _ in range(episodes):
            observation, _ = env.reset(seed=config.seed)
            total_reward = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                tensor = torch.as_tensor(observation, dtype=torch.float32)
                with torch.no_grad():
                    action = model(tensor).numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
            rewards.append(total_reward)
        env.close()
        return float(-np.mean(rewards))

    @staticmethod
    def _run_cma_es_training(
        config: OnPolicyConfig,
        stop_event: threading.Event,
        event_callback: Optional[EventCallback] = None,
        *,
        render_during_training: bool = False,
        frame_capture_steps: int = 64,
        pause_event: Optional[threading.Event] = None,
    ) -> dict[str, Any]:
        if cma is None:
            raise RuntimeError("Das Paket 'cma' fehlt. Installiere es mit: pip install cma")

        env = make_halfcheetah_env(render_mode="rgb_array" if render_during_training else None)
        input_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        model = CMAESPolicyNetwork(input_dim=input_dim, hidden_layers=(32, 16), output_dim=action_dim)
        x0 = flatten_policy_weights(model)
        sigma0 = 0.2
        options = {"popsize": 8, "verbose": -1, "maxiter": max(10, config.total_timesteps // 10_000)}
        optimizer = cma.CMAEvolutionStrategy(x0, sigma0, options)

        best_weights = x0.copy()
        best_reward = -np.inf
        rewards: list[float] = []
        iteration = 0
        if render_during_training:
            observation, _ = env.reset(seed=config.seed)
        while not stop_event.is_set() and iteration < options["maxiter"]:
            population = optimizer.ask()
            fitness_values = []
            for candidate in population:
                fitness_values.append(HalfCheetahTrainer._evaluate_cma_es_candidate(config, candidate, episodes=1))
                if event_callback is not None:
                    event_callback({"type": "progress", "method": config.method, "timesteps": iteration, "total_timesteps": options["maxiter"], "ratio": iteration / max(1, options["maxiter"])})
            optimizer.tell(population, fitness_values)
            best_idx = int(np.argmin(fitness_values))
            current_best = population[best_idx]
            current_best_reward = -fitness_values[best_idx]
            if current_best_reward > best_reward:
                best_reward = current_best_reward
                best_weights = current_best.copy()
            rewards.append(current_best_reward)
            if event_callback is not None:
                event_callback({
                    "type": "episode",
                    "method": config.method,
                    "reward": float(current_best_reward),
                    "length": 1,
                    "episodes": len(rewards),
                })
            if render_during_training and iteration % max(1, int(frame_capture_steps)) == 0:
                try:
                    frame = env.render()
                    if isinstance(frame, (list, tuple)):
                        frame = frame[0] if frame else None
                    elif isinstance(frame, np.ndarray) and frame.ndim == 4:
                        frame = frame[0]
                    if frame is not None:
                        event_callback({"type": "frame", "method": config.method, "frame": np.asarray(frame)})
                except Exception:
                    pass
            iteration += 1
        env.close()
        restore_policy_weights(model, best_weights)
        result = {
            "method": config.method,
            "rewards": rewards,
            "lengths": [1 for _ in rewards],
            "cancelled": stop_event.is_set(),
            "quality": assess_training_quality(rewards, config.method),
            "auto_saved_path": None,
        }
        try:
            path = Path(__file__).resolve().parent / "saved_models" / f"{config.method.lower()}_halfcheetah_model.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, weights=best_weights)
            result["auto_saved_path"] = str(path)
        except Exception:
            result["auto_saved_path"] = None
        return result

    @staticmethod
    def _make_env(config: OnPolicyConfig, render_mode: Optional[str] = None) -> gym.Env:
        _verify_mujoco_runtime()
        try:
            return gym.make(config.env_id, render_mode=render_mode)
        except Exception as exc:  # pragma: no cover - depends on OS/display backend
            message = (
                "MuJoCo konnte keinen gültigen Render-Kontext erstellen. "
                "Das passiert typischerweise in headless-/Remote-Umgebungen, "
                "Windows-VMs oder ohne echte GUI-/OpenGL-Session. "
                "Starte die App auf einem lokalen Desktop oder deaktiviere die Live-Animation."
            )
            raise RuntimeError(message) from exc

    @classmethod
    def _make_vec_env(cls, config: OnPolicyConfig, render_during_training: bool = False):
        render_mode = "rgb_array" if render_during_training else None
        factories = [
            (lambda r=render_mode: cls._make_env(config, r))
            for _ in range(config.n_envs)
        ]
        vector_env = gym.vector.SyncVectorEnv(factories)
        environment = SyncVectorEnvAdapter(vector_env)
        if config.seed is not None:
            environment.seed(config.seed)
        return environment

    @staticmethod
    def _create_model(config: OnPolicyConfig, environment: VecEnv):
        common = {
            "policy": "MlpPolicy",
            "env": environment,
            "learning_rate": config.learning_rate,
            "gamma": config.gamma,
            "batch_size": config.batch_size,
            "policy_kwargs": build_policy_kwargs(config),
            "seed": config.seed,
            "device": config.device,
            "verbose": config.verbose,
        }
        if config.method == "PPO":
            return PPO(
                n_steps=config.n_steps,
                n_epochs=config.n_epochs,
                gae_lambda=config.gae_lambda,
                ent_coef=float(config.ent_coef),
                vf_coef=config.vf_coef,
                max_grad_norm=config.max_grad_norm,
                clip_range=config.clip_range,
                **common,
            )
        if config.method == "CMA-ES":
            raise NotImplementedError(
                "CMA-ES ist als nativer Stable-Baselines3-Trainer nicht verfügbar. "
                "Das HalfCheetah-Backend verwendet jetzt einen eigenen CMA-ES-Wrapper über das Paket 'cma'."
            )
        off_policy = {
            **common,
            "buffer_size": config.buffer_size,
            "learning_starts": config.learning_starts,
            "tau": config.tau,
            "train_freq": config.train_freq,
            "gradient_steps": config.gradient_steps,
        }
        if config.method == "SAC":
            return SAC(ent_coef=config.ent_coef, **off_policy)
        return TD3(policy_delay=config.policy_delay, **off_policy)

    def train(
        self,
        config: OnPolicyConfig,
        stop_event: threading.Event,
        event_callback: Optional[EventCallback] = None,
        *,
        render_during_training: bool = False,
        frame_capture_steps: int = 64,
        pause_event: Optional[threading.Event] = None,
        auto_save: bool = True,
    ) -> dict[str, Any]:
        config.validate()
        if config.method == "CMA-ES":
            result = self._run_cma_es_training(
                config,
                stop_event,
                event_callback,
                render_during_training=render_during_training,
                frame_capture_steps=frame_capture_steps,
                pause_event=pause_event,
            )
            self.models[config.method] = result
            self.configs[config.method] = config
            return result

        learn_timesteps = int(config.total_timesteps)
        if config.training_stop_mode == "episodes" and config.target_episodes is not None:
            learn_timesteps = max(learn_timesteps, int(config.target_episodes) * 2000)
        environment = self._make_vec_env(config, render_during_training)
        callback = TrainingCallback(
            stop_event,
            event_callback,
            total_timesteps=learn_timesteps,
            method=config.method,
            training_stop_mode=config.training_stop_mode,
            target_episodes=config.target_episodes,
            pause_event=pause_event,
            render_during_training=render_during_training,
            frame_capture_steps=frame_capture_steps,
        )
        model = self._create_model(config, environment)
        self.models[config.method] = model
        self.configs[config.method] = config
        model.learn(total_timesteps=learn_timesteps, callback=callback, progress_bar=False)
        rewards = callback.episode_rewards
        result = {
            "method": config.method,
            "rewards": rewards,
            "lengths": callback.episode_lengths,
            "cancelled": stop_event.is_set(),
            "quality": assess_training_quality(rewards, config.method),
            "auto_saved_path": None,
        }
        if auto_save:
            try:
                path = Path(__file__).resolve().parent / "saved_models" / f"{config.method.lower()}_halfcheetah_model.zip"
                path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(path))
                result["auto_saved_path"] = str(path)
            except Exception:
                result["auto_saved_path"] = None
        return result

    def evaluate(self, model: Any, episodes: int = 10, seed: Optional[int] = None) -> list[float]:
        rewards: list[float] = []
        env = make_halfcheetah_env()
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode if seed is not None else None)
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
            rewards.append(total_reward)
        env.close()
        return rewards

    def save_model(self, method: str, path: str | Path) -> str:
        model = self.models.get(method)
        if model is None:
            raise ValueError(f"Kein geladenes Modell für {method} vorhanden.")
        full_path = str(path)
        model.save(full_path)
        return full_path

    def load_model(self, method: str, path: str | Path) -> Any:
        env = make_halfcheetah_env()
        if method == "PPO":
            model = PPO.load(str(path), env=env)
        elif method == "SAC":
            model = SAC.load(str(path), env=env)
        else:
            model = TD3.load(str(path), env=env)
        self.models[method] = model
        self.configs[method] = OnPolicyConfig(method=method)
        env.close()
        return model


Baselines3Trainer = HalfCheetahTrainer
GymnasiumVectorEnvAdapter = SyncVectorEnvAdapter


__all__ = [
    "EventCallback",
    "SUPPORTED_METHODS",
    "SUPPORTED_REWARD_MODES",
    "HALF_CHEETAH_ENV_ID",
    "HALF_CHEETAH_SOLVED_REWARD",
    "OnPolicyConfig",
    "get_default_parameters_for_method",
    "make_halfcheetah_env",
    "HalfCheetahTrainer",
    "Baselines3Trainer",
    "SyncVectorEnvAdapter",
    "GymnasiumVectorEnvAdapter",
    "assess_training_quality",
    "flatten_policy_weights",
    "restore_policy_weights",
    "CMAESPolicyNetwork",
    "parse_hidden_layers",
    "parse_architecture_variants",
]
