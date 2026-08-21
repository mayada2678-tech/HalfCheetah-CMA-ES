# HalfCheetah-v5 Workbench

## Überblick

Diese Workbench ist eine eigenständige Stable-Baselines3-Anwendung für die MuJoCo-Umgebung `HalfCheetah-v5`. Sie enthält dieselbe grundlegende Struktur wie die Hopper-V5-Version: GUI, Training, Live-Animation, Reward-Plot, Vergleichsmodus und Modell-Management.

Die Standard-Methoden in dieser Workbench sind `SAC`, `TD3` und `PPO`. Die Umgebung ist ein MuJoCo-Task und benötigt eine valide MuJoCo-Installation mit kompatiblem Interpreter.

---

## Installation

Im Ordner `Mayada/HalfCheetah-v5`:

```powershell
python -m pip install -r requirements.txt
```

Abhängigkeiten:

- `gymnasium[mujoco]>=0.29.1`
- `mujoco>=3.0.0`
- `stable-baselines3>=2.3.0`
- `torch>=2.2.0`
- `matplotlib>=3.7.0`
- `numpy>=1.24.0`

---

## Start

```powershell
python halfcheetah_app.py
```

---

## Unterstützte Methoden

- `SAC`
- `TD3`
- `PPO`

Die Auswahl erfolgt im UI über die aktive Methode und im Methodenfenster.

---

## Funktionen

- Training für `HalfCheetah-v5`
- Hyperparameter pro Methode
- Vergleich mehrerer Methoden
- Live-Animation während des Trainings
- Reward-Verlauf und Vollplot
- Pause/Resume und Abbruch
- Modell speichern und laden
- automatische Qualitätsbewertung
- Dark/Light Theme

---

## Projektstruktur

```text
HalfCheetah-v5/
├── halfcheetah_app.py
├── halfcheetah_gui.py
├── halfcheetah_logic.py
├── workbench_base.py
├── workbench.md
├── requirements.txt
├── saved_models/
└── test_halfcheetah_logic.py
```

---

## Wichtige Logik

Die Datei `halfcheetah_logic.py` enthält:

- `SUPPORTED_METHODS = ("SAC", "TD3", "PPO")`
- `HALF_CHEETAH_ENV_ID = "HalfCheetah-v5"`
- `OnPolicyConfig`
- `get_default_parameters_for_method(...)`
- `SyncVectorEnvAdapter`
- `TrainingCallback`
- `HalfCheetahTrainer`
- Qualitätsbewertung mit `assess_training_quality(...)`

Dabei gilt als Erfolgsmaßstab ein mittlerer Reward ab einem definierten Solved-Wert. Das genaue Muster entspricht der Hopper-V5-Implementation.

---

## Tests

```powershell
python -m pytest test_halfcheetah_logic.py -q
```

---

## Hinweis

Die MuJoCo-Umgebung ist anspruchsvoll; die Workbench prüft die Laufzeitumgebung beim Start des Trainings und zeigt einen klaren Fehler an, wenn MuJoCo auf der aktuellen Maschine nicht nutzbar ist.
