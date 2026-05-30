# Reinforcement Racing

Reinforcement Racing is a small Deep Q-Learning racing environment built with Pygame and TensorFlow. An agent drives on an image-based track, reads distance sensors, chooses one of four actions, receives rewards, and trains a Q-network from replayed experience.

The project is designed for experimentation with reward functions, sensor layouts, model architecture, and track design.

## Repository Layout

```text
src/
  Agent.py                  Agent physics, sensors, actions, reward, and experience storage
  base_model.py             TensorFlow/Keras Q-network architecture
  config.py                 Shared constants such as screen size, sensors, and state size
  main.py                   Training, evaluation, manual driving, model loading, and CLI
  Track.py                  Optional Tkinter/Pillow track painter
  current_track/track.png   Current image-based track
  models/                   Saved Keras models

requirements.txt            Core dependencies for training and evaluation
requirements-tools.txt      Optional dependencies for helper scripts/tools
EXPLANATION.md              Short technical walkthrough of agent and training loops
```

## Setup

Use a virtual environment from the repository root.

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Install optional helper-tool dependencies only if you want to use `Track.py` or `cuda_pruebas.py`:

```bash
pip install -r requirements-tools.txt
```

`tkinter` is used by `Track.py`, but it is part of the Python standard library on many installations. Some Linux distributions package it separately.

## Running The Project

Run commands from the repository root.

Train with rendering:

```bash
python src/main.py --mode train --rounds 4 --episode-size 800 --batch-size 30 --exploration 0.6
```

Train headlessly:

```bash
python src/main.py --mode train --headless --rounds 4 --episode-size 800 --batch-size 30 --exploration 0.6
```

Evaluate the saved model:

```bash
python src/main.py --mode eval
```

Manual driving mode:

```bash
python src/main.py --mode manual
```

Use a custom model path:

```bash
python src/main.py --mode train --headless --model-path src/models/my_model.keras
```

## Track Format

The environment loads the active track from:

```text
src/current_track/track.png
```

The track is an image. Pixel colors define behavior:

```text
black (0, 0, 0)      wall / crash
red   (255, 0, 0)    finish line
other colors         driveable space
```

The current screen and track size is `800x600`, defined in `src/config.py`.

Sensors treat the red finish line as transparent. This means sensors do not think the finish line is a wall. The agent still finishes when its own position reaches a red pixel.

## Track Painter

`src/Track.py` is an optional Tkinter-based track editor. It lets you draw black walls and red finish lines, erase with white, save named tracks into `src/tracks`, and choose one saved track as `src/current_track/track.png`.

Run it with:

```bash
python src/Track.py
```

Install optional dependencies first:

```bash
pip install -r requirements-tools.txt
```

Current agent logic only gives special meaning to black and red pixels. Other colors are treated as driveable unless you extend the environment logic.

## Agent State

The agent state is defined in `src/Agent.py` and `src/config.py`.

Each state contains:

```text
9 normalized sensor distances + 1 normalized speed = 10 values
```

The sensor angles are:

```python
(-80, -60, -40, -20, 0, 20, 40, 60, 80)
```

These angles are relative to the car direction. Each sensor casts a ray up to `SENSOR_LENGTH` pixels and returns a value from `0.0` to `1.0`:

```text
0.0    wall immediately in front of that sensor
1.0    no wall found within sensor range
```

The final speed value is:

```python
self.speed / MAX_SPEED
```

If you change the number of sensors, sensor history, or any other state input, update `STATE_SIZE` in `src/config.py`. Existing saved models may become incompatible.

## Agent Actions

The Q-network outputs four Q-values. The agent chooses the action with the highest Q-value unless it explores randomly.

```text
0    turn left
1    turn right
2    accelerate
3    brake
```

Turning changes the angle by `2.5` degrees. Acceleration and braking change speed within the configured speed bounds.

## Agent Flags

The agent uses three main status flags:

```text
finish    the car reached a red finish pixel
wall      the car hit a black wall or left the screen
active    the car is still being simulated
```

These flags are intentionally separate. `active=False` means the agent stopped being simulated, but `wall=True` specifically means a crash/failure. This avoids confusing a successful finish or forced stop with a wall collision.

## Reward Function

The reward is calculated in `Agent.calculate_step_reward()`.

Current behavior:

```text
finish:       +20.0
wall crash:   -20.0
inactive:       0.0
normal step:   0.3 * speed, with a small penalty if speed < 1.5
```

This reward avoids directly rewarding wide sensor readings. That is deliberate: rewarding open space can make the agent avoid narrow but correct parts of a track. Sensors are still part of the state, but the reward now mainly encourages fast movement while strongly penalizing crashes and strongly rewarding finish.

## Model Architecture

The Q-network is created in `src/base_model.py`.

Current architecture:

```text
Input(STATE_SIZE)
Dense(32, tanh)
Dense(32, tanh)
Dense(4, linear)
```

It is compiled with:

```text
optimizer: adam
loss: mean_squared_error
```

The output layer is linear because Q-learning predicts action values, not probabilities.

## Training Process

Training is managed in `src/main.py`.

High-level flow:

1. Load the track image.
2. Load an existing model or create a fresh model.
3. Clone the Q-model into a target model.
4. Simulate batches of agents.
5. Store transitions in replay memory:

```text
state, action, reward, next_state, done
```

6. Sample random transitions from replay memory.
7. Build Q-learning targets.
8. Fit the Q-model for one epoch on the sampled states and targets.
9. Periodically copy Q-model weights into the target model.
10. Periodically save the model.

The code uses a Double DQN-style target:

```text
best next action: chosen by q_model
future value:     evaluated by target_model
```

For non-terminal transitions:

```text
target = reward + DISCOUNT_FACTOR * target_model(next_state)[best_next_action]
```

For terminal transitions:

```text
target = reward
```

The discount factor is currently:

```python
DISCOUNT_FACTOR = 0.99
```

## Exploration

The agent uses epsilon-greedy exploration.

During training:

```text
with probability epsilon: choose a random action
otherwise:                choose the action with highest predicted Q-value
```

`episode()` decays exploration from the provided value toward `0.05` across the episode.

## Model Files

The default model path is:

```text
src/models/upmodel_compact.keras
```

`main.py` can also load or save a model through `--model-path`.

The loader is defensive:

1. If no file exists, it creates a fresh model.
2. If the full `.keras` model loads, it uses it.
3. If Keras cannot deserialize the architecture, it creates the current architecture and tries to load only the saved weights.
4. If the shape is incompatible, it creates a fresh model.

This is useful when TensorFlow/Keras versions differ between machines.

## Changing The State Or Model

If you change the state representation, update these together:

```text
src/config.py       STATE_SIZE and related constants
src/Agent.py        state construction
src/base_model.py   input layer shape through STATE_SIZE
saved models        retrain or discard incompatible files
```

Common state changes that break old models:

```text
changing sensor count
adding previous sensor frames
adding velocity history
changing action count
changing model layer shapes
```

## Troubleshooting

### Keras `quantization_config` Error

If loading a model fails with:

```text
Unrecognized keyword arguments passed to Dense: {'quantization_config': None}
```

the model was probably saved with a newer Keras version than the one loading it. `main.py` now attempts to recover by loading only the weights into the local model architecture.

To reduce these issues, keep TensorFlow pinned consistently in `requirements.txt`.

### TensorFlow CUDA Warnings

Warnings about CUDA, cuDNN, cuFFT, or cuBLAS usually mean TensorFlow looked for GPU support and did not find a usable GPU. If the process continues, these warnings are normally harmless.

### Pygame Display Issues

Use headless mode on servers or environments without a display:

```bash
python src/main.py --mode train --headless
```

Headless mode sets Pygame to use a dummy video driver.

### Training Is Slow

The sensor simulation is Python-heavy. Every agent casts 9 rays, and each ray can scan up to `SENSOR_LENGTH` pixels. Training speed depends strongly on:

```text
batch size
episode size
sensor length
number of agents
CPU speed
TensorFlow overhead
```

Reducing `batch_size`, `episode_size`, or `SENSOR_LENGTH` can make experiments faster.

## Developer Notes

Keep constants in `src/config.py` when possible. This avoids hard-coding screen size, action count, sensor count, or state size across multiple files.

Prefer changing one learning idea at a time. For example, do not change reward, state, model size, and discount factor all in the same experiment unless you are deliberately testing a combined design.
