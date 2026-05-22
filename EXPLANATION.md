# Reinforcement Racing Explanation

## 1. How the agent interacts with the environment

Each simulation frame follows this loop:

1. The agent reads its current state.
   - It casts 9 front-facing sensors from left to right at angles `-80, -60, -40, -20, 0, 20, 40, 60, 80` degrees relative to the car direction.
   - Each sensor returns a normalized distance from `0.0` to `1.0`, where `1.0` means no wall or finish line was found within the sensor range.
   - The current speed is normalized by the maximum speed and appended to the sensors.
   - The final state therefore has 10 values: 9 sensors plus speed.

2. The agent chooses an action.
   - With probability `epsilon`, it explores by choosing a random action.
   - Otherwise, it sends the state into the Q model.
   - The model returns 4 Q values, one per action.
   - The agent chooses the action with the highest Q value.

3. The action changes the car controls.
   - Action `0`: turn left.
   - Action `1`: turn right.
   - Action `2`: accelerate.
   - Action `3`: brake.

4. The car moves.
   - Its position changes according to its speed and angle.
   - The position is clipped to the screen bounds.
   - Distance and time counters are updated.

5. The environment checks the result.
   - If the car reaches black pixels, it has hit a wall and becomes inactive.
   - If the car reaches red pixels, it has finished the track.
   - Otherwise, it remains active.

6. The agent reads the next state.
   - The 9 sensors are read again after movement.
   - The new normalized speed is appended again.

7. The reward is calculated.
   - Finishing gives a large positive reward.
   - Hitting a wall gives a negative reward.
   - While driving, reward increases with speed and with free space in front of the car.
   - Driving very slowly receives a small penalty.

8. The transition is stored.
   - The replay item is:
     `state, action, reward, next_state, done`
   - `done` is `1.0` when the car finished or crashed, otherwise `0.0`.

## 2. How the training process works

Training uses Deep Q Learning with a target model:

1. `main.py` loads the track image and creates or loads the Q model.
   - The model input size must be 10.
   - If an older saved model has the old 6-input shape, it is ignored and a fresh model is created.

2. A target model is cloned from the Q model.
   - The Q model is trained every replay step.
   - The target model is updated less often, which makes Q targets more stable.

3. For each training round, an episode loop runs many simulation batches.
   - Each batch creates several agents with the current Q model.
   - Each agent drives until it crashes, finishes, or reaches the step limit.
   - All generated transitions are added to a replay memory buffer.

4. Exploration decays during the episode.
   - Early in training, the agent takes more random actions.
   - Later in training, it relies more on the Q model.
   - The minimum exploration value used by the schedule is `0.05`.

5. Once the replay memory has enough data, a random sample is selected.
   - Random replay avoids training only on the most recent driving sequence.
   - The code currently waits for at least 256 transitions.
   - It trains from up to 1200 sampled transitions.

6. The model builds Q-learning targets.
   - It predicts current Q values for each sampled state.
   - It predicts next-state actions with the Q model.
   - It evaluates those next-state actions with the target model.
   - This is the Double DQN update pattern.

7. Each sampled action target is replaced.
   - If the transition ended the run, the target is just the immediate reward.
   - Otherwise:
     `target = reward + discount_factor * target_model_value_for_best_next_action`
   - The discount factor is `0.95`.

8. The Q model trains for one epoch on the sampled states and updated targets.

9. Every 10 replay updates, the target model copies the Q model weights.

10. Every 25 replay updates, the Q model is saved.

11. After all rounds finish, the trained Q model is saved to `models/upmodel_compact.keras`.
