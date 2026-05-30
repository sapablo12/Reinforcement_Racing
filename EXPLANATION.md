# Technical Explanation

This file is a short walkthrough of the core learning loop. The complete project documentation is in `README.md`.

## Agent Step

Each active agent step follows this sequence:

1. Read the current state.
   - Cast 9 front-facing sensors.
   - Normalize each sensor distance to `0.0` through `1.0`.
   - Append normalized speed.
   - The current state size is 10.

2. Choose an action.
   - With probability `epsilon`, choose a random action.
   - Otherwise, run the state through the Q-network and choose the action with the highest Q-value.

3. Apply the action.
   - `0`: turn left.
   - `1`: turn right.
   - `2`: accelerate.
   - `3`: brake.

4. Move the car.
   - Position changes using speed and heading.
   - Leaving the screen marks `wall=True` and `active=False`.

5. Check track pixels.
   - Black pixels mark wall collision.
   - Red pixels mark finish.
   - The red finish line is transparent to sensors but still detected by the car position.

6. Read the next state.

7. Calculate reward.
   - Finish gives a strong positive reward.
   - Wall collision gives a strong negative reward.
   - Normal driving rewards speed and lightly penalizes very low speed.

8. Store an experience item:

```text
state, action, reward, next_state, done
```

`done` is true when the agent finished or stopped being active.

## Training Loop

Training uses DQN-style replay with a target network:

1. Load or create the Q-network.
2. Clone it into a target network.
3. Simulate agents to collect experiences.
4. Add experiences to replay memory.
5. Once enough data exists, sample random experiences.
6. Predict current Q-values for sampled states.
7. Predict next actions with the Q-network.
8. Evaluate those next actions with the target network.
9. Replace only the Q-value for the action that was actually taken.
10. Fit the Q-network on the updated targets.
11. Periodically copy Q-network weights into the target network.
12. Periodically save the model.

For non-terminal transitions:

```text
target = reward + discount_factor * future_value
```

For terminal transitions:

```text
target = reward
```

The current discount factor is `0.99`, so future finish/crash rewards influence earlier decisions more strongly than with a lower discount value.
