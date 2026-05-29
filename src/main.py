import argparse
import os
import random

import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.models import clone_model, load_model
from tqdm import tqdm

from Agent import Agent
from base_model import create_model
from config import ACTION_COUNT, SCREEN_HEIGHT, SCREEN_WIDTH, STATE_SIZE

DISCOUNT_FACTOR = 0.99
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "upmodel_compact.keras")


def init_pygame(headless: bool):
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()


def load_track():
    folder_path = os.path.join(os.path.dirname(__file__), "current_track")
    file_path = os.path.join(folder_path, "track.png")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Track image not found at {file_path}")
    return pygame.image.load(file_path)


def save_model(model_to_save, path=MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model_to_save.save(path)


def run_simulation(agents, track, render=True, max_steps=1800, fps=60):
    if render:
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Racing Game Simulation")
        timer_font = pygame.font.Font(None, 36)
        clock = pygame.time.Clock()
    else:
        screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        timer_font = None
        clock = None

    running = True
    game_active = True
    steps = 0

    while running:
        if render:
            screen.fill((255, 255, 255))
            screen.blit(track, (0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                    save_model(agents[0].model)
        else:
            pygame.event.pump()

        if game_active:
            for agent in agents:
                agent.update()
                if render:
                    agent.draw(screen)

            steps += 1
            if render:
                elapsed_seconds = steps / float(fps)
                timer_text = timer_font.render(f"Time: {elapsed_seconds:.2f} s", True, (0, 0, 0))
                screen.blit(timer_text, (20, 20))

            if steps >= max_steps:
                for agent in agents:
                    agent.active = False

            if all(agent.finish or not agent.active for agent in agents):
                game_active = False
        else:
            running = False

        if render:
            pygame.display.flip()
            clock.tick(fps)

    experience_batch = []
    for agent in agents:
        experience_batch.extend(agent.data)
    return experience_batch


def get_experience(size, q_model, track, exploration, render=False):
    all_experiences = []
    remaining_agents = size

    while remaining_agents > 0:
        run_size = min(30, remaining_agents)
        current_agents = [
            Agent(
                track,
                model=q_model,
                exploration=exploration,
                color="green" if (render and i == 0) else "blue",
            )
            for i in range(run_size)
        ]
        all_experiences.extend(run_simulation(current_agents, track=track, render=render))
        remaining_agents -= run_size

    return all_experiences


def episode(q_model, target_model, track, exploration=0.85, size=1000, batch_size=30, render=False):
    memory_buffer = []
    min_replay_sample = 256
    replay_sample = 1200
    exp=exploration
    for i in tqdm(range(1, size + 1), desc=f"Exploration={exp:.2f}"):
        exp = exploration - (exploration - 0.05) * (i / size)
        memory_buffer.extend(get_experience(size=batch_size, q_model=q_model, track=track, exploration=exp, render=render))

        current_sample_size = min(len(memory_buffer), replay_sample)
        if current_sample_size < min_replay_sample:
            continue

        experiences = random.sample(memory_buffer, current_sample_size)
        states = np.array([data.state for data in experiences], dtype=np.float32)
        next_states = np.array([data.next_state for data in experiences], dtype=np.float32)

        targets = target_model.predict(states, verbose=0)
        next_q_values = q_model.predict(next_states, verbose=0)
        actions_argmax = np.argmax(next_q_values, axis=1)
        target_next_q = target_model.predict(next_states, verbose=0)

        for idx, data in enumerate(experiences):
            if data.done:
                targets[idx][data.action] = data.reward
            else:
                targets[idx][data.action] = data.reward + DISCOUNT_FACTOR * target_next_q[idx][actions_argmax[idx]]

        q_model.fit(states, targets, epochs=1, verbose=0)

        if i % 10 == 0:
            target_model.set_weights(q_model.get_weights())
            if i % 100 == 0:
                memory_buffer = memory_buffer[int(len(memory_buffer) * 0.1) :]

        if i % 25 == 0:
            save_model(q_model)

    return q_model, target_model


def train(
    q_model,
    track,
    rounds=4,
    episode_size=800,
    batch_size=30,
    exploration=0.6,
    render=False,
):
    target_model = clone_model(q_model)
    target_model.set_weights(q_model.get_weights())

    for _ in range(rounds):
        q_model, target_model = episode(
            q_model,
            target_model,
            track=track,
            exploration=exploration,
            size=episode_size,
            batch_size=batch_size,
            render=render,
        )

    return q_model


def try_model(q_model, track, runs=100):
    for _ in range(runs):
        agent = Agent(track, model=q_model, exploration=0.01, color="green")
        run_simulation([agent], track=track, render=True)


def manual(q_model, track):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Racing Manual Drive")
    agent = Agent(track, model=q_model, exploration=0.0, color="green")

    running = True
    clock = pygame.time.Clock()
    timer_font = pygame.font.Font(None, 36)
    game_active = True
    steps = 0

    while running:
        screen.fill((255, 255, 255))
        screen.blit(track, (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if game_active:
            agent.update_manual()
            agent.draw(screen)

            steps += 1
            elapsed_seconds = steps / 60.0
            timer_text = timer_font.render(f"Time: {elapsed_seconds:.2f} s", True, (0, 0, 0))
            screen.blit(timer_text, (20, 20))
            if steps >= 120 * 60:
                agent.active = False

            if agent.finish or not agent.active:
                game_active = False
        else:
            running = False

        pygame.display.flip()
        clock.tick(60)

    return list(agent.data)


def load_or_create_model(path=MODEL_PATH):
    if not os.path.exists(path):
        loaded_model = create_model()
        save_model(loaded_model, path)
        return loaded_model

    try:
        loaded_model = load_model(path, compile=False)
    except (TypeError, ValueError) as exc:
        print(f"Could not deserialize saved model at {path}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
        print("Trying to load the saved weights into the current model architecture.")
        loaded_model = create_model()
        try:
            loaded_model.load_weights(path)
        except Exception as weight_exc:
            print(f"Could not load weights from {path}: {weight_exc}")
            print("Using a fresh model instead.")
            loaded_model = create_model()

    input_size = loaded_model.input_shape[-1]
    output_size = loaded_model.output_shape[-1]
    if input_size != STATE_SIZE or output_size != ACTION_COUNT:
        print(
            f"Ignoring incompatible saved model at {path}: "
            f"expected input/output {STATE_SIZE}/{ACTION_COUNT}, got {input_size}/{output_size}."
        )
        loaded_model = create_model()

    loaded_model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    return loaded_model


def parse_args():
    parser = argparse.ArgumentParser(description="Reinforcement racing trainer")
    parser.add_argument("--mode", choices=["train", "eval", "manual"], default="train")
    parser.add_argument("--headless", action="store_true", help="Disable the graphical window (for CI/servers)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--episode-size", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--exploration", type=float, default=0.6)
    parser.add_argument("--eval-runs", type=int, default=10)
    parser.add_argument("--model-path", default=MODEL_PATH)
    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    init_pygame(headless=args.headless)

    try:
        track = load_track()
        q_model = load_or_create_model(args.model_path)

        if args.mode == "train":
            trained_model = train(
                q_model,
                track=track,
                rounds=args.rounds,
                episode_size=args.episode_size,
                batch_size=args.batch_size,
                exploration=args.exploration,
                render=not args.headless,
            )
            save_model(trained_model, args.model_path)
            return

        if args.mode == "eval":
            if args.headless:
                raise ValueError("Evaluation mode requires display. Remove --headless.")
            try_model(q_model, track=track, runs=100)
            return

        if args.mode == "manual":
            if args.headless:
                raise ValueError("Manual mode requires display. Remove --headless.")
            manual(q_model, track=track)
            return
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()

