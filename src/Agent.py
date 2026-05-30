from dataclasses import dataclass

import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.models import Model

from config import (
    ACTION_COUNT,
    CENTER_SENSOR_INDEX,
    MAX_SPEED,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
    SENSOR_ANGLES,
    SENSOR_COUNT,
    SENSOR_LENGTH,
)

WALL_COLOR = (0, 0, 0)
FINISH_COLOR = (255, 0, 0)
CAR_COLORS = {
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
}


@dataclass
class Experience:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: float


class Agent:
    def __init__(self, track, model: Model | None, exploration: float, color: str = "blue"):
        self.track = track
        self.model = model
        self.epsilon = exploration
        self.color = color

        self.x_initial = 0.0
        self.y_initial = 65.0
        self.x = self.x_initial
        self.y = self.y_initial
        self.size = 20
        self.angle = 0.0
        self.speed = 1.0

        self.sensors = np.zeros(SENSOR_COUNT, dtype=np.float32)
        self.finish = False
        self.wall=False
        self.active = True
        self.total_distance = 0.0
        self.total_time = 0
        self.displacements = []
        self.data: list[Experience] = []

        self.font = pygame.font.Font(None, 24)

    @staticmethod
    def _rgb(color):
        return color[:3]

    def draw(self, screen):
        half_size = self.size / 2
        corners = [
            (self.x - half_size, self.y - half_size),
            (self.x + half_size, self.y - half_size),
            (self.x + half_size, self.y + half_size),
            (self.x - half_size, self.y + half_size),
        ]
        rotated_corners = [(int(px), int(py)) for px, py in (self.rotate_point(x, y) for x, y in corners)]

        pygame.draw.polygon(screen, CAR_COLORS.get(self.color, CAR_COLORS["blue"]), rotated_corners)

        if self.color == "green":
            self.draw_sensors(screen)
            self.display_sensor_values(screen)
            self.display_reward(screen)

    def rotate_point(self, px, py):
        radians = np.radians(self.angle)
        sin_a = np.sin(radians)
        cos_a = np.cos(radians)
        x_shifted = px - self.x
        y_shifted = py - self.y
        return (
            x_shifted * cos_a - y_shifted * sin_a + self.x,
            x_shifted * sin_a + y_shifted * cos_a + self.y,
        )

    def draw_sensors(self, screen):
        for angle_offset, sensor_value in zip(SENSOR_ANGLES, self.sensors):
            angle = self.angle + angle_offset
            distance = sensor_value * SENSOR_LENGTH
            end_x = self.x + distance * np.cos(np.radians(angle))
            end_y = self.y + distance * np.sin(np.radians(angle))
            pygame.draw.line(screen, (0, 0, 255), (int(self.x), int(self.y)), (int(end_x), int(end_y)), 2)

    def update(self):
        if not self.active or self.finish:
            return

        state = self.get_state()
        action = self.choose_action(state)

        self.apply_action(action)
        self.move()
        self.update_status()

        next_state = self.get_state()
        reward = self.calculate_step_reward()
        done = 1.0 if self.finish or not self.active else 0.0
        self.data.append(Experience(state, action, reward, next_state, done))

    def update_manual(self):
        if not self.active or self.finish:
            return

        state = self.get_state()
        action = self.apply_manual_controls()

        self.move()
        self.update_status()

        next_state = self.get_state()
        reward = self.calculate_step_reward()
        done = 1.0 if self.finish or not self.active else 0.0
        self.data.append(Experience(state, action, reward, next_state, done))

    def get_state(self):
        self.sensors = self.read_sensors()
        speed_value = np.array([self.speed / MAX_SPEED], dtype=np.float32)
        return np.concatenate([self.sensors, speed_value]).astype(np.float32)

    def read_sensors(self):
        distances = [
            self.distance_to_obstacle(self.angle + angle_offset) / SENSOR_LENGTH
            for angle_offset in SENSOR_ANGLES
        ]
        return np.array(distances, dtype=np.float32)

    def distance_to_obstacle(self, angle):
        radians = np.radians(angle)
        cos_a = np.cos(radians)
        sin_a = np.sin(radians)

        for distance in range(SENSOR_LENGTH + 1):
            sensor_x = self.x + distance * cos_a
            sensor_y = self.y + distance * sin_a

            if not self.inside_screen(sensor_x, sensor_y):
                return distance

            color = self._rgb(self.track.get_at((int(sensor_x), int(sensor_y))))
            if color == WALL_COLOR:
                return distance

        return SENSOR_LENGTH

    def choose_action(self, state):
        if self.model is None or np.random.rand() < self.epsilon:
            return int(np.random.choice(ACTION_COUNT))

        q_values = self.model(tf.expand_dims(state, axis=0), training=False)
        return int(tf.argmax(q_values, axis=1).numpy()[0])

    def apply_action(self, action):
        actions = {
            0: self.turn_left,
            1: self.turn_right,
            2: self.accelerate,
            3: self.brake,
        }
        actions[action]()

    def apply_manual_controls(self):
        keys = pygame.key.get_pressed()
        action = 0

        if keys[pygame.K_a]:
            self.turn_left()
            action = 0
        if keys[pygame.K_d]:
            self.turn_right()
            action = 1
        if keys[pygame.K_w]:
            self.accelerate()
            action = 2
        if keys[pygame.K_s]:
            self.brake()
            action = 3

        return action

    def move(self):
        self.x += self.speed * np.cos(np.radians(self.angle))
        self.y += self.speed * np.sin(np.radians(self.angle))
        if not self.inside_screen(self.x, self.y):
            self.wall=True
            self.active = False

        self.x = float(np.clip(self.x, 0, SCREEN_WIDTH))
        self.y = float(np.clip(self.y, 0, SCREEN_HEIGHT))

        self.total_distance += self.speed
        self.total_time += 1
        self.displacements.append((self.x, self.y))

    def update_status(self):
        if not self.active:
            return

        if not self.inside_screen(self.x, self.y):
            self.active = False
            return

        color = self._rgb(self.track.get_at((int(self.x), int(self.y))))
        if color == WALL_COLOR:
            self.wall=True
            self.active = False
        elif color == FINISH_COLOR:
            self.finish = True

    @staticmethod
    def inside_screen(x, y):
        return 0 <= x < SCREEN_WIDTH and 0 <= y < SCREEN_HEIGHT

    def calculate_step_reward(self):
        if self.finish:
            return 20.0
        if self.wall:
            return -20.0
        if not self.active:
            return 0.0
        

        """center_clearance = self.sensors[CENTER_SENSOR_INDEX]
        front_slice = self.sensors[CENTER_SENSOR_INDEX - 1 : CENTER_SENSOR_INDEX + 2]
        front_clearance = float(np.mean(front_slice))
        overall_clearance = float(np.mean(self.sensors))

        raw_reward = (
            1.2 * self.speed
            - 3.0 * (1.0 - center_clearance)
            - 1.5 * (1.0 - front_clearance)
            - 1.0 * (1.0 - overall_clearance)
        )
        normalized_reward = 2.0 * (raw_reward + 5.5) / 15.5 - 1.0"""

        reward = 0.3 * self.speed
        if self.speed < 1.5:
            reward -= 0.2

        return float(reward)

    def display_sensor_values(self, screen):
        for i, value in enumerate(self.sensors):
            sensor_text = self.font.render(f"Sensor {i + 1}: {value:.2f}", True, (0, 0, 0))
            screen.blit(sensor_text, (20, 50 + i * 20))

    def display_reward(self, screen):
        reward_text = self.font.render(f"Reward: {self.calculate_step_reward():.2f}", True, (0, 0, 0))
        speed_text = self.font.render(f"Speed: {self.speed:.2f}", True, (0, 0, 0))
        screen.blit(reward_text, (600, 550))
        screen.blit(speed_text, (600, 570))

    def turn_left(self):
        self.angle -= 2.5

    def turn_right(self):
        self.angle += 2.5

    def accelerate(self):
        self.speed = min(MAX_SPEED, self.speed + 0.15)

    def brake(self):
        self.speed = max(0.0, self.speed - 0.25)
