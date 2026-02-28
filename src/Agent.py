import pygame
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, clone_model

from base_model import assign_weights

SENSOR_LENGTH = 250
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600


class Info:
    def __init__(self, state, output, act, step_reward, next_state, done_flag):
        self.state = state
        self.output = output
        self.action = int(act)
        self.step_reward = step_reward
        self.done_flag = done_flag
        self.next_state = next_state


class Agent:
    def __init__(self, track, model: Model, weights, exploration, color="blue"):
        self.track = track
        self.x_initial = 0
        self.y_initial = 65
        self.x = self.x_initial
        self.y = self.y_initial
        self.displacements = []
        self.frame = 0
        self.frame_skip = 1
        self.skip_counter = 0
        self.last_action = None
        self.last_output = None
        self.size = 20
        self.angle = 0
        self.speed = 1
        self.sensors = tf.Variable(tf.zeros([1, 5]), dtype=tf.float32)
        self.finish = False
        self.active = True
        self.model: Model = clone_model(model)
        self.epsilon = exploration
        assign_weights(model=self.model, flat_weights=weights)
        self.total_distance = 0
        self.total_time = 0
        self.reward = 0
        self.font = pygame.font.Font(None, 24)
        self.data = []
        self.color = color

    @staticmethod
    def _rgb(color):
        return (color[0], color[1], color[2])

    def draw(self, screen):
        half_size = self.size / 2
        points = [
            (self.x - half_size, self.y - half_size),
            (self.x + half_size, self.y - half_size),
            (self.x + half_size, self.y + half_size),
            (self.x - half_size, self.y + half_size),
        ]
        rotated_points = [self.rotate_point(px, py, self.angle) for px, py in points]
        rotated_points = [(int(px), int(py)) for px, py in rotated_points]
        pygame.draw.polygon(screen, (0, 0, 255) if self.color == "blue" else (0, 255, 0), rotated_points)

        self.draw_sensors(screen)
        if self.color == "green":
            self.display_sensor_values(screen)
            self.display_reward(screen)

    def rotate_point(self, px, py, angle):
        rad_angle = np.radians(angle)
        sin_a = np.sin(rad_angle)
        cos_a = np.cos(rad_angle)
        cx, cy = self.x, self.y
        x_shifted, y_shifted = px - cx, py - cy
        x_new = x_shifted * cos_a - y_shifted * sin_a + cx
        y_new = x_shifted * sin_a + y_shifted * cos_a + cy
        return (x_new, y_new)

    def check_wall(self):
        if 0 <= self.x < SCREEN_WIDTH and 0 <= self.y < SCREEN_HEIGHT:
            color = self._rgb(self.track.get_at((int(self.x), int(self.y))))
            if color == (0, 0, 0):
                self.active = False

    def draw_sensors(self, screen):
        sensor_angles = np.array([0, -45, 45, -90, 90])
        sensor_colors = (0, 0, 255)

        new_sensor_values = []
        for angle_offset in sensor_angles:
            angle = self.angle + angle_offset
            distance = self.calculate_sensor_distance(angle, SENSOR_LENGTH)
            new_sensor_values.append(distance / SENSOR_LENGTH)

            if self.color == "green":
                end_x = self.x + distance * np.cos(np.radians(angle))
                end_y = self.y + distance * np.sin(np.radians(angle))
                pygame.draw.line(screen, sensor_colors, (int(self.x), int(self.y)), (int(end_x), int(end_y)), 2)

        self.sensors.assign(tf.convert_to_tensor([new_sensor_values], dtype=tf.float32))

    def update_sensors(self):
        sensor_angles = np.array([0, -45, 45, -90, 90])
        new_sensor_values = []

        for angle_offset in sensor_angles:
            angle = self.angle + angle_offset
            distance = self.calculate_sensor_distance(angle, SENSOR_LENGTH)
            new_sensor_values.append(distance / SENSOR_LENGTH)

        return tf.convert_to_tensor([new_sensor_values], dtype=tf.float32)

    def calculate_sensor_distance(self, angle, max_length):
        for distance in range(max_length):
            sensor_x = self.x + distance * np.cos(np.radians(angle))
            sensor_y = self.y + distance * np.sin(np.radians(angle))

            if 0 <= sensor_x < SCREEN_WIDTH and 0 <= sensor_y < SCREEN_HEIGHT:
                rgb_color = self._rgb(self.track.get_at((int(sensor_x), int(sensor_y))))
                if rgb_color == (0, 0, 0):
                    if distance == 0:
                        self.active = False
                    return distance
                if rgb_color == (255, 0, 0):
                    if distance == 0:
                        self.finish = True
                    return distance
            else:
                return distance
        return max_length

    def update(self):
        if self.active and self.model is not None:
            self.sensors.assign(self.update_sensors())
            self.frame += 1
            state = tf.squeeze(tf.concat([self.sensors, tf.constant([[self.speed / 10.0]])], axis=1), axis=0)

            if self.skip_counter == 0:
                output = self.model(tf.expand_dims(state, axis=0))
                act = tf.argmax(output, axis=1).numpy()[0]
                if np.random.rand() < self.epsilon:
                    act = np.random.choice(4)
                self.last_action = act
                self.last_output = output
                self.skip_counter = self.frame_skip
            else:
                act = self.last_action
                self.skip_counter -= 1

            actions = {
                0: self.dec_angle,
                1: self.inc_angle,
                2: self.inc_speed,
                3: self.dec_speed,
            }
            actions.get(act, lambda: None)()

            self.x += self.speed * np.cos(np.radians(self.angle))
            self.y += self.speed * np.sin(np.radians(self.angle))
            self.displacements.append((self.x, self.y))
            self.total_distance += self.speed

            if self.x <= 0 or self.x >= SCREEN_WIDTH or self.y <= 0 or self.y >= SCREEN_HEIGHT:
                self.active = False
            else:
                color = self._rgb(self.track.get_at((int(self.x), int(self.y))))
                if color == (0, 0, 0):
                    self.active = False
                elif color == (255, 0, 0):
                    self.finish = True

            if self.skip_counter == 0:
                step_reward = self.calculate_step_reward()
                next_sensors = self.update_sensors().numpy().tolist()
                next_state = tf.squeeze(
                    tf.concat([tf.convert_to_tensor(next_sensors, dtype=tf.float32), tf.constant([[self.speed / 10.0]])], axis=1),
                    axis=0,
                )
                done_flag = 1.0 if (self.finish or not self.active) else 0.0
                self.data.append(Info(state, self.last_output, act, step_reward, next_state, done_flag))

            self.x = np.clip(self.x, 0, SCREEN_WIDTH)
            self.y = np.clip(self.y, 0, SCREEN_HEIGHT)
            return

        new_displacement = (self.x, self.y)
        self.displacements.append(new_displacement)
        step_reward = self.calculate_step_reward()
        done_flag = 1.0 if (self.finish or not self.active) else 0.0
        output = tf.zeros([1, 4])
        state = tf.squeeze(tf.concat([tf.zeros([1, 5], dtype=tf.float32), tf.constant([[self.speed / 10.0]])], axis=1), axis=0)
        self.data.append(Info(state, output, 0, step_reward, state, done_flag))

    def update2(self):
        if self.active:
            self.sensors.assign(self.update_sensors())
            state = tf.squeeze(tf.concat([self.sensors, tf.constant([[self.speed / 10.0]])], axis=1), axis=0)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_a]:
                self.dec_angle()
            if keys[pygame.K_d]:
                self.inc_angle()
            if keys[pygame.K_w]:
                self.inc_speed()
            if keys[pygame.K_s]:
                self.dec_speed()

            self.x += self.speed * np.cos(np.radians(self.angle))
            self.y += self.speed * np.sin(np.radians(self.angle))
            self.total_distance += self.speed
            self.total_time += 1
            self.displacements.append((self.x, self.y))

            if self.x <= 0 or self.x >= SCREEN_WIDTH or self.y <= 0 or self.y >= SCREEN_HEIGHT:
                self.active = False
            else:
                color = self._rgb(self.track.get_at((int(self.x), int(self.y))))
                if color == (0, 0, 0):
                    self.active = False
                elif color == (255, 0, 0):
                    self.finish = True

            step_reward = self.calculate_step_reward()
            next_sensors = self.update_sensors().numpy().tolist()
            next_state = tf.squeeze(
                tf.concat([tf.convert_to_tensor(next_sensors, dtype=tf.float32), tf.constant([[self.speed / 10.0]])], axis=1),
                axis=0,
            )
            done_flag = 1.0 if self.finish else 0.0
            self.data.append(Info(state, 0, 0, step_reward, next_state, done_flag))
            self.x = np.clip(self.x, 0, SCREEN_WIDTH)
            self.y = np.clip(self.y, 0, SCREEN_HEIGHT)
            return

        self.displacements.append((self.x, self.y))
        step_reward = self.calculate_step_reward()
        done_flag = 1.0 if self.finish else 0.0
        if not self.finish:
            step_reward = -100
            done_flag = 0.0
        output = tf.zeros([1, 4])
        state = tf.squeeze(tf.concat([tf.zeros([1, 5], dtype=tf.float32), tf.constant([[self.speed / 10.0]])], axis=1), axis=0)
        self.data.append(Info(state, output, 0, step_reward, state, done_flag))

    def calculate_step_reward(self):
        v = self.speed
        fs = -3 * (1 - self.sensors.numpy()[0][0])
        avgf = -1.5 * (1 - tf.reduce_mean(self.sensors[0, 1:3]).numpy())
        avgs = -1 * tf.reduce_mean(self.sensors[0, 2:]).numpy()
        max_reward = 10
        min_reward = -5.5

        reward = v + fs + avgs + avgf
        reward = 2 * (reward - min_reward) / (max_reward - min_reward) - 1

        if self.active:
            if not self.finish:
                if self.speed < 0.8:
                    reward = reward - 0.3
                return reward
            return 10

        if self.finish:
            return 10
        return -5

    def calculate_mean_speed(self):
        if self.total_time > 0:
            return self.total_distance / self.total_time
        return 0

    def display_sensor_values(self, screen):
        sensor_values = self.sensors.numpy()[0]
        for i, value in enumerate(sensor_values):
            sensor_text = self.font.render(f"Sensor {i+1}: {value:.2f}", True, (0, 0, 0))
            screen.blit(sensor_text, (20, 50 + i * 20))

    def display_reward(self, screen):
        reward = self.calculate_step_reward()
        reward_text = self.font.render(f"Reward: {reward}", True, (0, 0, 0))
        screen.blit(reward_text, (600, 550))

        speed_text = self.font.render(f"Speed: {self.speed:.2f}", True, (0, 0, 0))
        screen.blit(speed_text, (600, 570))

    def dec_angle(self):
        self.angle -= 2.5

    def inc_angle(self):
        self.angle += 2.5

    def inc_speed(self):
        self.speed = min(10, self.speed + 0.15)

    def dec_speed(self):
        self.speed = max(0, self.speed - 0.25)
