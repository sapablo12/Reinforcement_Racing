import pygame
import sys
import math
import numpy as np  # Added import for NumPy
import tensorflow as tf
from tensorflow.keras.models import Model,clone_model  # Import the Model class from Keras
from tqdm import tqdm  # Add import for tqdm
from base_model import *  # Import the functions from base_model.py

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600

# Add new class Info for storing update data
class Info:
    def __init__(self, state, output, act,step_reward, next_state, done_flag):
        self.state = state
        self.output = output
        self.action=act
        self.step_reward = step_reward
        self.done_flag = done_flag
        self.next_state=next_state

class Agent:
    def __init__(self, track, model: Model,weights,exploration):  # Add type hint for model
        self.track = track
        self.x_initial = 0
        self.y_initial = 420
        self.x=self.x_initial
        self.y=self.y_initial
        self.displacements = []  # Initialize self.displacements as an empty list
        self.size = 20  # Size of the agent (square)
        self.angle = 0  # Angle the agent is facing (in degrees)
        self.speed = 1  # Movement speed
        self.sensors = tf.Variable(tf.zeros([1, 5]), dtype=tf.float32)  # Distance values for the 5 sensors as a tensor
        self.finish = False
        self.active = True  # Flag to check if agent is active
        self.model: Model = clone_model(model)
        self.epsilon=exploration
        assign_weights(model=self.model,flat_weights=weights)  # Explicitly declare the type of self.model
        self.total_distance = 0  # Total distance traveled by the agent
        self.total_time = 0  # Total time the agent has been active
        self.reward=0
        self.font = pygame.font.Font(None, 24)  # Font for displaying sensor values
        self.data = []  # Initialize self.data as an empty list
        # Initialize manual controls for continuous user input
    def draw(self, screen):
        # Draw the agent as a square
        half_size = self.size / 2
        points = [
            (self.x - half_size, self.y - half_size),
            (self.x + half_size, self.y - half_size),
            (self.x + half_size, self.y + half_size),
            (self.x - half_size, self.y + half_size)
        ]
        rotated_points = [self.rotate_point(px, py, self.angle) for px, py in points]
        rotated_points = [(int(px), int(py)) for px, py in rotated_points]  # Ensure points are number pairs
        pygame.draw.polygon(screen, (0, 0, 255), rotated_points)

        # Draw sensors
        self.draw_sensors(screen)
        self.display_sensor_values(screen)
    def rotate_point(self, px, py, angle):
        # Rotate a point around the agent's center using NumPy
        rad_angle = np.radians(angle)
        sin_a = np.sin(rad_angle)
        cos_a = np.cos(rad_angle)
        cx, cy = self.x, self.y
        x_shifted, y_shifted = px - cx, py - cy
        x_new = x_shifted * cos_a - y_shifted * sin_a + cx
        y_new = x_shifted * sin_a + y_shifted * cos_a + cy
        return (x_new, y_new)
    
    def check_wall(self):
        # Check if the agent has hit a wall (black pixel)
        if 0 <= self.x < SCREEN_WIDTH and 0 <= self.y < SCREEN_HEIGHT:
            color = self.track.get_at((int(self.x), int(self.y)))
            if color == (0, 0, 0) :  # Black  or green color indicates a wall
                self.active = False

    def dispersion(self,a):
        # Calculate the sum of the distance from the last three displacements
        if len(self.displacements) < a:
            return 0  # Not enough displacements to calculate
        distances=0.0
        last_displacements = self.displacements[-a:]
        for i in range(0, a, 5):
            distances += np.linalg.norm(np.array(last_displacements[i]) - np.array([self.x, self.y]))

        
        """for i in range(a):
            distances += np.linalg.norm(np.array(last_displacements[i]) - np.array([self.x, self.y]))
        """
        return distances

        
    def draw_sensors(self, screen):
        sensor_length = 100
        sensor_angles = np.array([0, -45, 45, -90, 90])  # Front, front-left, front-right, left, right
        sensor_colors = (0, 0, 255)  # blue color for sensors

        new_sensor_values = []

        for i, angle_offset in enumerate(sensor_angles):
            angle = self.angle + angle_offset
            distance = self.calculate_sensor_distance(angle, sensor_length)
            new_sensor_values.append(distance / 100)

            # Calculate the end point of the sensor line
            end_x = self.x + distance * np.cos(np.radians(angle))
            end_y = self.y + distance * np.sin(np.radians(angle))
            pygame.draw.line(screen, sensor_colors, (int(self.x), int(self.y)), (int(end_x), int(end_y)), 2)

        self.sensors.assign(tf.convert_to_tensor([new_sensor_values], dtype=tf.float32))
    
    def update_sensors(self):
        sensor_length = 100
        sensor_angles = np.array([0, -45, 45, -90, 90])  # Front, front-left, front-right, left, right

        new_sensor_values = []

        for i, angle_offset in enumerate(sensor_angles):
            angle = self.angle + angle_offset
            distance = self.calculate_sensor_distance(angle, sensor_length)
            new_sensor_values.append(distance / 100)

        return tf.convert_to_tensor([new_sensor_values], dtype=tf.float32)


    def calculate_sensor_distance(self, angle, max_length):
        # Calculate the distance from the agent to the point where the sensor meets a black wall
        for distance in np.arange(0, max_length, 1):  # Decrease step size for higher resolution
            sensor_x = self.x + distance * np.cos(np.radians(angle))
            sensor_y = self.y + distance * np.sin(np.radians(angle))
            
            # Check if the sensor point is within bounds
            if 0 <= sensor_x < SCREEN_WIDTH and 0 <= sensor_y < SCREEN_HEIGHT:
                # Get the color of the pixel at the sensor point
                color = self.track.get_at((int(sensor_x), int(sensor_y)))
                if color == (0, 0, 0):  # Black color indicates a wall
                    if distance == 0:
                        self.active = False
                    return distance
                elif color == (255, 0, 0):  # Red color indicates a finish line
                    if distance == 0:
                        self.active = False
                        self.finish = True
                    return distance
            else:
                 # Sensor is out of bounds
                return distance
        return max_length
    
    def update(self):
        if self.active:
            if self.model is not None:
                state = tf.concat([self.sensors, [[self.speed / 10.0]]], axis=1)  # Normalize speed to 0-1
                output = self.model(state)
                act = tf.argmax(output, axis=1).numpy()[0]
                actions = {
                        0: self.dec_angle,
                        1: self.inc_angle,
                        2: self.inc_speed,
                        3: self.dec_speed,
                    }
                if np.random.rand() < self.epsilon:  # Exploration
                    act=np.random.choice(4)
                actions.get(act, lambda: None)()
                # Update the position of the agent based on speed and angle
                self.x += self.speed * np.cos(np.radians(self.angle))
                self.y += self.speed * np.sin(np.radians(self.angle))
                self.total_distance += self.speed
                self.total_time += 1  # Assuming each update is one time unit

                # Append new displacement data (x, y) to self.displacements
                new_displacement = (self.x, self.y)
                self.displacements.append(new_displacement)

                # Keep the agent within screen bounds
                if self.x <= 0 or self.x >= SCREEN_WIDTH or self.y <= 0 or self.y >= SCREEN_HEIGHT:
                    self.active = False
                else:
                    color = self.track.get_at((int(self.x), int(self.y)))
                    if color == (0, 0, 0) or color == (0, 255, 0):
                        self.active = False
                    elif color == (255, 0, 0):
                        self.finish = True
                
                # Calculate reward for this step
                step_reward = self.calculate_step_reward()
                print(step_reward)
                # Update sensors for the next time step
                
                next_sensors = self.update_sensors().numpy().tolist()

                # Determine if the agent is done
                done_flag = 1.0 if self.finish else 0.0
                next_state=tf.concat([next_sensors, [[self.speed / 10.0]]], axis=1)
                # Replace tuple with Info instance
                new_data = Info(state, output.numpy().tolist(),act, step_reward, next_state, done_flag)
                self.data.append(new_data)
                
                self.x = np.clip(self.x, 0, SCREEN_WIDTH)
                self.y = np.clip(self.y, 0, SCREEN_HEIGHT)
        else:
            new_displacement = (self.x, self.y)
            self.displacements.append(new_displacement)
            step_reward = self.calculate_step_reward()
            print(step_reward)
            

            # Determine if the agent is done
            done_flag = 1.0 if self.finish else 0.0
            if not self.finish:
                step_reward = -100
                done_flag = 0.0
            else:
                done_flag = 1.0
                print("Ha muerto" + done_flag)
            # Define output for consistency in the inactive branch
            output = tf.zeros([1, 2])
            # Replace tuple with Info instance
            new_data = Info(tf.zeros([1, 5]).numpy().tolist(), output.numpy().tolist(), np.random.rand(0, 4), step_reward, tf.zeros([1, 5]).numpy().tolist(), done_flag)
            self.data.append(new_data)

    def calculate_step_reward(self):
       if self.active:
            if self.finish:
                    return self.speed + 1*self.dispersion() + 1000
            else:
                    return self.speed + 0.1*self.dispersion(150)
       else:
            if self.finish:
                return self.speed+1000
            else:
                return -1000


    def calculate_mean_speed(self):
        if self.total_time > 0:
            
            return self.total_distance / self.total_time
        return 0

    def display_sensor_values(self, screen):
        sensor_values = self.sensors.numpy()[0]
        for i, value in enumerate(sensor_values):
            sensor_text = self.font.render(f"Sensor {i+1}: {value:.2f}", True, (0, 0, 0))
            screen.blit(sensor_text, (20, 50 + i * 20))

    
    def update2(self):
            if self.active:
                keys = pygame.key.get_pressed()
                # Compute continuous delta values
                # Adjust angle directly with left/right arrows
                if keys[pygame.K_a]:
                    self.dec_angle()
                if keys[pygame.K_d]:
                    self.inc_angle()
                # Adjust speed continuously with up/down arrows
                
                if keys[pygame.K_w]:
                    self.inc_speed()
                if keys[pygame.K_s]:
                    self.dec_speed()

                # Accumulate and clamp the continuous values between 0.0 and 1.0

                # Create user output using the continuous values
                output = tf.convert_to_tensor([[0,0]], dtype=tf.float32)
                
                # ...existing code for updating position...
                self.x += self.speed * np.cos(np.radians(self.angle))
                self.y += self.speed * np.sin(np.radians(self.angle))
                self.total_distance += self.speed
                self.total_time += 1

                new_displacement = (self.x, self.y)
                self.displacements.append(new_displacement)

                if self.x <= 0 or self.x >= SCREEN_WIDTH or self.y <= 0 or self.y >= SCREEN_HEIGHT:
                    self.active = False
                else:
                    color = self.track.get_at((int(self.x), int(self.y)))
                    if color == (0, 0, 0) or color == (0, 255, 0):
                        self.active = False
                    elif color == (255, 0, 0):
                        self.finish = True

                step_reward = self.calculate_step_reward()
                #zprint(step_reward)
                next_sensors = self.update_sensors().numpy().tolist()
                done_flag = 1.0 if self.finish else 0.0
                new_data = Info(self.sensors.numpy().tolist(), output.numpy().tolist(), step_reward, next_sensors, done_flag)
                self.data.append(new_data)

                self.x = np.clip(self.x, 0, SCREEN_WIDTH)
                self.y = np.clip(self.y, 0, SCREEN_HEIGHT)
            else:
                new_displacement = (self.x, self.y)
                self.displacements.append(new_displacement)
                step_reward = self.calculate_step_reward()
                #print(step_reward)
                done_flag = 1.0 if self.finish else 0.0
                if not self.finish:
                    step_reward = -100
                    done_flag = 0.0
                else:
                    done_flag = 1.0
                output = tf.zeros([1, 2])
                new_data = Info(tf.zeros([1, 5]).numpy().tolist(), output.numpy().tolist(), step_reward, tf.zeros([1, 5]).numpy().tolist(), done_flag)
                self.data.append(new_data)

    def dec_angle(self):
        self.angle -= 2.5

    def inc_angle(self):
        # Turn right: increase angle by 5 degrees
        self.angle += 2.5

    def inc_speed(self):
        # Increase speed by 0.5
        self.speed = min(10, self.speed+0.1)  # Ensure speed doesn't exceed 7.5

    def dec_speed(self):
        # Decrease speed by 0.5, ensuring it doesn't drop below zero
        self.speed = max(0, self.speed - 0.2)