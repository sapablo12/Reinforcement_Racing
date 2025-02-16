import os
import sys
import math
from Agent import Agent
import pygame
import tensorflow as tf
from tensorflow.keras.models import load_model, Model, clone_model
import numpy as np
from tqdm import tqdm  # Add import for tqdm
from base_model import *
import random
#// vscode-fold=#
# Initialize Pygame
pygame.init()
LEARNING_RATE = 0.01
DISCOUNT_FACTOR = 0.95
# Define screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Racing Game Simulation")

def load_track():
    folder_path = os.path.join(os.path.dirname(__file__), 'current_track')
    file_path = os.path.join(folder_path, 'track.png')
    if os.path.exists(file_path):
        return pygame.image.load(file_path)
    else:
        print("No track found in the current_track folder.")
        return None

track = load_track()


def save_model(model):
    model.save("src/upmodel.keras")
    print("Model saved to src/upmodel.keras")

def run_simulation(agents):
    running = True
    clock = pygame.time.Clock()
    start_ticks = pygame.time.get_ticks()
    timer_font = pygame.font.Font(None, 36)
    game_active = True
    elapsed_seconds = 0

    while running:
        screen.fill((255, 255, 255))
        if track:
            screen.blit(track, (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Check for S key press to save the model
            if event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                save_model(agents[0].model)

        if game_active:
            for agent in agents:
                agent.update()
                agent.draw(screen)

            elapsed_seconds = (pygame.time.get_ticks() - start_ticks) / 1000
            timer_text = timer_font.render(f"Time: {elapsed_seconds:.2f} s", True, (0, 0, 0))
            screen.blit(timer_text, (20, 20))
            #reward_text = timer_font.render(f"Reward: {agents.calculate_step_reward():.2f}", True, (0, 0, 0))
            #screen.blit(reward_text, (SCREEN_WIDTH - reward_text.get_width() - 20, SCREEN_HEIGHT - reward_text.get_height() - 20))
            if elapsed_seconds >= 30:
                for agent in agents: 
                    if elapsed_seconds >= 30:
                        agent.active = False
                    if agent.speed < 0.01:
                        agent.active = False
                agent.check_wall()
                if agent.finish or not agent.active:
                    game_active = False
        else:
            running = False

        pygame.display.flip()
        clock.tick(60)
    experience_batch=[]
    for agent in agents:
        for data in agent.data:
            experience_batch.append(data)    
    return experience_batch

def tweak_random(flat_weights, sigma, ratio):
    index = []
    cant = int(round((ratio / 100) * len(flat_weights)))
    #print("cantidad: "+str(cant))
    for i in range(cant):
        number = np.random.randint(0, len(flat_weights))  # Ensure the index is within bounds
        flat_weights[number] = flat_weights[number] + sigma * np.random.uniform(-1, 1)
    return flat_weights


def get_experience(size, model, exploration):  
    flat_weights = flatten_weights(model.trainable_variables)
    current_agents = [Agent(track, model=model, weights=flat_weights, exploration=exploration,
                            color="green" if i == 0 else "blue")
                      for i in range(size)]
    return run_simulation(current_agents)

def episode(Q_model,target_model,exploration):   
    for i in tqdm(range(100), desc="Fase 1"):
        experiences = get_experience(size=20, model=Q_model, exploration=0.7)
        random.shuffle(experiences)
        states = np.array([data.state for data in experiences])
    
        targets = target_model.predict(states, verbose=0) 
        next_states = np.array([data.next_state for data in experiences])
        next_q_values = target_model.predict(next_states, verbose=0)  
    
    for idx, data in enumerate(experiences):
        new_target_max = data.step_reward + DISCOUNT_FACTOR * np.max(next_q_values[idx])
        targets[idx][data.action] = new_target_max
    
    Q_model.fit(states, targets, epochs=1, verbose=0) 
    target_model = clone_model(Q_model)
    target_model.set_weights(Q_model.get_weights())
    return Q_model,target_model

def train(Q_model):
    target_model = clone_model(Q_model)
    target_model.set_weights(Q_model.get_weights())
    Q_model,target_model=episode(Q_model,target_model,0.7)
    Q_model,target_model=episode(Q_model,target_model,0.4)
    Q_model,target_model=episode(Q_model,target_model,0.2)
   
    return Q_model

def try_model(model):
    agent = Agent(track, model,flatten_weights(model.trainable_variables))
    run_simulation(agent)

def main():
    
    Q_model = load_model("src/upmodel.keras")
    model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics=['accuracy'])  # Updated to native Keras format
    wmodel = train(Q_model)
    wmodel.save("src/defmodel.keras")  # Save using the native Keras format
    #try_model(model)

if __name__ == "__main__":
    main()
