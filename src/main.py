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
DISCOUNT_FACTOR = 0.95
# Define screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SENSOR_LENGTH = 250
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
    model.save("src/upmodel_compact.keras")
    print("Model saved to src/upmodel_compact.keras")

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
            if elapsed_seconds >= 60:
                for agent in agents: 

                    agent.active = False
            if all(agent.finish or not agent.active for agent in agents):
                game_active = False
        else:
            running = False

        pygame.display.flip()
        clock.tick(60)
    experience_batch=[]
    
    for agent in agents: 
        experience_batch.extend(agent.data)    
    return experience_batch


def get_experience(size, model, exploration):
    run_size=30  
    all_experiences = []
    priority_batch=[]
    num_simulations = -(-size // run_size)  # Ceiling division
    for _ in range(num_simulations):
        flat_weights = flatten_weights(model.trainable_variables)
        current_agents = [Agent(track, model=model, weights=flat_weights, exploration=exploration,
                                color="green" if i == 0 else "blue")
                          for i in range(run_size)]
        experiences = run_simulation(current_agents)
        all_experiences.extend(experiences)
    return all_experiences

def episode(Q_model, target_model,exploration=0.85,size=10000,batch_size=30):
    #Priority discarded
    memory_buffer = []
    current=2700   
    for i in tqdm(range(current,size), desc="Fase exploration = "+str(exploration)):
        exp = exploration - (exploration - 0.05) * (i / size)
        print("\n"+str(current))
        print("Exploration: " + str(exp))
        current+=1
        new_experiences = get_experience(size=batch_size, model=Q_model, exploration=exp)
        print("Experience size: ",len(new_experiences))
        memory_buffer.extend(new_experiences)
        experiences = random.sample(memory_buffer, 256)
        states = np.array([data.state for data in experiences])
        targets = target_model.predict(states, verbose=0) 
        next_states = np.array([data.next_state for data in experiences])
        next_q_values = Q_model.predict(next_states, verbose=0)  
        actions_argmax = np.argmax(next_q_values, axis=1)
        target_next_q= target_model.predict(next_states, verbose=0)
        for idx, data in enumerate(experiences):
            new_target_max = data.step_reward + DISCOUNT_FACTOR * target_next_q[idx][actions_argmax[idx]]
            targets[idx][data.action] = new_target_max
    
        Q_model.fit(states, targets, epochs=1, verbose=0)
        """priority_states=states[-n:]
        priority_targets=targets[-n:]
        Q_model.fit(priority_states,priority_targets, epochs=3, verbose=0)"""
        if i % 10 == 0:  # Update target model periodically
            target_model.set_weights(Q_model.get_weights())
        Q_model.save("src/upmodel_compact.keras")
    return Q_model, target_model

def train(Q_model):
    target_model = clone_model(Q_model)
    target_model.set_weights(Q_model.get_weights())
    Q_model, target_model = episode(Q_model, target_model,0.7)
    """Q_model, target_model = episode(Q_model, target_model,0.7)
    Q_model, target_model = episode(Q_model, target_model,0.7)
    Q_model, target_model = episode(Q_model, target_model,0.4)
    Q_model, target_model = episode(Q_model, target_model,0.2)"""
   
    return Q_model

def try_model(Q_model):
    flat_weights = flatten_weights(Q_model.trainable_variables)
    for i in range(100):
        agent = Agent(track, model=Q_model, weights=flat_weights, exploration=0.4, color="green")
        run_simulation([agent])


def manual(Q_model):
    # Initialize the agent
    flat_weights = flatten_weights(Q_model.trainable_variables)
    agent = Agent(track, model=Q_model, weights=flat_weights, exploration=0.0, color="green")

    # Set up the simulation environment
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

        if game_active:
            agent.update2()
            agent.draw(screen)

            elapsed_seconds = (pygame.time.get_ticks() - start_ticks) / 1000
            timer_text = timer_font.render(f"Time: {elapsed_seconds:.2f} s", True, (0, 0, 0))
            screen.blit(timer_text, (20, 20))
            if elapsed_seconds >= 120:
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

    experience_batch = []
    for data in agent.data:
        experience_batch.append(data)
    return experience_batch

def main():
    
    Q_model = load_model("src/upmodel_compact.keras")
    Q_model.compile(optimizer='adam', 
              loss='mean_squared_error', 
              metrics=['accuracy']) 
    """"for i in range(100):
        manual(Q_model)"""
    
    wmodel = train(Q_model)
    wmodel.save("src/upmodel_compact.keras")  # Save using the native Keras format
    #try_model(model)
    """while True:
        manual(Q_model) # Updated to native Keras format"""
if __name__ == "__main__":
    main()