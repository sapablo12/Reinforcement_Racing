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

def agent_touch_green_spot(agent_x, agent_y, green_spots, canvas, draw_image):
    for spot in green_spots:
        x, y, size = spot
        if (x - size <= agent_x <= x + size) and (y - size <= agent_y <= y + size):
            green_spots.remove(spot)
            canvas.create_oval(x-size, y-size, x+size, y+size, fill='white', outline='white')
            draw_image.ellipse([x-size, y-size, x+size, y+size], fill='white', outline='white')
            return 100  # Reward for touching a green spot
    return 0

def run_simulation(agent, check_rate=2):
    running = True
    clock = pygame.time.Clock()
    start_ticks = pygame.time.get_ticks()
    timer_font = pygame.font.Font(None, 36)
    game_active = True
    elapsed_seconds = 0
    last_displacement_check = 0

    while running:
        screen.fill((255, 255, 255))
        if track:
            screen.blit(track, (0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if game_active:
            agent.update()
            agent.draw(screen)
            elapsed_seconds = (pygame.time.get_ticks() - start_ticks) / 1000
            timer_text = timer_font.render(f"Time: {elapsed_seconds:.2f} s", True, (0, 0, 0))
            screen.blit(timer_text, (20, 20))
            
            # Check displacement every 5 seconds
            if elapsed_seconds - last_displacement_check >= check_rate:
                
                last_displacement_check = elapsed_seconds

            if elapsed_seconds >= 4: 
                if elapsed_seconds>=30:
                    agent.active = False
                    
                if agent.speed < 0.1:
                    agent.active = False
                # Modified displacements check to compute Euclidean distance
                if len(agent.displacements) > 1 and np.linalg.norm(np.array(agent.displacements[-1]) - np.array(agent.displacements[-3])) < 0.5:
                    agent.active = False
                    

            agent.check_wall()
            if agent.finish or not agent.active:
                game_active = False
        else:
            running = False

        pygame.display.flip()
        clock.tick(60)
    return agent.data  # Moved return outside of the loop

def tweak_random(flat_weights, sigma, ratio):
    index = []
    cant = int(round((ratio / 100) * len(flat_weights)))
    #print("cantidad: "+str(cant))
    for i in range(cant):
        number = np.random.randint(0, len(flat_weights))  # Ensure the index is within bounds
        flat_weights[number] = flat_weights[number] + sigma * np.random.uniform(-1, 1)
    return flat_weights


def train_batch(size,model,check_rate,exploration):
    experiences = []
    flat_weights=flatten_weights(model.trainable_variables)
    current_agent = Agent(track, model=model,weights=flat_weights,exploration=exploration)
    for i in range(size): 
        run_simulation(current_agent,check_rate)  
    return experiences
        
def train(Q_model):
    target_model = clone_model(Q_model)
    target_model.set_weights(Q_model.get_weights())
    for i in tqdm(range(30), desc="Fase 1"):
        experiences = train_batch(size=20, model=Q_model, check_rate=10,exploration=0.4)
        np.random.shuffle(experiences)
        for data in experiences:
            target_value=target_model.predict(data.state)
            new_target_max=data.reward+DISCOUNT_FACTOR*np.max(target_model.predict(data.next_state))
            target_value[0][data.action]=new_target_max
            Q_model.fit(data.state,target_value, verbose=0)
        target_model = clone_model(Q_model)
        target_model.set_weights(Q_model.get_weights())    
    for i in tqdm(range(30), desc="Fase 2"):
        experiences = train_batch(size=20, model=Q_model, check_rate=10,exploration=0.2)
        np.random.shuffle(experiences)
        for data in experiences:
            target_value=target_model.predict(data.state)
            new_target_max=data.reward+DISCOUNT_FACTOR*np.max(target_model.predict(data.next_state))
            target_value[0][data.action]=new_target_max
            Q_model.fit(data.state,target_value, verbose=0)
        target_model = clone_model(Q_model)
        target_model.set_weights(Q_model.get_weights())    
    
    return Q_model

def try_model(model):
    agent = Agent(track, model,flatten_weights(model.trainable_variables))
    run_simulation(agent)

def main():
    
    Q_model = load_model("src/Q_model.keras")  # Updated to native Keras format
    wmodel = train(Q_model)
    wmodel.save("src/defmodel.keras")  # Save using the native Keras format
    #try_model(model)

if __name__ == "__main__":
    main()
