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
            agent.update2()
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


def train_batch(size,model,check_rate):
    experiences = []
    flat_weights=flatten_weights(model.trainable_variables)
    current_agent = Agent(track, model=model,weights=flat_weights)
    for i in range(size): 
        run_simulation(current_agent,check_rate)  
    return experiences
        
def train(model):
    wmodel=model
    for i in tqdm(range(100), desc="Fase 1"):
        experiences=train_batch(size=10,model=wmodel,check_rate=10)
  
    return wmodel
def try_model(model):
    agent = Agent(track, model,flatten_weights(model.trainable_variables))
    run_simulation(agent)

def main():
    model = load_model("src/model.keras")  # Updated to native Keras format
    wmodel = train(model)
    wmodel.save("src/defmodel.keras")  # Save using the native Keras format
    try_model(model)

if __name__ == "__main__":
    main()
