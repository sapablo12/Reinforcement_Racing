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

def run_simulation(agent,check_rate=2):
    running = True
    clock = pygame.time.Clock()
    
    # Initialize timer
    start_ticks = pygame.time.get_ticks()  # Record the starting time
    timer_font = pygame.font.Font(None, 36)

    game_active = True  # Track active state for the agent
    elapsed_seconds = 0  # Initialize elapsed time for the agent
    last_displacement_check = 0  # Time of the last displacement check

    while running:
        screen.fill((255, 255, 255))  # Fill screen with white

        # Draw the track if it exists
        if track:
            screen.blit(track, (0, 0))

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if game_active:
            # Update and draw the agent
            agent.update2()
            agent.draw(screen)

            # Check for interaction with green spots
            #reward = agent_touch_green_spot(agent.x, agent.y, painter.green_spots, painter.canvas, painter.draw_image)
            #agent.reward += reward

            # Calculate and display timer
            elapsed_seconds = (pygame.time.get_ticks() - start_ticks) / 1000  # Convert to seconds
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
            # Check if the agent is finished or inactive
            if  agent.finish:
                game_active = False
                
                return agent.data
            if not agent.active:
                game_active = False
                
                return agent.data
            

        # Check if the agent is finished or inactive
        if not game_active:
            running = False

        # Update display
        pygame.display.flip()
        clock.tick(60)  # Cap the frame rate at 60 FPS

def tweak_random(flat_weights, sigma, ratio):
    index = []
    cant = int(round((ratio / 100) * len(flat_weights)))
    #print("cantidad: "+str(cant))
    for i in range(cant):
        number = np.random.randint(0, len(flat_weights))  # Ensure the index is within bounds
        flat_weights[number] = flat_weights[number] + sigma * np.random.uniform(-1, 1)
    return flat_weights


def train_batch(size,model,sigma,ratio,check_rate,exploration=10):
    databatch = []
    models = []
    databatch.append(run_simulation(Agent(track, model=model,weights=flatten_weights(model.trainable_variables)),check_rate=check_rate))
    flat_weights = flatten_weights(model.trainable_variables)
    for i in range(size): 
        current_agent = Agent(track, model=model,weights=flat_weights)
        #print(compare_models(model,current_agent.model))
        databatch.append(run_simulation(current_agent,check_rate))   
    #Some aggressive tweaking
    for j in range(exploration):    
        new_weights=tweak_random(flat_weights=flat_weights, sigma=sigma, ratio=50)
        current_agent = Agent(track, model=model,weights=new_weights)
        databatch.append(run_simulation(current_agent,check_rate))
    
    return flat_weights
        
    

def train(model):
    wmodel=model
    
    for i in tqdm(range(1), desc="Fase 1"):
        win_weights=train_batch(size=10,model=wmodel,sigma=0.05,ratio=30,check_rate=1)
        assign_weights(wmodel, win_weights)
        # Add your logic for the first loop here

    for j in tqdm(range(1), desc="Fase 2"):
        win_weights=train_batch(size=10,model=wmodel,sigma=0.05,ratio=10,check_rate=1.5)
        assign_weights(wmodel, win_weights)
        # Add your logic for the second loop here

    for k in tqdm(range(1), desc="Fase 3"):
        win_weights=train_batch(size=10,model=wmodel,sigma=0.01,ratio=5,check_rate=2)
        assign_weights(wmodel, win_weights)
    
    return wmodel
def try_model(model):
    agent = Agent(track, model,flatten_weights(model.trainable_variables))
    run_simulation(agent)

def main():
    model = load_model("model.h5")
    wmodel=train(model) 
    #wmodel.save("defmodel.h5")
    try_model(model)

if __name__ == "__main__":
    main()

# wmodel=load_model("model.h5")
# weights=flatten_weights(wmodel.trainable_variables)
# cmodel=clone_model(wmodel)
# assign_weights(cmodel,weights)
# print("antes: " +str(compare_models(wmodel,cmodel)))
# fweights=flatten_weights(cmodel.trainable_variables)
# fweights=tweak_random(flat_weights=fweights, sigma=0.1, ratio=20)
# assign_weights(cmodel,fweights)
# print("despues: " +str(compare_models(wmodel,cmodel)))