import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageTk
import os

class TrackPainter:
    def __init__(self, root):
        self.root = root
        self.root.title("Track Painter")
        
        # Set up the canvas
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()
        
        # Variables to keep track of drawing
        self.drawing = False
        self.last_x, self.last_y = None, None
        self.current_color = 'black'
        self.green_spots = []  # List to keep track of green spots

        # Bind mouse events to canvas
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)

        # Frame to hold buttons in a neat table
        self.button_frame = tk.Frame(root, relief=tk.RIDGE, borderwidth=2, bg='#f0f0f0')
        self.button_frame.pack(pady=10, padx=10)

        # Add buttons in a grid layout
        self.black_button = tk.Button(self.button_frame, text="Switch to Black", command=self.switch_to_black, width=20)
        self.black_button.grid(row=0, column=0, padx=5, pady=5)
        
        self.red_button = tk.Button(self.button_frame, text="Switch to Red", command=self.switch_to_red, width=20)
        self.red_button.grid(row=0, column=1, padx=5, pady=5)
        
        self.green_button = tk.Button(self.button_frame, text="Switch to Green", command=self.switch_to_green, width=20)
        self.green_button.grid(row=0, column=2, padx=5, pady=5)
        
        self.fill_button = tk.Button(self.button_frame, text="Fill Area", command=self.fill_area, width=20)
        self.fill_button.grid(row=1, column=2, padx=5, pady=5)
        
        self.save_current_button = tk.Button(self.button_frame, text="Save Track to Current Folder", command=self.save_track_current_folder, width=20)
        self.save_current_button.grid(row=1, column=0, padx=5, pady=5)
        
        self.view_current_button = tk.Button(self.button_frame, text="View Current Track", command=self.view_current_track, width=20)
        self.view_current_button.grid(row=1, column=1, padx=5, pady=5)
        
        # Create an image to draw on (for saving purposes)
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw_image = ImageDraw.Draw(self.image)

    def start_drawing(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            # Draw on the canvas
            if self.current_color == 'green':
                size = 10  # Bigger size for green spots
                self.green_spots.append((x, y, size))  # Add green spot to the list
            else:
                size = 3
            self.canvas.create_line((self.last_x, self.last_y, x, y), fill=self.current_color, width=6)
            self.canvas.create_oval(x-size, y-size, x+size, y+size, fill=self.current_color, outline=self.current_color)
            # Draw on the image
            self.draw_image.line((self.last_x, self.last_y, x, y), fill=self.current_color, width=6)
            self.draw_image.ellipse([x-size, y-size, x+size, y+size], fill=self.current_color, outline=self.current_color)
            self.last_x, self.last_y = x, y

    def stop_drawing(self, event):
        self.drawing = False
        self.last_x, self.last_y = None, None

    def switch_to_black(self):
        self.current_color = 'black'

    def switch_to_red(self):
        self.current_color = 'red'
    
    def switch_to_green(self):
        self.current_color = 'green'

    def fill_area(self):
        # Fill the entire canvas with the current color
        if self.current_color == 'green':
            fill_color = '#00FF00'  # Hex code for (0, 255, 0)
        else:
            fill_color = self.current_color
        self.canvas.create_rectangle(0, 0, self.canvas_width, self.canvas_height, fill=fill_color, outline=fill_color)
        self.draw_image.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=fill_color, outline=fill_color)

    def save_track_current_folder(self):
        # Define the path to the 'current_track' folder
        folder_path = os.path.join(os.path.dirname(__file__), 'current_track')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Define the file path for the track image
        file_path = os.path.join(folder_path, 'track.png')
        
        # If a file already exists, it will be replaced
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Save the current track
        self.image.save(file_path)
        print(f"Track saved to {file_path}")

    def view_current_track(self):
        # Define the path to the 'current_track' folder
        folder_path = os.path.join(os.path.dirname(__file__), 'current_track')
        file_path = os.path.join(folder_path, 'track.png')
        
        # Check if the track file exists
        if os.path.exists(file_path):
            # Open the image and display it on the canvas
            track_image = Image.open(file_path)
            self.image = track_image.copy()
            self.draw_image = ImageDraw.Draw(self.image)
            self.track_photo = ImageTk.PhotoImage(track_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.track_photo)
        else:
            print("No track found in the current_track folder.")

if __name__ == "__main__":
    root = tk.Tk()
    painter = TrackPainter(root)
    root.mainloop()
