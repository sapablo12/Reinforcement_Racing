import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk


CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
TRACKS_DIR = os.path.join(os.path.dirname(__file__), "tracks")
CURRENT_TRACK_DIR = os.path.join(os.path.dirname(__file__), "current_track")
CURRENT_TRACK_PATH = os.path.join(CURRENT_TRACK_DIR, "track.png")

COLORS = {
    "Black": "black",
    "Red": "red",
    "Eraser": "white",
}


class TrackPainter:
    def __init__(self, root):
        self.root = root
        self.root.title("Track Painter")
        self.root.configure(bg="#f6f7f9")

        os.makedirs(TRACKS_DIR, exist_ok=True)
        os.makedirs(CURRENT_TRACK_DIR, exist_ok=True)

        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.track_photo = None

        self.color_name = tk.StringVar(value="Black")
        self.brush_size = tk.IntVar(value=8)
        self.status_text = tk.StringVar(value="")

        self.image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
        self.draw_image = ImageDraw.Draw(self.image)

        self.build_ui()
        self.bind_canvas_events()
        self.view_current_track(show_missing_message=False)

    def build_ui(self):
        main_frame = ttk.Frame(self.root, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(toolbar, text="Brush").grid(row=0, column=0, sticky="w", padx=(0, 8))
        color_menu = ttk.OptionMenu(toolbar, self.color_name, self.color_name.get(), *COLORS.keys())
        color_menu.grid(row=0, column=1, sticky="w", padx=(0, 18))

        ttk.Label(toolbar, text="Width").grid(row=0, column=2, sticky="w", padx=(0, 8))
        size_slider = ttk.Scale(
            toolbar,
            from_=2,
            to=40,
            orient=tk.HORIZONTAL,
            variable=self.brush_size,
            command=self.update_brush_label,
            length=170,
        )
        size_slider.grid(row=0, column=3, sticky="w")

        self.brush_label = ttk.Label(toolbar, text=str(self.brush_size.get()), width=3)
        self.brush_label.grid(row=0, column=4, sticky="w", padx=(8, 18))

        ttk.Button(toolbar, text="Save Track", command=self.save_track).grid(row=0, column=5, padx=(0, 8))
        ttk.Button(toolbar, text="Choose Current Track", command=self.choose_current_track).grid(row=0, column=6, padx=(0, 8))
        ttk.Button(toolbar, text="View Current Track", command=self.view_current_track).grid(row=0, column=7)

        toolbar.columnconfigure(8, weight=1)

        canvas_frame = ttk.Frame(main_frame, borderwidth=1, relief=tk.SOLID)
        canvas_frame.pack()

        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="white",
            highlightthickness=0,
        )
        self.canvas.pack()

        status = ttk.Label(main_frame, textvariable=self.status_text, anchor="w")
        status.pack(fill=tk.X, pady=(10, 0))

    def bind_canvas_events(self):
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def update_brush_label(self, _value=None):
        self.brush_label.configure(text=str(int(float(self.brush_size.get()))))

    def current_color(self):
        return COLORS[self.color_name.get()]

    def current_brush_size(self):
        return int(float(self.brush_size.get()))

    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        self.draw(event)

    def draw(self, event):
        if not self.drawing:
            return

        x = self.clamp(event.x, 0, CANVAS_WIDTH - 1)
        y = self.clamp(event.y, 0, CANVAS_HEIGHT - 1)
        color = self.current_color()
        width = self.current_brush_size()
        radius = width / 2

        self.canvas.create_line(self.last_x, self.last_y, x, y, fill=color, width=width, capstyle=tk.ROUND, joinstyle=tk.ROUND)
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=color, outline=color)

        self.draw_image.line((self.last_x, self.last_y, x, y), fill=color, width=width)
        self.draw_image.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=color)

        self.last_x = x
        self.last_y = y

    def stop_drawing(self, _event):
        self.drawing = False
        self.last_x = None
        self.last_y = None

    def save_track(self):
        path = filedialog.asksaveasfilename(
            title="Save Track",
            initialdir=TRACKS_DIR,
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
        )
        if not path:
            return

        self.image.save(path)
        self.status_text.set(f"Saved track: {path}")

    def choose_current_track(self):
        path = filedialog.askopenfilename(
            title="Choose Current Track",
            initialdir=TRACKS_DIR,
            filetypes=[("PNG image", "*.png")],
        )
        if not path:
            return

        shutil.copyfile(path, CURRENT_TRACK_PATH)
        self.view_current_track(show_missing_message=False)
        self.status_text.set(f"Current track set to: {path}")

    def view_current_track(self, show_missing_message=True):
        if not os.path.exists(CURRENT_TRACK_PATH):
            self.clear_canvas()
            message = f"No current track found at {CURRENT_TRACK_PATH}"
            self.status_text.set(message)
            if show_missing_message:
                messagebox.showinfo("Current Track", message)
            return

        with Image.open(CURRENT_TRACK_PATH) as track_image:
            self.image = track_image.convert("RGB").resize((CANVAS_WIDTH, CANVAS_HEIGHT))

        self.draw_image = ImageDraw.Draw(self.image)
        self.redraw_canvas_image()
        self.status_text.set(f"Viewing current track: {CURRENT_TRACK_PATH}")

    def clear_canvas(self):
        self.image = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "white")
        self.draw_image = ImageDraw.Draw(self.image)
        self.redraw_canvas_image()

    def redraw_canvas_image(self):
        self.canvas.delete("all")
        self.track_photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.track_photo)

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(minimum, min(value, maximum))


if __name__ == "__main__":
    root = tk.Tk()
    painter = TrackPainter(root)
    root.mainloop()
