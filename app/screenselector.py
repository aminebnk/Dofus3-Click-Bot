import tkinter as tk
from tkinter import messagebox, simpledialog
from cbt import screenshot_high_res, resource_path
import mss
import cv2
import numpy as np
import os


""" This file takes care of the name tag registering process. Creates a sub app that takes a screenshot of the user entry zone and cuts out the nametag automatically so it can be used by the main loop. """
SAVE_FOLDER = resource_path("templates/names")

class ScreenSelector(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        # Hide main window
        master.withdraw()

        self.overrideredirect(True)
        # Get main screen size
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        # Maximize window manually (not macOS full-screen mode)
        self.geometry(f"{screen_w}x{screen_h}+0+0")
        self.lift()
        self.attributes("-topmost", True)
        self.attributes('-alpha', 0.2)
        self.configure(bg='black')

        self.canvas = tk.Canvas(self, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.start_x = self.start_y = None
        self.rect_id = None

        self.bind("<ButtonPress-1>", self.on_press)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.bind("<Escape>", self.cancel)

    def on_press(self, event):
        self.start_x, self.start_y = event.x, event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)

        self.rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline="red", width=2
        )

    def on_drag(self, event):
        if self.rect_id:
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        if not self.rect_id:
            return

        # Normalize coords
        x1, y1, x2, y2 = self.canvas.coords(self.rect_id)
        x1, x2 = sorted([int(x1), int(x2)])
        y1, y2 = sorted([int(y1), int(y2)])

        w = x2 - x1
        h = y2 - y1

        if w < 3 or h < 3:
            messagebox.showerror("Error", "Selection too small.")
            self.cancel()
            return

        # Close overlay
        self.destroy()
        self.master.update()

        # Capture region
        self.capture_region(x1, y1, w, h)

        self.master.deiconify()

    def cancel(self, event=None):
        self.destroy()
        self.master.deiconify()

    def capture_region(self, x, y, w, h):
        img = screenshot_high_res(x, y, w, h)

        min_x = 3000
        max_x = 0
        min_y = 3000
        max_y = 0

        for i, line in enumerate(img):
            for j, pixel in enumerate(line):
                if pixel[0] == 111 and pixel[1] == 255 and pixel[2] == 243:
                    if i < min_y:
                        min_y = i
                    if i > max_y:
                        max_y = i
                    if j < min_x:
                        min_x = j
                    if j > max_x:
                        max_x = j
        
        min_x = round(min_x * 0.5)
        min_y = round(min_y * 0.5)
        max_x = round(max_x * 0.5)
        max_y = round(max_y * 0.5)

        with mss.mss() as sct:
            region = {
                "top": y + min_y - 1,
                "left": x + min_x - 1,
                "width": max_x - min_x + 3,
                "height": max_y - min_y + 3
            }
            name_tag = sct.grab(region)
            name_tag = np.array(name_tag)
            name_tag = cv2.cvtColor(name_tag, cv2.COLOR_BGRA2BGR)

        name = simpledialog.askstring("Save As", "Enter filename:")
        if not name:
            return
        
        filepath = os.path.join(SAVE_FOLDER, f"{name}.png")

        cv2.imwrite(filepath, name_tag)
        messagebox.showinfo("Saved", f"Saved as:\n{filepath}")

def run(master):
    return ScreenSelector(master)

    