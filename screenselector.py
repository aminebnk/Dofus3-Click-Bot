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

        # Convert to HSV to detect yellow text robustly
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Yellow range in HSV (OpenCV Scale: H: 0-179, S: 0-255, V: 0-255)
        # Yellow is roughly 30. We take a range around it.
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        points = cv2.findNonZero(mask)
        
        if points is None:
            messagebox.showerror("Error", "Could not find nametag in the selected area (yellow text not found).")
            return

        x_coords, y_coords, w_rect, h_rect = cv2.boundingRect(points)
        min_x = x_coords
        min_y = y_coords
        max_x = x_coords + w_rect
        max_y = y_coords + h_rect

        # Check if we are likely on a Retina display or high-DPI scaling
        # If the screenshot img has different dimensions than w, h
        scale_x = img.shape[1] / w
        scale_y = img.shape[0] / h
        
        # Adjust min/max coordinates back to screen coordinates
        # We found min_x inside img (scaled). We want coordinates relative to x, y (screen).
        
        # If scale is 2 (Retina), min_x is in 2x coords. We need to divide by 2.
        # But wait, mss monitor uses physical coords usually? Or logical?
        # mss.grab(region) expects coordinates in the same system as the desktop geometry.
        
        # If the initial screenshot_high_res returned an image that is scaled (e.g. 2x),
        # then we need to scale down our found coordinates to pass them back to mss.grab
        
        min_x = int(min_x / scale_x)
        min_y = int(min_y / scale_y)
        max_x = int(max_x / scale_x)
        max_y = int(max_y / scale_y)

        # Ensure width and height are positive
        width = max_x - min_x + 3
        height = max_y - min_y + 3
        
        if width <= 0 or height <= 0:
             messagebox.showerror("Error", "Invalid nametag dimensions.")
             return

        with mss.mss() as sct:
            region = {
                "top": y + min_y - 1,
                "left": x + min_x - 1,
                "width": width,
                "height": height
            }
            try:
                name_tag = sct.grab(region)
                name_tag = np.array(name_tag)
                name_tag = cv2.cvtColor(name_tag, cv2.COLOR_BGRA2BGR)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to capture nametag: {e}")
                return

        name = simpledialog.askstring("Save As", "Enter filename:")
        if not name:
            return
        
        filepath = os.path.join(SAVE_FOLDER, f"{name}.png")

        cv2.imwrite(filepath, name_tag)
        messagebox.showinfo("Saved", f"Saved as:\n{filepath}")

class RegionSelector(tk.Toplevel):
    def __init__(self, master, callback):
        super().__init__(master)
        self.master = master
        self.callback = callback

        # Hide main window
        master.withdraw()

        self.overrideredirect(True)
        # Get main screen size
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        # Maximize window manually
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
            outline="blue", width=2
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
        self.master.deiconify()
        
        # Check for Retina/High DPI scaling by comparing screenshot size to screen size
        # We take a dummy screenshot of the whole screen to check dimensions
        with mss.mss() as sct:
             # Grab the monitor that corresponds to (0,0) - usually monitor 1
             # We assume the window is on the primary monitor for simplicity in calibration
             monitor = sct.monitors[1]
             # monitor width/height
             mon_w = monitor["width"]
             mon_h = monitor["height"]
             
             # Screen size reported by tkinter
             scr_w = self.winfo_screenwidth()
             scr_h = self.winfo_screenheight()
             
             scale_x = mon_w / scr_w
             scale_y = mon_h / scr_h

        # Adjust coordinates
        final_x = int(x1 * scale_x)
        final_y = int(y1 * scale_y)
        final_w = int(w * scale_x)
        final_h = int(h * scale_y)

        if self.callback:
            self.callback(final_x, final_y, final_w, final_h)

    def cancel(self, event=None):
        self.destroy()
        self.master.deiconify()

def run(master):
    return ScreenSelector(master)

def select_region(master, callback):
    return RegionSelector(master, callback)

if __name__ == "__main__":
    def test_cb(x, y, w, h):
        messagebox.showinfo("Result", f"Selected: x={x}, y={y}, w={w}, h={h}")

    root = tk.Tk()
    root.geometry("200x200")
    tk.Button(root, text="Test Selection", command=lambda: select_region(root, test_cb)).pack(expand=True)
    root.mainloop()
