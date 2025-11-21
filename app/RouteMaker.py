import tkinter as tk
from tkinter import font, messagebox, filedialog
import math
import os

GRID_SIZE = 200
HALF = GRID_SIZE // 2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTES_FOLDER = os.path.join(BASE_DIR, "resources", "routes")

class RouteMaker(tk.Toplevel): 
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Route Maker")
        self.geometry("900x600")

        # Top bar
        top = tk.Frame(self)
        top.pack(side="top", fill="x")
        self.save_btn = tk.Button(top, text="Enregistrer", command=self.open_save_window)
        self.save_btn.pack(side="right", padx=6, pady=6)

        self.load_btn = tk.Button(top, text="Charger", command=self.load_route)
        self.load_btn.pack(side="right", padx=6, pady=6)

        tk.Label(
            top,
            text="Clic gauche = ajouter une case | Retour arrière = supprimer la dernière case"
        ).pack(side="left", padx=6)

        # Canvas
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(fill="both", expand=True)

        # State
        self.cell_size = 80
        self.offset_x = 0
        self.offset_y = 0
        self.route = []  # route = list of tiles (x,y)

        self.font_small = font.Font(size=8)

        # Mouse state
        self.pan_start = None
        self.pan_offset_start = None
        self.click_pos = None
        self.dragged = False

        # Bind events
        self.canvas.bind("<Configure>", self.on_configure)
        self.canvas.bind("<ButtonPress-1>", self.on_left_press)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)

        # Bind Backspace
        self.bind("<BackSpace>", self.on_backspace)

        self.redraw()

    # # # # # # # # # # # # #
    # coordinate transforms #
    # # # # # # # # # # # # #

    def world_to_screen(self, wx, wy):
        cx = self.canvas.winfo_width() // 2
        cy = self.canvas.winfo_height() // 2
        sx = cx + int(wx * self.cell_size) + self.offset_x
        sy = cy + int(wy * self.cell_size) + self.offset_y
        return sx, sy

    def screen_to_world_float(self, sx, sy):
        cx = self.canvas.winfo_width() // 2
        cy = self.canvas.winfo_height() // 2
        wx = (sx - cx - self.offset_x) / self.cell_size
        wy = (sy - cy - self.offset_y) / self.cell_size
        return wx, wy

    # # # # # #
    # drawing #
    # # # # # #

    def redraw(self):
        self.canvas.delete("all")
        self.draw_grid()
        self.draw_labels()
        self.draw_route()

    def draw_grid(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        left_wx, top_wy = self.screen_to_world_float(0, 0)
        right_wx, bottom_wy = self.screen_to_world_float(w, h)

        min_x = max(math.floor(left_wx - 0.5), -HALF)
        max_x = min(math.ceil(right_wx + 0.5), HALF)
        min_y = max(math.floor(top_wy - 0.5), -HALF)
        max_y = min(math.ceil(bottom_wy + 0.5), HALF)

        for gx in range(min_x, max_x + 1):
            sx1, sy1 = self.world_to_screen(gx + 0.5, min_y)
            sx2, sy2 = self.world_to_screen(gx + 0.5, max_y)
            self.canvas.create_line(sx1, sy1, sx2, sy2, fill="#ccc")

        for gy in range(min_y, max_y + 1):
            sx1, sy1 = self.world_to_screen(min_x, gy + 0.5)
            sx2, sy2 = self.world_to_screen(max_x, gy + 0.5)
            self.canvas.create_line(sx1, sy1, sx2, sy2, fill="#ccc")

    def draw_labels(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        left_wx, top_wy = self.screen_to_world_float(0, 0)
        right_wx, bottom_wy = self.screen_to_world_float(w, h)

        min_x = max(math.floor(left_wx) - 1, -HALF)
        max_x = min(math.ceil(right_wx) + 1, HALF - 1)
        min_y = max(math.floor(top_wy) - 1, -HALF)
        max_y = min(math.ceil(bottom_wy) + 1, HALF - 1)

        for gy in range(min_y, max_y + 1):
            for gx in range(min_x, max_x + 1):
                sx, sy = self.world_to_screen(gx, gy)
                txt = f"({gx},{gy})"
                self.canvas.create_text(
                    sx, sy, text=txt, font=self.font_small, fill="#333"
                )

    def draw_route(self):
        if not self.route:
            return

        # drawing the lines
        for i in range(len(self.route) - 1):
            a = self.route[i]
            b = self.route[i + 1]
            ax, ay = self.world_to_screen(*a)
            bx, by = self.world_to_screen(*b)
            self.canvas.create_line(ax, ay, bx, by, fill="black", width=2)

        # drawing the points
        for (x, y) in self.route:
            sx, sy = self.world_to_screen(x, y)
            r = 4  # rayon du point
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill="black")

    # # # # # #
    # events  #
    # # # # # #

    def on_configure(self, event):
        self.redraw()

    def on_left_press(self, event):
        self.click_pos = (event.x, event.y)
        self.pan_start = (event.x, event.y)
        self.pan_offset_start = (self.offset_x, self.offset_y)
        self.dragged = False

    def on_left_drag(self, event):
        if self.pan_start is None:
            return
        dx = event.x - self.pan_start[0]
        dy = event.y - self.pan_start[1]
        if abs(dx) > 2 or abs(dy) > 2:
            self.dragged = True
        self.offset_x = self.pan_offset_start[0] + dx
        self.offset_y = self.pan_offset_start[1] + dy
        self.redraw()

    def on_left_release(self, event):
        if not self.dragged:
            self.add_to_route(event.x, event.y)
            self.redraw()
        self.click_pos = None
        self.pan_start = None
        self.pan_offset_start = None
        self.dragged = False

    def on_backspace(self, event):
        """Gets rid of the last added point"""
        if self.route:
            self.route.pop()
            self.redraw()

    # # # # # # # # #
    # route editing #
    # # # # # # # # # 

    def add_to_route(self, sx, sy):
        wx, wy = self.screen_to_world_float(sx, sy)
        cx = round(wx)
        cy = round(wy)
        self.route.append((cx, cy))

    # # # # # # # #
    # save / load #
    # # # # # # # #

    def open_save_window(self):
        popup = tk.Toplevel(self)
        popup.title("Entrez le nom du chemin à sauvegarder")
        popup.attributes("-topmost", True)
        tk.Label(popup, text="Nom du chemin:").pack(padx=10, pady=5)
        entry = tk.Entry(popup)
        entry.pack(padx=10, pady=5)
        entry.focus_set()

        def validate():
            name = entry.get().strip()
            if not name:
                messagebox.showerror("Erreur", "Entrez un nom pour le chemin à sauvegarder")
                return
            self.try_save_route(name)
            popup.destroy()

        entry.bind("<Return>", lambda e: validate())

        tk.Button(popup, text="Valider", command=validate).pack(pady=5)

    def try_save_route(self, name):
        def save_route(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(";".join(parts))
            messagebox.showinfo(
                "Enregistrer", f"Chemin de {len(self.route)} cases sauvegardé dans {filepath}"
            )
        if not self.route:
            messagebox.showinfo("Enregistrer", "Aucun chemin à enregistrer.")
            return
        parts = [f"({x},{y})" for (x, y) in self.route]
        filepath = os.path.join(ROUTES_FOLDER, name + ".txt")
        if os.path.exists(filepath):
            confirm = tk.Toplevel(self)
            confirm.title("Ce nom de chemin existe déjà")
            confirm.attributes("-topmost", True)
            tk.Label(confirm, text="Un chemin est déjà enregistré sous ce nom. Voulez-vous écraser les données ?").pack()
            tk.Button(confirm, text="Oui", command=lambda: (confirm.destroy(), save_route(filepath))).pack(pady=5)
            tk.Button(confirm, text="Non", command=confirm.destroy).pack(pady=5)
        else:
            save_route(filepath)

    def load_route(self):
        file_path = filedialog.askopenfilename(
            title="Charger un chemin",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            points = []
            if content:
                for part in content.split(";"):
                    part = part.strip(" ()")
                    if not part:
                        continue
                    x_str, y_str = part.split(",")
                    points.append((int(x_str), int(y_str)))
            self.route = points
            self.redraw()
            messagebox.showinfo("Charger", f"Chemin de {len(self.route)} cases chargé depuis {file_path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le fichier : {e}")

def run():
    return RouteMaker()


if __name__ == "__main__":
    app = RouteMaker()
    app.mainloop()

