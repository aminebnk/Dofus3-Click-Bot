import tkinter as tk
from tkinter import font, messagebox, filedialog
import math
import os

GRID_SIZE = 80
HALF = GRID_SIZE // 2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WALLS_FOLDER = os.path.join(BASE_DIR, "resources", "walls")

class WallMapper(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Wall Mapper")
        self.geometry("900x600")

        # Top bar
        top = tk.Frame(self)
        top.pack(side="top", fill="x")
        self.save_btn = tk.Button(top, text="Enregistrer", command=self.open_save_window)
        self.save_btn.pack(side="right", padx=6, pady=6)

        self.load_btn = tk.Button(top, text="Charger", command=self.load_walls)
        self.load_btn.pack(side="right", padx=6, pady=6)

        tk.Label(top,text="Clic gauche = activer/désactiver une arête | Maintenu = faire glisser la grille").pack(side="left", padx=6)

        # Canvas
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(fill="both", expand=True)

        # States
        self.cell_size = 80
        self.offset_x = 0
        self.offset_y = 0
        self.walls = set()
        self.edge_items = {}
        self.file_name = ""

        self.font_small = font.Font(size=12)

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

        self.redraw()

    # coordinate transforms

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


    # drawing

    def redraw(self):
        self.canvas.delete("all")
        self.draw_grid()
        self.draw_labels()
        self.draw_walls()

    def draw_grid(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        left_wx, top_wy = self.screen_to_world_float(0, 0)
        right_wx, bottom_wy = self.screen_to_world_float(w, h)

        min_x = max(math.floor(left_wx - 0.5), -HALF)
        max_x = min(math.ceil(right_wx + 0.5), HALF)
        min_y = max(math.floor(top_wy - 0.5), -HALF)
        max_y = min(math.ceil(bottom_wy + 0.5), HALF)

        # Vertical lines
        for gx in range(min_x, max_x + 1):
            sx1, sy1 = self.world_to_screen(gx + 0.5, min_y)
            sx2, sy2 = self.world_to_screen(gx + 0.5, max_y)
            self.canvas.create_line(sx1, sy1, sx2, sy2, fill="#ccc")

        # Horizontal lines
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

    def draw_walls(self):
        for (a, b) in self.walls:
            ax, ay = a
            bx, by = b
            if abs(ax - bx) + abs(ay - by) != 1:
                continue
            if ax == bx:  # horizontal wall
                wx1, wy1 = self.world_to_screen(ax - 0.5, (ay + by) / 2)
                wx2, wy2 = self.world_to_screen(ax + 0.5, (ay + by) / 2)
            else:  # vertical wall
                wx1, wy1 = self.world_to_screen((ax + bx) / 2, ay - 0.5)
                wx2, wy2 = self.world_to_screen((ax + bx) / 2, ay + 0.5)
            self.canvas.create_line(wx1, wy1, wx2, wy2, fill="red", width=3)


    # events

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
            self.toggle_nearest_edge(event.x, event.y)
            self.redraw()
        self.click_pos = None
        self.pan_start = None
        self.pan_offset_start = None
        self.dragged = False


    # toggle edges

    def normalize_edge(self, a, b):
        return (a, b) if a <= b else (b, a)

    def toggle_nearest_edge(self, sx, sy):
        wxf, wyf = self.screen_to_world_float(sx, sy)
        cx = round(wxf)
        cy = round(wyf)

        candidates = [
            ((cx, cy), (cx + 1, cy)),
            ((cx - 1, cy), (cx, cy)),
            ((cx, cy), (cx, cy + 1)),
            ((cx, cy - 1), (cx, cy)),
        ]

        best = None
        best_dist = float("inf")
        for a, b in candidates:
            mx = (a[0] + b[0]) / 2
            my = (a[1] + b[1]) / 2
            d = math.hypot(mx - wxf, my - wyf)
            if d < best_dist:
                best_dist = d
                best = (a, b)

        if best and best_dist < 0.45:
            edge = self.normalize_edge(*best)
            if edge in self.walls:
                self.walls.remove(edge)
            else:
                self.walls.add(edge)

    # save / load

    def open_save_window(self):
        popup = tk.Toplevel(self)
        popup.title("Entrez le nom de la zone concernée")
        popup.attributes("-topmost", True)
        tk.Label(popup, text="Nom de la zone concernée:").pack(padx=10, pady=5)
        entry = tk.Entry(popup)
        entry.pack(padx=10, pady=5)
        entry.focus_set()

        def validate():
            name = entry.get().strip()
            if not name:
                messagebox.showerror("Erreur", "Entrez un nom pour la zone concernée")
                return
            self.try_save_walls(name)
            popup.destroy()

        entry.bind("<Return>", lambda e: validate())

        tk.Button(popup, text="Valider", command=validate).pack(pady=5)

    def try_save_walls(self, name):
        def save_walls(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(";".join(parts))
            messagebox.showinfo(
                "Enregistrer", f"{len(self.walls)} arêtes sauvegardées dans pour la zone {name}"
            )
        if not self.walls:
            messagebox.showinfo("Enregistrer", "Aucun mur à enregistrer.")
            return
        parts = [
            f"(({a[0]},{a[1]}),({b[0]},{b[1]}));(({b[0]},{b[1]}),({a[0]},{a[1]}))"
            for a, b in sorted(self.walls)
        ]
        filepath = os.path.join(WALLS_FOLDER, name + ".txt")
        if os.path.exists(filepath):
            confirm = tk.Toplevel(self)
            confirm.title("Zone déjà enregistrée")
            confirm.attributes("-topmost", True)
            tk.Label(confirm, text="Une zone est déjà enregistrée sous ce nom. Voulez-vous écraser les données ?").pack()
            tk.Button(confirm, text="Oui", command=lambda: (save_walls(filepath), confirm.destroy())).pack(pady=5)
            tk.Button(confirm, text="Non", command=confirm.destroy).pack(pady=5)
        else:
            save_walls(filepath)

    def load_walls(self):
        file_path = filedialog.askopenfilename(
            title="Charger un fichier d'arêtes",
            filetypes=[("Fichiers texte", "*.txt"), ("Tous les fichiers", "*.*")]
        )
        if not file_path:
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            edges = set()
            if content:
                for part in content.split(";"):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        a_str, b_str = part.split("),(")
                        a_str = a_str.replace("(", "").replace(")", "")
                        b_str = b_str.replace("(", "").replace(")", "")
                        ax, ay = map(int, a_str.split(","))
                        bx, by = map(int, b_str.split(","))
                        edges.add(self.normalize_edge((ax, ay), (bx, by)))
                    except Exception:
                        continue
            self.walls = edges
            self.redraw()
            messagebox.showinfo("Charger", f"{len(self.walls)} arêtes chargées depuis {file_path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le fichier : {e}")

def run():
    app = WallMapper()

if __name__ == "__main__":
    app = WallMapper()
    app.mainloop()