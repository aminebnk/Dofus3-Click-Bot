import tkinter as tk
from tkinter import font, messagebox, filedialog
import math

GRID_SIZE = 200
HALF = GRID_SIZE // 2
OUTPUT_FILE = "path.txt"

class PathMaker(tk.Toplevel): 
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Path Maker")
        self.geometry("900x600")

        # Top bar
        top = tk.Frame(self)
        top.pack(side="top", fill="x")
        self.save_btn = tk.Button(top, text="Enregistrer", command=self.save_path)
        self.save_btn.pack(side="right", padx=6, pady=6)

        self.load_btn = tk.Button(top, text="Charger", command=self.load_path)
        self.load_btn.pack(side="right", padx=6, pady=6)

        tk.Label(
            top,
            text="Clic gauche = ajouter une case | Glisser gauche = déplacer la vue | Retour arrière = supprimer dernier point"
        ).pack(side="left", padx=6)

        # Canvas
        self.canvas = tk.Canvas(self, bg="white")
        self.canvas.pack(fill="both", expand=True)

        # State
        self.cell_size = 40
        self.offset_x = 0
        self.offset_y = 0
        self.path = []  # chemin = liste de cases (x,y)

        self.font_small = font.Font(size=8)

        # Mouse state
        self._pan_start = None
        self._pan_offset_start = None
        self._click_pos = None
        self._dragged = False

        # Bind events
        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)

        # Bind Backspace
        self.bind("<BackSpace>", self._on_backspace)

        self.redraw()

    # ---------------------------
    # coordinate transforms
    # ---------------------------
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

    # ---------------------------
    # drawing
    # ---------------------------
    def redraw(self):
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_labels()
        self._draw_path()

    def _draw_grid(self):
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

    def _draw_labels(self):
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

    def _draw_path(self):
        if not self.path:
            return

        # dessiner les traits
        for i in range(len(self.path) - 1):
            a = self.path[i]
            b = self.path[i + 1]
            ax, ay = self.world_to_screen(*a)
            bx, by = self.world_to_screen(*b)
            self.canvas.create_line(ax, ay, bx, by, fill="black", width=2)

        # dessiner les points
        for (x, y) in self.path:
            sx, sy = self.world_to_screen(x, y)
            r = 4  # rayon du point
            self.canvas.create_oval(sx - r, sy - r, sx + r, sy + r, fill="black")

    # ---------------------------
    # events
    # ---------------------------
    def _on_configure(self, event):
        self.redraw()

    def _on_left_press(self, event):
        self._click_pos = (event.x, event.y)
        self._pan_start = (event.x, event.y)
        self._pan_offset_start = (self.offset_x, self.offset_y)
        self._dragged = False

    def _on_left_drag(self, event):
        if self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        if abs(dx) > 2 or abs(dy) > 2:
            self._dragged = True
        self.offset_x = self._pan_offset_start[0] + dx
        self.offset_y = self._pan_offset_start[1] + dy
        self.redraw()

    def _on_left_release(self, event):
        if not self._dragged:
            self._add_to_path(event.x, event.y)
            self.redraw()
        self._click_pos = None
        self._pan_start = None
        self._pan_offset_start = None
        self._dragged = False

    def _on_backspace(self, event):
        """Supprime le dernier point ajouté au chemin"""
        if self.path:
            self.path.pop()
            self.redraw()

    # ---------------------------
    # path editing
    # ---------------------------
    def _add_to_path(self, sx, sy):
        wx, wy = self.screen_to_world_float(sx, sy)
        cx = round(wx)
        cy = round(wy)
        self.path.append((cx, cy))

    # ---------------------------
    # save / load
    # ---------------------------
    def save_path(self):
        if not self.path:
            messagebox.showinfo("Enregistrer", "Aucun chemin à enregistrer.")
            return
        parts = [f"({x},{y})" for (x, y) in self.path]
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(";".join(parts))
        messagebox.showinfo(
            "Enregistrer", f"Chemin de {len(self.path)} cases sauvegardé dans {OUTPUT_FILE}"
        )

    def load_path(self):
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
            self.path = points
            self.redraw()
            messagebox.showinfo("Charger", f"Chemin de {len(self.path)} cases chargé depuis {file_path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le fichier : {e}")

def run():
    app = PathMaker()


if __name__ == "__main__":
    app = PathMaker()
    app.mainloop()

