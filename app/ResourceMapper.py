import sqlite3
import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkFont
from pynput import mouse
from pynput.mouse import Button
import os
import cv2
from bot_script import screenshot_high_res, get_map
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_PATH = os.path.join(BASE_DIR, "resources", "resources.db")

def init_db():
    conn = sqlite3.connect(RESOURCE_PATH)
    c = conn.cursor()
    c.execute("""
            CREATE TABLE IF NOT EXISTS resources (
                map_x INTEGER NOT NULL,
                map_y INTEGER NOT NULL,
                type TEXT NOT NULL,
                pos_x INTEGER NOT NULL,
                pos_y INTEGER NOT NULL,
                zone TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_resources(map_pos, resources, zone="Douze", overwrite=False):
    conn = sqlite3.connect(os.path.join(BASE_DIR, "resources", "resources.db"))
    c = conn.cursor()
    c.execute("SELECT DISTINCT type FROM resources WHERE map_x = ? AND map_y = ? AND zone = ?", (map_pos[0], map_pos[1], zone))
    # On identifie les types de ressources déjà enregistrés pour la map
    existing_resources = [row[0] for row in c.fetchall()]
    conflicting_resources = []
    for type, pos in resources.items(): # resource = dictionnaire clé = type et value = listes des positions
        if pos and type in existing_resources: # si la liste de positions est vide -> pas de conflit
            conflicting_resources.append(type) # On liste les types déjà présent que l'utilisateur veut ajouter

    if conflicting_resources:
        if not overwrite:
            return conflicting_resources # La fonction est appelée une première fois sans overwrite: on renvoie la liste des types de ressources conflictuels
        else:
            for resource in conflicting_resources: # La fonction est rappelée avec overwrite: on supprime les types conflictuels 
                c.execute("DELETE FROM resources WHERE type = ? AND map_x = ? AND map_y = ? AND zone = ?", (resource, map_pos[0], map_pos[1], zone))
    # Pas de conflit ou overwrite activé: on met à jour la base de donnée
    for key, value in resources.items():      
        for pos in value:
            c.execute("INSERT INTO resources (map_x, map_y, type, pos_x, pos_y, zone) VALUES (?, ?, ?, ?, ?, ?)",
                    (map_pos[0], map_pos[1], key, pos[0], pos[1], zone))
    conn.commit()
    conn.close()
    return None

class ResourceMapper(tk.Tk):
    def __init__(self):
        super().__init__()
        self.attributes("-topmost", True)
        self.title("Resourde Mapper")
        # état
        self.active_resource = None
        self.resources = {}
        self.resource_buttons = {}
        self.resource_labels = {}
        self.clear_buttons = {}
        self.delete_buttons = {}
        self.selected_mode = tk.StringVar(value="automatique")
        self.window_focused = tk.IntVar(value=1)

        self.options = ["automatique", "manuel"]

        # === FRAME 1 : coordonnées ===
        self.top_frame = tk.Frame(self)
        self.top_frame.pack(side="top", fill="x", pady=5)

        tk.Label(self.top_frame, text="Coordonnées:").grid(row=0, column=0, sticky="w")
        count = 1
        for opt in self.options:
            rb = tk.Radiobutton(self.top_frame, text=opt, variable=self.selected_mode, value=opt)
            rb.grid(row=0, column=count, sticky="w")
            count += 1

        tk.Label(self.top_frame, text="X:").grid(row=0, column=3, sticky="w")
        self.x_spin = tk.Spinbox(self.top_frame, from_=-100, to=100, width=5)
        self.x_spin.grid(row=0, column=4, sticky="w")
        self.x_spin.delete(0, "end")
        self.x_spin.insert(0, "0")

        tk.Label(self.top_frame, text="Y:").grid(row=0, column=5, sticky="w")
        self.y_spin = tk.Spinbox(self.top_frame, from_=-100, to=100, width=5)
        self.y_spin.grid(row=0, column=6, sticky="w")
        self.y_spin.delete(0, "end")
        self.y_spin.insert(0, "0")

        # === FRAME 2 : ressources dynamiques ===
        self.resource_frame = tk.Frame(self)
        self.resource_frame.pack(anchor="w", pady=5)

        self.add_btn = tk.Button(self.resource_frame, text="+", command=self.open_add_resource_popup)
        self.add_btn.grid(row=0, column=0, pady=5)

        # === FRAME 3 : boutons bas ===
        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(side="top", pady=10)

        tk.Label(self.bottom_frame, text="Zone:").pack(side="left")
        self.zone_entry = tk.Entry(self.bottom_frame, width=15)
        self.zone_entry.pack(side="left")
        self.save_btn = tk.Button(self.bottom_frame, text="Save Map", command=self.save_map)
        self.save_btn.pack(side="left")

        # listener souris
        self.listener = mouse.Listener(on_click=self.on_click)
        self.listener.start()

        def clear_focus(event):
            if event.widget != self.zone_entry:
                self.focus_set()

        def get_focus(event):
            return
        def lose_focus(event):
            self.window_focused = 1

        self.bind("<Button-1>", clear_focus)

        self.bind("<FocusIn>", get_focus)
        self.bind("<FocusOut>", lose_focus)

    def open_add_resource_popup(self):
        # création du pop-up
        popup = tk.Toplevel(self)
        popup.title("Ajouter un type de ressource")
        popup.attributes("-topmost", True)

        # entrée de la ressource
        tk.Label(popup, text="Ressource à ajouter:").pack(padx=10, pady=5)
        entry = tk.Entry(popup)
        entry.pack(padx=10, pady=5)
        entry.focus_set()

        def validate():
            name = entry.get().strip()
            if not name:
                messagebox.showerror("Erreur", "Entrez un nom pour la ressource")
                return
            if name in self.resources:
                messagebox.showerror("Erreur", "Cette ressource existe déjà")
                return
            self.add_resource_button(name)
            popup.destroy()

        entry.bind("<Return>", lambda e: validate())

        tk.Button(popup, text="Valider", command=validate).pack(pady=5)

    def add_resource_button(self, name):
        row = len(self.resource_buttons) + 1 # ligne à laquelle on ajoute le bouton 
        small_font = tkFont.Font(size=8)

        # On ajoute le bouton de la resource 
        btn = tk.Button(self.resource_frame, text=name + " (OFF)", command=lambda n=name: self.toggle_resource(n))
        btn.grid(row=row, column=0, sticky="w", pady=2)
        clear_btn = tk.Button(self.resource_frame, text="clear", command=lambda n=name: self.clear(n))
        clear_btn.grid(row=row, column=1, sticky="w", pady=2)
        delete_btn = tk.Button(self.resource_frame, text="x", command=lambda n=name: self.delete(n))
        delete_btn.grid(row=row, column=2, sticky="w", pady=2)


        # On met à jour les dictionnaires de l'app
        self.resources[name] = []
        self.resource_buttons[name] = btn
        self.clear_buttons[name] = clear_btn
        self.delete_buttons[name] = delete_btn

        # On rajoute un Label à côté du bouton
        lbl = tk.Label(self.resource_frame, text="", anchor="w", justify="left", font=small_font, wraplength=200)
        lbl.grid(row=row, column=3, sticky="w", pady=2)
        self.resource_labels[name] = lbl

        # On descend le bouton "+"
        self.add_btn.grid(row=row+1, column=0, pady = 5)

    def toggle_resource(self, name):
        # On éteint le bouton actif
        if self.active_resource:
            self.resource_buttons[self.active_resource].config(text=self.active_resource + " (OFF)")
            # Si la ressource était déjà activée, on la désactive
            if name == self.active_resource:
                self.active_resource = None
            # Sinon on active la ressource appelée
            else:
                self.active_resource = name
                self.resource_buttons[name].config(text=name + " (ON)")
        # Si aucun bouton n'était actif
        else:
            self.active_resource = name
            self.resource_buttons[name].config(text=name + " (ON)")

    # Efface les positions associées à un type de ressources
    def clear(self, name):
        self.resources[name] = []
        self.resource_labels[name].config(text="")

    # Supprime le type de ressource
    def delete(self, name):
        del self.resources[name]
        self.resource_buttons[name].destroy()
        self.clear_buttons[name].destroy()
        self.delete_buttons[name].destroy()
        self.resource_labels[name].destroy()
        if self.active_resource == name:
            self.active_resource = None


    def on_click(self, x, y, button, pressed):
        if pressed and button == Button.left and self.active_resource:
            win_x = self.winfo_x()
            win_y = self.winfo_y()
            win_w = self.winfo_width()
            win_h = self.winfo_height()

            TITLEBAR_HEIGHT = 25
            # les clicks dans la fenêtre sont ignorés
            if win_x <= x <= win_x + win_w + 1 and win_y - TITLEBAR_HEIGHT<= y <= win_y + win_h + TITLEBAR_HEIGHT + 1:
                return
            if self.window_focused < 1:
                # On ajoute la ressource
                self.resources[self.active_resource].append((int(x), int(y)))
                # On recalcule le label
                resource_label = " ".join(f"({rx},{ry})" for rx, ry in self.resources[self.active_resource])
                self.resource_labels[self.active_resource].config(text=resource_label)
            else:
                self.window_focused -= 1

    def save_map(self):
        if self.active_resource:
            self.resource_buttons[self.active_resource].config(text=self.active_resource+" (OFF)")
            self.active_resource = None
        mode = self.selected_mode.get()
        if mode == "manuel":
            map_x = int(self.x_spin.get().strip())
            map_y = int(self.y_spin.get().strip())
        else:
            try:
                map_x, map_y = get_map()
                self.x_spin.delete(0, "end")
                self.x_spin.insert(0, str(map_x))
                self.y_spin.delete(0, "end")
                self.y_spin.insert(0, str(map_y))
            except ValueError:
                print("Impossible de récupérer la position de la map")
        
        zone = self.zone_entry.get()
        if not zone:
            zone = "Douze"
        if map_x is None or map_y is None:
            messagebox.showerror("Error", "Les coordonnées de la carte ne peuvent être vides")
            return
        map_pos = [map_x, map_y]
        conflicting_resources = save_resources(map_pos, self.resources, zone)
        if conflicting_resources:
            # création du pop-up
            popup = tk.Toplevel(self)
            popup.title("Ressources déjà enregistrées")
            popup.attributes("-topmost", True)

            # entrée de la ressource
            label = "Les ressources suivantes ont déjà été enregistrées pour cette carte:\n"
            for resource in conflicting_resources:
                label += resource + "\n"
            label += "Voulez-vous écraser les données ?"
            tk.Label(popup, text=label, wraplength=300, anchor="w", justify="center").pack(padx=10, pady=5)
            tk.Button(popup, text="Oui", command=lambda: (popup.destroy(), save_resources(map_pos, self.resources, zone, overwrite=True), self.cleanup())).pack(pady=5)
            tk.Button(popup, text="Non", command=popup.destroy).pack(pady=5)
        else: 
            self.cleanup()

    def cleanup(self):
        for key in self.resources: 
            self.resources[key] = []
        for key in self.resource_labels:
            self.resource_labels[key].config(text="")
    

def run():
    init_db()
    app = ResourceMapper()

if __name__ == "__main__":
    init_db()
    app = ResourceMapper()
    app.mainloop()
