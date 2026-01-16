import sqlite3
import tkinter as tk
from tkinter import messagebox
import tkinter.font as tkFont
from pynput import mouse
from pynput.mouse import Button
import os
import json
from bot_script import get_map, resource_path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCE_PATH = os.path.join(BASE_DIR, "resources", "resources.db")
CONFIG_PATH = os.path.join(BASE_DIR, "resources", "config", "config.json")

## Database to store the resource positions
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

def save_resources(map_pos, resources, zone="Amakna", overwrite=False):
    """
    self explanatory: add a resource to the database.
    resources is a dictionnary whose keys are the resources types and values are the list of positions 
    on the screen corresponding to that type. 
    If of the resource types are already in the database for that map, we return those conflicting resources. 
    If the function is called with overwrite=True, we delete the old data and replace it with the new.
    """
    conn = sqlite3.connect(os.path.join(BASE_DIR, "resources", "resources.db"))
    c = conn.cursor()
    c.execute("SELECT DISTINCT type FROM resources WHERE map_x = ? AND map_y = ? AND zone = ?", (map_pos[0], map_pos[1], zone))
    # Get the resources already registered for that map, if any
    existing_resources = [row[0] for row in c.fetchall()]
    conflicting_resources = []
    for type, pos in resources.items(): # Check wether the positions we want to add conflict with the database
        if pos and type in existing_resources:
            conflicting_resources.append(type) 

    if conflicting_resources: # If they do,
        if not overwrite:
            return conflicting_resources # Return the liste of conflicting resources to warn the user
        else:
            for resource in conflicting_resources: # Overwrite only after getting user approval 
                c.execute("DELETE FROM resources WHERE type = ? AND map_x = ? AND map_y = ? AND zone = ?", (resource, map_pos[0], map_pos[1], zone))
    # If no conflicts, simply save the data
    for key, value in resources.items():      
        for pos in value:
            c.execute("INSERT INTO resources (map_x, map_y, type, pos_x, pos_y, zone) VALUES (?, ?, ?, ?, ?, ?)",
                    (map_pos[0], map_pos[1], key, pos[0], pos[1], zone))
    conn.commit()
    conn.close()
    return None

class ResourceMapper(tk.Tk):
    """
    This is the sub app used to save the resource positions in the database. 
    To do so, we need: 
    -the type of resource to add, for example 'Châtaigner'
    -the map position, for example [-17, 5]
    -the game zone the resource is in. Different game zones make coordinates degenerate so this is important. 
    Defaults to 'Amakna', which is the main continent of the game.
    -the list of positions to add to the database, where the bot actually needs to click.
    
    Once the user has added a resource to the app, he can activate it to then add any position he right clicks on. 
    He then confirms and all those positions are added to the database.

    For the map coordinates, there are two modes:
    -Automatic, the coordinates are retrieved by using the get_map function. The default.
    -Manual, the user inputs it by hand
    """
    def __init__(self):
        super().__init__()
        self.attributes("-topmost", True)
        self.title("Resourde Mapper")

        # Internal states used by the app
        self.active_resource = None # The resource we are currently clicking on
        self.resources = {} # The types of resources to add
        self.resource_buttons = {}
        self.resource_labels = {}
        self.clear_buttons = {}
        self.delete_buttons = {}
        self.selected_mode = tk.StringVar(value="automatique") 
        self.window_focused = tk.IntVar(value=1)

        self.options = ["automatique", "manuel"] # The two modes to get the map position

        # === Top frame: input the coordinates of the map ===
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

        # === Middle frame : dynamic resource buttons ===
        self.resource_frame = tk.Frame(self)
        self.resource_frame.pack(anchor="w", pady=5)
        # We hereby only define the add button. The resource buttons are added by the user using this add button. They are displayed row by row.
        self.add_btn = tk.Button(self.resource_frame, text="+", command=self.open_add_resource_popup) 
        self.add_btn.grid(row=0, column=0, pady=5)

        # === Bottom frame: input the zone and add the map ===
        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(side="top", pady=10)

        tk.Label(self.bottom_frame, text="Zone:").pack(side="left")
        self.zone_entry = tk.Entry(self.bottom_frame, width=15)
        self.zone_entry.pack(side="left")
        self.save_btn = tk.Button(self.bottom_frame, text="Save Map", command=self.save_map)
        self.save_btn.pack(side="left")
        
        self.calibrate_btn = tk.Button(self.bottom_frame, text="Calibrate Map", command=self.calibrate_map_zone)
        self.calibrate_btn.pack(side="left", padx=10)

        # Mouse listener. Clicks are handled by the on_click function
        self.listener = mouse.Listener(on_click=self.on_click)
        self.listener.start()

        # This part is a bit cryptic. We want the entry zone_entry to lose the focus when we click anywhere else, which it doesn't by default.
        def clear_focus(event):
            if event.widget != self.zone_entry:
                self.focus_set()

        def get_focus(event):
            return
        # When the window loses focus, we initialise an internal variable. This ensures we ignore the first click outside the window.
        def lose_focus(event):
            self.window_focused = 1

        self.bind("<Button-1>", clear_focus)
        self.bind("<FocusIn>", get_focus)
        self.bind("<FocusOut>", lose_focus)

    def open_add_resource_popup(self):
        # Create the popup
        popup = tk.Toplevel(self)
        popup.title("Ajouter un type de ressource")
        popup.attributes("-topmost", True)

        # Add an entry for the resource type name
        tk.Label(popup, text="Ressource à ajouter:").pack(padx=10, pady=5)
        entry = tk.Entry(popup)
        entry.pack(padx=10, pady=5)
        entry.focus_set()

        def validate(): # If the resource is not already added as a button, we add it and destroy the popup
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
        row = len(self.resource_buttons) + 1 # Row at which we add the button. Equal to the number of resource buttons that already exist, plus one.
        small_font = tkFont.Font(size=8)

        # We add three buttons: one to activate/deactivate the resource. One to clear the positions. One to delete the resource button. One button to rule them all and into darknes bind them.
        btn = tk.Button(self.resource_frame, text=name + " (OFF)", command=lambda n=name: self.toggle_resource(n))
        btn.grid(row=row, column=0, sticky="w", pady=2)
        clear_btn = tk.Button(self.resource_frame, text="clear", command=lambda n=name: self.clear(n))
        clear_btn.grid(row=row, column=1, sticky="w", pady=2)
        delete_btn = tk.Button(self.resource_frame, text="x", command=lambda n=name: self.delete(n))
        delete_btn.grid(row=row, column=2, sticky="w", pady=2)


        # We add that resource to the dictionary, together with its buttons.
        self.resources[name] = []
        self.resource_buttons[name] = btn
        self.clear_buttons[name] = clear_btn
        self.delete_buttons[name] = delete_btn

        # We add a label next to the buttons. It will be updated with the positions we click when the resource is active.
        lbl = tk.Label(self.resource_frame, text="", anchor="w", justify="left", font=small_font, wraplength=200)
        lbl.grid(row=row, column=3, sticky="w", pady=2)
        self.resource_labels[name] = lbl

        # We added a row. So the add button needs to go down a row to always be at the bottom of the resource buttons.
        self.add_btn.grid(row=row+1, column=0, pady = 5)

    def toggle_resource(self, name):
        """
        This function is called to activate/deactivate a resource button
        If a another button was already activated, it is deactivated. Only one resource can be active at a time.
        """
        # If a resource is already active, deactivate it
        if self.active_resource:
            self.resource_buttons[self.active_resource].config(text=self.active_resource + " (OFF)") # Turn off its button
            if name == self.active_resource: # if the button clicked was the active resource, then we are left with no active resource
                self.active_resource = None
            else: # Else we activate the button that was pushed
                self.active_resource = name
                self.resource_buttons[name].config(text=name + " (ON)")
        else: # If no resource was already active, we simply activate the button that was pushed
            self.active_resource = name
            self.resource_buttons[name].config(text=name + " (ON)")

    # Clear the positions associated with a resource
    def clear(self, name):
        self.resources[name] = []
        self.resource_labels[name].config(text="")

    # Delete everything from a ressource. Called when we press the delete button
    def delete(self, name):
        del self.resources[name]
        self.resource_buttons[name].destroy()
        self.clear_buttons[name].destroy()
        self.delete_buttons[name].destroy()
        self.resource_labels[name].destroy()
        if self.active_resource == name:
            self.active_resource = None


    def on_click(self, x, y, button, pressed):
        if pressed and button == Button.left:
            if self.active_resource:
                win_x = self.winfo_x()
                win_y = self.winfo_y()
                win_w = self.winfo_width()
                win_h = self.winfo_height()

                TITLEBAR_HEIGHT = 25
                # The clicks inside the window are ignored for the purpose of adding resource positions
                if win_x <= x <= win_x + win_w + 1 and win_y - TITLEBAR_HEIGHT<= y <= win_y + win_h + TITLEBAR_HEIGHT + 1:
                    return
                # The first click outside the window of the resource mapper is ignored (recall that self.window_focused is updated to 1 when the app loses focus)
                if self.window_focused < 1:
                    self.resources[self.active_resource].append((int(x), int(y))) # We add the position to the list inside the dictionary
                    resource_label = " ".join(f"({rx},{ry})" for rx, ry in self.resources[self.active_resource]) # We update the label next to the resource button
                    self.resource_labels[self.active_resource].config(text=resource_label)
                else:
                    self.window_focused -= 1

    def save_map(self):
        """
        Save the map. Updates the database with the resource positions.
        If a conflict is detected, asks the user for confirmation.
        """
        if self.active_resource:
            self.resource_buttons[self.active_resource].config(text=self.active_resource+" (OFF)")
            self.active_resource = None
        mode = self.selected_mode.get()
        if mode == "manuel":
            map_x = int(self.x_spin.get().strip())
            map_y = int(self.y_spin.get().strip())
        else:
            try:
                result = get_map()
                if result is None:
                    messagebox.showerror("Error", "Impossible de récupérer la position de la map")
                    return
                map_x, map_y = result
                self.x_spin.delete(0, "end")
                self.x_spin.insert(0, str(map_x))
                self.y_spin.delete(0, "end")
                self.y_spin.insert(0, str(map_y))
            except ValueError:
                print("Impossible de récupérer la position de la map")
        
        zone = self.zone_entry.get()
        if not zone:
            zone = "Amakna"
        if map_x is None or map_y is None:
            messagebox.showerror("Error", "Les coordonnées de la carte ne peuvent être vides")
            return
        map_pos = [map_x, map_y]
        conflicting_resources = save_resources(map_pos, self.resources, zone)
        if conflicting_resources:
            # pop up creation
            popup = tk.Toplevel(self)
            popup.title("Ressources déjà enregistrées")
            popup.attributes("-topmost", True)

            # Ask the user wether he wants to overwrite the existing data
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
    
    def calibrate_map_zone(self):
        messagebox.showinfo("Calibration", "Select the area containing the MAP COORDINATES.")
        from screenselector import select_region
        select_region(self, self.on_map_selected)

    def on_map_selected(self, x, y, w, h):
        config = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                try:
                    config = json.load(f)
                except json.JSONDecodeError:
                    pass
        
        config["map_coordinates"] = {
            "x": x,
            "y": y,
            "width": w,
            "height": h
        }
        
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=4)
            
        messagebox.showinfo("Calibration", "Map coordinates updated! Restart the bot for changes to take effect.")


def run():
    init_db()
    app = ResourceMapper()

if __name__ == "__main__":
    init_db()
    app = ResourceMapper()
    app.mainloop()
