import tkinter as tk
import RouteMaker
import ResourceMapper
import WallMapper 
import bot_script as bot_script

from pynput import keyboard
import os
import multiprocessing as mp


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTES_DIR = os.path.join(BASE_DIR, "resources", "routes")
WALLS_DIR = os.path.join(BASE_DIR, "resources", "walls")
NAMES_DIR = os.path.join(BASE_DIR, "resources", "templates", "names")
RESOURCES_DIR = os.path.join(BASE_DIR, "resources", "templates", "resources")

class ClickBot:
    def __init__(self, root):
        self.root = root
        root.title("Click Bot")

        # App buttons
        top = tk.Frame(root)
        top.pack(side="top", fill="x")

        inner = tk.Frame(top)
        inner.pack(expand=True)

        self.routeMaker_btn = tk.Button(inner, text="Run Route Maker", command=lambda: self.open_sub_app(RouteMaker.run, routes_dropdown, ROUTES_DIR, self.selected_route))
        self.routeMaker_btn.pack(side="left", padx=6, pady=6)

        selects = tk.Frame(root)
        selects.pack(side="top", fill="x")

        self.selected_walls = tk.StringVar(root)
        self.selected_route = tk.StringVar(root)
        self.selected_name = tk.StringVar(root)

        tk.Label(selects, text="Sélectionner les paramètres du script:").pack(side="top")
        
        ## Drop-dowm menu to select the text-file containing the walls
        walls = [f for f in os.listdir(WALLS_DIR) if f.endswith(".txt")]
        if walls:
            self.selected_walls.set(walls[0])
        else:
            self.selected_walls.set(None)
        
        wall_frame = tk.Frame(root)
        wall_frame.pack(side="top", fill="x")
        tk.Label(wall_frame, text="Murs de la map:").pack(side="left", padx=10)
        walls_dropdown = tk.OptionMenu(wall_frame, self.selected_walls, *walls)
        walls_dropdown.pack(side="right", pady=10, padx=30)

        ## Drop-down menu to select the text file containing the pick up path
        routes = [f for f in os.listdir(ROUTES_DIR) if f.endswith(".txt")]
        if routes:
            self.selected_route.set(routes[0])
        else:
            self.selected_route.set(None)
        
        routes_frame = tk.Frame(root)
        routes_frame.pack(side="top", fill="x")
        tk.Label(routes_frame, text="Chemin de récolte à suivre:").pack(side="left", padx=10)
        routes_dropdown = tk.OptionMenu(routes_frame, self.selected_route, *routes)
        routes_dropdown.pack(side="right", padx=30, pady=10)

        ## Drop-down menu to select the text file containing the character's name picture
        names = [f for f in os.listdir(NAMES_DIR) if f.endswith(".png")]
        if names:
            self.selected_name.set(names[0])
        else:
            self.selected_name.set(None)
        names_frame = tk.Frame(root)
        names_frame.pack(side="top", fill="x")
        tk.Label(names_frame, text="Image du nom du personnage:").pack(side="left", padx=10)
        names_dropdown = tk.OptionMenu(names_frame, self.selected_name, *names)
        names_dropdown.pack(side="right", padx=30, pady=10)

        resources = [f for f in os.listdir(RESOURCES_DIR) if f.endswith(".png")]
        self.resources_dict = {}
        resources_frame = tk.Frame(root)
        resources_frame.pack(side="top", fill="x")
        tk.Label(resources_frame, text="Ressources à récolter:").pack(side="left", padx=10)
        checkboxes_frame = tk.Frame(resources_frame)
        checkboxes_frame.pack(side="right", fill="x", padx=10)
        for resource in resources:
            resource, _ = os.path.splitext(resource)
            var = tk.IntVar()
            chk = tk.Checkbutton(checkboxes_frame, text=resource, variable=var)
            chk.pack(anchor="w", padx=10, pady=2)
            self.resources_dict[resource] = var

        tk.Button(root, text="Lancer le script", command=self.listen_script).pack(pady=10)

    def refresh_dropdown(self,dropdown, DIR, selected):
        menu = dropdown["menu"]
        menu.delete(0, "end")
        new_files = [f for f in os.listdir(DIR) if f.endswith(".txt")]
        for file in new_files:
            menu.add_command(
                label=file,
                command=lambda value=file: selected.set(value)
            )

    def open_sub_app(self, App, dropdown, DIR, selected):
        sub_window = App()
        self.root.wait_window(sub_window)
        self.refresh_dropdown(dropdown, DIR, selected)
    
    def listen_script(self):
        route = self.selected_route.get()
        walls = self.selected_walls.get()
        zone, _ = os.path.splitext(walls)
        name = self.selected_name.get()
        name, _ = os.path.splitext(name)
        route = os.path.join(ROUTES_DIR, route)
        walls = os.path.join(WALLS_DIR, walls)
        resource_list = []
        for resource in self.resources_dict:
            selected = self.resources_dict[resource].get()
            if selected == 1:
                resource_list.append(resource)
        p = mp.Process(
            target=bot_script.run_script,
            args=(walls, route, resource_list, name, zone),
            daemon=True
        )
        p.start()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x600")
    app = ClickBot(root)
    root.mainloop()