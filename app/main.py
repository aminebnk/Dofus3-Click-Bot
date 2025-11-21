import tkinter as tk
import RouteMaker
import bot_script as bot_script

import os
import multiprocessing as mp


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTE_DIR = os.path.join(BASE_DIR, "resources", "routes")
ZONE_DIR = os.path.join(BASE_DIR, "resources", "walls")
NAME_DIR = os.path.join(BASE_DIR, "resources", "templates", "names")
RESOURCES_DIR = os.path.join(BASE_DIR, "resources", "templates", "resources")

class ClickBot:
    def __init__(self, root):
        self.root = root
        root.title("Click Bot")

        # Route Maker button: this button lauches a sub-app to define a root and save it as a text file in the dedicated directory for later use
        top = tk.Frame(root)
        top.pack(side="top", fill="x")

        inner = tk.Frame(top)
        inner.pack(expand=True)

        self.routeMaker_btn = tk.Button(inner, text="Run Route Maker", command=lambda: self.open_sub_app(RouteMaker.run, routes_dropdown, ROUTE_DIR, self.selected_route))
        self.routeMaker_btn.pack(side="left", padx=6, pady=6)

        # Drop-down menus to define the parameters of the BOT: the zone, the route, the name of the bot. 
        selects = tk.Frame(root)
        selects.pack(side="top", fill="x")

        self.selected_zone = tk.StringVar(root)
        self.selected_route = tk.StringVar(root)
        self.selected_name = tk.StringVar(root)

        tk.Label(selects, text="Sélectionner les paramètres du script:").pack(side="top")
        
        # Drop-dowm menu to select the text-file containing the walls for the zone
        zones = [f for f in os.listdir(ZONE_DIR) if f.endswith(".txt")]
        if zones:
            self.selected_zone.set(zones[0])
        else:
            self.selected_zone.set(None)
        
        zone_frame = tk.Frame(root)
        zone_frame.pack(side="top", fill="x")
        tk.Label(zone_frame, text="Murs de la map:").pack(side="left", padx=10)
        walls_dropdown = tk.OptionMenu(zone_frame, self.selected_zone, *zones)
        walls_dropdown.pack(side="right", pady=10, padx=30)

        # Drop-down menu to select the text file containing the path
        routes = [f for f in os.listdir(ROUTE_DIR) if f.endswith(".txt")]
        if routes:
            self.selected_route.set(routes[0])
        else:
            self.selected_route.set(None)
        
        routes_frame = tk.Frame(root)
        routes_frame.pack(side="top", fill="x")
        tk.Label(routes_frame, text="Chemin de récolte à suivre:").pack(side="left", padx=10)
        routes_dropdown = tk.OptionMenu(routes_frame, self.selected_route, *routes)
        routes_dropdown.pack(side="right", padx=30, pady=10)

        # Drop-down menu to select the character's name. The directory \name must include a picture of the character's name tag under the filename "character_name.png". Press 'p' in game to activate the name tags.
        names = [f for f in os.listdir(NAME_DIR) if f.endswith(".png")]
        if names:
            self.selected_name.set(names[0])
        else:
            self.selected_name.set(None)
        names_frame = tk.Frame(root)
        names_frame.pack(side="top", fill="x")
        tk.Label(names_frame, text="Image du nom du personnage:").pack(side="left", padx=10)
        names_dropdown = tk.OptionMenu(names_frame, self.selected_name, *names)
        names_dropdown.pack(side="right", padx=30, pady=10)

        # Radio buttons to select the resources you want the bot to pick up along its path
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

    def refresh_dropdown(self, dropdown, DIR, selected):
        """
        This function is called when a route is added via RouteMaker to update the scroll-down menu.
        """
        menu = dropdown["menu"]
        menu.delete(0, "end")
        new_files = [f for f in os.listdir(DIR) if f.endswith(".txt")]
        for file in new_files:
            menu.add_command(
                label=file,
                command=lambda value=file: selected.set(value)
            )

    def open_sub_app(self, App, dropdown, DIR, selected):
        """
        This function is called to open the RouteMaker sub-app
        """
        sub_window = App()
        self.root.wait_window(sub_window)
        self.refresh_dropdown(dropdown, DIR, selected)
    
    def listen_script(self):
        """
        This function calls the run_script function with the arguments selected inside the app.
        See the afore mentioned function in the bot_script.py file to get more insight.
        Because the run_script function listens to the keyboard input, it needs to be called in a 
        separate process.
        """
        route = self.selected_route.get()
        zone = self.selected_zone.get()
        name = self.selected_name.get()
        resource_list = []
        for resource in self.resources_dict:
            selected = self.resources_dict[resource].get()
            if selected == 1:
                resource_list.append(resource)
        p = mp.Process(
            target=bot_script.run_script,
            args=(route, resource_list, name, zone),
            daemon=True
        )
        p.start()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x600")
    app = ClickBot(root)
    root.mainloop()