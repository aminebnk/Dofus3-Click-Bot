import tkinter as tk
from tkinter import messagebox
import routemaker
import bot_script as bot_script
from cbt import resource_path
import os
import multiprocessing as mp
import screenselector
import json

""" This file is the main app layout. """

class ClickBot:
    def __init__(self, root):
        self.root = root
        root.title("Click Bot")

        # Route Maker button: this button lauches a sub-app to define a root and save it as a text file in the dedicated directory for later use
        top = tk.Frame(root)
        top.pack(side="top", fill="x")

        inner = tk.Frame(top)
        inner.pack(expand=True)

        self.routeMaker_btn = tk.Button(inner, text="Créer un nouveau chemin de récolte", command=lambda: self.open_sub_app(routemaker.run, routes_dropdown, resource_path("routes"), self.selected_route))
        self.routeMaker_btn.pack(side="left", padx=6, pady=6)

        self.nametag_screenshot_btn = tk.Button(inner, text="Enregistrer un nouveau personnage", command=lambda: self.open_sub_app(screenselector.run, names_dropdown, resource_path("templates/names"), self.selected_name, ".png", self.root))
        self.nametag_screenshot_btn.pack(side="right", padx=6, pady=6)

        # Drop-down menus to define the parameters of the BOT: the zone, the route, the name of the bot. 
        selects = tk.Frame(root)
        selects.pack(side="top", fill="x")

        self.selected_zone = tk.StringVar(root)
        self.selected_route = tk.StringVar(root)
        self.selected_name = tk.StringVar(root)

        tk.Label(selects, text="Sélectionner les paramètres du script:").pack(side="top")
        
        self.calibrate_btn = tk.Button(selects, text="CALIBRER LES ZONES", bg="yellow", command=self.calibrate_interface)
        self.calibrate_btn.pack(side="top", pady=5)
        
        # Drop-dowm menu to select the text-file containing the walls for the zone
        zones = [f for f in os.listdir(resource_path("walls")) if f.endswith(".txt")]
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
        routes = [f for f in os.listdir(resource_path("routes")) if f.endswith(".txt")]
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
        names = [f for f in os.listdir(resource_path("templates/names")) if f.endswith(".png")]
        if not names:
            names = ["Aucun personnage enregistré"]
        self.selected_name.set(names[0])
        names_frame = tk.Frame(root)
        names_frame.pack(side="top", fill="x")
        tk.Label(names_frame, text="Nametag du personnage:").pack(side="left", padx=10)
        names_dropdown = tk.OptionMenu(names_frame, self.selected_name, *names)
        names_dropdown.pack(side="right", padx=30, pady=10)

        # Radio buttons to select the resources you want the bot to pick up along its path
        resources = [f for f in os.listdir(resource_path("templates/resources")) if f.endswith(".png")]
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

        launch_frame = tk.Frame(root)
        launch_frame.pack(side="top", fill="x")
        tk.Button(launch_frame, text="Lancer le script", command=self.listen_script).pack(pady=10)
        self.launch_text = tk.Label(launch_frame, text="")
        self.launch_text.pack(side="bottom", padx=10)

    def calibrate_interface(self):
        messagebox.showinfo("Calibration", "Please select the area containing the MAP COORDINATES (e.g. '12, -5').")
        screenselector.select_region(self.root, self.calib_map_coords)

    def calib_map_coords(self, x, y, w, h):
        self.new_config = {}
        self.new_config["map_coordinates"] = {"x": x, "y": y, "width": w, "height": h}
        
        messagebox.showinfo("Calibration", "Please select the FIGHT 'READY' / 'END TURN' BUTTON.")
        screenselector.select_region(self.root, self.calib_fight_btn)

    def calib_fight_btn(self, x, y, w, h):
        if "fight_coordinates" not in self.new_config:
            self.new_config["fight_coordinates"] = {}
        self.new_config["fight_coordinates"]["CORNER_FIGHT_BTN"] = [x, y]
        self.new_config["fight_coordinates"]["SIZE_FIGHT_BTN"] = [w, h]
        self.new_config["fight_coordinates"]["END_TURN_POS"] = [x + w//2, y + h//2]

        messagebox.showinfo("Calibration", "Please select the SPELL ICON you want to use.")
        screenselector.select_region(self.root, self.calib_spell)

    def calib_spell(self, x, y, w, h):
        self.new_config["fight_coordinates"]["SPELL_POS"] = [x + w//2, y + h//2]

        messagebox.showinfo("Calibration", "Please select the ENTIRE FIGHT SCENE (the area where characters move).")
        screenselector.select_region(self.root, self.calib_scene)

    def calib_scene(self, x, y, w, h):
        self.new_config["fight_coordinates"]["FIGHT_SCN_TOP"] = y
        self.new_config["fight_coordinates"]["FIGHT_SCN_LEFT"] = x
        self.new_config["fight_coordinates"]["FIGHT_SCN_WIDTH"] = w
        self.new_config["fight_coordinates"]["FIGHT_SCN_HEIGHT"] = h
        
        self.save_config()
        messagebox.showinfo("Calibration", "Calibration Complete! Configuration saved.")

    def save_config(self):
        try:
            with open(resource_path("config/config.json"), "r") as f:
                old_config = json.load(f)
        except:
            old_config = {}
        
        old_config.update(self.new_config)
        
        with open(resource_path("config/config.json"), "w") as f:
            json.dump(old_config, f, indent=4)

    def start_selection(self):
        screenselector.ScreenSelector(self.root)

    def refresh_dropdown(self, dropdown, DIR, selected, ext):
        """
        This function is called when a route is added via RouteMaker to update the scroll-down menu.
        """
        menu = dropdown["menu"]
        menu.delete(0, "end")
        new_files = [f for f in os.listdir(DIR) if f.endswith(ext)]
        for file in new_files:
            menu.add_command(
                label=file,
                command=lambda value=file: selected.set(value)
            )

    def open_sub_app(self, App, dropdown, DIR, selected, ext=".txt", App_arg = None):
        """
        This function is called to open the RouteMaker sub-app
        """
        if App_arg:
            sub_window = App(App_arg)
        else:
            sub_window = App()
        self.root.wait_window(sub_window)
        self.refresh_dropdown(dropdown, DIR, selected, ext=ext)
    
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
            daemon=False
        )
        p.start()
        self.launch_text.config(text="Script lancé. Appuyez sur 'n' pour le démarrer et 'z' pour l'interrompre")
        self.check_process(p)

    def check_process(self, p):
        if p.is_alive():
            self.root.after(100, lambda: self.check_process(p))
        else:
            self.launch_text.config(text="Script interrompu.")
