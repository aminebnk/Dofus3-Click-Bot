import tkinter as tk

import PathMaker
import ResourceMapper
import WallMapper 

class ClickBot:
    def __init__(self, root):
        self.root = root
        root.title("Click Bot")

        # App buttons
        top = tk.Frame(root)
        top.pack(side="top", fill="x")

        inner = tk.Frame(top)
        inner.pack(expand=True)

        self.pathMaker_btn = tk.Button(inner, text="Run Path Maker", command=PathMaker.run)
        self.pathMaker_btn.pack(side="left", padx=6, pady=6)
        
        self.wallMapper_btn = tk.Button(inner, text="Run Wall Mapper", command=WallMapper.run)
        self.wallMapper_btn.pack(side="left", padx=6, pady=6)

        self.resourceMapper_btn = tk.Button(inner, text="Run Reource Mapper", command=ResourceMapper.run)
        self.resourceMapper_btn.pack(side="left", padx=6, pady=6)

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("900x600")
    app = ClickBot(root)
    root.mainloop()