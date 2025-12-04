from clickbot import ClickBot
import tkinter as tk
import multiprocessing as mp

def main():
    root = tk.Tk()
    root.geometry("900x600")
    app = ClickBot(root)
    root.mainloop()

if __name__ == "__main__":
    mp.freeze_support()
    main()