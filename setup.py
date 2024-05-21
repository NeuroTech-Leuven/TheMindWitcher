import tkinter as tk
from tkinter import filedialog as fd

# This code activates a GUI to select the directory where the game The Witcher 3 is installed.
# It then modifies the necessary file so you can activate the debug console in-game.  

window = tk.Tk()

# Code that activates the debug console.
def activateConsole():
    dirPath = fd.askdirectory(parent=window, title='Select directories', mustexist=True)
    # Activate debug console
    try:
        with open(f"{dirPath}/bin/config/base/general.ini", "r+") as file:
            for line in file:
                if line.rstrip() == "DBGConsoleOn=true":
                    window.destroy()
                    return
            # Check if last line has a new line starting already
            if line[-1:] != '\n':
                file.write("\n")
            file.write("DBGConsoleOn=true")
    # Catch the event wherein the wrong directory was selected.
    except FileNotFoundError:
        print("This directory does not contain an install of The Witcher 3. The right directory looks like " +
              "C:/{CUSTOM_PATH}/The Witcher 3 Wild Hunt GOTY")
    finally:
        window.destroy()

# Set window title
window.title('File Explorer')

#Set window background color
window.config(background = "white")

# Create a File Explorer label
label_dir_explorer = tk.Label(window, 
							text = "Select the directory where The Witcher 3 is installed",
							width = 100, height = 4, 
							fg = "black",
                            font=('Segoe UI', 12))

# Browse files button.	
button_explore = tk.Button(window, 
						text = "Browse Files",
						command = activateConsole) 

# Grid method is chosen for placing the widgets at respective positions 
# in a table like structure by specifying rows and columns
label_dir_explorer.grid(column = 0, row = 0, rowspan=3)
button_explore.grid(column = 0, row = 5, rowspan=2)

window.mainloop()


