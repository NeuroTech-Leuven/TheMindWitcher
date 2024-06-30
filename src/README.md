## Files and structure
The source directory is organized into several folders, each containing specific components of the project:

### 1. Emotions

Folder: `Emotions`. 
Contains all files to do with the emotions pipeline, from generating model parameters to the python script executing in-game actions.

### 2. Imaginary Movement

Folder: `Imaginary Movement`. 
Contains all files to do with the imaginary movement pipeline, from generating model parameters to the python script executing in-game actions.

### 3. Game Modification
Folder: `Modding`. 
Contains all files to do with modifying the game.

### Loose files
This directory also has the .xml OpenVIBE files that provide the link between the raw signals from the headset and python code execution. `main.xml` is the one used in the final result and is called upon by `main.py` in the root directory of the project. The other .xml files can be used to run the different model pipelines, see their respective documentation for more info. They need to be in this directory to have access to the modding functionality.