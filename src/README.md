## Files and structure
The source directory is organized into several folders, each each related to specific components of the project and their respective documentation:

### 1. Emotion

Folder: `Emotions`. 
Contains all files pertaining to the emotions pipeline. These files and directories are:
- A python file which handles the link between data acquisition in OpenVIBE and modifying the game.
- The notebook which details the chosen model, this model is trained and the determined parameters are saved.
- The directory `models`, in which model parameters are stored.

### 2. Imaginary Movement

Folder: `Imaginary Movement`. 
Contains all files pertaining to the imagined movement pipeline. It has the same type of files as detailed [above](#1-emotion), for the two different models.

### 3. Game Modification
Folder: `Modding`. 
Contains all files in relation to modifying the game.

### Loose files
This directory also has the .xml OpenVIBE files that provide the link between the raw signals from the headset and python code execution. `main.xml` is the one used in the final result and is called upon by `main.py` in the root directory of the project. The other .xml files can be used to run the different model pipelines, see their respective documentation for more info. They need to be in this directory to have access to the modding functionality.
