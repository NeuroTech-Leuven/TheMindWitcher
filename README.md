# Neurotech Leuven - The Mindwitcher Project

## Introduction
Welcome to the repository for the Mindwitcher project, developed by the Team Neurotech Leuven. This project is our submission for the Neurotech X competition 2024, where we present an innovative modification to the popular game *The Witcher 3*. The Mindwitcher project integrates EEG signals to control certain actions within the game, creating a unique gaming experience that merges neuroscience and technology.

## Video
_Here comes the video_

## Project Overview
From beginning to end, the Mindwitcher project focuses on utilizing EEG signals to control actions in *The Witcher 3* game. Specifically, we have implemented two main functionalities:
1. **Spell Casting and Horse calling**: EEG signals related to movement and imagined movement are used to call for the player's horse and cast spells within the game.
2. **Emotion-based Weather Modification**: The EEG headset also captures emotional signals from the player, which dynamically modify the weather conditions within the game environment.

For a full overview of the project the following diagram may be of use:

![](docs/Overview.svg)

#### Stimuli and headset
The stimuli in this project are your emotions and (imaginary) movement by your left and right hand. The brain signals are measured using the AntNeuro headset _SPECIFY THE EXACT MODEL?_. Of our available choices for the headset, it provided the best node placement. Some gel is needed to apply to the nodes.

#### Data processing
The signals obtained from the headset are perturbed by noise due to various effects, such as powerline noise, movement artefacts and other brain activity. So data needs to be processed with this in mind. Depending on what we want to see from the data, different classifying models were built. For more details on the imaginary movement models, see [here](<docs/Imaginary Movement.md>). Two were developed, a Machine Learning model using a Common Spatial Pattern (CSP) filter and a Deep Learning model. For classifying emotions, only a Deep Learning model was developed, more details to be found [here](docs/Emotions.md).

The OpenVIBE files provide the link between this data and the classifying decision. It receives the data and applies temporal filters and time epoching. For imaginary movement, a decision is made each second based on the data of the past two seconds. For the emotions, we make a decision every half minute based on data of the five seconds before. Further processing is done by some python scripts, in which the different models are applied and a classification is made. 

#### Game modification
Once the classification is decided, the right action needs to be executed in the game. This is done using keyboard commands, where python code simulates a key press. From the imaginary movement to casting a spell or calling your horse, it is just a matter of pressing the right key that controls that action. Changing the weather is not so straightforward and so we make use of the in-built debug console. More details are available [here](docs/Modding.md).

## Repository Structure
The repository is organized into several folders, each containing specific components of the project:

### 1. Model notebooks
Folder: `notebooks`
- This folder contains the python notebooks used to define and train the several data processing models. Model parameters were exported from here and used in the final pipeline.

### 2. Code
Folder: `Code`
- It has the .xml OpenVIBE files that provide the link between the raw signals from the headset and python code execution. 
- This folder further has two subfolders:
  - `models`: Contains the model parameters exported using the notebooks (cf. supra).
  - `pythonscripts`: Contains the python code that is imported in OpenVIBE. It applies the models to the data and executes the classifying decision. 

### 3. Documentation
Folder: `docs`
- Extra documentation regarding the different parts of the project. 
- Driver files to use in installation

## Installation and Usage
The project requires a Windows installation to run its two main software components: the game and OpenVIBE, with python code to do all calculations and communication. 

OpenVIBE requires a specific python version and is not compatible with a virtual environment. To be able to run all code, install [Python version 3.10.11](https://www.python.org/downloads/release/python-31011/) and edit your Windows environment variables to put this installation on top of your PATH. All dependencies for the project can then be installed with
```bash
pip install -r requirements.txt
```

With the right version of python installed, now install the latest version of [OpenVIBE](https://openvibe.inria.fr). Next install the game [The Witcher 3](https://www.thewitcher.com/pl/en/witcher3), which only officially runs on Windows. To be able to mod the game, execute
```bash
python setup.py
```
and navigate to the directory where the game is installed. Typically, this is something like C:/Games/The Witcher 3.

To properly use the AntNeuro headset, the right drivers need to be installed. First connect the headset with its USB port to your pc. Then open your device manager and follow the steps from [step 8](https://www.wikihow.com/Copy-Drivers-from-One-Computer-to-Another-on-PC-or-Mac) onward. The driver files are included in the repository in `drivers`. Finally, configure the OpenVIBE acquisition server with the settings as seen in [this image](<docs/OpenVIBE Acquisition Server setup.png>).

Now all that's left to do is run the `main.py` file. It sets up a training/calibration session for the CSP model used for Imaginary Movement. Once that's done, it starts running the model pipelines so just open the game and start playing!

## Limitations and future plans
- Due to mismatch between the data the models were trained on and the data we get from the headset, our models underperform quite heavily. Future plans would include to gather lots of data with our own headset and train the models with that. 
- For accuracy reasons, a real movement is highly preferred over imaginary movement.

## Contact
For any inquiries or feedback regarding the Mindwitcher project, please contact hq@ntxl.org.

We hope you enjoy exploring the Mindwitcher project and experiencing the fusion of neuroscience and gaming!
