# Imaginary Movement
This file contains the details on the implementation of an (imaginary) movement classification system using EEG (electroencephalography) data. This project aims to recognize two different states: moving your left hand, right hand or no hand at all.

## Files and structure
This directory contains both the Python notebooks used to construct the data models and the Python files that connect the OpenVIBE functionality with the game modification.

### Models
Folder: `models`. 
Contains all models used in this part of the project.

### Training the CSP model
Folder: `trainCSP`. 
Contains all functionality to train the CSP model. To do this, simply run the file `main.py`. It contains OpenVIBE .xml files and Python scripts identically to this folder and the `src` folder.

## Model implementations
In order to achieve the imagined movement (IM) detection and classification goal, different models were trained using a public dataset. Once pre-trained, the models can then be finetuned for new hardware. In total, two pipelines are foreseen for the IM classification task; a machine learning pipeline and a deep learning pipeline. These models will be further discussed below.


## Used dataset
The public dataset that was used to pre-train our models is the [Physionet dataset](https://physionet.org/content/eegmmidb/1.0.0/). This dataset consists of over 1500 one- and two-minute EEG recordings, sampled at 160 Hz, obtained from 109 volunteers. In summary, the experimental runs were:
- Run 1: Baseline, eyes open
- Run 2: Baseline, eyes closed
- Run 3: Task 1 (open and close left or right fist)
- Run 4: Task 2 (imagine opening and closing left or right fist)
- Run 5: Task 3 (open and close both fists or both feet)
- Run 6: Task 4 (imagine opening and closing both fists or both feet)
- Run 7: Task 1
- Run 8: Task 2
- Run 9: Task 3
- Run 10: Task 4
- Run 11: Task 1
- Run 12: Task 2
- Run 13: Task 3
- Run 14: Task 4


Each run contains annotations, with annotation including one of three codes of events (T0=1, T1=2, or T2=3):
- T0 corresponds to rest
- T1 corresponds to onset of motion (real or imagined) of
    > the left fist (in runs 3, 4, 7, 8, 11, and 12)
    > both fists (in runs 5, 6, 9, 10, 13, and 14)
- T2 corresponds to onset of motion (real or imagined) of
    > the right fist (in runs 3, 4, 7, 8, 11, and 12)
    > both feet (in runs 5, 6, 9, 10, 13, and 14)


We used the data from runs 4, 8, and 12. Our different classification classes are thus rest, imagined opening or closing of the left fist, and imagined opening or closing of the right fist.


## Used preprocessing
As the raw EEG data may contain drift, high frequency noise and/or powerline noise, the EEG data is first bandpass filtered using the cut-off frequencies 0.5 Hz and 40 Hz. Higher frequency information removal from the signal is no issue here, as we're mainly interested in the 8-30 Hz frequency range where signals from the motor cortex are found. After this, the EEG data is rereferenced using average rereferencing, as this is considered to be beneficial for EEG classification tasks in general. In a last step, the data is epoched into 2 second windows with a given label. This data is then transfered to any of our model pipelines discussed below.


## Machine learning model
In the machine learning pipeline, three pipeline components were designed to work together meticulously. First, a Common Spatial Pattern (CSP) filter estimation step takes place for every subject. Here, a spatial filter is estimated in a subject-dependent manner to increase the discriminativity between our two IM classes. In short, this is done through the maximization of a class-variance ratio criterion, but more details and a more in-depth theoretical background can be found in the extensively documented jupyter notebooks. After this, a feature extraction step takes place, where Power Spectral Density (PSD) features and temporal features are calculated for the 8-15 Hz frequency band. In a last step, a gradient boosted classifier is trained on all the subjects' epochs' extracted features.

![Slide5](https://github.com/NeuroTech-Leuven/TheMindWitcher/assets/141845184/5f90df8b-c703-4ca5-b6c2-6dbb8e38acaa)

In the case a BCI2000 EEG system is used, comprising 64 channels, this strategy results in a classification performance with an f1 score of 84%, which is generally considered to be quite good for a multi-subject EEG application.


In the case an Ant-Neuro headset is used, however, comprising only 8 channels, the performance drops to 68%.


The fact that this pipeline contains a subject-specific component, the CSP filter, is both a blessing and a curse. On the one hand, it facilitates the obtention of good results, but on the other hand, it does result in the requirement of calibration data for each new subject trying out the system. This calibration data can for example be acquired through the use of passive movements, where another person moves the limbs of the person of interest to simulate IM EEG data.


## Deep learning models
In the deep learning pipeline, two deep learning models were considered, both of which are transformers. Our first transformer was based on the following non-academic [github repository](https://github.com/reshalfahsi/eeg-motor-imagery-classification/tree/master). Our second transformer was based on the model presented in the following [academic paper](https://ieeexplore.ieee.org/document/9991178). As this second model outperformed the first, this was also chosen as the model used in our final deep learning-based classification pipeline.


The transformer-model, called ComfyNet, is an attention-based network consisting of three modules:
- PatchEmbedding
- TransformerEncoder
- ClassificationHead


It takes the preprocessed data as its input, without further modifications, and returns the probabilities of the input data belonging to our considered classification classes.


In the case a BCI2000 EEG system is used, comprising 64 channels, this strategy results in a classification performance with an f1 score of 80%, which is again generally considered to be quite good for a multi-subject EEG application, especially if the pipeline is completely subject-independent, in contrast to the Machine Learning pipeline.


In the case an Ant-Neuro headset is used, however, comprising only 8 channels, the performance drops to 62%.
