# Imaginary Movement
This folder and its readme contain the details on the implementation of an (imaginary) movement classification system using EEG (electroencephalography) data. This project aims to recognize three different states: moving your left hand, right hand or no hand at all.

The directory contains both the Python notebooks used to construct the data models and the Python files that connect the OpenVIBE functionality with the game modification. The subfolder `models` contains all models used in this part of the project. The subfolder `trainCSP`. contains all functionality to train the CSP model.

In what follows, the different steps in the imaginary movement pipeline are explained: the pre-training, the preprocessing and the 2 machine learning decoders(machine learning and deep learning).

## Dataset
The public dataset that was used to pre-train our models is the [PhysioNet dataset](https://physionet.org/content/eegmmidb/1.0.0/). This dataset consists of over 1500 1 and 2 minute  annotated EEG recordings, sampled at 160 Hz, obtained from 109 volunteers. The recorded events are rest, left, right and both fist motion (real and imagined) and motion of both feet (real and imagined). In our project we used the EEG segments representing rest, imagined opening or closing of the left fist, and imagined opening or closing of the right fist.

## Preprocessing
As the raw EEG data may contain drift, high frequency noise and/or powerline noise, the EEG data is first band-pass filtered using the cut-off frequencies 0.5 Hz and 40 Hz. Higher frequency information removal from the signal is no issue here, as we're mainly interested in the 8-30 Hz frequency range where signals from the motor cortex are found. After this, the EEG data is rereferenced using common average rereferencing, as this is considered to be beneficial for EEG classification tasks in general. In a last step, the data is epoched into 2 second windows with a given label. This data is then passed on to any of our model pipelines discussed below.

## Model implementations
In order to achieve the imagined movement (IM) detection and classification goal, different models were trained using a public dataset. Once pre-trained, the models can then be finetuned for new hardware. In total, two pipelines are foreseen for the IM classification task; a machine learning pipeline and a deep learning pipeline. These models will be further discussed below.

## Decoder 1: Machine learning
As illustrated in the figure below, the machine learning pipeline consists out of 3 major steps:

![Slide5](https://github.com/NeuroTech-Leuven/TheMindWitcher/assets/141845184/5f90df8b-c703-4ca5-b6c2-6dbb8e38acaa)

- First, a Common Spatial Patterns (CSP) filter estimation step takes place for every subject. Here, a spatial filter is estimated in a subject-dependent manner to increase the discrimination between our two IM classes. In short, this is done through the maximization of a class-variance ratio criterion. More details and a more in-depth theoretical background can be found in the extensively documented [jupiter notebooks](./CSP_pipeline_all_subjects_2_classes.ipynb)
.
- After this, a feature extraction step takes place, where Power Spectral Density (PSD) features and temporal features are calculated. PSD features capture the distribution of power across different frequency components. Temporal features provide additional information about the dynamics of EEG signals over time.
- In the final step, a gradient boosted classifier is trained on the extracted features from all subjects' epochs. Gradient boosting combines multiple weak learners to build a robust predictive model. This ensemble approach iteratively improves the model's predictive performance by focusing on areas where previous models have performed poorly. Through this process, the classifier is optimized to accurately classify EEG data based on the features extracted earlier.

The subfolder `trainCSP`. contains all functionality to train the CSP model. To do this, simply run the file `main.py`. It contains OpenVIBE .xml files and Python scripts identically to this folder and the `src` folder.

### Results ML approach
For data from the PhysioNet MI dataset, containing 64 channels, this strategy results in a classification performance with an f1 score of 84%, which is generally considered to be quite good for a multi-subject EEG application. In the case an Ant-Neuro headset is used, however, comprising only 8 channels, the performance drops to 68%.

The fact that this pipeline contains a subject-specific component, the CSP filter, is both a blessing and a curse. On the one hand, it enhances performance, but on the other hand, it requires of calibration data for each new subject trying out the system. This calibration data can for example be acquired by the subject executing cued performed or imagined movement. 

## Decoder 2: Deep learning
In the deep learning pipeline, different deep learning models (more specifically transformers) were considered. The chosen transformer  was based on the model presented in the following [academic paper](https://ieeexplore.ieee.org/document/9991178).

The transformer-model, called ComfyNet, is an attention-based network consisting of three modules:
- PatchEmbedding
- TransformerEncoder
- ClassificationHead

It takes the preprocessed data as its input, without further modifications, and returns the probabilities of the input data belonging to our considered classification classes.

### Results DL approach
For the PhysioNet dataset with 64 channels, this strategy results in a classification performance with an f1 score of 80%, which is again generally considered to be quite good for a multi-subject EEG application, especially if the pipeline is completely subject-independent, in contrast to the Machine Learning pipeline.

In the case an Ant-Neuro headset is used, however, comprising only 8 channels, the performance drops to 62%.
