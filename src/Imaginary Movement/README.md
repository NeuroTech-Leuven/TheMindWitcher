# Imaginary Movement
The imaginary movement pipeline aims to recognize three distinct states: moving the left hand, moving the right hand, and no movement. Imagined movement involves the player mentally simulating sensory and motor inputs, which can be challenging and requires training. To address this complexity, we initially focused on real movements, where the player's physical actions trigger in-game responses. For example, moving the left hand casts a magic spell, while moving the right hand calls a horse. Interestingly, the neural signals from real movements closely resemble those from imagined movements. In the future, we plan to leverage transfer learning to transition from real to imagined movements, thereby making the game accessible to players unable to physically use their hands.

In what follows, the different steps in the imaginary movement pipeline are explained: 
- To classify the movements of our users we trained two machine learning decoders; one machine learning approach and one deep learning approach.
- As every EEG may contain different kinds of noise, we pre-process our data.
- To pre-train our classifiers we leverage the Physionet dataset.
- Finally our results are presented.

## Model implementations
In order to achieve the imagined movement (IM) detection and classification goal, different models were trained using a public dataset. Once pre-trained, the models can then be finetuned for new hardware. In total, two pipelines are foreseen for the IM classification task; a machine learning pipeline and a deep learning pipeline. A first overview of these models can be found below.

|**Model**|Machine Learning|Deep Learning|
|-|-|-|
|**Description**| Consists of four major steps: <br>  1.    Preprocessing & Epoching <br>     2. CSP filter (individual calibration) <br>     3. Time & frequency domain features <br>     4. Gradient Boosted classifier (trained on PhysioNet database) | Utilizes a transformer model (ComfyNet) with three modules: <br>     1. PatchEmbedding <br>     2. TransformerEncoder <br>     3. ClassificationHead <br> |
|**Advantages**| * Higher classification performance because of subject-specific calibration | * Subject-independent <br> * Requires less preprocessing <br> |
|**Disadvantages**| * Requires calibration for each subject <br> | * Slightly lower performance compared to machine learning when calibrated for individual subjects |
|**Results**| 84% F1 score with 64 channels <br> 68% F1 score with 8 channels | 80% F1 score with 64 channels <br> 62% F1 score with 8 channels |

## Decoder 1: Machine learning
As illustrated in the figure below, the machine learning pipeline consists out of 4 major steps:

![Slide5](https://github.com/NeuroTech-Leuven/TheMindWitcher/assets/141845184/5f90df8b-c703-4ca5-b6c2-6dbb8e38acaa)

- After the preprocessing step, a Common Spatial Patterns (CSP) filter estimation step takes place for every subject. The CSP algorithm calculates mixtures of channels with mixture coefficients that are designed to maximize the difference in variance between our two classes (left versus right hand movement) - it maximizes their variance ratio - in their output. These mixures are thus essentially the outputs of specially designed spatial filters. In our pipeline, a spatial filter is estimated in a subject-dependent manner to increase the discrimination between our two IM classes.
  
- After this, a feature extraction step takes place, where Power Spectral Density (PSD) features and temporal features are calculated. PSD features capture the distribution of power across different frequency components. Temporal features provide additional information about the dynamics of EEG signals over time.
  
- In the final step, a gradient boosted classifier is trained on the extracted features from all subjects' epochs. Gradient boosting combines multiple weak learners, in our project decision trees, to build a robust predictive model. This ensemble approach iteratively improves the model's predictive performance by focusing on areas where previous models have performed poorly. Through this process, the classifier is optimized to accurately classify EEG data based on the features extracted earlier.

The subfolder `trainCSP`. contains all functionality to train the CSP model. To do this, simply run the file `main.py`. It contains OpenViBE .xml files and Python scripts identically to this folder and the `src` folder.


## Decoder 2: Deep learning
For the deep learning pipeline, various deep learning models, specifically transformers, were evaluated for classifying movements. The selected model, ComfyNet, is based on the transformer architecture described in a recent [academic paper](https://ieeexplore.ieee.org/document/9991178).

Transformers are a type of deep learning model that work by using self-attention mechanisms to weigh the importance of different parts of the input data, allowing them to capture long-range dependencies and contextual relationships more effectively than traditional neural networks.

The ComfyNet transformer model comprises three main modules:
- Patch Embedding: This module divides the input data into smaller patches and embeds them into a format suitable for the transformer network.
- Transformer Encoder: This core component applies self-attention and feed-forward neural network layers to process the embedded patches, capturing the intricate patterns and relationships within the data.
- Classification Head: This final module takes the encoded data and predicts the probabilities of the input data belonging to each classification class.

The key advantages of transformers include their ability to handle long-range dependencies, parallelize training processes, and achieve superior performance on complex tasks. In our pipeline, ComfyNet takes the preprocessed data as input, without requiring further modifications, and outputs the probabilities for each of the considered classification classes. This approach leverages the strengths of transformers to accurately classify the different movement states.

## Preprocessing
As the raw EEG data may contain drift, high frequency noise and/or powerline noise, the EEG data is first band-pass filtered using the cut-off frequencies 0.5 Hz and 40 Hz. Higher frequency information removal from the signal is no issue here, as we're mainly interested in the 8-30 Hz frequency range where signals from the motor cortex are found. After this, the EEG data is rereferenced using common average rereferencing, as this is considered to be beneficial for EEG classification tasks in general. In a last step, the data is epoched into 2 second windows with a given label. This data is then passed on to any of our model pipelines discussed below.

## Dataset
The public dataset that was used to pre-train our models is the [PhysioNet dataset](https://physionet.org/content/eegmmidb/1.0.0/). This dataset consists of over 1500 one and two minute  annotated EEG recordings, sampled at 160 Hz, obtained from 109 volunteers. The recorded events are rest, left, right for fist (real and imagined) and feet (real and imagined) motions. In our project we used the EEG segments representing rest, imagined opening or closing of the left fist, and imagined opening or closing of the right fist.

## Results

### Results ML approach
For data from the PhysioNet MI dataset, containing 64 channels, this strategy results in a classification performance with an f1 score of 84%, which is generally considered to be quite good for a multi-subject EEG application. In the case an Ant-Neuro headset is used, however, comprising only 8 channels, the performance drops to 68%.

### Results DL approach
For the PhysioNet dataset with 64 channels, this strategy results in a classification performance with an f1 score of 80%, which is again generally considered to be quite good for a multi-subject EEG application, especially if the pipeline is completely subject-independent, in contrast to the Machine Learning pipeline.
In the case an Ant-Neuro headset is used, however, comprising only 8 channels, the performance drops to 62%.

## Repository structure
This folder and its readme contain the details on the implementation of an (imaginary) movement classification system using EEG (electroencephalography) data.

The directory contains both the Python notebooks used to construct the data models and the Python files that connect the OpenViBE functionality with the game modification. The subfolder `models` contains all models used in this part of the project. The subfolder `trainCSP`. contains all functionality to train the CSP model.
