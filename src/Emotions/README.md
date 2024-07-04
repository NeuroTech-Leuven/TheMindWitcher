# Emotions Classification: A Machine Learning Approach

This folder and its readme  contain the details on the implementation of an emotion classification system using EEG (electroencephalography) data. This part of the project aims to recognize four emotions— neutral, happy, sad, and fear — based on real-time EEG signals of the user playing 'The Witcher 3'.

To achieve the emotion classification goal, two different decorders were initially considered: a deep learning and a machine learning approach. However, since we observed the DL model to overfit the training data significantly, we further developed the ML approach as the most viable option for this application. 

In what follows, the different steps in the emotion pipeline are explained: the preprocessing, the machine learning decoder (including feature extraction and classification) and the pre-training.

## Preprocessing

As the raw EEG data may contain drift, high-frequency noise, and/or powerline noise, the EEG data is first bandpass filtered using the cut-off frequencies 0.5 Hz and 40 Hz. After this, the EEG data is epoched into 4-second windows with a 50% overlap. This data is then transferred to the model pipeline discussed below.

## Machine Learning Decoder
Following preprocessing and preceding classification, the relevant features are extracted to facilitate emotion recognition from EEG data. This project employs Power Spectral Density (PSD) features and Differential Entropy features, which are computed across 8 EEG channels and 5 frequency bands: Delta, Theta, Alpha, Sigma, and Beta.

### Feature extraction
Power Spectral Density (PSD): PSD measures the power distribution of the EEG signal as a function of frequency. It provides insights into the energy present in different frequency bands, which is crucial for understanding the underlying neural activity associated with different emotional states.

Differential Entropy: Differential entropy is a measure derived from information theory that quantifies the complexity or unpredictability of the EEG signal. It provides a robust measure of signal variability and is particularly useful in characterizing the non-linear dynamics of brain activity. Higher entropy values indicate more irregular and complex brain activity, which can be linked to specific emotional states.

### Classification
After feature extraction, a gradient-boosted classifier is trained on the extracted features from all subjects' epochs. Gradient boosting is a powerful ensemble learning technique that builds a strong predictive model by combining multiple weak learners. To optimize the model's performance, a grid search is employed to systematically explore the hyperparameter space and identify the best combination of parameters.

## Pre-training the model

The public dataset [SEED IV dataset](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html) was used to pre-train our models. Once pre-trained, the model could be used in a "plug-and-play" manner. 

The dataset consists of 15 users' recordings across 3 sessions of 24 trials each. Each of these trials is an EEG recording of a total of 2m50s, sampled at 200 Hz. The dataset includes 6 male subjects and 9 female subjects, each experiencing the four different states of emotions for which we aimed to build the classifier.


## Results and future plans
After transitioning to the 8-channel EEG Ant Neuro headset, we achieved a limited model's performance of 66%, as not all channels provide relevant information for accurate classification.  However, if a 64-channel headset were to be used, this accuracy could be further enhanced.






