# Emotions Classification: A Machine Learning Approach

The emotion pipeline aims to recognize 4 emotions — neutral, happy, sad, and fear — based on real-time EEG (electroencephalography) signals of the user playing 'The Witcher 3'. The emotions are translated in the game as wheater changes. E.g. when a sad/fear emotion is detected, a storm will arise.

![GIF_storm](https://github.com/NeuroTech-Leuven/TheMindWitcher/assets/141845184/1e4cb4d7-b21d-40fd-9eef-b0644b83a800)


In what follows, the different steps in the emotion pipeline are explained:
- To classify the emotions of our users we train a machine learning algorithm.
- As every EEG may contain different kinds of noise, we pre-process our data.
- To pre-train our machine learning pipeline we leverage the public dataset SEED IV.
- Finally our results are presented.

## Machine Learning Decoder
Machine learning plays a pivotal role in the emotion classification pipeline, enabling the translation of raw EEG data into recognizable emotional states. This includes the extraction of relevant features from preprocessed EEG signals, training a classifier, and fine-tuning the model to achieve high accuracy in emotion detection.

### Feature extraction
This project employs Power Spectral Density (PSD) features and Differential Entropy features, which are computed across 8 EEG channels and 5 frequency bands: Delta, Theta, Alpha, Sigma, and Beta. The standard ranges of frequencies in human EEG are illustrated in the figure below:

![freq_bands](https://github.com/NeuroTech-Leuven/TheMindWitcher/assets/141845184/a5f40ec9-08b3-42b1-9dd8-5e2a8b5ee64d)

- Power Spectral Density (PSD): 
  PSD measures the power distribution of the EEG signal as a function of frequency. It provides insights into the energy present in different frequency bands, which is         crucial for understanding the underlying neural activity associated with different emotional states. For example, a high PSD value in the alpha band might                    indicate a relaxed and calm state, whereas a high PSD value in the beta band might be associated with increased cognitive activity or stress.

- Differential Entropy: 
  Differential entropy is a measure derived from information theory that quantifies the complexity or unpredictability of the EEG signal. It provides a robust measure of       signal variability and is particularly useful in characterizing the non-linear dynamics of brain activity. For example, higher differential entropy values in the EEG         signal might correspond to more complex and irregular brain activity typically observed during states of anxiety or excitement, while lower entropy values might indicate     more regular and predictable patterns associated with states of relaxation or boredom.

### Classification: gradient boosting
After feature extraction, a gradient-boosted classifier is trained on the extracted features from all subjects' epochs. Gradient boosting is a powerful ensemble learning technique that builds a strong predictive model by combining multiple weak learners, in our project simple decision trees. To illustrate this concept we can apply the following analogy: Imagine building a robust bridge. Instead of using one massive block of concrete (a single complex model), we use many smaller bricks (simple decision trees). Each brick alone might not support much weight (a weak learner), but when carefully stacked and reinforced together, they create a sturdy and reliable bridge (a strong predictive model). 

To optimize the model's performance, a grid search is employed to systematically explore the hyperparameter space and identify the best combination of parameters.
The grid search process is akin to experimenting with different types of reinforcement and arrangements for our bridge. We test various materials and structural designs (hyperparameters) to find the most effective combination that ensures our bridge can withstand the heaviest loads and toughest conditions.


## Preprocessing

The raw EEG data may contain different kinds of noise e.g., drift, high-frequency noise, and/or powerline noise. Hence, the data needs to be filtered. We employed a classical filtering technique, where the EEG data is bandpass filtered using the cut-off frequencies 0.5 Hz and 40 Hz. 

Working with real-time data requires the data to be split up into windows. Therefore, after the bandpass filtering, the EEG data is epoched into 4-second windows with a 50% overlap. The clean, preprocessed data is then transferred to the model pipeline discussed above.

## Dataset

The public dataset [SEED IV dataset](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html) was used to pre-train our models. Once pre-trained, the model could be used in a "plug-and-play" manner. 

The dataset consists of 15 users' recordings across 3 sessions of 24 trials each. Each of these trials is an EEG recording of a total of 2m50s, sampled at 200 Hz. The dataset includes 6 male subjects and 9 female subjects, each experiencing the four different states of emotions for which we aimed to build the classifier.

## Results 
After transitioning to the 8-channel EEG Ant Neuro headset, the model's performance drops from 75% (with the 64-channel data) to 66%, as not all channels provide relevant information for accurate classification.  However, if a 64-channel headset were to be used, this accuracy could be further enhanced.

## Repository structure
This folder and its readme  contain the details on the implementation of an emotion classification system using EEG data. The subfolder `models` contains all the models used in this part of the project.



