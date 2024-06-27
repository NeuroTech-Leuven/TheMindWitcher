# Emotions Classification: A Machine Learning Approach
This file contains the details on the implementation of an emotion classification system using EEG (electroencephalography) data. This project aims to recognize four emotions—neutral, happy, sad, and fear based on real-time EEG signals of the user playing 'The Witcher 3'.


## Model implementations
In order to achieve the emotion classification goal, two different approaches were initially viewed; the deep learning and the machine learning approaches. However, since we observed the DL model to overfit the training data significantly, we further developped the ML approach as the most viable option for this application. 

To train the model we ussed a private dataset, SEED IV, with recording made by the BCMI laboratory. Once the model was pre-trained, they were "plug and play" ready. The ML model will be further discussed below.


## Used dataset
The public dataset that was used to pre-train our models is the [SEED IV dataset](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html). This dataset consists of 15 users recordings across 3 sessions of each 24 trials. Each of those trials being EEG recordings of total 2 min 50s. These recordings sampled at 200 Hz. 
The dataset contains the following split 

male_sers:1、2、6、7、12、13
female_users:3、4、5、8、9、10、11、14、15

Label:
The labels of the three sessions for the same subjects are as follows,
- session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3];
- session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1];
- session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0];

where each trial label corresponds to: 
- 0: neutral
- 1: sad 
- 2: fear
- 3: happy


## Used preprocessing
As the raw EEG data may contain drift, high frequency noise and/or powerline noise, the EEG data is first bandpass filtered using the cut-off frequencies 0.5 Hz and 40 Hz. After this, the EEG data is epoched into 4 second windows with a 50% overlap. This data is then transfered to the model pipeline discussed below.


## Machine learning model

In the machine learning pipeline, components were designed to work together meticulously.The feature extraction step takes place, where Power Spectral Density (PSD) features and Differential Entropy features were computed over 8 channels and across 5 frequency Bands. The frequency bands are as follows: bands = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']

In a last step, a gradient boosted classifier with grid search is trained on all the subjects' epochs' extracted features.



Since the 8 channel EEG ant Neuro headset us used, the performance of the model is of 66%, because these channels do not all contain relevant information to classify the information.
However if a 64 channel headset were to be used, this accuracy could be further enhanced. 

