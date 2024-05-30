"""
Import necessary libraries for numerical computations, machine learning model handling, statistical analysis, 
signal processing, and data manipulation.
"""
import numpy as np
from joblib import load
import pandas as pd
from scipy.signal import windows

# Get location of this file to find path to models
from inspect import getsourcefile
from os.path import dirname
modelsDic = dirname(dirname(getsourcefile(lambda:0))) + "/models"

# Ignore inconsistent version warnings
# import warnings
# warnings.filterwarnings("ignore", category=sklearn.exceptions.InconsistentVersionWarning)

def psd_EM(signal, fs, flip=False):
    """
    Calculate the Power Spectral Density (PSD) of a given EEG signal.
    
    Parameters:
    - signal: The EEG signal data.
    - fs: Sampling frequency of the EEG signal.
    - flip: Boolean flag to indicate whether to flip the frequency axis.
    
    Returns:
    - amplitude: Amplitude spectrum of the signal.
    - freq: Frequency bins corresponding to the amplitude spectrum.
    """
    windowed_signal = signal * windows.blackman(signal.shape[0])
    eeg_fft = np.fft.fft(windowed_signal, axis=0)
    n = eeg_fft.shape[0]

    amplitude, phase = 2 * np.absolute(eeg_fft)[:n // 2], np.angle(eeg_fft, deg=True)[:n // 2]
    freq = np.fft.fftfreq(signal.shape[0], d=1.0 / fs)[:n // 2] if not flip else np.fft.fftfreq(signal.shape[0], d=1.0 / fs)[-n // 2 - (n + 1) % 2::-1]
    return amplitude, freq

def extract_features_EM(eeg_data, sfreq, num_channels=8):
    """
    Extracts features from EEG data for each frequency band (Delta, Theta, Alpha, Sigma, Beta).
    
    Parameters:
    - eeg_data: Multi-channel EEG data.
    - sfreq: Sampling frequency of the EEG data.
    - num_channels: Number of channels in the EEG data.
    
    Returns:
    - DataFrame containing PSD and DE values for each frequency band and channel.
    """
    bands = ['Delta', 'Theta', 'Alpha', 'Sigma', 'Beta']
    column_titles = [f'PSD_{band}_{channel}' for channel in range(num_channels) for band in bands]
    column_titles += [f'DE_{band}_{channel}' for channel in range(num_channels) for band in bands]
    df = pd.DataFrame(columns=column_titles)

    for channel in range(num_channels):
        PSD, freq = psd_EM(eeg_data[channel, :], sfreq)

        for band in bands:
            lower_bound, upper_bound = {'Delta': (0.5, 4.5), 'Theta': (4.5, 8.5), 'Alpha': (8.5, 11.5), 'Sigma': (11.5, 15.5), 'Beta': (15.5, 45)}[band]
            band_PSD = PSD[(freq >= lower_bound) & (freq < upper_bound)]
            mean_band_PSD = np.mean(band_PSD)
            sigma_squared = np.var(band_PSD)
            de = 0.5 * np.log(2 * np.pi * np.e * sigma_squared)

            df.loc[0, f'PSD_{band}_{channel}'] = mean_band_PSD
            df.loc[0, f'DE_{band}_{channel}'] = de

    return df

class EMBox(OVBox):
    """
    Custom OVBox implementation for processing EEG signals.
    Inherits from OVBox to utilize OpenViBE framework capabilities.
    """
    def __init__(self):
        super().__init__()
        self.signalHeader = None

    def process(self):
        """
        Process incoming EEG signal chunks. Handles both headers and buffers.
        
        - Saves signal header information upon receiving a header.
        - Processes EEG signal buffers to extract features and load a predictive model.
        - Outputs processed data or errors as needed.
        """
        for chunkIndex in range(len(self.input[0])):
            if isinstance(self.input[0][chunkIndex], OVSignalHeader):
                self.signalHeader = self.input[0].pop()
                outputHeader = OVSignalHeader(
                    self.signalHeader.startTime,
                    self.signalHeader.endTime,
                    self.signalHeader.dimensionSizes,
                    self.signalHeader.dimensionLabels,
                    self.signalHeader.samplingRate)
                # self.output[0].append(outputHeader)

            elif isinstance(self.input[0][chunkIndex], OVSignalBuffer):
                chunk = self.input[0].pop()
                concatenated = np.array(chunk).tolist()
                X_ML = []
                for i in range(self.signalHeader.dimensionSizes[0]):
                    start_idx = i * int(len(concatenated) / self.signalHeader.dimensionSizes[0])
                    end_idx = start_idx + int(len(concatenated) / self.signalHeader.dimensionSizes[0])
                    channel = concatenated[start_idx:end_idx]
                    X_ML.append(channel)
                X_ML = np.array([X_ML], dtype=object)
                X_ML = np.transpose(X_ML, (1, 2, 0))

                # print("Extracting features...")
                ML_features = extract_features_EM(X_ML, 256)

                try:
                    grid_search_cv = load(f"{modelsDic}/EM_model.pkl")
                    best_estimator = grid_search_cv.best_estimator_
                    # print("Model loaded. Type:", type(best_estimator))
                except Exception as e:
                    print("Error loading model:", e)
                    return
                
                if hasattr(best_estimator, 'predict_proba'):
                    # probabilities = [neutral, sad, fear, happy]
                    probabilities = best_estimator.predict_proba(ML_features)[0]
                    print("Probabilities:", probabilities)
                    maxProb = max(probabilities)
                    # Neutral
                    if probabilities[0] == maxProb:
                        typeCommand("Mid_Clouds")
                    # Sad
                    elif probabilities[1] == maxProb:
                        typeCommand("Rain_Storm")
                    # Fear
                    elif probabilities[2] == maxProb:
                        typeCommand(["Blizzard", "Snow"])
                    # Happy
                    else:
                        typeCommand("stoprain", WTcommand=False)
                else:
                    print("Best estimator does not have 'predict_proba' method.")

box = EMBox()
