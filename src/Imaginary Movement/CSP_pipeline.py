import numpy as np
import pickle

from scipy.stats import kurtosis, skew
import pandas as pd

# Ignore inconsistent version warnings
# import warnings
# warnings.filterwarnings("ignore", category=sklearn.exceptions.InconsistentVersionWarning)
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def psd_computation_CSP(signal, fs, flip=False):
    """
    Compute the Power Spectral Density (PSD) of a signal.
 
    Parameters:
    - signal: 1D numpy array containing the signal.
    - fs: Sampling frequency of the signal.
    - flip_w
 
    Returns:
    - freqs: Array of sample frequencies.
    - psd: Power spectral density of the signal.
    """
    eeg_fft = np.fft.fft(signal * np.blackman(signal.shape[0]), axis=0)
    n = eeg_fft.shape[0]
 
    # get amplitude and phase
    amplitude, phase = 2 * np.absolute(eeg_fft)[:n // 2], np.angle(eeg_fft, deg=True)[:n // 2]
 
    # get frequencies
    if not flip:
        freq = np.fft.fftfreq(signal.shape[0], d=1.0/fs)[:n // 2]  # get positive frequency spectrum half
    else:
        freq = np.fft.fftfreq(signal.shape[0], d=1.0/fs)[-n // 2 - (n + 1) % 2::-1]  # get positive frequency spectrum and flip
 
    return amplitude, freq

def psd_CSP(trials):
    '''
    Calculates for each trial the Power Spectral Density (PSD).

    Parameters
    ----------
    trials : 3d-array (channels x samples x trials)
        The EEG signal

    Returns
    -------
    trial_PSD : 3d-array (channels x PSD x trials)
        the PSD for each trial.
    freqs : list of floats
        Yhe frequencies for which the PSD was computed (useful for plotting later)
    '''
    nchannels = trials.shape[0]
    ntrials = trials.shape[2]

    # Iterate over trials and channels
    for trial in range(ntrials):
        for ch in range(nchannels):
            # Calculate the PSD according to Welch's method
            # For numpy data, we can use the following: https://het.as.utexas.edu/HET/Software/Matplotlib/api/mlab_api.html#matplotlib.mlab.psd
            # Fs = math.floor(len(trials[ch,:,trial])/2)
            Fs = 512
            # print(np.array(trials[ch,:,trial]))
            (PSD, freqs) = psd_computation_CSP(np.array(trials[ch,:,trial]), Fs)

            if trial == 0 and ch == 0:
                trials_PSD = np.zeros((nchannels, len(PSD), ntrials))

            trials_PSD[ch, :, trial] = PSD

    return trials_PSD, freqs

def apply_mix_CSP(W, trials):
    ''' Apply a mixing matrix to each trial (basically multiply W with the EEG signal matrix)'''
    nchannels = trials.shape[0]
    nsamples = trials.shape[1]
    ntrials = trials.shape[2]
    trials_csp = np.zeros((nchannels, nsamples, ntrials))
    for i in range(ntrials):
        trials_csp[:,:,i] = W.T.dot(trials[:,:,i])
    return trials_csp

def extract_features_CSP(trials_PSDs, trials_time, CSP_components,freqs):

    component_1 = CSP_components[0]
    component_2 = CSP_components[1]

    df = pd.DataFrame(columns=['psd_power_1', 'psd_max_1', 'psd_kurtosis_1', 'psd_skew_1',
                               'psd_power_2', 'psd_max_2', 'psd_kurtosis_2', 'psd_skew_2',
                               'time_logvar_1', 'time_rms_1', 'time_kurtosis_1', 'time_skew_1',
                               'time_logvar_2', 'time_rms_2', 'time_kurtosis_2', 'time_skew_2'])
    for trial in range(trials_PSDs.shape[2]):
        freqs_temp = freqs[0:trials_PSDs[component_1,:,trial].shape[0]]
        psd_component_1 = trials_PSDs[component_1,:,trial][(freqs_temp>=8) & (freqs_temp<=15)]
        psd_component_2 = trials_PSDs[component_2,:,trial][(freqs_temp>=8) & (freqs_temp<=15)]

        power_1 = np.trapz(psd_component_1)
        power_2 = np.trapz(psd_component_2)
        max_1 = max(psd_component_1)
        max_2 = max(psd_component_2)
        kurtosis_1 = kurtosis(psd_component_1)
        kurtosis_2 = kurtosis(psd_component_2)
        skew_1 = skew(psd_component_1)
        skew_2 = skew(psd_component_2)

        time_component_1 = trials_time[component_1,:,trial]
        time_component_2 = trials_time[component_2,:,trial]

        logvar_1 = np.log(np.var(time_component_1))
        logvar_2 = np.log(np.var(time_component_2))
        rms_1 = np.sqrt(np.mean(np.square(time_component_1)))
        rms_2 = np.sqrt(np.mean(np.square(time_component_2)))
        time_kurtosis_1 = kurtosis(time_component_1)
        time_kurtosis_2 = kurtosis(time_component_2)
        time_skew_1 = skew(time_component_1)
        time_skew_2 = skew(time_component_2)

        row = [power_1, max_1, kurtosis_1, skew_1,
               power_2, max_2, kurtosis_2, skew_2,
               logvar_1, rms_1, time_kurtosis_1, time_skew_1,
               logvar_2, rms_2, time_kurtosis_2, time_skew_2]

        df.loc[len(df)] = row

    return df

# we use numpy to compute the mean of an array of values
import numpy

# let's define a new box class that inherits from OVBox
class CSPBox(OVBox):
    def __init__(self):
        OVBox.__init__(self)
        # we add a new member to save the signal header information we will receive
        self.signalHeader = None

    # The process method will be called by openvibe on every clock tick
    def process(self):
       # we iterate over all the input chunks in the input buffer
        for chunkIndex in range(len(self.input[0])):
            # if it's a header we save it and send the output header (same as input, except it has only one channel named 'Mean')
            if type(self.input[0][chunkIndex]) == OVSignalHeader:
                self.signalHeader = self.input[0].pop()
                outputHeader = OVSignalHeader(
                    self.signalHeader.startTime,
                    self.signalHeader.endTime,
                    self.signalHeader.dimensionSizes,
                    self.signalHeader.dimensionLabels,
                    self.signalHeader.samplingRate)

            # if it's a buffer we pop it and put it directly in the box output buffer
            elif type(self.input[0][chunkIndex]) == OVSignalBuffer:
                chunk = self.input[0].pop()
                concatenated = numpy.array(chunk).tolist()
                X_ML = []
                for i in range(self.signalHeader.dimensionSizes[0]-2):
                    start_idx = i * int(len(concatenated)/self.signalHeader.dimensionSizes[0])
                    end_idx = start_idx + int(len(concatenated)/self.signalHeader.dimensionSizes[0])
                    channel = concatenated[start_idx:end_idx]
                    X_ML.append(channel)
                X_ML = np.array([X_ML], dtype=object)
                X_ML = np.transpose(X_ML, (1, 2, 0))

                with open(f"models/W_CSP.pkl", 'rb') as f:
                    W = pickle.load(f)[0]

                X_ML_CSP = apply_mix_CSP(W,X_ML)
                X_ML_CSP_PSD, freqs = psd_CSP(X_ML)
                ML_features= extract_features_CSP(X_ML_CSP_PSD, X_ML_CSP, [0, -2],freqs)
                model = 2
                if model == 3:
                    with open(f"models/Physionet_3_class_best.pkl", 'rb') as f:
                        loaded_model = pickle.load(f)
                else:   
                    with open(f"models/Physionet_2_class_best.pkl", 'rb') as f:
                        loaded_model = pickle.load(f)

                # probabilities = [left, right, ~nothing]
                probabilities = loaded_model.predict_proba(ML_features)[0]
                print(probabilities)
                if probabilities[0] > 0.55:
                    # Left action
                    print("LEFT: Cast sign.")
                    pressKey(SPELL_KEY)
                elif probabilities[1] > 0.7:
                    # Right action
                    print("RIGHT: Call horse")
                    pressKey(HORSE_KEY)
                else:
                    print("No action")
                

# Finally, we notify openvibe that the box instance 'box' is now an instance of MyOVBox.
# Don't forget that step!!
box = CSPBox()
