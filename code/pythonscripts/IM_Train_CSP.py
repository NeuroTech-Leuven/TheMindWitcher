import mne
import numpy as np
from numpy import linalg
import pickle
from inspect import getsourcefile
from os.path import dirname

sourceDic = dirname(dirname(getsourcefile(lambda:0)))

def cov(trials):
    ''' Calculate the covariance for each trial and return their average '''
    nsamples = trials.shape[1]
    ntrials = trials.shape[2]
    covs = [ trials[:,:,i].dot(trials[:,:,i].T) / nsamples for i in range(ntrials) ]
    return np.mean(covs, axis=0)


def whitening(sigma):
    ''' Calculate a whitening matrix for covariance matrix sigma. '''
    U, l, _ = linalg.svd(sigma)
    return U.dot( np.diag(l ** -0.5) )


def csp(trials_l, trials_r):
    '''
    Calculate the CSP transformation matrix W.
    arguments:
        trials_r - Array (channels x samples x trials) containing right hand movement trials
        trials_l - Array (channels x samples x trials) containing left hand movement trials
    returns:
        Mixing matrix W
    '''
    cov_r = cov(trials_r)
    cov_l = cov(trials_l)
    P = whitening(cov_r + cov_l)
    B, _, _ = linalg.svd( P.T.dot(cov_l).dot(P) )
    W = P.dot(B)
    return W

def load_and_process(file_path):
    raw_obj = mne.io.read_raw_edf(file_path, preload=True)

    if len(raw_obj.ch_names) > 8:
        raw_obj.pick_channels(raw_obj.ch_names[:8])

    raw_obj = raw_obj.load_data().filter(0.5, 45)
    raw_obj = raw_obj.set_eeg_reference('average')
    raw_obj = raw_obj.load_data().filter(8, 15)

    # To exclude any bad eeg data automatically from the epochs if there is any
    eeg_channel_inds = mne.pick_types(
        raw_obj.info,
        meg=False,
        eeg=True,
        stim=False,
        eog=False,
        exclude='bads',
    )
    
    # Extract the epochs from the dataset
    events = mne.events_from_annotations(raw_obj, event_id='auto')[0][:]
    epoched = mne.Epochs(
        raw_obj,
        events,
        dict(left=5, right=6), # Class labels we're interested in
        tmin=0.5,
        tmax=2.5,
        proj=False,
        # picks=eeg_channel_inds,
        baseline=None,
        preload=True
    )

    # Calculate the CSP filter
    X_ML = np.transpose((epoched.get_data() * 1e3).astype(np.float32), (1, 2, 0))
    y = (epoched.events[:, 2] - 5).astype(np.int64)
    mask_0 = y==0
    mask_1 = y==1
    W = csp(X_ML[:,:,mask_0], X_ML[:,:,mask_1])

    print(W.shape)
    with open(f"{sourceDic}/models/W_CSP.pkl", 'wb') as f:
        pickle.dump([W], f)

class MyOVBox(OVBox):
   def __init__(self):
      OVBox.__init__(self)

   def initialize(self):
      # nop
      return

   def process(self):
        # Get location of this file to find path to models
        file_path = f"{sourceDic}/signals/IM_Acquisition_CSP.edf"
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(file_path)
        load_and_process(file_path)
        stimSet = OVStimulationSet(self.getCurrentTime(), self.getCurrentTime()+1./self.getClock())
        stimSet.append(OVStimulation(32770, self.getCurrentTime, 0.5))             
        self.output[0].append(stimSet)
        return

   def uninitialize(self):
      # nop
      return

box = MyOVBox()
