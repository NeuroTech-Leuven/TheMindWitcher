import pickle

with open('../models/W_CSP.pkl', 'rb') as f:
    W = pickle.load(f)[0]