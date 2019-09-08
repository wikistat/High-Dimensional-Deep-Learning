import pandas as pd
import numpy as np

def my_read_csv(filename):
    return pd.read_csv(filename, delim_whitespace=True, header=None)

def load_signal(dataset, signal):
    filename = dataset+'/Inertial Signals/'+signal+'_'+dataset+'.txt'
    x = my_read_csv(filename).as_matrix()
    return x 

def load_signals(dataset, signals):
    signals_data = []
    for signal in signals:
        signals_data.append(load_signal(dataset, signal)) 
    X = np.transpose(signals_data, (1, 2, 0))
        
    return X 

def load_y(dataset = "train"):
    filename = dataset+'/y_'+dataset+'.txt'
    y = my_read_csv(filename)[0]
    Y = y.as_matrix()
    return Y