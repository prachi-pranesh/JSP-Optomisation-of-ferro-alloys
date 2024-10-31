import numpy as np
import pandas as pd

def exp_dim(col):
    return (np.expand_dims(col.values,axis=1))

def load_and_preprocess_data(df1):
    X=list(np.concatenate((exp_dim(df1['SiMn']),exp_dim(df1['HCFeMn']),exp_dim(df1['FeSi']),exp_dim(df1['FeNb']),exp_dim(df1['CPC']),exp_dim(df1['Lime']),exp_dim(df1['Al wire'])),axis=1)) #Getting the inputs
    Y=list(np.concatenate((exp_dim(df1['Mn']),exp_dim(df1['C']),exp_dim(df1['Si']),exp_dim(df1['S']),exp_dim(df1['Nb']),exp_dim(df1['Al'])),axis=1))#getting the outputs
    X=np.array(X,dtype=np.float32)/1000.0
    Y=np.array(Y,dtype=np.float32)
    return [X,Y]
