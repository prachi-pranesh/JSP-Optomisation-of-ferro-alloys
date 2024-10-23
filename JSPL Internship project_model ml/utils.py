import numpy as np

def load_and_preprocess_data(df1):
    X=list(np.concatenate((exp_dim(df1['SiMn ']),exp_dim(df1['HCFeMn']),exp_dim(df1['FeSi']),exp_dim(df1['FeNb']),exp_dim(df1['CPC']),exp_dim(df1['Lime']),exp_dim(df1['Al wire '])),axis=1)) #Getting the inputs
    Y=list(np.concatenate((exp_dim(df1['Mn']),exp_dim(df1['C']),exp_dim(df1['Si']),exp_dim(df1['S']),exp_dim(df1['Nb']),exp_dim(df1['Al'])),axis=1))#getting the outputs
    X1=list()
    X1.append(X[0])
    n =len(X)
    suma=1
    heat=df1['heat number ']
    heat=heat.isnull().values
    for i in range(1,n):
            if(not heat[i]):
                suma=0
            X1.append(X[i]+suma*X1[i-1])
            suma=1
        
    X1=np.array(X1[1:],dtype=np.float32)/1000.0
    Y1=np.array(Y[1:],dtype=np.float32)
    
    return [X1,Y1]

def exp_dim(col):
    return (np.expand_dims(col.values,axis=1))