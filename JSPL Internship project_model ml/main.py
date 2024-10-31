import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from utils import load_and_preprocess_data

df = pd.read_excel(r"data_sheet.xlsx",sheet_name=1)   
y,x=load_and_preprocess_data(df)

def Linearfit(X,Y):
    regr = LinearRegression()   
    regr.fit(X, Y) 
    return regr

SiMn_pred=Linearfit(x,y[:,0])
HCFeMn_pred=Linearfit(x,y[:,1])
FeSi_pred=Linearfit(x,y[:,2])
FeNb_pred=Linearfit(x,y[:,3])
CPC_pred=Linearfit(x,y[:,4])
Lime_pred=Linearfit(x,y[:,5])
Al_pred=Linearfit(x,y[:,6])

def calculate_alloys(Mn0,C0,Si0,S0,Nb0,Al0):
    Mn=1.3-Mn0
    C=0.17-C0
    Si=0.17-Si0
    S=0.003-S0
    Nb=0.02-Nb0
    Al=0.03-Al0
    test=[Mn,C,Si,S,Nb,Al]
    test=np.array(test,dtype=np.float32)
    x_t=np.expand_dims(test,axis=0)
    SiMn=SiMn_pred.predict(x_t)
    HCFeMn=HCFeMn_pred.predict(x_t)
    FeSi=FeSi_pred.predict(x_t)
    FeNb=FeNb_pred.predict(x_t)
    CPC=CPC_pred.predict(x_t)
    Lime=Lime_pred.predict(x_t)
    Al=Al_pred.predict(x_t)
    print(f"SiMn: {1000*SiMn} \n HCFeMn: {1000*HCFeMn}\n FeSi: {1000*FeSi}\n FeNb: {1000*FeNb}\n CPC: {1000*CPC} \n Lime: {1000*Lime} \n Al wire: {1000*Al}")
print("Example:")  
calculate_alloys(0.75,0.1,0.08,0.021,0,0.025)
