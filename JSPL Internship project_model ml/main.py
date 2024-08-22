import pandas as pd
import numpy as np
from RecoveryCalculator import RecoveryCalculator
from CalculateAlloys import CalculateAlloys
from tensorflow.keras.optimizers import Adam 

df = pd.read_excel(r"data_sheet.xlsx",sheet_name=1)   
obj=RecoveryCalculator(df)
df1=pd.read_excel(r"data_sheet.xlsx",sheet_name='INPUT') 
x,y=obj.load_and_preprocess_data(df1)
obj.model.compile(optimizer='Adam',loss='MSE',metrics='accuracy')
hist=obj.model.fit(x=x,y=y,batch_size=33,epochs=25) #
print("Training of Recovery Calculator model completed. Starting second phase of taining")      
calculator=CalculateAlloys(obj.model)
calculator.compile(Adam(learning_rate=0.0001))
calculator.fit(y,batch_size=1,epochs=100)
Mn=0.75
C=0.1
Si=0.08
S=0.021
Nb=0
Al=0.025
test=[Mn,C,Si,S,Nb,Al]
test=np.array(test,dtype=np.float32) 
output=calculator.predict(np.expand_dims(test,axis=0))
print(f"SiMn: {1000*output[0,0]} \n HCFeMn: {1000*output[0,1]}\n FeSi: {1000*output[0,2]}\n FeNb: {1000*output[0,3]}\n CPC: {1000*output[0,4]} \n Lime: {1000*output[0,5]} \n Al: {1000*output[0,6]}")
calculator.save(r"C:\Users\anura\OneDrive\Desktop\clutter file (1)\Prach_Pranesh_JSPL_project")