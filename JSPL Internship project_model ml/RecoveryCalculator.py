import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Concatenate
from keras.models import Model 
#Inputs -> Manganese, Carbon, Silicon, SiMn, HCFeMn, LCFeMn.

class RecoveryCalculator:
    def __init__(self, df):
        mn_values=df['Mn']
        self.SiMn_Mn_efficiency=mn_values[0]/100.0
        self.HCFeMn_Mn_efficiency=mn_values[1]/100.0
        self.Mn_Mn_efficiency=mn_values[2]/100.0
        
        c_values=df['C']
        self.SiMn_C_efficiency=c_values[0]/100.0
        self.HCFeMn_C_efficiency=c_values[1]/100.0
        self.FeSi_C_efficiency=c_values[3]/100.0
        self.FeNb_C_efficiency=c_values[8]/100.0
        self.CPC_C_efficiency=c_values[9]/100.0
        
        si_values=df['Si']
        self.SiMn_Si_efficiency=si_values[0]/100.0
        self.HCFeMn_Si_efficiency=si_values[1]/100.0
        self.FeSi_Si_efficiency=si_values[3]/100.0
        self.FeNb_Si_efficiency=si_values[8]/100.0
        
        s_values=df['S']
        self.HCFeMn_S_efficiency=s_values[1]/100.0
        self.Mn_S_efficiency=s_values[2]/100.0
        self.FeSi_S_efficiency=s_values[3]/100.0
        self.FeNb_S_efficiency=s_values[8]/100.0
        self.CPC_S_efficiency=s_values[9]/100.0        
        
        p_values=df['P']
        self.SiMn_P_efficiency=p_values[0]/100.0
        self.HCFeMn_P_efficiency=p_values[1]/100.0
        self.FeSi_P_efficiency=p_values[3]/100.0
        self.FeNb_P_efficiency=p_values[8]/100.0
        
        self.model=self.get_model() 
    
    #Code which uses formula to get the recovered values
   
    def Mn_recovered(self, Lo_Manganese,SiMn,HCFeMn,Mn,ladle_weight,recovery):
        intermediate= (SiMn*self.SiMn_Mn_efficiency*recovery + HCFeMn*self.HCFeMn_Mn_efficiency*recovery + Mn*self.Mn_Mn_efficiency*recovery)/ladle_weight 
        return Lo_Manganese+intermediate
    
    def C_recovered(self, Lo_Carbon,SiMn,HCFeMn,FeSi,CBC,ladle_weight,recovery):
        intermediate= (SiMn*self.SiMn_C_efficiency*recovery + HCFeMn*self.HCFeMn_C_efficiency*recovery + FeSi*self.FeSi_C_efficiency*recovery + CBC*self.CBC_C_efficiency*recovery)/ladle_weight 
        return Lo_Carbon+intermediate
    
    def Si_recovered(self, Lo_Silicon,SiMn,HCFeMn,FeSi,FeNb,ladle_weight,recovery):
        intermediate= (SiMn*self.SiMn_Si_efficiency*recovery + HCFeMn*self.HCFeMn_Si_efficiency*recovery + FeSi*self.FeSi_Si_efficiency*recovery + FeNb*self.FeNb_Si_efficiency*recovery)/ladle_weight 
        return Lo_Silicon+intermediate
    
    def S_recovered(self, Lo_Sulphur,HCFeMn,Mn,FeSi,FeNb,ladle_weight,recovery):
        intermediate= ( HCFeMn*self.HCFeMn_S_efficiency*recovery + FeSi*self.FeSi_S_efficiency*recovery + Mn*self.Mn_S_efficiency*recovery + FeNb*self.FeNb_S_efficiency*recovery)/ladle_weight 
        return Lo_Sulphur+intermediate
    def P_recovered(self, Lo_Phosphorous,SiMn,HCFeMn,FeSi,FeNb,ladle_weight,recovery):
        intermediate= (SiMn*self.SiMn_P_efficiency*recovery + HCFeMn*self.HCFeMn_P_efficiency*recovery + FeSi*self.FeSi_P_efficiency*recovery + FeNb*self.FeNb_P_efficiency*recovery)/ladle_weight 
        return Lo_Phosphorous+intermediate
    
   # A function which defines the deep learning model
    def get_model(self):
        Alloy_values=Input(shape=(7)) #This layer is for taking as input the 6 values which we give to the ladle or whatever.(SiMn,HCFeMn,FeSi,Al,Mn,Lime)
        d=Dense(20,activation="LeakyReLU")(Alloy_values) # A layer of 20 neurons, which takes the Chemistry_values as input and processes them
        d=Dense(100,activation="LeakyReLU")(d)
        d=Dense(300,activation="LeakyReLU")(d)
        d=Dense(100,activation="LeakyReLU")(d)
        out=Dense(6,activation='relu')(d) #This layer gives the output as (Mn_e,C_e,Si_e,S,p)
        return Model(Alloy_values,out) #'Model' basically refers to the deep learning model. We are defining the inputs adn outputs of the model
    
    def calculate_error(self,SiMn,HCFeMn,FeSi,Al,Mn,Lime,grade,ladle_weight,recovery,Lo_Manganese,Lo_Carbon,Lo_Silicon,LCFeMn):
        chemistry_values=np.array([SiMn,HCFeMn,FeSi,Al,Mn,Lime],dtype=np.float32)/1000.0
        Mn_e,C_e,Si_e=self.model.predict(chemistry_values) #Give corresponding inputs to the model to get the output
        
        '''The model does not actually output the recovered values, instead it calculates how much error was made by the 
        formula which is used to estimate the recovered values. Thus, the model output (Mn_e for example) will need to 
        be added to the value which we get from the formula
        '''
        Mn_i=self.Mn_recovered(Lo_Manganese, SiMn, HCFeMn, LCFeMn, Mn, ladle_weight, recovery)
        C_i=self.C_recovered(Lo_Carbon, SiMn, HCFeMn, LCFeMn, ladle_weight, recovery)
        Si_i=self.Si_recovered(Lo_Silicon, SiMn, HCFeMn, LCFeMn, ladle_weight, recovery)
        
        Final_Mn = Mn_i + Mn_e
        Final_C = C_i + C_e
        Final_Si = Si_i + Si_e
        
        return Final_Mn, Final_C, Final_Si
    
    def load_and_preprocess_data(self,df1):
        X=list(np.concatenate((np.expand_dims(df1['SiMn '].values,axis=1),np.expand_dims(df1['HCFeMn'].values,axis=1),np.expand_dims(df1['FeSi'].values,axis=1),np.expand_dims(df1['FeNb'].values,axis=1),np.expand_dims((df1['CPC'].values),axis=1),np.expand_dims(df1['Lime'].values,axis=1),np.expand_dims((df1['Al wire '].values),axis=1)),axis=1)) #Getting the inputs
        Y=list(np.concatenate((np.expand_dims(df1['Mn'].values,axis=1),np.expand_dims(df1['C'].values,axis=1),np.expand_dims(df1['Si'].values,axis=1),np.expand_dims(df1['S'].values,axis=1),np.expand_dims(df1['Nb'].values,axis=1),np.expand_dims(df1['Al'].values,axis=1)),axis=1)) #getting the outputs
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
        
        print(len(X1),len(Y))
        X1=np.array(X1[1:],dtype=np.float32)/1000.0
        Y1=np.array(Y[1:],dtype=np.float32)
        
        #Further code on pre-processing shall be written when the data is completely tabularized     
        return [X1,Y1]
    
