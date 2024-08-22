import tensorflow as tf
from keras.layers import Input, Dense, Concatenate
from keras.models import Model 




class CalculateAlloys(Model):
    def __init__(self,estimator):
        super(CalculateAlloys,self).__init__()
        self.estimator=estimator
        self.model=self.get_model()
    def get_model(self):
        Composition=Input(shape=(6))
        d=Dense(20,activation="LeakyReLU")(Composition) 
        d=Dense(100,activation="LeakyReLU")(d)
        d=Dense(500,activation="LeakyReLU")(d)
        d=Dense(100,activation="LeakyReLU")(d)
        out=Dense(7,activation='relu')(d) #This layer gives the output as (Mn_e,C_e,Si_e,S,p)
        return Model(Composition,out)
    def compute_grade_loss(self,out,inp):
        #out=tf.split(out,7,axis=1)
        estimate=self.estimator(out)
        estimate=tf.split(estimate,6,axis=1)
        inp=tf.split(inp,6,axis=1)
        loss_Mn=(estimate[0] + inp[0]-1.325)**2
        loss_C=(estimate[1]+inp[1]-0.18)**2
        loss_Si=(estimate[2]+inp[2]-0.195)**2
        loss_S=(estimate[3]+inp[3])**2
        loss_Nb=(estimate[4]+inp[4]-0.0225)**2
        loss_Al=(estimate[5]+inp[5]-0.0275)**2
        return 10*loss_Mn+ loss_C+1.5*loss_Si+loss_S+loss_Nb+loss_Al
    def compute_cost(self,out):
        out=tf.split(out,7,axis=1)
        cost=(0.42663*out[4] + 0.77284*out[0] + 0.73028*out[1] + 7.14*out[3] + 0.1*out[5] + 0.85*(out[2]  + out[6]) - 6.5)**2
        return cost
    
    def compile(self,optimizer):
        super(CalculateAlloys,self).compile()
        self.optimizer=optimizer
    
    def call(self,x):
        return self.model(x)
    
    def train_step(self, data):
        x=data
        with tf.GradientTape() as tape:
            out=self.model(x)
            loss1=self.compute_grade_loss(out,x)
            loss2=self.compute_cost(out)
            loss=100*loss1 + 1e-9*loss2
        grad=tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        return {"Grade_loss":loss1, "Cost":loss2}
    

