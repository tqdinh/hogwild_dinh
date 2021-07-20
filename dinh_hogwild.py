import copy
from threading import Lock
import numpy as np
from itertools import tee
from dataclasses import dataclass
from random import seed
from random import random
from threading import Thread
import threading
# seed random number generator
seed(1)
import copy
from random import randint

import time
import warnings
from numpy.core.fromnumeric import shape
from prompt_toolkit.layout.dimension import D
from tensorflow.python.keras.layers.merge import dot
from tensorflow.python.ops.numpy_ops.np_array_ops import empty
from torch.onnx.symbolic_opset9 import detach
from zmq.sugar.constants import NULL
warnings.filterwarnings("ignore")
from sklearn.datasets import load_boston
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from numpy import random
from sklearn.model_selection import train_test_split
from threading import Thread, Lock
print("DONE")


NUM_OF_SUB_MODEL=10
NUM_OF_CLIENT=10
DELTA_TIME_STAMP=500

class CHUNK:
    chunk_weights=[]
    time_stamp=0;
    lock:Lock
    def __init__(self,_chunk_weight,_time_stamp):
        self.chunk_weights=copy.copy(_chunk_weight)
        self.time_stamp=_time_stamp
        self.lock=Lock()
    def update_chunk(self,new_chunk):
        self.lock.acquire()
        if new_chunk.time_stamp > (self.time_stamp-DELTA_TIME_STAMP):
            self.chunk_weights=copy.copy(new_chunk.chunk_weights)
            self.time_stamp=new_chunk.time_stamp
        self.lock.release()
    
    def update_chunk_data(self,_chunk_weight,_time_stamp):
        self.lock.acquire()
        if _time_stamp > self.time_stamp:
            self.chunk_weights=copy.copy(_chunk_weight)
            self.time_stamp=_time_stamp

            print("update chunk ",self.chunk_weights[0:3]," ---- ",self.time_stamp)

        self.lock.release()
    
    def update_chunk_with_new_model(self, new_chunk):
        Thread(target=self.update_chunk_data,args=(new_chunk.chunk_weights,new_chunk.time_stamp)).start()
    
    def update_model_time_stamp(self,_chunk_weight,_time_stamp):
        if _time_stamp > self.time_stamp:
            self.chunk_weights=copy.copy(_chunk_weight)
            self.time_stamp=_time_stamp

    def get_weights(self):
        return self.chunk_weights
        
class MY_COORDINATOR:

    def __init__(self,total_wight): 
        self.chunks : CHUNK =[]
        # 72 72 2880 
        model_weights=np.zeros(shape=(1,total_wight),dtype=float)
        array_weights=np.array_split(model_weights[0],NUM_OF_SUB_MODEL)
        for i in range(0,len(array_weights)):
            self.chunks.append(CHUNK(array_weights[i],0))
                
    
    def update_chunk_model(self,chunk_model,index):
        if len(self.chunks ) > index:
            self.chunks[index].update_chunk_with_new_model(chunk_model)

  
    def get_chunks_model(self):
        return copy.copy(self.chunks)

    def util_get_weights(self):
        ret=[]
        for i in range(0,len(self.chunks)):
            www=self.chunks[i].get_weights().tolist()
            ret=ret+www
        return ret


    def test_init(self):
        for i in range(0,len(self.chunks)):
            print(self.chunks[i].chunk_weights)
            print(self.chunks[i].time_stamp)
        

boston_data=pd.DataFrame(load_boston().data,columns=load_boston().feature_names)
boston_target=pd.DataFrame(load_boston().target)
Y=load_boston().target
X=load_boston().data
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3)
dinh_train_data=pd.DataFrame(x_train)
dinh_train_data['price']=y_train

dinh_train_data=np.array_split(dinh_train_data,NUM_OF_CLIENT)

global_model=np.zeros(boston_data.shape[1]+1,dtype=float)

print("X Shape: ",X.shape)
print("Y Shape: ",Y.shape)
print("X_Train Shape: ",x_train.shape)
print("X_Test Shape: ",x_test.shape)
print("Y_Train Shape: ",y_train.shape)
print("Y_Test Shape: ",y_test.shape)

# standardizing data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test=scaler.transform(x_test)

## Adding the PRIZE Column in the data
train_data=pd.DataFrame(x_train)
train_data['price']=y_train
train_data.head(3)

x_test=np.array(x_test)
y_test=np.array(y_test)

n_iter=100

my_model=MY_COORDINATOR(boston_data.shape[1]+1)


def DinhSGD(chunks,train_data_frame,learning_rate,n_iter,k, divideby):
    
    new_chunks=copy.copy(chunks)

    local_model=[]
    for i in range(0,len(chunks)):
        local_model+=chunks[i].get_weights().tolist()
    delta_tau=0
    
    arr_loss=[]
    arr_grad=[]
    arr_loss_i=[]

    cur_iter=1
    sleep_time=10/(randint(1,50))
    
    while( cur_iter<=n_iter):
        
        temp=train_data.sample(k)
        y=np.array(temp['price'])
        x=np.array(temp.drop('price',axis=1))
        new_one=np.ones(shape=(x.shape[0],1),dtype=float)
        x=np.hstack((new_one,x))
        
        w_grad=np.zeros(shape=(1,train_data_frame.shape[1]))
      
        loss =0
        
        for i in range(k):
            prediction=np.dot(local_model,x[i])
            w_grad=w_grad + (-2) * x[i]*              (y[i]-prediction)
            loss = (np.square(np.asscalar(prediction) -y[i] )).mean(axis=None)
         #   time.sleep(0.0001*sleep_time)
            
        w_grad_tmp=w_grad/k
        local_model=local_model-learning_rate *(w_grad_tmp)
        
        delta_tau+=1

        arr_grad.append(w_grad)
        arr_loss.append(loss)
        arr_loss_i.append(cur_iter)

        cur_iter=cur_iter+1

        learning_rate=learning_rate /divideby


    end_timestamp=time.time()
    #print("total time = ",(end_timestamp-start_timestamp)*10)    
    
   
    local_model=list(local_model)
      
    array_local_model=np.array_split(local_model[0],len(new_chunks))
    for i in range(0,len(array_local_model)):
        new_chunks[i].update_model_time_stamp(array_local_model[i],new_chunks[i].time_stamp+delta_tau)
  
    return local_model,new_chunks

def read_data():
    ret=False
    
    if len(dinh_train_data)>0:
        ret= dinh_train_data.pop()
    return ret


def predict(x,w):
    new_one=np.ones(shape=(x.shape[0],1),dtype=float)
    x=np.hstack((new_one,x))
    y_prd=[]
    for i in range(len(x)):
        y=np.asscalar(np.dot(w,x[i]))
        y_prd.append(y)
    return np.array(y_prd)


def thread_read_data_cal_grad(thread_index,sleep_explicit):
    
    my_model.util_get_weights()
    
    my_chunks=my_model.get_chunks_model()
    while True:
        _data_frame=read_data()
        if  isinstance(_data_frame, pd.DataFrame):
            new_weight,new_chunks=DinhSGD(my_chunks,_data_frame,learning_rate=0.001,n_iter=200,divideby=1.001,k=50)
        else:
            break    
        for i in range(0,len(new_chunks)):
            my_model.update_chunk_model(new_chunks[i],i)
        print("thread {0} update".format(thread_index)) 


if __name__ == "__main__":
    threads=[]
    for i in range(0,NUM_OF_CLIENT):
        threads.append(Thread(target=thread_read_data_cal_grad,args=(i,1) ))
        threads[i].start()
    for i in range(0,NUM_OF_CLIENT):
        threads[i].join()
    new_weight=my_model.util_get_weights()
    y_pred_customsgd=predict(x_test,new_weight)    
    print('Mean Squared Error :',mean_squared_error(y_test, y_pred_customsgd))    
    
    
   
