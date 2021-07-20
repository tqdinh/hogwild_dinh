
from server.server import data_weight
from numpy import random
from threading import Lock, Thread
import copy
import time
from typing import List
import numpy as np

NUMBER_OF_DATA_SPLITED=5
NUMBER_OF_CLIENT=10
NUMBER_OF_MODEL_SPLITED=5
TAU=3


class DATA_CHUNKS_HANDLER:
    class DATA_CHUNK:
        def __init__(self,_weights):
            self.data=copy.copy(_weights)
            
            self.write_lock=Lock()
            self.read_lock=Lock()
            
        
        def write_data_with_time_stamp(self,_data,_time_stamp):
            self.write_lock.acquire()
            self.data=copy.copy(_data)
            
            self.write_lock.release() 

        def write_data(self,_data):
            self.write_lock.acquire()
            self.data=copy.copy(_data)
    
            self.write_lock.release() 
        def read_data(self):
            ret=[]
            while True: 
                if True != self.write_lock.locked:
                    ret= self.data
                    break
                else:
                    print("thread is block")
                time.sleep(0.001)
            return ret
    
    def __init__(self,_init_weight:List):
        self.time_stamp_lock=Lock()
        self.time_stamp=0
        self.chunks=[]
        _init_weight_splited=np.array_split(_init_weight,NUMBER_OF_MODEL_SPLITED)

        for i in range(0,len(_init_weight_splited)):
            self.chunks.append(self.DATA_CHUNK(_init_weight_splited[i]))
    
    def read_time_stamp(self):
        ret=any
        while True:
            if True != self.time_stamp_lock.locked:
                ret=self.time_stamp
                break
            time.sleep(0.001)

        return ret 
    
    def write_time_stamp_plus_one(self):
        self.time_stamp_lock.acquire()
        self.time_stamp +=1
        self.time_stamp_lock.release()

    def update_time_stamp(self,_time_stamp):
        self.time_stamp_lock.acquire()
        self.time_stamp =_time_stamp
        self.time_stamp_lock.release()
    
    def update_data_chunks_thread(self,_data_chunks,_time_stamp):
        # if isinstance(_data_chunks,DATA_CHUNKS_HANDLER):
    
        print("update submodel with timestamp ",_time_stamp)
        print(self.util_get_weights()[0:3])
        
        
        for i in range(0, len(_data_chunks)):
            current_time_stamp=self.read_time_stamp()
            if _time_stamp >= current_time_stamp - TAU:
                chunk_data=_data_chunks[i].read_data()
                self.chunks[i].write_data(chunk_data)
            else:
                print("DELTA TIME IS TOO BIG {0}  vs {1} - {2}".format(_time_stamp," vs  ",current_time_stamp))
        
        current_time_stamp=self.read_time_stamp()
        
        if _time_stamp >= current_time_stamp:
            self.update_time_stamp(_time_stamp+1)
        
        
    
    def util_get_weights(self):
        ret=[]
        for chunk in self.chunks:
            list_data=chunk.read_data()
            ret+=list_data.tolist()
        return ret

random_list=np.random.rand(100)*0.1
print(random_list)
server = DATA_CHUNKS_HANDLER(random_list)

def my_calculation(special_thread):
    global server
    if True == special_thread:
        time.sleep(20)
    print("read_model")
    my_weights=server.util_get_weights()
    my_time_stamp=server.read_time_stamp()
    print("read_time_stamp",my_time_stamp)
    num_of_iterator=10

    if True == special_thread:
        print("special thread")
    while num_of_iterator>=0:
        
        time_to_sleep=np.random.randint(1,10)*0.1
        
        time.sleep(time_to_sleep)
      
        for i in range(0,len(my_weights)):
            my_weights[i]=np.random.rand()
      
        num_of_iterator-=1

    array_data_weight=np.array_split(my_weights,NUMBER_OF_MODEL_SPLITED)
    chunks=[]
    for j in range (0,len(array_data_weight)):
        list_data=array_data_weight[j]
        chunks.append(DATA_CHUNKS_HANDLER.DATA_CHUNK(list_data))
    
    server.update_data_chunks_thread(chunks,my_time_stamp)


if __name__ == "__main__":
    
    for i in range(0,NUMBER_OF_CLIENT):
    
        Thread(target=my_calculation,args=(False,) ).start()
        time.sleep(i)*0.5
    Thread(target=my_calculation,args=(True,) ).start()
        