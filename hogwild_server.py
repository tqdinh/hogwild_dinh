

from numpy import random
from threading import Lock, Thread
import copy
import time
from typing import List
import numpy as np
from config import *



class DATA_CHUNKS_HANDLER:


    __instance = None
    @staticmethod 
    def getInstance():
      """ Static access method. """
      if DATA_CHUNKS_HANDLER.__instance == None:
         DATA_CHUNKS_HANDLER()
      return DATA_CHUNKS_HANDLER.__instance
    def __init__(self):
        
        self.time_stamp_lock=Lock()
        self.time_stamp=0
        self.chunks=[]
        """ Virtually private constructor. """
        if DATA_CHUNKS_HANDLER.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            DATA_CHUNKS_HANDLER.__instance = self

    class DATA_CHUNK:
        def __init__(self,_weights):
            self.data=copy.copy(_weights)
            
            self.private_time_stamp_chunk=0

            self.write_lock=Lock()
            self.read_lock=Lock()
            
        
        # def write_data_with_time_stamp(self,_data,_time_stamp):
        #     self.write_lock.acquire()
        #     self.data=copy.copy(_data)
        #     self.private_time_stamp_chunk=_time_stamp
        #     self.write_lock.release() 

        def chunk_write_data(self,_data):
            self.write_lock.acquire()
            self.data=copy.copy(_data)
            self.private_time_stamp_chunk=self.private_time_stamp_chunk+1
            print("chunk written with time stamp= ",self.private_time_stamp_chunk)
            self.write_lock.release() 
        def chunk_read_data(self):
            ret=[]
            while True: 
                if True != self.write_lock.locked:
                    ret= self.data
                    break
                else:
                    print("thread is block")
                
            return ret
            #return (ret,self.private_time_stamp_chunk)
        def chunk_read_time_stamp(self):
            ret=0
            while True: 
                if True != self.write_lock.locked:
                    ret= self.private_time_stamp_chunk
                    break
                else:
                    print("thread is block")
                
            return ret
            #return (ret,self.private_time_stamp_chunk)
    
    def set_init_weight(self,_init_weight:List):
        
        if len(self.chunks) >0:
            return 
        
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
    
    def update_data_chunks_thread(self,n_th_thread,_data_chunks,_time_stamp):
        print("\n---------THREAD[{0}]---------- update  with timestamp {1}".format(n_th_thread,_time_stamp))
        
        for i in range(0, len(_data_chunks)):
            current_time_stamp=self.read_time_stamp()
            print("THREAD[{0}] read server time stamp{1}".format(n_th_thread,current_time_stamp))
            if _time_stamp >= current_time_stamp - TAU:
                chunk_data=_data_chunks[i].chunk_read_data()
                chunk_time_stamp=_data_chunks[i].chunk_read_time_stamp()
                self.chunks[i].chunk_write_data(chunk_data)
                #time_to_sleep=np.random.randint(1,RANDOM_SLEEP_TIME)*0.5
                #time.sleep(time_to_sleep)
                #self.chunks[i].write_data_with_time_stamp(chunk_data,_time_stamp)
            else:
                print("DELTA TIME IS TOO BIG {0}  vs {1} - {2}".format(_time_stamp," vs  ",current_time_stamp))
        
        current_time_stamp=self.read_time_stamp()
        
        if _time_stamp >= current_time_stamp - TAU:
            if(_time_stamp>=current_time_stamp):
                self.update_time_stamp(_time_stamp+1)
            else:
                current_time_stamp=self.read_time_stamp()
                self.update_time_stamp(current_time_stamp+1)
        
        
        
        
    
    def server_util_get_weights(self):
        ret=[]
        private_time_stamp=[]
        for chunk in self.chunks:
            list_data=chunk.chunk_read_data()
            ret+=list_data.tolist()
            private_time_stamp.append(chunk.chunk_read_time_stamp())
        return (ret,private_time_stamp)



def my_calculation(special_thread,server):
    
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
    ___server=DATA_CHUNKS_HANDLER(np.random.rand(100)*0.1)    
    for i in range(0,NUMBER_OF_CLIENT):
    
        Thread(target=my_calculation,args=(False,___server) ).start()
        time.sleep(i*0.5)
    #Thread(target=my_calculation,args=(True,___server) ).start()
        