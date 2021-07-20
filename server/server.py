# implement json
# @dataclass 
# class distributed_model:
import ast
from collections import namedtuple
from dataclasses import dataclass
from json import JSONEncoder
import json
import numpy as np
from numpy.core.defchararray import array
from numpy.core.fromnumeric import argmax, shape



# empty_array=np.array([])
# empty_array=np.append(empty_array,[100,200,300])
# empty_array=np.append(empty_array,[8,3.200,30.90])
# print(empty_array)
# print(len(empty_array))
# print(empty_array[2:3])

# empty_array=np.insert(empty_array,2,[0.004,0.0012,0.0006])
# print("---------------")
# print(empty_array)
np.set_printoptions(precision=3)
@dataclass
class data_weight:
    i_from:int
    i_to:int 
    i_tau:int
    arr_weights:str

    def __init__(self,* args) :
        if 1==len(args):
           
            data_w:data_weight=json.loads(args[0],object_hook=lambda d: namedtuple('X',d.keys())(*d.values()) )
            self.i_from=data_w.i_from
            self.i_to=data_w.i_to
            self.i_tau=data_w.i_tau
            
            list=[float(i) for i in data_w.arr_weights.split()]           
            self.arr_weights=''.join(str(x)+" " for x in list)

        else:
            self.i_from = args[0]
            self.i_to=args[1]
            self.i_tau=args[2]
            #self.arr_weights='0'
            self.arr_weights=''.join(str(x)+" " for x in args[3])
        
    def f_convert_to_json(self):
        return json.dumps(self.__dict__)
        
    
    def f_test_val_type(self):
        text="{0} {1}".format(self.i_from , type(self.i_from))
        text+="\n{0} {1}".format(self.i_to , type(self.i_to))
        text+="\n{0} {1}".format(self.i_tau , type(self.i_tau))
        text+="\n{0} {1}".format(self.arr_weights , type(self.arr_weights))
        print(text)
    def get_list_arr(self):
        listx= [float(i) for i in self.arr_weights.split()]
        return listx

@dataclass
class my_model:
    num_of_chunk:int
    num_elements_each_chunk:int
    my_weights_in_str:str
    my_weights_in_array:np.array
    my_real_model:np.ndarray
    def __init__(self,* args) :
        if 1 == len(args):
            pass
        else:
            self.num_elements_each_chunk=args[0]
            self.my_weights_in_str=''.join(str(x) +" " for x in args[1])
            self.my_weights_in_array=[float(i) for i in self.my_weights_in_str.split()]  
            self.num_of_chunk=round(len(self.my_weights_in_array)/self.num_elements_each_chunk)
            self.my_real_model=np.ndarray(shape=(self.num_of_chunk,self.num_elements_each_chunk),dtype=float)
            
            num_of_padding=self.num_of_chunk * self.num_elements_each_chunk-len(self.my_weights_in_array)
            padding=np.zeros(shape=(1,num_of_padding),dtype=float)
            self.my_weights_in_array.append(padding)

            self.my_real_model=np.reshape(self.my_weights_in_array,(self.num_of_chunk,self.num_elements_each_chunk))
            print(self.my_real_model.shape)
            print(self.my_real_model)
            



dinh_model=my_model(31,np.random.uniform(-10,10,123))