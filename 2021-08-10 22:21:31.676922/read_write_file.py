import numpy as np
import json
#array_list_weights=np.random.uniform(-100,100,10)

# rand_float=np.random.uniform(-50,49)
# org_list=range(-50,50)
# array_list_weights=np.array(org_list)+rand_float
# array_list_weights=np.reshape(array_list_weights,newshape=(4,5,5))
# list_w=array_list_weights.tolist()

# with open("xloss/list_file","w") as save:
#     # for val in array_list_weights:
#     #     save.write("%f"%val)
#     json.dump(list_w,save)
# print(array_list_weights)
# dest=[]
# with open("xloss/list_file","r") as read_file:
#     # for line in read_file:
#     #     current=line[:-1]
#     #     dest.append(current)
#     dest=json.load(read_file)
# print(dest)

def write_list(file_name,_list):
    with open(file_name,"w") as save:
        json.dump(_list,save)

def read_list(file_name):
    ret=[]
    with open(file_name,"r") as read_file:
        ret=json.load(read_file)
    return ret
