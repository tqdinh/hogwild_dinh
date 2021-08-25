import queue
from read_write_file import write_list
from threading import Thread
from typing import Dict

from numpy.core.defchararray import index
from model import Network
from inout import load_mnist, load_cifar, preprocess
import numpy as np
import ctypes as c
import matplotlib.pyplot as plt
from hogwild_server import *
from tensorflow import keras
import tensorflow as tf
from multiprocessing import shared_memory, Process, Manager,Queue
from multiprocessing import cpu_count, current_process
import multiprocessing as mp
import multiprocessing.managers as m
import matplotlib.pyplot as bar_plt

import os


from datetime import datetime
import shutil




# for i in range(0,len(plotting_info["n_thread_type"])):
#     # share_mem=mp.Array(c.c_double, )
#     # plotting_info["loss_vals"].append([])
#     # # 1 3 5 10
#     # for j in range(NUMBER_OF_EPOCH):
#     #     plotting_info["loss_vals"][i].append([])
#     num_of_thread=plotting_info["n_thread_type"][i]
#     print("thread_type=",num_of_thread)
#     mp_arr = mp.Array(c.c_double, num_of_thread*NUMBER_OF_EPOCH)
#     arr = np.frombuffer(mp_arr.get_obj()) # mp_arr and arr share the same memory
# # make it two-dimensional
#     b = arr.reshape((num_of_thread,NUMBER_OF_EPOCH)) # b and arr share the same memory

        


'''
    Hyper parameters
    
        - dataset_name              choose between 'mnist' and 'cifar'
        - num_epochs                number of epochs
        - learning_rate             learning rate
        - validate                  0 -> no validation, 1 -> validation
        - regularization            regularization term (i.e., lambda)
        - verbose                   > 0 --> verbosity
        - plot_weights              > 0 --> plot weights distribution
        - plot_correct              > 0 --> plot correct predicted digits from test set
        - plot_missclassified       > 0 --> plot missclassified digits from test set
        - plot_featurea_maps         > 0 --> plot feature maps of predicted digits from test set
'''




dataset_name = 'mnist'

learning_rate = 10e-4
validate = 0
regularization = 0
verbose = 1
plot_weights = 0
plot_correct = 0
plot_missclassified = 0
plot_feature_maps = 0
_today=datetime.now()

#server=DATA_CHUNKS_HANDLER.getInstance()


def run_train(thread_index,n_threads,_trains,
        _num_epochs,_learning_rate,_validate,_regularization,
        _plot_weights,_verbose,list_of_loss_thread,_server):
    
    _model=Network(_server)
    _model.build_model(dataset_name)
    plot=_model.train(
        n_threads,
        _trains,
    _num_epochs,
    _learning_rate,
    _validate,
    _regularization,
    _plot_weights,
    _verbose)
    
    list_of_loss_thread[thread_index]=plot
    #valx.append(plot)
    print("Thread type {0} thread index{1}".format(n_threads, thread_index),plot)

    # plotting_info["loss_vals"][0][1].append([1,2,3,4])
    # plotting_info["loss_vals"][n_threads][1].append(plot)
    
class MyManager(m.BaseManager):
    pass

MyManager.register("DATA_CHUNKS_HANDLER", DATA_CHUNKS_HANDLER)




if __name__ == "__main__":

    folder_today="{0}".format(_today)
    if not os.path.exists(folder_today):
        os.makedirs(folder_today)

        original = "plot_my_graph.py"
        target = "{0}/{1}".format(folder_today,original)
        shutil.copyfile(original, target)

        original1 = "read_write_file.py"
        target1 = "{0}/{1}".format(folder_today,original1)
        shutil.copyfile(original1, target1)

   
    print('\n--- Loading ' + dataset_name + ' dataset ---')                 # load dataset
    dataset = load_mnist() if dataset_name == 'mnist' else load_cifar()

    dataset = preprocess(dataset)

    train_images=dataset['train_images']
    train_lables=dataset['train_labels']
    validation_images=dataset['validation_images']
    validation_labels=dataset['validation_labels']


    for i_type in range(len(plotting_info["n_thread_type"])):
        initial_time = time.time()
        manager = MyManager()
        manager.start()
        my_server = manager.DATA_CHUNKS_HANDLER(TAU[6])
    
        my_queue=Queue()
        my_queue_accuracy=Queue()
        
        n_threads=plotting_info["n_thread_type"][i_type]

        folder_name="{0}/n_thread_{1}".format(folder_today,n_threads)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        list_of_loss_thread=[]
        list_of_accuracy_thread=[]
        for j in range(n_threads):
            list_of_loss_thread.append([])
            list_of_accuracy_thread.append([])
        
        #print(list_of_loss_thread)
        
        print('\n--- Building the model ---')                                   # build model

        trains_images_array=np.array_split(train_images,n_threads)
        trains_lables_array=np.array_split(train_lables,n_threads)
        trains=[]
        for i in range(len(trains_images_array)):
            trains.append({'train_images':trains_images_array[i], 'train_labels':trains_lables_array[i]})
        

        print('\n--- Training the model ---')                                   # train model    
        print("create {0} processes that run {1} epochs".format(n_threads,NUMBER_OF_EPOCH))
        processes=[]
        
        for i in range(0,n_threads):
            print("CREATE_thread",i)
            
            process=Network(i,i_type,trains[i],NUMBER_OF_EPOCH,
            learning_rate,validate,regularization,plot_weights,
            verbose,list_of_loss_thread,my_server,my_queue,my_queue_accuracy,folder_name)
            

            # time_to_sleep=np.random.randint(1,RANDOM_SLEEP_TIME)*0.2
            # time.sleep(time_to_sleep)
            process.start()

            processes.append(process)
        for i in range(0,n_threads):
            processes[i].join()
        
        for thread_index in reversed(range(0,n_threads)):
            list_of_loss_thread[thread_index]=my_queue.get()
            list_of_accuracy_threadoi[thread_index]=my_queue_accuracy.get()
    
        # k=my_server.read_time_stamp()
        # ws=my_server.server_util_get_weights()
        
        end_time=  time.time()
        execution_time=end_time-initial_time
        

        file_exe_time_name="{0}/time_execute".format(folder_name)
        write_list(file_exe_time_name,[execution_time])

        plotting_info["duration"].append(execution_time)

        # valid_model=Network(i,i_type,trains[i],NUMBER_OF_EPOCH,
        #         learning_rate,validate,regularization,plot_weights,
        #         verbose,list_of_loss_thread,my_server,my_queue)
        # valid_model.build_model(dataset_name)
        # valid_model.set_weights_for_layer(ws[0])

    

        # indices=np.random.permutation(dataset['validation_images'].shape[0])
        # val_loss,val_accuracy=valid_model.evaluate(validation_images[indices, :],
        #                         validation_labels[indices],
        #                         regularization,
        #                         plot_correct=0,
        #                         plot_missclassified=0,
        #                         plot_feature_maps=0,
        #                         verbose=0)
        # print('Valid Loss: %02.3f' % val_loss)
        # print('valid Accuracy: %02.3f' % val_accuracy)
        
        # val_acc="valid_loss={0} vallid_acc={1}  ".format(val_loss,val_accuracy)
        #plotting_info["info"].append(val_acc)
        
    
        tmp_loss_list_of_thread=[]
        for j in range(len(list_of_loss_thread)):
           tmp_loss_list_of_thread.append(list_of_loss_thread[j])
        tmp_loss_list_of_thread=np.array(tmp_loss_list_of_thread).T
        
        min_loss_epoch=[]
        for row in range(len(tmp_loss_list_of_thread)):
            min_loss_row=np.min(tmp_loss_list_of_thread[row])
            min_loss_epoch.append(min_loss_row)
        plotting_info["loss_vals"].append(min_loss_epoch)
        

        tmp_accuracy_list_of_thread=[]
        for j in range(len(list_of_accuracy_thread)):
           tmp_accuracy_list_of_thread.append(list_of_accuracy_thread[j])
        tmp_accuracy_list_of_thread=np.array(tmp_accuracy_list_of_thread).T
        
        mean_accuracy_epoch=[]
        for row in range(len(tmp_accuracy_list_of_thread)):
            mean_accuracy_row=np.mean(tmp_accuracy_list_of_thread[row])
            mean_accuracy_epoch.append(mean_accuracy_row)
        plotting_info["accuracy"].append(mean_accuracy_epoch)

        


    color_val=['b','g','r','c','m','y']
    for i in range(0,len(plotting_info["n_thread_type"])):
        n_threads=plotting_info["n_thread_type"][i]
        print("in type {0} thread".format(n_threads))
        epoch_type_val=plotting_info["loss_vals"][i]
        
        color=color_val[i%len(color_val)]
        #meta_info=plotting_info["info"][i]
        line_lable="n_thread = {0} ".format(n_threads)

        plt.plot(epoch_type_val, color, linewidth=1.0, label=line_lable)    
        plt.xlabel('epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend()
        plt.title('Loss with learning rate scheduled step decay ', fontsize=16)
        plt.savefig('lr_schedule_step_decay_3.png')
    plt.show()


    color_val=['b','g','r','c','m','y']
    for i in range(0,len(plotting_info["n_thread_type"])):
        n_threads=plotting_info["n_thread_type"][i]
        print("in type {0} thread".format(n_threads))
        epoch_type_acc=plotting_info["accuracy"][i]
        
        color=color_val[i%len(color_val)]
        #meta_info=plotting_info["info"][i]
        line_lable="n_thread = {0} ".format(n_threads)

        plt.plot(epoch_type_acc, color, linewidth=1.0, label=line_lable)    
        plt.xlabel('Loss', fontsize=16)
        plt.ylabel('Accuracy (%)', fontsize=16)
        plt.legend()
        plt.title('accuracy with learning rate scheduled step decay ', fontsize=16)
        plt.savefig('lr_accuracy.png')
    plt.show()
            
    lable=[]
    val=[]
    for i_type in range(len(plotting_info["n_thread_type"])):
        n_threads=plotting_info["n_thread_type"][i_type]
        _lable="{0}_thread".format(n_threads)
        _val=plotting_info["duration"][i_type]
        lable.append(_lable)
        val.append(_val)

    # creating the bar plot
    bar_plt.bar(lable, val, color ='maroon',
            width = 0.4)
    bar_plt.xlabel("No. of process")
    bar_plt.ylabel("Time in sec")
    bar_plt.title("Time to execute  ")
    bar_plt.savefig('time_execute.png')
    bar_plt.show()

