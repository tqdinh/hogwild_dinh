from threading import Thread
from typing import Dict

from model import Network
from inout import load_mnist, load_cifar, preprocess
import numpy as np
from config import *
import matplotlib.pyplot as plt
from hogwild_server import *
from tensorflow import keras
import tensorflow as tf
from multiprocessing import Process

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
        - plot_feature_maps         > 0 --> plot feature maps of predicted digits from test set
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


def run_train(n_th_thread,_trains,_num_epochs,_learning_rate,_validate,_regularization,_plot_weights,_verbose):
    _model=Network()
    _model.build_model(dataset_name)
    _model.train(
        n_th_thread,
        _trains,
    _num_epochs,
    _learning_rate,
    _validate,
    _regularization,
    _plot_weights,
    _verbose)


def run_valid(dataset, regularization,plot_correct,plot_missclassified,plot_feature_maps,verbose):
    indices=np.random.permutation(dataset['validation_images'].shape[0])
    val_loss,val_accuracy=model.evaluate(validation_images[indices, :],
                            validation_labels[indices],
                            regularization,
                            plot_correct=0,
                            plot_missclassified=0,
                            plot_feature_maps=0,
                            verbose=0)
    print('Test Loss: %02.3f' % val_loss)
    print('Test Accuracy: %02.3f' % val_accuracy)


if __name__ == "__main__":


    for i_type in range(len(plotting_info["n_thread_type"])):
        n_threads=plotting_info["n_thread_type"][i_type]
        print('\n--- Loading ' + dataset_name + ' dataset ---')                 # load dataset
        dataset = load_mnist() if dataset_name == 'mnist' else load_cifar()

        # print('\n--- Processing the dataset ---')                               # pre process dataset
        # dataset = preprocess(dataset)


        print('\n--- Building the model ---')                                   # build model


        train_images=dataset['train_images']
        train_lables=dataset['train_labels']

        trains_images_array=np.array_split(train_images,n_threads)
        trains_lables_array=np.array_split(train_lables,n_threads)
        trains=[]


        for i in range(len(trains_images_array)):
            trains.append({'train_images':trains_images_array[i], 'train_labels':trains_lables_array[i]})
            


        #print(type(dataset))
        validation_images=dataset['validation_images']
        validation_labels=dataset['validation_labels']



        print('\n--- Training the model ---')                                   # train model

        # models=[]
        # for _ in range(0,n_threads):
        #     model = Network()
        #     model.build_model(dataset_name)
        #     models.append(model)



        time.sleep(2)
        print("create {0} threads that run {1} epochs".format(n_threads,NUMBER_OF_EPOCH))
        threads=[]
        for i in range(0,n_threads):
            thread=Process(target=run_train,args=( i,trains[i],
                NUMBER_OF_EPOCH,
                learning_rate,
                validate,
                regularization,
                plot_weights,
                verbose))
                
            time_to_sleep=np.random.randint(1,RANDOM_SLEEP_TIME)*0.2
            time.sleep(time_to_sleep)
            thread.start()
            threads.append(thread)
        for i in range(0,n_threads):
            threads[i].join()
        
        # indices=np.random.permutation(dataset['validation_images'].shape[0])
        # val_loss,val_accuracy=models[i].evaluate(validation_images[indices, :],
        #                         validation_labels[indices],
        #                         regularization,
        #                         plot_correct=0,
        #                         plot_missclassified=0,
        #                         plot_feature_maps=0,
        #                         verbose=0)
        # print('Valid Loss: %02.3f' % val_loss)
        # print('valid Accuracy: %02.3f' % val_accuracy)

        
        

    # ‘b’	blue
    # ‘g’	green
    # ‘r’	red
    # ‘c’	cyan
    # ‘m’	magenta
    # ‘y’	yellow
    # ‘k’	black
    # ‘w’	white

    color_val=['b','g','r','c','m','y']

    for i in range(0,len(plotting_info["n_thread_type"])):
        n_threads=plotting_info["n_thread_type"][i]
        print("in type {0} thread".format(n_threads))
        epoch_type_val=[]
        for j in range(NUMBER_OF_EPOCH):
            loss_of_thread_in_epoch_nths=np.sort(plotting_info["loss_vals"][i][j])
            min_val_on_epoch=loss_of_thread_in_epoch_nths[0]
            epoch_type_val.append(min_val_on_epoch)
            print("epoch {0} thread min vallue = {1} ".format(j,min_val_on_epoch))

        color=color_val[i%len(color_val)]
        line_lable="n_thread = {0}".format(n_threads)

        plt.plot(epoch_type_val, color, linewidth=1.0, label=line_lable)    
        plt.xlabel('epoch', fontsize=16)
        plt.ylabel('Loss', fontsize=16)
        plt.legend()
    plt.show()
            


    # model.train(
    #     dataset,
    #     num_epochs,
    #     learning_rate,
    #     validate,
    #     regularization,
    #     plot_weights,
    #     verbose
    # )




    # model.train(
    #     dataset,
    #     num_epochs,
    #     learning_rate,
    #     validate,
    #     regularization,
    #     plot_weights,
    #     verbose
    # )
    # model.update_weights_to_server()

