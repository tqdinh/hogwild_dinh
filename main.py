from threading import Thread
from typing import Dict
from dinh_hogwild import NUM_OF_CLIENT
from model import Network
from inout import load_mnist, load_cifar, preprocess
import numpy as np
import time


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
num_epochs = 10
learning_rate = 0.01
validate = 0
regularization = 0
verbose = 0
plot_weights = 0
plot_correct = 0
plot_missclassified = 0
plot_feature_maps = 0

print('\n--- Loading ' + dataset_name + ' dataset ---')                 # load dataset
dataset = load_mnist() if dataset_name == 'mnist' else load_cifar()

print('\n--- Processing the dataset ---')                               # pre process dataset
dataset = preprocess(dataset)


print('\n--- Building the model ---')                                   # build model
model = Network()
model.build_model(dataset_name)

train_images=dataset['train_images']
train_lables=dataset['train_labels']

trains_images_array=np.array_split(train_images,NUM_OF_CLIENT)
trains_lables_array=np.array_split(train_lables,NUM_OF_CLIENT)
trains=[]


for i in range(len(trains_images_array)):
    trains.append({'train_images':trains_images_array[i], 'train_labels':trains_lables_array[i]})
    


print(type(dataset))
validation_images=dataset['validation_images']
validation_labels=dataset['validation_labels']



print('\n--- Training the model ---')                                   # train model

def run_train(_trains,_num_epochs,_learning_rate,_validate,_regularization,_plot_weights,_verbose):
    model.train(
    _trains,
    _num_epochs,
    _learning_rate,
    _validate,
    _regularization,
    _plot_weights,
    _verbose)

    model.update_weights_to_server()

threads=[]
for i in range(0,NUM_OF_CLIENT):
    thread=Thread(target=run_train,args=( trains[i],
        num_epochs,
        learning_rate,
        validate,
        regularization,
        plot_weights,
        verbose))
    #time.sleep(i*0.2)
    thread.start()
    threads.append(thread)
# for i in range(0,NUM_OF_CLIENT):
#     threads[i].join()

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





# print('\n--- Testing the model ---')                                    # test model
# model.evaluate(
#     dataset['test_images'],
#     dataset['test_labels'],
#     regularization,
#     plot_correct,
#     plot_missclassified,
#     plot_feature_maps,
#     verbose
# )
