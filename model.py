from numpy.lib.function_base import gradient
from layer import Convolutional, Pooling, FullyConnected, Dense, regularized_cross_entropy, lr_schedule
from inout import plot_sample, plot_learning_curve, plot_accuracy_curve, plot_histogram
import numpy as np
import time
from hogwild_server import *
from config import *
import random
 

class Network:
    def __init__(self) :
        self.layers=[]
      
        self.server=DATA_CHUNKS_HANDLER.getInstance()
        
        
    def add_layer(self,layer):
        self.layers.append(layer)
    
    
    def build_model(self,dataset_name):
        if dataset_name == 'mnist':
            self.add_layer(Convolutional(name='conv1', num_filters=8, stride=1, size=3, activation='xxrelu'))
            self.add_layer(Pooling(name='pool0', stride=2, size=2))
            self.add_layer(Dense(name='dense', nodes=8 * 13 * 13, num_classes=10))

            # self.add_layer(Convolutional(name='conv1', num_filters=6, stride=1, size=3, activation='relu'))
            # self.add_layer(Pooling(name='pool1', stride=2, size=2))
            # self.add_layer(Convolutional(name='conv2', num_filters=16, stride=1, size=3, activation='relu'))
            # self.add_layer(Pooling(name='pool2', stride=2, size=2))
            # self.add_layer(Dense(name='dense', nodes=400, num_classes=10))
            total_weights=0
            for i in range(len(self.layers)):
                w_layer_i=len(self.layers[i].get_weights())
                #print("layer w_ {0}".format(w_layer_i))
                total_weights+=w_layer_i
                
            rand=np.random.rand(total_weights)*0.1
            self.server = DATA_CHUNKS_HANDLER.getInstance()
            self.server.set_init_weight(rand)

                    
        else:
            self.add_layer(Convolutional(name='conv1', num_filters=32, stride=1, size=3, activation='relu'))
            self.add_layer(Convolutional(name='conv2', num_filters=32, stride=1, size=3, activation='relu'))
            self.add_layer(Pooling(name='pool1', stride=2, size=2))
            self.add_layer(Convolutional(name='conv3', num_filters=64, stride=1, size=3, activation='relu'))
            self.add_layer(Convolutional(name='conv4', num_filters=64, stride=1, size=3, activation='relu'))
            self.add_layer(Pooling(name='pool2', stride=2, size=2))
            self.add_layer(FullyConnected(name='fullyconnected', nodes1=64 * 5 * 5, nodes2=256, activation='relu'))
            self.add_layer(Dense(name='dense', nodes=256, num_classes=10))
    def forward(self,image,plot_feature_maps):
        global history
        for layer in self.layers:
        
            image=layer.forward(image)
        return image
    def backward(self,gradient,learning_rate):
        
        for layer in reversed(self.layers):
#            print('\nbackward class {0}'.format(layer.name))
            gradient=layer.backward(gradient,learning_rate)
            weight=layer.get_weights()
            #print('\nname={0}   weight {1}'.format(layer.name,weight.shape))
#           print('\nbackward  class gradient {0}'.format(gradient))
    #    print('\nbackward gradient {0}'.format(gradient))
        
    
    def train(self,type_n_thread,dataset,num_epochs,learning_rate,validate,regularization,plot_weights,verbose):    
        
        batch_size=1000

        is_show_debug_info=False
        # if 0==type_n_thread:
        #     is_show_debug_info=True

        #print("THREAD[{0}] start to train".format(type_n_thread))

        _time_stamp=self.server.read_time_stamp()
        (current_model_weights,list_time_stamp)=self.server.server_util_get_weights()
        
        t1=time.time()
        self.set_weights_for_layer(current_model_weights)
        if True==is_show_debug_info:
            print("THREAD[{0}]  set weight  in ".format(type_n_thread,time.time()-t1))

        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        for epoch in range(0, num_epochs ):
            print('\n--- Epoch {0} ---'.format(epoch))
            #print(list_time_stamp)
            
            permutation = np.random.permutation(len(dataset["train_images"]))
            train_images=dataset["train_images"]
            train_labels=dataset["train_labels"]
            train_images = train_images[permutation]
            train_labels = train_labels[permutation]

            train_images=train_images[0:batch_size]
            train_labels=train_labels[0:batch_size]
            
            
            loss = 0
            num_correct = 0
            initial_time = time.time()
            for i ,(image,label) in enumerate(zip(train_images,train_labels)):
                if i % 100 == 99:
                    executed_time=time.time()-initial_time
                    initial_time= time.time()

                    print('[Step %d] Past 100 steps [%.fs]: Average Loss %.3f | Accuracy: %d%%' %(i + 1,executed_time, loss / 100, num_correct))
                    loss = 0
                    num_correct = 0

                    history['loss'].append(loss)                # update history
                    history['accuracy'].append(num_correct/100)
                    

                    start_update_time=time.time()-initial_time
                    
                    self.update_weights_to_server(type_n_thread,_time_stamp)
                    end_update_time=time.time()-initial_time

                    if True==is_show_debug_info:
                        print("THREAD[{0}] step:{1} update done after {2}".format(type_n_thread,i+1,end_update_time-start_update_time))
                    
                    _time_stamp=self.server.read_time_stamp()
                    (current_model_weights,list_time_stamp)=self.server.server_util_get_weights()
                    print(list_time_stamp)
                    t2=time.time()
                    self.set_weights_for_layer(current_model_weights)
                    if True==is_show_debug_info:
                        print("THREAD[{0}]  step:{1} set weight  in {2}".format(type_n_thread,i+1,time.time()-t2))

                #image=np.pad(image, ((0,0),(2,2),(2,2)),constant_values=0)
                t3=time.time()
                tmp_output = self.forward(image, plot_feature_maps=0)       # forward propagation
                if True==is_show_debug_info:
                    print("THREAD[{0}]  step:{1} foward  in {2}".format(type_n_thread,i+1,time.time()-t3))
                l = -np.log(tmp_output[label])
                acc = 1 if np.argmax(tmp_output) == label else 0
                # compute (regularized) cross-entropy and update loss
                #tmp_loss += regularized_cross_entropy(self.layers, regularization, tmp_output[label])
                loss += l
                num_correct += acc

                gradient = np.zeros(10)
                gradient[label] = -1 / tmp_output[label]
                
                #gradient = np.zeros(10)
                                                     # compute initial gradient
                # gradient[label] = -1 / tmp_output[label] + np.sum(
                #     [2 * regularization * np.sum(np.absolute(layer.get_weights())) for layer in self.layers])

                learning_rate = lr_schedule(learning_rate, iteration=i)     # learning rate decay
                t4=time.time()
                self.backward(gradient, learning_rate)                      # backward propagation
                if True==is_show_debug_info:
                    print("THREAD[{0}] step{1} backward  in {2}".format(type_n_thread,i+1,time.time()-t4))
    
            print("loss  thread:{0}   epoch:{1}  loss:{2}".format(type_n_thread,epoch,loss))
            plotting_info["loss_vals"][type_n_thread][epoch].append(loss)
       
    
    def evaluate(self,X,y,regularization,plot_correct,plot_missclassified,plot_feature_maps,verbose):
        loss,num_correct=0,0
        for i in range(len(X)):
            tmp_output=self.forward(X[i],plot_feature_maps)

            loss+=regularized_cross_entropy(self.layers,regularization,tmp_output[y[i]])

            prediction =np.argmax(tmp_output)

            if prediction ==y[i]:
                num_correct +=1

                # if plot_correct:
                #     image=(X[i]*255)[0,:,:]
                #     plot_sample(image,y[i],prediction)
                #     plot_correct=1
                # else:
                #     if plot_missclassified:
                #         image=(X[i]*255)[0,:,:]
                #         plot_sample(image,y[i],prediction)
                #         plot_missclassified=1
            
        test_size=len(X)
        accuracy =(num_correct/test_size)*100
        loss = loss /test_size

        # if verbose:
        #     print('Test Loss: %02.3f' % loss)
        #     print('Test Accuracy: %02.3f' % accuracy)
        return loss,accuracy

    def get_layer_weights(self):
        weights=[]
        for i in range(0,self.layers):
            weights+=self.layers[i].get_weights()
        return weights

    def update_weights_to_server(self,n_th_thread,time_stamp):


        weights=[]
        for i in range(0,len(self.layers)):
            layer=self.layers[i]
            weight=layer.get_weights().tolist()
            weights+=weight

        array_weight=np.array_split(weights,NUMBER_OF_MODEL_SPLITED)
        
        chunks=[]
        
        for j in range(0,len(array_weight)):
            chunks.append(DATA_CHUNKS_HANDLER.DATA_CHUNK(array_weight[j]))
        
        self.server.update_data_chunks_thread(n_th_thread,chunks,time_stamp)        
        
            

        
    def update_new_chunk(self,chunk_model,index):
        if len(self.chunk_weights ) > index:
            self.chunk_weights[index].update_chunk_with_new_model(chunk_model)
    
    def set_weights_for_layer(self,weights):
        
        array_weights=[]
        
        start_index=0
        end_index=0

        for i in range(0,len(self.layers)):
            end_index+=len(self.layers[i].get_weights())
            www=weights[start_index:end_index]
            array_weights.append(www)
            start_index=end_index

            self.layers[i].set_weights(www)
        
        