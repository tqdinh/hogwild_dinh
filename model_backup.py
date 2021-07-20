from numpy.lib.function_base import gradient
from layer import Convolutional, Pooling, FullyConnected, Dense, regularized_cross_entropy, lr_schedule
from inout import plot_sample, plot_learning_curve, plot_accuracy_curve, plot_histogram
import numpy as np
import time
from hogwild_server import *

 

class Network:
    def __init__(self) :
        self.delta_time=0
        self.prev_time_stamp=0
        self.layers=[]
        
        self.server=any
        
    def add_layer(self,layer):
        self.layers.append(layer)
    
    def build_model(self,dataset_name):
        if dataset_name == 'mnist':
            self.add_layer(Convolutional(name='conv1', num_filters=8, stride=2, size=3, activation='relu'))
            self.add_layer(Convolutional(name='conv2', num_filters=8, stride=2, size=3, activation='relu'))
            self.add_layer(Dense(name='dense', nodes=8 * 6 * 6, num_classes=10))
            
            total_weights=0
            for i in range(len(self.layers)):
                total_weights+=len(self.layers[i].get_weights())
                
            rand=np.random.rand(total_weights)*0.1
            self.server = DATA_CHUNKS_HANDLER(rand)
            

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
        
        for layer in self.layers:
         #   print('\nforwarding class {0}'.format(layer.name))
            if plot_feature_maps:
                image=(image*255)[0,:,:]
                plot_sample(image,None,None)
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
        
    
    def train(self,trains,num_epochs,learning_rate,validate,regularization,plot_weights,verbose):    
        self.delta_time=0
        self.set_weights_for_layer()
        history={'loss':[],'accuracy':[],'val_loss':[],'val_accuracy':[]}
        for epoch in range(1,num_epochs+1):
            print('\n--- Epoch{0}---'.format(epoch))
            loss,tmp_loss,num_corr=0,0,0
            initial_time=time.time()

            data_len=(int)(len(trains['train_images']))
            for i in range(data_len):
                if 0==i%50 :
                #if 1==1:
                    accuracy = (num_corr/(i+1))*100
                    loss=tmp_loss/(i+1)

                    history['loss'].append(loss)                # update history
                    history['accuracy'].append(accuracy)

                    # if validate:
                    #     indices=np.random.permutation(dataset['validation_images'].shape[0])

                    #     val_loss,val_accuracy=self.evaluate(dataset['validation_images'][indices, :],
                    #         dataset['validation_labels'][indices],
                    #         regularization,
                    #         plot_correct=0,
                    #         plot_missclassified=0,
                    #         plot_feature_maps=0,
                    #         verbose=0)
                        
                    #     history['val_loss'].append(val_loss)
                    #     history['val_accuracy'].append(val_accuracy)

                    #     if verbose:
                    #         print('[Step %05d]: Loss %02.3f | Accuracy: %02.3f | Time: %02.2f seconds | '
                    #               'Validation Loss %02.3f | Validation Accuracy: %02.3f' %
                    #               (i + 1, loss, accuracy, time.time() - initial_time, val_loss, val_accuracy)
                    #         )
                    #     elif verbose:
                    #         print('[Step %05d]: Loss %02.3f | Accuracy: %02.3f | Time: %02.2f seconds' %
                    #             (i + 1, loss, accuracy, time.time() - initial_time))

                    #     initial_time = time.time()

                    image = trains['train_images'][i]
                    lable = trains['train_labels'][i]

                    #print('--Forward---')

                    tmp_output = self.forward(image,plot_feature_maps=0)

                    tmp_loss+=regularized_cross_entropy ( self.layers, regularization,tmp_output[lable])

                    if np.argmax (tmp_output) == lable:
                        num_corr+=1

                    gradient=np.zeros(10)

                    gradient[lable]=-1/tmp_output[lable] + np.sum( [2 * regularization * np.sum(np.absolute(layer.get_weights())) for layer in self.layers] )

                    learning_rate = lr_schedule(learning_rate,iteration=i)
                    
                    #print('--Gradient---  gradient[{0}] = {1}'.format(lable,gradient[lable]))
                    
                    self.backward(gradient,learning_rate)
                    self.delta_time+=1
            
            # for l in self.layers:
            #     weight=l.get_weights()
            #     print('\nname={0}   weight {1} {2}'.format(l.name,weight.shape,type(weight)))
                    
            if verbose:
                print('Train Loss: %02.3f' % (history['loss'][-1]))
                print('Train Accuracy: %02.3f' % (history['accuracy'][-1]))
                plot_learning_curve(history['loss'])
                plot_accuracy_curve(history['accuracy'], history['val_accuracy'])
            
            self.prev_time_stamp+=self.delta_time
            # if plot_weights:
            #     for layer in self.layers:
            #         if 'pool' not in layer.name:
            #             plot_histogram(layer.name, layer.get_weights())
            self.update_weights_to_server()
    
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

    def update_weights_to_server(self):
        weights=[]
        for i in range(0,len(self.layers)):
            layer=self.layers[i]
            www=layer.get_weights().tolist()
            weights+=www
            #print("weight layer  weights".format(self.layers[i].name),len(self.layers[i].get_weights()))

        #tmp_chunk_weights=[]
        array_weight=np.array_split(weights,NUM_OF_SUB_MODEL)
        for j in range(0,len(array_weight)):
            new_chunk=CHUNK(array_weight[j],self.prev_time_stamp)
            #tmp_chunk_weights.append(new_chunk)
            self.update_new_chunk(new_chunk,j)

        
    def update_new_chunk(self,chunk_model,index):
        if len(self.chunk_weights ) > index:
            self.chunk_weights[index].update_chunk_with_new_model(chunk_model)
    
    def set_weights_for_layer(self):
        weights=[]
        for i in range(0, len(self.chunk_weights)):
           weights+= self.chunk_weights[i].get_weights().tolist()
        
        array_weights=[]
        
        start_index=0
        end_index=0

        for i in range(0,len(self.layers)):
            end_index+=len(self.layers[i].get_weights())
            www=weights[start_index:end_index]
            time_stamp=self.chunk_weights[i].time_stamp
            array_weights.append(www)
            start_index=end_index

            self.layers[i].set_weights(www)
        
        