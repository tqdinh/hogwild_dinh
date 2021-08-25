from read_write_file import * 
import matplotlib.pyplot as plt
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


n_thread_array=[1,3,5,10,15]
folder_name=[]
loss_array_of_thread=[]
accuracy_array_of_thread=[]
for i in range(len(n_thread_array)):
    num_of_thread=n_thread_array[i]
    specific_folder_name="n_thread_{0}".format(num_of_thread)
    folder_name.append(specific_folder_name)
    
    
    list_of_accuracy=[]
    list_of_loss=[]
    for thread_index in range(num_of_thread):
        file_name_accuracy="{0}/{1}/accacy_process_{2}".format(dir_path,specific_folder_name,thread_index)
        _accuracy=read_list(file_name_accuracy)
        list_of_accuracy.append(_accuracy)

        file_name_loss="{0}/{1}/loss_process_{2}".format(dir_path,specific_folder_name,thread_index)
        _loss=read_list(file_name_loss)
        list_of_loss.append(_loss)
    

    list_of_loss=np.array(list_of_loss).T
    list_of_accuracy=np.array(list_of_accuracy).T

    array_min_loss=[]
    array_min_acc=[]

    for row in range(len(list_of_loss)):
        min_loss_row=np.min(list_of_loss[row])
        min_acc_row=np.min(list_of_accuracy[row])
       
        array_min_loss.append(min_loss_row)
        array_min_acc.append(min_acc_row)

    loss_array_of_thread.append(array_min_loss)
    accuracy_array_of_thread.append(array_min_acc)


# print(loss_array_of_thread)
# print(accuracy_array_of_thread)

color_val=['b','g','r','c','m','y']

lable_loss=[]
lable_acc= []
val_loss=loss_array_of_thread
val_acc=accuracy_array_of_thread
for index_folder_name in range(len(folder_name)):
    
    line_lable_loss="{0}_loss".format(folder_name[index_folder_name])
    lable_loss.append(line_lable_loss)
    
    color=color_val[index_folder_name%len(color_val)]
    plt.plot(val_loss[index_folder_name], color, linewidth=1.0, label=line_lable_loss)    
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.title('Loss with learning rate scheduled step decay ', fontsize=16)
    plt.savefig('{0}/Loss_threads.png'.format(dir_path))

plt.show()

for index_folder_name in range(len(folder_name)):
    line_lable_acc="{0}_acc".format(folder_name[index_folder_name])
    lable_acc.append(line_lable_acc)
    color=color_val[index_folder_name%len(color_val)]
    plt.plot(val_acc[index_folder_name], color, linewidth=1.0, label=line_lable_acc)    
    plt.xlabel('epoch', fontsize=16)
    plt.ylabel('acc', fontsize=16)
    plt.legend()
    plt.title('Accuracy with learning rate scheduled step decay ', fontsize=16)
    plt.savefig('{0}/accuracy_threads.png'.format(dir_path))

plt.show()