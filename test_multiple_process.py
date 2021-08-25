import time
from multiprocessing import Process, Queue
import multiprocessing.managers as m
from ShareClassTest import *
from Xworker import *
from hogwild_server import *


class MyManager(m.BaseManager):
    pass

MyManager.register("MySharedClass", MySharedClass)
MyManager.register("MYHOG", DATA_CHUNKS_HANDLER)
  
worker=any
worker2=any
worker3=any
def f0():
    pass

def main():
    manager = MyManager()
    manager.start()
    #a = manager.MySharedClass()
    a=manager.MYHOG()
    q = Queue() 
    worker = Worker(a,1,q)
    worker.start()

    worker2  = Worker(a,2,q)
    worker2.start()

    worker3 = Worker(a,3,q)
    worker3.start()

    worker.join()
    worker2.join()
    worker3.join()
    #print(a.geta())
    # print("run get val")
    # print(worker.get_val())
    # print(worker2.get_val())
    # print(worker3.get_val())
    #[proc.join() for proc in processes]
    while not q.empty():
        print("RESULT: {0}".format(q.get()) )  # get results from the queue...



if __name__ == '__main__':
    print(np.random.randint(1,3))
    