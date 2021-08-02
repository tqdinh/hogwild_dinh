from multiprocessing import Process
import time
class Worker(Process):
    
    def __init__(self,_a,_val,queue,**kwargs):
        Process.__init__(self,**kwargs)  # Call the parent class initialization method
        self.a=_a
        self.my_val=_val
        self.queue = queue
        self.kwargs = kwargs
    def run(self):
        time.sleep(5)
        self.set_val(9990+self.my_val)
        print("run done")
        self.queue.put(self.get_val())
        
        

    def train(self):
        return self.a.get_my_val()
       
        
    def set_val(self,_my_val):
        self.my_val=_my_val
    
    def get_val(self):
        return self.my_val
