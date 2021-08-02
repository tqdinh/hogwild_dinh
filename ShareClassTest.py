class MySharedClass:
    
    def __init__(self):
        self.a = 0

    def inc(self):
        self.a += 1
        print(self.a)

    def geta(self):
        return self.a
    def set_val(self,x):
        print("xxxxx")
        self.a=x