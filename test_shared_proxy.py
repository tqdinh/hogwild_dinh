import multiprocessing.managers as m
import multiprocessing

class MySharedClass:

    def __init__(self):
        self.a = 0

    def inc(self):
        self.a += 1
        print(self.a)

    def geta(self):
        return self.a

def f(a):
    a.inc()

def g(f):
    f.write('1')


class MyManager(m.BaseManager):
    pass

MyManager.register("MySharedClass", MySharedClass)
MyManager.register("open", open)



if __name__ == '__main__':
    manager = MyManager()
    manager.start()
    a = manager.MySharedClass()

    n = 100000
    p = multiprocessing.Pool()
    p.map(f, [a] * 5)
    print('{} == {}'.format(a.geta(), n)) # 99996 == 100000

    file = manager.open('test.txt', 'w')
    
    p.map(g, [file] * n)

    print('{} == {}'.format(len(open('test.txt').read()), n)) # 98316 == 100000