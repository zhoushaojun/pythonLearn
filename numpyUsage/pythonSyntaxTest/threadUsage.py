import threading

'''简单定义thread
def thread_job():
    print("this is thread %s"%threading.current_thread())

def main():
    thread=threading.Thread(target=thread_job,)
    thread.start()

if __name__ == '__main__':
    main()
'''

'''join
import threading
import time
def thread_job():
    print('T1 start\n')
    for i in range(10):
        time.sleep(0.1)
    print('T1 finish\n')

def T2_job():
    print('T2 start\n')
    print('T2 finish\n')

def main():
    added_thread = threading.Thread(target=thread_job, name='T1')
    thread2 = threading.Thread(target=T2_job, name='T2')
    added_thread.start()
    thread2.start()
    thread2.join()
    added_thread.join()
    print('all done\n')

if __name__ == '__main__':
    main()
'''

#queue
'''
from queue import Queue
import numpy as np
import threading

def thread_job(data,q):
    print('T1 start\n')
    for i in range(len(data)):
        data[i]= data[i]**2
    q.put(data)
    print('T1 finish\n')

def multithreading():
    q = Queue()
    threads=[]
    data = np.arange(9).reshape(3,3)
    print(data)
    for i in range(3):
        t = threading.Thread(target=thread_job,args=(data[i],q))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    result=[]
    for _ in range(3):
        result.append(q.get())
    print(result)

if __name__ == '__main__':
    multithreading()
'''

'''
import threading
from queue import Queue
import copy
import time

def job(l, q):
    res = sum(l)
    q.put(res)

def multithreading(l):
    q = Queue()
    threads = []
    for i in range(4):
        t = threading.Thread(target=job, args=(copy.copy(l), q), name='T%i' % i)
        t.start()
        threads.append(t)
    [t.join() for t in threads]
    total = 0
    for _ in range(4):
        total += q.get()
    print(total)

def normal(l):
    total = sum(l)
    print(total)

if __name__ == '__main__':
    l = list(range(1000000))
    s_t = time.time()
    normal(l*4)
    print('normal: ',time.time()-s_t)
    s_t = time.time()
    multithreading(l)
    print('multithreading: ', time.time()-s_t)
'''

#thread lock
import threading

def job1(lock):
    global A
    lock.acquire()
    for i in range(10):
        A += 1
        print('job1', A)
    lock.release()

def job2(lock):
    global A
    lock.acquire()
    for i in range(10):
        A += 10
        print('job2', A)
    lock.release()

if __name__ == '__main__':
    lock = threading.Lock()
    A = 0
    t1 = threading.Thread(target=job1,args=(lock,))
    t2 = threading.Thread(target=job2,args=(lock,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(A)