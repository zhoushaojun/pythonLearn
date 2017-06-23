import subprocess
from multiprocessing import Process, Queue
import os, time, random
from multiprocessing import Pool

#子进程为外部
'''
print('$ nslookup www.python.org')
r = subprocess.call(['nslookup', 'www.python.org'])
print('Exit code:', r)
'''

# 子进程要执行的代码
'''
def run_proc(name):
    print('Run child process %s (%s)...' % (name, os.getpid()))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('Child process will start.')
    p.start()
    p.join()
    print('Child process end.')
'''

#线程池
'''
def long_time_task(name):
    print('Run task %s (%s)...' % (name, os.getpid()))
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print('Task %s runs %0.2f seconds.' % (name, (end - start)))

if __name__=='__main__':
    print('Parent process %s.' % os.getpid())
    p = Pool(4)
    for i in range(5):
        p.apply_async(long_time_task, args=(i,))
    print('Waiting for all subprocesses done...')
    p.close()
    p.join()
    print('All subprocesses done.')
'''

#子线程之间通讯
'''
def write(q):
    print('Process to write: %s' % os.getpid())
    for value in ['A', 'B', 'C']:
        print('Put %s to queue...' % value)
        q.put(value)
        time.sleep(random.random())

# 读数据进程执行的代码:
def read(q):
    print('Process to read: %s' % os.getpid())
    while True:
        value = q.get(True)
        print('Get %s from queue.' % value)

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = Queue()
    pw = Process(target=write, args=(q,))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pw结束:
    pw.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    pr.terminate()
'''

#Thread
'''
#创建
import time,threading

def loop():
    print('thread %s is running...' % threading.currentThread().name)
    n = 0
    while n < 5:
        n = n + 1
        print('thread %s >>> %s' % (threading.current_thread().name, n))
        time.sleep(1)
    print('thread %s ended.' % threading.current_thread().name)
t1=threading.Thread(target=loop,name="loopthread1")
t2=threading.Thread(target=loop,name="loopthread2")
t1.start()
t2.start()
t1.join()
t2.join()
'''

#锁
'''
import time,threading

lock = threading.Lock()
class threadLock(object):

    def __init__(self,age):
        self.__age__=age

    @property
    def age(self):
        return self.__age__

    @age.setter
    def age(self,age):
        self.__age__=age

def change_age(tLock):
        age=tLock.age+1
        tLock.age = age

def run_thread(tLock):
    time.sleep(random.random() * 3)
    print(threading.currentThread().getName())
    change_age(tLock)



if __name__=='__main__':
    tLock=threadLock(5)
    t1 = threading.Thread(target=run_thread, args=(tLock,))
    t2 = threading.Thread(target=run_thread, args=(tLock,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    print(tLock.age)
'''

#threadlocal
import threading

# 创建全局ThreadLocal对象:
local_school = threading.local()

def process_student():
    # 获取当前线程关联的student:
    std = local_school.student
    print('Hello, %s (in %s)' % (std, threading.current_thread().name))

def process_thread(name):
    # 绑定ThreadLocal的student:
    local_school.student = name
    process_student()

t1 = threading.Thread(target= process_thread, args=('Alice',), name='Thread-A')
t2 = threading.Thread(target= process_thread, args=('Bob',), name='Thread-B')
t1.start()
t2.start()
t1.join()
t2.join()