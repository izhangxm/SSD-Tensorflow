import time
from multiprocessing import Process, JoinableQueue, Queue
from random import random
from scipy import misc

batch_img_queue = JoinableQueue(maxsize=10)
midle_reulst_queue = JoinableQueue()
cls_queen = Queue()


def msg(flag,info):
    print flag,info

def pre_process(batch_img_queue):
    for i in range(10):
        msg('pre_process','prepare bat_img....')
        time.sleep(0.2)
        img = misc.imread('dog.jpg')
        batch_img_queue.put(img)
    batch_img_queue.put(None)

def batch_extractFeature(batch_img_queue, midle_reulst_queue):
    while 1:
        msg('batch_extractFeature','extracting feature.....')
        time.sleep(0.5)
        img_bat = batch_img_queue.get()
        if img_bat is None:
            midle_reulst_queue.put(None)
            break
        else:
            midle_reulst_queue.put('[[13,24],[0.5,0.6]]')

def refine_result(midle_reulst_queue):
    while 1:
        msg('refine_result','geting final-result.....')
        time.sleep(0.2)
        midl_result = midle_reulst_queue.get()
        if midl_result is None:
            break

start = time.time()

processes = []
p = Process(target=pre_process, args=(batch_img_queue,))
p.start()
processes.append(p)


p = Process(target=batch_extractFeature, args=(batch_img_queue, midle_reulst_queue))
p.start()
processes.append(p)



p = Process(target=refine_result, args=(midle_reulst_queue,))
p.start()
processes.append(p)


batch_img_queue.join()
midle_reulst_queue.join()

for p in processes:
    p.join()

print (time.time() - start)*1000,'ms'
