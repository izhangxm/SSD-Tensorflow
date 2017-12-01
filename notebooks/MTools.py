import time

class Timer(object):
    pre = time.time()


    def start(self):
        self.pre = time.time()
    def consume(self,info = ''):
        end = time.time()
        print info,(end - self.pre)*1000,'ms'
        self.pre = end