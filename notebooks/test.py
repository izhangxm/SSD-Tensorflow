
import numpy as np
import  scipy
from scipy import misc
import matplotlib.image as mpimg
import tensorflow as tf
from preprocessing import ssd_vgg_preprocessing
def demo1():


    a = np.arange(6*2*10*10*21).reshape([6,2,10,10,21])


    print a.shape # (2, 3, 2)

def demo2():
    a = np.arange(30).reshape([3,2,5])
    b = np.arange(60).reshape([4,2,5])

    c_list = [a,b]

    d_arra = np.array(c_list)

    print d_arra


def demo3():
    a = np.arange(6 * 2 * 10 * 10 * 21).reshape([6, 2, 10, 10, 21])
    print a.shape

    b = np.expand_dims(a,axis=1)

    print b.shape



def demo4():
    a = np.arange(6 * 2 * 10 * 10 * 21).reshape([6, 2, 10, 10, 21])
    print a.shape



    b = np.expand_dims(a,axis=1)

    print b.shape

    c = np.array_split(b, b.shape[0], axis=0)
    print type(c[0]),len(c),c[0].shape


def demo5():
    image = mpimg.imread('dog.jpg')
    # VGG mean parameters.
    _R_MEAN = 123.
    _G_MEAN = 117.
    _B_MEAN = 104.
    means = [_R_MEAN, _G_MEAN, _B_MEAN]
    image = image - means


def demo6():
    ttf_img, _, _, _ = ssd_vgg_preprocessing.preprocess_for_eval(tf.constant(misc.imread('dog.jpg'), dtype=tf.uint8), None, None, (300,300), 'NHWC', resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    isess = tf.Session()
    rtfimg = isess.run([ttf_img])
    print rtfimg[0]
    """
    [[[ -66.          -59.          -54.        ]
      [ -62.43999863  -55.43999863  -50.43999863]
      [ -62.36000061  -55.36000061  -50.36000061]
      ..., 
      [  62.00018311  110.32000732   34.96002197]
      [  69.2800293   121.16003418   27.95996094]
      [  -3.76013184  -45.60009766  -61.88000488]]
    """

def demo7():
    pass
    img = misc.imread('dog.jpg')

    # VGG mean parameters.
    _R_MEAN = 123.
    _G_MEAN = 117.
    _B_MEAN = 104.

    means = [_R_MEAN, _G_MEAN, _B_MEAN]
    img = misc.imresize(img,(300,300,3),interp='bilinear')
    img = img - means

    print img[0]


def demo8():
    pass



demo7()