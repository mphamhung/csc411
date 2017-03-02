
from pylab import *
import numpy as np
import cPickle
from scipy.io import loadmat
import matplotlib.pyplot as plt
from numpy import *
from scipy.misc import imresize


# def softmax(y):
#     '''Return the output of the softmax function for the matrix of output y. y
#     is an NxM matrix where N is the number of outputs for a single case, and M
#     is the number of cases'''
#     return exp(y)/tile(sum(exp(y),0), (len(y),1))

def softmax(y):
    return np.exp(y)/np.tile(np.sum(np.exp(y),0), (y.shape[0],1))

def part2(X, W):
    #Input X is a flattened 28x28 vector representing the image and bias term
    #Input W is a 10,785 matrix representing the weight matrix and the dummy bias term
    #Ouput the softmax of the nine possible digits
    X = np.vstack( (ones((1, X.shape[1])), X))
    L0 = np.dot(W, X)
    output = softmax(L0)
    return L0, output
    
def f(X, Y, W):
    # assume vstack is 
    # returns cost function    
    X = np.vstack( (np.ones((1, X.shape[1])), X))
    
    #Forward Pass
    L0 = np.dot(W, X)
    # print("LO", L0)
    L1 = softmax(L0)
    
    # print("L1", L1)
    
    return -np.dot(Y.T, np.log(L1))

    
def df(X, Y, W): #df
    # Computing the Gradient w.r.t. the Negative Log Cost Function
    X = np.vstack( (np.ones((1, X.shape[1])), X))
    
    #Forward Pass
    L0 = np.dot(W, X)
    L1 = softmax(L0)
    
    # print(shape(dCdW(X, L1, Y)))
    #Sketchy backprop with skipped steps
    return np.dot((L1 - np.multiply(Y,L1)), X.T)
    
def grad_descent(f, df, x, y, init_t, alpha, divide_EPS_by):
    # Added parameter divide_EPS_by: used to make EPS smaller by a factor at each
    # time step
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    while np.linalg.norm(t - prev_t) >  EPS/divide_EPS_by and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 2000 == 0:
            print "Iter", iter
            # print ("f(x) = %.2f" % (f(x, y, t)) )
            print "Gradient: ", df(x, y, t).flatten(), "\n"
            
        iter += 1
        
    print("Total iterations: %d" % iter)
    return t  
      
def grad_descent_checker(f, df, x, y, init_t, alpha, divide_EPS_by):
    # Slightly modified version of gradient descent only used for part 6)d)
    # All the changes are within the (iter % 2000 == 0) conditional
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0
    while np.linalg.norm(t - prev_t) >  EPS/divide_EPS_by and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 2000 == 0:
            print "\nIter", iter
            #use a delta that's very small
            delta = alpha/1000
            # fill an array that's the same dimensions as theta (i.e. t)
            delta_theta = np.full(np.shape(t), delta)
            
            # compute the result manually using the def'n of derivate
            manually_computed = (f(x, y, t + delta_theta) - f(x, y, t))/(delta)
            print("Manually computed: %f" % manually_computed)
            # compute the result with the one computed by the gradient function
            df_computed = np.sum(df(x, y, t))
            print("Gradient function result: %f", df_computed)
                
        iter += 1
        
    print("Total iterations: %d" % iter)
    return t

    
def build_y(training_size, number):
    # constructs y such that the first training_size values are 1, and the last
    # are -1
    # y = np.concatenate( ( np.full((training_size, ), 1.), np.full((training_size, ), -1.) ) )
    y = np.zeros((training_size, 10))
    for i in range(training_size):
        y[i][number] = 1
    
    return y.T
    
def build_w():
    # return np.random.random((10, 785))/999.
    return np.zeros((10,785))

def part3(x, y, init_t):
    grad_descent_checker(f, df, x, y, init_t, alpha = 0.0000000005, divide_EPS_by = 80)
    
def part4():    
    M = loadmat("mnist_all.mat")
    x = M["train2"][:200].T

    # for i in range(10):
    #     cur_train = "train" + str(i)
    #     if i != 2:
    #         pass
    #         # new_x = np.array(M[cur_train][:20].T)
    #         # x = np.hstack((x, new_x))
    #         # y = np.hstack((y, build_y(np.shape(new_x)[1], i)))
    #     else:
    #         x = M[cur_train][:200].T
    #         y = build_y(np.shape(x)[1], i)
    
    init_w = build_w()
    
    print(np.shape(x), np.shape(y), np.shape(init_w))
    
    #normalizes X as a vector
    x = (x - 127*ones((x.shape)))/255.0
    
    w = grad_descent(f, df, x, y, init_w, alpha = 0.000000015, divide_EPS_by = 1)
    
    print(w[0][1:].shape)
    plt.close()
    for i in range(10):
        w_show = w[i][1:].reshape((28,28))
        w_resized = imresize(w_show, (320,320))
        # plt.figure(i+1)
        # imshow(w_resized, cmap=cm.gray)
        imsave('number_%d.png' % i, w_resized)
    # show()
    

if __name__ =='__main__':
    
    np.random.seed(0)
    
    # M = loadmat("mnist_all.mat")
    # snapshot = cPickle.load(open("snapshot50.pkl"))
    # W0 = snapshot["W0"]
    # b0 = snapshot["b0"].reshape((300,1))
    # W1 = snapshot["W1"]
    # b1 = snapshot["b1"].reshape((10,1))
    # x = M["train5"][148:149].T
    
    # part2(x, dot(W0,W1))
    y = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]).reshape((10,1))
    # print(shape(y))
    init_t = np.random.random((10, 785))/999.
    # part3(x, y, init_t)
    part4()
    
    
