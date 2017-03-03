from pylab import *
import numpy as np
import cPickle
from scipy.io import loadmat
import matplotlib.pyplot as plt
# from numpy import *
from scipy.misc import imresize
import random
import time

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

    #Sketchy backprop with skipped steps
    return np.dot((L1-Y),X.T)
    #return np.dot((L1-Y),X.T)
    
def grad_descent(f, df, x, y, init_t, alpha, divide_EPS_by, plot_perf = False, plot_name='learning_curves.png', max_iter = 15000, test_data = []):
    '''
    Added parameter divide_EPS_by: used to make EPS smaller by a factor at
    each time step. Also added an option to plot a learning curve
    '''
    if plot_perf:
        x_v, y_v, x_t, y_t = test_data
        x_axis = []
        result_v = []
        result_t = []
    
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter  = 0
    while np.linalg.norm(t - prev_t) >  EPS/divide_EPS_by and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t)
        if iter % 10 == 0:
            print "Iter", iter
            # print ("f(x) = %.2f" % (f(x, y, t)) )
            # print ("Gradient: ", df(x, y, t), "\n")
            # print "Weight: ", t, "\n"
            
            #checking performance
            if plot_perf:
                x_axis.append(iter)
                print "Validation set:"
                result_v.append(validate(x_v, y_v, t))
                print "Testing set:"
                result_t.append(validate(x_t, y_t, t))
        iter += 1
        
    print("Total iterations: %d" % iter)
    # print t
    if plot_perf:
        plt.close()
        
        plot(x_axis, result_v, 'go', label="Validation")    
        plot(x_axis, result_t, 'rx', label="Test")
        axis([-50, iter, 0, 105])
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy (%)")  
        plt.legend()
        savefig(plot_name, bbox_inches='tight')
    return t  
    
def grad_check(X,Y,W,f,df):
    '''Defining Variables'''
    row = np.linspace(1,W.shape[0],W.shape[0]).astype(int)
    col = np.linspace(1,W.shape[1],W.shape[1]).astype(int)
    
    ''' Testing Finite Differences for different components of Theta'''
    sum0 =0
    sum1=0
    for i in row[:-1]:
        for j in col[:-1]:
            theta0 = np.zeros(W.shape)
            theta1= np.zeros(W.shape)
            theta2 = np.zeros(W.shape)
            
            h = 0.0001
            theta1[i][j] = h
            theta2[i][j] = -h
        
            sum0 += (f(x,y,theta1)-f(x,y,theta2))/(2*h)
            sum1+= df(x, y, theta0)[i][j]
            
    print sum0
    print sum1

def part3(x, y, init_t):
    grad_check(x,y,init_t,f,df)
    
def build_y(training_size, number):
    # constructs y such that the first training_size values are 1, and the last
    # are -1
    # y = np.concatenate( ( np.full((training_size, ), 1.), np.full((training_size, ), -1.) ) )
    y = np.zeros((training_size, 10))
    for i in range(training_size):
        y[i][number] = 1
    
    return y.T
    
def build_data(train_size, valid_size, test_size, M, noise = (False, 0), normalize = True ):
    noise_size = 0
    if noise[0]:
        noise_size = int(train_size*noise[1])
        train_size = train_size - noise_size
        sigma = noise[1]
        
    for i in range(10):
        cur_train = "train" + str(i)
        random_imgs = random.sample(xrange(0, len(M[cur_train]) - 1), train_size +noise_size+ valid_size + test_size)
        training_imgs = random_imgs[:train_size]
        noise_imgs = random_imgs[train_size:train_size+noise_size]
        valid_imgs = random_imgs[train_size+noise_size:train_size+noise_size+valid_size]
        test_imgs = random_imgs[train_size+valid_size+noise_size:]
        
        training = []
        valid = []
        test = []
        
        for j in range(len(M[cur_train])):
            if j in training_imgs:
                training.append(M[cur_train][j])
            elif j in noise_imgs:
                # digits = range(10)
                # digits.remove(i)
                # rand_int = int(np.random.choice(digits, 1))
                # random_train = "train"+str(rand_int)
                # training.append(M[random_train][j%len(random_train)])
                noisy_img = (M[cur_train][j] + 255*np.random.normal(0, sigma, M[cur_train][j].shape))%255
                training.append(noisy_img)
                
            elif j in valid_imgs:
                valid.append(M[cur_train][j])
            elif j in test_imgs:
                test.append(M[cur_train][j])
            
        if i == 0:
            x = np.array(training).T
            x_v = np.array(valid).T
            x_t = np.array(test).T
            y = build_y(train_size+noise_size, 0)
            y_v = build_y(valid_size, 0)
            y_t = build_y(test_size, 0)
        else:
            x = np.hstack((x, np.array(training).T))
            x_v = np.hstack((x_v, np.array(valid).T))
            x_t = np.hstack((x_t, np.array(test).T))
            y = np.hstack((y, build_y(train_size+noise_size, i)))
            y_v = np.hstack((y_v, build_y(valid_size, i)))
            y_t = np.hstack((y_t, build_y(test_size, i)))
    
    if normalize:    
        #normalizes X as a vector

        x = (x - 127*ones((x.shape)))/255.0
        x_v = (x_v - 127*ones((x_v.shape)))/255.0
        x_t = (x_t - 127*ones((x_t.shape)))/255.0
    
    return x, x_v, x_t, y, y_v, y_t
        
    
def validate(x, y, w):
    x = x.T
    y = y.T
    result = 0
    valid_size = shape(x)[0]
    
    #error check
    if valid_size != shape(y)[0]:
        print "Size mismatch in x and y during validation"
    
    for image in range(valid_size):
        test = np.insert(x[image], 0, 1) # appends a 1 at the start of X[i]
        if argmax(np.dot(w, test)) == argmax(y[image]): # compares max of W.X[i] and Y[i]
            result += 1
        # else: # in case you wanna see which ones it's getting wrong
        #     print np.dot(w, test), argmax(np.dot(w, test)), y[image], argmax(y[image])
    
    print "Result: %f%%\nAccuracy: %d/%d" % (result*100/float(valid_size), result, valid_size)
    
    return result*100/float(valid_size) #in case you need this
    
def part4():    
    M = loadmat("mnist_all.mat")
    x, x_v, x_t, y, y_v, y_t = build_data(60, 20, 20, M)
    
    # print(shape(x), shape(x_v), shape(x_t), shape(y), shape(y_v), shape(y_t))
    
    init_w = np.random.random((10,785))/10000.
    
    # print(np.shape(x), np.shape(y), np.shape(init_w))
    
    test_data = [x_v, y_v, x_t, y_t]
    w = grad_descent(f, df, x, y, init_w, alpha = 0.0002, divide_EPS_by = 0.01, plot_perf = True, test_data = test_data)
    # w = grad_descent(f, df, x, y, init_w, alpha = 0.004, divide_EPS_by = 0.01)
    
    print(w[0][1:].shape)
    plt.close()
    for i in range(10):
        w_show = w[i][1:].reshape((28,28))
        w_resized = imresize(w_show, (320,320))
        # plt.figure(i+1)
        # imshow(w_resized, cmap=cm.gray)
        imsave('number_%d.png' % i, w_resized)
    # show()
    
    print "Testing with training set"
    validate(x, y, w)    
    print "Testing with validation set"
    validate(x_v, y_v, w)    
    print "Testing with testing set"
    validate(x_t, y_t, w)
    
    return x, w, y #in case you need this
def f_lin(X,y,theta):
    X = np.vstack( (np.ones((1, X.shape[1])), X))
    return np.sum((dot(theta,X) - y)**2)
    
def df_lin(X,y,theta):
    X = np.vstack( (np.ones((1, X.shape[1])), X))
    return 2*dot(X,(dot(theta,X) - y).T).T
    
def part5():
    M = loadmat("mnist_all.mat")
    x, x_v, x_t, y, y_v, y_t = build_data(60, 20, 20, M, noise = (True, 0.5))
    init_w = np.random.random((10,785))/10000.
    test_data = [x_v, y_v, x_t, y_t]
    
    print "________________________________________\n Training Linear Model"
    w_lin = grad_descent(f, df_lin, x, y, init_w, alpha = 0.00001, divide_EPS_by = 0.1, plot_perf = True, test_data = test_data, max_iter = 1000, plot_name = "Linear Model Performance.png")
    
    print "________________________________________\n Training Multinomial Log Model"
    w_log = grad_descent(f, df, x, y, init_w, alpha = 0.0002, divide_EPS_by = 0.01, plot_perf = True, test_data = test_data, max_iter = 1000, plot_name = "Multinomial Log Model Performance.png")
    

    
    plt.close()
    for i in range(10):
        w_show = w_log[i][1:].reshape((28,28))
        w_resized = imresize(w_show, (320,320))
        # plt.figure(i+1)
        # imshow(w_resized, cmap=cm.gray)
        imsave('number_%d.png' % i, w_resized)
    # show()
    
    
    #return x, w, y #in case you need this
    
def df_non_vec(X, Y, W):
    X = np.vstack( (np.ones((1, X.shape[1])), X))
    p = softmax(np.dot(W,X))
    grad = []
    for i in range(W.shape[0]):
        row = []
        for j in range(W.shape[1]):
            w_ij = np.dot((p[i]-y[i]),X[j])
            row.append(w_ij)
        grad.append(row)
    
    return np.array(grad)
            
    
def part6(x,y,w):
    #computing gradient wrt to each element
    t0 = time.time()
    g1 = df_non_vec(x,y,w)
    t1 = time.time()
    print "non vectorized gradient time = ", t1-t0
    
    t0 = time.time()
    g2 = df(x,y,w)
    t1 = time.time()
    print "vectorized gradient time = ", t1-t0
    
if __name__ =='__main__':
    
    np.random.seed(0)
    
    M = loadmat("mnist_all.mat")
    snapshot = cPickle.load(open("snapshot50.pkl"))
    W0 = snapshot["W0"]
    b0 = snapshot["b0"].reshape((300,1))
    W1 = snapshot["W1"]
    b1 = snapshot["b1"].reshape((10,1))
    x = M["train5"][148:149].T  
    
    # part2(x, dot(W0,W1))
    y = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]).reshape((10,1))
    # print(shape(y))
    init_t = np.random.random((10, 785))/999.
    #part3(x, y, init_t)
    #x,w,y = part4()
    #part5()
    #grad_check(x,y,init_t, f_lin, df_lin)
    part6(x,y,init_t)
    
