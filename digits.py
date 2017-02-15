from numpy import *
import cPickle
from scipy.io import loadmat


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def part2(X, W):
    #Input X is a flattened 28x28 vector representing the image and bias term
    #Input W is a 10x785 matrix representing the weight matrix and the dummy bias term
    #Ouput the softmax of the nine possible digits
    L0 = dot(W.T, X)
    output = softmax(L0)
    return L0, output
    
def part3(W, X, Y):
    # Computing the Gradient w.r.t. the Negative Log Cost Function
    
    #Forward Pass
    L0 = dot(W.T, X)
    L1 = softmax(L0)
    
    #Sketchy backprop with skipped steps
    dCdW = dot((L1 - Y), X.T)
    
    return 0
    #Checking the Gradient

    

if __name__ =='__main__':
    
    M = loadmat("mnist_all.mat")
    snapshot = cPickle.load(open("snapshot50.pkl"))
    W0 = snapshot["W0"]
    b0 = snapshot["b0"].reshape((300,1))
    W1 = snapshot["W1"]
    b1 = snapshot["b1"].reshape((10,1))
    x = M["train5"][148:149].T  
    
    part2(x, dot(W0,W1))
    
    
