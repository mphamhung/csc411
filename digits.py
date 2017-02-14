import numpy as np
import cPickle
from scipy.io import loadmat

M = loadmat("mnist_all.mat")
data = cPickle.load(open('snapshot50.pkl', 'rb'))



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
    dCdW = np.dot((L1 - Y), X.T)
    
    #Checking the Gradient
    
    
if __name__ =='__main__':
    