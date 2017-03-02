
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy import misc
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib



#act = list(set([a.split("\t")[0] for a in open("subset_actors.txt").readlines()]))
act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

#tries to find the "uncropped folder, and makes it if it does not exist
dir = "uncropped"
try:
    os.stat(dir)
except:
    os.mkdir(dir)
dir = "cropped"
try:
    os.stat(dir)
except:
    os.mkdir(dir)


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.


#Note: you need to create the uncropped folder first in order 
#for this to work
for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in open("faces_subset.txt"):
        if a in line:
            filename = line.split()[6]+'.'+line.split()[4].split('.')[-1]
            # filename = line.split()[6] + ".png"
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download

            #print(line.split()[5].split(','))
            if not os.path.isfile("uncropped/"+filename):
            
                timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                if not os.path.isfile("uncropped/"+filename):
                    continue
    
                try:            
                    #image downloaded successfully
                    face = misc.imread('uncropped/'+filename)
                    if not face is None:
                        y1 = int(line.split()[5].split(',')[1])
                        y2 = int(line.split()[5].split(',')[3])         
                        x1 = int(line.split()[5].split(',')[0])
                        x2 = int(line.split()[5].split(',')[2])
                        
                        #print([y1,y2,x1,x2])
                        crop_face = face[y1:y2,x1:x2]
                        grayscale_face = rgb2gray(crop_face)
                        resized_face = misc.imresize(grayscale_face, (32,32))
                        #plt.imshow(crop_face, cmap=plt.cm.gray) 
                        #plt.show()
                        
                        misc.imsave('cropped/'+filename, resized_face)
                        i += 1   
                        #j = 0
                        print (filename)
                except:
                    print("failed to download picture")          
                    #j += 1               
                    #misc.imsave('failed/'+filename+"_"+j, face)


