######################################################################
# Learning Objectives: 
# --------------------- 
# 1. Solve the problem without looking for book solution 
# 2. Optimize your solution 
# 3. Have a look on his solution vs mine 
# 4. May start a mindmap summarizing my solution techniques and steps 
# 5. Try pre-trained networks like (ImageNet - Inception ...) 
# 6. Different levels of logging 
# 7. Ambitious dream: what about strategy pattern to support different network architectures !!! :) 
# 
# Experiments: 
# -------------
# 1. Try the network with and without color 
#######################################################################

###########################################
# --- Imports ---
###########################################
import sys
sys.path.insert(0, '~/Desktop/TimeProfiler')
from profiler       import Profiler
print('Log: Start imports')
timer = Profiler()
timer.StartTime() 

from keras.datasets import cifar10
from keras.utils    import np_utils 
import numpy 


###########################################
# from skimage import color 
###########################################
from PIL            import Image 
from matplotlib     import pyplot

timer.EndTime()
timer.DeltaTime() 
print('Log: end imports')

###########################################
# --- Load data --- 
###########################################
(X_train, Y_train) , (X_test, Y_test) = cifar10.load_data() 


###########################################
# --- Fixed seed number for reproducibility --- 
###########################################
print('Log: start seed definition')
seed = 7 
numpy.random.seed(seed)
print('Log: end seed definition')


###########################################
# --- Convert color images into gray ones --- 
###########################################
print('Log: start training images conversion')
timer.StartTime()
X_train_gray = numpy.ndarray(shape=(32, 32))
# for imageIndex in range(0, X_train.shape[0]): 
# TODO: handling the dimension of the numpy array is pretty bad, find a better solution 
# TODO: numpy create an initial record, I need to remove it 
X_training_sample_size = 10
for imageIndex in range(0, X_training_sample_size): 
	img = Image.fromarray(X_train[imageIndex])
	img = img.convert('1')
	X_train_gray = numpy.dstack([X_train_gray, img])
	print("Log: training image number = {}".format(imageIndex))
X_train_gray = numpy.swapaxes(X_train_gray, 1, 2)
X_train_gray = numpy.swapaxes(X_train_gray, 0, 1)
timer.EndTime()
timer.DeltaTime() 
print('Log: end training image conversion') 

print('Log: start testing image conversion')
timer.StartTime()
X_test_gray = numpy.ndarray(shape=(32, 32))
#for imageIndex in range(0, X_test.shape[0]):
# TODO: apply training modifications here
# TODO: code is similar may need to put in a function 
# TODO: the function may be generic enough to put outside the code 
# TODO: think of a library of utilities to be on git-hub   
for imageIndex in range(0, X_training_sample_size/5): 
	img = Image.fromarray(X_test[imageIndex])
	X_test_gray = numpy.vstack([X_test_gray, img.convert('1')])
	print("Log: testing image number = {}".format(imageIndex))
timer.EndTime()     
timer.DeltaTime()
print('Log: end testing image conversion')

print('Log: start input manipulation')
timer.StartTime()


###########################################
# --- Reshape to be [sample][channel][width][height] 
###########################################
# TODO: this is a temp solution, the added one needs to removed, because the first item in the array is fake and needs to be removed 
X_train_gray = X_train_gray.reshape(X_training_sample_size +1, 1, 32, 32) 
X_test_gray  = X_test_gray.reshape(X_training_sample_size/5 +1, 1 , 32, 32)


###########################################
# --- Normalization --- 
###########################################
X_train_gray = X_train_gray/255 
X_test_gray  = X_test_gray/255 


###########################################
# --- One hot encoding --- 
###########################################
Y_train     = np_utils.to_categorical(Y_train)
Y_test      = np_utils.to_categorical(Y_test) 
num_classes = Y_test.shape[1]

###########################################
# --- DEBUG --- 
print('classes count = {}'.format(num_classes)) 
###########################################
timer.EndTime()
timer.DeltaTime() 
print('Log: end input manipulation')








																																																																																									
