######################################################################
# Learning Objectives: 
# --------------------- 
# 1. Solve the problem without looking for book solution 
# 2. Optimize your solution 
# 3. Have a look on his solution vs mine 
# 4. May start a mindmap summarizing my solution techniques and steps 
# 5. Try pre-trained networks like (ImageNet - Inception ...) 
# 
# Experiments: 
# -------------
# 1. Try the network with and without color 
#######################################################################

# --- Imports ---
from profiler       import Profiler
print('Log: Start imports')
timer = Profiler()
timer.StartTime() 

from keras.datasets import cifar10
from keras.utils    import np_utils 
import numpy 

# from skimage import color 
from PIL            import Image 
from matplotlib     import pyplot

timer.EndTime()
timer.DeltaTime() 
print('Log: end imports')
# --- Load data --- 
(X_train, Y_train) , (X_test, Y_test) = cifar10.load_data() 


# img = color.rgb2gray(X_train[0]) 
"""
imgCount = 10
img = Image.fromarray(X_train[imgCount])
img = img.convert('1')
"""
# --- Show Image --- 
"""
pyplot.subplot(211) 
pyplot.imshow(X_train[imgCount])
pyplot.subplot(212) 
pyplot.imshow(img)
pyplot.show()
"""

# --- Fixed seed number for reproducibility --- 
print('Log: start seed definition')
seed = 7 
numpy.random.seed(seed)
print('Log: end seed definition')
###########################################
# --- DEBUG --- 
"""
X_train_gray = numpy.ndarray(shape=(32, 32))
for imageIndex in range(0, 10): 
	img = Image.fromarray(X_train[imageIndex])
	# X_train_gray.append(img.convert('1'))
	X_train_gray = numpy.vstack([X_train_gray, img.convert('1')])
X_train_gray = X_train_gray.reshape(11, 1, 32, 32)
print("Finished Processing ... ")
"""
########################################### 


# --- Convert color images into gray ones --- 
print('Log: start training images conversion')
timer.StartTime()
X_train_gray = numpy.ndarray(shape=(32, 32))
# for imageIndex in range(0, X_train.shape[0]): 
X_training_sample_size = 10
for imageIndex in range(0, X_training_sample_size): 
	img = Image.fromarray(X_train[imageIndex])
	X_train_gray = numpy.vstack([X_train_gray, img.convert('1')])
	print("Log: training image number = {}".format(imageIndex))

timer.EndTime()
timer.DeltaTime() 
print('Log: end training image conversion') 

print('Log: start testing image conversion')
timer.StartTime()
X_test_gray = numpy.ndarray(shape=(32, 32))
#for imageIndex in range(0, X_test.shape[0]): 
for imageIndex in range(0, X_training_sample_size/5): 
	img = Image.fromarray(X_test[imageIndex])
	X_test_gray = numpy.vstack([X_test_gray, img.convert('1')])
	print("Log: testing image number = {}".format(imageIndex))
timer.EndTime()     
timer.DeltaTime()																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				print('Log: end testing image conversion')
###########################################
# --- DEBUG --- 

pyplot.subplot(111)
pyplot.imshow(X_train_gray[2])
pyplot.show()

###########################################

# --- Reshape to be [sample][channel][width][height] 
# X_train_gray = X_train_gray.reshape(X_train_gray.shape[0], 1, 32, 32) 
# X_test_gray  = X_test_gray.reshape(X_test_gray.shape[0], 1 , 32, 32)

X_train_gray = X_train_gray.reshape(X_train_gray.shape[0], 1, 32, 32) 
X_test_gray  = X_test_gray.reshape(X_test_gray.shape[0], 1 , 32, 32)


# --- Normalization --- 
X_train_gray = X_train_gray/255 
X_test_gray  = X_test_gray/255 

# --- One hot encoding --- 
Y_train     = np_utils.to_categorical(Y_train)
Y_test      = np_utils.to_categorical(Y_test) 
num_classes = Y_test.shape[1]

###########################################
# --- DEBUG --- 
print('classes count = {}'.format(numclasses)) 
###########################################



		







																																																																																									
