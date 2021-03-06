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
from keras.models   import Sequential 
from keras.layers   import Dense 


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
(X_train, y_train) , (X_test, y_test) = cifar10.load_data() 


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
"""
print('Log: start training images conversion')
timer.StartTime()
X_train_gray = numpy.ndarray(shape=(32, 32))
# for imageIndex in range(0, X_train.shape[0]): 
# TODO: handling the dimension of the numpy array is pretty bad, find a better solution 
# TODO: numpy create an initial record, I need to remove it 
# TODO: modify the sample size to 10000
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
	img = img.convert('1')
	X_test_gray = numpy.dstack([X_test_gray, img])
	print("Log: testing image number = {}".format(imageIndex))
X_test_gray = numpy.swapaxes(X_test_gray, 1, 2)	
X_test_gray = numpy.swapaxes(X_test_gray, 0, 1)
timer.EndTime()     
timer.DeltaTime()
print('Log: end testing image conversion')

print('Log: start input manipulation')
timer.StartTime()

print('Log: X_test_gray = {}'.format(X_test_gray))
"""
###########################################
# --- Extract a sample ---
###########################################
training_sample_size = 10000
X_train = X_train[0:training_sample_size]
y_train = y_train[0:training_sample_size]

X_test  = X_test[0:training_sample_size/5]
y_test  = y_test[0:training_sample_size/5]


###########################################
# --- Flatten Input ---
###########################################
num_pixels = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
print ("num_pixels = {}".format(num_pixels))
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test  = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

###########################################
# --- Normalization --- 
###########################################

X_train = X_train/255 
X_test  = X_test/255 


###########################################
# --- One hot encoding --- 
###########################################
y_train     = np_utils.to_categorical(y_train)
y_test      = np_utils.to_categorical(y_test) 
num_classes = y_test.shape[1]

###########################################
# --- Define baseline model ---
###########################################
def baseline_model(): 
	# Create model 
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, init='normal', activation='relu'))
	model.add(Dense(num_classes, init='normal', activation='softmax'))
	# Compile model 
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 
	return model


###########################################
# --- Build the model ---
###########################################
model = baseline_model() 


###########################################
# --- Fit the model ---
###########################################
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)


###########################################
# --- Final evaluation ---
###########################################
scores = model.evaluate(X_test, y_test, verbose=0) 
print('Log: score = {} %'.format(scores))












																																																																																									
