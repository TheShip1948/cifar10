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

from keras.wrappers.scikit_learn import KerasClassifier 
from sklearn.model_selection import GridSearchCV 

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
# --- Extract a sample ---
###########################################
training_sample_size = 100
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
def baseline_model(optimizer='adam', init='normal'): 
	# Create model 
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, init=init, activation='relu'))
	model.add(Dense(num_classes, init=init, activation='softmax'))
	# Compile model 
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) 
	return model


###########################################
# --- Build the model ---
###########################################
# model = baseline_model() 
model = KerasClassifier(build_fn=baseline_model, verbose=0)


###########################################
# --- Grid search for values ---
###########################################
optimizers = ['adam' , 'sgd', 'rmsprop']
init = ['glorot_uniform' , 'normal' , 'uniform']
epochs = numpy.array([10 , 20 , 30])
batches = numpy.array([5, 50 , 200, 300])

###########################################
# --- Fit the model ---
###########################################
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)


param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init) 
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X_train, y_train)


###########################################
# --- Summarize results ---
###########################################
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
	print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))










																																																																																									
