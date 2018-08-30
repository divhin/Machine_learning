
#Importing the necessary packages
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

#Loading the data and assigning to X and Y vectors
mat = scipy.io.loadmat('hw2data_2.mat')
x = mat['X']
y = mat['Y']


def sigmoid(z):
	"""
	Sigmoid function definition - Calculates and returns the Sigmoid of a matrix input z
		input z: a numpy matrix of any shape
	
		output s: a numpy matrix of the same shape as z
	"""
	s = 1/(1 + np.exp(-z))
	return s

def forward_propogation(x,w1,b1,w2,b2):
	"""
	Forward propogation definition - Calculates the output of the Neural Network
		input x: training samples array of shape ()
		inputs w1: Hidden Layer 1 weight matrix array of shape ()
		input b1: Hidden Layer 1 bias vector of shape ()
		inputs w2: Hidden Layer 2 weight matrix array of shape ()
		input b2: Hidden Layer 2 bias vector of shape ()

		output: The ouputs of the computaion steps of both the layers
		a2, a1: The output of the sigmoid function of layer 1 and layer 2 respectively
		z2, z1: The output of the linear function of layer 1 and layer 2 respectively
	"""
	
	#Layer-1 computation
	z1 = np.dot(x,w1.T) + b1.T
	a1 = sigmoid(z1)
	
	#Layer-2 computaion
	z2 = np.dot(a1,w2) + b2.T
	a2 = sigmoid(z2)

	return (a2,z2,a1,z1)

def backward_propogation(x,w1,b1,w2,b2,del_z,a2,z2,a1,z1):
	"""
	input x: training samples array of shape ()
	inputs w1: Hidden Layer 1 weight matrix array of shape ()
	input b1: Hidden Layer 1 bias vector of shape ()
	inputs w2: Hidden Layer 2 weight matrix array of shape ()
	input b2: Hidden Layer 2 bias vector of shape ()
	input del_z: The gradient of the error function with respect to the predicted value
	input a1, a2: The output of the sigmoid function of layer 1 and layer 2 respectively
	input z1, z2: The output of the linear function of layer 1 and layer 2 respectively
	
	output: The updated weights 
	"""

	del_e_w2 = np.multiply(del_z, a1).mean(axis=0)

	del_e_b2 = del_z.mean(axis=0)

	sigma_z1_term = np.multiply(sigmoid(z1),1-sigmoid(z1))

	del_e_w1 = (del_z.dot(w2.T)*(sigma_z1_term*x)).mean(axis=0)

	del_e_b1 = np.multiply(del_z.dot(w2.T),sigma_z1_term).mean(axis=0)    

	w2_new = w2 - eta* np.reshape(del_e_w2, (del_e_w2.shape[0],1))
	b2_new = b2 - eta* np.reshape(del_e_b2, (del_e_b2.shape[0],1))
	w1_new = w1 - eta* np.reshape(del_e_w1, (del_e_w1.shape[0],1))
	b1_new = b1 - eta* np.reshape(del_e_b1, (del_e_b1.shape[0],1))

	return (w2_new,b2_new,w1_new,b1_new)


n = x.shape[0]
d = x.shape[1]
k = 25

#Giving a seed to the numpy random function for reproducibility
np.random.seed(0)

#Initializing weights and bias randomly
w1 = np.random.randn(k,d)
b1 = np.random.randn(k,1) 
w2 = np.random.randn(k,1)
b2 = np.random.randn(1)

epoch = 0
#eta: Learning rate for updating weights
eta = 0.1

#minerror: error threshold for stopping further weight updates and model training
minerror=0.00015

#error: the training error of the model. Initializing it with a random large value to start the loop.
error = 10000

#training the model
while(error > minerror):
	epoch += 1
	(a2,z2,a1,z1) = forward_propogation(x,w1,b1,w2,b2)
	y_nn = a2
	error = ((1/(2*n))*np.dot((y_nn - y).T, (y_nn - y)))[0][0]    
	sigma_z2_term = np.multiply(sigmoid(z2),(1- sigmoid(z2)))
	del_z= np.multiply(y_nn-y,sigma_z2_term)    
	(w2,b2,w1,b1)= backward_propogation(x,w1,b1,w2,b2,del_z,a2,z2,a1,z1)

	#reducing learning rate with every 100000 epochs
	if epoch % 100000 == 0:
	    eta /= 2

	if epoch > 1400000:
		break


plt.figure(figsize=(10,10))
plt.scatter(x,y,alpha=0.7,s=1,c='b')
plt.scatter(x,y_nn,c='r',alpha=0.7,s=1)
plt.xlabel('X')
plt.ylabel('Y')
plt.suptitle('X vs Y Plot')
plt.grid()
plt.show()