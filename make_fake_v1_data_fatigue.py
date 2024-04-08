import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor_kernel
import tensorflow as tf

# This code is used to make the simulated data set of responses of n neurons to
# different numbers of images (i.e. see [1000, 3000, 10000, 30000] below)
# You can specify the number of neurons (n_neurons below) and the numbers of images 
# you want to expose each neuron to. The responses of each simulated neuron to each
# image in the training set are saved in a CSV file (see below)

# Import the cifar100 dataset from keras. It'll probably download it the first time.
# The tf.keras.datasets.cifar100.load_data() function in TensorFlow is used to load 
# the CIFAR-100 dataset, which is a collection of 60,000 32x32 color images in 100 different classes
# In Linux it will download to:
#  "~/.keras/datasets/" where ~ is your home directory
(x_train, _), (x_test, _) = tf.keras.datasets.cifar100.load_data()

#greyscale, retype and normalize data. Neural networks like their data to be floating points. If there were lots
#Of different data types, we'd probably want them all centered around 0... z-scored.
x_train = np.mean(x_train, axis=3).astype('float32') / 255
x_test = np.mean(x_test, axis=3).astype('float32') / 255
 
n_train_images, height, width = x_train.shape


#Our neuron class. Which is really just a mechanism
#to generate a little gabor filter and then do the dot-product
# with the image.

class Neuron:
	#__init__ is called whenever an new instance of the class
	# is made.
	def __init__(self, width, height):
		scale = np.random.random()*3 + 1
		self.pos_x = np.random.randint(width)
		self.pos_y = np.random.randint(height)
		self.freq = 0.1*scale
		self.theta = self._randpi()
		self.sigma_x = 3/scale
		self.sigma_y = 3/scale
		self.offset = self._randpi()
		self.rf = self._makerf(width, height)
		self.previous_activity = 0

	#__str__ is called whenever you try to print() and object
	# This was purely for debugging purposes
	def __str__(self):
		str = "pos_x = {}".format(self.pos_x) 
		str += "\npos_y = {}".format(self.pos_y)
		str += "\nfreq = {}".format(self.freq)
		str += "\ntheta = {}".format(self.theta)
		str += "\nsigma_x = {}".format(self.sigma_x)
		str += "\nsigma_y = {}".format(self.sigma_y)
		str += "\noffset = {}".format(self.offset)
		return str

	#Returns all the properties of the neuron. largely for debugging purposes
	def get_props(self):
		return [self.pos_x, self.pos_y, self.freq, self.theta, self.sigma_x, self.sigma_y, self.offset]

	#This initializes the receptive field. Python methods that start with _
	# are labelled to make them private. They're not *Actually* private
	# But it's just an indicator that "you porbably don't want to call"
	# this function".
	def _makerf(self, width, height):
		rf = np.zeros((height, width))
		k = np.real(gabor_kernel(self.freq, 
                           theta = self.theta, 
                           sigma_x = self.sigma_x, 
                           sigma_y = self.sigma_y, 
                           offset = self.offset))
		x_idx = np.arange(self.pos_x, self.pos_x+k.shape[1])%width
		y_idx = np.arange(self.pos_y, self.pos_y+k.shape[0])%height
		x_idx, y_idx = np.meshgrid(x_idx, y_idx)
		rf[y_idx, x_idx] = k
		rf = rf / np.sqrt(np.sum(rf * rf)) #scale so rf^2 = 1.
		return rf

	# The meat of the sandwhich. Just do element-wise multiplication of the receptive field
	# with the image. Sum it up, and multiply by 4. Then feed into a sigmoid, and tack on a random.
	def get_activity(self, input):
		# Calculate the basic activity as before
		basic_activity = np.sum(self.rf * input) * 4
		activity = 1 / (1 + np.exp(-basic_activity))
		adaptive_noise = 1
		# Introduce an adaptive noise component based on previous activity
		adaptive_noise_scale = max(abs(adaptive_noise * self.previous_activity), 0.01)
		adaptive_noise = np.random.normal(scale=adaptive_noise_scale)

		# Update the activity with adaptive noise and store it for next time
		final_activity = activity + adaptive_noise
		self.previous_activity = final_activity

		return final_activity

	#Just a helper method to generate a random number between 0 and pi.
	def _randpi(self):
		return np.random.rand()*2*np.pi


#Make 1000 (default) neurons and put them in a list
n_neurons = [250,500,1000,2000]
noise_scale = 0.1
for n_neurons in n_neurons:
	neurons = []
	for _ in range(n_neurons):
		neurons.append(Neuron(width,height))

	replicates = 1

	#loop over every training image, and repeat it replicates times.
	#build up a row to go in our results (output) table
	#that first contains a flattened version of our image
	#The for every neuron we have, feed it our image, and append the activity
	#to our row [1000, 3000, 10000, 30000]

	# numbers of images to generate simulated responses to:
	numbers_of_images = [1000, 3000, 10000, 30000]

	for n_img in numbers_of_images:
		print('Started generating responses to ' + str(n_img) + ' images...')
		results = []
		n_train_images = n_img
		for i in range(n_train_images):
			for r in range(replicates):
				row = []
				# flatten the image
				row += list(x_train[i,:,:].flatten())
				# calculate response of every neuron to the image
				for neuron in neurons:
					row.append(neuron.get_activity(x_train[i,:,:]))
			results.append(row)
			if i%1000 == 0:
				print("Calculated response to image {}".format(i))


		# Save the output
		# Output will be saved to the "home/username/data/Response_Simulation/A" folder
		save_dir = '~/data/Response_Simulation/A/'
		save_dir = os.path.expanduser(save_dir)
		# Check if the directory does not exist
		if not os.path.exists(save_dir):
			# Create the directory
			os.makedirs(save_dir)

		adaptive_noise=1

		fname = 'neurons_to_cifar_' + str(n_neurons) + 'n_' + str(replicates) +'rep' + str(n_img) + 'n_img'+'_'+str(adaptive_noise)+'fatigue'
		fname = os.path.join(save_dir,fname)
		np.savetxt(fname +'.csv', results, delimiter=',')

		neuron_prop = []
		for neuron in neurons:
			neuron_prop.append(neuron.get_props())

		#This contains the properties of the neurons
		np.savetxt(fname + '_neuron_prop.csv', neuron_prop, delimiter=',')