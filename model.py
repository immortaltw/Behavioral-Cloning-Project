import csv
import cv2
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Cropping2D, Convolution2D
from keras.layers import Lambda
from keras.optimizers import Adam

def read_from_csv(data_dir):
	"""
	Read csv files from data_dir to a list and return lines as list
	"""

	lines = []

	for d in os.listdir(data_dir):
		if "data" not in d:
			continue
		with open(os.path.join(data_dir, os.path.join(d, "driving_log.csv")), 'r') as f:
			reader = csv.reader(f)
			for row in reader:
				for i in range(0, 3):
					row[i] = os.path.join(data_dir, os.path.join(d, "IMG"))+ "/" + row[i].split('/')[-1]
				lines.append(row)

	return lines

def read_image_and_angle(lines):
	"""
	Utility function to read image and angle

	return: (images, angles)
	"""

	images = []
	angles = []
	aug_images = []
	aug_angles = []

	# Load image and convert from BGR to RGB
	images = [cv2.cvtColor(cv2.imread(i[0]), cv2.COLOR_BGR2RGB) for i in lines]
	angles = [float(l[3]) for l in lines]

	# Flip the image
	aug_images = [cv2.flip(i, 1) for i in images]
	aug_angles = [-1*m for m in angles]

	# Append original images to flipped images
	aug_images.extend(images)
	aug_angles.extend(angles)

	return aug_images, aug_angles

def generator(data, batch_size=32):
	"""
	Training and test set generator.
	"""

	data_size = len(data)
	while True:
		sklearn.utils.shuffle(data)

		images = []
		angles = []
		aug_images = []
		aug_angles = []

		for i in range(0, data_size, batch_size):
			cur_batch = data[i:i+batch_size]

			images = [cv2.imread(i[0]) for i in cur_batch]
			angles = [float(l[3]) for l in cur_batch]

			# Flip the image
			aug_images = [cv2.flip(i, 1) for i in images]
			aug_angles = [-1*m for m in angles]

			# Append original images to flipped images
			aug_images.extend(images)
			aug_angles.extend(angles)

		X_train = np.array(aug_images)
		y_train = np.array(aug_angles)

		yield sklearn.utils.shuffle(X_train, y_train)

def nvidia():
	"""
	Nvdia's model.

	return: model
	"""

	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((75,25), (0,0))))
	model.add(Convolution2D(24,5,5,activation='elu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(36,5,5,activation='elu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(48,5,5,activation='elu'))
	model.add(Convolution2D(64,3,3,activation='elu'))
	model.add(Convolution2D(64,3,3,activation='elu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(1140))
	model.add(Dropout(0.5))
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-9, decay=0.0001)
	model.compile(loss='mse', optimizer=adam)

	return model

def lenet():
	"""
	Lenet training model

	return: model
	"""

	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((75,25), (0,0))))
	model.add(Convolution2D(6,5,5,activation='relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(6,5,5,activation='relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(1))

	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-9, decay=0.0001)
	model.compile(loss='mse', optimizer=adam)

	return model

def main():

	# In my environment, generator run 100 times slower so I turned it off.
	use_generator = False
	num_epochs = 3

	lines = read_from_csv("./training_data")
	lines = sklearn.utils.shuffle(lines)
	train_samples_, test_samples = train_test_split(lines, test_size=0.2)
	train_samples, validation_samples = train_test_split(train_samples_, test_size=0.2)

	model = nvidia()

	if use_generator:
		train_generator = generator(train_samples)
		valid_generator = generator(validation_samples)
		model.fit_generator(train_generator, samples_per_epoch= \
	            len(train_samples)/32, validation_data=valid_generator, \
	            nb_val_samples=len(validation_samples)/32, nb_epoch=num_epochs)
	else:
		train_images, train_angles = read_image_and_angle(train_samples)
		X_train = np.array(train_images)
		y_train = np.array(train_angles)
		valid_images, valid_angles = read_image_and_angle(validation_samples)
		X_valid = np.array(valid_images)
		y_valid = np.array(valid_angles)

		model.fit(X_train, y_train, validation_data=(X_valid, y_valid), shuffle=True, epochs=num_epochs)

	model.save("model.h5")

	test_images, test_angles = read_image_and_angle(test_samples)
	X_test = np.array(test_images)
	y_test = np.array(test_angles)
	test_result = model.evaluate(X_test, y_test)
	print(test_result)

if __name__ == "__main__":
	main()
