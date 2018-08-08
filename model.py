import csv
import cv2
import os
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

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

def train(x, y, num_epochs=2):
	"""
	Nvdia's model.
	"""

	from keras.models import Sequential
	from keras.layers import Dense, Flatten, Dropout
	from keras.layers import Conv2D
	from keras.layers.pooling import MaxPooling2D
	from keras.layers.convolutional import Cropping2D, Convolution2D
	from keras.layers import Lambda
	from keras.optimizers import Adam

	model = Sequential()
	model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
	model.add(Cropping2D(cropping=((75,25), (0,0))))
	model.add(Convolution2D(24,5,5,activation='relu'))
	model.add(MaxPooling2D())
	model.add(Convolution2D(36,5,5,activation='relu'))
	model.add(Convolution2D(48,5,5,activation='relu'))
	model.add(Convolution2D(64,3,3,activation='relu'))
	model.add(Convolution2D(64,3,3,activation='relu'))
	model.add(MaxPooling2D())
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))

	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-9, decay=0.0001)
	model.compile(loss='mse', optimizer=adam)

	# 80/20 split, shuffled
	model.fit(x, y, validation_split=.2, shuffle=True, epochs=num_epochs)
	model.save("model.h5")

def lenet(x, y, num_epochs=5):
	"""
	Lenet training model
	"""

	from keras.models import Sequential
	from keras.layers import Dense, Flatten, Dropout
	from keras.layers import Conv2D
	from keras.layers.pooling import MaxPooling2D
	from keras.layers.convolutional import Cropping2D, Convolution2D
	from keras.layers import Lambda
	from keras.optimizers import Adam

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

	# 80/20 split, shuffled
	model.fit(x, y, validation_split=.2, shuffle=True, epochs=num_epochs)
	model.save("model.h5")

def main():
	lines = read_from_csv("./training_data")
	print(len(lines))

	# Load image and convert from BGR to RGB
	images = [cv2.cvtColor(cv2.imread(i[0]), cv2.COLOR_BGR2RGB) for i in lines]
	meas = [float(l[3]) for l in lines]

	# Flip the image
	aug_images = [cv2.flip(i, 1) for i in images]
	aug_meas = [-1*m for m in meas]

	# Append original images to flipped images
	aug_images.extend(images)
	aug_meas.extend(meas)

	X_train = np.array(aug_images)
	y_train = np.array(aug_meas)

	train(X_train, y_train)

if __name__ == "__main__":
	main()
