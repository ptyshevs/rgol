import numpy as np
import pandas as pd
import os

from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, BatchNormalization
from keras.callbacks import EarlyStopping

TRAIN_FILE = "./resources/train.csv"
TEST_FILE_Y = "./resources/test.csv"
TEST_FILE_X = "./resources/sampleSubmission.csv"

conv_shape = (20,20,1)
kernel = (5,5)

np.random.seed(42)

def create_model():
	model = Sequential()
	model.add(Conv2D(256, kernel_size=kernel, padding='same', activation='relu', input_shape=conv_shape))
	model.add(BatchNormalization())
	model.add(Conv2D(256, kernel_size=kernel, padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(256, kernel_size=kernel, padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(256, kernel_size=kernel, padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(256, kernel_size=kernel, padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(256, kernel_size=kernel, padding='same', activation='relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(1, kernel_size=kernel, padding='same', activation='sigmoid'))
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	return model

def create_matrices(df, delta=1):
	X = []
	y = []

	for i, row in df[df['delta'] == delta].iterrows():
		tmp_x = row.iloc[1:401].values.reshape((20,20))
		tmp_y = row.iloc[401:801].values.reshape((20,20))

		X.append(tmp_x)
		y.append(tmp_y)

	return X, y

def train(filename):
	train_data = pd.read_csv(filename, index_col="id")

	if not os.path.isdir("./model"):
		os.mkdir("model")

	for delta in [1,2,3,4,5]:
		print("Learning model for delta =", delta)
		X, y = create_matrices(train_data, delta=delta)

		X = np.array(X).reshape(len(X),20,20,1)
		y = np.array(y).reshape(len(y),20,20,1)

		model = create_model()
		es = EarlyStopping(monitor="loss", patience=9, min_delta=0.001)
		model.fit(X, y, epochs=50, validation_split=0.2, callbacks=[es])
		model.save_weights("./model/cnn_delta_" + str(delta) + ".h5")

def evaluate(X_file, y_file):
	test_data_y = pd.read_csv(y_file, index_col="id")
	test_data_x = pd.read_csv(x_file, index_col="id")
	test_data = pd.concat([test_data_x, test_data_y], axis=1)

	acc = 0.0

	for delta in [1,2,3,4,5]:
		X_test, y_test = create_matrices(test_data, delta=delta)
		X_test = np.array(X_test).reshape(len(X_test),20,20,1)
		y_test = np.array(y_test).reshape(len(y_test),20,20,1)

		model = create_model()
		model.load_weights("./model/cnn_delta_" + str(delta) + ".h5")
		loss = model.evaluate(X_test, y_test)
		acc += loss[1]

	acc = acc / 5
	print("Average accuracy =", acc)

	return acc

if __name__ == "__main__":
	train(TRAIN_FILE)
	evaluate(TEST_FILE_X, TEST_FILE_Y)


