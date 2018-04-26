import numpy as np
np.random.seed(1)

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten,ActivityRegularization,Conv1D,GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from operator import itemgetter
from collections import defaultdict

import os


class CNN:

	def __init__(self, max_length, vector_size, minibatch_size, ngram, epochs):
		self.max_length = max_length
		self.vector_size = vector_size
		self.minibatch_size = minibatch_size
		self.ngram = ngram
		self.epochs = epochs
		self.level = "char"


	def make_model_old(self, verbose=True, weights=None):
		input_layer = Input(shape=(self.max_length,), name="input_ngrams_{}".format(self.ngram), dtype="int32")
		embedding = Embedding(len(self.vocab), self.vector_size, input_length=self.max_length)(input_layer)
		conv = Conv1D(self.vector_size, 5, padding="same", activation="relu")(embedding)
		pool = GlobalMaxPooling1D()(conv)
		dense = Dense(self.vector_size)(pool)
		out = Dense(1, activation="sigmoid")(dense)
		optimizer = Adam(lr=0.0001)
		model = Model(inputs=input_layer, outputs=out)
		model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
		if verbose:
			model.summary()
		return model

	def make_model(self, verbose=True):
		input_layer = Input(shape=(self.max_length,), dtype="int32")
		embeddings = Embedding(len(self.vocab), 200, input_length=self.max_length)(input_layer)
		conv_3 = Conv1D(200, 3, activation="relu")(Conv1D(400, 3, activation="relu")(embeddings))
		conv_4 = Conv1D(200, 3, activation="relu")(Conv1D(400, 4, activation="relu")(embeddings))
		conv_5 = Conv1D(200, 3, activation="relu")(Conv1D(400, 5, activation="relu")(embeddings))
		pool_3 = GlobalMaxPooling1D()(conv_3)
		pool_4 = GlobalMaxPooling1D()(conv_4)
		pool_5 = GlobalMaxPooling1D()(conv_5)
		conc = Concatenate(axis=1)([pool_3, pool_4, pool_5])
		drop = Dropout(0.25)(conc)
		dense = Dense(150, activation="tanh")(drop)
		out = Dense(1, activation="sigmoid")(dense)

		model = Model(inputs=[input_layer], outputs=[out])
		optimizer = Adam(lr=0.0001)
		model.compile(optimizer=optimizer, metrics=["accuracy"], loss="binary_crossentropy")
		if verbose:
			model.summary()
		return model



	def ngram_generator(self, text, level, ngram_length, max_length=1000000000):
		if level == "word":
			text = text.split(" ")
		for i in range(0, len(text)-ngram_length):
			if i >= max_length: break
			yield text[i:i+ngram_length]

	def read_vocabulary(self, data):
		vocab = {}
		vocab["<MASK>"] = 0
		vocab["<NOTFOUND>"] = 1
		feat_id = 2
		for text in data:
			for ngram in self.ngram_generator(text, self.level, self.ngram):
				if ngram not in vocab:
					vocab[ngram] = feat_id
					feat_id += 1
		return vocab

	def matrixify(self, X_data, y_data, info):
		X = np.zeros((len(X_data), self.max_length), np.int)
		y = np.zeros((len(X_data), 1), np.int) ## 2 classes == binary
		for row, text in enumerate(X_data):
			for column, ngram in enumerate(self.ngram_generator(text, self.level, self.ngram, self.max_length)):
				X[row, column] = self.vocab.get(ngram, 1)
			y[row] = info[row][3]
		return X, y

	def optimize_parameters(self, parameters, data):
		optimized_params = {}
		print(parameters)
		for key, value in parameters.items():
			if key == "epochs":
				optimized_params[key] = self.optimize_epochs(data, value)
		print(optimized_params)
		return optimized_params

	def extract_nested(self, X, y, I):
		nX = []
		nY = []
		nI = []
		for i in range(len(X)):
			book = X[i]
			auth = y[i]
			info = I[i]
			for text in book:
				nX.append(text)
				nY.append(auth)
				nI.append(info)
		return nX, nY, nI

	def optimize_parameters(self, parameters, datagen):
		results = self.optimize(datagen, parameters["epochs"])

	def optimize(self, datagen, epoch_count):
		n_splits = 5
		kf = KFold(n_splits=n_splits, shuffle=True, random_state=100)
		X, y, info, _, _, _, _, _ = next(datagen)
		results = []
		foldn = 0
		for train_indexes, test_indexes in kf.split(X):
			print("Fold: {}/{}".format(foldn+1, n_splits))
			foldn += 1
			X_train, y_train, train_info = [X[i] for i in train_indexes], [y[i] for i in train_indexes], [info[i] for i in train_indexes]
			X_test, y_test, test_info = [X[i] for i in test_indexes], [y[i] for i in test_indexes], [info[i] for i in test_indexes]

			if type(X_train[0]) == list: ## book_split
				X_train, y_train, train_info = self.extract_nested(X_train, y_train, train_info)
				X_test, y_test, test_info = self.extract_nested(X_test, y_test, test_info)

			##Shuffle data here
			X_train, y_train, train_info = shuffle(X_train, y_train, train_info, random_state=1)
			X_test, y_test, test_info = shuffle(X_test, y_test, test_info, random_state=1)

			self.vocab = self.read_vocabulary(X_train)
			self.model = self.make_model(verbose=True)

			X_train, y_train = self.matrixify(X_train, y_train, train_info)
			X_test, y_test = self.matrixify(X_test, y_test, test_info)

			cb = EarlyStopping(monitor="val_acc", verbose=1, patience=10000)

			res = self.model.fit(X_train, y_train, batch_size=self.minibatch_size, shuffle=True, validation_data=(X_test, y_test), epochs=epoch_count, callbacks=[cb])
			results.append(res.history)
			#print(res.history)
		maxn = 0

		for i in len(results[0]["val_acc"]):
			val = sum(results[i]["val_acc"]) / len(results[i]["val_acc"])
			maxn += max(results[i]["val_acc"])
			print("Epoch: {}\t Value: {}".format(i, val))

		print("Max: {}".format(maxn / len(results[0]["val_acc"])))
		return results

	def classify(self, threads, data_gen, classification_type):
		predictions = []
		for X_train, y_train, train_info, X_test, y_test, test_info, iteration, max_iter in data_gen:
			print(test_info)

			if type(X_train[0]) == list: ## book_split
				X_train, y_train, train_info = self.extract_nested(X_train, y_train, train_info)
				X_test, y_test, test_info = self.extract_nested(X_test, y_test, test_info)

			##Shuffle data here
			X_train, y_train, train_info = shuffle(X_train, y_train, train_info, random_state=1)
			X_test, y_test, test_info = shuffle(X_test, y_test, test_info, random_state=1)

			self.vocab = self.read_vocabulary(X_train)
			self.model = self.make_model()

			X_train, y_train = self.matrixify(X_train, y_train, train_info)
			X_test, y_test = self.matrixify(X_test, y_test, test_info)

			es = EarlyStopping(monitor="val_acc", verbose=1, patience=400) ## don't stop
			model_path = "models/model_{}_{}.hdf5".format(self.ngram, iteration)
			if not os.path.exists("models"):
				os.makedirs("models")

			mc = ModelCheckpoint(model_path, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
			self.model.fit(X_train, y_train, batch_size=self.minibatch_size, shuffle=True, validation_data=(X_test, y_test), epochs=self.epochs, callbacks=[es, mc])
			#self.make_model(weights="models/model.hdf5")
			self.model.load_weights(model_path)
			prediction = self.model.predict(X_test)

			predictions.append((prediction, test_info, y_test))
		print(predictions)
		return predictions
