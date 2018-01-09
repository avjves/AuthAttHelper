import numpy as np
np.random.seed(1)




from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from joblib import Parallel, delayed
from operator import itemgetter
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Merge, Input, merge, Flatten,ActivityRegularization,Conv1D,GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from collections import defaultdict

class ClassifierClass():

	def __init__(self, vectorizers=None, ngram_range=(1,2), analyzer="word", verbose=True):
		self.verbose = verbose
		if not vectorizers:
			self.vectorizers = [TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer, sublinear_tf=True)]
		else:
			self.vectorizers = vectorizers

	def vectorize(self, data, train):
		vects = []
		for vectorizer in self.vectorizers:
			if train:
				vects.append(vectorizer.fit_transform(data))
			else:
				vects.append(vectorizer.transform(data))

		return hstack(vects)

	def classify(self, threads, generator, clsf_level):
		if clsf_level == "multiclass":
			multiclass = True
		else:
			multiclass = False
		results = Parallel(n_jobs=threads)(delayed(self.test)(X_train,y_train,train_info,X_test,y_test,test_info,iter,max_iter, multiclass=multiclass) for X_train,y_train,train_info,X_test,y_test,test_info,iter,max_iter in generator)
		return results

	def test(self):
		pass

class SVM(ClassifierClass):

	def __init__(self, C, vectorizers=None, ngram_range=(1,2), analyzer="word", verbose=True):
		ClassifierClass.__init__(self, vectorizers, ngram_range, analyzer, verbose)
		self.classifier = LinearSVC(C=C, class_weight="balanced")

	def change_C(self, C):
		self.classifier.set_params(C=C)

	def fit(self, X, y, sample_weights=None):
		if sample_weights:
			self.classifier.fit(X, y, sample_weights)
		else:
			self.classifier.fit(X, y)

	def decision(self, X_test):
		return self.classifier.decision_function(X_test)

	def predict(self, X_train, y_train, X_test, y_test):
		X_train = self.vectorize(X_train, True)
		X_test = self.vectorize(X_test, False)

		self.fit(X_train, y_train)

		predictions = self.classifier.predict(X_test)

		return predictions


	def test(self, X_train, y_train, train_info, X_test, y_test, test_info, iter, max_iter, sample_weights=None, multiclass=False):
		if self.verbose:
			print("Iteration: {} / {}".format(iter, max_iter), end="\r")

		if len(X_test) == 1:
			single = True
		else:
			single = False


		X_train = self.vectorize(X_train, True)
		X_test = self.vectorize(X_test, False)


		self.fit(X_train, y_train, sample_weights)
		decision = self.decision(X_test)

		classes = self.classifier.classes_

		if multiclass:
			return decision, test_info, classes
		else:
			if single:

				if decision <= 0:
					got = classes[0]
				else:
					got = classes[1]
			else:
				got = None
			return decision, test_info, got==y_test[0]

	def get_best_features(self, X, y, num_feats, num_iter=100):
		all_feats = {}
		X = self.vectorize(X, True)
		print("\n")
		for i in range(num_iter):
			print("Iteration {}/{}".format(i, num_iter), end="\r")
			clsf = LinearSVC(C=10000, penalty="l1", dual=False)
			clsf.fit(X, y)
			coef = clsf.coef_
			vocab = self.get_vocab()
			feats = self.coef_to_feats(coef, vocab)
			for feat_key, feat_value in feats.items():
				all_feats[feat_key] = all_feats.get(feat_key, 0) + feat_value

		feat_list = []
		for key, value in all_feats.items():
			feat_list.append([key, value/num_iter])
		feat_list.sort(key=itemgetter(1))

		feats = {"POS": [], "NEG": []}
		for feat in feat_list:
			if feat[1] < 0 and len(feats["NEG"]) < num_feats:
				feats["NEG"].append(feat)
			else:
				continue
		for feat in reversed(feat_list):
			if feat[1] > 0 and len(feats["POS"]) < num_feats:
				feats["POS"].append(feat)
			else:
				continue

		return feats

	def get_vocab(self):
		feats = {}
		feat_count = 0
		for vectorizer in self.vectorizers:
			vocab = vectorizer.vocabulary_
			for key, value in vocab.items():
				feats[value + feat_count] = key
			feat_count += len(vocab)
		return feats

	def coef_to_feats(self, coef, vocab):
		feats = {}
		nonzeros = np.nonzero(coef)
		for val in nonzeros[1]:
			feats[vocab[val]] = coef[0][val]
		return feats


 ##TODO Not done
class GI(ClassifierClass):

	def __init__(self, vectorizers=None, ngram_range=(1,2), analyzer="word"):
		ClassifierClass.__init__(self, vectorizers, ngram_range, analyzer)
		self.threshold = threshold

	def test(self, X_train, y_train, train_info, X_test, y_test, test_info, iter, max_iter):
		pass

## TODO Not done
class VectDist(ClassifierClass):

	def __init__(self, vectorizers=None, ngram_range=(1,2), analyzer="word"):
		ClassifierClass.__init__(self, vectorizers, ngram_range, analyzer)

	def classify(self, threads, generator, clsf_level):
		results = Parallel(n_jobs=threads)(delayed(test)(X_train,y_train,train_info,X_test,y_test,test_info,iter,max_iter) for X_train,y_train,train_info,X_test,y_test,test_info,iter,max_iter in generator)
		if clsf_level == "CV":
			vals = numpy.array([res[0] for res in results])
			min_val, max_val = np.min(vals[np.nonzero(vals)]), np.max(vals[np.nonzero(vals)])
			self.min_val = min_val
			self.max_val = max_val
			return results
		elif clsf_level == "test":
			for result in results:
				return "LO"



	def calculate_author_style(data):
		#data = data.todense()
		means = np.mean(data, axis=1)
		return means


	def test(self, X_train, y_train, train_info, X_test, y_test, test_info, iter, max_iter):
		## two_choices here, all neg = class or ever neg has its own style, first all neg = class
		positive_indexes = [i for i in range(0, len(X_train)) if y_train[i] == 1]
		negative_indexes = [i for i in range(0, len(X_train))] - positive_indexes
		X_train = self.vectorize(X_train, True)
		X_test = self.vectorize(X_test, False)
		positive_style_vect = self.calculate_author_style(X_train[positive_indexes])
		#negative_style_vect = self.calculate_author_style(X_train[negative_indexes])
		test_case_style = calculate_author_style(X_test)
		dist = self.minmax(test_case_style, positive_style_vect)

		return dist, test_info


	def minmax(self, a, b):
		min_sum, max_sum = 0, 0
		for i in range(0, len(a)):
			min_sum += min(a[i], b[i])
			max_sum += max(a[i], b[i])

		return min_sum / max_sum






class CNN:
	def __init__(self, max_length, vector_size, minibatch_size, ngram, epochs):
		self.max_length = max_length
		self.vector_size = vector_size
		self.minibatch_size = minibatch_size
		self.ngram = ngram
		self.epochs = epochs
		self.level = "char"


	def make_model(self, verbose=True, weights=None):
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


	def ngram_generator(self, text, level, ngram_length, max_length=1000000000):
		if level == "word":
			text = text.split(" ")
		for i in range(0, len(text)-ngram_length):
			if i >= max_length: break
			yield text[i:i+ngram_length]

	def read_vocabulary(self, data):
		vocab = {}
		vocab["<NOTFOUND>"] = 0
		feat_id = 1
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
				X[row, column] = self.vocab.get(ngram, 0)
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

	# def optimize_epochs(self, datagen, epoch_range):
	# 	n_splits = 5
	# 	kf = KFold(n_splits=n_splits, shuffle=True, random_state=100)
	# 	X, y, info, _, _, _, _, _ = next(datagen)
	# 	results = []
	# 	for e in epoch_range:
	# 		print("Testing epoch count: {}".format(e))
	# 		e_res = []
	# 		foldn = 0
	# 		for train_indexes, test_indexes in kf.split(X):
	# 			print("Fold: {}/{}".format(foldn+1, n_splits))
	# 			foldn += 1
	# 			X_train, y_train, train_info = [X[i] for i in train_indexes], [y[i] for i in train_indexes], [info[i] for i in train_indexes]
	# 			X_test, y_test, test_info = [X[i] for i in test_indexes], [y[i] for i in test_indexes], [info[i] for i in test_indexes]
	#
	# 			if type(X_train) == list: ## book_split
	# 				X_train, y_train, train_info = self.extract_nested(X_train, y_train, train_info)
	# 				X_test, y_test, test_info = self.extract_nested(X_test, y_test, test_info)
	#
	# 			self.vocab = self.read_vocabulary(X_train)
	# 			self.model = self.make_model(verbose=False)
	#
	# 			X_train, y_train = self.matrixify(X_train, y_train, train_info)
	# 			X_test, y_test = self.matrixify(X_test, y_test, test_info)
	#
	# 			res = self.model.fit(X_train, y_train, batch_size=self.minibatch_size, shuffle=False, validation_data=(X_test, y_test), epochs=e)
	# 			e_res.append(res.history["val_acc"][-1])
	# 		results.append(sum(e_res) / len(e_res))
	# 		print(results[-1])

	def optimize_epochs(self, datagen, epoch_range):
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

			if type(X_train) == list: ## book_split
				X_train, y_train, train_info = self.extract_nested(X_train, y_train, train_info)
				X_test, y_test, test_info = self.extract_nested(X_test, y_test, test_info)

			##Shuffle data here
			X_train, y_train, train_info = shuffle(X_train, y_train, train_info, random_state=1)
			X_test, y_test, test_info = shuffle(X_test, y_test, test_info, random_state=1)

			self.vocab = self.read_vocabulary(X_train)
			self.model = self.make_model(verbose=False)

			X_train, y_train = self.matrixify(X_train, y_train, train_info)
			X_test, y_test = self.matrixify(X_test, y_test, test_info)

			cb = EarlyStopping(monitor="val_acc", verbose=1, patience=10)

			res = self.model.fit(X_train, y_train, batch_size=self.minibatch_size, shuffle=False, validation_data=(X_test, y_test), epochs=list(epoch_range)[-1], callbacks=[cb])
			results.append(res.history)
			#print(res.history)
		maxn = 0

		for i in len(results[0]["val_acc"]):
			val = sum(results[i]["val_acc"]) / len(results[i]["val_acc"])
			maxn += max(results[i]["val_acc"])
			print("Epoch: {}\t Value: {}".format(i, val))

		print("Max: {}".format(maxn / len(results[0]["val_acc"])))
	#	print(results)
		return results

	def classify(self, threads, data_gen, classification_type):
		predictions = []
		for X_train, y_train, train_info, X_test, y_test, test_info, iteration, max_iter in data_gen:
			print(test_info)

			if type(X_train) == list: ## book_split
				X_train, y_train, train_info = self.extract_nested(X_train, y_train, train_info)
				X_test, y_test, test_info = self.extract_nested(X_test, y_test, test_info)

			##Shuffle data here
			X_train, y_train, train_info = shuffle(X_train, y_train, train_info, random_state=1)
			X_test, y_test, test_info = shuffle(X_test, y_test, test_info, random_state=1)

			self.vocab = self.read_vocabulary(X_train)
			self.model = self.make_model()

			X_train, y_train = self.matrixify(X_train, y_train, train_info)
			X_test, y_test = self.matrixify(X_test, y_test, test_info)

			es = EarlyStopping(monitor="val_acc", verbose=1, patience=4)
			model_path = "models/model_{}_{}.hdf5".format(self.ngram, iteration)
			mc = ModelCheckpoint(model_path, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
			self.model.fit(X_train, y_train, batch_size=self.minibatch_size, shuffle=False, validation_data=(X_test, y_test), epochs=self.epochs, callbacks=[es, mc])
			#self.make_model(weights="models/model.hdf5")
			self.model.load_weights(model_path)
			prediction = self.model.predict(X_test)

			predictions.append((prediction, test_info, y_test))
		print(predictions)
		return predictions
