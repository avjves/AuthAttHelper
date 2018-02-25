import numpy as np
np.random.seed(1)

from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from joblib import Parallel, delayed
from operator import itemgetter
from collections import defaultdict


class LinearClassifierClass:

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


	def classify(self, threads, generator, clsf_level):
		if clsf_level == "multiclass":
			multiclass = True
		else:
			multiclass = False
		results = Parallel(n_jobs=threads)(delayed(self.test)(X_train,y_train,train_info,X_test,y_test,test_info,iter,max_iter, multiclass=multiclass) for X_train,y_train,train_info,X_test,y_test,test_info,iter,max_iter in generator)
		return results

	def test(self):
		pass

class SVM(LinearClassifierClass):
	def __init__(self, C, vectorizers=None, ngram_range=(1,2), analyzer="word", verbose=True):
		super().__init__(vectorizers, ngram_range, analyzer, verbose)
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

		if type(X_train) == list: ## book_split
			X_train, y_train, train_info = self.extract_nested(X_train, y_train, train_info)
			X_test, y_test, test_info = self.extract_nested(X_test, y_test, test_info)


		X_train = self.vectorize(X_train, True)
		X_test = self.vectorize(X_test, False)

		self.fit(X_train, y_train)

		predictions = self.classifier.predict(X_test)

		return predictions


	def test(self, X_train, y_train, train_info, X_test, y_test, test_info, iter, max_iter, sample_weights=None, multiclass=False):


		if type(X_train[0]) == list: ## book_split
			X_train, y_train, train_info = self.extract_nested(X_train, y_train, train_info)
			X_test, y_test, test_info = self.extract_nested(X_test, y_test, test_info)
			if len(X_train) == 0 or len(X_test) == 0:
				return []

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

	def get_best_features_nsampling(self, X, y, text_percentage, num_iter, num_feats=50):
		feat_count = {}
		feat_values = {}
	#	X = self.vectorize(X, True)
		print()
		for i in range(num_iter):
			print("Iteration {}/{}".format(i, num_iter), end="\r")
			sample_indexes = np.random.choice(np.arange(0, len(X)), size=int(len(X)*text_percentage), replace=False)
			iter_X, iter_y = [X[i] for i in sample_indexes], [y[i] for i in sample_indexes]

			iter_X = self.vectorize(iter_X, True)
			clsf = LinearSVC(C=10000, penalty="l1", dual=False)
			clsf.fit(iter_X, iter_y)
			coef = clsf.coef_
			vocab = self.get_vocab()
			feats = self.coef_to_feats(coef, vocab)
			for feat_key, feat_value in feats.items():
				feat_count[feat_key] = feat_count.get(feat_key, 0) + 1
				feat_values[feat_key] = feat_values.get(feat_key, 0) + feat_value

		feat_list = []
		for key in feat_count:
			feat_list.append([key, feat_count[key], feat_values[key]])

		feat_list.sort(key=itemgetter(1, 2), reverse=True)


		feats = {"POS": [], "NEG": []}
		for feat in feat_list:
			if feat[2] < 0 and len(feats["NEG"]) < num_feats:
				feats["NEG"].append(feat)
			else:
				continue

		for feat in reversed(feat_list):
			if feat[2] > 0 and len(feats["POS"]) < num_feats:
				feats["POS"].append(feat)
			else:
				continue
		return feats


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
