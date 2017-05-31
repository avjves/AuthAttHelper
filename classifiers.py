from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from operator import itemgetter
import numpy as np
class SVM():

	def __init__(self, C, vectorizers=None, ngram_range=(1,2), analyzer="word"):
		self.classifier = LinearSVC(C=C)
		if not vectorizers:
			self.vectorizers = [TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer, sublinear_tf=True)]
		else:
			self.vectorizers = vectorizers
			
	def change_C(self, C):
		self.classifier.set_params(C=C)

	def fit(self, X, y):
		self.classifier.fit(X, y)

	def decision(self, X_test):
		return self.classifier.decision_function(X_test)

	def vectorize(self, data, train):
		vects = []
		for vectorizer in self.vectorizers:
			if train:
				vects.append(vectorizer.fit_transform(data))
			else:
				vects.append(vectorizer.transform(data))
		
		return hstack(vects)
	
	def test(self, X_train, y_train, train_info, X_test, y_test, test_info):
		#Xtr, Xte = [], []
		#Xtr, Xte = self.vectorize(X_train, true), self.vectorize(X_test, false)
		#for vectorizer in self.vectorizers:
		#	Xtr.append(vectorizer.fit_transform(X_train))
		#	Xte.append(vectorizer.transform(X_test))
		#X_train = hstack(Xtr)
		#X_test = hstack(Xte)
	
		X_train = self.vectorize(X_train, True)
		X_test = self.vectorize(X_test, False)

		self.fit(X_train, y_train)
		decision = self.decision(X_test)
		
		classes = self.classifier.classes_
		if decision <= 0:
			got = classes[0]
		else:
			got = classes[1]
			
		#print(decision, test_info)
		return decision, test_info, got  
	
	def get_best_features(self, X, y, num_feats, num_iter=100):
		all_feats = {}
		X = self.vectorize(X, True)
		
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
			if feat[1] > 0 and len(feats["POS"]) < num_feats:
				feats["POS"].append(feat)
			elif feat[1] < 0 and len(feats["NEG"]) < num_feats:
				feats["NEG"].append(feat)
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
			
		
		

