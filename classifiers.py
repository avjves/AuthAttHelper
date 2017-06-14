from sklearn.svm import LinearSVC
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
from operator import itemgetter
import numpy as np

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
		results = Parallel(n_jobs=threads)(delayed(self.test)(X_train,y_train,train_info,X_test,y_test,test_info,iter,max_iter) for X_train,y_train,train_info,X_test,y_test,test_info,iter,max_iter in generator)
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


	def test(self, X_train, y_train, train_info, X_test, y_test, test_info, iter, max_iter, sample_weights=None):
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
		
		if single:
		
			if decision <= 0:
				got = classes[0]
			else:
				got = classes[1]
		else:
			got = None
		
		
		return decision, test_info, got==y_test[0]  
	
	def get_best_features(self, X, y, num_feats, num_iter=1):
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
			


class GI(ClassifierClass):
	
	def __init__(self, vectorizers=None, ngram_range=(1,2), analyzer="word"):
		ClassifierClass.__init__(self, vectorizers, ngram_range, analyzer)
		self.threshold = threshold
		
	def test(self, X_train, y_train, train_info, X_test, y_test, test_info, iter, max_iter):
		pass
	
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
		
		
		
			
			

class CNN():
	def __init__(self, args):
		self.args = args
		model = self.make_model()
	
		
	
	
		

