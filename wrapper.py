from data_handler import DataHandler
from plotting import Plotter
from joblib import Parallel, delayed
import numpy as np

class Handler():

	def __init__(self, train_data, test_data, clean_info, cleaner_file, cleaner_class, split_size, split_type, classifier, threads):
		self.data_handler = DataHandler(train_data, test_data, clean_info, cleaner_file, cleaner_class, split_size, split_type)
		self.classifier = classifier
		self.threads = threads
		self.plotter = Plotter(style="ggplot", verbose=True)
		self.decisions = []
		self.infos = []


	def cross_validate(self):
		results = Parallel(n_jobs=self.threads)(delayed(self.classifier.test)(X_train, y_train, train_info, X_test, y_test, test_info) for X_train, y_train, train_info, X_test, y_test, test_info in self.data_handler.cross_validation("book"))
		for tuple in results:
			self.decisions.append(tuple[0])
			self.infos.append(tuple[1])
			#print(tuple)
			
	def optimize_C(self):
		C_values = [2**i for i in range(-30, 20)]
		accuracies = []
		for c_index, c_value in enumerate(C_values):
			print("Testing C: {}\tIteration: {}/{}".format(c_value, c_index, len(C_values)), end="\r")
			self.classifier.change_C(c_value)
			corrects = []
			results = Parallel(n_jobs=self.threads)(delayed(self.classifier.test)(X_train, y_train, train_info, X_test, y_test, test_info) for X_train, y_train, train_info, X_test, y_test, test_info in self.data_handler.cross_validation("book"))
			for tuple in results:
				if tuple[1][0] == tuple[2]:
					corrects.append(1)
				else:
					corrects.append(0)
			
			accuracies.append(sum(corrects) / len(corrects))
		sorted_indexes = np.argsort(accuracies)
		print("\nOptimal C:{}".format(C_values[sorted_indexes[-1]]))
		self.classifier.change_C(C_values[sorted_indexes[-1]])
			


	def attribute_testdata(self):
		X_train, y_train, train_info, X_test, y_test, test_info = self.data_handler.true_data("book")
		decision, test_info, got = self.classifier.test(X_train, y_train, train_info, X_test, y_test, test_info)
		self.decisions.append(decision)
		self.infos.append(test_info)
		print(test_info, decision)



	def plot_values(self):
		authors = self.get_authors()
		mapping = {authors[0]: "blue", authors[1]: "red"}
		self.plotter.plot(self.decisions, self.infos, mapping)


	def get_authors(self):
		authors = []
		for author in self.data_handler.train_data:
			authors.append(author)
		return authors
	
	def get_best_features(self, num_feats=50):
		X_train, y_train, _, _, _, _ = self.data_handler.true_data("book")
		features = self.classifier.get_best_features(X_train, y_train, num_feats)
		print("Positive features:\n")
		self.print_feat(features["POS"])
		print("Negative features:\n")
		self.print_feat(features["NEG"])
	
	def print_feat(self, feats):
		for feat in feats:
			print("{}\t{}".format(feat[0], feat[1]))
