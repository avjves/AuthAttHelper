from data_handler import DataHandler
from plotting import Plotter
from joblib import Parallel, delayed
import numpy as np
from operator import itemgetter
from collections import OrderedDict

class Handler():

	def __init__(self, train_data, test_data, positive_classes, split_size, split_type, data_type, classifier, threads, iter=-1, clean_stuff=None):
		self.data_handler = DataHandler(train_data, test_data, positive_classes, split_size, split_type, iter, clean_stuff)
		self.classifier = classifier
		self.threads = threads
		self.data_type = data_type
		self.positive_classes = positive_classes
		self.plotter = Plotter(style="ggplot")
		self.decisions = []
		self.infos = []


	def cross_validate(self):
		results = self.classifier.classify(self.threads, self.data_handler.cross_validation(self.data_type), "CV")
		#tuple is (classification value, information about said classification)
		for tuple in results: 
			self.decisions.append(tuple[0])
			self.infos.append(tuple[1])
			
	def optimize_C(self, C_values):
		accuracies = []
		for c_index, c_value in enumerate(C_values):
			if len(accuracies) == 0: best_acc, best_c = "-", "-"
			else: best_acc, best_c = max(accuracies), C_values[np.argsort(accuracies)[-1]]
			
			print("Testing C: {}\tIteration: {}/{} Current best Accuracy and C: {} {}".format(c_value, c_index, len(C_values), best_acc, best_c))
			self.classifier.change_C(c_value)
			corrects = 0
			results = self.classifier.classify(self.threads, self.data_handler.cross_validation(self.data_type), "Opt")
			for result in results:
				if result[2] == True:
					corrects += 1
			accuracies.append(corrects / len(results))
			
			
		sorted_indexes = np.argsort(accuracies)
		print("\nOptimal C:{}".format(C_values[sorted_indexes[-1]]))
		self.classifier.change_C(C_values[sorted_indexes[-1]])
			


	def attribute_testdata(self):
		results = self.classifier.classify(self.threads, self.data_handler.true_data(self.data_type), "test")
		for result in results:
			self.decisions.append(result[0])
			self.infos.append(result[1])



	def plot_values(self, scale=False):
		self.plotter.scale=scale
		if self.data_type == "single":
			self.aggregate_results()
		authors = self.get_authors()
		mapping = {}
		for author in authors:
			if author not in self.positive_classes:
				mapping[author] = "blue"
			else:
				mapping[author] = "red"
		self.plotter.plot(self.decisions, self.infos, mapping)


	def aggregate_results(self):
		data = OrderedDict()
		for i in range(0, len(self.decisions)):
			book = self.infos[i][1]
			data[book] = data.get(book, [[],[]])
			data[book][0].append(self.decisions[i])
			data[book][1].append(self.infos[i])
		
		new_decisions = []
		new_info = []
		for key, value in data.items():
			#print(value)
			new_decision = sum(value[0]) / len(value[0])
			new_decisions.append(new_decision)
			new_info.append([value[1][0][0], value[1][0][1], "full"])
		
		self.decisions = new_decisions
		self.infos = new_info
			

	def get_authors(self):
		authors = []
		for author in self.data_handler.train_data:
			authors.append(author)
		return authors
	
	def get_best_features(self, num_feats=50):
		X_train, y_train, _, _, _, _, _, _ = next(self.data_handler.true_data(self.data_type))
		features = self.classifier.get_best_features(X_train, y_train, num_feats)
		print("Positive features:\n")
		self.print_feat(features["POS"])
		print("Negative features:\n")
		self.print_feat(features["NEG"])
	
	def print_feat(self, feats):
		for feat in feats:
			print("{}\t{}".format(feat[0], feat[1]))
			
	
	def get_results(self):
		pass
	


	def norm_value(self, old_value, old_min, old_max, new_min, new_max):
		return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

	def print_results(self, normalize=False):
		if self.data_type == "single":
			self.aggregate_results()
		min_val, max_val = min(self.decisions), max(self.decisions)
		res = sorted(zip(self.decisions, self.infos), key=itemgetter(0))
		for val in res:
			if normalize:
				value = self.norm_value(val[0], min_val, max_val, -1, 1)
			else:
				value = val[0]
			print("Author: {}\tBook: {}\tDecision: {}".format(val[1][0], val[1][1], value[0]))
		
