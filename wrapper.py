import numpy as np
np.random.seed(1)
from data_handler import DataHandler, MultiClassDataHandler
from plotting import Plotter
from joblib import Parallel, delayed
import os, pickle
from operator import itemgetter
from collections import OrderedDict
import math

class Handler():

	def __init__(self, train_data, test_data, positive_classes, split_size, split_type, data_type, classifier, threads, iteration=-1, clean_stuff=None):
		self.data_handler = DataHandler(train_data, test_data, positive_classes, split_size, split_type, iteration, clean_stuff)
		self.classifier = classifier
		self.threads = threads
		self.iteration = iteration
		self.data_type = data_type
		self.positive_classes = positive_classes
		self.plotter = Plotter(style="ggplot")
		self.decisions = []
		self.infos = []


	def cross_validate(self):
	#	if self.iteration == -1:
		self.results = self.classifier.classify(self.threads, self.data_handler.cross_validation(self.data_type), "CV")
		#tuple is (classification value, information about said classification)
		for res_tuple in self.results:
			if len(res_tuple) == 0:
				continue
			self.decisions.append(res_tuple[0])
			self.infos.append(res_tuple[1][0])
#	else:
			#res = self.classifier.classify(self.data_handler.cross_validation())

	def optimize_C(self, C_values, threshold=1):
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
			if corrects / len(results) >= threshold:
				break


		sorted_indexes = np.argsort(accuracies)
		print("\nOptimal C:{}".format(C_values[sorted_indexes[-1]]))
		self.classifier.change_C(C_values[sorted_indexes[-1]])

	def optimize_cnn(self, parameters):
		print("Optimizing CNN parameters..")
		self.classifier.optimize_parameters(parameters, self.data_handler.true_data(self.data_type))

	def attribute_testdata(self, keep_test_seperate=False):
		print()
		self.keep_test_seperate = keep_test_seperate
		results = self.classifier.classify(self.threads, self.data_handler.true_data(self.data_type), "test")
		for result in results:
			self.decisions.append(result[0])
			self.infos.append(result[1][0])



	def plot_values(self, scale=False, title=" "):
		self.plotter.scale=scale
	#	if self.data_type == "single" or self.data_type == "book_split":
		self.aggregate_results()
		#else:
			#self.format_results()

		authors = self.get_authors()
		mapping = {}
		for author in authors:
			if author not in self.positive_classes:
				mapping[author] = "blue"
			else:
				mapping[author] = "red"

		self.plotter.plot(self.decisions, self.infos, mapping, title)



	def aggregate_results(self):

		data = OrderedDict()
		for i in range(0, len(self.decisions)):
			author = self.infos[i][0]
			book = self.infos[i][1]
			name = author + "_" + book
			data[name] = data.get(name, [[],[]])
			data[name][0].append(self.decisions[i])
			data[name][1].append((author, book))

		new_decisions = []
		new_info = []
		for key, value in data.items():
			if "test" in key and self.keep_test_seperate == True:
				for val_i, val in enumerate(value[0]):
					new_decision = val
					new_decisions.append(new_decision)
					new_info.append([value[1][0][0], value[1][0][1] + "_"+ str(val_i), "full"])

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
		print("\nNegative features:\n")
		self.print_feat(features["NEG"])

	def get_best_features_nsampling(self, text_percentage, sampling_count):
		X_train, y_train, _, _, _, _, _, _ = next(self.data_handler.true_data(self.data_type))
		features = self.classifier.get_best_features_nsampling(X_train, y_train, text_percentage, sampling_count)
		print("Positive features: \n")
		self.print_feat(features["POS"])
		print("\nNegative features:\n")
		self.print_feat(features["NEG"])

	def print_feat(self, feats):
		if len(feats[0]) == 2: ## normal feat
			for feat in feats:
				print("{}\t{}".format(feat[0], round(feat[1], 3)))
		elif len(feats[0]) == 3: ##nsampled
			for feat in feats:
				print("{}\t{}\t{}".format(feat[0], feat[1], round(feat[2], 3)))

	def get_results(self):
		pass



	def norm_value(self, old_value, old_min, old_max, new_min, new_max):
		return ((old_value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

	def print_results(self, normalize=False):
	#	if self.data_type == "single" or self.data_type == "book_split":
		self.aggregate_results()
		min_val, max_val = min(self.decisions), max(self.decisions)
		res = sorted(zip(self.decisions, self.infos), key=itemgetter(0), reverse=True)
		for val in res:
			if normalize:
				value = self.norm_value(val[0], min_val, max_val, -1, 1)
			else:
				value = val[0]
			value = math.floor(value[0] * 1000) / 1000.0
			print("Author: {}\tBook: {}\tDecision: {}".format(val[1][0], val[1][1], value))

	def load_iteration_results(self, result_folder):
		files = os.listdir(result_folder)
		for file_i, filename in enumerate(files):
			with open(result_folder + "/" + filename, "rb") as pklf:
				res = pickle.load(pklf)

			for i in range(len(res[0][0])):
				self.decisions.append(res[0][0][i])
				info = res[0][1][i]
				self.infos.append(info)
				#print(info, filename)
			print("Read: {}".format(file_i), end="\r")
			print()

class MultiClassHandler(Handler):


	def __init__(self, train_data, test_data, split_size, split_type, data_type, classifier, threads, iter=-1, clean_stuff=None):
		self.data_handler = MultiClassDataHandler(train_data, test_data, split_size, split_type, iter, clean_stuff)
		self.classifier = classifier
		self.threads = threads
		self.data_type = data_type
		self.plotter = Plotter(style="ggplot")
		self.decisions = []
		self.infos = []

	def cross_validate(self):
		print("No CV for multiclass")
		return
		results = self.classifier.classify(self.threads, self.data_handler.cross_validation(self.data_type), "multiclass")
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
			results = self.classifier.classify(self.threads, self.data_handler.cross_validation(self.data_type), clsf_level="multiclass")
			for result in results:
				ind = np.argmax(result[0])
				if result[2][ind] == result[1][0]:
					corrects += 1
				#if result[2] == True:
				#	corrects += 1
			accuracies.append(corrects / len(results))
			if corrects / len(results) == 1:
				break


		sorted_indexes = np.argsort(accuracies)
		print("\nOptimal C:{}".format(C_values[sorted_indexes[-1]]))
		self.classifier.change_C(C_values[sorted_indexes[-1]])



	def attribute_testdata(self):
		self.results = self.classifier.classify(self.threads, self.data_handler.true_data(self.data_type), "multiclass")
		for result in self.results:
			self.decisions.append(result[0])
			self.infos.append(result[1])
			self.classes = result[2]


	def plot_values(self, scale=False, title=" "):
		print("No plotting for multiclass")

	def aggregate_results(self):

		data = OrderedDict()
		for i in range(0, len(self.decisions)):
			author = self.infos[i][0]
			book = self.infos[i][1]
			name = author + "_" + book
			data[name] = data.get(name, [[],[]])
			data[name][0].append(self.decisions[i])
			data[name][1].append(self.infos[i])

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


	def print_results(self, normalize=False):
		if self.data_type == "single":
			self.aggregate_results()
		res = zip(self.decisions, self.infos)
		print("Classes: {}".format(self.classes))
		for val in res:
			print("Author: {}\tBook: {}\tDecisions: {}".format(val[1][0], val[1][1], val[0][0]))
