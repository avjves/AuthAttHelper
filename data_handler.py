from collections import OrderedDict
import importlib
import numpy as np
np.random.seed(1)
from sklearn.utils import shuffle
#from clean_warpper import CleanWrapper
class DataHandler:

	def __init__(self, train_data, test_data, positive_classes, split_size, split_type, iteration, clean_stuff=None):
		## Cleaning shouldn't be part of this shit
		if clean_stuff != None:
			self.cleaner = getattr(importlib.import_module(clean_stuff[1]), clean_stuff[2])()
			self.clean_info = clean_stuff[0]
			self.need_clean = True
		else:
			self.need_clean = False
		self.split_size = split_size
		self.split_type = split_type
		self.iteration = iteration
		self.positive_classes = positive_classes
		self.train_data = self.read_data(train_data, train=True)
		self.test_data = self.read_data(test_data, train=False)


	def read_data(self, data, train):
		new_data = OrderedDict()
		#for author, books in data.items():
		for author in sorted(data):
			books = data[author]
			new_data[author] = new_data.get(author, OrderedDict())
			#for book, text_data in books.items():
			for book in sorted(books):
				text_data = books[book]
				new_data[author][book] = self.preprocess_text(text_data, author, book)
		return new_data


	def preprocess_text(self, text, author, book):
		if self.need_clean:
			text = self.cleaner.clean(text, self.clean_info[author][book])
		return list(self.split_text(text))

	def split_text(self, text):
		if self.split_size == -1 or self.split_size == 0:
			return [text]
		else:
			chunks = []
			for chunk in self.split(text):
				if self.split_type == "word":
					chunks.append(" ".join(chunk))
				else:
					chunks.append(chunk)
			return chunks


	def split(self, text):
		if self.split_type == "word":
			text = text.split(" ")
		for i in range(0, len(text), self.split_size):
			if len(text[i:i + self.split_size]) > self.split_size*0.9:
				yield text[i:i + self.split_size]


	def binarify_classes(self, y):
		new_y = []
		for val in y:
			if val in self.positive_classes:
				new_y.append(1)
			else:
				new_y.append(0)

		return new_y

	def data_to_list(self, level, data):
		X = []
		y = []
		bookinfo = []
		for author, books in data.items():
			if author in self.positive_classes:
				positive_status = 1 ## Pos example
			else:
				positive_status = 0  ## Neg example
			for book, texts in books.items():
				if level == "book":
					X.append("".join(texts))
					bookinfo.append([author, book, "full", positive_status])
					y.append(author)
				elif level == "book_split":
					X.append(texts)
					bookinfo.append([author, book, "book_split", positive_status])
					y.append(author)
				elif level == "single":
					for text_index, text in enumerate(texts):
						X.append(text)
						y.append(author)
						bookinfo.append([author, book, "{}_{}".format(text_index*self.split_size, (text_index+1)*self.split_size), positive_status])
				else:
					raise NotImplementedError("Level {} not valid.".format(level))
		X, y, bookinfo = shuffle(X, y, bookinfo, random_state=1)
		y = self.binarify_classes(y)
		return X, y, bookinfo

	def cross_validation(self, level):
		data, y, info = self.data_to_list(level, self.train_data)
		if self.iteration < 0:
			for i in range(0, len(data)):
				X_train, y_train, train_info, X_test, y_test, test_info = self.extract_sets(i, data, y, info)
				yield X_train, y_train, train_info, X_test, y_test, test_info, i+1, len(data)
		else:
			X_train, y_train, train_info, X_test, y_test, test_info = self.extract_sets(self.iteration, data, y, info)
			yield X_train, y_train, train_info, X_test, y_test, test_info, self.iteration+1, -1

	def extract_sets(self, i, data, y, info):
		X_train = list(data)
		y_train = list(y)
		train_info = list(info)
		X_test = [X_train.pop(i)]
		y_test = [y_train.pop(i)]
		test_info = [train_info.pop(i)]

		return X_train, y_train, train_info, X_test, y_test, test_info

	def true_data(self, level):
		train_data, train_y, train_info = self.data_to_list(level, self.train_data)
		test_data, test_y, test_info = self.data_to_list(level, self.test_data)
		for i in range(0, len(test_data)):
			X_test = [list(test_data).pop(i)]
			y_test = [list(test_y).pop(i)]
			t_info = list(test_info).pop(i)
			yield train_data, train_y, train_info, X_test, y_test, t_info, i+1, len(test_data)
			#yield train_data, train_y, train_info, test_data, test_y, test_info, 0, 0

class MultiClassDataHandler(DataHandler):

	def __init__(self, train_data, test_data, split_size, split_type, iteration, clean_stuff=None):
		self.need_clean = False
		self.split_size = split_size
		self.split_type = split_type
		self.iteration = iteration
		self.train_data = self.read_data(train_data, train=True)
		self.test_data = self.read_data(test_data, train=False)

	def data_to_list(self, level, data):
		X = []
		y = []
		bookinfo = []
		for author, books in data.items():
			for book, texts in books.items():
				if level == "book":
					X.append("".join(texts))
					bookinfo.append([author, book, "full"])
					y.append(author)
				elif level == "single":
					for text_index, text in enumerate(texts):
						X.append(text)
						y.append(author)
						bookinfo.append([author, book, "{}_{}".format(text_index*self.split_size, (text_index+1)*self.split_size)])
				else:
					raise NotImplementedError("Level {} not valid.".format(level))

		return X, y, bookinfo
