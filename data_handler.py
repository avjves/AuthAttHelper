from collections import OrderedDict
import importlib
#from clean_warpper import CleanWrapper
class DataHandler():
	
	def __init__(self, train_data, test_data, clean_info, cleaner_file, cleaner_class, split_size, split_type):
		self.cleaner = getattr(importlib.import_module(cleaner_file), cleaner_class)()
		self.clean_info = clean_info
		self.split_size = split_size
		self.split_type = split_type
		self.train_data = self.read_data(train_data, train=True)
		self.test_data = self.read_data(test_data, train=False)


	def read_data(self, data, train):
		new_data = OrderedDict()
		for author, books in data.items():
			new_data[author] = new_data.get(author, OrderedDict())
			for book, text_data in books.items():
				new_data[author][book] = self.preprocess_text(text_data, author, book)
		return new_data
	
	
	def preprocess_text(self, text, author, book):
		text = self.cleaner.clean(text, self.clean_info[author][book])
		return list(self.split_text(text))
	
	def split_text(self, text):
		if self.split_size == -1 or self.split_size == 0:
			yield [text]
		else:
			for chunk in self.chunks(text):
				yield chunk
	
	def split(self, text):
		for i in range(0, len(text), self.split_size):
			if i + self.split_size < len(text)*0.90:
				yield text[i:i + self.split_size]
				
	def data_to_list(self, level, data):
		X = []
		y = []
		bookinfo = []
		for author, books in data.items():
			for book, texts in books.items():
				if level == "book":
					bookd = []
					for text_index, text in enumerate(texts):
						bookd.append("".join(text))
						bookinfo.append([author, book, "{}_{}".format(text_index*self.split_size, (text_index+1)*self.split_size)])
					X.append("".join(bookd))
					y.append(author)
				elif level == "single":
					for text in texts:
						X.append(text)
						y.append(author)
				else:
					raise NotImplementedError("Level {} not valid.".format(level))
		return X, y, bookinfo
	
	def cross_validation(self, level):
		data, y, info = self.data_to_list(level, self.train_data)
		for i in range(0, len(data)):
			X_train = list(data)
			y_train = list(y)
			train_info = list(info)
			X_test = [X_train.pop(i)]
			y_test = [y_train.pop(i)]
			test_info = [train_info.pop(i)]
			
			yield X_train, y_train, train_info, X_test, y_test, test_info


	def true_data(self, level):
		train_data, train_y, train_info = self.data_to_list(level, self.train_data)
		test_data, test_y, test_info = self.data_to_list(level, self.test_data)

		return train_data, train_y, train_info, test_data, test_y, test_info
