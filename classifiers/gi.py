
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
