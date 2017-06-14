from matplotlib import pyplot as plt
import time
import numpy as np

class Plotter():

	def __init__(self, verbose=False, style=None, scale=False):
		self.verbose = verbose
		self.scale = scale
		if style == "ggplot":
			plt.style.use("ggplot")

	def conv(self, old_value, old_min, old_max, new_min, new_max):
			return ( ( old_value - old_min ) / ( old_max - old_min) ) * ( new_max - new_min) + new_min


	def plot(self, values, infos, mapping):
		x = [val[0] for val in values]
		if self.scale and len(x) > 1:
			max_val, min_val = max(x), min(x)
			x = [self.conv(val, min_val, max_val, -1, 1) for val in x]
		y = []
		for info in infos:
			if info[0] not in mapping or info[0] == "Test":
				y.append("Green")
			else:
				y.append(mapping[info[0]])

		fig, ax = plt.subplots(figsize=(6,1))
		ax.scatter(x, [1]*len(x), c=y, marker="s", s=100)
		fig.autofmt_xdate()

		ax.yaxis.set_visible(False)
		ax.spines["right"].set_visible(False)
		ax.spines["left"].set_visible(False)
		ax.spines["top"].set_visible(False)
		ax.xaxis.set_ticks_position("bottom")

		sorts = np.argsort(x)
		if self.verbose:
			for i in range(len(sorts)):
				j = sorts[i]
				print("Author: {}\tBook: {}\t Value: {}".format(infos[j][0], infos[j][1], x[j]))

		time.sleep(1) ## So print finishes
		plt.show()
