from matplotlib import pyplot as plt
import time

class Plotter():

	def __init__(self, verbose=False, style=None):
		self.verbose = verbose
		if style == "ggplot":
			plt.style.use("ggplot")


	def plot(self, values, infos, mapping):
		x = values
		y = []
		for info in infos:
			if info[0][0] not in mapping:
				y.append("Green")
			else:
				y.append(mapping[info[0][0]])

		fig, ax = plt.subplots(figsize=(6,1))
		ax.scatter(values, [1]*len(x), c=y, marker="s", s=100)
		fig.autofmt_xdate()

		ax.yaxis.set_visible(False)
		ax.spines["right"].set_visible(False)
		ax.spines["left"].set_visible(False)
		ax.spines["top"].set_visible(False)
		ax.xaxis.set_ticks_position("bottom")

		if self.verbose:
			for i in range(len(x)):
				print("Author: {}\tBook: {}\t Value: {}".format(infos[i][0][0], infos[i][0][1], x[i]))

		time.sleep(1) ## So print finishes
		plt.show()
