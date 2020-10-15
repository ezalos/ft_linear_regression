import matplotlib.pyplot as plt
import math

class PlotMyLinearRegression():
	def __init__(self):
		pass

	def get_shape(self, x):
		nb_features = x.shape[1]
		nb_graphs = nb_features + 2

		cols = math.ceil(nb_graphs ** 0.5)

		for i in range(cols + 1):
			rows = i
			if cols * rows >= nb_graphs:
				break

		return (cols, rows)

	def init_plot(self, x, y, cost=None):
		plt.ion()
		self.graph = True
		plot_dim = self.get_shape(x)
		self.fig, axs = plt.subplots(plot_dim[0], plot_dim[1], figsize=[4 * plot_dim[0], 4 * plot_dim[1]])
		self.axs = []
		self.last_reg = []

		for sublist in axs:
			for item in sublist:
				self.axs.append(item)

		#Plot data points
		for i, feature in enumerate(x.T):
			step = (feature.min() - feature.max()) / 5
			self.axs[i].set_xlim(feature.min() - step, feature.max() + step)
			step = (y.min() - y.max()) / 5
			self.axs[i].set_ylim(y.min() - step, y.max() + step)
			
			self.axs[i].plot(feature, y, 'o', markersize=3)


	def clean_plot(self):
		for fig_art in self.last_reg:
			fig_art.remove()
		self.last_reg = []

	def plot_cost(self, x, y, y_, cost):
		artist_fig = self.axs[-2].scatter(x.T[0], y, s=3, c='b', label="h(x)")
		self.last_reg.append(artist_fig)
		artist_fig = self.axs[-2].scatter(x.T[0], y_, s=1, c='r', label="h(x)")
		self.last_reg.append(artist_fig)
		self.axs[-1].plot(cost, c='y')

	def multi_plot(self, x, y, y_, theta, cost):
		self.clean_plot()

		for i, feature in enumerate(x.T):
			artist_fig, = self.axs[i].plot(feature, theta[1 + i] * feature + theta[0], c='r')
			self.last_reg.append(artist_fig)

		if cost:
			self.plot_cost(x, y, y_, cost)
		plt.pause(0.000000000001)
		plt.draw()

	def close_plot(self):
		plt.close()
		plt.ioff()
