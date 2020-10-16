import matplotlib.pyplot as plt
import numpy as np
import math
import random

class PlotMyLinearRegression():
	def __init__(self):
		pass

	def get_shape(self, x, bonus=0):
		nb_features = x.shape[1]
		nb_graphs = nb_features + bonus

		cols = math.ceil(nb_graphs ** 0.5)

		for i in range(cols + 1):
			rows = i
			if cols * rows >= nb_graphs:
				break

		return (cols, rows)

	def init_plot(self, x, y, cost=None):
		plt.ion()
		plot_dim = self.get_shape(x, bonus=2)
		self.fig, axs = plt.subplots(plot_dim[0], plot_dim[1],
						figsize=[4 * plot_dim[1], 4 * plot_dim[0]])
		self.axs = []
		self.last_reg = []

		for sublist in axs:
			for item in sublist:
				self.axs.append(item)

		#Plot data points
		for i, feature in enumerate(x.T):
			self.axs[i].plot(feature, y, 'o', markersize=3)
			self.axs[i].set_title("Regression of feature " + str(i))
		self.axs[-2].set_title("Dataset prediction")
		self.axs[-1].set_title("Cost function")


	def clean_plot(self):
		for fig_art in self.last_reg:
			fig_art.remove()
		self.last_reg = []

	def plot_pred(self, axs, x, y, y_):
		# print(y)
		# print(y_)
		artist_fig = axs.scatter(x.T[0], y_, s=1, c='r', label="h(x)")
		self.last_reg.append(artist_fig)
		axs.relim()
		# axs.autoscale_view()
		# axs.autoscale_view()
		artist_fig = axs.scatter(x.T[0], y, s=3, c='b', label="y")
		self.last_reg.append(artist_fig)
		# axs.relim()
		axs.autoscale_view()
		axs.legend(loc='best')

	def plot_cost(self, axs, cost):
		artist_fig, = axs.plot(cost, c='y', label="MSE(y, y_)")
		self.last_reg.append(artist_fig)
		axs.legend(loc='best')

	def plot_reg_features(self, x, y, y_, theta):
		for i, feature in enumerate(x.T):
			# print("θ: ", theta)
			my_label = "θ" + str(i + 1) + " * x + θ0"
			feature = feature.reshape(-1, 1)
			y_pred = (theta[1 + i] * (feature)) + theta[0]
			# print("feat: ", feature)
			# print("y_: ", y_pred)
			# print("y: ", y)
			artist_fig, = self.axs[i].plot(feature, y_pred, c='r', label=my_label)
			self.last_reg.append(artist_fig)
			self.axs[i].relim()
			self.axs[i].autoscale_view()
			self.axs[i].legend(loc='best')


	def multi_plot(self, x, y, y_, theta, cost):
		self.clean_plot()
		# mylist=['r', 'b', 'k', 'y']
		# random.shuffle(mylist)

		self.plot_reg_features(x, y, y_, theta)
		self.plot_pred(self.axs[-2], x, y, y_)
		self.plot_cost(self.axs[-1], cost)
		plt.pause(0.000000000001)
		plt.draw()

	def close_plot(self):
		plt.close()
		plt.ioff()

	def plot_results(self, x, y, y_, theta):
		plot_dim = self.get_shape(x, bonus=0)
		self.fig, axs = plt.subplots(plot_dim[0], plot_dim[1],
						figsize=[4 * plot_dim[1], 4 * plot_dim[0]])
		self.axs = []
		for sublist in axs:
			for item in sublist:
				self.axs.append(item)

		#Plot data points
		print(x.T.shape)
		for i, feature in enumerate(x.T):
			# print(i.shape)
			artist_fig = self.axs[i].scatter(feature, y_, s=1, c='r', label="h(x)")
			artist_fig = self.axs[i].scatter(feature, y,  s=3, c='b', label="y")
			self.axs[i].set_title("Prediction of feature " + str(i))
		plt.show()
