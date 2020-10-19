import numpy as np
import matplotlib.pyplot as plt
import math
from listy import ft_progress
from visuals import PlotMyLinearRegression

def add_intercept(x):
	vec_one = np.ones(x.shape[0])
	result = np.column_stack((vec_one, x))
	return result

class MyLinearRegression():
	def __init__(self, thetas=[0, 0], alpha=0.00015, n_cycle=100000, visual=False):
		self.alpha = alpha
		self.theta = np.array(thetas).reshape(-1, 1)
		self.n_cycle = n_cycle
		self.visual = visual
		self.cost = []
		self.plot = PlotMyLinearRegression()

	def gradient(self, x, y):
		m = x.shape[0]
		gradient = (x.T @ ((x @ self.theta) - y)) / m
		# print(gradient)
		return gradient

	def plot_results(self, x, y):
		if x.shape[1] +1 != self.theta.shape[0]:
			print(x.shape, self.theta.shape)
			print("Error: Theta dimension dont fit with X")
			return None
		self.plot.plot_results(x, y, self.predict(x), self.theta)

	def fit_routine(self, x, y, i):
		# print(i * 100 / self.n_cycle, "%")
		# if x.shape[1] > 1:
		# if not i % (update * 5):
		if self.visual:
			self.plot.multi_plot(x, y, self.predict(x), self.theta, self.cost)
		# else:
		# 	self.plot(x, y)
		# print(self.theta)

		# print(self.cost[-1])

	def fit(self, x, y):
		if x.shape[1] +1 != self.theta.shape[0]:
			print(x.shape, self.theta.shape)
			print("Error: Theta dimension dont fit with X")
			return None
		update = self.n_cycle // 100
		self.cost = []
		if self.visual:
			self.plot.init_plot(x, y)
		x_ = add_intercept(x)
		for i in ft_progress(self.n_cycle + 1):
			if not i % (update * 5):
				if np.isnan(self.theta).any():
					print("\nError: Theta has NaN values.\
\nTry with a smaller alpha or apply preprocessing to data")
					return None
				self.fit_routine(x, y, i)
			theta_ = self.gradient(x_, y) * self.alpha
			self.theta = self.theta - theta_
			self.cost.append(self.mse_(self.predict(x), y).sum())
		if self.visual:
			self.plot.close_plot()
		return self.theta

	def mse_(self, y, y_hat):
		res = (1 / (y.shape[0])) * (y_hat - y).T.dot(y_hat - y)
		return abs(res)

	def predict(self, x):
		if x.shape[1] +1 != self.theta.shape[0]:
			print(x.shape, self.theta.shape)
			print("Error: Theta dimension dont fit with X")
			return None
		if len(x) == 0:
			return None
		x = add_intercept(x)
		if len(self.theta) != x.shape[1]:
			return None
		return x @ self.theta

	def __str__(self):
		msg = "Regression: " + "\n"
		msg += "\ttheta: " + "\n"
		msg += "\t\t" + str(self.theta).replace("\n", "\n\t\t") + "\n"
		msg += "\talpha:   " + str(self.alpha) + "\n"
		msg += "\tn_cycle: " + str(self.n_cycle) + "\n"
		msg += "\tvisual:  " + str(self.visual) + "\n"
		return msg
