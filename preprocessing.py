import numpy as np

class Preprocessing():
	def __init__(self, data, scaler="", polynomial=None):
		self.min = []
		self.max = []
		self.scaler = scaler
		self.polynomial = polynomial
		if scaler == "minmax":
			self.data = self.minmax_scaler(data)

	def add_polynomial_features(x, power):
		new = []
		new.append(x)
		for i in range(power - 1):
			new.append(x ** (i + 2))
		return np.concatenate(tuple(new),axis=1)

	def stdscaler(self, x):
		if len(x.shape) == 1:
			x = x.reshape(-1, 1)
			zs = x - x.mean()
			zs = zs / x.std()
		else:
			new_array = []
			for i in x:
				new_array.append((i - i.mean()) / i.std)
			zs = np.array(new_array)
		return zs

	def minmax_elem(self, x, percent=5):
		x = x.reshape(-1, 1)
		min, max = np.percentile(x, [percent, 100 - percent])
		mnmx = (x - min) / (max - min)
		self.min.append(min)
		self.max.append(max)
		return mnmx

	def minmax_scaler(self, x, percent=5):
		if len(x.shape) == 1 or x.shape[1] == 1:
			mnmx = self.minmax_elem(x, percent)
		else:
			new_array = []
			for i in x.T:
				elem = self.minmax_elem(i, percent)
				new_array.append(elem.reshape(-1))
			mnmx = np.array(new_array).T
		return mnmx

	def re_apply_minmax(self, data):
		if self.min == [] or self.max == []:
			# raise Exceptions()
			return None

		scaler = lambda x, min, max: x * (max - min) + min
		if len(data.shape) == 1 or data.shape[1] == 1:
			mnmx = scaler(data, self.min[0], self.max[0])
		else:
			new_array = []
			for i in data.T:
				elem = scaler(data, self.min[i], self.max[i])
				new_array.append(elem.reshape(-1))
			mnmx = np.array(new_array).T
		return mnmx

if __name__ == '__main__':
	x = np.array([	[1. , 2., 3.],
					[10., 4., 0.],
					[8. , 3., 0.],
					[5.5, 4., 1.]])
	y = np.array([	[1.],
					[10.],
					[7.],
					[5.5]])

	thing = Preprocessing(x, )
