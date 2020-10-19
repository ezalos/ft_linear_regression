import numpy as np

class Preprocessing():
	def __init__(self, data, scaler=False):
		self.min = []
		self.max = []
		self.scaler = scaler
		if scaler:
			self.data = self.minmax_scaler(data)
		else:
			self.data = data

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


	def unapply_minmax(self, data):
		if self.min == [] or self.max == []:
			return None

		scaler = lambda x, min, max: x * (max - min) + min
		if len(data.shape) == 1 or data.shape[1] == 1:
			if len(self.min) != 1:
				print("Error: preprocessing size do not fit data size")
				return None
			mnmx = scaler(data, self.min[0], self.max[0])
		else:
			if len(self.min) != len(data.T):
				print("Error: preprocessing size do not fit data size")
				return None
			new_array = []
			for idx, val in enumerate(data.T):
				elem = scaler(val, self.min[idx], self.max[idx])
				new_array.append(elem.reshape(-1))
			mnmx = np.array(new_array).T
		return mnmx

	def re_apply_minmax(self, data):
		if self.min == [] or self.max == []:
			return None

		# scaler = lambda x, min, max: x * (max - min) + min
		print("Re-applying minmax preprocessing, as the one used for training")
		scaler = lambda x, min, max: (x - min) / (max - min)
		if len(data.shape) == 1 or data.shape[1] == 1:
			if len(self.min) != 1:
				print("Error: preprocessing size do not fit data size")
				return None
			mnmx = scaler(data, self.min[0], self.max[0])
		else:
			if len(self.min) != len(data.T):
				print("Error: preprocessing size do not fit data size")
				return None
			new_array = []
			for idx, val in enumerate(data.T):
				elem = scaler(val, self.min[idx], self.max[idx])
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
