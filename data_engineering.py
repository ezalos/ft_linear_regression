import numpy as np

class Preprocessing():
	def __init__(self, data, scaler="", polynomial_fit=None):
		pass

	def minmax(x):
		x = x.reshape(-1, 1)
		# print(x)
		min, max = np.percentile(x, [5, 95])
		mnmx = (x - min) / (max - min)
		# print(mnmx)
		return mnmx

	def minmax_all(x):
		# print("IN: ", x.shape)
		or_shape = x.shape
		if len(x.shape) == 1 or x.shape[1] == 1:
			mnmx =  minmax(x)
		else:
			new_array = []
			for i in x.T:
				# print("MDL: ", i.shape)
				new_array.append(minmax(i).reshape(-1))
				# print("na: ", new_array[-1].shape)
			mnmx = np.array(new_array).T
		# print("OUT: ", mnmx.shape)
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

	print(minmax_all(x))
	print(minmax_all(y))
