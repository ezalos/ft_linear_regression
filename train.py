import sys
import pandas as pd
import numpy as np
import pickle
from my_linear_regression import MyLinearRegression
import matplotlib.pyplot as plt

class DotDict(dict):
	"""
	a dictionary that supports dot notation
	as well as dictionary access notation
	usage: d = DotDict() or d = DotDict({'val1':'first'})
	set attributes: d.val2 = 'second' or d['val2'] = 'second'
	get attributes: d.val2 or d['val2']
	"""
	__getattr__ = dict.__getitem__
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

def usage():
	usage_message = ""
	usage_message += "usage: " + "python3 " + sys.argv[0] + " "
	usage_message += " [-v or --visual] "
	usage_message += " [-h or --help] "
	usage_message += " [-f or --file] "
	usage_message += "\"path/data.csv\""
	print(usage_message)

def arg_parse(av, ARGS):
	for arg in av:
		if arg == "-v" or arg == "--visual":
			ARGS.visual = True
		elif arg == "-h" or arg == "--help":
			ARGS.help = True
		elif arg == "-l" or arg == "--load":
			ARGS.load = True
		elif ARGS.data == None:
			ARGS.data = arg
		else:
			usage()
			sys.exit()
	if ARGS.help:
		usage()
		sys.exit()
	if ARGS.data == None:
		ARGS.data = input("Please input your csv location: ")
	return ARGS


def stdscaler(x):
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

def minmaxscaler(x):
	x = x.reshape(-1, 1)
	min, max = np.percentile(x, [5, 95])
	mnmx = (x - min) / (max - min)
	return mnmx

def minmax_all(x):
	if len(x.shape) == 1 or x.shape[1] == 1:
		mnmx = minmax(x)
	else:
		new_array = []
		for i in x.T:
			new_array.append(minmax(i).reshape(-1))
		mnmx = np.array(new_array).T
	return mnmx

if __name__ == '__main__':
	ARGS = DotDict({
	'help' : False,
	'visual' : False,
	'file' : False,
	'load' : False,
	'data' : None,
	'alpha' : 1e-3,
	'n_cycle' : 1000000,
	'pickle_dir' : "pickles/",
	})

	ARGS = arg_parse(sys.argv[1:], ARGS)

	df = pd.read_csv(ARGS.data)
	X = np.array(df.iloc[:, 0:-1]).reshape(-1, len(df.columns) - 1)
	Y = np.array(df.iloc[:, -1]).reshape(-1,1)
	# plt.scatter(X.T[0], Y)
	# plt.show()
	X = minmax_all(X)
	Y = minmax_all(Y)
	# plt.scatter(X.T[0], Y)
	# plt.show()
	# print(X.shape)
	# print(Y.shape)
	# sys.exit()
	if ARGS.load:
		try:
			with open(ARGS.pickle_dir + "model" + ".pkl", 'rb') as f:
				theta = pickle.load(f)
				print("Model loaded!")
				print(theta)
		except:
			theta = [1] * (X.shape[1] + 1)
	else:
		theta = [1] * (X.shape[1] + 1)
	lr = MyLinearRegression(theta, alpha=1e-4, n_cycle=1000000, visual=ARGS.visual)
	# lr.fit(X, Y)

	try:
		with open(ARGS.pickle_dir + "model" + ".pkl", 'wb+') as f:
			data = lr.theta
			theta = pickle.dump(data, f)
			print("Model saved!")
			print(data)
	except Exceptions as e:
		print("Error while saving model")
		print("Theta: ", lr.theta)

	if ARGS.visual:
		plt.ioff()
		lr.plot_results(X, Y)
