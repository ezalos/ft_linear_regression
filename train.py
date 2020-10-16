import sys
import pandas as pd
import numpy as np
import pickle
from my_linear_regression import MyLinearRegression
from preprocessing import Preprocessing
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
	usage_message += " [-s or --scaler AND minmax or std ] "
	usage_message += " [-p or --polynomial] "
	usage_message += " [-a or --alpha] "
	usage_message += " [-n or --n_cycle] "
	usage_message += " [-f or --file] "
	usage_message += "\"path/data.csv\""
	print(usage_message)

def arg_parse(arg, ARGS):
	i = 0
	ac = len(arg)
	err = False
	while i < ac and err == False:
		if arg[i] == "-v" or arg[i] == "--visual":
			ARGS.visual = True
		elif arg[i] == "-h" or arg[i] == "--help":
			ARGS.help = True
		elif arg[i] == "-a" or arg[i] == "--alpha":
			i += 1
			if i < ac:
				try:
					ARGS.alpha = float(arg[i])
				except:
					err = True
					print("Error, Alpha must be float: ", arg[i])
			else:
				print("Error: Alpha value")
		elif arg[i] == "-n" or arg[i] == "--n_cycle":
			i += 1
			if i < ac:
				try:
					ARGS.n_cycle = int(arg[i])
				except:
					err = True
					print("Error, n_cycle must be int: ", arg[i])
			else:
				print("Error: Alpha value")
		elif arg[i] == "-l" or arg[i] == "--load":
			ARGS.load = True
		elif arg[i] == "-s" or arg[i] == "--scaler":
			i += 1
			if i < ac:
				if arg[i] == "minmax" or arg[i] == "std":
					ARGS.scaler = arg[i]
				else:
					print("Error, Bad scaler type: ", arg[i])
					err = True
			else:
				print("Error: Missing scaler type")
				err = True
			ARGS.load = True
		elif ARGS.data == None:
			ARGS.data = arg[i]
		else:
			print("Error while parsing arg: ", arg[i])
			err = True
		i += 1

	if ARGS.help or err:
		usage()
		sys.exit()

	if ARGS.data == None:
		ARGS.data = input("Please input your csv location: ")

	return ARGS

if __name__ == '__main__':
	ARGS = DotDict({
	'help' : False,
	'visual' : False,
	'file' : False,
	'load' : False,
	'data' : None,
	'scaler' : "",
	'polynomial' : None,
	'alpha' : 1e-3,
	'n_cycle' : 1000000,
	'pickle_dir' : "pickles/",
	})

	ARGS = arg_parse(sys.argv[1:], ARGS)

	df = pd.read_csv(ARGS.data)
	X = np.array(df.iloc[:, 0:-1]).reshape(-1, len(df.columns) - 1)
	Y = np.array(df.iloc[:, -1]).reshape(-1,1)

	PreP_x = Preprocessing(X, scaler="minmax", polynomial=None)
	PreP_y = Preprocessing(Y, scaler="minmax", polynomial=None)
	X = PreP_x.data
	Y = PreP_y.data

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
	lr = MyLinearRegression(theta,
							alpha=ARGS.alpha,
							n_cycle=ARGS.n_cycle,
							visual=ARGS.visual)
	print(lr)
	lr.fit(X, Y)

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
