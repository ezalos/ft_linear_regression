import sys
import pandas as pd
import numpy as np
import pickle
from my_linear_regression import MyLinearRegression
from preprocessing import Preprocessing
import matplotlib.pyplot as plt
from data_handler import DataHandler
from arg_parse import arg_parse_predict, DotDict

if __name__ == '__main__':
	ARGS = DotDict({
	'help' : False,
	'visual' : False,
	'load' : "",
	'data' : None,
	'values' : None,
	'pickle_dir' : "pickles/",
	'pickle_name' : "model",
	'alpha' : 1e-3,
	'n_cycle' : 100000,
	})

	ARGS = arg_parse_predict(sys.argv[1:], ARGS)

	if ARGS.data:
		df = pd.read_csv(ARGS.data)
		# print(df.shape)
		if (df.shape[1] > 1):
			X = np.array(df.iloc[:, 0:-1]).reshape(-1, len(df.columns) - 1)
			Y = np.array(df.iloc[:, -1]).reshape(-1,1)
		else:
			X = np.array(df.iloc[:, :])
			print("Dataset without results, if visual asked, an array of zeros will be used")
			Y = np.zeros_like(X)
	else:
		X = ARGS["values"]
	if ARGS.load:
		pkl = DataHandler(ARGS)
		PreP_x, PreP_y, theta = pkl.load()
		X = PreP_x.re_apply_minmax(X)
		Y = PreP_y.re_apply_minmax(Y)
		if type(X) == type(None):
			sys.exit()
	else:
		theta = [0] * (X.shape[1] + 1)

	print("Theta is: ", theta)

	lr = MyLinearRegression(theta, visual=ARGS.visual)
	value = lr.predict(X)
	print("Predicted value(s):\n", value)
	if ARGS.load:
		print("\twithout preprocessing:\n", PreP_y.unapply_minmax(value))

	if ARGS.visual:
		lr.plot_results(X, Y)
