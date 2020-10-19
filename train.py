import sys
import pandas as pd
import numpy as np
import pickle
from my_linear_regression import MyLinearRegression
from preprocessing import Preprocessing
import matplotlib.pyplot as plt
from data_handler import DataHandler
from arg_parse import arg_parse_train, DotDict

if __name__ == '__main__':
	ARGS = DotDict({
	'help' : False,
	'visual' : False,
	'file' : False,
	'load' : "",
	'data' : None,
	'scaler' : "",
	'polynomial' : None,
	'alpha' : 1e-3,
	'n_cycle' : 100000,
	'pickle_dir' : "pickles/",
	'pickle_name' : "model",
	})

	ARGS = arg_parse_train(sys.argv[1:], ARGS)

	df = pd.read_csv(ARGS.data)
	X = np.array(df.iloc[:, 0:-1]).reshape(-1, len(df.columns) - 1)
	Y = np.array(df.iloc[:, -1]).reshape(-1,1)

	pkl = DataHandler(ARGS)

	if ARGS.load:
		PreP_x, PreP_y, theta = pkl.load()
		if PreP_x.scaler:
			X = PreP_x.re_apply_minmax(X)
		if PreP_y.scaler:
			Y = PreP_y.re_apply_minmax(Y)
		if type(X) == type(None) or type(Y) == type(None):
			sys.exit()
	else:
		PreP_x = Preprocessing(X, scaler=ARGS.scaler)
		PreP_y = Preprocessing(Y, scaler=ARGS.scaler)
		X = PreP_x.data
		Y = PreP_y.data
		theta = [1] * (X.shape[1] + 1)

	lr = MyLinearRegression(theta,
							alpha=ARGS.alpha,
							n_cycle=ARGS.n_cycle,
							visual=ARGS.visual)
	err = lr.fit(X, Y)
	if type(err) == type(None):
		sys.exit()

	pkl.save(PreP_x, PreP_y, lr.theta)
	if ARGS.visual:
		lr.plot_results(X, Y)
