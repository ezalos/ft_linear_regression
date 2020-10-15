import sys
import pandas as pd
import numpy as np
import pickle
from my_linear_regression import MyLinearRegression

def usage():
	usage_message = ""
	usage_message += "usage: " + "python3 " + sys.argv[0] + " "
	usage_message += " [-v or --visual] "
	usage_message += " [-h or --help] "
	usage_message += " [-f or --file] "
	usage_message += "\"path/data.csv\""
	print(usage_message)

def arg_pase(av):
	arg_dic = {
		'help' : False,
		'visual' : False,
		'file' : False,
		'data' : None,
	}
	for arg in av:
		if arg == "-v" or arg == "--visual":
			arg_dic['visual'] = True
		elif arg == "-h" or arg == "--help":
			arg_dic['help'] = True
		elif arg_dic['data'] == None:
			arg_dic['data'] = arg
		else:
			usage()
			sys.exit()
	if arg_dic['help']:
		usage()
		sys.exit()
	if arg_dic['data'] == None:
		arg_dic['data'] = input("Please input your csv location: ")
	return arg_dic


if __name__ == '__main__':
	arg_dic = arg_pase(sys.argv[1:])

	df = pd.read_csv(arg_dic['data'])
	X = np.array(df.iloc[:, 0:-1]).reshape(-1, len(df.columns) - 1)
	Y = np.array(df.iloc[:, -1]).reshape(-1,1)
	print(X.shape)
	print(Y.shape)
	# sys.exit()
	try:
		with open('pickles/model.pkl', 'rb') as f:
			theta = pickle.load(f)
			print("Model loaded!")
			print(theta)
	except:
		theta = [1] * (X.shape[1] + 1)
	theta = [1] * (X.shape[1] + 1)
	lr = MyLinearRegression(theta, alpha=1e-10, n_cycle=1000000)
	lr.fit(X, Y)

	try:
		with open('pickles/model.pkl', 'wb+') as f:
			data = lr.theta
			theta = pickle.dump(data, f)
			print("Model saved!")
			print(data)
	except Exceptions as e:
		print("Error while saving model")
		print("Theta: ", lr.theta)
