import sys
import numpy as np

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

def usage_train():
	usage_message = ""
	usage_message += "usage: " + "python3 " + sys.argv[0] + " "
	usage_message += "\n\t\t" + " [-v or --visual] "
	usage_message += "\n\t\t" + " [-h or --help] "
	usage_message += "\n\t\t" + " [-s or --scaler] "
	usage_message += "\n\t\t" + " [-a or --alpha AND int] "
	usage_message += "\n\t\t" + " [-n or --n_cycle AND int] "
	usage_message += "\n\t\t" + " [-l or --load AND '/path/to_model.pkl'] "
	usage_message += "\n\t\t" + " [-m or --model AND 'saved_model_name'] "
	usage_message += "\n\t\t" + "\"path/data.csv\""
	usage_message += "\n\n" + "Options details: "
	usage_message += "\n\t" + "scaler: " + "will apply minmax to data"
	usage_message += "\n\t" + "load:   " + "will load theta and preprocessing "
	usage_message += "from file and apply them. It ignore sacler option."
	usage_message += "\n\t" + "model:  " + "Default value is pickles/model.pkl"

	print(usage_message)

def arg_parse_train(arg, ARGS):
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
					print("Error, Alpha must be float: ", arg[i])
					err = True
			else:
				print("Error: Missing Alpha value")
				err = True
		elif arg[i] == "-n" or arg[i] == "--n_cycle":
			i += 1
			if i < ac:
				try:
					ARGS.n_cycle = int(float(arg[i]))
				except:
					print("Error, n_cycle must be int: ", arg[i])
					err = True
			else:
				print("Error: Missing n_cycle value")
				err = True
		elif arg[i] == "-m" or arg[i] == "--model":
			i += 1
			if i < ac:
				ARGS.pickle_name = arg[i]
			else:
				print("Error: Missing name of model to save")
				err = True
		elif arg[i] == "-l" or arg[i] == "--load":
			i += 1
			if i < ac:
				ARGS.load = arg[i]
			else:
				print("Error: Missing model to load")
				err = True
		elif arg[i] == "-s" or arg[i] == "--scaler":
			ARGS.scaler = True
		elif ARGS.data == None:
			ARGS.data = arg[i]
		else:
			print("Error while parsing arg: ", arg[i])
			err = True
		i += 1

	if ARGS.help or err:
		usage_train()
		sys.exit()

	if ARGS.data == None:
		ARGS.data = input("Please input your csv location: ")

	return ARGS



def usage_predict():
	usage_message = ""
	usage_message += "usage: " + "python3 " + sys.argv[0] + " "
	usage_message += "\n\t\t" + " [-v or --visual] "
	usage_message += "\n\t\t" + " [-h or --help] "
	usage_message += "\n\t\t" + " [-d or --data AND '/path/to_data.csv'] "
	usage_message += "\n\t\t" + " [-l or --load AND '/path/to_model.pkl'] "
	print(usage_message)

def arg_parse_predict(arg, ARGS):
	i = 0
	ac = len(arg)
	err = False
	while i < ac and err == False:
		if arg[i] == "-v" or arg[i] == "--visual":
			ARGS.visual = True
		elif arg[i] == "-h" or arg[i] == "--help":
			ARGS.help = True
		elif arg[i] == "-l" or arg[i] == "--load":
			i += 1
			if i < ac:
				ARGS.load = arg[i]
			else:
				print("Error: Missing model path")
				err = True
		elif arg[i] == "-d" or arg[i] == "--data":
			i += 1
			if i < ac:
				ARGS.data = arg[i]
			else:
				print("Error: Missing data path")
				err = True
		else:
			print("Error while parsing arg: ", arg[i])
			err = True
		i += 1

	if ARGS.help or err:
		usage_predict()
		sys.exit()

	if ARGS.data == None:
		val = input("Please input your value(s) to predict,\n\
if multiple features are necessary use comma separated list:\n")
		val = val.split(",")
		values = []
		for i in val:
			try:
				elem = float(i)
			except Exception as e:
				print("Error, '" + i + "' is not float compatible")
				sys.exit()
			values.append(elem)
		X = np.array(values).reshape(1, -1)
		ARGS["values"] = X
		print(ARGS["values"])

	return ARGS
