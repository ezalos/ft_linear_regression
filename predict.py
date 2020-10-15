import sys

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
		help : False,
		visual : False,
		equation : None,
	}
	for arg in av:
		if arg == "-v" or arg == "--visual":
			arg_dic['visual'] = True
		elif arg == "-h" or arg == "--help":
			arg_dic['help'] = True
		else:
			arg_dic['equation'] = True
	if help:
		usage()
		sys.exit()
	elif equation == None:
		equation = input("Please input your equation: ")
	return equation, visual


if __name__ == '__main__':
	equation, visual = arg_pase(sys.argv[1:])
	with open('model.bin', 'rb') as f:
		theta = pickle.load(f) 
	ComputorV1(equation, visual)
