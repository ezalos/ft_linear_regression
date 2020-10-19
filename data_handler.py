import pickle


class DataHandler():
	def __init__(self, ARGS):
		self.ARGS = ARGS
		pass

	def save(self, PreP_x, PreP_y, Theta):
		data = [PreP_x, PreP_y, Theta]
		try:
			with open(ARGS.pickle_dir + "model" + ".pkl", 'wb+') as f:
				# data = lr.theta
				pickle.dump(data, f)
				print("Model saved!")
				print(data)
		except Exceptions as e:
			print("Error while saving model")
			print("Theta: ", Theta)
		pass

	def load(self):
		if ARGS.load:
			try:
				with open(ARGS.pickle_dir + "model" + ".pkl", 'rb') as f:
					data = pickle.load(f)
					print("Model loaded!")
					print(theta)
			except:
				theta = [1] * (X.shape[1] + 1)
		else:
			theta = [1] * (X.shape[1] + 1)
		return data[0], data[1], data[2]
