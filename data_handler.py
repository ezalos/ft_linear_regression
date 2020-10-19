import pickle


class DataHandler():
	def __init__(self, ARGS):
		self.ARGS = ARGS
		pass

	def save(self, PreP_x, PreP_y, Theta):
		data = [PreP_x, PreP_y, Theta]
		try:
			save_file = self.ARGS.pickle_dir + self.ARGS.pickle_name + ".pkl"
			with open(save_file, 'wb+') as f:
				pickle.dump(data, f)
				print("Model " + save_file + " saved!")
		except Exception as e:
			print("Error while saving model: ", e)

	def load(self):
		try:
			with open(self.ARGS.load, 'rb') as f:
				data = pickle.load(f)
				print("Model " + self.ARGS.load + " loaded!")
			return data[0], data[1], data[2]
		except Exception as e:
			print("Error while loading model: ", e)
