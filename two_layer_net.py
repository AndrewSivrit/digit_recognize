import numpy as np

class ClassName(object):
	"""
	Двухслойная full-connected нейронная сеть. Сеть имеет входной размер N
	скрытый слой размером H и выполняет классификацию C классов.
	Я обучал сеть с функцией потери softmax и L2 регуляризации на метриках весов. 
	Сеть имеет ReLU функцией активации после первого full-connected слоя

	input - fully connected layer - ReLU - fully connected layer - softmax

	Кол-во выходов второго fully-connected слоя равно кол-ву классов.
	"""
	def __init__(self, weights, input_size=28*28, hidden_size=100, output_size=10):
		"""
		Initialize the model. Weights are passed into the class. Weights and biases are stored in the
		variable self.params, which is a dictionary with the following keys:
		W1: First layer weights; has shape (D, H)
		b1: First layer biases; has shape (H,)
		W2: Second layer weights; has shape (H, C)
		b2: Second layer biases; has shape (C,)
		Inputs:
		- input_size: The dimension D of the input data.
		- hidden_size: The number of neurons H in the hidden layer.
		- output_size: The number of classes C.
		"""
		self.params = {}
		self.params['W1'] = weights['W1']
		self.params['b1'] = weights['b1']
		self.params['W2'] = weights['W2']
		self.params['b2'] = weights['b2']
		