

import torch
import torch.nn.functional as F
import torch.nn as nn

class autoencoderMLP4Layer(nn.Module):

	def __init__(self, N_input=784, N_bottleneck=8, N_output=784):
		super(autoencoderMLP4Layer, self).__init__()
		N_middle = 392
		self.fc1 = nn.Linear(N_input, N_middle)
		self.fc2 = nn.Linear(N_middle, N_bottleneck)
		self.fc3 = nn.Linear(N_bottleneck, N_middle)
		self.fc4 = nn.Linear(N_middle, N_output)
		self.type = 'MLP4'
		self.input_shape = (1, N_input)


	def forward(self, X):
		return self.decode(self.encode(X))

	def encode(self, X):
		# Encoder
		X = self.fc1(X)
		X = F.relu(X)
		X = self.fc2(X)
		X = F.relu(X)

		return X

	def decode(self, X):
		# Decoder
		X = self.fc3(X)
		X = F.relu(X)
		X = self.fc4(X)
		X = F.sigmoid(X)

		return X




