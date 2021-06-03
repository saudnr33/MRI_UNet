import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print("WOOOOAH4")

from tifffile import imsave
import matplotlib.pyplot as plt

class NeuralNet(torch.nn.Module):
	def __init__(self, lrate, loss_fn):
		super(NeuralNet, self).__init__()

		self.latent = []

		self.conv1 = nn.Conv3d(4, 128, 2, padding = 1)
		self.conv2 = nn.Conv3d(128,128, 2, padding = 1)


		self.conv3 = nn.Conv3d(128, 256, 3,padding = 1)
		self.conv4 = nn.Conv3d(256, 256, 3,padding = 1)

		self.conv5 = nn.Conv3d(256, 512, 3, stride=3,padding = 1)
		self.conv6 = nn.Conv3d(512, 512, 3, stride=3,padding = 1)


		self.convTrans1 = nn.ConvTranspose3d(512, 512, 3, stride = 3,padding = 1)
		self.convTrans2 = nn.ConvTranspose3d(512, 256, 3, stride = 3,padding = 1)

		self.convTrans3 = nn.ConvTranspose3d(256, 256, 3,stride = 3,padding = 1)
		self.convTrans4 = nn.ConvTranspose3d(256, 128,3, stride = 3,padding = 1)

		self.convTrans5 = nn.ConvTranspose3d(128, 128, 2,padding = 1)
		self.convTrans6 = nn.ConvTranspose3d(128, 4, 2,padding = 1)

		self.loss_fn = loss_fn
		self.lrate = lrate
		self.relu =  nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.upsample = nn.Upsample()
	def get_parameters(self):
		return self.parameters()

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x =  self.relu(x)
    # print(x.size())
		# x = self.conv3(x)
		# x = self.conv4(x)
		# x =  self.relu(x)
		# #
		# # x = self.conv5(x)
		# # x = self.conv6(x)
		# # x =  self.relu(x)


		# # x = self.convTrans1(x)
		# # x = self.convTrans2(x)
    # # x =  self.relu(x)

		# x = self.convTrans3(x)
		# x = self.convTrans4(x)
		# x =  self.relu(x)
    # # print(x.size())

		x = self.convTrans5(x)
		x = self.convTrans6(x)
		x =  self.relu(x)

		return x

	def step(self, x, y, i):
		optimizer = optim.Adam(self.parameters(), lr=1e-4, eps=1e-6, weight_decay=1e-5)
		# zero the parameter gradients
		if i%10 == 0:
			optimizer.zero_grad()
		# forward + backward + optimize
		outputs = self(x)
		# print("Version 1: ",outputs.size(), y.size())
		loss = self.loss_fn(outputs, y)
		loss.backward()
		if i%10 == 8:
			optimizer.step()
		return loss.item()
