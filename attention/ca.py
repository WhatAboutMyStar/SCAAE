import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelWiseAttention(nn.Module):
	def __init__(self, in_channels=32):
		super(ChannelWiseAttention, self).__init__()
		self.fc1 = nn.Linear(in_features=in_channels, out_features=in_channels//4)
		self.fc2 = nn.Linear(in_features=in_channels//4, out_features=in_channels//4)
		self.fc3 = nn.Linear(in_features=in_channels//4, out_features=in_channels)

	def forward(self, x):
		"""
		:param x: (B, C, D, H, W)
		:return: (B, C, D, H, W)
		"""
		batch, channels, d, h, w = x.size()
		feature = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(batch, channels)
		feature = F.gelu(self.fc1(feature))
		feature = F.gelu(self.fc2(feature))
		feature = self.fc3(feature)
		feature = torch.sigmoid(feature)

		weight = feature.view(batch, channels, 1, 1, 1)
		feature = weight.expand_as(x).clone()
		return feature, weight

if __name__ == '__main__':
	ca = ChannelWiseAttention().cuda()
	img = torch.randn(1, 32, 49, 58, 47).cuda()

	total_par = 0
	for layer in ca.parameters():
		total_par += layer.numel()
	print(total_par)

	feature, weight = ca(img)
	print(feature.shape)
	print(weight.shape)
