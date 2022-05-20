import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import ChannelWiseAttention, SpatialAttention

class ConvMixer(nn.Module):
	def __init__(self, kernel_size=3, padding=1, groups=32):
		super(ConvMixer, self).__init__()
		self.conv = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=32,
											kernel_size=kernel_size, padding=padding, groups=groups),
								  nn.GELU(),
								  nn.BatchNorm3d(32))
		self.mixer = nn.Sequential(nn.Conv3d(in_channels=32, out_channels=32, kernel_size=1),
								   nn.GELU(),
								   nn.BatchNorm3d(32))
	def forward(self, x):
		x_in = x
		x = self.conv(x)
		x = x + x_in
		x = self.mixer(x)
		return x

class ConvBlock(nn.Module):
	def __init__(self, layer_num=3, kernel_size=3, padding=1, groups=1):
		super(ConvBlock, self).__init__()
		self.conv = nn.ModuleList([nn.Conv3d(in_channels=32, out_channels=32,
											 kernel_size=kernel_size, padding=padding, groups=groups)
								   for _ in range(layer_num)])
		self.bn = nn.BatchNorm3d(32)

	def forward(self, x):
		x_in = x
		for layer in self.conv:
			x = F.gelu(layer(x))
		x = x + x_in
		x = self.bn(x)
		return x

class ConvBlockShortCut(nn.Module):
	def __init__(self, in_channels=32, out_channels=32):
		super(ConvBlockShortCut, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2)
		self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
		self.conv3 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
		self.bn1 = nn.BatchNorm3d(out_channels)
		self.bn2 = nn.BatchNorm3d(out_channels)
		self.bn3 = nn.BatchNorm3d(out_channels)

	def forward(self, x):
		x1 = self.conv1(x)
		x1 = self.bn1(x1)

		x2 = self.conv2(x)
		x2 = self.bn2(x2)

		x3 = self.conv3(x)
		x3 = self.bn3(x3)

		if self.in_channels == self.out_channels:
			x = x1 + x2 + x3 + x
		else:
			x = x1 + x2 + x3
		x = F.gelu(x)
		return x
class SAEncoder(nn.Module):
	def __init__(self):
		super(SAEncoder, self).__init__()
		self.conv = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
		self.convBlock = nn.Sequential(ConvBlockShortCut(),
									   ConvBlockShortCut())
		self.convBlock1 = ConvBlockShortCut(in_channels=32, out_channels=16)
		self.convBlock2 = ConvBlockShortCut(in_channels=32, out_channels=16)

		self.sa1 = SpatialAttention(in_channels=16)
		self.sa2 = SpatialAttention(in_channels=16)
		# self.sa3 = SpatialAttention(in_channels=8)
		# self.sa4 = SpatialAttention(in_channels=8)

		self.ca = ChannelWiseAttention(in_channels=32)
		# self.sa = SpatialAttention(in_channels=32)
		self.PatchesNum = 64
		self.PatchesEmbedding = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(8, 8, 8),
										  stride=(8, 8, 8), groups=2)
		self.conv_blocks = nn.ModuleList([ConvBlockShortCut() for _ in range(3)])
		self.down_sample = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=2, stride=2)
		self.mlp_list = nn.ModuleList([nn.Sequential(nn.Linear(in_features=32, out_features=16),
													 nn.GELU(),
													 nn.Linear(in_features=16, out_features=32),
													 nn.GELU(),
													 nn.Linear(in_features=32, out_features=1))
									   for _ in range(self.PatchesNum)])

	def forward(self, img):
		x = F.gelu(self.conv(img))
		x = self.convBlock(x)
		x1 = self.convBlock1(x)
		sa1 = self.sa1(x1)
		x1 = torch.sigmoid(sa1) * x1

		x2 = self.convBlock2(x)
		sa2 = self.sa2(x2)
		x2 = torch.sigmoid(sa2) * x2

		sa_raw = torch.cat([sa1, sa2], dim=1)
		x = torch.cat([x1, x2], dim=1)
		ca, ca_weight = self.ca(x)
		x = x * ca
		# sa_raw = self.sa(x)
		# sa = torch.sigmoid(sa_raw)
		# x = x * sa
		x = F.pad(x, (8, 9, 3, 3, 7, 8), "constant", 0)
		x = F.gelu(self.PatchesEmbedding(x))
		for layer in self.conv_blocks:
			x = layer(x)
		x = F.gelu(self.down_sample(x))
		x = x.flatten(2).transpose(1, 2)
		# print(x.shape)
		for i, layer in enumerate(self.mlp_list):
			if i == 0:
				feature_encode = layer(x[:, i, :])
			else:
				feature_encode = torch.cat([feature_encode, layer(x[:, i, :])], dim=1)

		return feature_encode, ca_weight, sa_raw

class PatchesReconstruction(nn.Module):
	def __init__(self):
		super(PatchesReconstruction, self).__init__()
		self.conv = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
		self.deconv_list = nn.Sequential(
			ConvBlock(kernel_size=5, padding=2, groups=32),
			nn.ConvTranspose3d(in_channels=32, out_channels=32,
							   kernel_size=2, stride=2, padding=0),
			nn.GELU(),
			ConvBlock(kernel_size=5, padding=2, groups=32),
			nn.ConvTranspose3d(in_channels=32, out_channels=32,
							   kernel_size=2, stride=2, padding=0),
			nn.GELU(),
			ConvBlock(kernel_size=5, padding=2, groups=32),
			nn.ConvTranspose3d(in_channels=32, out_channels=32,
							   kernel_size=2, stride=2, padding=0),
			nn.GELU(),
			ConvBlock(kernel_size=5, padding=2),
			ConvBlock(kernel_size=5, padding=2),
			nn.ConvTranspose3d(in_channels=32, out_channels=32,
							   kernel_size=2, stride=2, padding=0),
			nn.GELU(),
			ConvBlock(kernel_size=5, padding=2)
		)
		self.decoder = nn.Conv3d(in_channels=32, out_channels=1, kernel_size=1)

	def forward(self, x):
		x = x.view(-1, 1, 4, 4, 4)
		x = F.gelu(self.conv(x))
		x = self.deconv_list(x)
		x = self.decoder(x)
		return x

class SpatialActivation(nn.Module):
	def __init__(self):
		super(SpatialActivation, self).__init__()
		self.sa_encoder = SAEncoder()
		self.decoder = PatchesReconstruction()

	def forward(self, x):
		encode, ca_weight, sa = self.sa_encoder(x)
		decode = self.decoder(encode)
		decode = decode[:, :, 7:-8, 3:-3, 8:-9]
		return decode, ca_weight, sa

if __name__ == '__main__':
	sae = SAEncoder().cuda()
	img = torch.randn(4, 1, 49, 58, 47).cuda()
	total_par = 0
	for layer in sae.parameters():
		total_par += layer.numel()
	print(total_par)