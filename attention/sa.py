import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
	def __init__(self, in_channels=32):
		super(SpatialAttention, self).__init__()
		self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, padding=3, kernel_size=7)
		self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, padding=1, kernel_size=3)

	def forward(self, x):
		x = F.gelu(self.conv1(x))
		x = self.conv2(x)
		return x

if __name__ == '__main__':
	sa = SpatialAttention()
	img = torch.randn(1, 32, 49, 58, 47)
	sa.cuda()
	img = img.cuda()

	total_par = 0
	for layer in sa.parameters():
		total_par += layer.numel()
	print(total_par)

	out = sa(img)
	print(out.shape)