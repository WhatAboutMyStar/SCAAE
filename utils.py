import torch
import numpy as np
from nilearn.input_data import NiftiMasker
from nilearn.plotting import plot_prob_atlas
from nilearn.image import iter_img
from nilearn.plotting import plot_stat_map, show
import nibabel as nib
from sklearn.neural_network import MLPClassifier
from model import SpatialActivation

def IoU(n1, n2):
	"""
	:param n1: 1*N
	:param n2: 1*N
	:return: IoU
	"""
	intersect = np.logical_and(n1, n2)
	union = np.logical_or(n1, n2)
	I = np.count_nonzero(intersect)
	U = np.count_nonzero(union)
	return I / U

def flip(row):
	if np.sum(row > 0) < np.sum(row < 0):
		row *= -1
	return row

def thresholding(array):
	array1 = array

	for idx, row in enumerate(array):
		row = flip(row)
		row[row < 0] = 0
		T = np.amax(row) * 0.3
		row[np.abs(row) < T] = 0

		row = row / np.std(row)
		array1[idx, :] = row
	return array1

def transform2d(img, mask_img):
	"""
	:param img:(D, H, W, T)
	:param mask_img:(D, H, W)->bool
	:return: 2d signal (T, n_features)
	"""
	img = np.array(img)
	mask = np.array(mask_img, dtype=np.bool)
	return img[mask].T

def inverse_transform(n_features, mask_img):
	"""
	:param n_features:(T, n_features)
	:param mask_img: (D, H, W)->bool
	:return: (D, H, W, T)
	"""
	mask_img = np.array(mask_img, dtype=np.bool)
	data = np.zeros(mask_img.shape + (n_features.shape[0], ), dtype=n_features.dtype)
	data[mask_img, :] = n_features.T
	return data


class SAFuncActivate:
	def __init__(self, img_path="./data/adhd/data/0010042/0010042_rest_tshift_RPI_voreg_mni.nii.gz",
				 mask_path="./data/ADHD200_mask_152_4mm.nii.gz",
				 device="cuda", model_path="./model/se_3.pth"):
		self.img_path = img_path
		self.mask_path = mask_path
		self.batch_size = 1
		self.masker = NiftiMasker(mask_img=mask_path,
								  standardize=True,
								  detrend=1,
								  smoothing_fwhm=6.)
		self.masker.fit()

		self.device = device
		self.model = SpatialActivation()
		self.model.load_state_dict(torch.load(model_path))
		self.model.to(device)
		self.model.eval()

		fmri_masked = self.masker.transform(self.img_path)
		x_train_3D = self.masker.inverse_transform(fmri_masked)
		self.img = torch.tensor(x_train_3D.get_fdata(), dtype=torch.float)
		self.affine = x_train_3D.affine
		del x_train_3D
		self.img = self.img.permute(3, 0, 1, 2)
		self.img = self.img.unsqueeze(1)
		self.len = self.img.shape[0]
		self.start_index = 0
		self.end_index = self.start_index + self.batch_size

		self.imgs = self.getImgInput()

	def setImg(self, img_path):
		self.img_path = img_path
		fmri_masked = self.masker.transform(self.img_path)
		x_train_3D = self.masker.inverse_transform(fmri_masked)
		self.img = torch.tensor(x_train_3D.get_fdata(), dtype=torch.float)
		self.affine = x_train_3D.affine
		del x_train_3D
		self.img = self.img.permute(3, 0, 1, 2)
		self.img = self.img.unsqueeze(1)
		self.len = self.img.shape[0]
		self.start_index = 0
		self.end_index = self.start_index + self.batch_size

	def __len__(self):
		return self.len

	def setDevice(self, device='cpu'):
		self.device = device
		self.model.to(device)

	def getImgInput(self):
		while True:
			yield self.img[self.start_index:self.end_index, ...]
			self.start_index = self.end_index
			self.end_index += self.batch_size
			if self.start_index == self.len:
				self.start_index = 0
				self.end_index = self.start_index + self.batch_size
			elif self.end_index > self.len:
				self.end_index = self.len

	def getAffine(self):
		return self.affine

	def resetIndex(self):
		self.start_index = 0
		self.end_index = self.start_index + self.batch_size
		del self.imgs
		self.imgs = self.getImgInput()

	def plot_one_net(self, cut_coords=10, colorbar=True):
		img = next(self.imgs)
		img = img.to(self.device)
		_, ca, sa = self.model(img)

		sa = sa.squeeze(0)
		sa = (sa - sa.flatten(1).min(dim=1)[0].view(32, 1, 1, 1).expand_as(sa)) / \
			 (sa.flatten(1).max(dim=1)[0].view(32, 1, 1, 1).expand_as(sa) - sa.flatten(1).min(dim=1)[0].view(32, 1, 1, 1).expand_as(sa))
		sa = sa ** 2

		ca = ca.flatten().detach().cpu().numpy()
		sa = sa.permute(1, 2, 3, 0).detach().cpu().numpy()

		img2d = transform2d(sa, self.masker.mask_img_.get_fdata())
		img3d = inverse_transform(img2d, self.masker.mask_img_.get_fdata())
		components_img = nib.Nifti1Image(img3d, affine=self.getAffine())

		plot_prob_atlas(components_img, title='All components', colorbar=True)
		for i, cur_img in enumerate(iter_img(components_img)):
			plot_stat_map(cur_img, display_mode="z", title="index={} weight={:.4f}".format(i, ca[i]),
						  cut_coords=cut_coords, colorbar=colorbar)
			show()

	def plot_one_atlas(self):
		img = next(self.imgs)
		img = img.to(self.device)
		_, ca, sa = self.model(img)
		sa = sa.squeeze(0)
		sa = sa.permute(1, 2, 3, 0).detach().cpu().numpy()
		img2d = transform2d(sa, self.masker.mask_img_.get_fdata())
		img3d = inverse_transform(img2d, self.masker.mask_img_.get_fdata())
		components_img = nib.Nifti1Image(img3d, affine=self.getAffine())
		plot_prob_atlas(components_img, title='All components', colorbar=True)
		show()

	def save_net(self, save_name="./test.nii.gz"):
		img = next(self.imgs)
		img = img.to(self.device)
		_, ca, sa = self.model(img)
		sa = sa.squeeze(0)
		sa = (sa - sa.flatten(1).min(dim=1)[0].view(32, 1, 1, 1).expand_as(sa)) / \
			 (sa.flatten(1).max(dim=1)[0].view(32, 1, 1, 1).expand_as(sa) - sa.flatten(1).min(dim=1)[0].view(32, 1, 1, 1).expand_as(sa))
		sa = sa ** 2
		sa = sa.permute(1, 2, 3, 0).detach().cpu().numpy()
		img2d = transform2d(sa, self.masker.mask_img_.get_fdata())
		img3d = inverse_transform(img2d, self.masker.mask_img_.get_fdata())
		components_img = nib.Nifti1Image(img3d, affine=self.getAffine())
		components_img.to_filename(save_name)
	
	def plot_weight_net(self, cut_coords=10, colorbar=True,
						black_bg=False, annotate=False, group_step=16, thresholding=False):
		img = next(self.imgs)
		img = img.to(self.device)
		_, ca, sa = self.model(img)

		ca = ca.flatten().detach().cpu().numpy()
		sa = sa.squeeze(0)
		sa = sa.permute(1, 2, 3, 0).detach().cpu().numpy()
		img2d = transform2d(sa, self.masker.mask_img_.get_fdata())
		for i in range(0, 32, group_step):
			tmp2d = img2d[i:i+group_step]
			tmpca = ca[i:i+group_step].view(-1, 1)
			tmp2d = torch.tensor(tmp2d, dtype=torch.float)
			tmp2d = torch.sum(tmpca.expand_as(tmp2d) * tmp2d, dim=0).view(1, -1).detach().numpy()
			tmp2d = (tmp2d - tmp2d.min()) / (tmp2d.max() - tmp2d.min())
			tmp2d = tmp2d ** 2
			if thresholding:
				tmp2d[tmp2d < thresholding] = 0
			img3d = inverse_transform(tmp2d, self.masker.mask_img_.get_fdata())
			components_img = nib.Nifti1Image(img3d, affine=self.getAffine())
			plot_stat_map(components_img, display_mode="z",
						  title=None,
						  cut_coords=cut_coords, colorbar=colorbar,
						  black_bg=black_bg, annotate=annotate)
			show()