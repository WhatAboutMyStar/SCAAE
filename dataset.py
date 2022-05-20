from torch.utils import data
import torch
import torch.nn.functional as F
import numpy as np
from nilearn.input_data import NiftiMasker

import os

from utils import inverse_transform

class LoadADHD(data.Dataset):
	def __init__(self, mask_path="./data/ADHD200_mask_152_4mm.nii.gz", img_path="./data/adhd40.npy",
				 label_path="./data/adhd40_labels.npy"):
		self.mask_path = mask_path
		self.img_path = img_path
		self.label_path = label_path
		self.masker = NiftiMasker(mask_img=self.mask_path,
								  standardize=True,
								  detrend=1,
								  smoothing_fwhm=6.)
		self.masker.fit()
		fmri_masked = np.load(self.img_path)
		print(fmri_masked.shape)
		self.labels = np.load(self.label_path)
		self.labels = torch.tensor(self.labels, dtype=torch.float).view(-1, 1)

		x_train_3D = inverse_transform(fmri_masked, self.masker.mask_img_.get_fdata())
		self.img = torch.tensor(x_train_3D, dtype=torch.float)
		del x_train_3D

		self.img = self.img.permute(3, 0, 1, 2)
		self.img = self.img.unsqueeze(1)

		self.len = fmri_masked.shape[0]

	def __getitem__(self, item):
		return self.img[item], self.labels[item]

	def __len__(self):
		return self.len

class LoadADHD200(data.Dataset):
	def __init__(self, img_path="./data/adhd/data/", mask_path="./data/ADHD200_mask_152_4mm.nii.gz",
				 save_fmri=False, save_path="./data/adhd200.npy"):
		self.mask_img = mask_path
		self.img_path = img_path
		self.masker = NiftiMasker(mask_img=self.mask_img,
							 standardize=True,
							 detrend=1,
							 smoothing_fwhm=6.)
		self.masker.fit()
		self.filename = os.listdir(img_path)
		if save_fmri:
			for i, index in enumerate(self.filename):
				print(i)
				fmri_masked = self.masker.transform(
					self.img_path + index + '/' + index + "_rest_tshift_RPI_voreg_mni.nii.gz")
				if i == 0:
					iterator = fmri_masked
				else:
					iterator = np.concatenate((iterator, fmri_masked), axis=0)

			np.save(save_path, iterator)
		else:
			fmri_masked = np.load(save_path)
		
		self.len = fmri_masked.shape[0]

		x_train_3D = self.masker.inverse_transform(fmri_masked)
		self.img = torch.tensor(x_train_3D.get_fdata(), dtype=torch.float)
		del x_train_3D
		self.img = self.img.permute(3, 0, 1, 2)
		self.img = self.img.unsqueeze(1)
		# self.img = F.pad(self.img, (8, 9, 3, 3, 7, 8), "constant", 0)

	def __getitem__(self, item):
		return self.img[item], self.img[item]

	def __len__(self):
		return self.len

class Load_adhd200_fmri(data.Dataset):
	def __init__(self, fmri_path="./data/adhd200.npy", mask_path="./data/ADHD200_mask_152_4mm.nii.gz",
				 load_img=True):
		self.mask_img = mask_path
		self.fmri_path = fmri_path
		self.load_img = load_img
		self.masker = NiftiMasker(mask_img=self.mask_img,
								  standardize=True,
								  detrend=1,
								  smoothing_fwhm=6.)
		self.masker.fit()

		self.fmri_masked = np.load(self.fmri_path)
		self.fmri_masked = torch.tensor(self.fmri_masked, dtype=torch.float)

		if load_img:
			x_train_3D = self.masker.inverse_transform(self.fmri_masked)
			self.img = torch.tensor(x_train_3D.get_fdata(), dtype=torch.float)
			del x_train_3D
			self.img = self.img.permute(3, 0, 1, 2)
			self.img = self.img.unsqueeze(1)

	def __getitem__(self, item):
		if self.load_img:
			return self.img[item], self.fmri_masked[item]
		else:
			return self.fmri_masked[item]

	def __len__(self):
		return self.fmri_masked.shape[0]

	def get_fmri_masked(self):
		return self.fmri_masked

if __name__ == '__main__':
	dataset = LoadADHD200()
	img = dataset[0]
	print(img.shape)