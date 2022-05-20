import torch
import torch.nn as nn
from torch.utils import data
from torch.optim import Adam, SGD
from tensorboardX import SummaryWriter
from tqdm import trange
from datetime import datetime
import argparse

from dataset import LoadADHD200
from model import SpatialActivation

def train(lr=0.001, device='cuda', epochs=10,
		  img_path = "./data/adhd/data/",save_path="./model/",
		  load_model=True, batch_size=4, load_epochs=1, encoder="se",
		  optim='sgd', momentum=0.9, step_size=2, gamma=0.95, parallel=False):
	AutoEncoder = SpatialActivation()
	AutoEncoder.to(device)

	if load_model:
		AutoEncoder.load_state_dict(torch.load("{}{}_{}.pth".format(save_path, encoder, load_epochs)))

	if parallel:
		AutoEncoder = nn.DataParallel(AutoEncoder, device_ids=[0, 1])

	if optim == 'sgd':
		optimizer = SGD(AutoEncoder.parameters(), lr=lr, momentum=momentum)
	elif optim == 'adam':
		optimizer = Adam(AutoEncoder.parameters(), lr=lr)

	mse_loss = nn.MSELoss()
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

	print("loading data......")
	data_loader = data.DataLoader(LoadADHD200(img_path=img_path),
								  batch_size=batch_size,
								  shuffle=True)
	print("data load complete.")

	TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
	writer = SummaryWriter("./logdir/" + TIMESTAMP)

	for epoch in trange(1, epochs + 1):
		total_loss = 0
		for img, target_img in data_loader:
			img = img.to(device)

			decode, _, _ = AutoEncoder(img)
			loss = mse_loss(decode, img)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
		total_loss = total_loss / len(data_loader)
		writer.add_scalar("loss", total_loss, global_step=epoch)
		writer.add_scalar("learning rate", optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
		scheduler.step()

		AutoEncoder_path = save_path + "{}_{}.pth".format(encoder, load_epochs + epoch)
		if parallel:
			torch.save(AutoEncoder.module.state_dict(), AutoEncoder_path)
		else:
			torch.save(AutoEncoder.state_dict(), AutoEncoder_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--lr', default=0.001, type=float)
	parser.add_argument('--device', default='cuda', type=str)
	parser.add_argument('--epochs', default=20, type=int)
	parser.add_argument('--img_path', default="./data/adhd/data/", type=str)
	parser.add_argument('--save_path', default="./model/", type=str)
	parser.add_argument('--load_model', default=False, type=bool)
	parser.add_argument('--load_epochs', default=0, type=int)
	parser.add_argument('--encoder', default='se', type=str)
	parser.add_argument('--batch_size', default=4, type=int)
	parser.add_argument('--optim', default='sgd', type=str)
	parser.add_argument('--momentum', default=0.9, type=float)
	parser.add_argument('--step_size', default=2, type=int)
	parser.add_argument('--gamma', default=0.95, type=float)
	parser.add_argument('--parallel', default=False, type=bool)
	args = parser.parse_args()

	train(lr=args.lr, device=args.device, epochs=args.epochs,
		  img_path=args.img_path, save_path=args.save_path,
		  load_model=args.load_model, batch_size=args.batch_size,
		  load_epochs=args.load_epochs, encoder=args.encoder,
		  optim=args.optim, momentum=args.momentum,
		  step_size=args.step_size, gamma=args.gamma, parallel=args.parallel)
