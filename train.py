import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
from model import Edge_Detector,Res_Block,UResNet_P
import argparse
from dataloader import TrainDataSet

def train(config):
	device = torch.device("cuda:" + str(config.cuda_id))
	model = UResNet_P().to(device)
	for param in model.edge_detector.parameters():
	    param.requires_grad=False
	edge_detector = Edge_Detector().to(device)
	for param in edge_detector.parameters():
	    param.requires_grad=False

	if config.loss_type =='MSE':
		criterion = nn.MSELoss().to(device)
	if config.loss_type == 'L1':
		criterion == nn.L1Loss().to(device)

	optimizer = optim.Adam(model.parameters(), lr = config.lr)
	# if config.train_mode == 'N' or 'P-S':	
	# 	scheduler = lr_scheduler.StepLR(optimizer,step_size=config.step_size,gamma=config.decay_rate)
	# if config.train_mode == 'P-A':
	# 	scheduler = lr_scheduler.StepLR(optimizer,step_size=2*config.step_size,gamma=config.decay_rate)

	transform_list = [transforms.Resize((config.resize,config.resize)),transforms.ToTensor()] 
	tsfms = transforms.Compose(transform_list)
	train_dataset = TrainDataSet(config.input_images_path,config.label_images_path,tsfms)
	train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size = config.batch_size,shuffle = False)	
	img_loss_lst = []
	edge_loss_lst = []
	total_loss_lst = []
	for epoch in range(config.num_epochs):
		img_loss_tmp = []
		edge_loss_tmp = []
		total_loss_tmp = []
		if epoch>1 and epoch % config.step_size == 0:
			for param_group in optimizer.param_groups:
				param_group['lr']=param_group['lr']*0.7

		for input_img,label_img in train_dataloader:
			input_img = input_img.to(device)
			label_img = label_img.to(device)
			
			if config.train_mode == 'N':
				model.zero_grad()
				generate_img,edge_map = model(input_img)
				loss = criterion(generate_img,label_img)
				img_loss_tmp.append(loss.item())
				loss.backward()
				optimizer.step()

			if config.train_mode == 'P-A':

				for flag in range(2):
					model.zero_grad()
					generate_img,edge_map = model(input_img)
					if flag == 0:
						edge_label = edge_detector(label_img)
						edge_loss = criterion(edge_map,edge_label)
						edge_loss.backward()
					if flag == 1:
						img_loss = criterion(generate_img,label_img)
						img_loss.backward()

					scheduler.step()
			if config.train_mode == 'P-S':
				model.zero_grad()
				generate_img,edge_map = model(input_img)
				img_loss = criterion(generate_img,label_img)
				edge_label = edge_detector(label_img)
				edge_loss = criterion(edge_map,edge_label)
				loss = img_loss + weight * edge_label
				total_loss_tmp.append(loss.item())
				loss.backward()

		if config.train_mode == 'N':
			img_loss_lst.append(np.mean(img_loss_tmp))
		
		if config.train_mode == 'P-S':
			total_loss_lst.append(np.mean(total_loss_tmp))

		if config.train_mode == 'P-A':
			img_loss_lst.append(np.mean(img_loss_tmp))
			edge_loss_lst.append(np.mean(edge_loss_tmp))

		if epoch % config.print_feq == 0:
			if config.train_mode == 'N' :			
				print('epoch:[{}]/[{}], image loss:{}'.format(epoch,config.num_epochs,str(img_loss_lst[epoch])))
			if config.train_mode == 'P-A':
				print('epoch:[{}]/[{}], image loss:{},edge difference loss:{}'.format(epoch,config.num_epochs,str(img_loss_lst[epoch]),str(edge_loss_lst[epoch])))
			if config.train_mode == 'P-S':
				print('epoch:[{}]/[{}], total loss:{}'.format(epoch,config.num_epochs,str(total_loss_lst[epoch])))

		if epoch % config.snapshot_iter == 0:
			torch.save(model, config.snapshots_folder + 'model_epoch_{}.ckpt'.format(epoch))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--input_images_path', type=str, default="./data/input/",help='path of input images(underwater images) default:./data/input/')
    parser.add_argument('--label_images_path', type=str, default="./data/label/",help='path of label images(clear images) default:./data/label/')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--decay_rate', type=float, default=0.7,help='Learning rate decay default: 0.7')
    parser.add_argument('--step_size',type=int,default=400,help="Period of learning rate decay")
    parser.add_argument('--loss_type',type=str,default="MSE",help="loss type to train model, L1 or MSE default: MSE")
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--train_mode',type=str,default="N",help="N for UResNet;P-S for URseNet-P-S;P-A for UResnet-P-S. default:N")
    parser.add_argument('--batch_size', type=int, default=1,help="default : 1")
    parser.add_argument('--resize', type=int, default=256,help="resize images, default:resize images to 256*256")
    parser.add_argument('--cuda_id', type=int, default=0,help="id of cuda device,default:0")
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--print_feq', type=int, default=1)    
    parser.add_argument('--snapshot_iter', type=int, default=50)
    parser.add_argument('--snapshots_folder', type=str, default="./snapshots/")
    # parser.add_argument('--sample_output_folder', type=str, default="samples/")

    config = parser.parse_args()

    # if not os.path.exists(config.snapshots_folder):
    #     os.mkdir(config.snapshots_folder)
    # if not os.path.exists(config.sample_output_folder):
    #     os.mkdir(config.sample_output_folder)

    train(config)







