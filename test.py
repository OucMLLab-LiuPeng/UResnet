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
from dataloader import TrainDataSet,TestDataSet
from model import UResNet_P



def test(config):

	device = torch.device("cuda:" + str(config.cuda_id))
	test_model = torch.load(config.snapshot_pth).to(device)
	tsfm_lst = [transforms.ToTensor()]
	tsfm = transforms.Compose(tsfm_lst)

	testset = TestDataSet(config.test_path,tsfm)
	test_dataloader = torch.utils.data.DataLoader(testset,batch_size = config.batch_size,shuffle = False)

	for i,(img,name) in enumerate(test_dataloader):
		with torch.no_grad():
			img = img.to(device)
			generate_img,_ = test_model(img)
			torchvision.utils.save_image(generate_img,config.output_pth + name[0] + '-output.png')
			print('process image [{}]/[{}]'.format(str(i+1),str(len(testset))))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--cuda_id', type=int, default=0,help='default:0')
	parser.add_argument('--snapshot_pth',type=str,default=None,help='snapshot path,such as :xxx/snapshots/model.ckpt default:None')
	parser.add_argument('--test_path',type=str,default='./data/test/',help='path of test images. default:./data/test/ ')
	parser.add_argument('--batch_size',type=int,default=1)
	parser.add_argument('--output_pth',type=str,default='./results/',help='path to save generated image. default:./results/')


