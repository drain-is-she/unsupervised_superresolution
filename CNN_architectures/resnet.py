import torch
import torch.nn as nn 
from torch.utils.data import Dataset,Dataloader
from torchvision import transformS
from PIL import Image 


class Dataset:
	def __init__(self,image_paths , labels , transform = None ):
		self.image_paths = image_paths 
		self.labels = labels 
		self.transform = transform


	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self,idx):
		image = Image.open(self.image_paths[idx]).convert("RGB")
		label = self.labels[idx]
		if self.transform : 
			image = self.transform(image)

		return image , labels 


transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(mean =[0,0,0,0],std = [0,0,0,0])])

class BasicBlock(nn.Module):
	expansion = 1 

	def __init__(self, in_channels , out_channels , stride = 1 , downsample = None):
		super(BasicBlock, self).__init__()

		self.conv1 = nn.Conv2d(
			in_channels = in_channels, 
			out_channels=out_channels, 
			kernel_size = 3 , 
			stride = stride , 
			bias = False
			)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace = True)

		self.conv2 = nn.Conv2d(
			in_channels = in_channels, 
			out_channels = out_channels, 
			kernel_size = 3 , 
			stride = stride, 
			bias = False
			)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.downsample = downsample 



	def forward(self,x):
		identity = x 
		out = self.conv1(x) 
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None : 
			identity = self.downsample(x)



		out = out + identity

		out = self.relu(out)

		return out 


class ResNet(nn.Module):
	def __init__(self, block , layers , num_classes = 10 ):
		super(ResNet,self).__init__()

		self.in_channels = 64 

		self.conv1 = Conv2d(in_channels = 3 , out_channels = 64 , kernel_size = 7 , stride = 2 , paddinh = 3 , bias= False)


        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size=3 , stride = 2, padding = 1)


        self.layer1 = self._make_layer(block, 64 , layers[0])
        self.layer2 = self._make_layer(block , 128 , layers[1],stride =2)
        self.layer3 = self._make_layer(block , 256 , layers[2],stride = 2 )
        self.layer4 = self._make_layer(block , 512 , layers[3],stride = 2 )


        #residual 
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion , num_classes)