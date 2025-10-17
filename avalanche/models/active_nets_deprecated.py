import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.autograd import Variable
from copy import deepcopy
from tqdm import tqdm
import torch.nn.init as init

class Net:
	def __init__(self, net, device):
		self.net = net
		self.device = device
		self.clf = self.net().to(device)
	
	def predict(self, data):
		self.clf.eval()
		preds = torch.zeros(len(data), dtype=int)
		loader = DataLoader(list(enumerate(data)), shuffle=False, batch_size=64)
		with torch.no_grad():
			for idxs, (x, y, _) in loader:
				# print(idxs, x, y)
				x, y = x.to(self.device), y.to(self.device)
				out, e1 = self.clf.forward(x, repr=True)
				pred = out.max(1)[1]
				preds[idxs] = pred.cpu()
		return preds
    
	def predict_logit(self, data):
		self.clf.eval()
		# import pdb;pdb.set_trace()
		logits = torch.zeros([len(data), self.clf.classifier.out_features])
		loader = DataLoader(list(enumerate(data)), shuffle=False, batch_size=64)
		with torch.no_grad():
			for idxs, (x,y,_) in loader:
				x, y = x.to(self.device), y.to(self.device)
				out, e1 = self.clf.forward(x, repr=True)
				logits[idxs] = out.cpu()
		return logits
	
	def predict_prob(self, data):
		self.clf.eval()
		probs = torch.zeros([len(data), self.clf.classifier.out_features])
		loader = DataLoader(list(enumerate(data)), shuffle=False, batch_size=64)
		with torch.no_grad():
			for idxs, (x,y,_) in loader:
				x, y = x.to(self.device), y.to(self.device)
				out, e1 = self.clf.forward(x, repr=True)
				prob = F.softmax(out, dim=1)
				probs[idxs] = prob.cpu()
		return probs

	def predict_prob_dropout(self, data, n_drop=10):
		self.clf.train()
		probs = torch.zeros([len(data), self.clf.classifier.out_features])
		loader = DataLoader(list(enumerate(data)), shuffle=False, batch_size=64)
		for i in range(n_drop):
			with torch.no_grad():
				for idxs, (x,y,_) in loader:
					x, y = x.to(self.device), y.to(self.device)
					out, e1 = self.clf.forward(x, repr=True)
					prob = F.softmax(out, dim=1)
					probs[idxs] += prob.cpu()
		probs /= n_drop
		return probs

	def predict_prob_dropout_split(self, data, n_drop=10):
		self.clf.train()
		probs = torch.zeros([n_drop, len(data), self.clf.classifier.out_features])
		loader = DataLoader(list(enumerate(data)), shuffle=False, batch_size=64)
		for i in range(n_drop):
			with torch.no_grad():
				for idxs, (x,y,_) in loader:
					x, y = x.to(self.device), y.to(self.device)
					out, e1 = self.clf.forward(x, repr=True)
					prob = F.softmax(out, dim=1)
					probs[i][idxs] += F.softmax(out, dim=1).cpu()
		return probs

	def get_model(self):
		return self.clf

	def get_embeddings(self, data):
		self.clf.eval()
		embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
		loader = DataLoader(list(enumerate(data)), shuffle=False, batch_size=64)
		with torch.no_grad():
			for idxs, (x,y,_) in loader:
				# print(x.shape, y.shape)
				x, y = x.to(self.device), y.to(self.device)
				out, e1 = self.clf.forward(x, repr=True)
				embeddings[idxs] = e1.cpu()
		return embeddings

	def get_grad_embeddings(self, data):
		self.clf.eval()
		embDim = self.clf.get_embedding_dim()
		nLab = self.clf.classifier.out_features
		embeddings = np.zeros([len(data), embDim * nLab])

		loader = DataLoader(list(enumerate(data)), shuffle=False, batch_size=64)
		with torch.no_grad():
			for idxs, (x,y,_) in loader:
				x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
				cout, out = self.clf.forward(x, repr=True)
				out = out.data.cpu().numpy()
				batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
				maxInds = np.argmax(batchProbs,1)
				for j in range(len(y)):
					for c in range(nLab):
						if c == maxInds[j]:
							embeddings[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c]) * -1.0
						else:
							embeddings[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c]) * -1.0

		return embeddings

	def get_p_out(self, data):
		self.clf.eval()
		hs = torch.zeros([len(data), self.clf.classifier.out_features])
		gs = torch.zeros(len(data))
		loader = DataLoader(list(enumerate(data)), shuffle=False, batch_size=64)
		with torch.no_grad():
			for idxs, (x,y,_) in loader:
				x, y = x.to(self.device), y.to(self.device)
				h, g = self.clf.get_p_out(x)
				hs[idxs] = h.cpu()
				gs[idxs] = g.flatten().cpu()
		return hs, gs


class MLP400(nn.Module):
	def __init__(self, num_classes = 10):
		super().__init__()
		self.fc1 = nn.Linear(28*28, 400)  # Input to hidden layer with 400 units
		self.relu = nn.ReLU()  # ReLU activation function
		self.classifier = nn.Linear(400, num_classes)  # Hidden layer to output
		self.dim = 400

	def forward(self, x, repr = False):
		x = x.flatten(1)
		x = self.fc1(x)
		x = self.relu(x)
		output = self.classifier(x)
		if repr:
			return output, x
		return output
	
	def get_embedding_dim(self):
		return self.dim
		

class MNIST_Net(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		self.conv = nn.Conv2d(1, 3, kernel_size = 1)
		self.conv1 = nn.Conv2d(1, 3, kernel_size=7, stride=2, padding=3, bias=False)
		self.classifier = nn.Linear(resnet18.fc.in_features,num_classes)
		self.dim = resnet18.fc.in_features
		
	
	def forward(self, x, repr=False):
		x = self.conv(x)
		feature  = self.features(x)	
		x = feature.view(feature.size(0), -1)	
		output = self.classifier(x)
		if repr:
			return output, x
		return output
	
	def get_embedding_dim(self):
		return self.dim

class CIFAR10_Net(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		features_tmp = nn.Sequential(*list(resnet18.children())[:-1])
		features_tmp[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.features = nn.Sequential(*list(features_tmp))
		self.classifier = nn.Linear(512, num_classes)
		self.dim = resnet18.fc.in_features
	
	def forward(self, x, repr=False):
		feature  = self.features(x)
		x = feature.view(feature.size(0), -1)		
		output = self.classifier(x)
		if repr:
			return output, x
		return output
	
	def get_embedding_dim(self):
		return self.dim
	
class CIFAR32_Net(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet34(pretrained=pretrained)
		features_tmp = nn.Sequential(*list(resnet18.children())[:-1])
		features_tmp[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.features = nn.Sequential(*list(features_tmp))
		self.classifier = nn.Linear(512, num_classes)
		self.dim = resnet18.fc.in_features
	
	def forward(self, x, repr=False):
		feature  = self.features(x)
		x = feature.view(feature.size(0), -1)		
		output = self.classifier(x)
		if repr:
			return output, x
		return output
	
	def get_embedding_dim(self):
		return self.dim
	
class CIFAR10_ODIN_Net(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		features_tmp = nn.Sequential(*list(resnet18.children())[:-1])
		features_tmp[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.features = nn.Sequential(*list(features_tmp))

		### odin architectures
		self.batch_norm = nn.BatchNorm2d(512)  # Modify the number of channels accordingly
		self.relu = nn.ReLU()
		self.avg_pool = nn.AvgPool2d(kernel_size=8)  # Modify kernel_size if needed
		self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
		self.dense_h = nn.Linear(512, num_classes)  # Modify the input size accordingly
		self.dense_g = nn.Linear(512, 1)
		self.batch_norm_g = nn.BatchNorm1d(1)
		self.sigmoid = nn.Sigmoid()

		self.classifier = nn.Linear(512, num_classes)
		self.dim = resnet18.fc.in_features
	
	def forward(self, x, repr=False):
		feature  = self.features(x)
		x = feature.view(x.size(0), -1)

		h = self.dense_h(x)

		g = self.dense_g(x)
		g = self.batch_norm_g(g)
		g = self.sigmoid(g)

		output = h/g
		if repr:
			return output, x
		return output
	
	def get_p_out(self, x):
		feature  = self.features(x)
		x = feature.view(x.size(0), -1)

		h = self.dense_h(x)

		g = self.dense_g(x)
		g = self.batch_norm_g(g)
		g = self.sigmoid(g)

		return h,g
	
	def get_embedding_dim(self):
		return self.dim

class CORe50_Net(nn.Module):
	def __init__(self, pretrained=False, num_classes=10):
		super().__init__()
		resnet34 = models.resnet34(pretrained=pretrained)

		# Modify the first convolutional layer to accept 3 channels and 128x128 images
		resnet34.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

		# Extract features except the final classification layer
		features_tmp = nn.Sequential(*list(resnet34.children())[:-1])
		self.features = nn.Sequential(*list(features_tmp))

		self.classifier = nn.Linear(512, num_classes)  # Output layer
		self.dim = resnet34.fc.in_features  # Get the embedding dimension

	def forward(self, x, repr=False):
		feature = self.features(x)
		x = feature.view(feature.size(0), -1)
		output = self.classifier(x)
		if repr:
			return output, x
		return output

	def get_embedding_dim(self):
		return self.dim

    
class Domain_Net(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 345):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		features_tmp = nn.Sequential(*list(resnet18.children())[:-1])
		features_tmp[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.features = nn.Sequential(*list(features_tmp))
		self.classifier = nn.Linear(512, num_classes)
		self.dim = resnet18.fc.in_features
	
	def forward_deepal(self, x):
		feature  = self.features(x)
		x = feature.view(feature.size(0), -1)		
		output = self.classifier(x)
		return output, x
	
	def forward(self, x):
		feature  = self.features(x)
		x = feature.view(feature.size(0), -1)		
		output = self.classifier(x)
		return output
	
	def get_embedding_dim(self):
		return self.dim

class openml_Net(nn.Module):
    def __init__(self, dim = 28 * 28, embSize=256, pretrained=False, num_classes = 10):
        super(openml_Net, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, num_classes)
    
    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        out = self.lm2(emb)
        return out, emb
    
    def get_embedding_dim(self):
        return self.embSize

class PneumoniaMNIST_Net(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		features_tmp = nn.Sequential(*list(resnet18.children())[:-1])
		features_tmp[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
		self.features = nn.Sequential(*list(features_tmp))
		self.classifier = nn.Linear(512, num_classes)
		self.dim = resnet18.fc.in_features
		
	
	def forward(self, x):
		feature  = self.features(x)
		x = feature.view(feature.size(0), -1)		
		output = self.classifier(x)
		return output, x
	
	def get_embedding_dim(self):
		return self.dim

class waterbirds_Net(nn.Module):
	def __init__(self, dim = 28 * 28, pretrained=False, num_classes = 10):
		super().__init__()
		resnet18 = models.resnet18(pretrained=pretrained)
		self.features = nn.Sequential(*list(resnet18.children())[:-1])
		self.classifier = nn.Linear(resnet18.fc.in_features,num_classes)
		self.dim = resnet18.fc.in_features
		
	def forward(self, x):
		feature  = self.features(x)
		x = feature.view(feature.size(0), -1)		
		output = self.classifier(x)
		return output, x
	
	def get_embedding_dim(self):
		return self.dim


# VAE for VAAL method

def get_net_vae(name):
	if name == 'MNIST':
		return VAE_MNIST, Discriminator
	elif name == 'MNIST_pretrain':
		return VAE_MNIST, Discriminator
	elif name == 'FashionMNIST':
		return VAE_MNIST, Discriminator
	elif name == 'EMNIST':
		return VAE_MNIST, Discriminator
	elif name == 'SVHN':
		return VAE_CIFAR10, Discriminator
	elif name == 'CIFAR10':
		return VAE_CIFAR10, Discriminator
	elif name == 'CIFAR10_imb':
		return VAE_CIFAR10, Discriminator
	elif name == 'CIFAR100':
		return VAE_CIFAR10, Discriminator
	elif name == 'TinyImageNet':
		return VAE_ImageNet, Discriminator
	elif name == 'openml':
		raise NotImplementedError
	elif name == 'BreakHis':
		return VAE_CIFAR10, Discriminator
	elif name == 'PneumoniaMNIST':
		return VAE_CIFAR10, Discriminator
	elif name == 'waterbirds':
		return VAE_waterbirds, Discriminator
	elif name == 'waterbirds_pretrain':
		return VAE_waterbirds, Discriminator
	else:
		raise NotImplementedError


class VAE_CIFAR10(nn.Module):
		"""Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
		def __init__(self, z_dim=32, nc=3):
				super(VAE_CIFAR10, self).__init__()
				self.z_dim = z_dim
				self.nc = nc
				self.encoder = nn.Sequential(
						nn.Conv2d(nc, 128, 4, 2, 1, bias=False),		
						nn.BatchNorm2d(128),
						nn.ReLU(True),
						nn.Conv2d(128, 256, 4, 2, 1, bias=False),	
						nn.BatchNorm2d(256),
						nn.ReLU(True),
						nn.Conv2d(256, 512, 4, 2, 1, bias=False),	
						nn.BatchNorm2d(512),
						nn.ReLU(True),
						nn.Conv2d(512, 1024, 4, 2, 1, bias=False),		
						nn.BatchNorm2d(1024),
						nn.ReLU(True),
						
						View((-1, 1024*2*2)),												
				)

				self.fc_mu = nn.Linear(1024*2*2, z_dim)											
				self.fc_logvar = nn.Linear(1024*2*2, z_dim)											
				self.decoder = nn.Sequential(
						nn.Linear(z_dim, 1024*4*4),										
						View((-1, 1024, 4, 4)),								
						nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False), 
						nn.BatchNorm2d(512),
						nn.ReLU(True),
						nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),	
						nn.BatchNorm2d(256),
						nn.ReLU(True),
						nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
						nn.BatchNorm2d(128),
						nn.ReLU(True),
						nn.ConvTranspose2d(128, nc, 1),			
				)
				self.weight_init()

		def weight_init(self):
				for block in self._modules:
						try:
								for m in self._modules[block]:
										kaiming_init(m)
						except:
								kaiming_init(block)

		def forward(self, x):
				z = self._encode(x)
				mu, logvar = self.fc_mu(z), self.fc_logvar(z)
				z = self.reparameterize(mu, logvar)
				x_recon = self._decode(z)

				return x_recon, z, mu, logvar

		def reparameterize(self, mu, logvar):
				stds = (0.5 * logvar).exp()
				epsilon = torch.randn(*mu.size())
				if mu.is_cuda:
						stds, epsilon = stds.cuda(), epsilon.cuda()
				latents = epsilon * stds + mu
				return latents

		def _encode(self, x):
				return self.encoder(x)

		def _decode(self, z):
				return self.decoder(z)

class VAE_MNIST(nn.Module):
		"""Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
		def __init__(self, dim = 32, nc=1):
				super(VAE_MNIST, self).__init__()
				dim = np.prod(dim)
				self.z_dim = dim
				self.nc = nc

				self.encoder = nn.Sequential(
					nn.Conv2d(nc, 32, 4, 1, 2),  
					nn.ReLU(True),
					nn.Conv2d(32, 32, 4, 2, 1), 
					nn.ReLU(True),
					nn.Conv2d(32, 64, 4, 2, 1), 
				)
				
				self.fc_mu = nn.Linear(64 * 7 * 7, dim)
				self.fc_logvar = nn.Linear(64 * 7 * 7, dim)
				
				self.upsample = nn.Linear(dim, 64 * 7 * 7)
				self.decoder = nn.Sequential(
				   	nn.ConvTranspose2d(64, 32, 4, 2, 1), 
					nn.ReLU(True),
					nn.ConvTranspose2d(32, 32, 4, 2, 1, 1),
					nn.ReLU(True),
					nn.ConvTranspose2d(32, nc, 4, 1, 2) 
				)
				self.weight_init()
				self.conv = nn.Conv2d(1, 3, kernel_size = 1)

		def weight_init(self):
				for block in self._modules:
						try:
								for m in self._modules[block]:
										kaiming_init(m)
						except:
								kaiming_init(block)

		def forward(self, x):
				z = self._encode(x).relu().view(x.size(0), -1)
				mu, logvar = self.fc_mu(z), self.fc_logvar(z)
				z = self.reparameterize(mu, logvar)
				x_recon = self._decode(self.upsample(z).relu().view(-1, 64, 7, 7))
				return x_recon, z, mu, logvar

		def reparameterize(self, mu, logvar):
				stds = (0.5 * logvar).exp()
				epsilon = torch.randn(*mu.size())
				if mu.is_cuda:
						stds, epsilon = stds.cuda(), epsilon.cuda()
				latents = epsilon * stds + mu
				return latents

		def _encode(self, x):
				return self.encoder(x)

		def _decode(self, z):
				return self.decoder(z)


class VAE_ImageNet(nn.Module):
		"""Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
		def __init__(self, z_dim=32, nc=3):
				super(VAE_ImageNet, self).__init__()
				self.z_dim = z_dim
				self.nc = nc
				self.encoder = nn.Sequential(
						nn.Conv2d(nc, 128, 4, 1, 2, bias=False),			
						nn.BatchNorm2d(128),
						nn.ReLU(True),
						nn.Conv2d(128, 256, 4, 2, 1, bias=False),			
						nn.BatchNorm2d(256),
						nn.ReLU(True),
						nn.Conv2d(256, 512, 4, 2, 1, bias=False),				
						nn.BatchNorm2d(512),
						nn.ReLU(True),
						nn.Conv2d(512, 1024, 4, 2, 1, bias=False),					
						nn.BatchNorm2d(1024),
						nn.ReLU(True),
						View((-1, 1024*4*4)),											
				)

				self.fc_mu = nn.Linear(1024*4*4, z_dim)												
				self.fc_logvar = nn.Linear(1024*4*4, z_dim)												
				self.decoder = nn.Sequential(
						nn.Linear(z_dim, 1024*8*8),												

						View((-1, 1024, 8, 8)),													
						nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False), 
						nn.BatchNorm2d(512),
						nn.ReLU(True),
						nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),		
						nn.BatchNorm2d(256),
						nn.ReLU(True),
						nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),	
						nn.BatchNorm2d(128),
						nn.ReLU(True),
						nn.ConvTranspose2d(128, nc, 1),								
				)
				self.weight_init()

		def weight_init(self):
				for block in self._modules:
						try:
								for m in self._modules[block]:
										kaiming_init(m)
						except:
								kaiming_init(block)

		def forward(self, x):
				z = self._encode(x)
				mu, logvar = self.fc_mu(z), self.fc_logvar(z)
				z = self.reparameterize(mu, logvar)
				x_recon = self._decode(z)

				return x_recon, z, mu, logvar

		def reparameterize(self, mu, logvar):
				stds = (0.5 * logvar).exp()
				epsilon = torch.randn(*mu.size())
				if mu.is_cuda:
						stds, epsilon = stds.cuda(), epsilon.cuda()
				latents = epsilon * stds + mu
				return latents

		def _encode(self, x):
				return self.encoder(x)

		def _decode(self, z):
				return self.decoder(z)

class VAE_waterbirds(nn.Module):
		"""Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
		def __init__(self, z_dim=32, nc=3):
				super(VAE_waterbirds, self).__init__()
				self.z_dim = z_dim
				self.nc = nc
				self.encoder = nn.Sequential(
						nn.Conv2d(nc, 128, 3, 2, 0, bias=False),				
						nn.BatchNorm2d(128),
						nn.ReLU(True),
						nn.Conv2d(128, 256, 3, 2, 0, bias=False),		
						nn.BatchNorm2d(256),
						nn.ReLU(True),
						nn.Conv2d(256, 512, 3, 2, 0, bias=False),			
						nn.BatchNorm2d(512),
						nn.Conv2d(512, 1024, 3, 2, 0, bias=False),		
						nn.BatchNorm2d(1024),
						nn.ReLU(True),
						nn.Conv2d(1024, 2048, 3, 2, 0, bias=False),			
						nn.BatchNorm2d(2048),
						nn.ReLU(True),

						nn.Conv2d(2048, 4096, 3, 2, 0, bias=False),			
						nn.BatchNorm2d(4096),
						nn.ReLU(True),
						
						View((-1, 4096*3*3)),
				)

				self.fc_mu = nn.Linear(4096*3*3, z_dim)										
				self.fc_logvar = nn.Linear(4096*3*3, z_dim)												
				self.decoder = nn.Sequential(
						nn.Linear(z_dim, 4096*3*3),											  
						View((-1, 4096, 3, 3)),											
						nn.ConvTranspose2d(4096, 2048, 3, 2, 0, bias=False),   
						nn.BatchNorm2d(2048),
						nn.ReLU(True),
						nn.ConvTranspose2d(2048, 1024, 3, 2, 0, bias=False),	
						nn.BatchNorm2d(1024),
						nn.ReLU(True),
						nn.ConvTranspose2d(1024, 512, 3, 2, 0, bias=False),		
						nn.BatchNorm2d(512),
						nn.ReLU(True),
						nn.ConvTranspose2d(512, 256, 3, 2, 0, bias=False),		
						nn.BatchNorm2d(256),
						nn.ReLU(True),
						nn.ConvTranspose2d(256, 128, 3, 2, 0, bias=False),		
						nn.BatchNorm2d(128),
						nn.ReLU(True),
						nn.ConvTranspose2d(128, 64, 3, 2, 0, bias=False),	
						nn.BatchNorm2d(64),
						nn.ReLU(True),
						nn.ConvTranspose2d(64, nc, 1),				
				)
				self.weight_init()

		def weight_init(self):
				for block in self._modules:
						try:
								for m in self._modules[block]:
										kaiming_init(m)
						except:
								kaiming_init(block)

		def forward(self, x):
				z = self._encode(x)
				mu, logvar = self.fc_mu(z), self.fc_logvar(z)
				z = self.reparameterize(mu, logvar)
				x_recon = self._decode(z)

				return x_recon, z, mu, logvar

		def reparameterize(self, mu, logvar):
				stds = (0.5 * logvar).exp()
				epsilon = torch.randn(*mu.size())
				if mu.is_cuda:
						stds, epsilon = stds.cuda(), epsilon.cuda()
				latents = epsilon * stds + mu
				return latents

		def _encode(self, x):
				return self.encoder(x)

		def _decode(self, z):
				return self.decoder(z)

class Discriminator(nn.Module):
		"""Adversary architecture(Discriminator) for WAE-GAN."""
		def __init__(self, dim=32):
				super(Discriminator, self).__init__()
				self.dim = np.prod(dim)
				self.net = nn.Sequential(
						nn.Linear(self.dim, 512),
						nn.ReLU(True),
						nn.Linear(512, 512),
						nn.ReLU(True),
						nn.Linear(512,1),
						nn.Sigmoid(),
				)
				self.weight_init()

		def weight_init(self):
				for block in self._modules:
						for m in self._modules[block]:
								kaiming_init(m)

		def forward(self, z):
				return self.net(z).reshape(-1)

class View(nn.Module):
		def __init__(self, size):
				super(View, self).__init__()
				self.size = size

		def forward(self, tensor):
				return tensor.view(self.size)

def kaiming_init(m):
		if isinstance(m, (nn.Linear, nn.Conv2d)):
				init.kaiming_normal(m.weight)
				if m.bias is not None:
						m.bias.data.fill_(0)
		elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
				m.weight.data.fill_(1)
				if m.bias is not None:
						m.bias.data.fill_(0)

def normal_init(m, mean, std):
		if isinstance(m, (nn.Linear, nn.Conv2d)):
				m.weight.data.normal_(mean, std)
				if m.bias.data is not None:
						m.bias.data.zero_()
		elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
				m.weight.data.fill_(1)
				if m.bias.data is not None:
						m.bias.data.zero_()


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print('p',x.shape)      #print(x.shape)
        return x