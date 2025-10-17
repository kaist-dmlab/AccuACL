from tqdm import tqdm
from copy import deepcopy

import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.distributions.categorical import Categorical

from acl.augment import RandAugment
from acl.data import get_statistics
from avalanche.training.checkpoint import maybe_load_checkpoint
from avalanche.training.determinism.rng_manager import RNGManager
from .util import DataHandler 

def are_state_dicts_equal(state_dict1, state_dict2, device):
    for key1, key2 in zip(state_dict1.keys(), state_dict2.keys()):
        if key1 != key2:
            return False
        tensor1 = state_dict1[key1]
        tensor2 = state_dict2[key2]
        if not torch.allclose(tensor1.to(device), tensor2.to(device), rtol=1e-3, atol=1e-5):
            return False
    return True

class Memory(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.targets = y
        self.transform = transform

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform:
            x = self.transform(x)
        y = self.targets[index]
        return x, y

    def __len__(self):
        return self.targets.size(0)

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class Strategy:
    def __init__(self, X, Y, idxs_lb, cl_strategy, args, fname=None, **kwargs):
        self.X = X
        self.Y = Y
        self.idxs_lb = idxs_lb
        self.cl_strategy = cl_strategy
        self.device = cl_strategy.device
        self.n_pool = len(Y)
        self.n_classes = args.n_classes
        self.fname = fname
        self.num_workers = args.num_workers
        self.dim = cl_strategy.model.get_embedding_dim()
        self.mean = get_statistics(args.data)['mean']
        self.std = get_statistics(args.data)['std']
        use_cuda = torch.cuda.is_available()
        self.mem_X = None
        self.mem_Y = None
        if len(self.cl_memory())!=0:
            self.mem_X = torch.stack([data[0] for data in self.cl_memory('eval')])
            self.mem_Y = torch.tensor(self.cl_memory().targets)

    def query(self, n, **kwargs):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb
    
    def cl_memory(self, transform='train'):
        if hasattr(self.cl_strategy, 'eca_memory'):
            return self.cl_strategy.storage_policy.buffer.eval().with_transforms(transform)
        if hasattr(self.cl_strategy, 'rp'):
            return self.cl_strategy.rp.storage_policy.buffer.eval().with_transforms(transform)
        elif hasattr(self.cl_strategy, 'mir'):
            return self.cl_strategy.mir.storage_policy.buffer.eval().with_transforms(transform)
        elif hasattr(self.cl_strategy, 'storage_policy'):
            return self.cl_strategy.storage_policy.buffer.eval().with_transforms(transform)
        elif hasattr(self.cl_strategy.plugins[0], 'memory_dataset'):
            return self.cl_strategy.plugins[0].memory_dataset.eval().with_transforms(transform)
        elif hasattr(self.cl_strategy.plugins[0], 'storage_policy'):
            return self.cl_strategy.plugins[0].storage_policy.buffer.eval().with_transforms(transform)
        elif hasattr(self.cl_strategy.plugins[0], 'ext_mem_list_x'):
            return Memory(
                    X = self.cl_strategy.plugins[0].ext_mem_list_x,
                    y = self.cl_strategy.plugins[0].ext_mem_list_y,
                )
        else:
            raise ValueError('no corresponding memory for cl strategy')
 
    def train(self, exp, no_update=False, first=False, full_data=False, fname=None, seed=None, epochs=None):        # if not first:
        if fname: 
            self.cl_strategy, initial_exp = maybe_load_checkpoint(self.cl_strategy, fname)
        else:
            self.cl_strategy, initial_exp = maybe_load_checkpoint(self.cl_strategy, self.fname)
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        if seed:
            RNGManager.set_random_seeds(seed)
        if epochs:
            self.cl_strategy.train_epochs=epochs
        self.cl_strategy.model.train()
        self.cl_strategy.train(exp, num_workers=self.num_workers, custom_selection_result = idxs_train, no_update = no_update, full_data = full_data)

    def get_dist(self, epochs, nEns=1, opt='adam', verbose=False):

        def weight_reset(m):
            newLayer = deepcopy(m)
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                newLayer.reset_parameters()
                m.reset_parameters()

        if verbose: print(' ',flush=True)
        if verbose: print('training to indicated number of epochs', flush=True)

        ce = nn.CrossEntropyLoss()
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(DataHandler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long()), shuffle=True, batch_size=64, num_workers=self.num_workers)
        dataSize = len(idxs_train)        
        N = np.round((epochs * len(loader_tr)) ** 0.5)
        allAvs = []
        allWeights = []
        for m in range(nEns):

            # initialize new model and optimizer
            net =  self.net.apply(weight_reset).to(self.device)
            if opt == 'adam': optimizer = optim.Adam(net.parameters(), lr=self.args['lr'], weight_decay=0)
            if opt == 'sgd': optimizer = optim.SGD(net.parameters(), lr=self.args['lr'], weight_decay=0)
        
            nUpdates = k = 0
            ek = (k + 1) * N
            pVec = torch.cat([torch.zeros_like(p).cpu().flatten() for p in self.cl_strategy.model.parameters()])

            avIterates = []
            for epoch in range(epochs + 1):
                correct = lossTrain = 0.
                net = net.train()
                for ind, (x, y, _) in enumerate(loader_tr):
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    output, _ = net(x)
                    correct += torch.sum(output.argmax(1) == y).item()
                    loss = ce(output, y)
                    loss.backward()
                    lossTrain += loss.item() * len(y)
                    optimizer.step()
                    flat = torch.cat([deepcopy(p.detach().cpu()).flatten() for p in net.parameters()])
                    pVec = pVec + flat
                    nUpdates += 1
                    if nUpdates > ek:
                        avIterates.append(pVec / N)
                        pVec = torch.cat([torch.zeros_like(p).cpu().flatten() for p in net.parameters()])
                        k += 1
                        ek = (k + 1) * N

                lossTrain /= dataSize
                accuracy = correct / dataSize
                if verbose: print(m, epoch, nUpdates, accuracy, lossTrain, flush=True)
            allAvs.append(avIterates)
            allWeights.append(torch.cat([deepcopy(p.detach().cpu()).flatten() for p in net.parameters()]))

        for m in range(nEns):
            avIterates = torch.stack(allAvs[m])
            if k > 1: avIterates = torch.stack(allAvs[m][1:])
            avIterates = avIterates - torch.mean(avIterates, 0)
            allAvs[m] = avIterates

        return allWeights, allAvs, optimizer, net

    def getNet(self, params):
        i = 0
        model = deepcopy(self.cl_strategy.model).to(self.device)
        for p in model.parameters():
            L = len(p.flatten())
            param = params[i:(i + L)]
            p.data = param.view(p.size())
            i += L
        return model

    def fitBatchnorm(self, model):
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(DataHandler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long()), shuffle=True, batch_size=64, num_workers=self.num_workers)
        model = model.to(self.device)
        for ind, (x, y, _) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            output = self.cl_strategy.model(x, repr=True)
        return model

    def sampleNet(self, weights, iterates):
        nEns = len(weights)
        k = len(iterates[0])
        i = np.random.randint(nEns)
        z = torch.randn(k, 1)
        weightSample = weights[i].view(-1) - torch.mm(iterates[i].t(), z).view(-1) / np.sqrt(k)
        sampleNet = self.getNet(weightSample).to(self.device)
        sampleNet = self.fitBatchnorm(sampleNet)
        return sampleNet

    def getPosterior(self, weights, iterates, X, Y, nSamps=50):
        net = self.fitBatchnorm(self.sampleNet(weights, iterates))
        output = self.predict_prob(X, Y, model=net) / nSamps
        print(' ', flush=True)
        ce = nn.CrossEntropyLoss()
        print('sampling models', flush=True)
        for i in range(nSamps - 1):
            net = self.fitBatchnorm(self.sampleNet(weights, iterates))
            output = output + self.predict_prob(X, Y, model=net) / nSamps
            print(i+2, torch.sum(torch.argmax(output, 1) == Y).item() / len(Y), flush=True)
        return output.numpy()

    def predict(self, X, Y):
        if type(X) is np.ndarray:
            loader_te = DataLoader(DataHandler(X, Y),
                            shuffle=False, batch_size=64, num_workers=self.num_workers)
        else: 
            loader_te = DataLoader(DataHandler(X.numpy(), Y),
                            shuffle=False, batch_size=64, num_workers=self.num_workers)

        self.cl_strategy.model.eval()
        P = torch.zeros(len(Y)).long()
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                out, e1 = self.cl_strategy.model(x, repr=True)
                pred = out.max(1)[1]
                P[idxs] = pred.data.cpu()
        return P

    def predict_prob(self, X, Y, model=[], exp=True):
        
        model = self.cl_strategy.model
        loader_te = DataLoader(DataHandler(X, Y), shuffle=False, batch_size=64, num_workers=self.num_workers)
        model = model.eval()
        probs = torch.zeros([len(Y), self.n_classes])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                out, e1 = self.cl_strategy.model(x, repr=True)
                if exp: out = F.softmax(out, dim=1)
                probs[idxs] = out.cpu().data
        
        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        loader_te = DataLoader(DataHandler(X, Y),
                            shuffle=False, batch_size=64, num_workers=self.num_workers)

        self.cl_strategy.model.train()
        probs = torch.zeros([len(Y), self.n_classes])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                    out, e1 = self.cl_strategy.model(x, repr=True)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += out.cpu().data
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        loader_te = DataLoader(DataHandler(X, Y),
                            shuffle=False, batch_size=64, num_workers=self.num_workers)

        self.cl_strategy.model.train()
        probs = torch.zeros([n_drop, len(Y), self.n_classes])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                    out, e1 = self.cl_strategy.model(x, repr=True)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def get_embedding(self, X, return_probs=False):
        loader_te = DataLoader(DataHandler(X, torch.zeros(len(X))),
                            shuffle=False, batch_size=64, num_workers=self.num_workers)
        self.cl_strategy.model.eval()
        embedding = torch.zeros([len(X), self.cl_strategy.model.get_embedding_dim()])
        probs = torch.zeros(len(X), self.n_classes)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.cl_strategy.model(x, repr=True)
                embedding[idxs] = e1.data.cpu()
                if return_probs:
                     pr = F.softmax(out,1)
                     probs[idxs] = pr.data.cpu()
        if return_probs: return embedding, probs
        return embedding
    
    def get_embedding_exp_class(self, X, exp_class, return_probs=False, temp=1):
        loader_te = DataLoader(DataHandler(X, torch.zeros_like(X)),
                            shuffle=False, batch_size=64, num_workers=self.num_workers)
        self.cl_strategy.model.eval()
        embedding = torch.zeros([len(X), self.cl_strategy.model.get_embedding_dim()])
        probs = torch.zeros(len(X), len(exp_class))
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.cl_strategy.model(x, repr=True)
                embedding[idxs] = e1.data.cpu()
                if return_probs:
                     pr = F.softmax(out[:,exp_class]/temp,1)
                     probs[idxs] = pr.data.cpu()
        if return_probs: return embedding, probs
        return embedding

    # gradient embedding for badge (assumes cross-entropy loss)
    def get_grad_embedding(self, X, Y, model=[], exp_classes=None):

        model=self.cl_strategy.model     
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = self.n_classes

        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(DataHandler(X, Y),
                            shuffle=False, batch_size=64, num_workers=self.num_workers)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                cout, out = self.cl_strategy.model(x, repr=True)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)
    
    def get_exp_grad_embedding(self, X, probs=[], model=[], exp_classes=None, temp=1):

        model=self.cl_strategy.model     
        embDim = model.get_embedding_dim()
        model.eval()
        if exp_classes is None:
            nLab = self.n_classes
        else:
            nLab = len(exp_classes)
        embedding = np.zeros([len(X), nLab, embDim * nLab])
        for ind in range(nLab):
            loader_te = DataLoader(DataHandler(X, torch.zeros(len(X))),
                            shuffle=False, batch_size=64, num_workers=self.num_workers)
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                    cout, out = self.cl_strategy.model(x, repr=True)
                    out = out.data.cpu().numpy()
                    batchProbs = F.softmax(cout[:,exp_classes]/temp, dim=1).data.cpu().numpy()
                    for j in range(len(y)):
                        for c in range(nLab):
                            if c == ind:
                                embedding[idxs[j]][ind][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])/temp
                            else:
                                embedding[idxs[j]][ind][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])/temp
                        if len(probs) > 0: embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(probs[idxs[j]][ind])
                        else: embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * np.sqrt(batchProbs[j][ind])
        return torch.Tensor(embedding)
    
    def get_labeled_grad_embedding(self, X, Y, exp_classes=None):
        model=self.cl_strategy.model     
        embDim = model.get_embedding_dim()
        model.eval()
        if exp_classes is None:
            nLab = self.n_classes
        else:
            nLab = len(exp_classes)

        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(DataHandler(X, Y),
                            shuffle=False, batch_size=64, num_workers=self.num_workers)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                cout, out = self.cl_strategy.model(x, repr=True)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout[:, exp_classes], dim=1).data.cpu().numpy()
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == y[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
        return torch.Tensor(embedding)
    
    def get_fisher_embedding(self, X, exp_classes=None):

        model = self.cl_strategy.model
        embDim = model.get_embedding_dim()
        model.eval()

        if exp_classes is None:
            nLab = self.n_classes
        else:
            nLab = len(exp_classes)

        embedding = np.zeros([len(X), embDim*nLab])
        loader_te = DataLoader(DataHandler(X,torch.zeros(len(X))), shuffle=False, batch_size=64, num_workers=self.num_workers)
        with torch.no_grad():
            for x, _, idxs in loader_te:
                x = Variable(x.to(self.device))
                cout, out = self.cl_strategy.model(x, repr=True)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout[:,exp_classes], dim=1).data.cpu().numpy()
                for j in range(len(x)):
                    for c in range(nLab):
                            embedding[idxs[j]][embDim*c:embDim*(c+1)] += batchProbs[j][c]*(1-batchProbs[j][c])*np.square(deepcopy(out[j]))

        return torch.Tensor(embedding)
    
    def get_labeled_fisher_embedding(self, X, Y, exp_classes=None):
        model = self.cl_strategy.model
        embDim = model.get_embedding_dim()
        model.eval()
        if exp_classes is None:
            nLab = self.n_classes
        else:
            nLab = len(exp_classes)

        embedding = np.zeros([len(X), embDim*nLab])
        loader_te = DataLoader(DataHandler(X, Y), shuffle=False, batch_size=64, num_workers=self.num_workers)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x = Variable(x.to(self.device))
                cout, out = self.cl_strategy.model(x, repr=True)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout[:,exp_classes], dim=1).data.cpu().numpy()
                for j in range(len(x)):
                    for c in range(nLab):
                        if c == y[j]:
                            embedding[idxs[j]][embDim*c:embDim*(c+1)] = np.square(deepcopy(out[j]) * (1-batchProbs[j][c]))
                        else:
                            embedding[idxs[j]][embDim*c:embDim*(c+1)] = np.square(deepcopy(out[j]) * (-1*batchProbs[j][c]))
        return torch.Tensor(embedding)


    def get_diag_fisher(self, X, Y, exp_classes, temp=1, model=[]):
        model=self.cl_strategy.model.eval()
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(exp_classes)

        embedding = np.zeros([len(Y), nLab, embDim * nLab])
        for ind in tqdm(range(nLab)):
            loader_te = DataLoader(DataHandler(X, Y),
                            shuffle=False, batch_size=64, num_workers=self.num_workers)
            with torch.no_grad():
                for x, y, idxs in loader_te:
                    x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                    cout, out = self.cl_strategy.model(x, repr=True)
                    out = out.data.cpu().numpy()
                    batchProbs = F.softmax(cout[:, exp_classes]/temp, dim=1).data.cpu().numpy()
                    for j in range(len(y)):
                        for c in range(nLab):
                            if c == ind:
                                embedding[idxs[j]][ind][embDim * c : embDim * (c+1)] = np.square(deepcopy(out[j]) * (1 - batchProbs[j][c])/temp)
                            else:
                                embedding[idxs[j]][ind][embDim * c : embDim * (c+1)] = np.square(deepcopy(out[j]) * (-1 * batchProbs[j][c])/temp)
                        embedding[idxs[j]][ind] = embedding[idxs[j]][ind] * batchProbs[j][ind]
        return torch.Tensor(embedding.sum(1))
    
    def get_diag_fisher_labeled(self, X, Y, model=[]):
        model=self.cl_strategy.model
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = self.n_classes
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(DataHandler(X, Y),
                            shuffle=False, batch_size=64, num_workers=self.num_workers)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                cout, out = self.cl_strategy.model(x, repr=True)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = np.square(deepcopy(out[j]) * (1 - batchProbs[j][c]))
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = np.square(deepcopy(out[j]) * (-1 * batchProbs[j][c]))
            return torch.Tensor(embedding)
    
    def get_sample_fim(self, X, Y):
        embs = np.zeros([len(X), len(self.cl_strategy.model.conv1.weight.flatten())])
        loader_te = DataLoader(DataHandler(X, Y), shuffle=False, batch_size=64, num_workers=self.num_workers)
        for x,y,idxs in tqdm(loader_te):
            x= x.to(self.device)
            self.cl_strategy.optimizer.zero_grad()
            cout, out = self.cl_strategy.model(x, repr=True)
            # import pdb;pdb.set_trace()
            logits = F.log_softmax(cout, dim=1).max(dim=1).values
            for i in range(len(x)):
                self.cl_strategy.optimizer.zero_grad()
                torch.autograd.backward(logits[i], retain_graph=True)
                embs[idxs[i]] = self.cl_strategy.model.conv1.weight.grad.data.flatten().detach().cpu().pow(2).numpy()
        return torch.Tensor(embs)
    
    def get_fisher_odin(self,X,Y):
        embs = np.zeros([len(X), len(self.cl_strategy.model.g[0].weight.flatten())])
        loader_te = DataLoader(DataHandler(X, Y), shuffle=False, batch_size=64, num_workers=self.num_workers)
        for x,y,idxs in tqdm(loader_te):
            x= x.to(self.device)
            self.cl_strategy.optimizer.zero_grad()
            cout, p_x = self.cl_strategy.model(x, p_x=True)
            for i in range(len(x)):
                self.cl_strategy.optimizer.zero_grad()
                torch.autograd.backward(torch.log(p_x[i]), retain_graph=True)
                embs[idxs[i]] = self.cl_strategy.model.g[0].weight.grad.data.flatten().detach().cpu().pow(2).numpy()
        return torch.Tensor(embs)
    
    def get_fisher_cnn_odin(self,X,Y):
        embs = np.zeros([len(X), len(self.cl_strategy.model.conv1.weight.flatten())])
        loader_te = DataLoader(DataHandler(X, Y), shuffle=False, batch_size=64, num_workers=self.num_workers)
        for x,y,idxs in tqdm(loader_te):
            x= x.to(self.device)
            self.cl_strategy.optimizer.zero_grad()
            cout, p_x = self.cl_strategy.model(x, p_x=True)
            for i in range(len(x)):
                self.cl_strategy.optimizer.zero_grad()
                torch.autograd.backward(torch.log(p_x[i]), retain_graph=True)
                embs[idxs[i]] = self.cl_strategy.model.conv1.weight.grad.data.flatten().detach().cpu().pow(2).numpy()
        return torch.Tensor(embs)
    
    def get_fim_linear(self, X, Y):
        embs = np.zeros([len(X), len(self.cl_strategy.model.linear.weight.flatten())])
        loader_te = DataLoader(DataHandler(X, Y), shuffle=False, batch_size=64, num_workers=self.num_workers)
        for x,y,idxs in tqdm(loader_te):
            x= x.to(self.device)
            self.cl_strategy.optimizer.zero_grad()
            cout, out = self.cl_strategy.model(x, repr=True)
            logits = F.log_softmax(cout, dim=1).max(dim=1).values
            for i in range(len(x)):
                self.cl_strategy.optimizer.zero_grad()
                torch.autograd.backward(logits[i], retain_graph=True)
                embs[idxs[i]] = self.cl_strategy.model.linear.weight.grad.data.flatten().detach().cpu().pow(2).numpy()
        return torch.Tensor(embs)
    
    def get_fim_diag(self, embs, probs, dim, temp=100):
        embedding = torch.zeros([len(probs)*dim]).cpu()
        for idx, p in enumerate(probs.flatten()):
            embedding[dim*idx:dim*(idx+1)] += p*(1-p)*embs.pow(2)/temp**2
        return embedding
    
    def get_fim_prompt(self, X, Y):
        embs = np.zeros([len(X), len(self.cl_strategy.model.prompt.prompt.flatten())])
        loader_te = DataLoader(DataHandler(X, Y), shuffle=False, batch_size=64, num_workers=self.num_workers)
        for x,y,idxs in tqdm(loader_te):
            x= x.to(self.device)
            self.cl_strategy.optimizer.zero_grad()
            cout, out = self.cl_strategy.model(x, repr=True)
            logits = F.log_softmax(cout, dim=1).max(dim=1).values
            for i in range(len(x)):
                self.cl_strategy.optimizer.zero_grad()
                torch.autograd.backward(logits[i], retain_graph=True)
                embs[idxs[i]] = self.cl_strategy.model.prompt.prompt.grad.data.flatten().detach().cpu().pow(2).numpy()
        return torch.Tensor(embs)
    
    def get_fim_g_prompt(self, X, Y):
        embs = np.zeros([len(X), len(self.cl_strategy.model.g_prompt.flatten())])
        loader_te = DataLoader(DataHandler(X, Y), shuffle=False, batch_size=64, num_workers=self.num_workers)
        for x,y,idxs in tqdm(loader_te):
            x= x.to(self.device)
            self.cl_strategy.optimizer.zero_grad()
            cout, out = self.cl_strategy.model(x, repr=True)
            logits = F.log_softmax(cout, dim=1).max(dim=1).values
            for i in range(len(x)):
                self.cl_strategy.optimizer.zero_grad()
                torch.autograd.backward(logits[i], retain_graph=True)
                embs[idxs[i]] = self.cl_strategy.model.g_prompt.grad.data.flatten().detach().cpu().pow(2).numpy()
        return torch.Tensor(embs)

    def get_fim_cnn(self, X, exp_classes):
        dim=0
        dim+=len(self.cl_strategy.model.conv1.weight.grad.flatten())
        embs = np.zeros([len(X), dim])
        loader_te = DataLoader(DataHandler(X, torch.zeros(len(X))), shuffle=False, batch_size=64, num_workers=self.num_workers)
        for x,y,idxs in tqdm(loader_te):
            x= x.to(self.device)
            self.cl_strategy.optimizer.zero_grad()
            cout, out = self.cl_strategy.model(x, repr=True)
            # import pdb;pdb.set_trace()
            probs = F.softmax(cout[:, exp_classes], dim=1)
            for i in range(len(x)):
                emb=0
                for p in probs[i]:
                    self.cl_strategy.optimizer.zero_grad()
                    torch.autograd.backward(torch.log(p), retain_graph=True)
                    emb+=p.detach().cpu().numpy()*self.cl_strategy.model.conv1.weight.grad.data.detach().cpu().pow(2).flatten().numpy()
                embs[idxs[i]] = emb
        return torch.Tensor(embs)

    def get_fim_cnn_deep(self, X, exp_classes):
        dim=0
        for k, v, in self.cl_strategy.model.layer1.named_parameters():
            if 'conv' in k:
                dim+=len(v.flatten())
        embs = np.zeros([len(X), dim])
        loader_te = DataLoader(DataHandler(X, torch.zeros(len(X))), shuffle=False, batch_size=64, num_workers=self.num_workers)
        for x,y,idxs in tqdm(loader_te):
            x= x.to(self.device)
            self.cl_strategy.optimizer.zero_grad()
            cout, out = self.cl_strategy.model(x, repr=True)
            # import pdb;pdb.set_trace()
            probs = F.softmax(cout[:, exp_classes], dim=1)
            for i in range(len(x)):
                emb=0
                for p in probs[i]:
                    self.cl_strategy.optimizer.zero_grad()
                    torch.autograd.backward(torch.log(p), retain_graph=True)
                    grads=[]
                    for k, v in self.cl_strategy.model.layer1.named_parameters():
                        if 'conv' in k:
                            grads.append(v.grad.data.detach().cpu().pow(2).numpy())
                    emb+=p.detach().cpu().numpy()*np.array(grads).flatten()
                embs[idxs[i]] = emb
        return torch.Tensor(embs)
    
    def get_fim_cnn_emp(self, X, exp_classes):
        dim=0
        dim+=len(self.cl_strategy.model.conv1.weight.grad.flatten())
        embs = np.zeros([len(X), dim])
        loader_te = DataLoader(DataHandler(X, torch.zeros(len(X))), shuffle=False, batch_size=64, num_workers=self.num_workers)
        for x,y,idxs in tqdm(loader_te):
            x= x.to(self.device)
            self.cl_strategy.optimizer.zero_grad()
            cout, out = self.cl_strategy.model(x, repr=True)
            # import pdb;pdb.set_trace()
            probs = F.softmax(cout[:,exp_classes], dim=1)
            for i in range(len(x)):
                self.cl_strategy.optimizer.zero_grad()
                torch.autograd.backward(torch.log(probs[i].max(dim=0).values), retain_graph=True)
                embs[idxs[i]] = self.cl_strategy.model.conv1.weight.grad.data.detach().cpu().pow(2).flatten().numpy()
        return torch.Tensor(embs)

    def get_fisher_cnn_deep(self, X, Y):
        dim=0
        for k, v, in self.cl_strategy.model.layer1.named_parameters():
            if 'conv' in k:
                dim+=len(v.flatten())
        embs = np.zeros([len(X), dim])
        loader_te = DataLoader(DataHandler(X, Y), shuffle=False, batch_size=64, num_workers=self.num_workers)
        for x,y,idxs in tqdm(loader_te):
            x= x.to(self.device)
            self.cl_strategy.optimizer.zero_grad()
            logit, out = self.cl_strategy.model(x, repr=True)
            outdx = Categorical(logits=logit).sample().detach().unsqueeze(1)
            samples = logit.gather(1, outdx)
            # import pdb;pdb.set_trace()
            # logits = F.log_softmax(cout, dim=1).max(dim=1).values
            for i in range(len(x)):
                self.cl_strategy.optimizer.zero_grad()
                torch.autograd.backward(samples[i], retain_graph=True)
                grads=[]
                for k, v in self.cl_strategy.model.layer1.named_parameters():
                    if 'conv' in k:
                        grads.append(v.grad.data.flatten().detach().cpu().pow(2).numpy()) 
                embs[idxs[i]] = np.array(grads).flatten()
        return torch.Tensor(embs)