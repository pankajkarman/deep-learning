import numpy as np
import pandas as pd
import torch, copy, random
import torch.nn as nn, heapq
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from model import ADDSTCN
import networkx as nx
import matplotlib.pyplot as plt

class CausalDiscovery(object):
    def __init__(self, file, target, seed=1111, significance=0.8, Network=ADDSTCN):
        self.file   = file
        self.target = target
        self.seed   = seed
        self.significance = significance
        self.Network = Network
    
    def discover(self, cuda=False, epochs=1000, kernel_size=4, layers=1, log_interval=100, lr=0.01, optimizername='Adam', dilation_c=4):
        self.cuda = cuda
        print("\n", "Target under consideration: ", self.target)
        torch.manual_seed(self.seed)
        X_train, Y_train = self.prepare()
        X_train = X_train.unsqueeze(0).contiguous()
        Y_train = Y_train.unsqueeze(2).contiguous()
        input_channels = X_train.size()[1]
        
        data    = pd.read_csv(self.file)
        columns = data.columns
        self.possible_causes = columns
        targetidx = columns.get_loc(self.target)
        model = self.Network(targetidx, input_channels, layers, kernel_size=kernel_size, cuda=cuda, dilation_c=dilation_c)
        self.model = model
        if cuda:
            model.cuda()
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()
        optimizer = getattr(optim, optimizername)(model.parameters(), lr=lr)  
        scores, firstloss = self.train(1, X_train, Y_train, model, optimizer, log_interval, epochs)
        self.firstloss = firstloss.cpu().data.item()
        for epoch in range(2, epochs+1):
            scores, realloss = self.train(epoch, X_train, Y_train, model, optimizer,log_interval,epochs)
        self.realloss = realloss.cpu().data.item()
        self.score    = scores.view(-1).cpu().detach().numpy()
        potential_ids = self.get_potential_causes(self.score)
        validated_ids = self.validate_causes(potential_ids, X_train, Y_train)
        delays        = self.get_delays(targetidx, validated_ids, layers, dilation_c)
        potentials    = columns[potential_ids].tolist()
        validated     = columns[validated_ids].tolist() 
        return potentials, validated, delays
        
    def get_potential_causes(self, sc):
        s = sorted(sc, reverse=True)
        indices = np.argsort(-1*sc)
        
        if len(s)<=5:
            potentials = []
            for i in indices:
                if sc[i]>1.0:
                    potentials.append(i)
        else:
            potentials = []
            gaps = []
            for i in range(len(s)-1):
                if s[i]<1.0: 
                    break
                gap = s[i]-s[i+1]
                gaps.append(gap)
            sortgaps = sorted(gaps, reverse=True)
            for i in range(0, len(gaps)):
                largestgap = sortgaps[i]
                index = gaps.index(largestgap)
                ind = -1
                if index<((len(s)-1)/2): 
                    if index>0:
                        ind=index
                        break
            if ind<0:
                ind = 0
            potentials = indices[:ind+1].tolist()
        return potentials 
    
    def validate_causes(self, potentials, X_train, Y_train):
        validated = copy.deepcopy(potentials)
        for idx in potentials:
            random.seed(self.seed)
            X_test2 = X_train.clone().cpu().numpy()
            random.shuffle(X_test2[:,idx,:][0])
            shuffled = torch.from_numpy(X_test2)
            if self.cuda:
                shuffled = shuffled.cuda()
            self.model.eval()
            output = self.model(shuffled)
            testloss = F.mse_loss(output, Y_train)
            self.testloss = testloss.cpu().data.item()

            diff = self.firstloss - self.realloss
            testdiff = self.firstloss - self.testloss

            if testdiff > (diff*self.significance): 
                validated.remove(idx) 
        return validated
    
    def get_delays(self, targetidx, validated, layers, dilation_c):
        weights = []
        for layer in range(layers):
            wts = self.model.dwn.network[layer].net[0].weight
            weight = wts.abs().view(wts.size()[0], wts.size()[2])
            weights.append(weight)
            
        causeswithdelay = dict()    
        for v in validated: 
            totaldelay=0    
            for k in range(len(weights)):
                w = weights[k]
                row = w[v]
                m1, m2 = heapq.nlargest(2, row)
                if m1 > m2:
                    index_max = len(row) - 1 - max(range(len(row)), key=row.__getitem__)
                else:
                    index_max = 0
                delay = index_max *(dilation_c**k)
                totaldelay += delay
            target, cause = self.possible_causes[[targetidx, v]]
            if targetidx != v:
                causeswithdelay[(target, cause)] = totaldelay
            else:
                causeswithdelay[(target, cause)] = totaldelay+1
        return causeswithdelay
        
    def prepare(self):
        target  = self.target
        df_data = pd.read_csv(self.file)
        df_y = df_data.copy(deep=True)[[target]]
        df_x = df_data.copy(deep=True)
        df_yshift = df_y.copy(deep=True).shift(periods=1, axis=0)
        df_yshift[target] = df_yshift[target].fillna(0.)
        df_x[target] = df_yshift
        data_x = df_x.values.astype('float32').transpose()    
        data_y = df_y.values.astype('float32').transpose()
        data_x = torch.from_numpy(data_x)
        data_y = torch.from_numpy(data_y)
        x, y = Variable(data_x), Variable(data_y)
        return x, y    
    
    def train(self, epoch, traindata, traintarget, model, optimizer, log_interval, epochs):
        x, y = traindata, traintarget
        model.train()
        optimizer.zero_grad()
        output = model(x)
        attentionscores = model.fs_attention.data
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        epochpercentage = (epoch/float(epochs))*100
        if epoch % log_interval ==0 or epoch % epochs == 0 or epoch==1:
            print('Epoch: {:2d} [{:.0f}%] \tLoss: {:.6f}'.format(epoch, epochpercentage, loss))
        return attentionscores, loss
        
class TCDF(object):
    def __init__(self, file, cuda=False):
        self.file = file
        self.data = pd.read_csv(file)
        self.cuda = cuda
        
    def run(self, epochs=1000, kernel_size=4, layers = 1, \
            log_interval=500, lr=0.01, optimizer='Adam'):
        self.causes = dict()
        self.delays = dict()
        self.losses = dict()
        self.attention_scores = dict()
        for c in self.data.columns:
            causal = CausalDiscovery(self.file, c)
            potential, cause, delay = causal.discover(epochs=epochs, \
                                kernel_size=kernel_size, layers = layers, \
                                log_interval=log_interval, lr=lr, optimizername=optimizer, \
                                cuda=self.cuda)
            print('Potential Causes: ', potential)
            print('Validated Causes: ', cause)
            print('Delays:', delay)
            self.causes[c] = cause            
            self.delays.update(delay)
            self.losses[c] = causal.realloss
            self.attention_scores[c] = causal.score
        return self.causes, self.delays
    
    def causal_graph(self, ax=None):
        G = nx.DiGraph()
        for column in self.data.columns:
            G.add_node(column)
        for nodepair in self.delays:
            G.add_edges_from([nodepair], weight = self.delays[nodepair])
        edge_labels = dict([((u, v), d['weight']) for u, v, d in G.edges(data=True)])
        pos = nx.circular_layout(G)
        nx.draw_networkx_edge_labels(G,pos, edge_labels = edge_labels)
        nx.draw(G, pos, node_color = 'white', edge_color='black', \
                node_size=1000, with_labels = True)
        if ax:
            ax.collections[0].set_edgecolor("#000000") 
        else:
            ax = plt.gca()
            ax.collections[0].set_edgecolor("#000000") 
            plt.show()
