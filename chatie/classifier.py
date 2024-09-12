import ast
import torch.utils
import torch.utils.data
import torch
from typing import Dict, List
from sklearn.decomposition import PCA
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import pickle

from chatie.net.net import CustomNet
from chatie.config import *
from model.embed import Embed

class Classifier:
    def __init__(self, embedding:Embed, templates:Dict, input_dim:int=NET_INPUT_DIM, batch_size:int=NET_BATCH_SIZE, learning_rate:int=NET_LR):
        # attributes for params
        self.params = []
        self.p2l = {}  # reflection from param's name to label
        self.l2p = []  # reflection from label to param's name
        # hyper-parameters
        self.num_classes = len(self.params)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # net utilities
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.net = CustomNet(input_dim, self.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        # models
        self.templates = templates
        self.embed = embedding
        self.pca = PCA(n_components=input_dim, copy=True, whiten=True)
    
    # set params
    def set_params(self, params:List[tuple]):
        self.params = params

    # train PCA
    def pca_train(self):
        phrases = [name for name, symbol in self.params]
        embeddings = self.embed.embedding(phrases)
        self.pca.fit(embeddings)
    
    # generate labels
    def label_train(self):
        for i, (name, symbol) in enumerate(self.params):
            label = [0]*len(self.params)
            label[i] = 1
            self.p2l[name] = label
            self.l2p.append(name)
    
    # relation between label&param's name
    def phrase2label(self, phrase:str):
        return self.p2l[phrase]
    def label2phrase(self, label:int):
        return self.l2p[label]
    


    # train net
    def net_train(self, datas:List[str], labels:List[str]):
        # init
        datas = self.pca.transform(self.embed.embedding(datas))
        labels = [self.phrase2label(label) for label in labels]
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        self.net.train()       
        self.net.zero_grad()
        optimizer.zero_grad()
        datas = torch.tensor(datas).float()
        labels = torch.tensor(labels).float()
        # dataset
        dataset = torch.utils.data.TensorDataset(datas, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        # 这是一个内插问题，所以迭代轮数可以很高，“过拟合”是一种好的现象
        # train
        losses = []
        for epoch in range(NET_EPOCH):
            epoch_loss = []
            print(f'epoch:{epoch}')
            for batch_data, batch_label in dataloader:
                # forward
                batch_data, batch_label = batch_data.to(self.device), batch_label.to(self.device)
                outputs = self.net(batch_data)
                loss = criterion(outputs, batch_label)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print
                epoch_loss.append(loss.item())
            losses.extend(epoch_loss)
        return losses
    

    # use net
    def net_predict(self, data:str)->str:
        data = self.pca.transform(self.embed.embedding(data))
        data = torch.tensor(data).float().to(self.device)
        output = self.net(data)
        print(output)
        _, predict = torch.max(output, 1)
        predict = int(predict[0])
        phrase = self.label2phrase(predict)
        return phrase


    # load&save net
    def net_load(self):
        self.net.load_state_dict(torch.load(NET_MODEL_WEIGHT_PATH, weights_only=True))
    def net_save(self):
        torch.save(self.net.state_dict(), NET_MODEL_WEIGHT_PATH)