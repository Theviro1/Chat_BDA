import pickle
import torch
from typing import List
from sklearn.decomposition import PCA
import torch.utils
import torch.utils.data

from chatie.config import *


class CustomNet(torch.nn.Module):
    def __init__(self, input_dim:int, num_classes:int, hidden_state:int=NET_HIDDEN_STATE):
        super(CustomNet, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim, hidden_state)
        self.layer2 = torch.nn.Linear(hidden_state, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = torch.nn.functional.sigmoid(x)
        x = self.layer2(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


class Net:
    def __init__(self, num_classes:int, input_dim:int=NET_INPUT_DIM, batch_size:int=NET_BATCH_SIZE, learning_rate:int=NET_LR):
        # hyper-parameters
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # net utilities
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.net = CustomNet(input_dim, self.num_classes).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()
        # models
        self.pca = PCA(n_components=input_dim, copy=True, whiten=True)
    

    # train PCA
    def pca_train(self, datas:List[List[float]]):
        self.pca.fit(datas)
    

    # train net
    def net_train(self, datas:List[List[float]], labels:List[List[float]])->tuple:
        datas = self.pca.transform(datas)
        # init
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        self.net.train()       
        self.net.zero_grad()
        optimizer.zero_grad()
        # batch_training
        datas = torch.tensor(datas).float()
        labels = torch.tensor(labels).float()
        dataset = torch.utils.data.TensorDataset(datas, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        # 这是一个内插问题，所以迭代轮数可以很高，“过拟合”是一种好的现象
        # train
        losses = []
        iters = []
        iter = 0
        for epoch in range(NET_EPOCH):
            epoch_loss = []
            print(f'training epoch {epoch}')
            for batch_data, batch_label in dataloader:
                # forward
                batch_data, batch_label = batch_data.to(self.device), batch_label.to(self.device)
                outputs = self.net(batch_data)
                loss = criterion(outputs, batch_label)
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # update
                iter += 1
                iters.append(iter)
                epoch_loss.append(loss.item())
            losses.extend(epoch_loss)
        return iters, losses
    

    # use net
    def net_predict(self, data:List[float])->str:
        data = self.pca.transform(data)
        data = torch.tensor(data).float().to(self.device)
        output = self.net(data)
        _, predict = torch.max(output, 1)
        predict = int(predict[0])
        return predict


    # load&save net
    def net_load(self):
        with open(NET_PCA_PATH, 'rb') as f:
            self.pca = pickle.load(f)
        self.net.load_state_dict(torch.load(NET_MODEL_WEIGHT_PATH, weights_only=True))

    def net_save(self):
        with open(NET_PCA_PATH, 'wb') as f:
            pickle.dump(self.pca, f)
        torch.save(self.net.state_dict(), NET_MODEL_WEIGHT_PATH)