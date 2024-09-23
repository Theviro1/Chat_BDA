import ast
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import pickle
from typing import Dict, List
import matplotlib.pyplot as plt

from chatie.net.net import Net
from chatie.config import *
from model.embed import Embed
from model.model import CustomLLM
from utils.logs import ChatIELogger

class Classifier:
    def __init__(self, params:List[tuple], llm:CustomLLM, embed:Embed, templates:Dict):
        # models
        self.net = Net(num_classes=len(params))
        self.llm = llm
        self.embed = embed
        self.templates = templates
        self.params = params
        self.logger = ChatIELogger()
        # record for labels
        self.phrase2label = {}  # reflection from param's name to label
        self.label2phrase = []  # reflection from label to param's name
        # init
        self.init()
    
    # all the initialize process
    def init(self):
        try:
            self.generate_labels()
            self.load_net()
        except:
            self.train_net()
    
    # print loss
    def print_loss(self, x:List[int], y:List[float]):
        plt.title('classifier nerual-network loss trend')
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.plot(x, y)
        plt.savefig(LOSS_FIGURE_PATH)
        

    # generate labels
    def generate_labels(self):
        for i, (name, _, _) in enumerate(self.params):
            label = [0]*len(self.params)
            label[i] = 1
            self.phrase2label[name] = label
            self.label2phrase.append(name)
        
     # auto generate training data
    def generate_data(self, save_path:str=NET_TRAINING_DATA_PATH)->List[tuple]:
        # use LLM to generate structured training data
        data_list=[]
        for name, _, _ in self.params:
            prompt = PromptTemplate.from_template(self.templates['data_generation_prompt'])
            chain = LLMChain(llm=self.llm, prompt=prompt)
            r = chain(inputs={'input_text':name})['text']
            self.logger.info(f"param {name}'s generation is:{r}")
            # format should be a python list, because it's an easy task so we can fully trust LLM
            while True:  # in case LLM return in other formations, try using a loop until the result can be 'literal_eval'
                try:
                    results = ast.literal_eval(r)  
                    break   # if success, break
                except:
                    r = chain(inputs={'input_text':name})['text']  # if not, continue generating
            data_list.extend([(result, name) for result in results])  # 'result' is a replaced synonyms
        with open(save_path, 'wb') as f:
            pickle.dump(data_list, f)
        return data_list
    
    # train the classifier net, return losses
    def train_net(self):
        # generate data using LLM
        generate_data = self.generate_data()
        # preprocessing data
        with open(NET_TRAINING_DATA_PATH, 'rb') as f:
            arr = pickle.load(f)
        datas = [name_replaced for name_replaced, name in arr]
        labels = [name for name_replaced, name in arr]
        datas = self.embed.embedding(datas)
        labels = [self.phrase2label[label] for label in labels]
        # train net
        self.net.pca_train(datas)
        iters, losses = self.net.net_train(datas, labels)
        # save net
        self.net.net_save()
        self.print_loss(iters, losses)
    
    # load&init classifier net
    def load_net(self):
        self.net.net_load()

    # classify
    def classify(self, phrase:str)->str:
        data = self.embed.embedding(phrase)
        label = self.net.net_predict(data)
        fixed_phrase = self.label2phrase[label]
        return fixed_phrase