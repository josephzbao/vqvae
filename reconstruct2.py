import numpy as np
import torch
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
import pickle
import csv
from utils import reconstruct

modelfilepath = "/Users/jbao/vqvae/saved_models//model.pkl"
embeddingfilepath = "/Users/jbao/vqvae/saved_models//embedding.pkl"
tokensfilepath = "/Users/jbao/vqvae/saved_models//real_data.txt"
frequenciesfilepath = "/Users/jbao/vqvae/saved_models//frequencies.pkl"

def readTokens(path):
    with open(path, mode='r') as csvfile:
        tokens_reader = csv.reader(csvfile, delimiter=' ')
        all_tokens = []
        for row in tokens_reader:
            all_tokens.append(list(map(lambda x: int(x), row)))
        return all_tokens

with open(modelfilepath, 'rb') as file:
    model = torch.load(modelfilepath)
    model.eval()
    file.close()

with open(embeddingfilepath, 'rb') as file:
    embedding_weights = pickle.load(file)
    file.close()

with open(frequenciesfilepath, 'rb') as file:
    frequencies = pickle.load(file)
    file.close()

path = "/Users/jbao/Downloads/IoTFlowGenerator-main/devices/EdimaxPlug1101W"

all_tokens = readTokens(tokensfilepath)
n_embeddings = 2000

all_features = []

for tokens in all_tokens:
    one_hots = []
    for token in tokens:
        one_hot = [0.0] * n_embeddings
        one_hot[token] = 1.0
        one_hots.append(one_hot)
    z_q = torch.matmul(torch.tensor(one_hots).clone().detach(), torch.tensor(embedding_weights).clone().detach())
    reconstructed = model.decoder(z_q.view([1, 20, 128]))
    all_features.append(reconstruct(path, reconstructed[0], frequencies))

all_real_features = utils.convert_path(path)

print("generated features")
print(all_features[1])

print("real_features")
print(all_real_features[1])

with open("saved_models/EdimaxPlug1101W/fake_device_features.pkl", mode='wb') as pklfile:
    pickle.dump(all_features, pklfile)
    pklfile.close()

with open("saved_models/EdimaxPlug1101W/real_device_features.pkl", mode='wb') as pklfile:
    pickle.dump(all_real_features, pklfile)
    pklfile.close()

