import numpy as np
import torch
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
import pickle
import csv
from utils import reconstruct

def readTokens(path):
    with open(path, mode='r') as csvfile:
        tokens_reader = csv.reader(csvfile, delimiter=' ')
        all_tokens = []
        for row in tokens_reader:
            all_tokens.append(list(map(lambda x: int(x), row)))
        return all_tokens

duration_cluster_sizes = [25, 50, 100]
n_embeddings = [500, 2000, 5000]
embedding_dims = [128, 512]
devices = ["EdimaxPlug1101W", "WeMoSwitch", "MAXGateway", "HomeMaticPlug", "D-LinkCam", "EdimaxCam2", "EdnetCam1", "Aria", "D-LinkDayCam", "D-LinkDoorSensor", "D-LinkHomeHub", "D-LinkSensor", "D-LinkSiren", "D-LinkSwitch", "D-LinkWaterSensor", "EdimaxCam1", "EdimaxPlug2101W", "EdnetCam2", "EdnetGateway", "HueBridge", "HueSwitch", "Lightify", "TP-LinkPlugHS100", "TP-LinkPlugHS110", "WeMoInsightSwitch", "WeMoInsightSwitch2", "WeMoLink", "WeMoSwitch2", "Withings"]

for duration_cluster_size in duration_cluster_sizes:
    for n_embedding in n_embeddings:
        for embedding_dim in embedding_dims:
            for device_path in devices:
                modelfilepath = "saved_models/" + device_path + "/" + str(duration_cluster_size) + "/" + str(n_embedding) + "/" + str(embedding_dim) + "/model.pkl"
                embeddingfilepath = "saved_models/" + device_path + "/" + str(duration_cluster_size) + "/" + str(n_embedding) + "/" + str(embedding_dim) + "/embedding.pkl"
                frequenciesfilepath = "saved_models/" + device_path + "/" + str(duration_cluster_size) + "/" + str(n_embedding) + "/" + str(embedding_dim) + "/frequencies.pkl"
                tokensfilepath = "saved_models/" + device_path + "/" + str(duration_cluster_size) + "/" + str(n_embedding) + "/" + str(embedding_dim) + "/real_data.txt"

                with open(modelfilepath, 'rb+') as file:
                    model = torch.load(modelfilepath)
                    model.eval()
                    file.close()

                with open(embeddingfilepath, 'rb+') as file:
                    embedding_weights = pickle.load(file)
                    file.close()

                with open(frequenciesfilepath, 'rb+') as file:
                    frequencies = pickle.load(file)
                    file.close()

                path = "devices/" + device_path + "/" + str(duration_cluster_size)

                all_tokens = readTokens(tokensfilepath)

                all_features = []

                for tokens in all_tokens:
                    one_hots = []
                    for token in tokens:
                        one_hot = [0.0] * n_embedding
                        one_hot[token] = 1.0
                        one_hots.append(one_hot)
                    z_q = torch.matmul(torch.tensor(one_hots).clone().detach(),
                                       torch.tensor(embedding_weights).clone().detach())
                    reconstructed = model.decoder(z_q.view([1, 20, embedding_dim]))
                    all_features.append(reconstruct(path, reconstructed[0], frequencies))

                all_real_features = utils.convert_path(path)

                print("generated features")
                print(all_features[1])

                print("real_features")
                print(all_real_features[1])

                with open("saved_models/" + device_path + "/" + str(duration_cluster_size) + "/" + str(n_embedding) + "/" + str(embedding_dim) + "/fake_device_features.pkl", mode='wb+') as pklfile:
                    pickle.dump(all_features, pklfile)
                    pklfile.close()

                with open("saved_models/" + device_path + "/" + str(duration_cluster_size) + "/" + str(n_embedding) + "/" + str(embedding_dim) + "/real_device_features.pkl", mode='wb+') as pklfile:
                    pickle.dump(all_real_features, pklfile)
                    pklfile.close()

