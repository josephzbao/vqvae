import numpy as np
import torch
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
import pickle
import csv

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=256)
parser.add_argument("--n_embeddings", type=int, default=2000)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset",  type=str, default='CIFAR10')

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

"""
Load data and define batch data loaders
"""

training_loader, featureVectors, tensor_x, frequencies = utils.load_data("/Users/jbao/Downloads/IoTFlowGenerator-main/devices/EdimaxPlug1101W")
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

print(frequencies)

model = VQVAE(np.asarray(featureVectors).shape[2], np.asarray(featureVectors).shape[1], args.n_embeddings, args.embedding_dim, args.beta).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}

def write_tokens(all_tokens, location):
    with open(location, mode='w') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerows(all_tokens)


def train():

    for i in range(args.n_updates):
        x = next(iter(training_loader))[0]
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity, min_encodings, min_encoding_indices, embedding_weights = model(x)
        recon_loss = torch.mean((x_hat - x)**2)
        loss = recon_loss + embedding_loss

        print(embedding_loss)
        print(recon_loss)
        print(perplexity)

        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % args.log_interval == 0:
            """
            save model and print values
            """
            if args.save:
                hyperparameters = args.__dict__
                utils.save_model_and_results(
                    model, results, hyperparameters, args.filename)

            print('Update #', i, 'Recon Error:',
                  np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))


    model.eval()
    embedding_loss, x_hat, perplexity, min_encodings, min_encoding_indices, embedding_weights = model(tensor_x)

    modelfilepath = "/Users/jbao/vqvae/saved_models//model.pkl"
    embeddingfilepath = "/Users/jbao/vqvae/saved_models//embedding.pkl"
    frequenciesfilepath = "/Users/jbao/vqvae/saved_models//frequencies.pkl"
    tokensfilepath = "/Users/jbao/vqvae/saved_models//real_data.txt"

    with open(modelfilepath, 'wb') as file:
        torch.save(model, file)
        file.close()

    with open(embeddingfilepath, 'wb') as file:
        pickle.dump(embedding_weights, file)
        file.close()

    with open(frequenciesfilepath, 'wb') as file:
        pickle.dump(frequencies, file)
        file.close()

    all_tokens = []
    for encoding in min_encoding_indices:
        all_tokens.append(int(encoding[0]))

    n = 20

    x = [all_tokens[i:i + n] for i in range(0, len(all_tokens), n)]

    write_tokens(x, tokensfilepath)

if __name__ == "__main__":
    train()
