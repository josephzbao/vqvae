
import torch
import torch.nn as nn
import numpy as np
from models.quantizer import VectorQuantizer
from models.transformer import TSTransformerEncoder


class VQVAE(nn.Module):
    def __init__(self, feat_dim, max_len,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = TSTransformerEncoder(feat_dim, embedding_dim, max_len)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = TSTransformerEncoder(embedding_dim, feat_dim, max_len)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):

        z_e = self.encoder(x)
        embedding_loss, z_q, perplexity, min_encodings, min_encoding_indices, embedding_weights = self.vector_quantization(
            z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, min_encodings, min_encoding_indices, embedding_weights


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((3, 20, 5))
    x = torch.tensor(x).float()

    # test transformer
    decoder = VQVAE(5, 20, 100, 10, 0.25)
    decoder_out = decoder(x)
    print('Dncoder out shape:', decoder_out[1].shape)
    print(decoder_out[4].shape)

    filepath = "/saved_models/D-LinkCam/saved.pkl"

    with open(filepath, 'w') as file:
      torch.save(decoder, filepath)

    with open(filepath, 'r') as file:
      # Then later:
      model = torch.load(filepath)
      encoder = model.encoder
      encoder_output = encoder(x)
      print(encoder_output)


