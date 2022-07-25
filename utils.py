import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np
import random
import pickle
from sklearn.cluster import DBSCAN, KMeans

def data_loaders(train_data, val_data, batch_size):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader

def mapAllElements(all_tokens, mapping):
    to_return = []
    for tokens in all_tokens:
        to_return.append(list(map(lambda x: mapping[x], tokens)))
    return to_return

def mapAllProtocolConfElements(all_tokens, mapping):
    to_return = []
    for tokens in all_tokens:
        to_return.append(list(map(lambda x: mapping[x], tokens)))
    return to_return

def mapDurationTokens(all_tokens, mapping):
    all_to_return = []
    for tokens in all_tokens:
        to_return = []
        for token in tokens:
            duration_set = mapping[token]
            to_return.append(random.choice(duration_set))
        all_to_return.append(to_return)
    return all_to_return

def one_hot(a, num_classes):
    all_seqs = []
    for x in a:
        seqs = []
        for j in x:
            empty = [0] * num_classes
            empty[j] = 1
            seqs.append(empty)
        all_seqs.append(seqs)
    return all_seqs

def extractPacketFrameLengths(all_packet_frame_lengths):
    all_abs_lengths = []
    for packet_frame_lengths in all_packet_frame_lengths:
        abs_lengths = []
        for packet_frame_length in packet_frame_lengths:
            abs_lengths.append([abs(packet_frame_length)])
        all_abs_lengths.append(abs_lengths)
    return all_abs_lengths

def extractPayloads(all_packet_frame_lengths):
    all_abs_lengths = []
    for packet_frame_lengths in all_packet_frame_lengths:
        abs_lengths = []
        for packet_frame_length in packet_frame_lengths:
            if packet_frame_length:
                abs_lengths.append([1])
            else:
                abs_lengths.append([0])
        all_abs_lengths.append(abs_lengths)
    return all_abs_lengths

def extractPacketFrameDirections(all_packet_frame_lengths):
    all_abs_lengths = []
    for packet_frame_lengths in all_packet_frame_lengths:
        abs_lengths = []
        for packet_frame_length in packet_frame_lengths:
            if packet_frame_length < 0:
                abs_lengths.append([0])
            else:
                abs_lengths.append([1])
        all_abs_lengths.append(abs_lengths)
    return all_abs_lengths

def normalize_packet(all_packet_frame_lengths):
    maxLength = float(np.amax(all_packet_frame_lengths))
    all_abs_lengths = []
    for packet_frame_lengths in all_packet_frame_lengths:
        abs_lengths = []
        for packet_frame_length in packet_frame_lengths:
            abs_lengths.append([float(packet_frame_length[0])/maxLength])
        all_abs_lengths.append(abs_lengths)
    return all_abs_lengths

def convert_tensor(tensors):
    items = []
    for tensor in tensors:
        items.append(tensor.item())
    return items

def get_closest(myList, myNumber):
    return min(myList, key=lambda x: abs(x - myNumber))

def get_token(token_to_frames, x):
    for key, value in token_to_frames.items():
        if x in value:
            return key

def convert_frame_lengths(all_packet_frame_lengths, all_packet_sigs):
    all_extra_frame_lengths = []
    token_to_frames = dict()
    for i in range(len(all_packet_frame_lengths)):
        packet_frame_lengths = all_packet_frame_lengths[i]
        packet_sigs = all_packet_sigs[i]
        for j in range(len(packet_frame_lengths)):
            packet_frame_length = packet_frame_lengths[j][0]
            packet_sig = packet_sigs[j]
            all_extra_frame_lengths.append([packet_frame_length])
    clustering = DBSCAN(eps=2, min_samples=2).fit(np.array(all_extra_frame_lengths))
    leftovers = []
    for i in range(len(clustering.labels_)):
        packet_frame_length = all_extra_frame_lengths[i][0]
        if clustering.labels_[i] == -1:
            leftovers.append([packet_frame_length])
        else:
            token = clustering.labels_[i]
            if token not in token_to_frames.keys():
                token_to_frames[token] = [packet_frame_length]
            else:
                if packet_frame_length not in token_to_frames[token]:
                    frames = token_to_frames[token]
                    frames.append(packet_frame_length)
                    token_to_frames[token] = frames
    max_token = max(token_to_frames.keys())
    if len(leftovers) != 0:
        kmeans = KMeans(n_clusters=len(leftovers), random_state=0).fit(np.array(leftovers))
        for i in range(len(kmeans.labels_)):
            packet_frame_length = leftovers[i][0]
            token = max_token + kmeans.labels_[i] + 1
            if token not in token_to_frames.keys():
                token_to_frames[token] = [packet_frame_length]
            else:
                if packet_frame_length not in token_to_frames[token]:
                    frames = token_to_frames[token]
                    frames.append(packet_frame_length)
                    token_to_frames[token] = frames
    all_new_packet_sigs = []
    for i in range(len(all_packet_sigs)):
        packet_sigs = all_packet_sigs[i]
        new_packet_sigs = []
        packet_frame_lengths = all_packet_frame_lengths[i]
        for j in range(len(packet_sigs)):
            packet_sig = packet_sigs[j]
            packet_frame_length = packet_frame_lengths[j][0]
            new_packet_sigs.append(get_token(token_to_frames, packet_frame_length))
        all_new_packet_sigs.append(new_packet_sigs)
    return all_new_packet_sigs, token_to_frames

def convert_frame_lengths1(all_packet_frame_lengths, all_packet_sigs):
    all_extra_frame_lengths = []
    frequencies = dict()
    token_to_frames = dict()
    for i in range(len(all_packet_frame_lengths)):
        packet_frame_lengths = all_packet_frame_lengths[i]
        packet_sigs = all_packet_sigs[i]
        for j in range(len(packet_frame_lengths)):
            packet_frame_length = packet_frame_lengths[j][0]
            packet_sig = packet_sigs[j]
            if packet_sig == -1:
                all_extra_frame_lengths.append([packet_frame_length])
            else:
                if packet_sig not in token_to_frames.keys():
                    token_to_frames[packet_sig] = [packet_frame_length]
                    frequencies[packet_sig] = {packet_frame_length: 1}
                else:
                    if packet_frame_length not in token_to_frames[packet_sig]:
                        frames = token_to_frames[packet_sig]
                        frames.append(packet_frame_length)
                        token_to_frames[packet_sig] = frames
                        ps_freqs = frequencies[packet_sig]
                        ps_freqs[packet_frame_length] = 1
                        frequencies[packet_sig] = ps_freqs
                    else:
                        ps_freqs = frequencies[packet_sig]
                        ps_freqs[packet_frame_length] = ps_freqs[packet_frame_length] + 1
                        frequencies[packet_sig] = ps_freqs
    if len(token_to_frames.keys()) == 0:
        max_token = -1
    else:
        max_token = max(token_to_frames.keys())
    clustering = DBSCAN(eps=2, min_samples=2).fit(np.array(all_extra_frame_lengths))
    leftovers = []
    for i in range(len(clustering.labels_)):
        packet_frame_length = all_extra_frame_lengths[i][0]
        if clustering.labels_[i] == -1:
            leftovers.append([packet_frame_length])
        else:
            token = max_token + clustering.labels_[i] + 1
            if token not in token_to_frames.keys():
                token_to_frames[token] = [packet_frame_length]
                frequencies[token] = {packet_frame_length: 1}
            else:
                if packet_frame_length not in token_to_frames[token]:
                    frames = token_to_frames[token]
                    frames.append(packet_frame_length)
                    token_to_frames[token] = frames
                    ps_freqs = frequencies[token]
                    ps_freqs[packet_frame_length] = 1
                    frequencies[token] = ps_freqs
                else:
                    ps_freqs = frequencies[token]
                    ps_freqs[packet_frame_length] = ps_freqs[packet_frame_length] + 1
                    frequencies[token] = ps_freqs
    if len(token_to_frames.keys()) == 0:
        max_token = -1
    else:
        max_token = max(token_to_frames.keys())
    if len(leftovers) != 0:
        kmeans = KMeans(n_clusters=len(leftovers), random_state=0).fit(np.array(leftovers))
        for i in range(len(kmeans.labels_)):
            packet_frame_length = leftovers[i][0]
            token = max_token + kmeans.labels_[i] + 1
            if token not in token_to_frames.keys():
                token_to_frames[token] = [packet_frame_length]
                frequencies[token] = {packet_frame_length: 1}
            else:
                if packet_frame_length not in token_to_frames[token]:
                    frames = token_to_frames[token]
                    frames.append(packet_frame_length)
                    token_to_frames[token] = frames
                    ps_freqs = frequencies[token]
                    ps_freqs[packet_frame_length] = 1
                    frequencies[token] = ps_freqs
                else:
                    ps_freqs = frequencies[token]
                    ps_freqs[packet_frame_length] = ps_freqs[packet_frame_length] + 1
                    frequencies[token] = ps_freqs
    all_new_packet_sigs = []
    for i in range(len(all_packet_sigs)):
        packet_sigs = all_packet_sigs[i]
        new_packet_sigs = []
        packet_frame_lengths = all_packet_frame_lengths[i]
        for j in range(len(packet_sigs)):
            packet_sig = packet_sigs[j]
            packet_frame_length = packet_frame_lengths[j][0]
            if packet_sig == -1:
                new_packet_sigs.append(get_token(token_to_frames, packet_frame_length))
            else:
                new_packet_sigs.append(packet_sig)
        all_new_packet_sigs.append(new_packet_sigs)
    return all_new_packet_sigs, token_to_frames, frequencies

def stringToSignature(item):
    item.replace(" ", "")
    arr = item.split(',')
    int_arr = [int(numeric_string) for numeric_string in arr]
    sig = []
    for i in range(0, len(int_arr), 2):
        sig.append((int_arr[i], int_arr[i + 1]))
    return sig

def compress_packet_ranges(token_to_packet_range):
    compressed = dict()
    for key, value in token_to_packet_range.items():
        if isinstance(value, str):
            range = stringToSignature(value)[0]
            if range[0] == range[1]:
                compressed[key] = range[0]
            else:
                compressed[key] = range
        else:
            compressed[key] = value
    return compressed

def chooseFromFrequencies(frequencies):
    items = []
    weights = []
    for key, value in frequencies.items():
        items.append(key)
        weights.append(value)
    return random.choices(items, weights=weights, k=1)[0]

def reconstruct(path, featureVectors, frequencies):
    with open(path + "/real_features.pkl", mode='rb') as pklfile, open(path + "/max_duration.pkl", mode='rb') as dur_file, open(path + "/token_to_frames.pkl", mode='rb') as tokenfile:
        features = pickle.load(pklfile)
        max_duration = features[15]
        pklfile.close()
        protocol_conf_length = len(features[8])
        src_ports_length = len(features[10])
        dst_ports_length = len(features[12])
        durations_length = len(features[6])
        all_generated_packets = []
        for featureVector in featureVectors:
            packet = []
            packet_frame_length_length = max(frequencies.keys()) + 1
            packet_frame_length = convert_tensor(featureVector[0:packet_frame_length_length])
            packet_frame_length = chooseFromFrequencies(frequencies[np.argmax(packet_frame_length)])
            packet_direction_index = packet_frame_length_length
            packet_direction = 1 if featureVector[packet_direction_index] > 0.5 else 0
            protocol_conf_index = packet_direction_index + 1
            protocol_confs = convert_tensor(featureVector[protocol_conf_index:protocol_conf_index+protocol_conf_length])
            protocol_confs = features[8][np.argmax(protocol_confs)]
            src_port_index = protocol_conf_index+protocol_conf_length
            src_ports = convert_tensor(featureVector[src_port_index:src_port_index + src_ports_length])
            src_ports = features[10][np.argmax(src_ports)]
            dst_port_index = src_port_index + src_ports_length
            dst_ports = convert_tensor(featureVector[dst_port_index:dst_port_index + dst_ports_length])
            dst_ports = features[12][np.argmax(dst_ports)]
            durations_index = dst_port_index + dst_ports_length
            duration = convert_tensor(featureVector[durations_index:durations_index + durations_length])
            duration = features[6][np.argmax(duration)]
            duration = random.choice(duration)
            payload = 1 if featureVector[durations_index + durations_length] > 0.5 else 0
            packet.append(True if payload == 1 else False)
            packet.append(abs(packet_frame_length))
            packet.append(True if packet_direction == 1 else False)
            packet.append(duration * max_duration)
            packet.append(protocol_confs)
            packet.append(src_ports)
            packet.append(dst_ports)
            all_generated_packets.append(packet)
        return all_generated_packets

def reconstruct1(path, featureVectors):
    with open(path + "/real_features.pkl", mode='rb') as pklfile, open(path + "/max_duration.pkl", mode='rb') as dur_file, open(path + "/token_to_frames.pkl", mode='rb') as tokenfile:
        token_to_frames = pickle.load(tokenfile)
        features = pickle.load(pklfile)
        max_duration = features[15]
        pklfile.close()
        protocol_conf_length = len(features[8])
        src_ports_length = len(features[10])
        dst_ports_length = len(features[12])
        durations_length = len(features[6])
        all_generated_packets = []
        for featureVector in featureVectors:
            packet = []
            packet_frame_length_length = max(token_to_frames.keys()) + 1
            packet_frame_length = convert_tensor(featureVector[0:packet_frame_length_length])
            packet_frame_lengths = token_to_frames[np.argmax(packet_frame_length)]
            packet_frame_length = random.choice(packet_frame_lengths)
            packet_direction_index = packet_frame_length_length
            packet_direction = 1 if featureVector[packet_direction_index] > 0.5 else 0
            protocol_conf_index = packet_direction_index + 1
            protocol_confs = convert_tensor(featureVector[protocol_conf_index:protocol_conf_index+protocol_conf_length])
            protocol_confs = features[8][np.argmax(protocol_confs)]
            src_port_index = protocol_conf_index+protocol_conf_length
            src_ports = convert_tensor(featureVector[src_port_index:src_port_index + src_ports_length])
            src_ports = features[10][np.argmax(src_ports)]
            dst_port_index = src_port_index + src_ports_length
            dst_ports = convert_tensor(featureVector[dst_port_index:dst_port_index + dst_ports_length])
            dst_ports = features[12][np.argmax(dst_ports)]
            durations_index = dst_port_index + dst_ports_length
            duration = convert_tensor(featureVector[durations_index:durations_index + durations_length])
            duration = features[6][np.argmax(duration)]
            duration = random.choice(duration)
            payload = 1 if featureVector[durations_index + durations_length] > 0.5 else 0
            packet.append(True if payload == 1 else False)
            packet.append(abs(packet_frame_length))
            packet.append(True if packet_direction == 1 else False)
            packet.append(duration * max_duration)
            packet.append(protocol_confs)
            packet.append(src_ports)
            packet.append(dst_ports)
            all_generated_packets.append(packet)
        return all_generated_packets

def convert_path(path):
    with open(path + "/real_features.pkl", mode='rb') as pklfile:
        features = pickle.load(pklfile)
        all_real_packet_frame_lengths = mapAllElements(features[3], features[4])
        all_real_protocol_confs = mapAllProtocolConfElements(features[7], features[8])
        all_real_src_ports = mapAllElements(features[9], features[10])
        all_real_dst_ports = mapAllElements(features[11], features[12])
        all_real_durations = mapDurationTokens(features[5], features[6])
        all_real_payloads = features[13]
        max_duration = features[15]
        all_real_traffic = []
        for i in range(len(all_real_packet_frame_lengths)):
            real_packet_frame_lengths = all_real_packet_frame_lengths[i]
            real_src_ports = all_real_src_ports[i]
            real_dst_ports = all_real_dst_ports[i]
            real_durations = all_real_durations[i]
            real_protocol_confs = all_real_protocol_confs[i]
            real_payloads = all_real_payloads[i]
            real_traffic = []
            for j in range(len(real_packet_frame_lengths)):
                real_packet = []
                real_packet.append(real_payloads[j])
                real_packet.append(abs(real_packet_frame_lengths[j]))
                real_packet.append(real_packet_frame_lengths[j] > 0)
                real_packet.append(real_durations[j] * max_duration)
                real_packet.append(real_protocol_confs[j])
                real_packet.append(real_src_ports[j])
                real_packet.append(real_dst_ports[j])
                real_traffic.append(real_packet)
            all_real_traffic.append(real_traffic)
        return all_real_traffic

def load_data(path, batch_size=32):
    with open(path + "/real_features.pkl", mode='rb') as pklfile:
        features = pickle.load(pklfile)
        pklfile.close()
        packet_frame_lengths = mapAllElements(features[3], features[4])
        packet_frame_directions = extractPacketFrameDirections(packet_frame_lengths)
        packet_frame_lengths, token_to_frames, frequencies = convert_frame_lengths1(extractPacketFrameLengths(packet_frame_lengths), features[16])
        packet_frame_lengths_length = max(frequencies.keys()) + 1
        packet_frame_lengths = one_hot(packet_frame_lengths, packet_frame_lengths_length)
        protocol_conf_length = len(features[8])
        protocol_confs = features[7]
        protocol_confs = one_hot(protocol_confs, protocol_conf_length)
        src_ports = features[9]
        src_ports_length = len(features[10])
        src_ports = one_hot(src_ports, src_ports_length)
        dst_ports = features[11]
        dst_ports_length = len(features[12])
        dst_ports = one_hot(dst_ports, dst_ports_length)
        durations = features[5]
        duration_length = len(features[6])
        durations = one_hot(durations, duration_length)
        payloads = extractPayloads(np.array(features[13]))
        featureVectors = np.concatenate((packet_frame_lengths, packet_frame_directions, protocol_confs, src_ports, dst_ports, durations, payloads), 2)
        tensor_x = torch.Tensor(featureVectors)  # transform to torch tensor
        my_dataset = TensorDataset(tensor_x)  # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size)  # create your dataloader
        with open(path + "/token_to_frames.pkl", 'wb') as file:
            pickle.dump(token_to_frames, file)
            file.close()
        return my_dataloader, featureVectors, tensor_x, frequencies

def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
               SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')
