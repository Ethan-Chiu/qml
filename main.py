import argparse
import time
import os
import random
from pathlib import Path

import torch
from tqdm import tqdm

from vggish import VGGish
from sentence_transformers import SentenceTransformer
from dataloader import get_clotho_loader
from utils.logger_util import get_logger

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer


# import jax
# from jax import numpy as jnp
# import jaxopt
# jax.config.update("jax_enable_x64", True)

dev = qml.device("default.qubit")

def entangle_all(N):
    for i in range(N):
        qml.CNOT(wires=[i, (i+1)%N])


def layer(N, layer_weights):
    for wire in range(N):
        qml.Rot(*layer_weights[wire], wires=wire)


def encoder(N, en_w, in_w, x):
    x = x.reshape(-1, N)
    wires = list(range(N))
    
    # Encode circuit
    for l, bx in enumerate(x):
        qml.AngleEmbedding(bx, wires=wires)
        layer(N, en_w[l])
        entangle_all(N)

    # Middle layers
    for layer_weights in in_w:
        layer(N, layer_weights)
        entangle_all(N)


def decoder(N, de_w, x):
    x = x.reshape(-1, N)
    wires = list(range(N))
    # circuit
    for l, bx in enumerate(x):
        qml.AngleEmbedding(bx, wires=wires)
        layer(N, de_w[l])
        entangle_all(N)


@qml.qnode(dev)
def circuit(N, en_w, in_w, de_w, x, y):
    wires = list(range(N))

    encoder(N, en_w, in_w, x)
    decoder(N, de_w, y)

    return qml.probs(wires=wires)

def log_prob_loss(prediction):
    # cross entropy when traget is [1, 0, ...]
    return -np.log(prediction[0])

# @jax.jit
def cost(en_w, in_w, de_w, X, Y, N):
    predictions = [circuit(N, en_w, in_w, de_w, x, y) for x, y in zip(X, Y)]
    loss = qml.math.stack([log_prob_loss(pred) for pred in predictions])
    loss = np.mean(loss)
    print(loss)
    return loss



# NOTE: validation
@qml.qnode(dev)
def decode_state(N, de_w, state, y):
    wires = list(range(N))
    
    qml.QubitStateVector(state, wires=wires)
    decoder(N, de_w, y)

    wires = list(range(N))
    return qml.probs(wires=wires)


def order_of_element(numbers, i):
    if i < 0 or i >= len(numbers):
        return "Index out of range"
    
    sorted_numbers = sorted(numbers, reverse=True)
    element = numbers[i]
    order = sorted_numbers.index(element) + 1
    
    return order


@qml.qnode(dev)
def encoder_state(N, en_w, in_w, x):
    encoder(N, en_w, in_w, x)
    return qml.state()


def retrieval(weights, X, Y, N):
    en_w, in_w, de_w = weights
    print("Retrieval test")
    mean_order = []
    for i, x in enumerate(tqdm(X)):
        f = encoder_state(N, en_w, in_w, x)
        sims = []
        for y in Y:
            sim = decode_state(N, de_w, f, y)[0]
            sims.append(sim)
        order = order_of_element(sims, i)
        mean_order.append(order)
    mean_order = sum(mean_order) / len(mean_order)
    return mean_order



def main(args):
    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # logger
    log_name = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    dir_name = args.exp_name
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(os.path.join(args.log_dir, dir_name)):
        os.mkdir(os.path.join(args.log_dir, dir_name))
    log_file = os.path.join(args.log_dir, dir_name, f'{log_name}.log')
    logger = get_logger(__name__, log_file)

    # config
    # cfg = Config.fromfile(args.cfg)
    # logger.info(cfg.pretty_text)
    # checkpoint_dir = os.path.join(args.checkpoint_dir, dir_name)

    # model
    audio_encoder = VGGish(
        freeze_audio_extractor=True,
        pretrained_vggish_model_path='/mnt/d/ethanfolder/qml/pretrained/vggish.pth',
        preprocess_audio_to_log_mel=True,
        postprocess_log_mel_with_pca=False,
        pretrained_pca_params_path=None
    )
    audio_encoder = audio_encoder.cuda()

    text_encoder = SentenceTransformer("all-MiniLM-L12-v2")

    # dataset
    batch_size = 8
    train_dataloader, val_dataloader = get_clotho_loader(
        Path("/mnt/d/clotho/dataset"), 
        "evaluation", 
        Path("/mnt/d/clotho/csv/clotho_test.csv"), 
        batch_size, 
        shuffle=False,
        nb_t_steps_pad='max'
    )

    opt = NesterovMomentumOptimizer(0.5)

    num_qubits = 8
    encoder_layer = 16
    inter_layers = 16
    decoder_layer = 48

    en_w_init = 0.01 * np.random.randn(encoder_layer, num_qubits, 3, requires_grad=True)
    in_w_init = 0.01 * np.random.randn(inter_layers, num_qubits, 3, requires_grad=True)
    de_w_init = 0.01 * np.random.randn(decoder_layer, num_qubits, 3, requires_grad=True)


    def process_data(batch_data):
        audio_names, text, text_ids = batch_data
        audio_names = list(audio_names)
        
        feature_dir = "/mnt/d/ethanfolder/qml/features/"
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        audio_feature_dir = os.path.join(feature_dir, "audio")
        text_feature_dir = os.path.join(feature_dir, "text")
        if not os.path.exists(audio_feature_dir):
            os.makedirs(audio_feature_dir)
        if not os.path.exists(text_feature_dir):
            os.makedirs(text_feature_dir)

        # NOTE: audio features
        all_features_exist = True
        for audio_file in audio_names:
            audio_feature_path = os.path.join(audio_feature_dir, f"AF_{os.path.basename(audio_file)}.npy")
            if not os.path.exists(audio_feature_path):
                all_features_exist = False
                break

        if not all_features_exist:
            # Calculate and save audio features if any are missing
            audio_feature = audio_encoder(audio_names)
            audio_feature = audio_feature.view(batch_size, -1, audio_feature.shape[1])
            audio_feature = audio_feature.mean(dim=1)
            audio_feature = audio_feature.detach().cpu().numpy()
            
            for idx, audio_file in enumerate(audio_names):
                audio_feature_path = os.path.join(audio_feature_dir, f"AF_{os.path.basename(audio_file)}.npy")
                with open(audio_feature_path, 'wb') as af_file:
                    np.save(af_file, audio_feature[idx])
        else:
            # Load existing audio features
            audio_feats = []
            for audio_file in audio_names:
                audio_feature_path = os.path.join(audio_feature_dir, f"AF_{os.path.basename(audio_file)}.npy")
                with open(audio_feature_path, 'rb') as af_file:
                    audio_feats.append(np.load(af_file))
            print(f"Loaded audio features from files")
            audio_feature = np.array(audio_feats)


        # NOTE: text features
        all_text_features_exist = True
        for t_id in text_ids:
            text_feature_path = os.path.join(text_feature_dir, f"TF_{t_id}.npy")
            if not os.path.exists(text_feature_path):
                all_text_features_exist = False
                break

        if not all_text_features_exist:
            text_feature = text_encoder.encode(text)
            
            for idx, t_id in enumerate(text_ids):
                t_f_path = os.path.join(text_feature_dir, f"TF_{t_id}.npy")
                with open(t_f_path, 'wb') as tf_file:
                    np.save(tf_file, text_feature[idx])
        else:
            t_feats = []
            for t_id in text_ids:
                t_f_path = os.path.join(text_feature_dir, f"TF_{t_id}.npy")
                with open(t_f_path, 'rb') as tf_file:
                    t_feats.append(np.load(tf_file))
            print(f"Loaded text features from files")
            text_feature = np.array(t_feats)
            

        return audio_feature, text_feature


    def validate(w):
        total_a = []
        total_t = []
        for n_iter, batch_data in enumerate(tqdm(val_dataloader)):
            audio_feature, text_feature = process_data(batch_data)
            total_a.append(torch.tensor(audio_feature))
            total_t.append(torch.tensor(text_feature))
        total_a = torch.cat(total_a, dim=0)
        total_t = torch.cat(total_t, dim=0)
        rank = retrieval(w, X=total_a, Y=total_t, N=num_qubits)
        print("Mean rank is:", rank)
        return rank
            


    # Train
    best_epoch = 0
    global_step = 0
    en_w, in_w, de_w = en_w_init, in_w_init, de_w_init
    weights = (en_w, in_w, de_w)

    for epoch in range(100):
        for n_iter, batch_data in enumerate(tqdm(train_dataloader)):
            audio_feature, text_feature = process_data(batch_data)

            # print("audio shape", audio_feature.shape) # 384
            # print("text shape", text_feature.shape)

            weights = opt.step(cost, *weights, X=audio_feature, Y=text_feature, N=num_qubits)

            global_step += 1
            # logger.info(f"Epoch: {epoch}, Iter: {n_iter}, ")
            
        if (epoch + 1) % 15 == 0:
            rank = validate(weights)
            logger.info(f"Retrieval rank: {rank}")
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str, help='config file path')
    parser.add_argument('--log_dir', type=str,
                        default='exp_dir', help='log dir')
    # parser.add_argument('--checkpoint_dir', type=str,
    #                     default='work_dir', help='dir to save checkpoints')

    args = parser.parse_args()
    main(args)
