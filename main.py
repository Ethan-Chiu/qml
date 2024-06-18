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

from circuit import encoder_state, decode_state, circuit
from process_data import process_data

# import jax
# from jax import numpy as jnp
# import jaxopt
# jax.config.update("jax_enable_x64", True)


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


def order_of_element(numbers, i):
    if i < 0 or i >= len(numbers):
        return "Index out of range"
    
    sorted_numbers = sorted(numbers, reverse=True)
    element = numbers[i]
    order = sorted_numbers.index(element) + 1
    
    return order


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
    # logger = get_logger(__name__, log_file)
    logger = get_logger(__name__)

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

    def validate(w):
        total_a = []
        total_t = []
        for n_iter, batch_data in enumerate(tqdm(val_dataloader)):
            audio_feature, text_feature = process_data(batch_data, audio_encoder, text_encoder)
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
            audio_feature, text_feature = process_data(batch_data, audio_encoder, text_encoder)

            # print("audio shape", audio_feature.shape) # 128
            # print("text shape", text_feature.shape) # 384

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
