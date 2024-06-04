import torch
import time
import torch.nn as nn
import os
import random
from pathlib import Path


from vggish import VGGish
from sentence_transformers import SentenceTransformer
from dataloader import get_clotho_loader

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer


dev = qml.device("default.qubit")

def entangle_all(N):
    for i in range(N):
        qml.CNOT(wires=[i, (i+1)%N])


def layer(N, layer_weights):
    for wire in range(N):
        qml.Rot(*layer_weights[wire], wires=wire)
    entangle_all(N)


def encode_data(N, x):
    x = x.reshape(-1, N)
    wires = list(range(N))
    for bx in x:
        qml.AngleEmbedding(bx, wires=wires)
        entangle_all(N)


@qml.qnode(dev)
def circuit(N, weights, x, y):
    encode_data(N, x)

    for layer_weights in weights:
        layer(N, layer_weights)

    encode_data(N, y)

    return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(N)])


def variational_circuit(N, weights, bias, x, y):
    return circuit(N, weights, x, y) + bias

def square_loss(prediction):
    return np.sum(prediction ** 2)

def cost(weights, bias, X, Y, N):
    predictions = [variational_circuit(N, weights, bias, x, y) for x, y in zip(X, Y)]
    loss = qml.math.stack([square_loss(pred) for pred in predictions])
    loss = np.mean(loss)
    print("loss", loss)
    return loss




def main():
    # Fix seed
    FixSeed = 123
    random.seed(FixSeed)
    np.random.seed(FixSeed)
    torch.manual_seed(FixSeed)
    torch.cuda.manual_seed(FixSeed)

    # logger
    # log_name = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    # dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    # if not os.path.exists(args.log_dir):
    #     os.mkdir(args.log_dir)
    # if not os.path.exists(os.path.join(args.log_dir, dir_name)):
    #     os.mkdir(os.path.join(args.log_dir, dir_name))
    # log_file = os.path.join(args.log_dir, dir_name, f'{log_name}.log')
    # logger = getLogger(log_file, __name__)
    # logger.info(f'Load config from {args.cfg}')

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
    batch_size = 5
    train_dataloader = get_clotho_loader(
        Path("/mnt/d/clotho/dataset"), 
        "evaluation", 
        Path("/mnt/d/clotho/csv/clotho_test.csv"), 
        batch_size, 
        shuffle=False,
        nb_t_steps_pad='max'
    )

    # max_step = (len(train_dataset) // cfg.dataset.train.batch_size) * \
    #     cfg.process.train_epochs

    opt = NesterovMomentumOptimizer(0.5)

    num_qubits = 4
    num_layers = 2
    weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
    bias_init = np.random.randn(num_qubits, requires_grad=True)

    print("Weights:", weights_init)
    print("Bias: ", bias_init)

    # Train
    best_epoch = 0
    global_step = 0
    weights = weights_init
    bias = bias_init
    for epoch in range(100):
        for n_iter, batch_data in enumerate(train_dataloader):
            audio, text = batch_data
            audio = list(audio)
            
            audio_feature = audio_encoder(audio)
            audio_feature = audio_feature.view(batch_size, -1, audio_feature.shape[1])
            audio_feature = audio_feature.mean(dim=1)
            audio_feature = audio_feature.detach().cpu().numpy()

            text_feature = text_encoder.encode(text)

            # print("audio shape", audio_feature.shape)
            # print("text shape", text_feature.shape)

            weights, bias = opt.step(cost, weights, bias, X=audio_feature, Y=text_feature, N=num_qubits)

            # Validate
            # Compute accuracy
            # predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X]
            # current_cost = cost(weights, bias, X, Y)

            global_step += 1
            # if (global_step - 1) % 20 == 0:
            #     train_log = 'Iter:%5d/%5d' % (
            #         global_step - 1, max_step)
                # logger.info(train_log)

        # Validation:


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('cfg', type=str, help='config file path')
    # parser.add_argument('--log_dir', type=str,
    #                     default='work_dir', help='log dir')
    # parser.add_argument('--checkpoint_dir', type=str,
    #                     default='work_dir', help='dir to save checkpoints')
    # parser.add_argument("--session_name", default="MS3",
    #                     type=str, help="the MS3 setting")

    # args = parser.parse_args()
    main()
