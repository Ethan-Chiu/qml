from tqdm import tqdm
import pennylane as qml
from pennylane import numpy as np


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




# NOTE: validation
@qml.qnode(dev)
def encoder_state(N, en_w, in_w, x):
    encoder(N, en_w, in_w, x)
    return qml.state()

@qml.qnode(dev)
def decode_state(N, de_w, state, y):
    wires = list(range(N))
    
    qml.QubitStateVector(state, wires=wires)
    decoder(N, de_w, y)

    wires = list(range(N))
    return qml.probs(wires=wires)


