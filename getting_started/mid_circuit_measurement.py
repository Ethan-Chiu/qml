import pennylane as qml

dev = qml.device("default.qubit")

@qml.qnode(dev)
def before():
    qml.Hadamard(0)  # Create |+> state
    return qml.expval(qml.X(0)), qml.expval(qml.Z(0))

b = before()
print(f"Expectation values before any measurement: {b[0]:.1f}, {b[1]:.1f}")
