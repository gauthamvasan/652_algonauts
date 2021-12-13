import torch
from torch import nn

# dict to enable loading activations based on a string
nn_activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
}


def mlp_hidden_layers(input_dim, hidden_sizes, activation="relu"):
    """ Helper function to create hidden MLP layers.
    N.B: The same activation is applied after every layer

    Args:
        input_dim: An int denoting the input size of the mlp
        hidden_sizes: A list with ints containing hidden sizes
        activation: A str specifying the activation function

    Returns:

    """
    activation = nn_activations[activation]
    dims = [input_dim] + hidden_sizes
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        layers.append(activation)
    return layers

def reset_weights(m):
    '''
      Try resetting model weights to avoid
      weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def orthogonal_weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class SimpleMLP(nn.Module):
    '''
      Simple Convolutional Neural Network
    '''

    def __init__(self, input_dim, output_dim, hidden_sizes=[2048, 2048], activation='relu', device=torch.device('cpu')):
        super().__init__()
        self.device = device

        layers = mlp_hidden_layers(input_dim, hidden_sizes, activation=activation)
        layers.append(nn.Linear(hidden_sizes[-1], output_dim))
        self.layers = nn.Sequential(*layers)
        self.apply(orthogonal_weight_init)
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        return self.layers(x)
