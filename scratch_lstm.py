import torch
from torch import nn


def make_gate(features, bias=False):
    unit = nn.Sequential(
        nn.Linear(in_features=features, out_features=1, bias=bias),
        nn.Sigmoid())
    return unit


class LSTMCell(nn.Module):
    def __init__(self, input_features, layer_size):
        super().__init__()
        self.total_inputs = input_features + layer_size

        self.forget_gate = make_gate(self.total_inputs)
        self.remember_gate = make_gate(self.total_inputs)
        self.output_gate = make_gate(self.total_inputs)
        self.output_activation = nn.Tanh()
        self.used_device = None

        self.input_unit = nn.Sequential(
            nn.Linear(in_features=self.total_inputs, out_features=1),
            nn.ReLU()
        )

        self.state = torch.zeros(1)

    def set_batch_size(self, size):
        self.state = torch.zeros((size, 1), device=torch.device('cuda:0'))

    def forward(self, x):
        input_act = self.input_unit(x)
        remember_act = self.remember_gate(x)
        forget_act = self.forget_gate(x)
        output_gate_act = self.output_gate(x)

        self.state = (self.state * forget_act) + (input_act * remember_act)
        return self.output_activation(self.state) * output_gate_act


class LSTMLayer(nn.Module):
    def __init__(self, input_features, layer_size):
        super().__init__()
        self.layer_size = layer_size
        self.input_features = input_features
        self.cells = nn.ModuleList([LSTMCell(input_features, layer_size) for _ in range(layer_size)])

    def forward(self, x_batch):
        batch_size = x_batch.shape[1]
        for cell in self.cells:
            cell.set_batch_size(batch_size)
        last_h = torch.zeros((batch_size, self.layer_size), device=torch.device('cuda:0'))
        for time_step in x_batch:
            cell_input = torch.cat([time_step, last_h], dim=1)
            last_h = [cell(cell_input) for cell in self.cells]
            last_h = torch.cat(last_h, dim=1)
        return last_h
