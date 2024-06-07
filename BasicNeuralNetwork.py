# <GPLv3_Header>
## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# \copyright
#                    Copyright (c) 2024 Nathan Ulmer.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# <\GPLv3_Header>

##
# \mainpage Basic 2 layer Neural network
#
# \copydoc BasicNeuralNetwork.py

##
# \file BasicNeuralNetwork.py
#
# \author Nathan Ulmer
#
# \date \showdate "%A %d-%m-%Y"
#
# This started by using the classic sigmoid activation function, but that appears

import numpy as np
import numpy.typing as npt
class BasicNeuralNetwork:
    def __init__(self,input_size: int,hidden_size: int,output_size: int):
        scale_factor = 0.01

        self.weights = np.random.randn(hidden_size,output_size) * scale_factor
        self.bias = np.zeros((1,output_size))

        self.weights_hidden = np.random.randn(input_size,hidden_size) * scale_factor
        self.bias_hidden = np.zeros((1,hidden_size))

    def sigmoid(self,x: npt.ArrayLike) -> npt.ArrayLike:
        x_cpy = np.clip(x, -500, 500) # Clip to prevent overflow
        return 1.0 / (1.0 + np.exp(-1.0 * x_cpy))

    def sigmoid_derivative(self,x: npt.ArrayLike) -> npt.ArrayLike:
        return x * (1.0 - x)

    def relu(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.maximum(0, x)

    def relu_derivative(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.where(x > 0, 1, 0)

    def forward(self, inputs: npt.ArrayLike):
        self.hidden_inputs = np.dot(inputs.T,self.weights_hidden) + self.bias_hidden
        self.hidden_outputs = self.relu(self.hidden_inputs)
        self.final_inputs = np.dot(self.hidden_outputs, self.weights) + self.bias
        self.final_outputs = self.final_inputs
        return self.final_outputs

    def train(self, inputs: npt.ArrayLike, targets: npt.ArrayLike, learning_rate: float) -> float:
        # Forward pass
        outputs = self.forward(inputs)

        # Calculate the error
        output_errors = targets - outputs
        output_delta = output_errors

        # Backpropagation
        hidden_errors = np.dot(output_delta, self.weights.T)
        hidden_delta = np.multiply(hidden_errors,  self.relu_derivative(self.hidden_outputs))

        # Update weights and biases
        self.weights += np.dot(self.hidden_outputs.T, output_delta) * learning_rate
        self.bias += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        print(np.size(inputs))
        print(np.size(hidden_delta))
        self.weights_hidden += np.dot(inputs, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

        # Mean squared error
        MAX_ERROR = 1e3
        output_errors = np.where(output_errors < MAX_ERROR, MAX_ERROR, output_errors)
        loss = np.mean(np.power(output_errors,2.0))
        print(loss)
        loss = np.fmin(np.fabs(loss), MAX_ERROR)
        return loss

    # <GPLv3_Footer>
    ################################################################################
    #                      Copyright (c) 2024 Nathan Ulmer.
    ################################################################################
    # <\GPLv3_Footer>