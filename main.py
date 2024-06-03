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
# \mainpage Procedural Music Generator
#
# \copydoc main.py

##
# \file main.py
#
# \author Nathan Ulmer
#
# \date \showdate "%A %d-%m-%Y"
#

from BasicNeuralNetwork import *
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def generate_data(num_samples: int, num_points: int, freq_range) -> npt.ArrayLike:
    X = np.zeros((num_samples, num_points))
    y = np.zeros(num_samples)

    for i in range(num_samples):
        freq = np.random.uniform(*freq_range)
        y[i] = freq
        t = np.linspace(0.0, 1.0, num_points)
        X[i] = np.sin(2.0 * np.pi * freq * t)

    return X, y



def main():
    np.random.seed(0)
    num_samples = 10
    num_points = 100
    freq_range = (1.0, 10.0)  # Frequency range from 1Hz to 10Hz
    X, y = generate_data(num_samples, num_points, freq_range)

    # Plot some samples
    plt.figure()
    print(len(X))
    for ii in range(len(X)):
        plt.plot(np.linspace(0, 1, num_points), X[ii])
    plt.title(f"Sample signal with frequency {y[0]:.2f} Hz")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # Normalize data
    epsilon = 1e-6
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std = np.where(X_std==0.0, epsilon, X_std)
    X = (X - X_mean) / X_std
    print(X)
    # Initialize the neural network
    input_size = num_points
    hidden_size = 50
    output_size = 1
    learning_rate = 0.01
    epochs = 10

    nn = BasicNeuralNetwork(input_size,hidden_size,output_size)

    # Train the neural network
    losses = []
    print_freq = 1
    for epoch in range(epochs):
        loss = nn.train(X, y.reshape(-1, 1), learning_rate)
        losses.append(loss)
        if epoch % print_freq == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Plot the loss
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Generate test data
    X_test, y_test = generate_data(100, num_points, freq_range)

    # Normalize test data
    X_test = (X_test - X_mean) / X_std

    # Predict
    predictions = nn.forward(X_test)

    # Plot predictions vs true values
    plt.figure()
    plt.scatter(y_test, predictions)
    plt.xlabel("True Frequencies")
    plt.ylabel("Predicted Frequencies")
    plt.title("True vs Predicted Frequencies")

    plt.figure()
    plt.plot(y_test-predictions)
    plt.xlabel("Test Num")
    plt.ylabel("Frequency Error")
    plt.title("True vs Predicted Frequencies")

    plt.show()


if __name__ == '__main__':
    main()

# <GPLv3_Footer>
################################################################################
#                      Copyright (c) 2024 Nathan Ulmer.
################################################################################
# <\GPLv3_Footer>