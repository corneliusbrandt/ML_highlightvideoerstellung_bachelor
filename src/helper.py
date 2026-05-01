import numpy as np


def load_data(dataset_path):
    data = np.load(dataset_path)

    X = data['X']
    y = data['y']

    return X, y

def format_data_for_pytorch(X, y):
    # reshape X to (num_samples, num_channels, num_time_steps)
    # because PyTorch expects (batch_size, num_channels, num_time_steps)
    X = np.transpose(X, (0, 2, 1))
    return X, y