from math import ceil


def get_steps(num_samples, batch_size):
    return ceil(num_samples / batch_size)
