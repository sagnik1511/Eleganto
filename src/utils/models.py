from typing import Tuple


def network_shape_update_conv(input_shape: Tuple[int, int], kernel_size: int = 3, stride: int = 1, padding: int = 0):
    height, width = input_shape
    updated_height = (height - kernel_size + 2 * padding) // stride + 1
    updated_width = (width - kernel_size + 2 * padding) // stride + 1

    return updated_height, updated_width


def network_shape_update_pool(input_shape: Tuple[int, int], kernel_size: int):
    height, width = input_shape
    updated_height = height // kernel_size
    updated_width = width // kernel_size

    return updated_height, updated_width
