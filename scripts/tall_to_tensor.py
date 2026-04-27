import numpy as np


def tall_to_tensor(arr):
    """Convert a tall 2D array to a 3D tensor.

    Col 0: item number (axis 0)
    Col 1: category (axis 1)
    Cols 2+: variables (axis 2 / planes)

    Returns: items, categories, tensor (num_items, num_categories, num_variables)
    """
    items = np.unique(arr[:, 0])
    categories = np.unique(arr[:, 1])
    num_vars = arr.shape[1] - 2

    item_idx = {v: i for i, v in enumerate(items)}
    cat_idx = {v: i for i, v in enumerate(categories)}

    tensor = np.full((len(items), len(categories), num_vars), np.nan)
    for row in arr:
        tensor[item_idx[row[0]], cat_idx[row[1]]] = row[2:].astype(float)

    return items, categories, tensor
