import matplotlib.pyplot as plt
from math import ceil, sqrt
import numpy as np
import random
from itertools import zip_longest

from typing import Union, Optional

IntPoint2d = Union[tuple[int, int], np.ndarray, list[int]]

def manhattan_distance(point1: IntPoint2d, point2: IntPoint2d) -> int:
    """Calculate Manhattan distance between two points"""
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def to_tuple(coords: any) -> tuple:
    """Convert from list/ndarray coordinates representation to tuple of ints"""
    return tuple(np.array(coords).flatten())

def get_random_coords(max_y: int, max_x: int) -> tuple[int, int]:
    """Get a random coordinate on a board of size (max_y, max_x)"""
    return (random.randrange(0, max_y), random.randrange(0, max_x))

def plot_featuremaps(data: np.ndarray,
                     title: Optional[str]=None,
                     fm_names: Optional[list[str]]=None,
                     vmin: Optional[float]=None,
                     vmax: Optional[float]=None,
                     shape: Optional[tuple[int, int]]=None,
                     separate_cbars: bool=False,
                     show: bool=True,
                     cmap: str='coolwarm',
                     cell_size: float=2.5) -> Optional[tuple[plt.Figure, np.ndarray]]:
    """
    Plots a stack of 2D featuremaps

    Parameters:
    data (ndarray): ndarray of 2D featuremaps with shape [N, H, W], where N - number of \
        featuremaps, H - height, W - width
    title (str): Title of the plot
    fm_names (list[str]) list of length N, with names of the featuremaps in `data`
    vmin, vmax: refer to plt.colorbar()
    shape (tuple[int, int]): Size of grid where featuremap will be plotted
    separate_cbars (bool): If True, each featuremap will have its own colorbar, \
        and `vmin` & `vmax` parameters will be ignored
    show: (bool): If True, plt.show() is called. Otherwise tuple (fig, axes) is returned
    cmap (str): colormap to use
    cell_size (float): basically represents figure size

    Returns:
    Tuple of (Figure, axes) if `show` == False, otherwise None
    """
    if shape is None:
        width = int(ceil(sqrt(data.shape[0])))
        height = int(ceil(data.shape[0] / width))
        shape = (height, width)
    
    if separate_cbars: 
        vmin = None
        vmax = None
    else:
        if vmin is None: vmin = np.min(data)
        if vmax is None: vmax = np.max(data)

    fig, ax_m = plt.subplots(*shape, dpi=100, figsize=np.array(shape[::-1]) * cell_size)
    if title is not None: fig.suptitle(title)
    axs = np.asarray(ax_m).ravel()
    imgs = []
    for i, (fm, ax) in enumerate(zip_longest(data[:len(axs)], axs)):
        if fm is None:
            ax.set_axis_off()
            continue

        imgs.append(ax.imshow(fm, cmap=cmap, vmin=vmin, vmax=vmax))
        if fm_names is not None:
            ax.set_title(fm_names[i])
        
        if separate_cbars:
            fig.colorbar(imgs[i], ax=ax)

    if not separate_cbars:
        fig.colorbar(imgs[0], ax=axs)
    
    if show: 
        if separate_cbars: plt.tight_layout()
        plt.show()
        return
    
    return fig, ax_m

def bytes_to_human_readable(bytes):
    if bytes < 1024:
        return f'{bytes} bytes'
    elif bytes < 1024**2:
        return f'{bytes / 1024:.2f} KB'
    elif bytes < 1024**3:
        return f'{bytes / (1024**2):.2f} MB'
    elif bytes < 1024**4:
        return f'{bytes / (1024**3):.2f} GB'
    else:
        return f'{bytes / (1024**4):.2f} TB'

def transform_matrix(matrix, func):
    result_matrix = np.empty_like(matrix)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result_matrix[i, j] = func(i, j, matrix[i, j])

    return result_matrix
