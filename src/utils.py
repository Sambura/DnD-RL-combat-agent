import matplotlib.pyplot as plt
from math import ceil, sqrt
import numpy as np
import random

from typing import Union, Optional

IntPoint2d = Union[tuple[int, int], np.ndarray, list[int]]

def manhattan_distance(point1: IntPoint2d, point2: IntPoint2d) -> int:
    """Calculate Manhattan distance between two points"""
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)

def to_tuple(coords: IntPoint2d) -> tuple[int, int]:
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
    data (ndarray): ndarray of 2D featuremaps with shape [N, H, W], where N - number of 
        featuremaps, H - height, W - width
    title (str): Title of the plot
    fm_names (list[str]) list of length N, with names of the featuremaps in `data`
    vmin, vmax: refer to plt.colorbar()
    shape (tuple[int, int]): Size of grid where featuremap will be plotted
    separate_cbars (bool): If True, each featuremap will have its own colorbar, 
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
    for i, (fm, ax) in enumerate(zip(data, axs)):
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
