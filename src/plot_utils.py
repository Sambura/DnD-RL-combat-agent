from itertools import zip_longest
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, sqrt
from typing import Optional

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

def plot_training_history(iters, eps=None, checkpoints=None, xlim=None, ylim=None, figsize=(11, 6), smoothness=[300, 3000], show=True):
    plt.figure(figsize=figsize)
    plt.title('Number of iterations per game')
    plt.xlabel('Game')
    plt.ylabel('Iterations')
    if xlim is None: xlim = [None, None]
    else: xlim = list(xlim)
    if xlim[0] is None: xlim[0] = 0
    if xlim[1] is None: xlim[1] = len(iters)
    if ylim is not None: 
        if not hasattr(ylim, '__len__'): plt.ylim(np.min(iters), ylim)
        else: plt.ylim(ylim)
    plt.xlim(xlim)

    ax: plt.Axes = plt.gca()
    ax.plot(iters, label='Iterations', alpha=0.65)
    if smoothness is not None:
        colors = ['orange', 'k', 'green'] * (len(smoothness) // 3 + 1)
        for n, color in zip(smoothness, colors):
            smooth = np.convolve(iters, np.ones(n) / n, mode='valid')
            ax.plot(range(n // 2, len(smooth) + n // 2), smooth, label='Iterations, averaged', color=color)

    artists, labels = ax.get_legend_handles_labels()

    if checkpoints is not None:
        for checkpoint in checkpoints:
            ax.axvline(checkpoint, c='k', alpha=0.6, lw=1.5, linestyle='--')
    
    if eps is not None:
        ax2 = plt.twinx()
        ax2.plot(eps, color='red', label='Epsilon')

        for x, y in zip((artists, labels), ax2.get_legend_handles_labels()): x += y

    ax2.legend(artists, labels)

    if show: plt.show()
