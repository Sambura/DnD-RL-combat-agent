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

def plot_training_history(iters, 
                          eps=None, 
                          checkpoints=None,
                          vlines=None,
                          xlim=None, 
                          ylim=None,
                          min_ymax=None,
                          figsize=(11, 6), 
                          smoothness=[300, 3000], 
                          average_last=1000,
                          show=True):
    plt.figure(figsize=figsize)
    plt.title('Number of iterations per game')
    plt.xlabel('Game')
    plt.ylabel('Iterations')
    if xlim is None: xlim = [None, None]
    else: xlim = list(xlim)
    if xlim[0] is None: xlim[0] = 0
    if xlim[1] is None: xlim[1] = len(iters)
    if ylim is None:
        y_min = np.min(iters)
        y_range = np.mean(iters[-2000:]) - y_min
        ylim = (y_min, np.max(iters) if min_ymax is None else max(min_ymax, y_min + y_range * 1.25))
    elif not hasattr(ylim, '__len__'): 
        ylim = (np.min(iters), ylim)

    plt.xlim(xlim)
    plt.ylim(ylim)

    ax: plt.Axes = plt.gca()
    ax.plot(iters, label='Iterations', alpha=0.65)
    if smoothness is not None:
        colors = ['orange', 'k', 'green'] * (len(smoothness) // 3 + 1)
        for n, color in zip(smoothness, colors):
            smooth = np.convolve(iters, np.ones(n) / n, mode='valid')
            ax.plot(range(n // 2, len(smooth) + n // 2), smooth, label='Iterations, averaged', color=color)

    artists, labels = ax.get_legend_handles_labels()

    if average_last is not None:
        avg = np.mean(iters[-average_last:])
        y_min, y_max = ax.get_ylim()
        yspan = y_max - y_min
        avg_nml = np.clip((avg - y_min) / yspan, 0.01, 0.99)
        ax.axhline(avg, c='k', lw=2, alpha=0.5)
        text_y_nml = avg_nml + (0.03 if avg_nml >= 0.5 else -0.05)
        text_y = text_y_nml * yspan + y_min
        text_x = xlim[0] * 0.95 + xlim[1] * 0.05
        ax.text(text_x, text_y, f'avg = {avg:0.2f}', fontsize=14)
    
    ax2 = None
    if eps is not None:
        ax2 = plt.twinx()
        ax2.plot(eps, color='red', label='Epsilon')

        for x, y in zip((artists, labels), ax2.get_legend_handles_labels()): x += y

    if vlines is None: vlines = []
    else: vlines = vlines.copy()
    if len(vlines) > 0 and not hasattr(vlines[0], '__len__'): vlines = [vlines]
    if checkpoints is not None: vlines.append({'data': checkpoints, 'c': 'k', 'alpha': 0.6, 'lw': 1.5, 'linestyle': '--'})

    for vline in vlines:
        if isinstance(vline, dict):
            points = vline.pop('data')
            kwargs = vline
        else:
            points = vline
            kwargs = { 'c': 'red', 'lw': 1, 'linestyle': '--', 'alpha': 0.8}
            
        for point in points:
            ax.axvline(point, **kwargs)

    ax.legend(artists, labels, loc='upper left')

    if show: plt.show()
    else: return (ax, ax2) if ax2 is not None else ax
