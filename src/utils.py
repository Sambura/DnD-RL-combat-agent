import numpy as np
import random
from math import ceil, sqrt
import matplotlib.pyplot as plt

def manhattan_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return abs(x1 - x2) + abs(y1 - y2)

def to_tuple(coords):
    return tuple(np.array(coords).flatten())

def get_random_coords(h, w):
    return (random.randrange(0, h), random.randrange(0, w))

def plot_featuremaps(data, name="", fm_names=None, vmin=None, vmax=None, shape=None, separate_cbars=False, show=True, cmap='coolwarm'):
    if shape is None:
        width = int(ceil(sqrt(data.shape[0])))
        height = int(ceil(data.shape[0] / width))
        shape = (height, width)
    
    if vmin is None and not separate_cbars: vmin = np.min(data)
    if vmax is None and not separate_cbars: vmax = np.max(data)

    fig, ax_m = plt.subplots(*shape, dpi=100)
    fig.suptitle(name)
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
