import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pylab as plt
from numba import jit, prange

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_alignment_to_numpy(alignment, title='', info=None, phoneme_seq=None,
                            vmin=None, vmax=None):
    if phoneme_seq:
        fig, ax = plt.subplots(figsize=(15, 10))
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment, aspect='auto', origin='lower',
                   interpolation='none', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')

    if phoneme_seq != None:
        # for debugging of phonemes and durs in maps. Not used by def in training code
        ax.set_yticks(np.arange(len(phoneme_seq)))
        ax.set_yticklabels(phoneme_seq)
        ax.hlines(np.arange(len(phoneme_seq)), xmin=0.0, xmax=max(ax.get_xticks()))

    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def save_plot(fname, attn_map):
    plt.imshow(attn_map)
    plt.savefig(fname)


@jit(nopython=True)
def mas(attn_map, width=1):
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]): # for each text dim
            prev_j = np.arange(max(0, j-width), j+1)
            prev_log = np.array([log_p[i-1, prev_idx] for prev_idx in prev_j])

            ind = np.argmax(prev_log)
            log_p[i, j] = attn_map[i, j] + prev_log[ind]
            prev_ind[i, j] = prev_j[ind]

    # now backtrack
    curr_text_idx = attn_map.shape[1]-1
    for i in range(attn_map.shape[0]-1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


# for very very large batch sizes
# usually slower than just running mas
@jit(nopython=True, parallel=True)
def b_mas(b_attn_map, in_lens, out_lens, width=1):
    attn_out = np.zeros_like(b_attn_map)

    for b in prange(b_attn_map.shape[0]):
        out = mas(b_attn_map[b, 0, :out_lens[b], :in_lens[b]], width=width)
        attn_out[b, 0, :out_lens[b], :in_lens[b]] = out
    return attn_out


if __name__ == '__main__':
    attn_ = np.load(sys.argv[1])
    attn = attn_.squeeze()
    save_plot('orig.png', attn)
    binarized = mas(attn)
    save_plot('binarized.png', binarized)
