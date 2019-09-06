"""
Makes an animation of sta/lta detector
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import obspy
from obspy.signal.trigger import classic_sta_lta


def _create_figure():
    """ Return a figure and its axis. """
    plt.clf()
    plt.cla()
    return plt.subplots(2, 1, sharex=True)


def _plot_crosshair(ax, start, stop, offset, length=50, **kwargs):
    """ Plots crosshairs defining window. """
    # first plot horizontal line
    ax.plot((start, stop), (offset, offset), **kwargs)
    # now vertical lines
    ax.plot((start, start), (offset - length, offset + length), **kwargs)
    ax.plot((stop, stop), (offset - length, offset + length), **kwargs)


def animate(tr, path, samps_per_frame=40, sta=5, lta=10, ton=1.5, toff=0.5):
    """
    Create the animation.
    """
    # ensure we have a path and create
    path = Path(__file__).parent / Path(path)
    path.mkdir(exist_ok=True, parents=True)
    # separate data, stats, sample rate, etc.
    data, stats = tr.data, tr.stats
    sr = stats.sampling_rate
    # get characteristic function
    sta_samp, lta_samp = int(sta * sr), int(lta * sr)
    char = classic_sta_lta(data, sta_samp, lta_samp)
    char[:lta_samp] = np.NaN
    # init subpots, plot
    data_abs = np.nanmax(abs(data))
    char_abs = np.nanmax(abs(char))
    # iterate time
    for num, i in enumerate(range(lta_samp, len(data), samps_per_frame)):
        fig, (ax1, ax2) = _create_figure()
        # fig.patch.set_facecolor('0.8')
        ar_, char_ = np.ones_like(data) * np.NAN, np.ones_like(char) * np.NAN
        ar_[: i], char_ = data[: i], char[: i]
        # plot char func and data
        ax1.plot(ar_, color='k')
        ax2.plot(char_, color='g')
        win_lta, win_sta = (i, i + lta_samp), (i, i + sta_samp)
        _plot_crosshair(ax1, i - sta_samp, i, 200, color='r', alpha=.6)
        _plot_crosshair(ax1, i - lta_samp, i, -200, color='b', alpha=.6)
        # finish plot
        ax1.set_xlim(0, len(data))
        ax1.set_ylim(-data_abs * 1.1, data_abs * 1.1)
        ax1.set_ylabel('amplitude', fontdict={'fontsize': 18})
        ax1.set_yticks([])
        ax2.set_xlim(0, len(data))
        ax2.set_ylim(0, char_abs * 1.1)
        ax2.set_xticks([])
        ax2.set_xlabel('time', fontdict={'fontsize': 18})
        ax2.set_ylabel('sta/lta', fontdict={'fontsize': 18})
        ax2.set_yticks([])
        ax2.axhline(ton, color='.5', ls='--')
        # save plot
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.savefig(path / f'{num:03d}.png')
        # clear state



if __name__ == "__main__":
    # get reference to output path
    data_path = "https://examples.obspy.org/ev0_6.a01.gse2"
    path = Path('stalta_single')
    # get the stream, array, and stats
    tr = obspy.read(data_path).select(component='Z').detrend('linear')[0]
    # create the images for the animation
    animate(tr, path)
