"""
Creates an animation for a network coincidence filter.
"""
from dataclasses import dataclass
from pathlib import Path

import bottleneck as bn
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numba
import numpy as np
import obspy
import pandas as pd
from obspy.signal.trigger import recursive_sta_lta


def get_streams():
    """ Just get the streams used in the tutoral. """
    st = obspy.Stream()
    files = ["BW.UH1..SHZ.D.2010.147.cut.slist.gz",
             "BW.UH2..SHZ.D.2010.147.cut.slist.gz",
             "BW.UH3..SHZ.D.2010.147.cut.slist.gz",
             "BW.UH4..SHZ.D.2010.147.cut.slist.gz"]
    for filename in files:
        st += obspy.read("https://examples.obspy.org/" + filename)

    assert len({len(x) for x in st}) == 1

    return st


def get_fig_axis():
    """ Use grid spec to create layout. """
    plt.cla(), plt.clf()
    plt.figure(figsize=(12, 8))
    gs = plt.subplots(5, 1, sharex='all')
    plt.tight_layout(h_pad=0)
    return gs


def get_char_funcs(st, sta=0.5, lta=10.0):
    """ return characteristic functions for each tr in st. """
    # get sampling rate
    srs = {tr.stats.sampling_rate for tr in st}
    assert len(srs) == 1
    sr = list(srs)[0]
    nsta = int(sta * sr)
    nlta = int(lta * sr)
    out = []
    for tr in st:
        char = recursive_sta_lta(tr.data, nsta, nlta)
        char[char == 0] = np.NaN
        out.append(char)
    return out


@dataclass()
class TriggerOnset:
    """ Class for getting trigger onset arrays. """

    thresh_on: float
    thresh_off: float

    def trigger_onset(self, charfct: np.ndarray):
        """ Get return arrays indicating if trigger is on for array. """
        out_ar = np.zeros_like(charfct)
        t1_on = charfct > self.thresh_on
        t2_off = charfct < self.thresh_off
        assert t1_on.shape == t2_off.shape
        out = np.zeros((np.sum(t1_on), 2))
        out = self._jited_onset(t1_on, t2_off, out)
        on_off = out[out[:, 1] > 0]
        for (on_ind, off_ind) in on_off:
            out_ar[int(on_ind): int(off_ind)] = 1
        return out_ar.astype(int)

    @staticmethod
    @numba.njit
    def _jited_onset(t1_on, t2_off, out):
        started = 0
        started_index = 0
        out_ind = 0
        for num in range(len(t1_on)):
            if t1_on[num] and not started:
                started = 1
                started_index = num
                continue
            if started and t2_off[num]:
                out[out_ind, :] = started_index, num - 1
                out_ind += 1
                started = 0
        return out

    def __call__(self, chars):
        """ boolean arrays indicating if trigger is on or off. """
        out = [self.trigger_onset(x) for x in chars]
        for char, ou in zip(chars, out):
            assert len(char) == len(ou)
        return out


def make_wf_plot(ax, tr, char, ison, ind):
    """ Make the plot for the waveform panel"""
    # normalize to make plotting easy
    data_ = (tr.data / np.nanmax(np.abs(tr.data)))[:ind]
    char_ = (char / np.nanmax(np.abs(char)))[:ind]
    ison_ = ison[:ind]

    off, on = np.ones(ind) * np.NaN, np.ones(ind) * np.NaN
    ison_bool = ison_.astype(bool)
    off[~ison_bool] = char_[~ison_bool]
    on[ison_bool] = char_[ison_bool]
    # plot the data and trigger on/off
    ax.plot(data_, color='0.5', alpha=.5)
    ax.plot(off, color='b', alpha=.5)
    ax.plot(on, color='r')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, len(char))
    ax.set_ylim(-1.1, 1.1)


def get_event_indicies(coin_sum, thresh=3, min_duration=100):
    """ Return a list of indicies for each event. """
    data = np.vstack([coin_sum, range(len(coin_sum))]).T
    df = pd.DataFrame(data, columns=['trig_count', 'index'])
    df['groupnum'] = (df['trig_count'] != df['trig_count'].shift()).cumsum()
    # df = df[df['trig_count'] >= thresh]
    gb = df.groupby('groupnum')
    start = gb['index'].min()
    end = gb['index'].max()
    count = gb['trig_count'].mean()
    duration = end - start
    # create data frame and filter
    out = pd.DataFrame([start, end]).T
    out = out[(duration >= min_duration) & (count >= thresh)]
    return out.values


def get_smoothed(trig_on, sum_window):
    ar = np.array(trig_on)
    rmax = bn.move_max(ar, window=sum_window, min_count=5, axis=1)
    return np.sum(rmax, axis=0)


def plot_bottom(ax, coin_sum_smoothed, event_inds, coin_thresh, ind):
    """ Plot the bottom panel. """
    buffer = 200
    sum_ = coin_sum_smoothed[:ind]
    yrange = (-1, np.nanmax(coin_sum_smoothed) + 1)

    # Plot event windows
    for (start, end) in event_inds:
        if ind < start:  # havent got to event yet, skip
            continue
        start_ind = start - buffer
        width = (min(end, ind) - start) + 2*buffer
        height = yrange[1] - yrange[0]
        rect = patches.Rectangle((start_ind, -1), width=width, height=height,
                                 facecolor='r', alpha=.3)
        ax.add_patch(rect)

    # plot sum
    ax.plot(sum_, c='b', alpha=.5)
    # plot threshold
    ax.axhline(coin_thresh, color='.4', ls='--', alpha=.5)

    ax.set_xticks([])
    ax.set_ylabel('count', fontdict={'size': 16})
    ax.set_xlabel('time', fontdict={'size': 16})
    ax.set_yticks([])
    ax.set_xlim(0, len(coin_sum_smoothed))
    ax.set_ylim(yrange)


if __name__ == "__main__":
    # set constants
    path = Path(__file__).parent / 'cofilt'
    thresh_on = 3.5
    thresh_off = 1.0
    sample_step = 30
    coin_sum_window = 100
    coin_thresh = 3

    path.mkdir(exist_ok=True, parents=True)
    # get stream and characteristic functions
    st = get_streams()
    chars = get_char_funcs(st)
    trig_on = TriggerOnset(thresh_on, thresh_off)(chars)
    # coin_sum = np.array(trig_on).sum(axis=0)
    coin_sum_smoothed = get_smoothed(np.array(trig_on), coin_sum_window)
    event_inds = get_event_indicies(coin_sum_smoothed, coin_thresh,
                                    coin_sum_window)

    # now iterate each time sample
    num_samps = len(chars[0])
    for pltnum, i in enumerate(range(400, len(chars[0]), sample_step)):
        fig, axs = get_fig_axis()  # recreate grids and such
        for num, ax in enumerate(axs[:-1]):  # plot char function and waveforms
            make_wf_plot(ax, st[num], chars[num], trig_on[num], i)
            ax.set_ylabel(f"sta {num + 1}",  fontdict={'size': 16})
        # plot bottom panel
        plot_bottom(axs[-1], coin_sum_smoothed, event_inds, coin_thresh, i)
        plt.subplots_adjust(hspace=0)
        plt.savefig(path / f"{pltnum:03d}.png", dpi=400)

    # for i in num_samps
