import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


plt.style.use('ggplot')


def plot_time_series(
        data,
        y,
        figsize=[25, 10],
        fig_name=None,
        same_plot=False,
        **kwargs):

    if isinstance(y, str):
        y = [y]

    if same_plot:
        nrows = 1

    else:
        nrows = len(y)

    figsize[1] = 2 * nrows

    f, a = plt.subplots(figsize=figsize, nrows=nrows, sharex=True)
    a = np.array(a).flatten()

    for idx, y_label in enumerate(y):
        if same_plot:
            idx = 0
        a[idx].set_title(y_label)
        data.plot(y=y_label, ax=a[idx], **kwargs)

    if fig_name:
        ensure_dir(fig_name)
        f.savefig(fig_name)

    return f


def plot_flex_episode(plot_data, fig_path='./'):

    f = plot_time_series(
        plot_data,
        y=['site_demand', 'site_consumption'],
        same_plot=True,
        fig_name=join(fig_path, 'fig1.png')
    )

    f = plot_time_series(
        plot_data,
        y=['electricity_price', 'setpoint', 'reward', 'site_consumption'],
        fig_name=join(fig_path, 'fig2.png')
    )
