from typing import Dict
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.special


def plot_implied_timescales(
        its, step=1, unit="ps", processes=None,
        ax=None, ax_props=None,
        line_props=None, ref_props=None):
    """Plot implied timescales vs. lag time

    Args:
        its: Mapping of integer keys (lag time) to sequences
            (time scales).
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if processes is None:
        processes = min([len(v) for v in its.values()])

    line_props_defaults = {
        }

    if line_props is not None:
        line_props_defaults.update(line_props)

    ref_props_defaults = {
        'linestyle': '--',
        'color': 'k',
        }

    if ref_props is not None:
        ref_props_defaults.update(ref_props)

    time = np.array([int(k) for k in its]) * step
    sorted_indices = np.argsort(time)
    time = time[sorted_indices]

    timescales = np.vstack([v for v in its.values()]) * step
    lines = ax.plot(time, timescales[sorted_indices, :], **line_props_defaults)
    ref = ax.plot(time, time, **ref_props_defaults)

    ax_props_defaults = {
        'xlabel': r"$\tau$ " + f" / {unit}",
        'ylabel': f"its / {unit}",
        'xlim': (time[0], time[-1]),
        }

    if ax_props is not None:
        ax_props_defaults.update(ax_props)

    ax.set(**ax_props_defaults)

    return (fig, ax, lines, ref)


def plot_eigenvectors(
        eigenvectors, states=None, norm=True, ax=None,
        ax_props=None, fill_props=None, line_props=None,
        style="clamp", clampfactor=4,
        grid=True, grid_props=None, invert=False):

    if style not in {"clamp", "step"}:
        raise ValueError(
            """Keyword argument 'style' must be one of "clamp", or "step" """
            )

    if states is None:
        states = list(range(eigenvectors.shape[0]))

    if invert:
        eigenvectors *= -1

    if norm:
        globalmax = np.max(
            [abs(np.max(eigenvectors)), abs(np.min(eigenvectors))]
            )
        eigenvectors /= globalmax

    drawn = eigenvectors.shape[1]
    if ax is None:
        figsize = mpl.rcParams["figure.figsize"]
        fig, Ax = plt.subplots(
            drawn, 1, figsize=(figsize[0], figsize[1] * drawn)
            )
        if drawn == 1:
            Ax = [Ax]
    else:
        if drawn == 1:
            fig = ax.get_figure()
            Ax = [ax]
        else:
            Ax = ax
            assert drawn == len(Ax)
            fig = Ax[0].get_figure()

    line_props_defaults = {
        'color': 'k',
        'lw': 1
    }
    if line_props is not None:
        line_props_defaults.update(line_props)

    fill_props_defaults = {
        'interpolate': True,
    }

    if fill_props is not None:
        fill_props_defaults.update(fill_props)

    grid_props_defaults = {
        'which': "minor",
        'axis': "x",
        'alpha': 0.5
    }

    if grid_props is not None:
        grid_props_defaults.update(grid_props)

    index = np.linspace(1, drawn, drawn)
    x = range(len(states))

    for axi, vector in enumerate(eigenvectors):

        if style == 'clamp':
            xpieces = []
            ypieces = []

            for c, value in enumerate(vector[:-1]):
                xpieces.append(np.linspace(0.8 + c, 1.2 + c, 2))
                ypieces.append([value, value])

                x = np.linspace(1.2 + c, 1.8 + c, 100)
                xpieces.append(x)

                smoothed = smoothstep(
                    x,
                    x_min=1.2 + c, x_max=1.8 + c,
                    y_min=0, y_max=1, N=clampfactor
                    )

                ypieces.append(
                    (smoothed * (vector[c + 1] - value)) + value
                    )

            xpieces.append(np.linspace(0.8 + c + 1, 1.2 + c + 1, 2))
            ypieces.append([vector[-1], vector[-1]])

            xt = np.concatenate(xpieces)
            yt = np.concatenate(ypieces)

            Ax[axi].plot(xt, yt, **line_props_defaults)
            Ax[axi].fill_between(
                xt, 0, yt,
                where=yt > 0,
                **fill_props_defaults
                )
            Ax[axi].fill_between(
                xt, 0, yt,
                where=yt < 0,
                **fill_props_defaults
                )

        elif style == "step":
            Ax[axi].step(x, vector, where='mid')

        Ax[axi].axhline(
            y=0, xmin=0, xmax=drawn,
            color='k', linestyle='--'
            )

        if grid:
            Ax[axi].set_xticks((index[:-1] + 0.5), minor=True)
            Ax[axi].grid(**grid_props_defaults)

        ax_props_defaults = {
            "xticks": (),
            "yticks": (),
            "xlim": (index[0] - 0.2, index[-1] + 0.2),
            "ylim": (-1.2, 1.2),
            "ylabel": f"{axi + 1}"
            }

        if ax_props is not None:
            ax_props_defaults.update(ax_props)

        Ax[axi].set(**ax_props_defaults)

    Ax[axi].set(**{
        "xticks": index,
        "xticklabels": [x + 1 for x in states],
        "xlabel": 'set',
        })

    fig.subplots_adjust(
        left=0.08,
        bottom=0.1,
        right=0.99,
        top=0.99,
        wspace=None,
        hspace=0
        )

    return fig, Ax


def smoothstep(x, x_min=0, x_max=1, y_min=0, y_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), y_min, y_max)

    result = 0
    for n in range(0, N + 1):
        a = scipy.special.comb(N + n, n)
        b = scipy.special.comb(2 * N + 1, N - n)
        result += a * b * (-x) ** n

    result *= x ** (N + 1)

    return result
