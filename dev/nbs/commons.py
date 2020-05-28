"""Commons functions for nbs."""

import numpy as np
import scipy.stats as st


def conf95(a):
    l, r = st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))
    return (l + r) / 2, (r - l) / 2


def merge(b, d):
    result = []
    for b1, d1 in zip(b, d):
        result.extend([b1, d1])

    result.append(b[-1])
    return np.array(result)


def split(durs):
    return np.array(durs[::2]), np.array(durs[1::2])


def adjust_durs(b, d):
    """Shares 1s with durs."""
    b, d, t = b.copy(), d.copy(), sum(b) + sum(d)

    doable = sum(b) >= len(d)

    for i in range(len(d)):
        if d[i] == 0:

            l, r = i, i + 1
            while True:
                if l < 0 and r >= len(b):
                    break

                if (l >= 0 and b[l] > 0) and (r >= len(b) or b[l] >= b[r]):
                    d[i] = 1
                    b[l] -= 1
                    break

                if (r < len(b) and b[r] > 0) and (l < 0 or b[r] > b[l]):
                    d[i] = 1
                    b[r] -= 1
                    break

                if l >= 0 and b[l] == 0:  # noqa
                    l -= 1  # noqa

                if r < len(b) and b[r] == 0:
                    r += 1

    assert sum(b) + sum(d) == t
    if doable:
        assert (b < 0).sum() == 0

    return (b, d), doable
