"""Commons functions for nbs."""


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
