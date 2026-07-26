"""
femmi/convergence.py
Fitting convergence orders, and refusing to fit one that isn't there.

WHY THIS EXISTS
---------------
This project has twice reported a convergence order that was actually an error
FLOOR:

  * measuring the forward shear against the infinite-domain analytic shear of a
    Gaussian, which floors near 2.4e-2 and reads as "order ~1";
  * measuring the COUPLED C^1 solve the same way, which floored near 2.6e-2 and
    was written up as the square's corner singularity capping the rate at
    O(h^{5/3}). It wasn't. On a compactly supported field the same solver holds
    O(h^4) (MATH.md 18.3f).

Both times the data said the same thing and it was easy to miss: the error stops
improving while h keeps shrinking, so a least-squares slope through log h still
returns a number, and the number is meaningless.

`fit_order` returns that slope only when the curve is actually converging, and
otherwise says the reference is floored. The test for it is cheap: a converging
curve keeps a healthy LOCAL order at the fine end, while a floored one goes flat
there no matter how good the coarse end looked.

    h, err = ...                       # finest last
    order = fit_order(h, err)          # raises FloorError if the curve plateaus
    order = fit_order(h, err, on_floor="warn")
"""

from __future__ import annotations
import warnings
import numpy as np


class FloorError(RuntimeError):
    """Raised when a convergence order is requested from a plateaued curve."""


def local_orders(h, err):
    """Per-interval convergence orders. Length len(h) - 1."""
    h = np.asarray(h, float); err = np.asarray(err, float)
    return np.log(err[1:] / err[:-1]) / np.log(h[1:] / h[:-1])


def floor_diagnosis(h, err, min_tail_order=0.5, min_total_decay=None):
    """Decide whether an error curve is converging or has hit a floor.

    h, err   : arrays ordered COARSE to FINE (h decreasing).
    Returns a dict with the fitted order, local orders, total decay, and a
    `floored` flag with the reason.

    Two independent symptoms, either of which condemns the fit:

      * the LOCAL order at the fine end collapsed (`min_tail_order`) -- the curve
        has gone flat exactly where the asymptotic rate should be clearest. This
        is the test that does the work: it catches both curves this project got
        wrong, where the coarse end looked like a plausible order and the average
        over the whole range hid the plateau;
      * the error moved less than even `min_tail_order` would produce over the
        measured h-range. This threshold is RANGE-AWARE on purpose. An absolute
        decay requirement is wrong for short sweeps -- a 3-point O(h^2) run
        spanning 2x in h can only ever decay 4x, and a fixed "must decay 4x" rule
        rejects it as floored. The expected decay is (h_coarse/h_fine)^order, so
        that is what it is compared against.
    """
    h = np.asarray(h, float); err = np.asarray(err, float)
    if h.ndim != 1 or h.shape != err.shape or len(h) < 3:
        raise ValueError("need matching 1-D h and err with at least 3 points")
    if np.any(np.diff(h) >= 0):
        raise ValueError("h must be strictly decreasing (coarse to fine)")
    if np.any(err <= 0):
        raise ValueError("err must be positive to fit a log-log slope")

    loc = local_orders(h, err)
    total_decay = float(err[0] / err[-1])
    tail = float(loc[-1])
    fitted = float(np.polyfit(np.log(h), np.log(err), 1)[0])

    if min_total_decay is None:
        min_total_decay = float((h[0] / h[-1]) ** min_tail_order)

    reasons = []
    if total_decay < min_total_decay:
        reasons.append(f"error fell only {total_decay:.2f}x across an h-range of "
                       f"{h[0] / h[-1]:.2f}x (want >= {min_total_decay:.2f}, i.e. "
                       f"at least order {min_tail_order:g})")
    if tail < min_tail_order:
        reasons.append(f"local order at the finest pair is {tail:.2f} "
                       f"(want >= {min_tail_order:g}) -- the curve has gone flat")

    return dict(order=fitted, local=loc, total_decay=total_decay,
                tail_order=tail, floored=bool(reasons), reasons=reasons)


def fit_order(h, err, on_floor="raise", min_tail_order=0.5, min_total_decay=None,
              what="error"):
    """Fitted log-log convergence order, guarded against floors.

    on_floor : 'raise' (default), 'warn', or 'ignore'.

    Use 'raise' for anything that ends up in a paper or a docstring. The floor
    cases this guards against do not look like failures -- they return a
    perfectly ordinary-looking number.
    """
    h = np.asarray(h, float); err = np.asarray(err, float)
    if len(h) < 3:
        # Two points give a slope but no way to tell a rate from a plateau, so
        # the fit is reported UNVERIFIED rather than silently trusted. Callers
        # that only want the error values (e.g. an equal-resolution comparison)
        # are unaffected.
        warnings.warn(
            f"convergence order for {what} fitted from {len(h)} points: too few "
            "to check for a floor, so this slope is unverified -- do not quote it",
            RuntimeWarning, stacklevel=2)
        return float(np.polyfit(np.log(h), np.log(err), 1)[0])

    d = floor_diagnosis(h, err, min_total_decay=min_total_decay,
                        min_tail_order=min_tail_order)
    if d["floored"]:
        msg = (f"refusing to report a convergence order for {what}: "
               + "; ".join(d["reasons"])
               + ". This is the signature of a floored reference -- e.g. "
                 "comparing a finite-domain solve against an infinite-domain "
                 "analytic field. Fix the reference, do not quote the slope "
                 f"({d['order']:.2f}).")
        if on_floor == "raise":
            raise FloorError(msg)
        if on_floor == "warn":
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        elif on_floor != "ignore":
            raise ValueError(f"unknown on_floor={on_floor!r}")
    return d["order"]
