"""
vel_noise.py  --  inject the fitted EE-velocity observation noise at TRAINING time.

Reads the model written by fit_vel_noise_model.py on the robot computer

    vel_noise_model.json  ->  sigma_inject [m/s], white, per timestep

and adds it to the velocity channel of the network's input. This is the Spot
paper's observation-noise step: simulation's velocity channel is far cleaner
than the robot's, so a model trained on clean simulation has never been asked to
read a velocity trace through the hardware's noise floor.

WHERE THIS GOES, AND WHY IT MATTERS MORE THAN IT LOOKS
    1. On the INPUT, inside the training loop -- not baked into sim_channels.csv.
       Adding it per batch redraws it every epoch, which is the point: the model
       sees many noise realizations of the same push instead of memorizing one.
    2. BEFORE any velocity filter. On hardware the filter sees already-noisy
       sensor data. Filtering first and adding noise after models a sensor that
       does not exist, and for movmean13 the difference is a factor of ~17 in
       the surviving floor.
    3. In the RIGHT SPACE. sigma is in m/s. If your Dataset hands the model
       standardized tensors, the noise must be divided by that channel's
       training std first, or it arrives 1/std times too large -- typically two
       orders of magnitude. Use VelNoise.apply(..., std=...) for that case and
       the class does the conversion; there is no way for it to detect which
       space a tensor is in, so this is the one thing you must tell it.
    4. TRAINING ONLY by default. Evaluation measures what the model does with
       the data it will meet; real pushes already carry the real noise, and
       adding more on top double-counts.

STANDARDIZATION STATS
    Keep the existing .arm_stats.<variant>.npz sidecars. Recomputing them on
    noisy data would make the noised runs incomparable with the runs you already
    have, and the change is negligible anyway: a channel with std 0.01 becomes
    sqrt(0.01^2 + 0.0014^2) = 0.01010, a 1% shift. The function
    `std_inflation()` below prints the exact number for your sidecar so this is
    a measurement rather than an assertion.

USE
    from vel_noise import VelNoise
    noise = VelNoise.from_json("vel_noise_model.json")       # once, at startup
    ...
    x = noise.apply(x, channel=VEL_IDX, std=vel_std)         # per batch
"""

import os
import json
import argparse

import numpy as np

try:
    import torch
except Exception as _torch_exc:                              # noqa: BLE001
    # Not just ImportError: a half-installed torch raises OSError on a missing
    # shared library, and this module's numpy path works regardless. The reason
    # is kept rather than discarded so a broken install is visible if the
    # tensor path later appears to do nothing.
    torch, TORCH_IMPORT_ERROR = None, _torch_exc
else:
    TORCH_IMPORT_ERROR = None


DEFAULT_JSON = os.environ.get("VEL_NOISE_JSON", "vel_noise_model.json")


# =============================================================================
# THE MODEL
# =============================================================================
class VelNoise:
    """The fitted noise level, plus the one safe way to apply it.

    Deliberately a small object rather than a bare float: the sigma is only
    meaningful together with the space it is expressed in and the fs and band it
    was measured at, and carrying those alongside is what lets `apply` refuse
    the mistakes that otherwise pass silently.
    """

    def __init__(self, sigma, fs=100.0, band=(25.0, 50.0), source="",
                 meta=None, enabled=True):
        self.sigma = float(sigma)
        self.fs = float(fs)
        self.band = (float(band[0]), float(band[1]))
        self.source = source
        self.meta = meta or {}
        self.enabled = bool(enabled) and self.sigma > 0

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_json(cls, path=DEFAULT_JSON, key="sigma_inject", required=True):
        """Load the fitted model.

        `required` defaults to True on purpose. A missing file silently falling
        back to zero noise would produce a run that looks like a noised run,
        is named like one, and is not one -- the worst of the available
        failures, because nothing about the result says so.
        """
        if not os.path.exists(path):
            if required:
                raise FileNotFoundError(
                    f"{path} not found. Copy vel_noise_model.json from the robot "
                    f"computer, point VEL_NOISE_JSON at it, or construct "
                    f"VelNoise(sigma=...) explicitly if you mean to pick a level "
                    f"by hand.")
            print(f"[vel_noise] {path} not found; noise DISABLED")
            return cls(0.0, enabled=False, source=f"{path} (missing)")
        with open(path) as fh:
            d = json.load(fh)
        if key not in d:
            raise KeyError(f"{path} has no '{key}'. Keys present: {sorted(d)}")
        sigma = float(d[key])
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError(f"{path}: {key} = {sigma!r} is not a usable sigma")
        obj = cls(sigma, fs=d.get("fs", 100.0),
                  band=d.get("band_hz", (25.0, 50.0)), source=path, meta=d)
        obj._warn_about_fit()
        return obj

    @classmethod
    def disabled(cls):
        """An inert instance, so the call site needs no `if`."""
        return cls(0.0, enabled=False, source="disabled")

    def _warn_about_fit(self):
        v = self.meta.get("floor_verdict")
        if v == "coloured":
            print(f"[vel_noise] WARNING: the fitted floor is '{v}', not white. "
                  f"White injection cannot reproduce a coloured floor; this "
                  f"sigma is the best single-number approximation to it.")
        elif v == "rising":
            print(f"[vel_noise] note: fitted floor verdict '{v}' -- the real "
                  f"floor tilts up toward Nyquist, so one white sigma is an "
                  f"approximation accurate to a few percent across the band.")

    # -- application ----------------------------------------------------------
    def sigma_in(self, std=None):
        """The sigma to actually add, given the space of the tensor.

        std is that channel's standardization std, in m/s, from the
        .arm_stats.<variant>.npz sidecar. Passing it means "this tensor is
        standardized"; omitting it means "this tensor is in m/s".
        """
        if std is None:
            return self.sigma
        std = float(std)
        if not np.isfinite(std) or std <= 0:
            raise ValueError(
                f"std must be a positive number in m/s; got {std!r}. This is the "
                f"velocity channel's entry in the .arm_stats sidecar, not a "
                f"normalized value.")
        return self.sigma / std

    def apply(self, x, channel=None, std=None, generator=None, inplace=False):
        """x with white noise added to the velocity channel.

        x is a torch Tensor or a numpy array shaped [..., T, C] or [..., C, T];
        `channel` indexes the channel axis, which is inferred as the axis whose
        length is NOT the sequence length when that is unambiguous, and must be
        given as a (axis, index) pair when it is not.

        channel:
            int           -> the last axis is the channel axis
            (axis, index) -> explicit, always correct
            None          -> every element of x is velocity (a single-channel
                             tensor); the whole thing is noised.
        """
        if not self.enabled:
            return x
        sigma = self.sigma_in(std)

        is_torch = torch is not None and torch.is_tensor(x)
        if channel is None:
            noise = self._draw(x, sigma, generator, is_torch)
            return x.add_(noise) if inplace else x + noise

        axis, idx = channel if isinstance(channel, (tuple, list)) else (-1, channel)
        out = x if inplace else (x.clone() if is_torch else x.copy())
        sl = [slice(None)] * out.ndim
        sl[axis] = idx
        sl = tuple(sl)
        sub = out[sl]
        out[sl] = sub + self._draw(sub, sigma, generator, is_torch)
        return out

    @staticmethod
    def _draw(ref, sigma, generator, is_torch):
        if is_torch:
            # torch.randn_like draws from the global generator, which the
            # DataLoader reseeds per worker and per epoch -- so workers do not
            # hand out duplicate noise, which a numpy RNG created in __init__
            # would. Pass `generator` only to pin a run for a reproducibility
            # check.
            if generator is not None:
                n = torch.randn(ref.shape, generator=generator,
                                dtype=ref.dtype, device=ref.device)
            else:
                n = torch.randn_like(ref)
            return n * sigma
        # The generator is honoured here too. It used to be ignored on this
        # branch, which made a caller that passed a seeded default_rng believe
        # it had pinned the draw while every call still came from the global
        # RNG -- silent, and exactly wrong for anything that has to be
        # reproducible, such as a figure. np.random (the module) and a
        # Generator share the .normal signature, so one name covers both.
        rng = generator if generator is not None else np.random
        return rng.normal(0.0, sigma, size=np.shape(ref)).astype(
            np.asarray(ref).dtype, copy=False)

    # -- reporting ------------------------------------------------------------
    def tag(self):
        """A short token for run names, so a noised run is never mistaken for
        a clean one in a results directory six weeks from now."""
        return "noise_off" if not self.enabled else f"vn{self.sigma * 1e6:.0f}u"

    def describe(self, std=None):
        if not self.enabled:
            return "[vel_noise] DISABLED -- inputs are unmodified"
        s = (f"[vel_noise] sigma {self.sigma:.6f} m/s, white, per timestep "
             f"(fs {self.fs:g} Hz, band {self.band[0]:g}-{self.band[1]:g} Hz)")
        if std is not None:
            s += (f"\n[vel_noise] applied in STANDARDIZED space: "
                  f"{self.sigma:.6f} / {float(std):.6f} = "
                  f"{self.sigma_in(std):.6f} z-units")
        else:
            s += "\n[vel_noise] applied in RAW space (m/s)"
        s += f"\n[vel_noise] from {self.source}, tag {self.tag()}"
        return s

    def __repr__(self):
        return (f"VelNoise(sigma={self.sigma:.6g}, enabled={self.enabled}, "
                f"source={self.source!r})")


# =============================================================================
# SIDECAR HELPERS
# =============================================================================
def vel_std_from_sidecar(path, channel="input_vel"):
    """(index, std) for the velocity channel in a .arm_stats.<variant>.npz.

    Returns the index as well, because that index is also the channel position
    the model's input tensor uses -- the two come from the same channel list, so
    reading them together is what keeps them consistent.
    """
    d = np.load(path, allow_pickle=True)
    try:
        chans = [str(c) for c in np.atleast_1d(d["channels"]).tolist()]
    except KeyError:
        raise KeyError(f"{path} has no 'channels' array; cannot locate "
                       f"{channel!r}.")
    if channel not in chans:
        raise KeyError(f"{channel!r} is not in {path}: {chans}")
    k = chans.index(channel)
    std = float(np.asarray(d["std"], float).reshape(-1)[k])
    if "log_transform" in d:
        logs = np.atleast_1d(np.asarray(d["log_transform"])).astype(bool)
        if k < len(logs) and logs[k]:
            raise ValueError(
                f"{channel!r} is log-transformed in {path}. Additive white noise "
                f"in log space is multiplicative in m/s and this sigma does not "
                f"describe it -- inject BEFORE the log, in raw space.")
    return k, std


def std_inflation(sigma, std):
    """How much the channel's std grows once the noise is added.

    The number that decides whether the existing sidecars can be reused.
    """
    new = float(np.sqrt(float(std) ** 2 + float(sigma) ** 2))
    return new, 100.0 * (new / float(std) - 1.0)


# =============================================================================
# SELF TEST  --  does the injected noise re-measure at the level requested?
# =============================================================================
def _psd_floor(W, fs, band):
    """The same estimator fit_vel_noise_model.py used: linear detrend, Hann,
    one-sided PSD, band median, sqrt(S0 fs / 2)."""
    N = W.shape[1]
    tt = np.arange(N)
    A = np.vstack([tt, np.ones(N)]).T
    coef, *_ = np.linalg.lstsq(A, W.T, rcond=None)
    Xd = W - (A @ coef).T
    w = np.hanning(N)
    F = np.fft.rfft(Xd * w, axis=1)
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    S = 2.0 * np.abs(F) ** 2 / (fs * np.sum(w ** 2))
    S[:, 0] /= 2.0
    if N % 2 == 0:
        S[:, -1] /= 2.0
    S = S.mean(axis=0)
    ok = (f >= band[0]) & (f <= band[1]) & (f > 0) & (f < f[-1])
    return float(np.sqrt(np.median(S[ok]) * fs / 2.0))


def selftest(sigma=0.0014342774, fs=100.0, band=(25.0, 50.0), seq=60, n=400):
    """Inject into a clean synthetic push, then measure the floor back out.

    This is the only check that actually closes the loop: it proves that what
    this module adds is what fit_vel_noise_model.py would measure, in both
    spaces, rather than that the arithmetic looks right.
    """
    print("=" * 76)
    print(" VEL NOISE SELF TEST")
    print("=" * 76)
    rng = np.random.default_rng(0)
    t = np.arange(seq)[None, :]
    clean = 0.08 * (1.0 - np.exp(-t / 12.0)) + np.zeros((n, seq))

    vn = VelNoise(sigma, fs=fs, band=band, source="selftest")
    print(f"  requested sigma      {sigma:.6f} m/s")
    print(f"  clean signal floor   {_psd_floor(clean, fs, band):.3e}  "
          f"(a smooth ramp has no content in the band)")

    # raw space
    noisy = vn.apply(clean + 0.0, channel=None)
    got = _psd_floor(noisy, fs, band)
    print(f"\n  RAW space      measured {got:.6f}   ratio {got / sigma:.3f}")

    # standardized space: standardize, inject with std, destandardize, measure
    mu, sd = float(clean.mean()), 0.0123
    z = (clean - mu) / sd
    zn = vn.apply(z, channel=None, std=sd)
    back = zn * sd + mu
    got_z = _psd_floor(back, fs, band)
    print(f"  z space (std {sd:g})  measured {got_z:.6f}   "
          f"ratio {got_z / sigma:.3f}")
    print(f"     sigma in z-units was {vn.sigma_in(sd):.6f}")

    # The two mistakes this module exists to prevent. Both are silent: neither
    # raises, neither looks wrong in a loss curve, and both waste a full retrain.
    too_small = _psd_floor(vn.apply(z, channel=None) * sd + mu, fs, band)
    too_big = _psd_floor(clean + (vn.sigma_in(sd) * rng.standard_normal(clean.shape)),
                         fs, band)
    print(f"\n  what goes wrong if the space is mixed up (std {sd:g}):")
    print(f"     raw sigma into a STANDARDIZED tensor -> {too_small:.6f}, "
          f"{sigma / max(too_small, 1e-12):.0f}x too SMALL")
    print(f"        the run looks noised and is effectively clean.")
    print(f"     z sigma into a RAW tensor            -> {too_big:.6f}, "
          f"{too_big / sigma:.0f}x too LARGE")
    print(f"        the velocity channel is buried and the model learns to "
          f"ignore it.")

    # channel indexing on a realistic [B, T, C] batch
    C = 5
    batch = np.zeros((n, seq, C))
    batch[:, :, 0] = clean
    out = vn.apply(batch, channel=0)
    moved = [float(np.abs(out[:, :, c] - batch[:, :, c]).max()) for c in range(C)]
    print(f"\n  channel targeting on [B, T, C={C}]: channel 0 moved by "
          f"{moved[0]:.2e},")
    print(f"     channels 1..{C - 1} moved by {max(moved[1:]):.2e} "
          f"(must be 0.00e+00)")

    ok = (abs(got / sigma - 1) < 0.08 and abs(got_z / sigma - 1) < 0.08
          and max(moved[1:]) == 0.0)
    print(f"\n  {'PASS' if ok else 'FAIL'}: injection re-measures at the "
          f"requested level in both spaces,")
    print(f"  and touches only the channel it was told to.")
    print("=" * 76)
    return ok


# =============================================================================
# CLI  --  validate the model and print the integration you need
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--sidecar", default=None,
                    help=".arm_stats.<variant>.npz, to report the z-space sigma "
                         "and whether the existing stats can be reused")
    ap.add_argument("--channel", default="input_vel")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(0 if selftest() else 1)

    vn = VelNoise.from_json(a.json)
    print(vn.describe())
    print(f"\n  fitted on   {vn.meta.get('n_real', '?')} real push(es)")
    print(f"  sigma_real  {vn.meta.get('sigma_real', float('nan')):.6f}")
    print(f"  sigma_sim   {vn.meta.get('sigma_sim', float('nan')):.6f}")

    if a.sidecar:
        k, std = vel_std_from_sidecar(a.sidecar, a.channel)
        new, pct = std_inflation(vn.sigma, std)
        print(f"\n  sidecar {os.path.basename(a.sidecar)}")
        print(f"     {a.channel} is channel index {k}, std {std:.6f} m/s")
        print(f"     sigma in z-units: {vn.sigma_in(std):.6f}")
        print(f"     adding the noise moves that std to {new:.6f} "
              f"({pct:+.2f}%)")
        if abs(pct) < 2.0:
            print(f"     Under 2%: KEEP the existing sidecar, so the noised "
                  f"runs stay comparable with the clean ones.")
        else:
            print(f"     Over 2%: large enough that re-deriving the sidecar on "
                  f"noised data is worth considering -- but then the noised "
                  f"runs are no longer directly comparable with the clean ones.")

    print(f"\n  In the training loop, on the INPUT, before any velocity filter:")
    print(f"     x = noise.apply(x, channel=VEL_IDX, std=VEL_STD)")
    print(f"  and name the run with  {vn.tag()}  so it cannot be confused "
          f"with a clean one.")


if __name__ == "__main__":
    main()