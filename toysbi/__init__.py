r"""Simulation-based inference on toy exoplanet emission spectra.

Forward models (6, 30 and 50 parameters) in :mod:`toysbi.exo6`, :mod:`toysbi.exo30` and :mod:`toysbi.exo50`,
wrapped by :class:`Model`; neural posterior estimation, truncation and importance sampling in
:mod:`toysbi.inference`.
"""

from .model import MODELS, Model
