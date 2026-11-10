r"""Simulation-based inference on toy exoplanet emission spectra.

Forward models (6, 30 and 50 parameters) in :mod:`exomini.exo6`, :mod:`exomini.exo30` and :mod:`exomini.exo50`,
wrapped by :class:`Model`; neural posterior estimation, truncation and importance sampling in
:mod:`exomini.inference`.
"""

from .model import MODELS, Model
