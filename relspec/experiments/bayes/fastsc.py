"""Promoted to relspec.fastsc (it passed its gate — see fastsc_metrics.json).
This shim keeps bench_fastsc.py runnable in place; the implementation lives
in src/relspec/fastsc.py now and is what the platform serves.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'src'))
from relspec.fastsc import fast_sc, ies_full, ies_top_band, selfcheck  # noqa: F401,E402
