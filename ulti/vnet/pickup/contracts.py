"""Contract registry for pickup-net training & inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ulti.eval.dojo import (
    deal_betli, deal_durchmars_colorless, deal_parti, deal_ulti_biased,
)


@dataclass(frozen=True)
class ContractCfg:
    name:        str
    solver:      str       # solver name ('parti' / 'ulti' / 'betli' / 'durchmars')
    dealer:      Callable  # function (seed, alpha) -> BiasedDeal
    has_trump:   bool


CONTRACT_CONFIGS = {
    'betli':     ContractCfg('betli',     'betli',     deal_betli,
                             has_trump=False),
    'durchmars': ContractCfg('durchmars', 'durchmars', deal_durchmars_colorless,
                             has_trump=False),
    'parti':     ContractCfg('parti',     'parti',     deal_parti,
                             has_trump=True),
    'ulti':      ContractCfg('ulti',      'ulti',      deal_ulti_biased,
                             has_trump=True),
}
