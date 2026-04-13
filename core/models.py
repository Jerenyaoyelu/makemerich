from dataclasses import dataclass


@dataclass
class StockSignal:
    symbol: str
    name: str
    theme: str
    theme_strength: float
    sector_linkage: float
    stock_strength: float
    capital_support: float
    risk_tag: str = ""

