import streamlit as st
from collections import OrderedDict

FONTS = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'

# =============================================================================
# SYMBOL GROUPS
# =============================================================================

FUTURES_GROUPS = OrderedDict([
    ('Futures',   ['ES=F', 'NQ=F', 'GC=F', 'SI=F', 'CL=F', 'NG=F', 'ZC=F', 'ZS=F', 'BTC=F', 'ETH=F', 'SB=F', 'KC=F']),
    ('Indices',   ['ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'NKD=F']),
    ('Rates',     ['ZB=F', 'ZN=F', 'ZF=F', 'ZT=F']),
    ('FX',        ['6E=F', '6J=F', '6B=F', '6A=F', 'USDSGD=X']),
    ('Crypto',    ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']),
    ('Energy',    ['CL=F', 'NG=F', 'RB=F', 'HO=F']),
    ('Metals',    ['GC=F', 'SI=F', 'PL=F', 'HG=F']),
    ('Grains',    ['ZC=F', 'ZS=F', 'ZW=F', 'ZM=F']),
    ('Softs',     ['SB=F', 'KC=F', 'CC=F', 'CT=F']),
    ('Singapore', ['ES3.SI', 'S68.SI', 'MBH.SI', 'MMS.SI']),
    ('US Sectors',['XLB', 'XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLU', 'XLRE', 'SPY']),
    ('Countries', ['EWA', 'EWZ', 'EWC', 'GXC', 'EWQ', 'EWG', 'EWH', 'PIN', 'EWI', 'EWJ', 'EWM', 'EWW', 'EWS', 'EWY', 'EWP', 'EWT', 'EWU', 'VNM', 'KSA', 'ARGT']),
    ('Macro',     ['DBC', 'USO', 'GLD', 'SLV', 'CPER', 'BIL', 'HYG', 'LQD', 'TLT', 'BND', 'EMB', 'EEM', 'SPY', 'BTC-USD', 'ETH-USD']),
    ('Core 5',    ['IAU', 'VOO', 'VTI', 'SHV', 'IBIT']),
    ('Exchanges', ['ICE', 'NDAQ', 'CME', 'CBOE', 'X.TO', 'LSEG.L', 'DB1.DE', 'ENX.PA', '8697.T', '0388.HK', 'ASX.AX', 'S68.SI']),
])

SYMBOL_NAMES = {
    'ES=F': 'E-mini S&P 500', 'NQ=F': 'E-mini Nasdaq 100', 'YM=F': 'E-mini Dow',
    'RTY=F': 'E-mini Russell 2000', 'NKD=F': 'Nikkei 225',
    'ZB=F': '30Y T-Bond', 'ZN=F': '10Y T-Note', 'ZF=F': '5Y T-Note', 'ZT=F': '2Y T-Note',
    'GC=F': 'Gold', 'SI=F': 'Silver', 'PL=F': 'Platinum', 'HG=F': 'Copper',
    'CL=F': 'Crude Oil WTI', 'NG=F': 'Natural Gas', 'RB=F': 'RBOB Gasoline', 'HO=F': 'Heating Oil',
    'ZS=F': 'Soybeans', 'ZC=F': 'Corn', 'ZW=F': 'Wheat', 'ZM=F': 'Soybean Meal',
    'SB=F': 'Sugar', 'KC=F': 'Coffee', 'CC=F': 'Cocoa', 'CT=F': 'Cotton',
    'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum', 'SOL-USD': 'Solana', 'XRP-USD': 'XRP',
    'BTC=F': 'Bitcoin Futures', 'ETH=F': 'Ethereum Futures',
    '6E=F': 'Euro FX', '6J=F': 'Japanese Yen', '6B=F': 'British Pound', '6A=F': 'Australian Dollar',
    'USDSGD=X': 'USD/SGD',
    'ES3.SI': 'STI ETF', 'S68.SI': 'SGX', 'MBH.SI': 'Amova IG Bond', 'MMS.SI': 'SGD Money Mkt',
    'XLB': 'Materials', 'XLC': 'Comms', 'XLY': 'Cons Disc', 'XLP': 'Cons Staples',
    'XLE': 'Energy', 'XLF': 'Financials', 'XLV': 'Healthcare', 'XLI': 'Industrials',
    'XLK': 'Technology', 'XLU': 'Utilities', 'XLRE': 'Real Estate', 'SPY': 'S&P 500',
    'EWA': 'Australia', 'EWZ': 'Brazil', 'EWC': 'Canada', 'GXC': 'China',
    'EWQ': 'France', 'EWG': 'Germany', 'EWH': 'Hong Kong', 'PIN': 'India',
    'EWI': 'Italy', 'EWJ': 'Japan', 'EWM': 'Malaysia', 'EWW': 'Mexico',
    'EWS': 'Singapore', 'EWY': 'South Korea', 'EWP': 'Spain', 'EWT': 'Taiwan',
    'EWU': 'UK', 'VNM': 'Vietnam', 'KSA': 'Saudi Arabia', 'ARGT': 'Argentina',
    'DBC': 'Commodities', 'USO': 'Oil ETF', 'GLD': 'Gold ETF', 'SLV': 'Silver ETF',
    'CPER': 'Copper ETF', 'BIL': 'T-Bills', 'HYG': 'High Yield', 'LQD': 'IG Corp',
    'TLT': '20Y+ Treasury', 'BND': 'Total Bond', 'EMB': 'EM Bonds', 'EEM': 'EM Equity',
    'ICE': 'ICE', 'NDAQ': 'Nasdaq Inc', 'CME': 'CME Group', 'CBOE': 'Cboe Global',
    'X.TO': 'TMX Group', 'LSEG.L': 'LSEG', 'DB1.DE': 'Deutsche Börse',
    'ENX.PA': 'Euronext', '8697.T': 'JPX', '0388.HK': 'HKEX', 'ASX.AX': 'ASX Ltd',
    'IAU': 'iShares Gold', 'VOO': 'Vanguard S&P 500', 'VTI': 'Vanguard Total Mkt',
    'SHV': 'Short Treasury', 'IBIT': 'iShares Bitcoin',
}

# Google News search terms per symbol (charts tab news panel)
NEWS_TERMS = {
    'ES=F': 'S&P 500', 'NQ=F': 'Nasdaq 100', 'YM=F': 'Dow Jones', 'RTY=F': 'Russell 2000',
    'NKD=F': 'Nikkei 225', 'ZB=F': 'US treasury bonds', 'ZN=F': '10 year treasury yield',
    'ZF=F': '5 year treasury', 'ZT=F': '2 year treasury', '6E=F': 'EUR USD euro',
    '6J=F': 'USD JPY yen', '6B=F': 'GBP USD pound', '6A=F': 'AUD USD australian',
    'USDSGD=X': 'USD SGD Singapore dollar', 'CL=F': 'crude oil', 'NG=F': 'natural gas',
    'GC=F': 'gold price', 'SI=F': 'silver price', 'PL=F': 'platinum', 'HG=F': 'copper price',
    'ZS=F': 'soybean', 'ZC=F': 'corn grain', 'ZW=F': 'wheat', 'ZM=F': 'soybean meal',
    'SB=F': 'sugar commodity', 'KC=F': 'coffee arabica', 'CC=F': 'cocoa', 'CT=F': 'cotton commodity',
    'BTC-USD': 'bitcoin', 'ETH-USD': 'ethereum', 'SOL-USD': 'solana crypto', 'XRP-USD': 'XRP ripple',
    'BTC=F': 'bitcoin futures CME', 'ETH=F': 'ethereum futures CME',
}

# =============================================================================
# THEMES
# =============================================================================

THEMES = {
    'Dark': {
        'mode': 'dark',
        'pos': '#4ade80', 'neg': '#f59e0b',
        'zone_hi': '#4ade80', 'zone_amid': '#86efac', 'zone_bmid': '#fbbf24', 'zone_lo': '#f59e0b',
        'str_up': '#4ade80', 'str_dn': '#f59e0b', 'pull': '#fbbf24', 'bnce': '#93c5fd',
        'long': '#4ade80', 'short': '#f59e0b',
        'bg': '#0f1117', 'bg2': '#0a0f1a', 'bg3': '#0f172a',
        'border': '#1e293b', 'text': '#e2e8f0', 'text2': '#94a3b8', 'muted': '#475569',
        'accent': '#4ade80',
        'plot_bg': '#0f1117', 'grid': '#1a1f2e', 'axis_line': '#2a2a2a', 'tick': '#888888',
    },
}

CHART_CONFIGS = [
    ('Day (15m)', '15m', 'Session High/Low', 'session'),
    ('Weekly (4H)', '1h', 'Week High/Low', 'week'),
    ('Monthly (Daily)', '1d', 'Month High/Low', 'month'),
    ('Year (Weekly)', '1wk', 'Year High/Low', 'year'),
]

STATUS_LABELS = {
    'above_high': '▲ ABOVE HIGH', 'above_mid': '● ABOVE MID',
    'below_mid': '● BELOW MID', 'below_low': '▼ BELOW LOW',
}

# =============================================================================
# SHARED HELPERS
# =============================================================================

def clean_symbol(sym):
    return sym.replace('=F', '').replace('=X', '').replace('.SI', '')

def sym_name(sym):
    """Friendly name: SYMBOL_NAMES lookup with clean_symbol fallback."""
    return SYMBOL_NAMES.get(sym, clean_symbol(sym))

def get_theme():
    """Single source of truth for current theme — used by all tabs."""
    name = st.session_state.get('theme', 'Dark')
    return THEMES.get(name, THEMES['Dark'])

def surface():
    """Derived surface palette for HTML rendering — used by all tabs."""
    t = get_theme()
    is_light = t.get('mode') == 'light'
    bg  = t.get('bg', '#1e1e1e');  bg2 = t.get('bg2', '#0a0f1a')
    bg3 = t.get('bg3', '#0f172a'); bdr = t.get('border', '#1e293b')
    txt = t.get('text', '#e2e8f0'); txt2 = t.get('text2', '#94a3b8')
    muted = t.get('muted', '#475569')
    if is_light:
        return dict(bg=bg, bg2=bg2, bg3=bg3, card=bg2,
            border=bdr, text=txt, text2=txt2, muted=muted,
            off_dot='#d1d5db', off_name='#9ca3af', link='#334155',
            bar_bg=bdr, row_alt=bg3, hm_txt=txt)
    return dict(bg=bg, bg2=bg2, bg3=bg3, card=bg3,
        border=bdr, text=txt, text2=txt2, muted=muted,
        off_dot='#3a3a3a', off_name='#4a5568', link='#c9d1d9',
        bar_bg=bg3, row_alt='#0d1321', hm_txt=txt)

def zone_colors():
    t = get_theme()
    return {'above_high': t['zone_hi'], 'above_mid': t['zone_amid'],
            'below_mid': t['zone_bmid'], 'below_low': t['zone_lo']}
