from collections import OrderedDict

FONTS = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
MONO = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'

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

# =============================================================================
# THEMES — each carries surface / text palette for full theming
# =============================================================================

THEMES = {
    'Terminal': {
        'mode': 'dark',
        'pos': '#4ade80', 'neg': '#f59e0b',
        'zone_hi': '#4ade80', 'zone_amid': '#86efac', 'zone_bmid': '#fbbf24', 'zone_lo': '#f59e0b',
        'str_up': '#4ade80', 'str_dn': '#f59e0b', 'pull': '#fbbf24', 'bnce': '#93c5fd',
        'long': '#4ade80', 'short': '#f59e0b',
        'bg': '#1e1e1e', 'bg2': '#0a0f1a', 'bg3': '#0f172a',
        'border': '#1e293b', 'text': '#e2e8f0', 'text2': '#94a3b8', 'muted': '#475569',
        'accent': '#4ade80',
        'plot_bg': '#121212', 'grid': '#1f1f1f', 'axis_line': '#2a2a2a', 'tick': '#888888',
    },
    'Neon': {
        'mode': 'dark',
        'pos': '#38bdf8', 'neg': '#fb7185',
        'zone_hi': '#38bdf8', 'zone_amid': '#7dd3fc', 'zone_bmid': '#fda4af', 'zone_lo': '#fb7185',
        'str_up': '#38bdf8', 'str_dn': '#fb7185', 'pull': '#fbbf24', 'bnce': '#a78bfa',
        'long': '#38bdf8', 'short': '#fb7185',
        'bg': '#0c0c14', 'bg2': '#08081a', 'bg3': '#10102a',
        'border': '#1a1a3e', 'text': '#e2e8f0', 'text2': '#8888bb', 'muted': '#555580',
        'accent': '#38bdf8',
        'plot_bg': '#0c0c14', 'grid': '#151530', 'axis_line': '#1a1a3e', 'tick': '#8888bb',
    },
    'Light': {
        'mode': 'light',
        'pos': '#16a34a', 'neg': '#dc2626',
        'zone_hi': '#16a34a', 'zone_amid': '#4ade80', 'zone_bmid': '#fca5a5', 'zone_lo': '#dc2626',
        'str_up': '#16a34a', 'str_dn': '#dc2626', 'pull': '#d97706', 'bnce': '#2563eb',
        'long': '#16a34a', 'short': '#dc2626',
        'bg': '#ffffff', 'bg2': '#f8fafc', 'bg3': '#f1f5f9',
        'border': '#e2e8f0', 'text': '#1e293b', 'text2': '#64748b', 'muted': '#94a3b8',
        'accent': '#2563eb',
        'plot_bg': '#ffffff', 'grid': '#f1f5f9', 'axis_line': '#e2e8f0', 'tick': '#64748b',
    },
}

CHART_CONFIGS = [
    ('Session', '5m', 'Session High/Low', 'session'),
    ('4H', '1h', 'Week High/Low', 'week'),
    ('Daily', '1d', 'Month High/Low', 'month'),
    ('Weekly', '1wk', 'Year High/Low', 'year'),
]

STATUS_LABELS = {
    'above_high': '▲ ABOVE HIGH', 'above_mid': '● ABOVE MID',
    'below_mid': '● BELOW MID', 'below_low': '▼ BELOW LOW',
}

def clean_symbol(sym):
    return sym.replace('=F', '').replace('=X', '').replace('.SI', '')
