import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from collections import OrderedDict
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import logging
import feedparser
from urllib.parse import quote

# =============================================================================
# SETUP
# =============================================================================

logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

st.set_page_config(page_title="Chart Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# Design tokens
FONTS = 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
MONO = '"JetBrains Mono", "SF Mono", "Fira Code", Consolas, monospace'
BG      = '#0b0f19'
CARD    = '#111827'
SURFACE = '#1f2937'
BORDER  = '#2d3748'
ACCENT  = '#3b82f6'
MUTED   = '#64748b'
DIM     = '#475569'
TEXT    = '#e2e8f0'

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {{ background: {BG}; }}
    header[data-testid="stHeader"] {{ background: {BG}; }}
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding: 0.5rem 1rem 0 1rem; max-width: 100%; }}
    div[data-testid="stHorizontalBlock"] {{ gap: 0.25rem; }}
    div[data-testid="stMarkdownContainer"] p {{ margin-bottom: 0; }}

    /* Selectbox */
    .stSelectbox label {{
        color: {MUTED} !important; font-size: 9px !important; font-weight: 600 !important;
        text-transform: uppercase !important; letter-spacing: 0.1em !important;
        font-family: {FONTS} !important; margin-bottom: 2px !important;
    }}
    .stSelectbox [data-baseweb="select"] > div {{
        background: {CARD} !important; border: 1px solid {BORDER} !important;
        border-radius: 6px !important; min-height: 34px !important;
        transition: border-color 0.15s ease !important;
    }}
    .stSelectbox [data-baseweb="select"] > div:hover {{ border-color: {ACCENT} !important; }}
    .stSelectbox [data-baseweb="select"] > div > div {{
        color: {TEXT} !important; font-size: 12px !important; font-family: {FONTS} !important;
    }}
    .stSelectbox svg {{ fill: {MUTED} !important; }}
    [data-baseweb="popover"], [data-baseweb="menu"] {{
        background: {CARD} !important; border: 1px solid {BORDER} !important; border-radius: 8px !important;
    }}
    [role="option"] {{ color: #cbd5e1 !important; font-size: 12px !important; }}
    [role="option"]:hover {{ background: #1e3a5f !important; }}
    [aria-selected="true"][role="option"] {{ background: #1e3a5f !important; }}

    /* Buttons */
    .stButton > button {{
        font-family: {FONTS} !important; font-weight: 600 !important;
        border-radius: 6px !important; font-size: 11px !important;
        padding: 4px 12px !important; min-height: 30px !important;
        transition: all 0.15s ease !important; letter-spacing: 0.02em !important;
    }}
    button[kind="secondary"] {{
        background: {CARD} !important; color: #94a3b8 !important; border: 1px solid {BORDER} !important;
    }}
    button[kind="secondary"]:hover {{
        background: {SURFACE} !important; color: {TEXT} !important; border-color: #475569 !important;
    }}
    button[kind="primary"] {{
        background: linear-gradient(135deg, #1e40af, #3b82f6) !important;
        color: white !important; border: 1px solid #3b82f6 !important;
        box-shadow: 0 0 12px rgba(59,130,246,0.15) !important;
    }}

    /* Scanner */
    .scanner-wrap {{
        overflow-x: auto; -webkit-overflow-scrolling: touch;
        border: 1px solid {BORDER}; border-radius: 8px; background: {CARD};
    }}
    .scanner-wrap table {{ border-collapse: collapse; width: 100%; }}
    .scanner-wrap tr:hover {{ background: rgba(59,130,246,0.05) !important; }}

    /* Panels */
    .panel {{ border: 1px solid {BORDER}; border-radius: 8px; overflow: hidden; background: {CARD}; margin-bottom: 8px; }}
    .panel-hdr {{
        padding: 8px 12px; background: linear-gradient(135deg, {SURFACE}, {CARD});
        border-bottom: 1px solid {BORDER};
        display: flex; justify-content: space-between; align-items: center;
    }}

    /* Mobile */
    @media (max-width: 768px) {{
        .block-container {{ padding: 0.25rem 0.5rem 0 0.5rem !important; }}
        .stButton > button {{ font-size: 9px !important; padding: 2px 6px !important; min-height: 26px !important; }}
    }}
    @media (max-width: 480px) {{
        .stButton > button {{ font-size: 8px !important; padding: 1px 4px !important; min-height: 22px !important; }}
    }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA: SYMBOLS, THEMES, NAMES
# =============================================================================

FUTURES_GROUPS = OrderedDict([
    ('Futures',   ['ES=F','NQ=F','GC=F','SI=F','CL=F','NG=F','ZC=F','ZS=F','BTC=F','ETH=F','SB=F','KC=F']),
    ('Indices',   ['ES=F','NQ=F','YM=F','RTY=F','NKD=F']),
    ('Rates',     ['ZB=F','ZN=F','ZF=F','ZT=F']),
    ('FX',        ['6E=F','6J=F','6B=F','6A=F','USDSGD=X']),
    ('Crypto',    ['BTC-USD','ETH-USD','SOL-USD','XRP-USD']),
    ('Energy',    ['CL=F','NG=F','RB=F','HO=F']),
    ('Metals',    ['GC=F','SI=F','PL=F','HG=F']),
    ('Grains',    ['ZC=F','ZS=F','ZW=F','ZM=F']),
    ('Softs',     ['SB=F','KC=F','CC=F','CT=F']),
    ('Singapore', ['ES3.SI','S68.SI','MBH.SI','MMS.SI']),
    ('US Sectors',['XLB','XLC','XLY','XLP','XLE','XLF','XLV','XLI','XLK','XLU','XLRE','SPY']),
    ('Countries', ['EWA','EWZ','EWC','GXC','EWQ','EWG','EWH','PIN','EWI','EWJ','EWM','EWW','EWS','EWY','EWP','EWT','EWU','VNM','KSA','ARGT']),
    ('Macro',     ['DBC','USO','GLD','SLV','CPER','BIL','HYG','LQD','TLT','BND','EMB','EEM','SPY','BTC-USD','ETH-USD']),
    ('Core 5',    ['IAU','VOO','VTI','SHV','IBIT']),
    ('Exchanges', ['ICE','NDAQ','CME','CBOE','X.TO','LSEG.L','DB1.DE','ENX.PA','8697.T','0388.HK','ASX.AX','S68.SI']),
])

CHART_CONFIGS = [
    ('Session','5m','Session High/Low','session'),
    ('4H','1h','Week High/Low','week'),
    ('Daily','1d','Month High/Low','month'),
    ('Weekly','1wk','Year High/Low','year'),
]

THEMES = {
    'Blue / Rose': {'pos':'#60a5fa','neg':'#fb7185','zone_hi':'#60a5fa','zone_amid':'#93c5fd','zone_bmid':'#fda4af','zone_lo':'#fb7185','str_up':'#60a5fa','str_dn':'#fb7185','pull':'#fbbf24','bnce':'#4ade80'},
    'Emerald / Amber': {'pos':'#4ade80','neg':'#f59e0b','zone_hi':'#4ade80','zone_amid':'#86efac','zone_bmid':'#fbbf24','zone_lo':'#f59e0b','str_up':'#4ade80','str_dn':'#f59e0b','pull':'#fbbf24','bnce':'#93c5fd'},
    'Cyan / Red': {'pos':'#22d3ee','neg':'#f87171','zone_hi':'#22d3ee','zone_amid':'#67e8f9','zone_bmid':'#fca5a5','zone_lo':'#f87171','str_up':'#22d3ee','str_dn':'#f87171','pull':'#fbbf24','bnce':'#a78bfa'},
    'Teal / Coral': {'pos':'#2dd4bf','neg':'#fb923c','zone_hi':'#2dd4bf','zone_amid':'#5eead4','zone_bmid':'#fdba74','zone_lo':'#fb923c','str_up':'#2dd4bf','str_dn':'#fb923c','pull':'#fbbf24','bnce':'#93c5fd'},
    'Indigo / Gold': {'pos':'#818cf8','neg':'#fbbf24','zone_hi':'#818cf8','zone_amid':'#a5b4fc','zone_bmid':'#fde68a','zone_lo':'#fbbf24','str_up':'#818cf8','str_dn':'#fbbf24','pull':'#fb923c','bnce':'#4ade80'},
}

STATUS_LABELS = {'above_high':'▲ ABOVE HIGH','above_mid':'● ABOVE MID','below_mid':'● BELOW MID','below_low':'▼ BELOW LOW'}

SYMBOL_NAMES = {
    'ES=F':'E-mini S&P 500','NQ=F':'E-mini Nasdaq 100','YM=F':'E-mini Dow',
    'RTY=F':'E-mini Russell 2000','NKD=F':'Nikkei 225',
    'ZB=F':'30Y T-Bond','ZN=F':'10Y T-Note','ZF=F':'5Y T-Note','ZT=F':'2Y T-Note',
    'GC=F':'Gold','SI=F':'Silver','PL=F':'Platinum','HG=F':'Copper',
    'CL=F':'Crude Oil WTI','NG=F':'Natural Gas','RB=F':'RBOB Gasoline','HO=F':'Heating Oil',
    'ZS=F':'Soybeans','ZC=F':'Corn','ZW=F':'Wheat','ZM=F':'Soybean Meal',
    'SB=F':'Sugar','KC=F':'Coffee','CC=F':'Cocoa','CT=F':'Cotton',
    'BTC-USD':'Bitcoin','ETH-USD':'Ethereum','SOL-USD':'Solana','XRP-USD':'XRP',
    'BTC=F':'Bitcoin Futures','ETH=F':'Ethereum Futures',
    '6E=F':'Euro FX','6J=F':'Japanese Yen','6B=F':'British Pound','6A=F':'Australian Dollar',
    'USDSGD=X':'USD/SGD',
    'ES3.SI':'STI ETF','S68.SI':'SGX','MBH.SI':'Amova IG Bond','MMS.SI':'SGD Money Mkt',
    'XLB':'Materials','XLC':'Comms','XLY':'Cons Disc','XLP':'Cons Staples',
    'XLE':'Energy','XLF':'Financials','XLV':'Healthcare','XLI':'Industrials',
    'XLK':'Technology','XLU':'Utilities','XLRE':'Real Estate','SPY':'S&P 500',
    'EWA':'Australia','EWZ':'Brazil','EWC':'Canada','GXC':'China',
    'EWQ':'France','EWG':'Germany','EWH':'Hong Kong','PIN':'India',
    'EWI':'Italy','EWJ':'Japan','EWM':'Malaysia','EWW':'Mexico',
    'EWS':'Singapore','EWY':'South Korea','EWP':'Spain','EWT':'Taiwan',
    'EWU':'UK','VNM':'Vietnam','KSA':'Saudi Arabia','ARGT':'Argentina',
    'DBC':'Commodities','USO':'Oil ETF','GLD':'Gold ETF','SLV':'Silver ETF',
    'CPER':'Copper ETF','BIL':'T-Bills','HYG':'High Yield','LQD':'IG Corp',
    'TLT':'20Y+ Treasury','BND':'Total Bond','EMB':'EM Bonds','EEM':'EM Equity',
    'ICE':'ICE','NDAQ':'Nasdaq Inc','CME':'CME Group','CBOE':'Cboe Global',
    'X.TO':'TMX Group','LSEG.L':'LSEG','DB1.DE':'Deutsche Börse',
    'ENX.PA':'Euronext','8697.T':'JPX','0388.HK':'HKEX','ASX.AX':'ASX Ltd',
    'IAU':'iShares Gold','VOO':'Vanguard S&P 500','VTI':'Vanguard Total Mkt',
    'SHV':'Short Treasury','IBIT':'iShares Bitcoin',
}

def get_theme():
    return THEMES.get(st.session_state.get('theme','Blue / Rose'), THEMES['Blue / Rose'])

def zone_colors():
    t = get_theme()
    return {'above_high':t['zone_hi'],'above_mid':t['zone_amid'],'below_mid':t['zone_bmid'],'below_low':t['zone_lo']}

def clean_symbol(sym):
    return sym.replace('=F','').replace('=X','').replace('.SI','').replace('-USD','')

# =============================================================================
# HELPERS + PERIOD BOUNDARIES
# =============================================================================

def get_dynamic_period(bt):
    now = pd.Timestamp.now()
    if bt == 'session': return '3d'
    elif bt == 'week': return f'{int(now.weekday()+1+14+3)}d'
    elif bt == 'month': return f'{int(now.day+65+5)}d'
    elif bt == 'year': return '3y'
    return '90d'

@dataclass
class PeriodBoundary:
    idx: int; date: pd.Timestamp; prev_high: float; prev_low: float; prev_close: float

class PeriodBoundaryCalculator:
    @staticmethod
    def get_boundaries(df, bt, symbol=''):
        if df is None or len(df) == 0: return []
        boundaries = []
        rules = {
            'year':    lambda i: df.index[i].year != df.index[i-1].year,
            'month':   lambda i: df.index[i].month != df.index[i-1].month or df.index[i].year != df.index[i-1].year,
            'week':    lambda i: df.index[i].isocalendar()[1] != df.index[i-1].isocalendar()[1] or df.index[i].year != df.index[i-1].year,
            'session': lambda i: df.index[i].date() != df.index[i-1].date()
        }
        masks = {
            'year':    lambda i: df.index.year == df.index[i-1].year,
            'month':   lambda i: (df.index.month == df.index[i-1].month) & (df.index.year == df.index[i-1].year),
            'week':    lambda i: (df.index.map(lambda x: x.isocalendar()[1]) == df.index[i-1].isocalendar()[1]) & (df.index.year == df.index[i-1].year),
            'session': lambda i: df.index.date == df.index[i-1].date()
        }
        if bt not in rules: return boundaries
        for i in range(1, len(df)):
            if rules[bt](i):
                prev = df.loc[masks[bt](i)]
                if len(prev) > 0:
                    boundaries.append(PeriodBoundary(idx=i, date=df.index[i],
                        prev_high=prev['High'].max(), prev_low=prev['Low'].min(), prev_close=prev['Close'].iloc[-1]))
        return boundaries

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1: return np.nan
    delta = closes.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta).where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else np.nan

# =============================================================================
# METRICS FETCHER
# =============================================================================

@dataclass
class FuturesMetrics:
    symbol: str; price: float; change_day: float; change_wtd: float
    change_mtd: float; change_ytd: float; timestamp: datetime
    lag_minutes: float; decimals: int
    hist_vol: float = np.nan; day_sharpe: float = np.nan
    wtd_sharpe: float = np.nan; mtd_sharpe: float = np.nan; ytd_sharpe: float = np.nan
    day_status: str = ''; week_status: str = ''; month_status: str = ''; year_status: str = ''
    current_dd: float = np.nan
    day_reversal: bool = False; week_reversal: bool = False
    month_reversal: bool = False; year_reversal: bool = False

class FuturesDataFetcher:
    def __init__(self, symbol):
        self.symbol = symbol; self.ticker = yf.Ticker(symbol)
        self.est = pytz.timezone('US/Eastern')
        self.dec = 4 if symbol.endswith('=X') else 2

    def fetch(self):
        try:
            hist = self.ticker.history(period='1y')
            if hist.empty: return None
            hi = self.ticker.history(period='1d', interval='1m')
            if not hi.empty:
                cp = hi['Close'].iloc[-1]; do = hi['Open'].iloc[0]
                dc = ((cp-do)/do)*100; lt = hi.index[-1]
            else:
                cp = hist['Close'].iloc[-1]; do = hist['Open'].iloc[-1]
                dc = ((cp-do)/do)*100; lt = hist.index[-1]
            now_est = datetime.now(self.est)
            try: lag = (now_est - lt.tz_convert(self.est)).total_seconds()/60
            except: lag = 0
            wtd, mtd, ytd = self._returns(hist, cp)
            return FuturesMetrics(
                symbol=self.symbol, price=round(cp,self.dec), change_day=round(dc,2),
                change_wtd=round(wtd,2) if not np.isnan(wtd) else np.nan,
                change_mtd=round(mtd,2) if not np.isnan(mtd) else np.nan,
                change_ytd=round(ytd,2) if not np.isnan(ytd) else np.nan,
                timestamp=lt, lag_minutes=round(lag,0), decimals=self.dec,
                hist_vol=self._hv(hist), current_dd=self._dd(hist,cp),
                day_sharpe=self._intra_sharpe(hi) if not hi.empty else np.nan,
                wtd_sharpe=self._psharpe(hist,'wtd'), mtd_sharpe=self._psharpe(hist,'mtd'),
                ytd_sharpe=self._ytd_sharpe(hist,cp),
                day_status=self._status(hist,cp,'session'), week_status=self._status(hist,cp,'week'),
                month_status=self._status(hist,cp,'month'), year_status=self._status(hist,cp,'year'),
                day_reversal=self._rev(hist,'session'), week_reversal=self._rev(hist,'week'),
                month_reversal=self._rev(hist,'month'), year_reversal=self._rev(hist,'year'))
        except: return None

    def _hv(self, h):
        try:
            if len(h) < 20: return np.nan
            return round(h['Close'].pct_change().dropna().std()*np.sqrt(252)*100, 1)
        except: return np.nan

    def _dd(self, h, cp):
        try:
            if len(h) < 2: return np.nan
            pk = h['High'].max()
            return round(((cp-pk)/pk)*100, 2) if pk != 0 else np.nan
        except: return np.nan

    def _ytd_sharpe(self, h, cp):
        try:
            yh = h[h.index.map(lambda x: x.date()) >= pd.Timestamp.now().replace(month=1,day=1).date()]
            if len(yh) < 10: return np.nan
            r = yh['Close'].pct_change().dropna()
            return round((r.mean()/r.std())*np.sqrt(252), 2) if len(r)>=5 and r.std()!=0 else np.nan
        except: return np.nan

    def _psharpe(self, h, period):
        try:
            now = pd.Timestamp.now()
            sd = (now - pd.Timedelta(days=now.weekday())).date() if period=='wtd' else now.replace(day=1).date()
            mn = 2 if period=='wtd' else 3
            ph = h[h.index.map(lambda x: x.date()) >= sd]
            if len(ph) < mn: return np.nan
            r = ph['Close'].pct_change().dropna()
            return round((r.mean()/r.std())*np.sqrt(252), 2) if len(r)>=2 and r.std()!=0 else np.nan
        except: return np.nan

    def _intra_sharpe(self, hi):
        try:
            if len(hi) < 30: return np.nan
            r = hi['Close'].pct_change().dropna()
            return round((r.mean()/r.std())*np.sqrt(252*390), 2) if len(r)>=20 and r.std()!=0 else np.nan
        except: return np.nan

    def _status(self, h, cp, pt):
        try:
            now = pd.Timestamp.now()
            if h.index.tzinfo:
                try: now = now.tz_localize(h.index.tzinfo)
                except: now = now.tz_localize('UTC').tz_convert(h.index.tzinfo)
            if pt == 'session':
                pd_ = h[h.index.map(lambda x: x.date()) < now.date()]
                if pd_.empty: return ''
                pp = pd_.iloc[-1:]
            elif pt == 'week':
                wsd = (now - pd.Timedelta(days=now.weekday())).date()
                pd_ = h[h.index.map(lambda x: x.date()) < wsd]
                if pd_.empty: return ''
                pp = pd_[pd_.index.map(lambda x: x.date()) >= wsd - pd.Timedelta(days=7)]
                if pp.empty: pp = pd_.tail(5)
            elif pt == 'month':
                msd = now.replace(day=1).date()
                pd_ = h[h.index.map(lambda x: x.date()) < msd]
                if pd_.empty: return ''
                pm = (now.month-2)%12+1; py = now.year if now.month>1 else now.year-1
                pp = pd_[(pd_.index.month==pm)&(pd_.index.year==py)]
            elif pt == 'year':
                pd_ = h[h.index.map(lambda x: x.date()) < now.replace(month=1,day=1).date()]
                if pd_.empty: return ''
                pp = pd_[pd_.index.year == now.year-1]
            else: return ''
            if pp.empty: return ''
            ph, pl = pp['High'].max(), pp['Low'].min(); mid = (ph+pl)/2
            if cp > ph: return 'above_high'
            elif cp < pl: return 'below_low'
            elif cp > mid: return 'above_mid'
            else: return 'below_mid'
        except: return ''

    def _rev(self, h, pt):
        try:
            if len(h) < 3: return False
            now = pd.Timestamp.now()
            if h.index.tzinfo:
                try: now = now.tz_localize(h.index.tzinfo)
                except: now = now.tz_localize('UTC').tz_convert(h.index.tzinfo)
            if pt == 'session':
                pd_ = h[h.index.map(lambda x: x.date()) < now.date()]
                if pd_.empty: return False
                pp = pd_.iloc[-1:]; cb = h[h.index.map(lambda x: x.date()) >= now.date()]
            elif pt == 'week':
                wsd = (now - pd.Timedelta(days=now.weekday())).date()
                pd_ = h[h.index.map(lambda x: x.date()) < wsd]
                if pd_.empty: return False
                pp = pd_[pd_.index.map(lambda x: x.date()) >= wsd - pd.Timedelta(days=7)]
                if pp.empty: pp = pd_.tail(5)
                cb = h[h.index.map(lambda x: x.date()) >= wsd]
            elif pt == 'month':
                msd = now.replace(day=1).date()
                pd_ = h[h.index.map(lambda x: x.date()) < msd]
                if pd_.empty: return False
                pm = (now.month-2)%12+1; py = now.year if now.month>1 else now.year-1
                pp = pd_[(pd_.index.month==pm)&(pd_.index.year==py)]
                cb = h[h.index.map(lambda x: x.date()) >= msd]
            elif pt == 'year':
                ysd = now.replace(month=1,day=1).date()
                pd_ = h[h.index.map(lambda x: x.date()) < ysd]
                if pd_.empty: return False
                pp = pd_[pd_.index.year == now.year-1]
                cb = h[h.index.map(lambda x: x.date()) >= ysd]
            else: return False
            if pp.empty or cb.empty: return False
            ph, pl = pp['High'].max(), pp['Low'].min(); cc = cb['Close'].iloc[-1]
            if cb['High'].max() > ph and cc <= ph: return True
            if cb['Low'].min() < pl and cc >= pl: return True
            return False
        except: return False

    def _returns(self, h, cp):
        try:
            now = pd.Timestamp.now()
            periods = {'wtd':(now-pd.Timedelta(days=now.weekday())).date(),'mtd':now.replace(day=1).date(),'ytd':now.replace(month=1,day=1).date()}
            ret = []
            for _,sd in periods.items():
                ph = h[h.index.map(lambda x: x.date()) >= sd]
                ret.append(((cp-ph['Open'].iloc[0])/ph['Open'].iloc[0])*100 if not ph.empty else np.nan)
            return tuple(ret)
        except: return (np.nan,np.nan,np.nan)


# =============================================================================
# CACHED DATA FETCHING
# =============================================================================

@st.cache_data(ttl=120, show_spinner=False)
def fetch_sector_data(sector):
    return [r for s in FUTURES_GROUPS.get(sector,[]) if (r := FuturesDataFetcher(s).fetch())]

@st.cache_data(ttl=120, show_spinner=False)
def fetch_chart_data(symbol, period, interval):
    h = yf.Ticker(symbol).history(period=period, interval=interval)
    if '-USD' not in symbol and not h.empty: h = h[h.index.dayofweek < 5]
    return h

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(symbol):
    NEWS_TERMS = {
        'ES=F':'S&P 500','NQ=F':'Nasdaq 100','YM=F':'Dow Jones','RTY=F':'Russell 2000',
        'NKD=F':'Nikkei 225','ZB=F':'US treasury bonds','ZN=F':'10 year treasury yield',
        'ZF=F':'5 year treasury','ZT=F':'2 year treasury','6E=F':'EUR USD euro',
        '6J=F':'USD JPY yen','6B=F':'GBP USD pound','6A=F':'AUD USD australian',
        'USDSGD=X':'USD SGD Singapore dollar','CL=F':'crude oil','NG=F':'natural gas',
        'GC=F':'gold price','SI=F':'silver price','PL=F':'platinum','HG=F':'copper price',
        'ZS=F':'soybean','ZC=F':'corn grain','ZW=F':'wheat','ZM=F':'soybean meal',
        'SB=F':'sugar commodity','KC=F':'coffee arabica','CC=F':'cocoa','CT=F':'cotton commodity',
        'BTC-USD':'bitcoin','ETH-USD':'ethereum','SOL-USD':'solana crypto','XRP-USD':'XRP ripple',
        'BTC=F':'bitcoin futures CME','ETH=F':'ethereum futures CME',
    }
    term = NEWS_TERMS.get(symbol) or SYMBOL_NAMES.get(symbol,'') or clean_symbol(symbol)
    results = []; seen = set()
    for when in ['1d','3d']:
        if when=='3d' and len(results)>=2: break
        try:
            url = f"https://news.google.com/rss/search?q={quote(term)}+when:{when}&hl=en&gl=US&ceid=US:en"
            for e in feedparser.parse(url).entries[:12]:
                title = e.get('title','').strip()
                if not title or title in seen: continue
                link = e.get('link',''); prov = ''
                if ' - ' in title:
                    parts = title.rsplit(' - ',1)
                    if len(parts)==2 and len(parts[1])<40: title, prov = parts[0].strip(), parts[1].strip()
                ds = ''
                pub = e.get('published','') or e.get('updated','')
                if pub:
                    try:
                        dt = pd.Timestamp(pub)
                        nt = pd.Timestamp.now(tz=dt.tzinfo) if dt.tzinfo else pd.Timestamp.now()
                        dh = (nt-dt).total_seconds()/3600
                        ds = f"{int(dh*60)}m" if dh<1 else f"{int(dh)}h" if dh<24 else dt.strftime('%d %b')
                    except: pass
                seen.add(title)
                results.append({'title':title,'url':link,'provider':prov,'date':ds})
        except: pass
    return results[:10]


# =============================================================================
# SCANNER TABLE
# =============================================================================

def render_scanner(metrics, selected):
    if not metrics:
        st.markdown(f"<div style='padding:16px;color:{DIM};font-size:12px;font-family:{FONTS}'>No data — markets may be closed</div>", unsafe_allow_html=True)
        return
    t = get_theme(); zc = zone_colors(); pc, nc = t['pos'], t['neg']

    def fv(v):
        if pd.isna(v): return f"<span style='color:#1e293b'>—</span>"
        return f"<span style='color:{pc if v>=0 else nc};font-weight:600'>{'+' if v>=0 else ''}{v:.2f}%</span>"

    def dot(s, rev=False):
        ico = ""
        if s=='above_high': ico = f"<span style='color:{zc[s]};font-size:7px'>▲</span>"
        elif s=='below_low': ico = f"<span style='color:{zc[s]};font-size:7px'>▼</span>"
        if rev: ico += "<span style='color:#facc15;font-size:7px'>◆</span>"
        return f"<span style='display:inline-block;width:20px;text-align:left;margin-left:1px'>{ico}</span>"

    def chg(v, s, rev=False):
        return f"<span style='display:inline-block;width:52px;text-align:right;font-variant-numeric:tabular-nums'>{fv(v)}</span>{dot(s,rev)}"

    def sharpe(v):
        if pd.isna(v): return f"<span style='color:#1e293b'>—</span>"
        return f"<span style='color:{pc if v>=0 else nc};font-weight:500'>{v:+.2f}</span>"

    def trend(m):
        sts = [m.year_status, m.month_status, m.week_status, m.day_status]
        bull = sum(1 for s in sts if s in ('above_high','above_mid')); bear = 4-bull
        if bull>=3: conf = f"<span style='color:{pc};font-weight:700;font-size:10px'>{bull}/4▲</span>"
        elif bear>=3: conf = f"<span style='color:{nc};font-weight:700;font-size:10px'>{bear}/4▼</span>"
        else: conf = "<span style='color:#6b7280;font-weight:700;font-size:10px'>2/4─</span>"
        hb = all(s in ('above_high','above_mid') for s in sts[:2])
        hr = all(s in ('below_mid','below_low') for s in sts[:2])
        lb_ = any(s in ('above_high','above_mid') for s in sts[2:])
        lr_ = any(s in ('below_mid','below_low') for s in sts[2:])
        if bull==4: sig, sc = 'STR▲', t['str_up']
        elif bear==4: sig, sc = 'STR▼', t['str_dn']
        elif hb and lr_: sig, sc = 'PULL', t['pull']
        elif hr and lb_: sig, sc = 'BNCE', t['bnce']
        else: sig, sc = 'MIX', '#6b7280'
        return f"{conf} <span style='color:{sc};font-size:9px;font-weight:600'>{sig}</span>"

    th = f"padding:6px 8px;color:{MUTED};font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.05em;font-family:{FONTS};border-bottom:1px solid {BORDER};background:{SURFACE};"
    td = f"padding:5px 8px;border-bottom:1px solid #1e293b;font-family:{MONO};font-size:11px;"

    html = f"""<div class='scanner-wrap'><table>
        <thead><tr>
            <th style='{th}text-align:left' rowspan='2'></th><th style='{th}text-align:right' rowspan='2'>PRICE</th>
            <th style='{th}text-align:center;border-bottom:none' colspan='4'>CHANGE</th>
            <th style='{th}text-align:right' rowspan='2'>HV</th><th style='{th}text-align:right' rowspan='2'>DD</th>
            <th style='{th}text-align:center' rowspan='2'>TREND</th>
            <th style='{th}text-align:center;border-bottom:none' colspan='4'>SHARPE</th>
        </tr><tr>
            <th style='{th}text-align:left'>DAY</th><th style='{th}text-align:left'>WTD</th>
            <th style='{th}text-align:left'>MTD</th><th style='{th}text-align:left'>YTD</th>
            <th style='{th}text-align:right'>D</th><th style='{th}text-align:right'>W</th>
            <th style='{th}text-align:right'>M</th><th style='{th}text-align:right'>Y</th>
        </tr></thead><tbody>"""

    for m in metrics:
        sym = clean_symbol(m.symbol)
        bg = f'rgba(59,130,246,0.06)' if m.symbol==selected else 'transparent'
        bl = f'border-left:3px solid {ACCENT}' if m.symbol==selected else 'border-left:3px solid transparent'
        pf = f"{m.price:,.{m.decimals}f}"
        hv = f"<span style='color:#94a3b8'>{m.hist_vol:.1f}%</span>" if not pd.isna(m.hist_vol) else f"<span style='color:#1e293b'>—</span>"
        dd = f"<span style='color:{nc}'>{m.current_dd:.1f}%</span>" if not pd.isna(m.current_dd) else f"<span style='color:#1e293b'>—</span>"
        html += f"""<tr style='background:{bg}'>
            <td style='{td}{bl};color:{TEXT};font-weight:600'>{sym}</td>
            <td style='{td}color:white;font-weight:700;text-align:right'>{pf}</td>
            <td style='{td}text-align:left;white-space:nowrap'>{chg(m.change_day, m.day_status, m.day_reversal)}</td>
            <td style='{td}text-align:left;white-space:nowrap'>{chg(m.change_wtd, m.week_status, m.week_reversal)}</td>
            <td style='{td}text-align:left;white-space:nowrap'>{chg(m.change_mtd, m.month_status, m.month_reversal)}</td>
            <td style='{td}text-align:left;white-space:nowrap'>{chg(m.change_ytd, m.year_status, m.year_reversal)}</td>
            <td style='{td}text-align:right'>{hv}</td><td style='{td}text-align:right'>{dd}</td>
            <td style='{td}text-align:center;white-space:nowrap'>{trend(m)}</td>
            <td style='{td}text-align:right'>{sharpe(m.day_sharpe)}</td>
            <td style='{td}text-align:right'>{sharpe(m.wtd_sharpe)}</td>
            <td style='{td}text-align:right'>{sharpe(m.mtd_sharpe)}</td>
            <td style='{td}text-align:right'>{sharpe(m.ytd_sharpe)}</td></tr>"""
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# CHART BUILDER — shared logic for grid + mobile
# =============================================================================

def _get_zone(price, high, low, mid):
    if price > high: return 'above_high'
    elif price < low: return 'below_low'
    elif price > mid: return 'above_mid'
    else: return 'below_mid'

def _build_chart(fig, row, col, symbol, chart_type, label, interval, zone_desc, bt, cidx, live, zc, t):
    """Add one chart panel to a plotly figure. Returns (status, rsi, levels)."""
    hist = fetch_chart_data(symbol, get_dynamic_period(bt), interval)
    if hist.empty: return None, np.nan, None

    boundaries = PeriodBoundaryCalculator.get_boundaries(hist, bt, symbol)
    cp = live if live is not None else hist['Close'].iloc[-1]
    x = list(range(len(hist)))

    # Ticks
    if bt == 'session':
        ti = [i for i, dt in enumerate(hist.index) if dt.minute==0 and dt.hour%4==0]
        tl = [hist.index[i].strftime('%H:%M') for i in ti]
    else:
        n = 8; ti = list(range(0, len(hist), max(1, len(hist)//n)))
        fmt = '%a %d' if bt=='week' else '%d %b' if bt=='month' else "%b '%y"
        tl = [hist.index[i].strftime(fmt) for i in ti]

    def segments(x_data, closes, high, low, mid, off=0, dates=None):
        zones = [_get_zone(c, high, low, mid) for c in closes]
        i = 0
        while i < len(zones):
            z = zones[i]; si = i
            while i < len(zones) and zones[i] == z: i += 1
            se = min(i+1, len(closes))
            sx = x_data[si:se] if isinstance(x_data, list) else list(range(off+si, off+se))
            sy = closes[si:se]
            sd = [dt.strftime('%d %b %H:%M') if bt=='session' else dt.strftime('%d %b %Y') for dt in dates[si:se]] if dates else None
            ht = '%{customdata}<br>%{y:.2f}<extra></extra>' if sd else '%{y:.2f}<extra></extra>'
            fig.add_trace(go.Scatter(x=sx, y=sy, mode='lines', line=dict(color=zc[z], width=1.5),
                showlegend=False, customdata=sd, hovertemplate=ht), row=row, col=col)

    lc = '#4b5563'; status = None; levels = None

    if boundaries:
        lb = boundaries[-1]; mid = (lb.prev_high + lb.prev_low)/2
        lc = zc[_get_zone(cp, lb.prev_high, lb.prev_low, mid)]; bi = lb.idx

        if chart_type == 'bars':
            fig.add_trace(go.Candlestick(x=x, open=hist['Open'].values, high=hist['High'].values,
                low=hist['Low'].values, close=hist['Close'].values,
                increasing_line_color=t['pos'], decreasing_line_color=t['neg'],
                increasing_fillcolor=t['pos'], decreasing_fillcolor=t['neg'],
                showlegend=False, line=dict(width=1)), row=row, col=col)

        # Previous period
        if len(boundaries) >= 2:
            pb = boundaries[-2]; pm = (pb.prev_high+pb.prev_low)/2
            ps, pe = pb.idx, bi
            if pe > ps:
                if chart_type == 'line':
                    segments(x[ps:pe], hist['Close'].values[ps:pe], pb.prev_high, pb.prev_low, pm, ps, hist.index[ps:pe])
                seg = hist.iloc[ps:pe]
                if len(seg) > 0:
                    rh = seg['High'].expanding().max(); rl = seg['Low'].expanding().min()
                    rx = list(range(ps, pe))
                    fig.add_trace(go.Scatter(x=rx, y=((rh+pb.prev_low)/2).values, mode='lines', line=dict(color='#be185d', width=1, dash='dot'), showlegend=False, hovertemplate='Retrace Buy: %{y:.2f}<extra></extra>'), row=row, col=col)
                    fig.add_trace(go.Scatter(x=rx, y=((rl+pb.prev_high)/2).values, mode='lines', line=dict(color='#0284c7', width=1, dash='dot'), showlegend=False, hovertemplate='Retrace Sell: %{y:.2f}<extra></extra>'), row=row, col=col)

        # Grey pre-history
        ft = boundaries[-2].idx if len(boundaries)>=2 else bi
        if ft > 0 and chart_type == 'line':
            dl = [dt.strftime('%d %b %H:%M') if bt=='session' else dt.strftime('%d %b %Y') for dt in hist.index[:ft]]
            fig.add_trace(go.Scatter(x=x[:ft], y=hist['Close'].values[:ft], mode='lines', line=dict(color='#374151', width=1.5),
                showlegend=False, customdata=dl, hovertemplate='%{customdata}<br>%{y:.2f}<extra></extra>'), row=row, col=col)

        # Current period
        if bi < len(hist) and chart_type == 'line':
            segments(x[bi:], hist['Close'].values[bi:], lb.prev_high, lb.prev_low, mid, bi, hist.index[bi:])

        status = _get_zone(cp, lb.prev_high, lb.prev_low, mid)
        levels = {'high':lb.prev_high,'low':lb.prev_low,'mid':mid,'price':cp,'status':status,'label':label}

    elif not boundaries:
        if chart_type == 'bars':
            fig.add_trace(go.Candlestick(x=x, open=hist['Open'].values, high=hist['High'].values,
                low=hist['Low'].values, close=hist['Close'].values,
                increasing_line_color=t['pos'], decreasing_line_color=t['neg'],
                increasing_fillcolor=t['pos'], decreasing_fillcolor=t['neg'],
                showlegend=False, line=dict(width=1)), row=row, col=col)
        else:
            dl = [dt.strftime('%d %b %H:%M') if bt=='session' else dt.strftime('%d %b %Y') for dt in hist.index]
            fig.add_trace(go.Scatter(x=x, y=hist['Close'].values, mode='lines', line=dict(color='#374151', width=1.5),
                showlegend=False, customdata=dl, hovertemplate='%{customdata}<br>%{y:.2f}<extra></extra>'), row=row, col=col)

    rsi = calculate_rsi(hist['Close'])

    # MAs on weekly
    if bt == 'year':
        ma20 = hist['Close'].rolling(20).mean(); ma40 = hist['Close'].rolling(40).mean()
        if ma20.notna().any():
            fig.add_trace(go.Scatter(x=x, y=ma20.values, mode='lines', line=dict(color='#ffffff', width=0.7), showlegend=False, hovertemplate='MA20: %{y:.2f}<extra></extra>'), row=row, col=col)
        if ma40.notna().any():
            fig.add_trace(go.Scatter(x=x, y=ma40.values, mode='lines', line=dict(color='#a855f7', width=0.7), showlegend=False, hovertemplate='MA40: %{y:.2f}<extra></extra>'), row=row, col=col)

    # Boundary lines
    for j in range(min(2, len(boundaries))):
        b = boundaries[-(j+1)]; px = b.idx; ex = len(hist)-1 if j==0 else boundaries[-1].idx
        fig.add_vline(x=px, line=dict(color='#374151', width=1, dash='dot'), row=row, col=col)
        ml = (b.prev_high+b.prev_low)/2
        for yv, c, lbl in [(b.prev_high, zc['above_high'],'High'),(b.prev_low, zc['below_low'],'Low'),(b.prev_close,'#9ca3af','Close'),(ml,'#fbbf24','50%')]:
            w = 1 if lbl in ('High','Low') else 0.8
            fig.add_trace(go.Scatter(x=[px,ex], y=[yv]*2, mode='lines', line=dict(color=c, width=w, dash='dot'),
                showlegend=False, hovertemplate=f'{lbl}: {yv:.2f}<extra></extra>'), row=row, col=col)

    # Retrace lines current
    if boundaries:
        lb = boundaries[-1]; seg = hist.iloc[lb.idx:]
        if len(seg) > 0:
            rh = seg['High'].expanding().max(); rl = seg['Low'].expanding().min()
            rx = list(range(lb.idx, lb.idx+len(seg)))
            fig.add_trace(go.Scatter(x=rx, y=((rh+lb.prev_low)/2).values, mode='lines', line=dict(color='#be185d', width=1, dash='dot'), showlegend=False, hovertemplate='Retrace Buy: %{y:.2f}<extra></extra>'), row=row, col=col)
            fig.add_trace(go.Scatter(x=rx, y=((rl+lb.prev_high)/2).values, mode='lines', line=dict(color='#0284c7', width=1, dash='dot'), showlegend=False, hovertemplate='Retrace Sell: %{y:.2f}<extra></extra>'), row=row, col=col)

    # Reversal dots
    if boundaries:
        lb = boundaries[-1]; rx, ry = [], []
        for j in range(lb.idx, len(hist)-1):
            c0, c1 = hist['Close'].iloc[j], hist['Close'].iloc[j+1]
            if c0 > lb.prev_high and c1 <= lb.prev_high: rx.append(j+1); ry.append(c1)
            if c0 < lb.prev_low and c1 >= lb.prev_low: rx.append(j+1); ry.append(c1)
        if len(boundaries)>=2:
            pb = boundaries[-2]
            for j in range(pb.idx, lb.idx-1):
                c0, c1 = hist['Close'].iloc[j], hist['Close'].iloc[j+1]
                if c0 > pb.prev_high and c1 <= pb.prev_high: rx.append(j+1); ry.append(c1)
                if c0 < pb.prev_low and c1 >= pb.prev_low: rx.append(j+1); ry.append(c1)
        if rx:
            fig.add_trace(go.Scatter(x=rx, y=ry, mode='markers',
                marker=dict(color='#facc15', size=6, symbol='diamond', line=dict(color='#a16207', width=1)),
                showlegend=False, hovertemplate='Reversal: %{y:.2f}<extra></extra>'), row=row, col=col)

    # Axis config
    ax_n = '' if cidx==0 else str(cidx+1)
    fig.update_layout(**{
        f'xaxis{ax_n}': dict(tickmode='array', tickvals=ti, ticktext=tl, tickfont=dict(color=MUTED, size=8),
            range=[-2, len(hist)-1+int(len(hist)*0.35)]),
        f'yaxis{ax_n}': dict(range=[hist['Low'].min()-(hist['High'].max()-hist['Low'].min())*0.08,
            hist['High'].max()+(hist['High'].max()-hist['Low'].min())*0.08],
            side='right', tickfont=dict(size=9, color=MUTED))
    })

    # Price annotation
    dec = 4 if '=X' in symbol else 2
    fig.add_annotation(x=1.02, y=cp,
        xref=f'x{ax_n} domain', yref=f'y{ax_n}',
        text=f'<b>{cp:.{dec}f}</b>', showarrow=False,
        font=dict(color='white', size=9, family=MONO), bgcolor=lc,
        bordercolor=lc, borderwidth=1, borderpad=2, xanchor='left')

    return status, rsi, levels


def create_chart_grid(symbol, chart_type='line', mobile=False):
    """Build 2×2 (desktop) or 4×1 (mobile) chart grid."""
    zc = zone_colors(); t = get_theme(); ds = clean_symbol(symbol)

    live = None
    try:
        h = fetch_chart_data(symbol, '1d', '5m')
        if not h.empty: live = float(h['Close'].iloc[-1])
    except: pass

    if mobile:
        fig = make_subplots(rows=4, cols=1,
            subplot_titles=[f"{ds} {c[0]}  ·  {c[2]}" for c in CHART_CONFIGS], vertical_spacing=0.07)
        pos = [(1,1),(2,1),(3,1),(4,1)]
    else:
        fig = make_subplots(rows=2, cols=2,
            subplot_titles=[f"{ds} {c[0]}  ·  {c[2]}" for c in CHART_CONFIGS],
            vertical_spacing=0.14, horizontal_spacing=0.10)
        pos = [(1,1),(1,2),(2,1),(2,2)]

    all_levels = {}; meta = {}
    for idx, (label, interval, zd, bt) in enumerate(CHART_CONFIGS):
        r, c = pos[idx]
        status, rsi, levels = _build_chart(fig, r, c, symbol, chart_type, label, interval, zd, bt, idx, live, zc, t)
        meta[idx] = (status, rsi)
        if levels: all_levels[bt] = levels

    # Enhance subplot titles
    for idx, ann in enumerate(fig['layout']['annotations']):
        if hasattr(ann, 'text') and '·' in str(ann.text):
            status, rsi = meta.get(idx, (None, np.nan))
            parts = [ann.text]
            if not np.isnan(rsi):
                rc = zc['above_mid'] if rsi > 50 else zc['below_low']
                parts.append(f"<span style='color:{rc}'>RSI {rsi:.0f}</span>")
            if status:
                sc = zc.get(status, MUTED)
                parts.append(f"<b><span style='color:{sc}'>[{STATUS_LABELS.get(status,'')}]</span></b>")
            ann['text'] = '  '.join(parts)
            ann['font'] = dict(color='#94a3b8', size=11, family=FONTS)

    fig.update_layout(template='plotly_dark', height=1200 if mobile else 650,
        margin=dict(l=40, r=80, t=55, b=50),
        showlegend=False, plot_bgcolor=BG, paper_bgcolor=BG,
        dragmode='pan', hovermode='closest', autosize=True, font=dict(family=FONTS))
    fig.update_xaxes(gridcolor='#1e293b', linecolor='#1e293b', showgrid=True, tickangle=-45,
        rangeslider=dict(visible=False), fixedrange=False,
        showspikes=True, spikecolor=DIM, spikethickness=1, spikedash='dot', spikemode='across')
    fig.update_yaxes(gridcolor='#1e293b', linecolor='#1e293b', showgrid=True, side='right',
        fixedrange=False, showspikes=True, spikecolor=DIM, spikethickness=1, spikedash='dot', spikemode='across')

    return fig, all_levels


# =============================================================================
# KEY LEVELS PANEL
# =============================================================================

def render_levels(symbol, levels):
    zc = zone_colors(); t = get_theme()
    ds = clean_symbol(symbol); fn = SYMBOL_NAMES.get(symbol, symbol)
    if not levels: return

    tfo = ['year','month','week','session']
    sts = [levels.get(tf,{}).get('status','') for tf in tfo]
    bull = sum(1 for s in sts if s in ('above_high','above_mid')); bear = 4-bull
    hb = all(s in ('above_high','above_mid') for s in sts[:2])
    hr = all(s in ('below_mid','below_low') for s in sts[:2])
    lb_ = any(s in ('above_high','above_mid') for s in sts[2:])
    lr_ = any(s in ('below_mid','below_low') for s in sts[2:])
    if bull==4: sig, sc = 'STRONG ▲', t['str_up']
    elif bear==4: sig, sc = 'STRONG ▼', t['str_dn']
    elif hb and lr_: sig, sc = 'PULLBACK ↻', '#fbbf24'
    elif hr and lb_: sig, sc = 'BOUNCE ↻', '#a855f7'
    else: sig, sc = 'MIXED ─', '#6b7280'

    dec = 2; price = None
    for tf in ['session','week','month','year']:
        if tf in levels: price = levels[tf]['price']; dec = 2 if price > 10 else 4; break

    th = f"padding:5px 8px;color:{MUTED};text-align:right;font-size:9px;text-transform:uppercase;border-bottom:1px solid {BORDER};font-family:{FONTS};"
    td = f"padding:5px 8px;border-bottom:1px solid #1e293b;font-family:{MONO};font-size:11px;"

    html = f"""<div class='panel'>
        <div class='panel-hdr' style='font-family:{FONTS}'>
            <span><span style='color:white;font-size:13px;font-weight:700;letter-spacing:0.06em'>{ds}</span>
            <span style='color:{MUTED};font-size:10px;margin-left:6px'>{fn}</span></span>
            <span style='color:{sc};font-size:10px;font-weight:700;letter-spacing:0.04em'>{sig}</span>
        </div>
        <table style='border-collapse:collapse;width:100%'>
        <thead><tr><th style='{th}text-align:left'></th><th style='{th}'>HIGH</th><th style='{th}'>MID</th><th style='{th}'>LOW</th><th style='{th}text-align:center'>STATUS</th></tr></thead><tbody>"""

    if price is not None:
        html += f"<tr><td style='{td}color:white;font-weight:700'>PRICE</td><td style='{td}color:white;font-weight:700;text-align:right;font-size:14px'>{price:,.{dec}f}</td><td colspan='3' style='{td}'></td></tr>"

    lbl = {'session':'SESSION','week':'WEEK','month':'MONTH','year':'YEAR'}
    for tf in ['session','week','month','year']:
        if tf not in levels: continue
        l = levels[tf]; sco = zc.get(l['status'],DIM); stx = STATUS_LABELS.get(l['status'],'')
        html += f"""<tr><td style='{td}color:#94a3b8;font-weight:600'>{lbl[tf]}</td>
            <td style='{td}color:{TEXT};text-align:right'>{l['high']:,.{dec}f}</td>
            <td style='{td}color:{MUTED};text-align:right'>{l['mid']:,.{dec}f}</td>
            <td style='{td}color:{TEXT};text-align:right'>{l['low']:,.{dec}f}</td>
            <td style='{td}text-align:center'><span style='color:{sco};font-size:10px;font-weight:600'>{stx}</span></td></tr>"""
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# NEWS PANEL
# =============================================================================

def render_news(symbol):
    ds = clean_symbol(symbol); fn = SYMBOL_NAMES.get(symbol, symbol)
    news = fetch_news(symbol)

    html = f"""<div class='panel'>
        <div class='panel-hdr' style='font-family:{FONTS}'>
            <span><span style='color:white;font-size:12px;font-weight:700;letter-spacing:0.06em'>{ds} NEWS</span>
            <span style='color:{MUTED};font-size:10px;margin-left:6px'>{fn}</span></span>
        </div>"""

    if not news:
        html += f"<div style='padding:16px;color:{DIM};font-size:12px;font-family:{FONTS}'>No recent news</div>"
    else:
        html += "<div style='max-height:400px;overflow-y:auto'>"
        for item in news:
            tt, u, p, d = item['title'], item['url'], item['provider'], item['date']
            th = f"<a href='{u}' target='_blank' style='color:#cbd5e1;text-decoration:none;font-size:11px;line-height:1.4;font-family:{FONTS}'>{tt}</a>" if u else f"<span style='color:#cbd5e1;font-size:11px'>{tt}</span>"
            meta = []
            if p: meta.append(f"<span style='color:{ACCENT};font-weight:500'>{p}</span>")
            if d: meta.append(f"<span style='color:{DIM}'>{d}</span>")
            html += f"<div style='padding:8px 12px;border-bottom:1px solid #1e293b'>{th}<div style='font-size:10px;margin-top:3px'>{' · '.join(meta)}</div></div>"
        html += "</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    for k, v in [('sector','Indices'),('symbol','ES=F'),('chart_type','line'),('theme','Blue / Rose'),('layout','Desktop')]:
        if k not in st.session_state: st.session_state[k] = v

    est = pytz.timezone('US/Eastern'); sgt = pytz.timezone('Asia/Singapore')

    # ── Header ──
    st.markdown(f"""
        <div style='padding:10px 16px;background:linear-gradient(135deg, #0f172a, #1e293b);
            border:1px solid {BORDER};border-radius:8px;font-family:{FONTS};
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;margin-bottom:10px'>
            <div>
                <span style='color:white;font-size:15px;font-weight:700;letter-spacing:0.1em'>CHART DASHBOARD</span>
                <span style='color:{ACCENT};font-size:9px;font-weight:600;margin-left:8px;padding:2px 6px;
                    border:1px solid {ACCENT};border-radius:10px;vertical-align:middle'>LIVE</span>
            </div>
            <span style='color:{MUTED};font-size:11px;font-family:{MONO}'>
                {datetime.now(est).strftime('%a %d %b %Y  %H:%M %Z')} · {datetime.now(sgt).strftime('%H:%M SGT')}</span>
        </div>""", unsafe_allow_html=True)

    is_mobile = st.session_state.layout == 'Mobile'

    # ── Controls ──
    c1, c2, c3, c4 = st.columns([5, 1, 2, 1])
    with c1:
        sector = st.selectbox("SECTOR", list(FUTURES_GROUPS.keys()),
            index=list(FUTURES_GROUPS.keys()).index(st.session_state.sector), key='sel_sec')
        if sector != st.session_state.sector:
            st.session_state.sector = sector; st.session_state.symbol = FUTURES_GROUPS[sector][0]
    with c2:
        ct = st.selectbox("CHART", ['line','bars'],
            index=0 if st.session_state.chart_type=='line' else 1, key='sel_ct')
        st.session_state.chart_type = ct
    with c3:
        theme = st.selectbox("THEME", list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state.theme), key='sel_th')
        st.session_state.theme = theme
    with c4:
        layout = st.selectbox("LAYOUT", ['Desktop','Mobile'],
            index=0 if st.session_state.layout=='Desktop' else 1, key='sel_ly')
        st.session_state.layout = layout; is_mobile = layout=='Mobile'

    # ── Asset buttons ──
    st.markdown(f"<div style='color:{MUTED};font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;padding:4px 0 2px 2px;font-family:{FONTS}'>ASSET</div>", unsafe_allow_html=True)
    syms = FUTURES_GROUPS[st.session_state.sector]
    cpr = min(len(syms), 6) if is_mobile else min(len(syms), 12)
    for ri in range((len(syms)+cpr-1)//cpr):
        s, e = ri*cpr, min((ri+1)*cpr, len(syms))
        cols = st.columns(cpr)
        for j, sym in enumerate(syms[s:e]):
            with cols[j]:
                if st.button(clean_symbol(sym), key=f"s_{sym}", use_container_width=True,
                            type="primary" if sym==st.session_state.symbol else "secondary"):
                    st.session_state.symbol = sym; st.rerun()

    # ── Data ──
    with st.spinner('Loading...'):
        metrics = fetch_sector_data(st.session_state.sector)
    if metrics:
        render_scanner(metrics, st.session_state.symbol)

    # ── Charts + Panels ──
    if is_mobile:
        with st.spinner('Loading charts...'):
            try:
                fig, levels = create_chart_grid(st.session_state.symbol, st.session_state.chart_type, mobile=True)
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom':True,'displayModeBar':False,'responsive':True})
            except Exception as ex:
                st.error(f"Chart error: {ex}"); levels = {}
        render_levels(st.session_state.symbol, levels)
        render_news(st.session_state.symbol)
    else:
        cl, cr = st.columns([65, 35])
        with cl:
            with st.spinner('Loading charts...'):
                try:
                    fig, levels = create_chart_grid(st.session_state.symbol, st.session_state.chart_type, mobile=False)
                    st.plotly_chart(fig, use_container_width=True, config={
                        'scrollZoom':True,'displayModeBar':True,
                        'modeBarButtonsToAdd':['pan2d','zoom2d','resetScale2d'],'responsive':True})
                except Exception as ex:
                    st.error(f"Chart error: {ex}"); levels = {}
        with cr:
            render_levels(st.session_state.symbol, levels)
            render_news(st.session_state.symbol)

    # ── Footer ──
    st.markdown(f"""<div style='margin-top:12px;padding:6px 16px;border-top:1px solid #1e293b;font-family:{FONTS};
        display:flex;justify-content:space-between;align-items:center'>
        <span style='font-size:10px;color:{DIM}'>Data caches 2 min · Yahoo Finance</span>
        <span style='font-size:10px;color:{DIM}'>{datetime.now(est).strftime('%H:%M %Z')}</span></div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
