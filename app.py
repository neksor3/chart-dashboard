import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import logging
import feedparser
from urllib.parse import quote
import re

from config import (FUTURES_GROUPS, THEMES, SYMBOL_NAMES, CHART_CONFIGS,
                     STATUS_LABELS, FONTS, MONO, clean_symbol)

# =============================================================================
# SETUP
# =============================================================================

logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

st.set_page_config(page_title="Chart Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Dark theme CSS + Google Fonts
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { background-color: #1e1e1e; font-family: 'Inter', sans-serif; }
    header[data-testid="stHeader"] { background-color: #1e1e1e; }
    [data-testid="stSidebar"] { background-color: #16213e; }
    .stSelectbox > div > div { background-color: #16213e; color: #b0b0b0; font-family: 'Inter', sans-serif; }
    .stTextInput > div > div > input { font-family: 'Inter', sans-serif; }
    div[data-testid="stHorizontalBlock"] { gap: 0.3rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 0; background-color: #0f172a; padding: 0; border-radius: 0; border-bottom: 1px solid #2a4a6a; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent; color: #64748b; border: none;
        border-bottom: 2px solid transparent;
        padding: 8px 20px; font-size: 11px; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
        font-family: 'Inter', sans-serif;
    }
    .stTabs [aria-selected="true"] { background-color: transparent; color: #e2e8f0; border-bottom: 2px solid #60a5fa; }
    .stRadio > div { flex-direction: row; gap: 8px; }
    .stRadio > div > label { background-color: #16213e; padding: 4px 12px; border-radius: 3px;
        border: 1px solid #2a4a6a; color: #b0b0b0; font-size: 12px; }
    div[data-testid="stMarkdownContainer"] p { margin-bottom: 0; }
    .block-container { padding-top: 2.5rem; padding-bottom: 0rem; }
    button[kind="secondary"] { background-color: #16213e; color: white; border: 1px solid #2a4a6a; font-family: 'Inter', sans-serif; }
    .stButton > button { font-size: 11px !important; padding: 4px 8px !important; min-height: 30px !important; font-family: 'Inter', sans-serif !important; }
    @media (max-width: 768px) {
        .block-container { padding: 2.5rem 0.5rem 0 0.5rem !important; }
        .stButton > button { font-size: 9px !important; padding: 2px 4px !important; min-height: 24px !important; }
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# (FUTURES_GROUPS, CHART_CONFIGS imported from config.py)

# (THEMES imported from config.py)

def get_theme():
    name = st.session_state.get('theme', 'Emerald / Amber')
    return THEMES.get(name, THEMES['Emerald / Amber'])

def zone_colors():
    t = get_theme()
    return {'above_high': t['zone_hi'], 'above_mid': t['zone_amid'], 'below_mid': t['zone_bmid'], 'below_low': t['zone_lo']}


# (STATUS_LABELS, SYMBOL_NAMES, clean_symbol imported from config.py)

# =============================================================================
# HELPERS
# =============================================================================

def get_dynamic_period(boundary_type):
    now = pd.Timestamp.now()
    if boundary_type == 'session': return '3d'
    elif boundary_type == 'week': return f'{int(now.weekday() + 1 + 14 + 3)}d'
    elif boundary_type == 'month': return f'{int(now.day + 65 + 5)}d'
    elif boundary_type == 'year': return '3y'
    return '90d'

# =============================================================================
# PERIOD BOUNDARIES
# =============================================================================

@dataclass
class PeriodBoundary:
    idx: int
    date: pd.Timestamp
    prev_high: float
    prev_low: float
    prev_close: float

class PeriodBoundaryCalculator:
    @staticmethod
    def get_boundaries(df, boundary_type, symbol=''):
        if df is None or len(df) == 0: return []
        boundaries = []
        boundary_rules = {
            'year': lambda i: df.index[i].year != df.index[i-1].year,
            'month': lambda i: (df.index[i].month != df.index[i-1].month or df.index[i].year != df.index[i-1].year),
            'week': lambda i: (df.index[i].isocalendar()[1] != df.index[i-1].isocalendar()[1] or df.index[i].year != df.index[i-1].year),
            'session': lambda i: df.index[i].date() != df.index[i-1].date()
        }
        mask_funcs = {
            'year': lambda i: df.index.year == df.index[i-1].year,
            'month': lambda i: ((df.index.month == df.index[i-1].month) & (df.index.year == df.index[i-1].year)),
            'week': lambda i: ((df.index.map(lambda x: x.isocalendar()[1]) == df.index[i-1].isocalendar()[1]) & (df.index.year == df.index[i-1].year)),
            'session': lambda i: df.index.date == df.index[i-1].date()
        }
        if boundary_type not in boundary_rules: return boundaries
        for i in range(1, len(df)):
            if boundary_rules[boundary_type](i):
                prev_data = df.loc[mask_funcs[boundary_type](i)]
                if len(prev_data) > 0:
                    boundaries.append(PeriodBoundary(
                        idx=i, date=df.index[i],
                        prev_high=prev_data['High'].max(),
                        prev_low=prev_data['Low'].min(),
                        prev_close=prev_data['Close'].iloc[-1]))
        return boundaries

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1: return np.nan
    delta = closes.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if not rsi.empty else np.nan

# =============================================================================
# DATA FETCHING
# =============================================================================

@dataclass
class FuturesMetrics:
    symbol: str
    price: float
    change_day: float
    change_wtd: float
    change_mtd: float
    change_ytd: float
    timestamp: datetime
    lag_minutes: float
    decimals: int
    hist_vol: float = np.nan
    day_sharpe: float = np.nan
    wtd_sharpe: float = np.nan
    mtd_sharpe: float = np.nan
    ytd_sharpe: float = np.nan
    day_status: str = ''
    week_status: str = ''
    month_status: str = ''
    year_status: str = ''
    current_dd: float = np.nan
    day_reversal: bool = False
    week_reversal: bool = False
    month_reversal: bool = False
    year_reversal: bool = False

class FuturesDataFetcher:
    def __init__(self, symbol):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.est = pytz.timezone('US/Eastern')
        self.decimals = 4 if symbol.endswith('=X') else 2

    def fetch(self):
        try:
            hist_yearly = self.ticker.history(period='1y')
            if hist_yearly.empty: return None
            hist_intraday = self.ticker.history(period='1d', interval='1m')
            if not hist_intraday.empty:
                current_price = hist_intraday['Close'].iloc[-1]
                daily_open = hist_intraday['Open'].iloc[0]
                daily_change = ((current_price - daily_open) / daily_open) * 100
                last_timestamp = hist_intraday.index[-1]
            else:
                current_price = hist_yearly['Close'].iloc[-1]
                daily_open = hist_yearly['Open'].iloc[-1]
                daily_change = ((current_price - daily_open) / daily_open) * 100
                last_timestamp = hist_yearly.index[-1]

            now_est = datetime.now(self.est)
            try: lag_minutes = (now_est - last_timestamp.tz_convert(self.est)).total_seconds() / 60
            except: lag_minutes = 0
            wtd, mtd, ytd = self._calculate_period_returns(hist_yearly, current_price)
            day_status = self._calculate_period_status(hist_yearly, current_price, 'session')
            week_status = self._calculate_period_status(hist_yearly, current_price, 'week')
            month_status = self._calculate_period_status(hist_yearly, current_price, 'month')
            year_status = self._calculate_period_status(hist_yearly, current_price, 'year')
            hist_vol = self._calculate_hist_vol(hist_yearly)
            current_dd = self._calculate_current_dd(hist_yearly, current_price)
            ytd_sharpe = self._calculate_ytd_sharpe(hist_yearly, current_price)
            mtd_sharpe = self._calculate_period_sharpe(hist_yearly, 'mtd')
            wtd_sharpe = self._calculate_period_sharpe(hist_yearly, 'wtd')
            day_sharpe = self._calculate_intraday_sharpe(hist_intraday) if not hist_intraday.empty else np.nan
            day_rev = self._check_reversal(hist_yearly, 'session')
            week_rev = self._check_reversal(hist_yearly, 'week')
            month_rev = self._check_reversal(hist_yearly, 'month')
            year_rev = self._check_reversal(hist_yearly, 'year')
            return FuturesMetrics(
                symbol=self.symbol, price=round(current_price, self.decimals),
                change_day=round(daily_change, 2),
                change_wtd=round(wtd, 2) if not np.isnan(wtd) else np.nan,
                change_mtd=round(mtd, 2) if not np.isnan(mtd) else np.nan,
                change_ytd=round(ytd, 2) if not np.isnan(ytd) else np.nan,
                timestamp=last_timestamp, lag_minutes=round(lag_minutes, 0),
                decimals=self.decimals,
                hist_vol=round(hist_vol, 1) if not np.isnan(hist_vol) else np.nan,
                day_sharpe=round(day_sharpe, 2) if not np.isnan(day_sharpe) else np.nan,
                wtd_sharpe=round(wtd_sharpe, 2) if not np.isnan(wtd_sharpe) else np.nan,
                mtd_sharpe=round(mtd_sharpe, 2) if not np.isnan(mtd_sharpe) else np.nan,
                ytd_sharpe=round(ytd_sharpe, 2) if not np.isnan(ytd_sharpe) else np.nan,
                day_status=day_status, week_status=week_status,
                month_status=month_status, year_status=year_status,
                current_dd=round(current_dd, 2) if not np.isnan(current_dd) else np.nan,
                day_reversal=day_rev, week_reversal=week_rev,
                month_reversal=month_rev, year_reversal=year_rev)
        except Exception as e:
            return None

    def _calculate_hist_vol(self, hist):
        try:
            if len(hist) < 20: return np.nan
            dr = hist['Close'].pct_change().dropna()
            return dr.std() * np.sqrt(252) * 100 if len(dr) >= 20 else np.nan
        except: return np.nan

    def _calculate_current_dd(self, hist, current_price):
        try:
            if len(hist) < 2: return np.nan
            peak = hist['High'].max()
            return ((current_price - peak) / peak) * 100 if peak != 0 else np.nan
        except: return np.nan

    def _calculate_ytd_sharpe(self, hist, current_price):
        try:
            ytd_start = pd.Timestamp.now().replace(month=1, day=1).date()
            ytd_hist = hist[hist.index.map(lambda x: x.date()) >= ytd_start]
            if len(ytd_hist) < 10: return np.nan
            dr = ytd_hist['Close'].pct_change().dropna()
            if len(dr) < 5 or dr.std() == 0: return np.nan
            return (dr.mean() / dr.std()) * np.sqrt(252)
        except: return np.nan

    def _calculate_period_sharpe(self, hist, period):
        try:
            now = pd.Timestamp.now()
            if period == 'wtd': start_date = (now - pd.Timedelta(days=now.weekday())).date(); min_bars = 2
            elif period == 'mtd': start_date = now.replace(day=1).date(); min_bars = 3
            else: return np.nan
            ph = hist[hist.index.map(lambda x: x.date()) >= start_date]
            if len(ph) < min_bars: return np.nan
            dr = ph['Close'].pct_change().dropna()
            if len(dr) < 2 or dr.std() == 0: return np.nan
            return (dr.mean() / dr.std()) * np.sqrt(252)
        except: return np.nan

    def _calculate_intraday_sharpe(self, hist_intraday):
        try:
            if len(hist_intraday) < 30: return np.nan
            r = hist_intraday['Close'].pct_change().dropna()
            if len(r) < 20 or r.std() == 0: return np.nan
            return (r.mean() / r.std()) * np.sqrt(252 * 390)
        except: return np.nan

    def _calculate_period_status(self, hist, current_price, period_type):
        try:
            now = pd.Timestamp.now()
            if hist.index.tzinfo is not None:
                try: now = now.tz_localize(hist.index.tzinfo)
                except: now = now.tz_localize('UTC').tz_convert(hist.index.tzinfo)
            if period_type == 'session':
                prev_data = hist[hist.index.map(lambda x: x.date()) < now.date()]
                if prev_data.empty: return ''
                prev_period = prev_data.iloc[-1:]
            elif period_type == 'week':
                wsd = (now - pd.Timedelta(days=now.weekday())).date()
                prev_data = hist[hist.index.map(lambda x: x.date()) < wsd]
                if prev_data.empty: return ''
                pwsd = wsd - pd.Timedelta(days=7)
                prev_period = prev_data[prev_data.index.map(lambda x: x.date()) >= pwsd]
                if prev_period.empty: prev_period = prev_data.tail(5)
            elif period_type == 'month':
                msd = now.replace(day=1).date()
                prev_data = hist[hist.index.map(lambda x: x.date()) < msd]
                if prev_data.empty: return ''
                pm = (now.month - 2) % 12 + 1
                py = now.year if now.month > 1 else now.year - 1
                prev_period = prev_data[(prev_data.index.month == pm) & (prev_data.index.year == py)]
            elif period_type == 'year':
                ysd = now.replace(month=1, day=1).date()
                prev_data = hist[hist.index.map(lambda x: x.date()) < ysd]
                if prev_data.empty: return ''
                prev_period = prev_data[prev_data.index.year == now.year - 1]
            else: return ''
            if prev_period.empty: return ''
            ph, pl = prev_period['High'].max(), prev_period['Low'].min()
            pm = (ph + pl) / 2
            if current_price > ph: return 'above_high'
            elif current_price < pl: return 'below_low'
            elif current_price > pm: return 'above_mid'
            else: return 'below_mid'
        except: return ''

    def _check_reversal(self, hist, period_type):
        try:
            if len(hist) < 3: return False
            now = pd.Timestamp.now()
            if hist.index.tzinfo is not None:
                try: now = now.tz_localize(hist.index.tzinfo)
                except: now = now.tz_localize('UTC').tz_convert(hist.index.tzinfo)
            if period_type == 'session':
                today = now.date()
                prev_data = hist[hist.index.map(lambda x: x.date()) < today]
                if prev_data.empty: return False
                prev_period = prev_data.iloc[-1:]
                current_bars = hist[hist.index.map(lambda x: x.date()) >= today]
            elif period_type == 'week':
                wsd = (now - pd.Timedelta(days=now.weekday())).date()
                prev_data = hist[hist.index.map(lambda x: x.date()) < wsd]
                if prev_data.empty: return False
                pwsd = wsd - pd.Timedelta(days=7)
                prev_period = prev_data[prev_data.index.map(lambda x: x.date()) >= pwsd]
                if prev_period.empty: prev_period = prev_data.tail(5)
                current_bars = hist[hist.index.map(lambda x: x.date()) >= wsd]
            elif period_type == 'month':
                msd = now.replace(day=1).date()
                prev_data = hist[hist.index.map(lambda x: x.date()) < msd]
                if prev_data.empty: return False
                pm = (now.month - 2) % 12 + 1
                py = now.year if now.month > 1 else now.year - 1
                prev_period = prev_data[(prev_data.index.month == pm) & (prev_data.index.year == py)]
                current_bars = hist[hist.index.map(lambda x: x.date()) >= msd]
            elif period_type == 'year':
                ysd = now.replace(month=1, day=1).date()
                prev_data = hist[hist.index.map(lambda x: x.date()) < ysd]
                if prev_data.empty: return False
                prev_period = prev_data[prev_data.index.year == now.year - 1]
                current_bars = hist[hist.index.map(lambda x: x.date()) >= ysd]
            else: return False
            if prev_period.empty or current_bars.empty: return False
            ph, pl = prev_period['High'].max(), prev_period['Low'].min()
            current_close = current_bars['Close'].iloc[-1]
            period_high = current_bars['High'].max()
            if period_high > ph and current_close <= ph: return True
            period_low = current_bars['Low'].min()
            if period_low < pl and current_close >= pl: return True
            return False
        except: return False

    def _calculate_period_returns(self, hist, current_price):
        try:
            now = pd.Timestamp.now()
            periods = {
                'wtd': (now - pd.Timedelta(days=now.weekday())).date(),
                'mtd': now.replace(day=1).date(),
                'ytd': now.replace(month=1, day=1).date()
            }
            returns = []
            for pn, sd in periods.items():
                ph = hist[hist.index.map(lambda x: x.date()) >= sd]
                if not ph.empty:
                    sp = ph['Open'].iloc[0]
                    returns.append(((current_price - sp) / sp) * 100)
                else: returns.append(np.nan)
            return tuple(returns)
        except: return (np.nan, np.nan, np.nan)


# =============================================================================
# CACHED DATA FETCHING
# =============================================================================

@st.cache_data(ttl=120, show_spinner=False)
def fetch_sector_data(sector_name):
    symbols = FUTURES_GROUPS.get(sector_name, [])
    metrics = []
    for symbol in symbols:
        fetcher = FuturesDataFetcher(symbol)
        result = fetcher.fetch()
        if result: metrics.append(result)
    return metrics

@st.cache_data(ttl=120, show_spinner=False)
def fetch_chart_data(symbol, period, interval):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, interval=interval)
    is_crypto = '-USD' in symbol
    if not is_crypto and not hist.empty:
        hist = hist[hist.index.dayofweek < 5]
    return hist

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(symbol):
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
    st_term = NEWS_TERMS.get(symbol)
    if not st_term:
        fn = SYMBOL_NAMES.get(symbol, '')
        st_term = fn if fn else clean_symbol(symbol)
    results = []; seen = set()
    for when in ['1d', '3d']:
        if when == '3d' and len(results) >= 2: break
        try:
            url = f"https://news.google.com/rss/search?q={quote(st_term)}+when:{when}&hl=en&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            for entry in feed.entries[:12]:
                title = entry.get('title', '').strip()
                if not title or title in seen: continue
                link = entry.get('link', ''); provider = ''
                if ' - ' in title:
                    parts = title.rsplit(' - ', 1)
                    if len(parts) == 2 and len(parts[1]) < 40:
                        title, provider = parts[0].strip(), parts[1].strip()
                date_str = ''
                pub = entry.get('published', '') or entry.get('updated', '')
                if pub:
                    try:
                        dt = pd.Timestamp(pub)
                        now_ts = pd.Timestamp.now(tz=dt.tzinfo) if dt.tzinfo else pd.Timestamp.now()
                        dh = (now_ts - dt).total_seconds() / 3600
                        date_str = f"{int(dh*60)}m ago" if dh < 1 else f"{int(dh)}h ago" if dh < 24 else dt.strftime('%d %b %H:%M')
                    except: pass
                seen.add(title)
                results.append({'title': title, 'url': link, 'provider': provider, 'date': date_str})
        except: pass
    return results[:12]


# =============================================================================
# SCANNER TABLE
# =============================================================================

def render_scanner_table(metrics, selected_symbol):
    if not metrics:
        st.markdown(f"<div style='padding:10px;color:#6d6d6d;font-size:11px;'>No data — markets may be closed</div>", unsafe_allow_html=True)
        return

    t = get_theme(); zc = zone_colors()
    pos_c, neg_c = t['pos'], t['neg']

    def _fv(val):
        if pd.isna(val): return "<span style='color:#6d6d6d'>—</span>"
        c = pos_c if val >= 0 else neg_c
        return f"<span style='color:{c};font-weight:600'>{'+' if val >= 0 else ''}{val:.2f}%</span>"

    def _dot(status, reversal=False):
        ico = ""
        c = zc.get(status, '#3a3a3a')
        if status == 'above_high': ico = f"<span style='color:{c};font-weight:700;border:1px solid #ffffff80;padding:0 2px;border-radius:2px;font-size:8px'>▲</span>"
        elif status == 'below_low': ico = f"<span style='color:{c};font-weight:700;border:1px solid #ffffff80;padding:0 2px;border-radius:2px;font-size:8px'>▼</span>"
        if reversal: ico += "<span style='color:#facc15;font-weight:700;border:1px solid #ffffff80;padding:0 2px;border-radius:0;font-size:8px'>■</span>"
        return f"<span style='display:inline-block;width:28px;text-align:left;vertical-align:middle;margin-left:3px'>{ico}</span>"

    def _chg(val, status, reversal=False):
        return f"<span style='display:inline-block;width:56px;text-align:right;font-variant-numeric:tabular-nums'>{_fv(val)}</span>{_dot(status, reversal)}"

    def _sharpe(val):
        if pd.isna(val): return "<span style='color:#6d6d6d'>—</span>"
        c = pos_c if val >= 0 else neg_c
        return f"<span style='color:{c};font-weight:600'>{val:+.2f}</span>"

    def _trend(m):
        statuses = [m.year_status, m.month_status, m.week_status, m.day_status]
        bullish = sum(1 for s in statuses if s in ('above_high', 'above_mid')); bearish = 4 - bullish
        if bullish >= 3: conf = f"<span style='color:{pos_c};font-weight:700;font-size:10px'>{bullish}/4▲</span>"
        elif bearish >= 3: conf = f"<span style='color:{neg_c};font-weight:700;font-size:10px'>{bearish}/4▼</span>"
        else: conf = "<span style='color:#6b7280;font-weight:700;font-size:10px'>2/4─</span>"
        hb = all(s in ('above_high','above_mid') for s in statuses[:2])
        hr = all(s in ('below_mid','below_low') for s in statuses[:2])
        lb = any(s in ('above_high','above_mid') for s in statuses[2:])
        lr = any(s in ('below_mid','below_low') for s in statuses[2:])
        if bullish == 4: sig, sc = 'STR▲', t['str_up']
        elif bearish == 4: sig, sc = 'STR▼', t['str_dn']
        elif hb and lr: sig, sc = 'PULL', t['pull']
        elif hr and lb: sig, sc = 'BNCE', t['bnce']
        else: sig, sc = 'MIX', '#6b7280'
        return f"{conf} <span style='color:{sc};font-size:9px;font-weight:600'>{sig}</span>"

    th = "padding:3px 6px;border-bottom:1px solid #3a3a3a;color:#8a8a8a;font-weight:500;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;"
    td = "padding:4px 6px;border-bottom:1px solid #2a2a2a;"

    html = f"""<div style='overflow-x:auto;-webkit-overflow-scrolling:touch'><table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>
        <thead><tr>
            <th style='{th}text-align:left' rowspan='2'></th><th style='{th}text-align:right' rowspan='2'>PRICE</th>
            <th style='{th}text-align:center;border-bottom:none' colspan='4'>CHANGE</th>
            <th style='{th}text-align:right' rowspan='2'>HV</th><th style='{th}text-align:right' rowspan='2'>DD</th>
            <th style='{th}text-align:center' rowspan='2'>TREND</th>
            <th style='{th}text-align:center;border-bottom:none' colspan='4'>SHARPE</th>
        </tr><tr>
            <th style='{th}text-align:left'>DAY</th><th style='{th}text-align:left'>WTD</th>
            <th style='{th}text-align:left'>MTD</th><th style='{th}text-align:left'>YTD</th>
            <th style='{th}text-align:right'>DAY</th><th style='{th}text-align:right'>WTD</th>
            <th style='{th}text-align:right'>MTD</th><th style='{th}text-align:right'>YTD</th>
        </tr></thead><tbody>"""

    for m in metrics:
        pf = f"{m.price:,.{m.decimals}f}"
        ss = clean_symbol(m.symbol)
        bg = '#1a2744' if m.symbol == selected_symbol else 'transparent'
        hv = f"<span style='color:#9d9d9d'>{m.hist_vol:.1f}%</span>" if not pd.isna(m.hist_vol) else "<span style='color:#6d6d6d'>—</span>"
        dd = f"<span style='color:{neg_c};font-weight:600'>{m.current_dd:.1f}%</span>" if not pd.isna(m.current_dd) else "<span style='color:#6d6d6d'>—</span>"
        html += f"""<tr style='background:{bg}'>
            <td style='{td}color:#cccccc;font-weight:600;text-align:left;white-space:nowrap'>{ss}</td>
            <td style='{td}color:white;font-weight:700;text-align:right'>{pf}</td>
            <td style='{td}text-align:left;white-space:nowrap'>{_chg(m.change_day, m.day_status, m.day_reversal)}</td>
            <td style='{td}text-align:left;white-space:nowrap'>{_chg(m.change_wtd, m.week_status, m.week_reversal)}</td>
            <td style='{td}text-align:left;white-space:nowrap'>{_chg(m.change_mtd, m.month_status, m.month_reversal)}</td>
            <td style='{td}text-align:left;white-space:nowrap'>{_chg(m.change_ytd, m.year_status, m.year_reversal)}</td>
            <td style='{td}text-align:right'>{hv}</td>
            <td style='{td}text-align:right'>{dd}</td>
            <td style='{td}text-align:center;white-space:nowrap'>{_trend(m)}</td>
            <td style='{td}text-align:right'>{_sharpe(m.day_sharpe)}</td>
            <td style='{td}text-align:right'>{_sharpe(m.wtd_sharpe)}</td>
            <td style='{td}text-align:right'>{_sharpe(m.mtd_sharpe)}</td>
            <td style='{td}text-align:right'>{_sharpe(m.ytd_sharpe)}</td>
        </tr>"""
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# 4-CHART GRID
# =============================================================================

def create_4_chart_grid(symbol, chart_type='line', mobile=False):
    zc = zone_colors(); t = get_theme()
    display_symbol = clean_symbol(symbol)
    full_name = SYMBOL_NAMES.get(symbol, symbol)

    live_price = None
    try:
        hist_lag = fetch_chart_data(symbol, '1d', '5m')
        if not hist_lag.empty:
            live_price = float(hist_lag['Close'].iloc[-1])
    except: pass

    if mobile:
        fig = make_subplots(rows=4, cols=1,
            subplot_titles=[f"{display_symbol} {tf[0]}  ·  {tf[2]}" for tf in CHART_CONFIGS],
            vertical_spacing=0.07)
        positions = [(1,1),(2,1),(3,1),(4,1)]
    else:
        fig = make_subplots(rows=2, cols=2,
            subplot_titles=[f"{display_symbol} {tf[0]}  ·  {tf[2]}" for tf in CHART_CONFIGS],
            vertical_spacing=0.15, horizontal_spacing=0.12)
        positions = [(1,1),(1,2),(2,1),(2,2)]
    chart_statuses = {}; chart_rsis = {}; computed_levels = {}

    for chart_idx, (label, interval, zone_desc, boundary_type) in enumerate(CHART_CONFIGS):
        row, col = positions[chart_idx]
        period = get_dynamic_period(boundary_type)
        hist = fetch_chart_data(symbol, period, interval)
        if hist.empty: continue

        boundaries = PeriodBoundaryCalculator.get_boundaries(hist, boundary_type, symbol)
        current_price = hist['Close'].iloc[-1]
        if live_price is not None: current_price = live_price

        x_vals = list(range(len(hist)))

        if boundary_type == 'session':
            tick_indices = [i for i, dt in enumerate(hist.index) if dt.minute == 0 and dt.hour % 4 == 0]
            tick_labels = [hist.index[i].strftime('%H:%M') for i in tick_indices]
        elif boundary_type == 'week':
            n = 8; tick_indices = list(range(0, len(hist), max(1, len(hist)//n)))
            tick_labels = [hist.index[i].strftime('%a %d') for i in tick_indices]
        elif boundary_type == 'month':
            n = 8; tick_indices = list(range(0, len(hist), max(1, len(hist)//n)))
            tick_labels = [hist.index[i].strftime('%d %b') for i in tick_indices]
        else:
            n = 8; tick_indices = list(range(0, len(hist), max(1, len(hist)//n)))
            tick_labels = [hist.index[i].strftime("%b '%y") for i in tick_indices]

        line_color = '#6b7280'

        def get_zone(price, high, low, mid):
            if price > high: return 'above_high'
            elif price < low: return 'below_low'
            elif price > mid: return 'above_mid'
            else: return 'below_mid'

        def plot_colored_segments(x_data, closes, high, low, mid, start_offset=0, datetimes=None):
            zones = [get_zone(c, high, low, mid) for c in closes]
            i = 0
            while i < len(zones):
                zone = zones[i]; start_i = i
                while i < len(zones) and zones[i] == zone: i += 1
                seg_end = min(i + 1, len(closes))
                seg_x = x_data[start_i:seg_end] if isinstance(x_data, list) else list(range(start_offset+start_i, start_offset+seg_end))
                seg_y = closes[start_i:seg_end]
                seg_dt = [dt.strftime('%d %b %H:%M') if boundary_type == 'session' else dt.strftime('%d %b %Y') for dt in datetimes[start_i:seg_end]] if datetimes is not None else None
                hover = '%{customdata}<br>%{y:.2f}<extra></extra>' if seg_dt else '%{y:.2f}<extra></extra>'
                fig.add_trace(go.Scatter(x=seg_x, y=seg_y, mode='lines',
                    line=dict(color=zc[zone], width=1.5),
                    showlegend=False, customdata=seg_dt, hovertemplate=hover), row=row, col=col)

        if boundaries:
            last_b = boundaries[-1]
            mid = (last_b.prev_high + last_b.prev_low) / 2
            line_color = zc[get_zone(current_price, last_b.prev_high, last_b.prev_low, mid)]
            boundary_idx = last_b.idx

            if chart_type == 'bars':
                fig.add_trace(go.Candlestick(x=x_vals, open=hist['Open'].values, high=hist['High'].values,
                    low=hist['Low'].values, close=hist['Close'].values,
                    increasing_line_color=t['pos'], decreasing_line_color=t['neg'],
                    increasing_fillcolor=t['pos'], decreasing_fillcolor=t['neg'],
                    showlegend=False, line=dict(width=1)), row=row, col=col)

            if len(boundaries) >= 2:
                prev_b = boundaries[-2]; prev_mid = (prev_b.prev_high + prev_b.prev_low) / 2
                ps, pe = prev_b.idx, boundary_idx
                if pe > ps:
                    if chart_type == 'line':
                        plot_colored_segments(x_vals[ps:pe], hist['Close'].values[ps:pe], prev_b.prev_high, prev_b.prev_low, prev_mid, ps, hist.index[ps:pe])
                    prev_seg = hist.iloc[ps:pe]
                    if len(prev_seg) > 0:
                        rh = prev_seg['High'].expanding().max(); rl = prev_seg['Low'].expanding().min()
                        rx = list(range(ps, pe))
                        fig.add_trace(go.Scatter(x=rx, y=((rh + prev_b.prev_low)/2).values, mode='lines', line=dict(color='#be185d', width=1, dash='dot'), showlegend=False, hovertemplate='Retrace Buy: %{y:.2f}<extra></extra>'), row=row, col=col)
                        fig.add_trace(go.Scatter(x=rx, y=((rl + prev_b.prev_high)/2).values, mode='lines', line=dict(color='#0284c7', width=1, dash='dot'), showlegend=False, hovertemplate='Retrace Sell: %{y:.2f}<extra></extra>'), row=row, col=col)

            first_tracked = boundaries[-2].idx if len(boundaries) >= 2 else boundary_idx
            if first_tracked > 0 and chart_type == 'line':
                dt_labels = [dt.strftime('%d %b %H:%M') if boundary_type == 'session' else dt.strftime('%d %b %Y') for dt in hist.index[:first_tracked]]
                fig.add_trace(go.Scatter(x=x_vals[:first_tracked], y=hist['Close'].values[:first_tracked], mode='lines', line=dict(color='#6b7280', width=1.5), showlegend=False, customdata=dt_labels, hovertemplate='%{customdata}<br>%{y:.2f}<extra></extra>'), row=row, col=col)

            if boundary_idx < len(hist) and chart_type == 'line':
                plot_colored_segments(x_vals[boundary_idx:], hist['Close'].values[boundary_idx:], last_b.prev_high, last_b.prev_low, mid, boundary_idx, hist.index[boundary_idx:])

        elif not boundaries:
            if chart_type == 'bars':
                fig.add_trace(go.Candlestick(x=x_vals, open=hist['Open'].values, high=hist['High'].values,
                    low=hist['Low'].values, close=hist['Close'].values,
                    increasing_line_color=t['pos'], decreasing_line_color=t['neg'],
                    increasing_fillcolor=t['pos'], decreasing_fillcolor=t['neg'],
                    showlegend=False, line=dict(width=1)), row=row, col=col)
            else:
                dt_labels = [dt.strftime('%d %b %H:%M') if boundary_type == 'session' else dt.strftime('%d %b %Y') for dt in hist.index]
                fig.add_trace(go.Scatter(x=x_vals, y=hist['Close'].values, mode='lines', line=dict(color='#6b7280', width=1.5), showlegend=False, customdata=dt_labels, hovertemplate='%{customdata}<br>%{y:.2f}<extra></extra>'), row=row, col=col)

        if boundaries:
            zone_status = get_zone(current_price, last_b.prev_high, last_b.prev_low, mid)
            chart_statuses[chart_idx] = STATUS_LABELS[zone_status]
            computed_levels[boundary_type] = {'high': last_b.prev_high, 'low': last_b.prev_low, 'mid': mid, 'price': current_price, 'status': zone_status, 'label': label}

        rsi_value = calculate_rsi(hist['Close']); chart_rsis[chart_idx] = rsi_value

        # MAs on weekly chart
        if boundary_type == 'year':
            ma_20 = hist['Close'].rolling(window=20).mean(); ma_40 = hist['Close'].rolling(window=40).mean()
            if ma_20.notna().any():
                fig.add_trace(go.Scatter(x=x_vals, y=ma_20.values, mode='lines', line=dict(color='#ffffff', width=0.8), showlegend=False, hovertemplate='MA20: %{y:.2f}<extra></extra>'), row=row, col=col)
            if ma_40.notna().any():
                fig.add_trace(go.Scatter(x=x_vals, y=ma_40.values, mode='lines', line=dict(color='#a855f7', width=0.8), showlegend=False, hovertemplate='MA40: %{y:.2f}<extra></extra>'), row=row, col=col)

        # Boundary lines
        num_boundaries = min(2, len(boundaries))
        for j in range(num_boundaries):
            b = boundaries[-(j+1)]; px = b.idx; ex = len(hist)-1 if j == 0 else boundaries[-1].idx
            fig.add_vline(x=px, line=dict(color='#6b7280', width=1.5, dash='dot'), row=row, col=col)
            ml = (b.prev_high + b.prev_low) / 2
            fig.add_trace(go.Scatter(x=[px,ex], y=[b.prev_high]*2, mode='lines', line=dict(color=zc['above_high'], width=1, dash='dot'), showlegend=False, hovertemplate=f'High: {b.prev_high:.2f}<extra></extra>'), row=row, col=col)
            fig.add_trace(go.Scatter(x=[px,ex], y=[b.prev_low]*2, mode='lines', line=dict(color=zc['below_low'], width=1, dash='dot'), showlegend=False, hovertemplate=f'Low: {b.prev_low:.2f}<extra></extra>'), row=row, col=col)
            fig.add_trace(go.Scatter(x=[px,ex], y=[b.prev_close]*2, mode='lines', line=dict(color='#e5e7eb', width=1, dash='dot'), showlegend=False, hovertemplate=f'Close: {b.prev_close:.2f}<extra></extra>'), row=row, col=col)
            fig.add_trace(go.Scatter(x=[px,ex], y=[ml]*2, mode='lines', line=dict(color='#fbbf24', width=1, dash='dot'), showlegend=False, hovertemplate=f'50%: {ml:.2f}<extra></extra>'), row=row, col=col)

        # Retrace lines (current period)
        if boundaries:
            cp = hist.iloc[last_b.idx:]
            if len(cp) > 0:
                rh = cp['High'].expanding().max(); rl = cp['Low'].expanding().min()
                rx = list(range(last_b.idx, last_b.idx+len(cp)))
                fig.add_trace(go.Scatter(x=rx, y=((rh+last_b.prev_low)/2).values, mode='lines', line=dict(color='#be185d', width=1, dash='dot'), showlegend=False, hovertemplate='Retrace Buy: %{y:.2f}<extra></extra>'), row=row, col=col)
                fig.add_trace(go.Scatter(x=rx, y=((rl+last_b.prev_high)/2).values, mode='lines', line=dict(color='#0284c7', width=1, dash='dot'), showlegend=False, hovertemplate='Retrace Sell: %{y:.2f}<extra></extra>'), row=row, col=col)

        # Reversal dots
        if boundaries:
            rev_x, rev_y = [], []
            bi = last_b.idx; ph, pl = last_b.prev_high, last_b.prev_low
            for j in range(bi, len(hist) - 1):
                c0 = hist['Close'].iloc[j]; c1 = hist['Close'].iloc[j + 1]
                if c0 > ph and c1 <= ph: rev_x.append(j + 1); rev_y.append(c1)
                if c0 < pl and c1 >= pl: rev_x.append(j + 1); rev_y.append(c1)
            if len(boundaries) >= 2:
                pb = boundaries[-2]; end_i = last_b.idx
                for j in range(pb.idx, end_i - 1):
                    c0 = hist['Close'].iloc[j]; c1 = hist['Close'].iloc[j + 1]
                    if c0 > pb.prev_high and c1 <= pb.prev_high: rev_x.append(j + 1); rev_y.append(c1)
                    if c0 < pb.prev_low and c1 >= pb.prev_low: rev_x.append(j + 1); rev_y.append(c1)
            if rev_x:
                fig.add_trace(go.Scatter(x=rev_x, y=rev_y, mode='markers',
                    marker=dict(color='#facc15', size=7, symbol='circle', line=dict(color='#a16207', width=1)),
                    showlegend=False, hovertemplate='Reversal: %{y:.2f}<extra></extra>'), row=row, col=col)

        # Axis formatting
        if tick_indices:
            axis_name = f'xaxis{chart_idx+1}' if chart_idx > 0 else 'xaxis'
            fig.update_layout(**{axis_name: dict(tickmode='array', tickvals=tick_indices, ticktext=tick_labels, tickfont=dict(color='#888888', size=8))})

        y_min, y_max = hist['Low'].min(), hist['High'].max(); pad = (y_max - y_min) * 0.08
        yref = f'yaxis{chart_idx+1}' if chart_idx > 0 else 'yaxis'
        fig.update_layout(**{yref: dict(range=[y_min-pad, y_max+pad], side='right', tickfont=dict(size=9, color='#888888'))})
        xref = f'xaxis{chart_idx+1}' if chart_idx > 0 else 'xaxis'
        fig.update_layout(**{xref: dict(range=[-2, len(hist)-1+int(len(hist)*0.4)])})

        pd_dec = 4 if '=X' in symbol else 2
        fig.add_annotation(x=1.02, y=current_price,
            xref=f'x{chart_idx+1} domain' if chart_idx > 0 else 'x domain',
            yref=f'y{chart_idx+1}' if chart_idx > 0 else 'y',
            text=f'<b>{current_price:.{pd_dec}f}</b>', showarrow=False,
            font=dict(color='white', size=10), bgcolor=line_color,
            bordercolor=line_color, borderwidth=1, borderpad=3, xanchor='left')

    # Update subplot titles with status + RSI
    stc = {'▲ ABOVE HIGH': zc['above_high'], '● ABOVE MID': zc['above_mid'],
           '● BELOW MID': zc['below_mid'], '▼ BELOW LOW': zc['below_low']}
    for idx, ann in enumerate(fig['layout']['annotations']):
        if hasattr(ann, 'text') and '·' in str(ann.text):
            status = chart_statuses.get(idx, ''); rsi = chart_rsis.get(idx, np.nan)
            parts = [ann.text]
            if not np.isnan(rsi):
                rc = zc['above_mid'] if rsi > 50 else zc['below_low']
                parts.append(f"<span style='color:{rc}'>RSI {rsi:.0f}</span>")
            if status:
                c = stc.get(status, '#9d9d9d')
                parts.append(f"<b><span style='color:{c}'>[{status}]</span></b>")
            ann['text'] = '  '.join(parts); ann['font'] = dict(color='#9d9d9d', size=11)

    fig.update_layout(template='plotly_dark', height=1200 if mobile else 650, margin=dict(l=50,r=90,t=60,b=60),
        showlegend=False, plot_bgcolor='#121212', paper_bgcolor='#121212',
        dragmode='pan', hovermode='closest', autosize=True)
    fig.update_xaxes(gridcolor='#1f1f1f', linecolor='#2a2a2a', tickfont=dict(color='#888888', size=8),
        showgrid=True, showticklabels=True, tickangle=-45, rangeslider=dict(visible=False),
        fixedrange=False, showspikes=True, spikecolor='#6b7280', spikethickness=1, spikedash='dot', spikemode='across')
    fig.update_yaxes(gridcolor='#1f1f1f', linecolor='#2a2a2a', showgrid=True, side='right',
        fixedrange=False, showspikes=True, spikecolor='#6b7280', spikethickness=1, spikedash='dot', spikemode='across')

    return fig, computed_levels


# =============================================================================
# KEY LEVELS PANEL
# =============================================================================

def render_key_levels(symbol, levels):
    zc = zone_colors(); t = get_theme()
    ds = clean_symbol(symbol); fn = SYMBOL_NAMES.get(symbol, symbol)
    if not levels: return

    tfo = ['year','month','week','session']
    statuses = [levels.get(tf,{}).get('status','') for tf in tfo]
    bull = sum(1 for s in statuses if s in ('above_high','above_mid')); bear = 4 - bull
    hb = all(s in ('above_high','above_mid') for s in statuses[:2])
    hr = all(s in ('below_mid','below_low') for s in statuses[:2])
    lb = any(s in ('above_high','above_mid') for s in statuses[2:])
    lr = any(s in ('below_mid','below_low') for s in statuses[2:])
    if bull == 4: sig, sc = 'STRONG ▲', t['str_up']
    elif bear == 4: sig, sc = 'STRONG ▼', t['str_dn']
    elif hb and lr: sig, sc = 'PULLBACK ↻', '#fbbf24'
    elif hr and lb: sig, sc = 'BOUNCE ↻', '#a855f7'
    else: sig, sc = 'MIXED', '#6b7280'

    dec = 2; price = None
    for tf in ['session','week','month','year']:
        if tf in levels: price = levels[tf]['price']; dec = 2 if price > 10 else 4; break

    html = f"""<div style='padding:6px 10px;background-color:#16213e;display:flex;justify-content:space-between;align-items:center;font-family:{FONTS}'>
        <span><span style='color:#e2e8f0;font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase'>{ds} LEVELS</span>
        <span style='color:#6d6d6d;font-size:10px;margin-left:6px'>{fn}</span></span>
        <span style='color:{sc};font-size:10px;font-weight:600;letter-spacing:0.05em'>{sig}</span></div>"""

    html += f"""<div style='background-color:#1e1e1e'><table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%'>
        <thead><tr>
            <th style='padding:4px 8px;color:#6d6d6d;text-align:left;font-size:9px;text-transform:uppercase;border-bottom:1px solid #3a3a3a'></th>
            <th style='padding:4px 8px;color:#6d6d6d;text-align:right;font-size:9px;text-transform:uppercase;border-bottom:1px solid #3a3a3a'>HIGH</th>
            <th style='padding:4px 8px;color:#6d6d6d;text-align:right;font-size:9px;text-transform:uppercase;border-bottom:1px solid #3a3a3a'>MID</th>
            <th style='padding:4px 8px;color:#6d6d6d;text-align:right;font-size:9px;text-transform:uppercase;border-bottom:1px solid #3a3a3a'>LOW</th>
            <th style='padding:4px 8px;color:#6d6d6d;text-align:center;font-size:9px;text-transform:uppercase;border-bottom:1px solid #3a3a3a'>STATUS</th>
        </tr></thead><tbody>"""

    if price is not None:
        html += f"<tr><td style='padding:6px 8px;color:white;font-weight:700;border-bottom:1px solid #2a2a2a'>PRICE</td><td style='padding:6px 8px;color:white;font-weight:700;text-align:right;font-size:13px;border-bottom:1px solid #2a2a2a'>{price:,.{dec}f}</td><td colspan='3' style='border-bottom:1px solid #2a2a2a'></td></tr>"

    tfl = {'year':'YEAR','month':'MONTH','week':'WEEK','session':'SESSION'}
    for tf in ['session','week','month','year']:
        if tf not in levels: continue
        l = levels[tf]; sco = zc.get(l['status'],'#6d6d6d'); stx = STATUS_LABELS.get(l['status'],'')
        html += f"""<tr><td style='padding:4px 8px;color:#b0b0b0;font-weight:600;border-bottom:1px solid #2a2a2a'>{tfl[tf]}</td>
            <td style='padding:4px 8px;color:#e2e8f0;text-align:right;border-bottom:1px solid #2a2a2a'>{l['high']:,.{dec}f}</td>
            <td style='padding:4px 8px;color:#8a8a8a;text-align:right;border-bottom:1px solid #2a2a2a'>{l['mid']:,.{dec}f}</td>
            <td style='padding:4px 8px;color:#e2e8f0;text-align:right;border-bottom:1px solid #2a2a2a'>{l['low']:,.{dec}f}</td>
            <td style='padding:4px 8px;text-align:center;border-bottom:1px solid #2a2a2a'><span style='color:{sco};font-size:10px;font-weight:600'>{stx}</span></td></tr>"""
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# NEWS PANEL
# =============================================================================

def render_news_panel(symbol):
    ds = clean_symbol(symbol); fn = SYMBOL_NAMES.get(symbol, symbol)
    news = fetch_news(symbol)

    html = f"""<div style='padding:6px 10px;background-color:#16213e;font-family:{FONTS};margin-top:8px'>
        <span style='color:#e2e8f0;font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase'>{ds} NEWS</span>
        <span style='color:#6d6d6d;font-size:10px;margin-left:6px'>{fn}</span></div>"""

    if not news:
        html += f"<div style='padding:12px;background-color:#1e1e1e;color:#6d6d6d;font-size:11px'>No news available</div>"
    else:
        html += "<div style='background-color:#1e1e1e;max-height:400px;overflow-y:auto'>"
        for item in news:
            t_text = item['title']; u = item['url']; p = item['provider']; d = item['date']
            title_html = f"<a href='{u}' target='_blank' style='color:#c9d1d9;text-decoration:none;font-size:11px;line-height:1.4'>{t_text}</a>" if u else f"<span style='color:#c9d1d9;font-size:11px'>{t_text}</span>"
            meta = []
            if p: meta.append(f"<span style='color:#3b82f6'>{p}</span>")
            if d: meta.append(f"<span style='color:#6d6d6d'>{d}</span>")
            html += f"<div style='padding:8px 12px;border-bottom:1px solid #2a2a2a;font-family:{FONTS}'><div>{title_html}</div><div style='font-size:10px;margin-top:2px'>{' &middot; '.join(meta)}</div></div>"
        html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# MOBILE DETECTION
# =============================================================================

def _detect_mobile():
    try:
        ua = st.context.headers.get('User-Agent', '')
        return bool(re.search(r'iPhone|Android.*Mobile|Windows Phone', ua, re.I))
    except:
        return False


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    from spreads import render_spreads_tab
    from portfolio import render_portfolio_tab

    # Init session state
    if 'sector' not in st.session_state: st.session_state.sector = 'Indices'
    if 'symbol' not in st.session_state: st.session_state.symbol = 'ES=F'
    if 'chart_type' not in st.session_state: st.session_state.chart_type = 'line'
    if 'theme' not in st.session_state:
        st.session_state.theme = list(THEMES.keys())[0]
    elif st.session_state.theme not in THEMES:
        st.session_state.theme = list(THEMES.keys())[0]

    is_mobile = _detect_mobile()
    est = pytz.timezone('US/Eastern'); sgt = pytz.timezone('Asia/Singapore')
    ts_est = datetime.now(est).strftime('%a %d %b %Y  %H:%M %Z')
    ts_sgt = datetime.now(sgt).strftime('%H:%M SGT')

    # Tabs first — clean uppercase, no icons
    tab_charts, tab_spreads, tab_portfolio = st.tabs(["CHARTS", "SPREADS", "PORTFOLIO"])

    with tab_charts:
        st.markdown(f"""
            <div style='padding:10px 16px;background-color:#16213e;border-radius:4px;font-family:{FONTS};display:flex;justify-content:space-between;align-items:center;margin-bottom:10px'>
                <span style='color:#e2e8f0;font-size:13px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase'>CHARTS DASHBOARD</span>
                <span style='color:#9d9d9d;font-size:11px'>{ts_est} &nbsp;·&nbsp; {ts_sgt}</span>
            </div>""", unsafe_allow_html=True)
        _render_charts_tab(is_mobile, est)

    with tab_spreads:
        st.markdown(f"""
            <div style='padding:10px 16px;background-color:#16213e;border-radius:4px;font-family:{FONTS};display:flex;justify-content:space-between;align-items:center;margin-bottom:10px'>
                <span style='color:#e2e8f0;font-size:13px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase'>SPREADS DASHBOARD</span>
                <span style='color:#9d9d9d;font-size:11px'>{ts_est} &nbsp;·&nbsp; {ts_sgt}</span>
            </div>""", unsafe_allow_html=True)
        render_spreads_tab(is_mobile)

    with tab_portfolio:
        st.markdown(f"""
            <div style='padding:10px 16px;background-color:#16213e;border-radius:4px;font-family:{FONTS};display:flex;justify-content:space-between;align-items:center;margin-bottom:10px'>
                <span style='color:#e2e8f0;font-size:13px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase'>PORTFOLIO DASHBOARD</span>
                <span style='color:#9d9d9d;font-size:11px'>{ts_est} &nbsp;·&nbsp; {ts_sgt}</span>
            </div>""", unsafe_allow_html=True)
        render_portfolio_tab(is_mobile)


def _render_charts_tab(is_mobile, est):
    """Chart tab content — sector scanner, asset charts, levels, news."""
    _lbl = f"color:#e2e8f0;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"

    symbols = FUTURES_GROUPS[st.session_state.sector]
    sym_labels = [clean_symbol(s) for s in symbols]

    # Ensure current symbol belongs to current sector
    if st.session_state.symbol not in symbols:
        st.session_state.symbol = symbols[0]
    current_idx = symbols.index(st.session_state.symbol)

    # Sector change callback
    def _on_sector_change():
        new_sector = st.session_state.sel_sector
        st.session_state.sector = new_sector
        st.session_state.symbol = FUTURES_GROUPS[new_sector][0]
        # Clear asset key so selectbox picks up new list
        if 'sel_asset' in st.session_state:
            del st.session_state.sel_asset

    # Controls row: SECTOR + ASSET + SORT + CHART + THEME
    if is_mobile:
        col_sec, col_ast, col_sort, col_ct, col_th = st.columns([3, 2, 2, 1, 2])
    else:
        col_sec, col_ast, col_sort, col_ct, col_th = st.columns([3, 3, 2, 1, 2])

    with col_sec:
        st.markdown(f"<div style='{_lbl}'>SECTOR</div>", unsafe_allow_html=True)
        sector_names = list(FUTURES_GROUPS.keys())
        if st.session_state.get('sel_sector') != st.session_state.sector:
            st.session_state.sel_sector = st.session_state.sector
        sector = st.selectbox("Sector", sector_names,
            key='sel_sector', label_visibility='collapsed',
            on_change=_on_sector_change)
    with col_ast:
        st.markdown(f"<div style='{_lbl}'>ASSET</div>", unsafe_allow_html=True)
        selected_label = st.selectbox("Asset", sym_labels, index=current_idx,
            key='sel_asset', label_visibility='collapsed')
        selected_sym = symbols[sym_labels.index(selected_label)]
        if selected_sym != st.session_state.symbol:
            st.session_state.symbol = selected_sym
            st.rerun()
    with col_sort:
        st.markdown(f"<div style='{_lbl}'>SORT</div>", unsafe_allow_html=True)
        sort_options = ['Default', 'Day %', 'WTD %', 'MTD %', 'YTD %', 'HV', 'DD',
                        'Sharpe Day', 'Sharpe WTD', 'Sharpe MTD', 'Sharpe YTD']
        sort_by = st.selectbox("Sort", sort_options, index=0,
            key='scanner_sort', label_visibility='collapsed')
    with col_ct:
        st.markdown(f"<div style='{_lbl}'>CHART</div>", unsafe_allow_html=True)
        chart_options = ['Line', 'Bars']
        ct_idx = 0 if st.session_state.chart_type == 'line' else 1
        ct = st.selectbox("Chart", chart_options, index=ct_idx,
            key='chart_select', label_visibility='collapsed')
        st.session_state.chart_type = ct.lower()
    with col_th:
        st.markdown(f"<div style='{_lbl}'>THEME</div>", unsafe_allow_html=True)
        theme_names = list(THEMES.keys())
        # Use 'theme' directly as widget key — single source of truth
        if st.session_state.get('theme') not in theme_names:
            st.session_state.theme = theme_names[0]
        theme = st.selectbox("Theme", theme_names,
            key='theme', label_visibility='collapsed')

    # Fetch data
    with st.spinner('Loading market data...'):
        metrics = fetch_sector_data(st.session_state.sector)

    # Sort scanner data
    if metrics and sort_by != 'Default':
        sort_map = {
            'Day %': 'change_day', 'WTD %': 'change_wtd', 'MTD %': 'change_mtd', 'YTD %': 'change_ytd',
            'HV': 'hist_vol', 'DD': 'current_dd',
            'Sharpe Day': 'day_sharpe', 'Sharpe WTD': 'wtd_sharpe', 'Sharpe MTD': 'mtd_sharpe', 'Sharpe YTD': 'ytd_sharpe',
        }
        attr = sort_map.get(sort_by)
        if attr:
            reverse = sort_by not in ('HV', 'DD')  # higher is better for most, lower for HV/DD
            metrics = sorted(metrics, key=lambda m: getattr(m, attr, 0) if not pd.isna(getattr(m, attr, None)) else -999,
                           reverse=reverse)

    # Scanner table
    if metrics:
        render_scanner_table(metrics, st.session_state.symbol)

    # Charts + Levels + News
    if is_mobile:
        with st.spinner('Loading charts...'):
            try:
                fig, levels = create_4_chart_grid(st.session_state.symbol, st.session_state.chart_type, mobile=True)
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False, 'responsive': True})
            except Exception as e:
                st.error(f"Chart error: {str(e)}"); levels = {}
        render_key_levels(st.session_state.symbol, levels)
        render_news_panel(st.session_state.symbol)
    else:
        col_chart_area, col_right = st.columns([65, 35])
        with col_chart_area:
            with st.spinner('Loading charts...'):
                try:
                    fig, levels = create_4_chart_grid(st.session_state.symbol, st.session_state.chart_type, mobile=False)
                    st.plotly_chart(fig, use_container_width=True, config={
                        'scrollZoom': True, 'displayModeBar': True,
                        'modeBarButtonsToAdd': ['pan2d','zoom2d','resetScale2d'],
                        'responsive': True})
                except Exception as e:
                    st.error(f"Chart error: {str(e)}"); levels = {}
        with col_right:
            render_key_levels(st.session_state.symbol, levels)
            render_news_panel(st.session_state.symbol)

    # Footer
    ct_now = datetime.now(est).strftime('%H:%M %Z')
    st.markdown(f"""<div style='margin-top:16px;padding:8px 12px;background-color:#16213e;border-radius:4px;font-family:{FONTS}'>
        <span style='font-size:11px;color:#9d9d9d'>EST: <span style='color:#cccccc'>{ct_now}</span>
        &nbsp;·&nbsp; <span style='color:#6d6d6d'>Data refreshes every 2 minutes · Click symbol for analysis</span></span></div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
