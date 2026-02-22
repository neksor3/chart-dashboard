import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import feedparser
from urllib.parse import quote
from html import escape as html_escape
import re

from config import (FUTURES_GROUPS, THEMES, SYMBOL_NAMES, FONTS, clean_symbol)

logger = logging.getLogger(__name__)

# =============================================================================
# CHARTS-ONLY CONSTANTS
# =============================================================================

CHART_CONFIGS = [
    ('Day (15m)', '15m', 'Session High/Low', 'session'),
    ('Weekly (4H)', '1h', 'Week High/Low', 'week'),
    ('Monthly (Daily)', '1d', 'Month High/Low', 'month'),
    ('Year (Weekly)', '1wk', 'Year High/Low', 'year'),
]

STATUS_LABELS = {
    'above_high': '▲ ABOVE HIGH', 'above_mid': '– ABOVE MID',
    'below_mid': '– BELOW MID', 'below_low': '▼ BELOW LOW',
}

_STATUS_LABEL_KEYS = {v: k for k, v in STATUS_LABELS.items()}

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
# THEME HELPERS
# =============================================================================

def get_theme():
    name = st.session_state.get('theme', 'Dark')
    return THEMES.get(name, THEMES['Dark'])

def zone_colors():
    t = get_theme()
    return {'above_high': t['zone_hi'], 'above_mid': t['zone_amid'],
            'below_mid': t['zone_bmid'], 'below_low': t['zone_lo']}


# =============================================================================
# ZONE / PRICE HELPERS
# =============================================================================

def get_zone(price, high, low, mid):
    if price > high: return 'above_high'
    elif price < low: return 'below_low'
    elif price > mid: return 'above_mid'
    else: return 'below_mid'

def get_dynamic_period(boundary_type):
    now = pd.Timestamp.now()
    if boundary_type == 'session': return '5d'
    elif boundary_type == 'week': return f'{int(now.weekday() + 1 + 14 + 3)}d'
    elif boundary_type == 'month': return f'{int(now.day + 65 + 5)}d'
    elif boundary_type == 'year': return '3y'
    return '90d'

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
# TIMEZONE HELPERS
# =============================================================================

def _tz_now(hist_index):
    tz = hist_index.tz
    return pd.Timestamp.now(tz=tz) if tz is not None else pd.Timestamp.now()

def _to_date(ts):
    return ts.date()

def _slice_period(hist, period_type, now=None):
    if now is None:
        now = _tz_now(hist.index)
    dates = hist.index.map(_to_date)

    if period_type == 'session':
        gaps = hist.index.to_series().diff()
        median_gap = gaps.median()
        today = now.date()
        if median_gap < pd.Timedelta(hours=4):
            prev_data = hist[dates < today]
            if prev_data.empty: return None, None
            prev_date = prev_data.index[-1].date()
            prev_period = prev_data[prev_data.index.map(_to_date) == prev_date]
            current_bars = hist[dates >= today]
        else:
            prev_data = hist[dates < today]
            if prev_data.empty: return None, None
            prev_period = prev_data.iloc[-1:]
            current_bars = hist[dates >= today]
    elif period_type == 'week':
        wsd = (now - pd.Timedelta(days=now.weekday())).date()
        prev_data = hist[dates < wsd]
        if prev_data.empty: return None, None
        pwsd = wsd - pd.Timedelta(days=7)
        prev_period = prev_data[prev_data.index.map(_to_date) >= pwsd]
        if prev_period.empty: prev_period = prev_data.tail(5)
        current_bars = hist[dates >= wsd]
    elif period_type == 'month':
        msd = now.replace(day=1).date()
        prev_data = hist[dates < msd]
        if prev_data.empty: return None, None
        pm = (now.month - 2) % 12 + 1
        py = now.year if now.month > 1 else now.year - 1
        prev_period = prev_data[(prev_data.index.month == pm) & (prev_data.index.year == py)]
        current_bars = hist[dates >= msd]
    elif period_type == 'year':
        ysd = now.replace(month=1, day=1).date()
        prev_data = hist[dates < ysd]
        if prev_data.empty: return None, None
        prev_period = prev_data[prev_data.index.year == now.year - 1]
        current_bars = hist[dates >= ysd]
    else:
        return None, None

    if prev_period.empty: return None, None
    return prev_period, current_bars


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
        if boundary_type == 'session' and len(df) >= 2:
            is_break = lambda i: df.index[i].date() != df.index[i-1].date()
        else:
            is_break = {
                'year': lambda i: df.index[i].year != df.index[i-1].year,
                'month': lambda i: (df.index[i].month != df.index[i-1].month or df.index[i].year != df.index[i-1].year),
                'week': lambda i: (df.index[i].isocalendar()[1] != df.index[i-1].isocalendar()[1] or df.index[i].year != df.index[i-1].year),
            }.get(boundary_type, lambda i: False)

        prev_start = 0
        for i in range(1, len(df)):
            if is_break(i):
                prev_data = df.iloc[prev_start:i]
                if len(prev_data) > 0:
                    boundaries.append(PeriodBoundary(
                        idx=i, date=df.index[i],
                        prev_high=prev_data['High'].max(),
                        prev_low=prev_data['Low'].min(),
                        prev_close=prev_data['Close'].iloc[-1]))
                prev_start = i
        return boundaries


# =============================================================================
# DATA MODEL
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
    day_reversal: str = ''
    week_reversal: str = ''
    month_reversal: str = ''
    year_reversal: str = ''


# =============================================================================
# DATA FETCHER
# =============================================================================

class FuturesDataFetcher:
    def __init__(self, symbol):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.est = pytz.timezone('US/Eastern')
        self.decimals = 4 if symbol.endswith('=X') else 2
        self._hist_yearly = None
        self._hist_intraday = None

    def fetch(self):
        try:
            hist_yearly = self.ticker.history(period='1y')
            if hist_yearly.empty: return None
            hist_intraday = self.ticker.history(period='1d', interval='1m')
        except Exception as e:
            logger.warning(f"[{self.symbol}] fetch API error: {e}")
            return None
        return self._compute_metrics(hist_yearly, hist_intraday)

    def fetch_from_cache(self):
        hist_yearly = self._hist_yearly
        hist_intraday = self._hist_intraday if self._hist_intraday is not None else pd.DataFrame()
        if hist_yearly is None or hist_yearly.empty: return None
        return self._compute_metrics(hist_yearly, hist_intraday)

    def _compute_metrics(self, hist_yearly, hist_intraday):
        try:
            if hist_yearly.empty: return None
            if not hist_intraday.empty:
                current_price = hist_intraday['Close'].iloc[-1]
                daily_open = hist_intraday['Open'].iloc[0]
                last_timestamp = hist_intraday.index[-1]
            else:
                current_price = hist_yearly['Close'].iloc[-1]
                daily_open = hist_yearly['Open'].iloc[-1]
                last_timestamp = hist_yearly.index[-1]

            daily_change = ((current_price - daily_open) / daily_open) * 100
            try:
                lag_minutes = (datetime.now(self.est) - last_timestamp.tz_convert(self.est)).total_seconds() / 60
            except Exception:
                lag_minutes = 0

            wtd, mtd, ytd = self._calculate_period_returns(hist_yearly, current_price)
            _sr = self._safe_round
            return FuturesMetrics(
                symbol=self.symbol, price=round(current_price, self.decimals),
                change_day=round(daily_change, 2),
                change_wtd=_sr(wtd, 2), change_mtd=_sr(mtd, 2), change_ytd=_sr(ytd, 2),
                timestamp=last_timestamp, lag_minutes=round(lag_minutes, 0), decimals=self.decimals,
                hist_vol=_sr(self._calculate_hist_vol(hist_yearly), 1),
                day_sharpe=_sr(self._calculate_intraday_sharpe(hist_intraday) if not hist_intraday.empty else np.nan, 2),
                wtd_sharpe=_sr(self._calculate_period_sharpe(hist_yearly, 'wtd'), 2),
                mtd_sharpe=_sr(self._calculate_period_sharpe(hist_yearly, 'mtd'), 2),
                ytd_sharpe=_sr(self._calculate_ytd_sharpe(hist_yearly, current_price), 2),
                day_status=self._calculate_period_status(hist_yearly, current_price, 'session'),
                week_status=self._calculate_period_status(hist_yearly, current_price, 'week'),
                month_status=self._calculate_period_status(hist_yearly, current_price, 'month'),
                year_status=self._calculate_period_status(hist_yearly, current_price, 'year'),
                current_dd=_sr(self._calculate_current_dd(hist_yearly, current_price), 2),
                day_reversal=self._check_reversal(hist_yearly, 'session'),
                week_reversal=self._check_reversal(hist_yearly, 'week'),
                month_reversal=self._check_reversal(hist_yearly, 'month'),
                year_reversal=self._check_reversal(hist_yearly, 'year'))
        except Exception as e:
            logger.warning(f"[{self.symbol}] fetch failed: {e}")
            return None

    @staticmethod
    def _safe_round(val, decimals):
        return round(val, decimals) if not np.isnan(val) else np.nan

    def _calculate_hist_vol(self, hist):
        try:
            if len(hist) < 20: return np.nan
            dr = hist['Close'].pct_change().dropna()
            return dr.std() * np.sqrt(252) * 100 if len(dr) >= 20 else np.nan
        except Exception as e:
            logger.debug(f"[{self.symbol}] hist_vol error: {e}"); return np.nan

    def _calculate_current_dd(self, hist, current_price):
        try:
            if len(hist) < 2: return np.nan
            peak = hist['High'].max()
            return ((current_price - peak) / peak) * 100 if peak != 0 else np.nan
        except Exception as e:
            logger.debug(f"[{self.symbol}] current_dd error: {e}"); return np.nan

    def _calculate_ytd_sharpe(self, hist, current_price):
        try:
            ytd_start = _tz_now(hist.index).replace(month=1, day=1).date()
            ytd_hist = hist[hist.index.map(_to_date) >= ytd_start]
            if len(ytd_hist) < 10: return np.nan
            dr = ytd_hist['Close'].pct_change().dropna()
            if len(dr) < 5 or dr.std() == 0: return np.nan
            return (dr.mean() / dr.std()) * np.sqrt(252)
        except Exception as e:
            logger.debug(f"[{self.symbol}] ytd_sharpe error: {e}"); return np.nan

    def _calculate_period_sharpe(self, hist, period):
        try:
            now = _tz_now(hist.index)
            if period == 'wtd':
                start_date = (now - pd.Timedelta(days=now.weekday())).date(); min_bars = 2
            elif period == 'mtd':
                start_date = now.replace(day=1).date(); min_bars = 3
            else: return np.nan
            ph = hist[hist.index.map(_to_date) >= start_date]
            if len(ph) < min_bars: return np.nan
            dr = ph['Close'].pct_change().dropna()
            if len(dr) < 2 or dr.std() == 0: return np.nan
            return (dr.mean() / dr.std()) * np.sqrt(252)
        except Exception as e:
            logger.debug(f"[{self.symbol}] period_sharpe ({period}) error: {e}"); return np.nan

    def _calculate_intraday_sharpe(self, hist_intraday):
        try:
            if len(hist_intraday) < 30: return np.nan
            r = hist_intraday['Close'].pct_change().dropna()
            if len(r) < 20 or r.std() == 0: return np.nan
            return (r.mean() / r.std()) * np.sqrt(252 * 390)
        except Exception as e:
            logger.debug(f"[{self.symbol}] intraday_sharpe error: {e}"); return np.nan

    def _calculate_period_status(self, hist, current_price, period_type):
        try:
            prev_period, _ = _slice_period(hist, period_type)
            if prev_period is None: return ''
            ph, pl = prev_period['High'].max(), prev_period['Low'].min()
            return get_zone(current_price, ph, pl, (ph + pl) / 2)
        except Exception as e:
            logger.debug(f"[{self.symbol}] period_status ({period_type}) error: {e}"); return ''

    def _check_reversal(self, hist, period_type):
        try:
            if len(hist) < 3: return ''
            prev_period, current_bars = _slice_period(hist, period_type)
            if prev_period is None or current_bars is None or current_bars.empty: return ''
            ph, pl = prev_period['High'].max(), prev_period['Low'].min()
            current_close = current_bars['Close'].iloc[-1]
            if current_bars['High'].max() > ph and current_close <= ph: return 'sell'
            if current_bars['Low'].min() < pl and current_close >= pl: return 'buy'
            return ''
        except Exception as e:
            logger.debug(f"[{self.symbol}] check_reversal ({period_type}) error: {e}"); return ''

    def _calculate_period_returns(self, hist, current_price):
        try:
            now = _tz_now(hist.index)
            periods = [
                (now - pd.Timedelta(days=now.weekday())).date(),
                now.replace(day=1).date(),
                now.replace(month=1, day=1).date(),
            ]
            returns = []
            for sd in periods:
                ph = hist[hist.index.map(_to_date) >= sd]
                if not ph.empty:
                    sp = ph['Open'].iloc[0]
                    returns.append(((current_price - sp) / sp) * 100)
                else:
                    returns.append(np.nan)
            return tuple(returns)
        except Exception as e:
            logger.debug(f"[{self.symbol}] period_returns error: {e}")
            return (np.nan, np.nan, np.nan)


# =============================================================================
# CACHED DATA FETCHING
# =============================================================================

@st.cache_resource(ttl=900, show_spinner=False)
def fetch_sector_data(sector_name):
    symbols = FUTURES_GROUPS.get(sector_name, [])
    if not symbols: return []
    try:
        batch_daily = yf.download(symbols, period='1y', group_by='ticker', threads=True, progress=False)
    except Exception as e:
        logger.warning(f"Batch download failed for {sector_name}: {e}")
        batch_daily = pd.DataFrame()

    metrics = []
    for symbol in symbols:
        try:
            hist_yearly = pd.DataFrame()
            if not batch_daily.empty:
                if len(symbols) == 1:
                    hist_yearly = batch_daily.copy()
                elif symbol in batch_daily.columns.get_level_values(0):
                    hist_yearly = batch_daily[symbol].dropna(how='all')
            if hist_yearly.empty:
                try: hist_yearly = yf.Ticker(symbol).history(period='1y')
                except Exception as e:
                    logger.warning(f"[{symbol}] individual fallback failed: {e}"); continue
            if hist_yearly.empty: continue

            fetcher = FuturesDataFetcher(symbol)
            fetcher._hist_yearly = hist_yearly
            fetcher._hist_intraday = yf.Ticker(symbol).history(period='1d', interval='1m')
            result = fetcher.fetch_from_cache()
            if result: metrics.append(result)
        except Exception as e:
            logger.warning(f"[{symbol}] sector fetch error: {e}")
    return metrics

@st.cache_data(ttl=900, show_spinner=False)
def fetch_chart_data(symbol, period, interval):
    hist = yf.Ticker(symbol).history(period=period, interval=interval)
    if '-USD' not in symbol and not hist.empty:
        hist = hist[hist.index.dayofweek < 5]
    return hist

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news(symbol):
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
                    except Exception: pass
                seen.add(title)
                results.append({'title': html_escape(title), 'url': link, 'provider': html_escape(provider), 'date': date_str})
        except Exception as e:
            logger.debug(f"News fetch error for {symbol} ({when}): {e}")
    return results[:12]


# =============================================================================
# SHARED DISPLAY HELPERS
# =============================================================================

def _fmt_value(val, pos_c, neg_c, muted):
    if pd.isna(val): return f"<span style='color:{muted}'>—</span>"
    c = pos_c if val >= 0 else neg_c
    return f"<span style='color:{c};font-weight:600'>{'+' if val >= 0 else ''}{val:.2f}%</span>"

def _fmt_dot(status, reversal, zc, pos_c, neg_c, muted):
    ico = ""
    c = zc.get(status, muted)
    if status == 'above_high': ico = f"<span style='color:{c};font-weight:700;font-size:8px'>▲</span>"
    elif status == 'below_low': ico = f"<span style='color:{c};font-weight:700;font-size:8px'>▼</span>"
    if reversal == 'buy': ico += f"<span style='color:{pos_c};font-size:10px'>●</span>"
    elif reversal == 'sell': ico += f"<span style='color:{neg_c};font-size:10px'>●</span>"
    return f"<span style='display:inline-block;width:20px;text-align:left;vertical-align:middle;margin-left:2px'>{ico}</span>"

def _fmt_change(val, status, reversal, zc, pos_c, neg_c, muted):
    return (f"<span style='display:inline-block;width:56px;text-align:right;font-variant-numeric:tabular-nums'>"
            f"{_fmt_value(val, pos_c, neg_c, muted)}</span>"
            f"{_fmt_dot(status, reversal, zc, pos_c, neg_c, muted)}")

def _fmt_sharpe(val, pos_c, neg_c, muted):
    if pd.isna(val): return f"<span style='color:{muted}'>—</span>"
    c = pos_c if val >= 0 else neg_c
    return f"<span style='color:{c};font-weight:600'>{val:+.2f}</span>"

def _fmt_trend(m, t, pos_c, neg_c):
    statuses = [m.year_status, m.month_status, m.week_status, m.day_status]
    bullish = sum(1 for s in statuses if s in ('above_high', 'above_mid'))
    bearish = 4 - bullish
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


# =============================================================================
# SCANNER TABLE + BAR CHART
# =============================================================================

def render_return_bars(metrics, sort_by='Default'):
    t = get_theme(); pos_c = t['pos']; neg_c = t['neg']
    field_map = {
        'Default': ('change_day', 'DAY %', True), 'Day %': ('change_day', 'DAY %', True),
        'WTD %': ('change_wtd', 'WTD %', True), 'MTD %': ('change_mtd', 'MTD %', True),
        'YTD %': ('change_ytd', 'YTD %', True),
        'HV': ('hist_vol', 'HV %', False), 'DD': ('current_dd', 'DD %', False),
        'Sharpe Day': ('day_sharpe', 'SHARPE DAY', False), 'Sharpe WTD': ('wtd_sharpe', 'SHARPE WTD', False),
        'Sharpe MTD': ('mtd_sharpe', 'SHARPE MTD', False), 'Sharpe YTD': ('ytd_sharpe', 'SHARPE YTD', False),
    }
    attr, label, is_change = field_map.get(sort_by, ('change_day', 'DAY %', True))
    vals = [(clean_symbol(m.symbol), getattr(m, attr, 0) if not pd.isna(getattr(m, attr, 0)) else 0) for m in metrics]
    if not vals: return
    vals.sort(key=lambda x: x[1], reverse=(sort_by not in ('HV', 'DD')))
    max_abs = max(abs(v) for _, v in vals) or 1

    n = len(vals)
    scanner_h = 52 + n * 26 + 2
    row_h = max((scanner_h - 46) // n, 18) if n > 0 else 22

    rows = ""
    for sym, v in vals:
        bar_pct = max(abs(v) / max_abs * 95, 3)
        c = pos_c if (v >= 0 or sort_by in ('HV',)) else neg_c
        if sort_by in ('HV', 'DD'): c = neg_c
        elif is_change: c = pos_c if v >= 0 else neg_c
        sign = '+' if v > 0 and is_change else ''
        fmt = f"{sign}{v:.1f}" if abs(v) < 100 else f"{sign}{v:.0f}"

        if v >= 0 or sort_by in ('HV',):
            left_c = ""
            right_c = (f"<div style='height:15px;width:{bar_pct}%;background:linear-gradient(90deg,{c}15,{c}55);border-radius:0 3px 3px 0'></div>"
                       f"<span style='color:{c};font-size:8px;font-weight:700;margin-left:3px;font-family:{FONTS};white-space:nowrap;font-variant-numeric:tabular-nums'>{fmt}</span>")
        else:
            left_c = (f"<span style='color:{c};font-size:8px;font-weight:700;margin-right:3px;font-family:{FONTS};white-space:nowrap;font-variant-numeric:tabular-nums'>{fmt}</span>"
                      f"<div style='height:15px;width:{bar_pct}%;background:linear-gradient(270deg,{c}15,{c}55);border-radius:3px 0 0 3px'></div>")
            right_c = ""

        rows += f"""<div style='display:flex;align-items:center;height:{row_h}px'>
            <div style='flex:1;display:flex;align-items:center;justify-content:flex-end'>{left_c}</div>
            <span style='width:36px;text-align:center;color:{t.get("text2","#9d9d9d")};font-size:9px;font-weight:600;font-family:{FONTS};flex-shrink:0'>{sym}</span>
            <div style='flex:1;display:flex;align-items:center'>{right_c}</div>
        </div>"""

    _bg0 = t.get('bg3', '#0f1522'); _bdr0 = t.get('border', '#1e293b')
    html = f"""<div style='background:{_bg0};border:1px solid {_bdr0};border-radius:6px;padding:0 6px;overflow:hidden;height:{scanner_h}px'>
        <div style='display:flex;align-items:flex-end;height:46px;padding:0 2px'>
            <span style='color:#f8fafc;font-size:9px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;font-family:{FONTS}'>{label}</span>
            <div style='flex:1;height:1px;background:{_bdr0};margin-left:8px'></div>
        </div>{rows}</div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_scanner_table(metrics, selected_symbol):
    if not metrics:
        st.markdown(f"<div style='padding:10px;color:{get_theme().get('muted','#475569')};font-size:11px;'>No data — markets may be closed</div>", unsafe_allow_html=True)
        return

    t = get_theme(); zc = zone_colors()
    pos_c, neg_c = t['pos'], t['neg']
    _mut = t.get('muted', '#475569'); _bdr = t.get('border', '#1e293b')
    _bg3 = t.get('bg3', '#0f172a'); _row_alt = '#131d2e'
    _txt1 = t.get('text', '#e2e8f0'); _txt2 = t.get('text2', '#94a3b8')
    th = f"padding:5px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;text-align:center;"
    td = f"padding:4px 8px;border:none;"
    _scanner_h = 52 + len(metrics) * 26 + 2

    html = f"""<div style='overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid {_bdr};border-radius:6px;height:{_scanner_h}px;overflow-y:hidden'><table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>
        <thead style='background:{_bg3}'><tr>
            <th style='{th}text-align:left' rowspan='2'></th><th style='{th}' rowspan='2'>PRICE</th>
            <th style='{th}border-bottom:none' colspan='4'>CHANGE</th>
            <th style='{th}' rowspan='2'>HV</th><th style='{th}' rowspan='2'>DD</th>
            <th style='{th}' rowspan='2'>TREND</th>
            <th style='{th}border-bottom:none' colspan='4'>SHARPE</th>
        </tr><tr>
            <th style='{th}'>DAY</th><th style='{th}'>WTD</th><th style='{th}'>MTD</th><th style='{th}'>YTD</th>
            <th style='{th}'>DAY</th><th style='{th}'>WTD</th><th style='{th}'>MTD</th><th style='{th}'>YTD</th>
        </tr></thead><tbody>"""

    for idx, m in enumerate(metrics):
        ss = clean_symbol(m.symbol)
        bg = f'linear-gradient(90deg,{pos_c}08,{t.get("bg3","#1a2744")},{pos_c}08)' if m.symbol == selected_symbol else (_row_alt if idx % 2 == 1 else 'transparent')
        hv = f"<span style='color:{_txt2}'>{m.hist_vol:.1f}%</span>" if not pd.isna(m.hist_vol) else f"<span style='color:{_mut}'>—</span>"
        dd = f"<span style='color:{neg_c};font-weight:600'>{m.current_dd:.1f}%</span>" if not pd.isna(m.current_dd) else f"<span style='color:{_mut}'>—</span>"
        html += f"""<tr style='background:{bg}'>
            <td style='{td}color:{_txt1};font-weight:600;text-align:left;white-space:nowrap'>{ss}</td>
            <td style='{td}color:#f8fafc;font-weight:700;text-align:center'>{m.price:,.{m.decimals}f}</td>
            <td style='{td}text-align:center;white-space:nowrap'>{_fmt_change(m.change_day, m.day_status, m.day_reversal, zc, pos_c, neg_c, _mut)}</td>
            <td style='{td}text-align:center;white-space:nowrap'>{_fmt_change(m.change_wtd, m.week_status, m.week_reversal, zc, pos_c, neg_c, _mut)}</td>
            <td style='{td}text-align:center;white-space:nowrap'>{_fmt_change(m.change_mtd, m.month_status, m.month_reversal, zc, pos_c, neg_c, _mut)}</td>
            <td style='{td}text-align:center;white-space:nowrap'>{_fmt_change(m.change_ytd, m.year_status, m.year_reversal, zc, pos_c, neg_c, _mut)}</td>
            <td style='{td}text-align:center'>{hv}</td><td style='{td}text-align:center'>{dd}</td>
            <td style='{td}text-align:center;white-space:nowrap'>{_fmt_trend(m, t, pos_c, neg_c)}</td>
            <td style='{td}text-align:center'>{_fmt_sharpe(m.day_sharpe, pos_c, neg_c, _mut)}</td>
            <td style='{td}text-align:center'>{_fmt_sharpe(m.wtd_sharpe, pos_c, neg_c, _mut)}</td>
            <td style='{td}text-align:center'>{_fmt_sharpe(m.mtd_sharpe, pos_c, neg_c, _mut)}</td>
            <td style='{td}text-align:center'>{_fmt_sharpe(m.ytd_sharpe, pos_c, neg_c, _mut)}</td>
        </tr>"""
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# CHART DRAWING HELPERS
# =============================================================================

def _add_zone_line(fig, row, col, x_data, closes, datetimes, boundary_type, mid, t, width=2.0):
    all_x, all_y = list(x_data), list(closes)
    all_dt = [dt.strftime('%d %b %H:%M') if boundary_type == 'session' else dt.strftime('%d %b %Y') for dt in datetimes] if datetimes is not None else None
    hover = '%{customdata}<br>%{y:.2f}<extra></extra>' if all_dt else '%{y:.2f}<extra></extra>'
    zones = ['up' if c >= mid else 'dn' for c in closes]
    i = 0
    while i < len(zones):
        zone = zones[i]; start_i = i
        while i < len(zones) and zones[i] == zone: i += 1
        end_i = min(i + 1, len(all_x))
        fig.add_trace(go.Scatter(
            x=all_x[start_i:end_i], y=all_y[start_i:end_i], mode='lines',
            line=dict(color=t['pos'] if zone == 'up' else t['neg'], width=width, shape='spline', smoothing=0.3),
            showlegend=False, customdata=all_dt[start_i:end_i] if all_dt else None,
            hovertemplate=hover), row=row, col=col)

def _add_simple_line(fig, row, col, x_data, closes, datetimes, boundary_type, color, width):
    all_x, all_y = list(x_data), list(closes)
    all_dt = [dt.strftime('%d %b %H:%M') if boundary_type == 'session' else dt.strftime('%d %b %Y') for dt in datetimes] if datetimes is not None else None
    hover = '%{customdata}<br>%{y:.2f}<extra></extra>' if all_dt else '%{y:.2f}<extra></extra>'
    fig.add_trace(go.Scatter(x=all_x, y=all_y, mode='lines',
        line=dict(color=color, width=width, shape='spline', smoothing=0.3),
        showlegend=False, customdata=all_dt, hovertemplate=hover), row=row, col=col)

def _add_candlesticks(fig, row, col, x_vals, hist, t):
    fig.add_trace(go.Candlestick(
        x=x_vals, open=hist['Open'].values, high=hist['High'].values,
        low=hist['Low'].values, close=hist['Close'].values,
        increasing_line_color=t['pos'], decreasing_line_color=t['neg'],
        increasing_fillcolor=t['pos'], decreasing_fillcolor=t['neg'],
        showlegend=False, line=dict(width=1)), row=row, col=col)

def _add_boundary_levels(fig, row, col, boundaries, hist, zc):
    for j in range(min(2, len(boundaries))):
        b = boundaries[-(j+1)]
        px, ex = b.idx, len(hist) - 1 if j == 0 else boundaries[-1].idx
        fig.add_vline(x=px, line=dict(color='rgba(255,255,255,0.25)', width=0.8, dash='dot'), row=row, col=col)
        ml = (b.prev_high + b.prev_low) / 2
        for y_val, color, lbl, dash, w in [
            (b.prev_high, zc['above_high'], 'High', None, 0.9),
            (b.prev_low, zc['below_low'], 'Low', None, 0.9),
            (b.prev_close, '#475569', 'Close', 'dot', 0.6),
            (ml, '#d97706', '50%', 'dot', 0.6),
        ]:
            fig.add_trace(go.Scatter(x=[px, ex], y=[y_val]*2, mode='lines',
                line=dict(color=color, width=w, dash=dash), showlegend=False,
                hovertemplate=f'{lbl}: {y_val:.2f}<extra></extra>'), row=row, col=col)

def _add_reversal_dots(fig, row, col, boundaries, hist, t):
    if not boundaries: return
    last_b = boundaries[-1]
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    def _scan(start, end, ph, pl):
        for j in range(start, end - 1):
            c0, c1 = hist['Close'].iloc[j], hist['Close'].iloc[j + 1]
            if c0 > ph and c1 <= ph: sell_x.append(j + 1); sell_y.append(c1)
            elif c0 < pl and c1 >= pl: buy_x.append(j + 1); buy_y.append(c1)
    _scan(last_b.idx, len(hist), last_b.prev_high, last_b.prev_low)
    if len(boundaries) >= 2:
        pb = boundaries[-2]
        _scan(pb.idx, last_b.idx, pb.prev_high, pb.prev_low)
    mk = dict(size=7, symbol='circle', line=dict(color='rgba(0,0,0,0.3)', width=0.5))
    if buy_x:
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', marker=dict(color=t['pos'], **mk),
            showlegend=False, hovertemplate='Buy reversal: %{y:.2f}<extra></extra>'), row=row, col=col)
    if sell_x:
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', marker=dict(color=t['neg'], **mk),
            showlegend=False, hovertemplate='Sell reversal: %{y:.2f}<extra></extra>'), row=row, col=col)

def _get_tick_labels(hist, boundary_type):
    if boundary_type == 'session':
        indices = [i for i, dt in enumerate(hist.index) if dt.minute == 0 and dt.hour % 4 == 0]
        labels = [hist.index[i].strftime('%H:%M') for i in indices]
    else:
        n = 8; indices = list(range(0, len(hist), max(1, len(hist)//n)))
        fmt = {'week': '%a %d', 'month': '%d %b', 'year': "%b '%y"}[boundary_type]
        labels = [hist.index[i].strftime(fmt) for i in indices]
    return indices, labels


# =============================================================================
# 4-CHART GRID
# =============================================================================

def create_4_chart_grid(symbol, chart_type='line', mobile=False):
    zc = zone_colors(); t = get_theme()

    live_price = None
    try:
        hist_lag = fetch_chart_data(symbol, '1d', '5m')
        if not hist_lag.empty: live_price = float(hist_lag['Close'].iloc[-1])
    except Exception as e:
        logger.debug(f"[{symbol}] live price fetch error: {e}")

    fig = make_subplots(rows=4, cols=1,
        subplot_titles=[tf[0].upper() for tf in CHART_CONFIGS], vertical_spacing=0.06)
    chart_statuses = {}; chart_rsis = {}; computed_levels = {}

    for chart_idx, (label, interval, _, boundary_type) in enumerate(CHART_CONFIGS):
        row, col = chart_idx + 1, 1
        hist = fetch_chart_data(symbol, get_dynamic_period(boundary_type), interval)
        if hist.empty: continue

        boundaries = PeriodBoundaryCalculator.get_boundaries(hist, boundary_type, symbol)
        current_price = live_price if live_price is not None else hist['Close'].iloc[-1]
        x_vals = list(range(len(hist)))
        line_color = '#6b7280'

        if boundaries:
            last_b = boundaries[-1]
            mid = (last_b.prev_high + last_b.prev_low) / 2
            zone_status = get_zone(current_price, last_b.prev_high, last_b.prev_low, mid)
            line_color = t['pos'] if zone_status in ('above_high', 'above_mid') else t['neg']
            bi = last_b.idx

            if chart_type == 'bars':
                _add_candlesticks(fig, row, col, x_vals, hist, t)
            if len(boundaries) >= 2 and chart_type == 'line':
                prev_b = boundaries[-2]
                _add_zone_line(fig, row, col, x_vals[prev_b.idx:bi], hist['Close'].values[prev_b.idx:bi],
                               hist.index[prev_b.idx:bi], boundary_type, (prev_b.prev_high + prev_b.prev_low) / 2, t, 1.5)
            first_tracked = boundaries[-2].idx if len(boundaries) >= 2 else bi
            if first_tracked > 0 and chart_type == 'line':
                _add_simple_line(fig, row, col, x_vals[:first_tracked], hist['Close'].values[:first_tracked],
                                 hist.index[:first_tracked], boundary_type, 'rgba(255,255,255,0.25)', 1.0)
            if bi < len(hist) and chart_type == 'line':
                _add_zone_line(fig, row, col, x_vals[bi:], hist['Close'].values[bi:],
                               hist.index[bi:], boundary_type, mid, t, 2.0)

            chart_statuses[chart_idx] = STATUS_LABELS[zone_status]
            computed_levels[boundary_type] = {'high': last_b.prev_high, 'low': last_b.prev_low, 'mid': mid,
                                               'price': current_price, 'status': zone_status, 'label': label}
        else:
            if chart_type == 'bars':
                _add_candlesticks(fig, row, col, x_vals, hist, t)
            else:
                _add_simple_line(fig, row, col, x_vals, hist['Close'].values, hist.index, boundary_type, 'rgba(255,255,255,0.5)', 1.5)

        chart_rsis[chart_idx] = calculate_rsi(hist['Close'])

        if boundary_type == 'year':
            for window, color in [(20, 'rgba(255,255,255,0.3)'), (40, 'rgba(168,85,247,0.5)')]:
                ma = hist['Close'].rolling(window=window).mean()
                if ma.notna().any():
                    fig.add_trace(go.Scatter(x=x_vals, y=ma.values, mode='lines', line=dict(color=color, width=0.7),
                        showlegend=False, hovertemplate=f'MA{window}: %{{y:.2f}}<extra></extra>'), row=row, col=col)

        _add_boundary_levels(fig, row, col, boundaries, hist, zc)
        _add_reversal_dots(fig, row, col, boundaries, hist, t)

        # Axis formatting
        tick_indices, tick_labels = _get_tick_labels(hist, boundary_type)
        ax = f'xaxis{chart_idx+1}' if chart_idx > 0 else 'xaxis'
        if tick_indices:
            fig.update_layout(**{ax: dict(tickmode='array', tickvals=tick_indices, ticktext=tick_labels, tickfont=dict(color='#e2e8f0', size=9))})
        if boundary_type == 'session' and boundaries:
            x_left = (boundaries[-2].idx if len(boundaries) >= 2 else boundaries[-1].idx) - 3
        else:
            x_left = -2
        last_bar = len(hist) - 1
        fig.update_layout(**{ax: dict(range=[x_left, x_left + int((last_bar - x_left) / 0.6)])})

        y_low, y_high = hist['Low'].dropna(), hist['High'].dropna()
        y_min = y_low.quantile(0.005) if len(y_low) > 10 else y_low.min()
        y_max = y_high.quantile(0.995) if len(y_high) > 10 else y_high.max()
        pad = (y_max - y_min) * 0.08
        ay = f'yaxis{chart_idx+1}' if chart_idx > 0 else 'yaxis'
        fig.update_layout(**{ay: dict(range=[y_min-pad, y_max+pad], side='right', tickfont=dict(size=9, color='#94a3b8'))})

        pd_dec = 4 if '=X' in symbol else 2
        fig.add_annotation(x=1.02, y=current_price,
            xref=f'x{chart_idx+1} domain' if chart_idx > 0 else 'x domain',
            yref=f'y{chart_idx+1}' if chart_idx > 0 else 'y',
            text=f'<b>{current_price:.{pd_dec}f}</b>', showarrow=False,
            font=dict(color=line_color, size=9), bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)', borderwidth=0, borderpad=2, xanchor='left')

    # Subplot titles
    title_labels = [tf[0].upper() for tf in CHART_CONFIGS]
    _clean_sym = clean_symbol(symbol)
    for idx, ann in enumerate(fig['layout']['annotations']):
        txt = str(ann.text) if hasattr(ann, 'text') else ''
        if txt in title_labels:
            status = chart_statuses.get(idx, ''); rsi = chart_rsis.get(idx, np.nan)
            parts = [f"{_clean_sym}  {txt}"]
            if not np.isnan(rsi):
                parts.append(f"<span style='color:{zc['above_mid'] if rsi > 50 else zc['below_low']};font-size:9px'>RSI {rsi:.0f}</span>")
            if status:
                c = zc.get(_STATUS_LABEL_KEYS.get(status, ''), '#64748b')
                parts.append(f"<span style='color:{c};font-size:9px'>{status}</span>")
            ann['text'] = '  '.join(parts); ann['font'] = dict(color='#f8fafc', size=10)

    _pbg = t.get('plot_bg', '#121212')
    fig.update_layout(template='plotly_dark', height=1100 if mobile else 850,
        margin=dict(l=40, r=80, t=50, b=20), showlegend=False,
        plot_bgcolor=_pbg, paper_bgcolor=_pbg, dragmode='pan', hovermode='closest', autosize=True)
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.03)', linecolor='rgba(0,0,0,0)',
        tickfont=dict(color='#e2e8f0', size=9), showgrid=True, showticklabels=True, tickangle=0,
        rangeslider=dict(visible=False), fixedrange=False,
        showspikes=True, spikecolor='#475569', spikethickness=0.5, spikedash='dot', spikemode='across')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.03)', linecolor='rgba(0,0,0,0)',
        showgrid=True, side='right', tickfont=dict(color='#94a3b8', size=9), fixedrange=False,
        showspikes=True, spikecolor='#475569', spikethickness=0.5, spikedash='dot', spikemode='across')

    return fig, computed_levels


# =============================================================================
# KEY LEVELS PANEL
# =============================================================================

def render_key_levels(symbol, levels):
    zc = zone_colors(); t = get_theme(); pos_c = t['pos']
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

    _hdr_bg = t.get('bg3', '#1a2744'); _body_bg = t.get('bg', '#1e1e1e')
    _bdr = t.get('border', '#2a2a2a'); _txt1 = t.get('text', '#e2e8f0')
    _txt2 = t.get('text2', '#b0b0b0'); _mut = t.get('muted', '#6d6d6d')

    html = f"""<div style='padding:8px 12px;background:{_hdr_bg};border-left:2px solid {pos_c};display:flex;justify-content:space-between;align-items:center;font-family:{FONTS};border-radius:4px 4px 0 0'>
        <span><span style='color:#f8fafc;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase'>{ds} LEVELS</span>
        <span style='color:{_mut};font-size:10px;margin-left:6px;font-weight:400'>{fn}</span></span>
        <span style='color:{sc};font-size:10px;font-weight:700;letter-spacing:0.05em'>{sig}</span></div>"""

    th_s = f"padding:5px 8px;color:#f8fafc;font-size:9px;text-transform:uppercase;border-bottom:1px solid {_bdr};font-weight:600"
    html += f"""<div style='background-color:{_body_bg};border:1px solid {_bdr};border-top:none;border-radius:0 0 4px 4px'><table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>
        <thead><tr>
            <th style='{th_s};text-align:left'></th><th style='{th_s};text-align:right'>HIGH</th>
            <th style='{th_s};text-align:right'>MID</th><th style='{th_s};text-align:right'>LOW</th>
            <th style='{th_s};text-align:center'>STATUS</th>
        </tr></thead><tbody>"""

    if price is not None:
        html += f"<tr><td style='padding:6px 8px;color:{_txt1};font-weight:700;border-bottom:1px solid {_bdr}'>PRICE</td><td style='padding:6px 8px;color:{_txt1};font-weight:700;text-align:right;font-size:13px;border-bottom:1px solid {_bdr}'>{price:,.{dec}f}</td><td colspan='3' style='border-bottom:1px solid {_bdr}'></td></tr>"

    tfl = {'year':'YEAR','month':'MONTH','week':'WEEK','session':'SESSION'}
    for tf in ['session','week','month','year']:
        if tf not in levels: continue
        l = levels[tf]; sco = zc.get(l['status'], _mut); stx = STATUS_LABELS.get(l['status'],'')
        row_bg = f'linear-gradient(90deg,transparent,{sco}08)' if l['status'] in ('above_high','below_low') else 'transparent'
        td_s = f"padding:5px 8px;border-bottom:1px solid {_bdr};font-variant-numeric:tabular-nums"
        html += f"""<tr style='background:{row_bg}'><td style='{td_s};color:{_txt2};font-weight:600'>{tfl[tf]}</td>
            <td style='{td_s};color:{_txt1};text-align:right'>{l['high']:,.{dec}f}</td>
            <td style='{td_s};color:{_mut};text-align:right'>{l['mid']:,.{dec}f}</td>
            <td style='{td_s};color:{_txt1};text-align:right'>{l['low']:,.{dec}f}</td>
            <td style='{td_s};text-align:center'><span style='color:{sco};font-size:10px;font-weight:700'>{stx}</span></td></tr>"""
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# NEWS PANEL
# =============================================================================

def render_news_panel(symbol):
    ds = clean_symbol(symbol); fn = SYMBOL_NAMES.get(symbol, symbol)
    t = get_theme(); pos_c = t['pos']
    _il = t.get('mode') == 'light'
    _hdr_bg = t.get('bg3', '#1a2744'); _body_bg = t.get('bg', '#1e1e1e')
    _bdr = t.get('border', '#2a2a2a'); _mut = t.get('muted', '#6d6d6d')
    _link_c = '#334155' if _il else '#c9d1d9'
    news = fetch_news(symbol)

    html = f"""<div style='padding:8px 12px;background:{_hdr_bg};border-left:2px solid {pos_c};font-family:{FONTS};margin-top:8px;border-radius:4px 4px 0 0'>
        <span style='color:#f8fafc;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase'>{ds} NEWS</span>
        <span style='color:{_mut};font-size:10px;margin-left:6px;font-weight:400'>{fn}</span></div>"""

    if not news:
        html += f"<div style='padding:12px;background-color:{_body_bg};border:1px solid {_bdr};border-top:none;border-radius:0 0 4px 4px;color:{_mut};font-size:11px;font-family:{FONTS}'>No news available</div>"
    else:
        _row_alt = t.get('bg3', '#131b2e')
        html += f"<div style='background-color:{_body_bg};border:1px solid {_bdr};border-top:none;border-radius:0 0 4px 4px;max-height:300px;overflow-y:auto'>"
        for i, item in enumerate(news):
            row_bg = _body_bg if i % 2 == 0 else _row_alt
            u, title, p, d = item['url'], item['title'], item['provider'], item['date']
            title_el = f"<a href='{u}' target='_blank' style='color:{_link_c};text-decoration:none;font-size:10.5px;font-weight:500;overflow:hidden;text-overflow:ellipsis'>{title}</a>" if u else f"<span style='color:{_link_c};font-size:10.5px'>{title}</span>"
            meta = []
            if p: meta.append(f"<span style='color:{pos_c};font-weight:600'>{p}</span>")
            if d: meta.append(f"<span style='color:{_mut}'>{d}</span>")
            html += (f"<div style='padding:5px 12px;border-bottom:1px solid {_bdr}10;font-family:{FONTS};background:{row_bg};"
                     f"display:flex;align-items:baseline;gap:6px;white-space:nowrap;overflow:hidden'>"
                     f"<span style='font-size:9px;flex-shrink:0;display:flex;gap:6px;align-items:baseline'>"
                     f"{f' <span style=\"color:{_bdr}\">|</span> '.join(meta)}</span>{title_el}</div>")
        html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# CHARTS TAB — MAIN ENTRY POINT
# =============================================================================

def render_charts_tab(is_mobile, est):
    t = get_theme(); pos_c = t['pos']
    symbols = FUTURES_GROUPS[st.session_state.sector]
    sym_labels = [clean_symbol(s) for s in symbols]

    if st.session_state.symbol not in symbols:
        st.session_state.symbol = symbols[0]
    current_idx = symbols.index(st.session_state.symbol)

    def _on_sector_change():
        new_sector = st.session_state.sel_sector
        st.session_state.sector = new_sector
        st.session_state.symbol = FUTURES_GROUPS[new_sector][0]
        if 'sel_asset' in st.session_state: del st.session_state.sel_asset

    _lbl = f"color:#f8fafc;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"
    col_sec, col_ast, col_sort, col_ct = st.columns([3, 2 if is_mobile else 3, 2, 1])

    with col_sec:
        st.markdown(f"<div style='{_lbl}'>SECTOR</div>", unsafe_allow_html=True)
        if st.session_state.get('sel_sector') != st.session_state.sector:
            st.session_state.sel_sector = st.session_state.sector
        st.selectbox("Sector", list(FUTURES_GROUPS.keys()), key='sel_sector',
                     label_visibility='collapsed', on_change=_on_sector_change)
    with col_ast:
        st.markdown(f"<div style='{_lbl}'>ASSET</div>", unsafe_allow_html=True)
        selected_label = st.selectbox("Asset", sym_labels, index=current_idx, key='sel_asset', label_visibility='collapsed')
        selected_sym = symbols[sym_labels.index(selected_label)]
        if selected_sym != st.session_state.symbol:
            st.session_state.symbol = selected_sym; st.rerun()
    with col_sort:
        st.markdown(f"<div style='{_lbl}'>SORT</div>", unsafe_allow_html=True)
        sort_by = st.selectbox("Sort", ['Default', 'Day %', 'WTD %', 'MTD %', 'YTD %', 'HV', 'DD',
                    'Sharpe Day', 'Sharpe WTD', 'Sharpe MTD', 'Sharpe YTD'], index=0,
                    key='scanner_sort', label_visibility='collapsed')
    with col_ct:
        st.markdown(f"<div style='{_lbl}'>CHART</div>", unsafe_allow_html=True)
        ct = st.selectbox("Chart", ['Line', 'Bars'], index=0 if st.session_state.chart_type == 'line' else 1,
                          key='chart_select', label_visibility='collapsed')
        st.session_state.chart_type = 'line' if ct == 'Line' else 'bars'

    with st.spinner('Loading market data...'):
        metrics = fetch_sector_data(st.session_state.sector)

    if metrics and sort_by != 'Default':
        sort_map = {'Day %': 'change_day', 'WTD %': 'change_wtd', 'MTD %': 'change_mtd', 'YTD %': 'change_ytd',
                    'HV': 'hist_vol', 'DD': 'current_dd', 'Sharpe Day': 'day_sharpe', 'Sharpe WTD': 'wtd_sharpe',
                    'Sharpe MTD': 'mtd_sharpe', 'Sharpe YTD': 'ytd_sharpe'}
        attr = sort_map.get(sort_by)
        if attr:
            metrics = sorted(metrics, key=lambda m: getattr(m, attr, 0) if not pd.isna(getattr(m, attr, None)) else -999,
                           reverse=(sort_by not in ('HV', 'DD')))

    if metrics:
        col_scan, col_bars = st.columns([55, 45])
        with col_scan: render_scanner_table(metrics, st.session_state.symbol)
        with col_bars: render_return_bars(metrics, sort_by)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    _sym = st.session_state.symbol
    _ds = clean_symbol(_sym); _fn = SYMBOL_NAMES.get(_sym, _sym)
    _chart_hdr = (f"<div style='padding:8px 12px;background:{t.get('bg3','#1a2744')};"
        f"border-left:2px solid {pos_c};font-family:{FONTS};border-radius:4px 4px 0 0'>"
        f"<span style='color:#f8fafc;font-size:11px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase'>{_ds} CHARTS</span>"
        f"<span style='color:{t.get('muted','#475569')};font-size:10px;margin-left:6px;font-weight:400'>{_fn}</span></div>")
    _cfg = {'scrollZoom': True, 'displayModeBar': False, 'responsive': True}

    if is_mobile:
        st.markdown(_chart_hdr, unsafe_allow_html=True)
        with st.spinner('Loading charts...'):
            try:
                fig, levels = create_4_chart_grid(_sym, st.session_state.chart_type, mobile=True)
                st.plotly_chart(fig, use_container_width=True, config=_cfg)
            except Exception as e: st.error(f"Chart error: {str(e)}"); levels = {}
        render_key_levels(_sym, levels); render_news_panel(_sym)
    else:
        col_charts, col_right = st.columns([55, 45])
        with col_charts:
            st.markdown(_chart_hdr, unsafe_allow_html=True)
            with st.spinner('Loading charts...'):
                try:
                    fig, levels = create_4_chart_grid(_sym, st.session_state.chart_type, mobile=False)
                    st.plotly_chart(fig, use_container_width=True, config=_cfg)
                except Exception as e: st.error(f"Chart error: {str(e)}"); levels = {}
        with col_right:
            render_key_levels(_sym, levels); render_news_panel(_sym)
