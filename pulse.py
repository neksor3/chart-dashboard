import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from collections import OrderedDict
import pytz
import logging
from streamlit.components.v1 import html as st_html

from config import FUTURES_GROUPS, THEMES, SYMBOL_NAMES, FONTS, clean_symbol

logger = logging.getLogger(__name__)


def get_theme():
    tn = st.session_state.get('theme', 'Dark')
    return THEMES.get(tn, THEMES['Dark'])


def _s():
    """Return surface palette dict based on current theme."""
    t = get_theme()
    is_light = t.get('mode') == 'light'
    bg = t.get('bg', '#1e1e1e')
    bg2 = t.get('bg2', '#0a0f1a')
    bg3 = t.get('bg3', '#0f172a')
    bdr = t.get('border', '#1e293b')
    txt = t.get('text', '#e2e8f0')
    txt2 = t.get('text2', '#94a3b8')
    muted = t.get('muted', '#475569')
    if is_light:
        return dict(
            bg=bg, bg2=bg2, bg3=bg3, card=bg2,
            border=bdr, text=txt, text2=txt2, muted=muted,
            off_dot='#d1d5db', off_name='#9ca3af', link='#334155',
            bar_bg=bdr, row_alt=bg3, hm_txt=txt,
        )
    else:
        return dict(
            bg=bg, bg2=bg2, bg3=bg3, card=bg3,
            border=bdr, text=txt, text2=txt2, muted=muted,
            off_dot='#3a3a3a', off_name='#4a5568', link='#c9d1d9',
            bar_bg=bg3, row_alt='#131d2e', hm_txt=txt,
        )


def _wrap(body, height):
    s = _s()
    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap' rel='stylesheet'>"
        "<style>"
        "* { margin:0; padding:0; box-sizing:border-box; }"
        f"body {{ background:transparent; font-family:{FONTS}; color:{s['text']}; overflow:hidden; }}"
        f"a {{ color:{s['link']}; text-decoration:none; }}"
        "a:hover { text-decoration:underline; }"
        f"::-webkit-scrollbar {{ width:4px; }}"
        f"::-webkit-scrollbar-track {{ background:{s['bg2']}; }}"
        f"::-webkit-scrollbar-thumb {{ background:{s['border']}; border-radius:2px; }}"
        "</style></head><body>"
        f"{body}"
        "</body></html>"
    )
    st_html(page, height=height)


# ── DATA ─────────────────────────────────────────────────────────────────────

HERO_SYMBOLS = OrderedDict([
    ('ES=F',     {'label': 'S&P 500',  'fmt': ',.0f'}),
    ('NQ=F',     {'label': 'NASDAQ',   'fmt': ',.0f'}),
    ('BTC-USD',  {'label': 'BITCOIN',  'fmt': ',.0f'}),
    ('GC=F',     {'label': 'GOLD',     'fmt': ',.0f'}),
    ('CL=F',     {'label': 'CRUDE',    'fmt': ',.2f'}),
    ('ZN=F',     {'label': '10Y NOTE', 'fmt': ',.2f'}),
    ('USDSGD=X', {'label': 'USD/SGD',  'fmt': ',.4f'}),
    ('^STI',     {'label': 'STI',      'fmt': ',.0f'}),
])

HEATMAP_SECTORS = OrderedDict([
    ('Indices',   ['ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'NKD=F']),
    ('Crypto',    ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']),
    ('Energy',    ['CL=F', 'NG=F', 'RB=F', 'HO=F']),
    ('Metals',    ['GC=F', 'SI=F', 'PL=F', 'HG=F']),
    ('Grains',    ['ZC=F', 'ZS=F', 'ZW=F', 'ZM=F']),
    ('Softs',     ['SB=F', 'KC=F', 'CC=F', 'CT=F']),
    ('Rates',     ['ZB=F', 'ZN=F', 'ZF=F', 'ZT=F']),
    ('FX',        ['6E=F', '6J=F', '6B=F', '6A=F']),
    ('Singapore', ['ES3.SI', 'S68.SI']),
])

SPARKLINE_SYMBOLS = ['ES=F', 'BTC-USD', 'GC=F', 'CL=F', 'USDSGD=X', '^STI']


# ── FETCH ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_pulse_batch():
    all_syms = list(HERO_SYMBOLS.keys())
    for syms in HEATMAP_SECTORS.values():
        all_syms.extend(syms)
    all_syms = list(OrderedDict.fromkeys(all_syms))
    batch_syms = [s for s in all_syms if not s.startswith('^')]
    result = {}
    try:
        batch = yf.download(batch_syms, period='5d', group_by='ticker', threads=True, progress=False)
    except Exception as e:
        logger.warning(f"Pulse batch failed: {e}")
        batch = pd.DataFrame()
    for sym in all_syms:
        try:
            if sym.startswith('^'):
                # Index symbols with ^ often fail in batch download
                hist = yf.Ticker(sym).history(period='5d')
            elif batch.empty:
                hist = yf.Ticker(sym).history(period='5d')
            elif len(all_syms) == 1:
                hist = batch.copy()
            elif sym in batch.columns.get_level_values(0):
                hist = batch[sym].dropna(how='all')
            else:
                hist = yf.Ticker(sym).history(period='5d')
            if hist.empty:
                continue
            current = float(hist['Close'].iloc[-1])
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else current
            change = ((current - prev_close) / prev_close) * 100 if prev_close else 0
            result[sym] = {'price': current, 'change': round(change, 2)}
        except Exception as e:
            logger.debug(f"[{sym}] pulse fetch error: {e}")
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_sparklines():
    result = {}
    for sym in SPARKLINE_SYMBOLS:
        try:
            hist = yf.Ticker(sym).history(period='1mo')
            if not hist.empty:
                result[sym] = hist['Close'].values.tolist()
        except Exception:
            pass
    return result


# ── BREAKOUT SCANNER ─────────────────────────────────────────────────────────

# Symbols shown in the breakout panel — curated cross-asset watchlist
BREAKOUT_SYMBOLS = OrderedDict([
    ('ES=F',    'S&P'),
    ('NQ=F',    'NQ'),
    ('GC=F',    'Gold'),
    ('CL=F',    'Crude'),
    ('ZW=F',    'Wheat'),
    ('NG=F',    'NatGas'),
    ('BTC-USD', 'BTC'),
    ('ZN=F',    '10Y'),
    ('6J=F',    'JPY'),
    ('USDSGD=X','SGDUSD'),
    ('^STI',    'STI'),
    ('ZC=F',    'Corn'),
])


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_breakout_data():
    """
    Fetch 3 months of daily data per symbol.
    3mo gives ~63 trading days — enough for:
      - full previous calendar month (up to ~23 bars)
      - full previous ISO week (up to 5 bars)
      - current period bars
    Returns dict: sym -> DataFrame with OHLC columns
    """
    syms = list(BREAKOUT_SYMBOLS.keys())
    batch_syms = [s for s in syms if not s.startswith('^')]
    result = {}

    # Batch download for non-index symbols
    try:
        batch = yf.download(batch_syms, period='3mo', group_by='ticker',
                            threads=True, progress=False)
    except Exception as e:
        logger.warning(f"Breakout batch download failed: {e}")
        batch = pd.DataFrame()

    for sym in syms:
        try:
            if sym.startswith('^'):
                hist = yf.Ticker(sym).history(period='3mo')
            elif batch.empty:
                hist = yf.Ticker(sym).history(period='3mo')
            elif len(batch_syms) == 1:
                hist = batch.copy()
            elif sym in batch.columns.get_level_values(0):
                hist = batch[sym].dropna(how='all')
            else:
                hist = yf.Ticker(sym).history(period='3mo')

            if hist.empty or len(hist) < 10:
                continue
            # Ensure required columns exist
            for col in ('Open', 'High', 'Low', 'Close'):
                if col not in hist.columns:
                    break
            else:
                result[sym] = hist
        except Exception as e:
            logger.debug(f"[{sym}] breakout fetch error: {e}")

    return result


def _compute_breakout_status(hist, period_type):
    """
    Reuses charts.py _slice_period logic exactly — imported at call time
    to avoid circular import at module load.

    Returns dict with:
      prev_high, prev_low, prev_mid, prev_close  — the reference levels
      curr_price                                  — latest close
      pct_r    : Williams-style %R (0=at low, 100=at high, >100=breakout above, <0=breakdown)
      status   : 'above_high' | 'above_mid' | 'below_mid' | 'below_low'
      reversal : 'buy' | 'sell' | ''
      curr_high, curr_low                         — this period's high/low so far
    """
    from charts import _slice_period  # deferred to avoid circular import

    try:
        prev_period, current_bars = _slice_period(hist, period_type)
        if prev_period is None or prev_period.empty:
            return None

        ph = float(prev_period['High'].max())
        pl = float(prev_period['Low'].min())
        pc = float(prev_period['Close'].iloc[-1])
        pm = (ph + pl) / 2.0
        rng = ph - pl

        # Use latest close as current price
        if current_bars is not None and not current_bars.empty:
            curr_price = float(current_bars['Close'].iloc[-1])
            curr_high  = float(current_bars['High'].max())
            curr_low   = float(current_bars['Low'].min())
        else:
            # Period just started — no current bars yet (e.g. first bar of new week)
            # Fall back to the very last bar in hist
            curr_price = float(hist['Close'].iloc[-1])
            curr_high  = curr_price
            curr_low   = curr_price

        # %R: position within prev period range
        # >100 means above prev high (breakout), <0 means below prev low (breakdown)
        pct_r = ((curr_price - pl) / rng * 100) if rng > 0 else 50.0

        # Status — exact same 4-state logic as charts.py
        if curr_price > ph:
            status = 'above_high'
        elif curr_price < pl:
            status = 'below_low'
        elif curr_price > pm:
            status = 'above_mid'
        else:
            status = 'below_mid'

        # Reversal — exact same logic as charts.py _check_reversal
        reversal = ''
        if current_bars is not None and not current_bars.empty:
            if curr_high > ph and curr_price <= ph:
                reversal = 'sell'   # rejected at high
            elif curr_low < pl and curr_price >= pl:
                reversal = 'buy'    # bounced off low

        return {
            'prev_high':  ph,
            'prev_low':   pl,
            'prev_mid':   pm,
            'prev_close': pc,
            'curr_price': curr_price,
            'curr_high':  curr_high,
            'curr_low':   curr_low,
            'pct_r':      pct_r,
            'status':     status,
            'reversal':   reversal,
        }
    except Exception as e:
        logger.debug(f"_compute_breakout_status ({period_type}) error: {e}")
        return None


def _render_breakout_panel(breakout_data):
    """
    Movers-style breakout panel.

    Layout: two equal columns side by side.
      Left  — WEEK:  above-mid list (green bars, sorted highest %R first)
                     | divider |
                     below-mid list (amber bars, sorted lowest %R first)
      Right — MONTH: same structure.

    Bar semantics (matching movers panel exactly):
      - Green bar  : above prev period midpoint  (%R 50–100+)
      - Amber bar  : below prev period midpoint  (%R 0–50-)
      - Bar width  : proportional to distance from midpoint (not from zero)
      - ▲ BREAK label : price > prev period high  (%R > 100)
      - ▼ BREAK label : price < prev period low   (%R < 0)
      - ● dot       : reversal signal (bounced off low / rejected at high)

    No purple. No ambiguity.
    """
    t   = get_theme()
    s   = _s()
    pos_c = t['pos']    # #4ade80  green
    neg_c = t['neg']    # #f59e0b  amber
    bdr   = s['border']
    bg2   = s['bg2']
    bg3   = s['bg3']
    muted = s['muted']

    # ── Collect results ───────────────────────────────────────────────────────
    week_items  = []   # list of (label, sym, res)
    month_items = []

    for sym, label in BREAKOUT_SYMBOLS.items():
        hist = breakout_data.get(sym)
        if hist is None:
            continue
        wr = _compute_breakout_status(hist, 'week')
        mr = _compute_breakout_status(hist, 'month')
        if wr is not None:
            week_items.append((label, sym, wr))
        if mr is not None:
            month_items.append((label, sym, mr))

    if not week_items and not month_items:
        html = (
            f"<div style='padding:10px 12px;background:{bg2};border:1px solid {bdr};"
            f"border-radius:6px;color:{muted};font-size:10px;font-family:{FONTS}'>"
            f"Breakout data loading…</div>"
        )
        _wrap(html, 40)
        return

    # ── Split into above/below mid, sort ─────────────────────────────────────
    def _split_sort(items):
        # above mid: status in (above_high, above_mid) — sort highest %R first
        above = sorted(
            [(l, sym, r) for l, sym, r in items if r['status'] in ('above_high', 'above_mid')],
            key=lambda x: x[2]['pct_r'], reverse=True
        )
        # below mid: status in (below_mid, below_low) — sort lowest %R first (most broken down)
        below = sorted(
            [(l, sym, r) for l, sym, r in items if r['status'] in ('below_mid', 'below_low')],
            key=lambda x: x[2]['pct_r'], reverse=False
        )
        return above, below

    w_above, w_below = _split_sort(week_items)
    m_above, m_below = _split_sort(month_items)

    # ── Bar row builder (mirrors _render_movers exactly) ─────────────────────
    def _bar_rows(items, color, is_above, max_abs_dist):
        """
        items      : list of (label, sym, res)
        color      : pos_c or neg_c
        is_above   : True = green side, False = amber side
        max_abs_dist: max |%R - 50| across this list, for bar scaling
        """
        if not items:
            return ''
        html = ''
        for label, sym, res in items:
            pct_r  = res['pct_r']
            status = res['status']
            rev    = res['reversal']

            # Distance from midpoint (50) drives bar width, capped at 85%
            dist = abs(pct_r - 50)
            bar_pct = max(dist / max(max_abs_dist, 1) * 85, 8)

            # Label for breakouts
            if status == 'above_high':
                pct_label = f"▲{pct_r:+.0f}%"
            elif status == 'below_low':
                pct_label = f"▼{pct_r:.0f}%"
            else:
                pct_label = f"{pct_r:.0f}%"

            rev_dot = ''
            if rev == 'buy':
                rev_dot = f"<span style='color:{pos_c};font-size:8px;margin-left:2px'>●</span>"
            elif rev == 'sell':
                rev_dot = f"<span style='color:{neg_c};font-size:8px;margin-left:2px'>●</span>"

            grad_dir = '90deg' if is_above else '270deg'

            html += (
                f"<div style='display:flex;align-items:center;padding:4px 0;gap:5px'>"
                # Symbol name
                f"<div style='width:42px;flex-shrink:0'>"
                f"<span style='color:#e2e8f0;font-size:10px;font-weight:600'>{label}</span>"
                f"{rev_dot}"
                f"</div>"
                # Bar
                f"<div style='flex:1;position:relative;height:18px;"
                f"background:{bdr};border-radius:2px;overflow:hidden'>"
                f"<div style='position:absolute;top:0;"
                f"{'left' if is_above else 'right'}:0;"
                f"height:100%;width:{bar_pct:.0f}%;"
                f"background:linear-gradient({grad_dir},{color}20,{color}60);"
                f"border-radius:2px'></div>"
                f"<span style='position:absolute;top:50%;transform:translateY(-50%);"
                f"{'right' if is_above else 'left'}:5px;"
                f"color:{color};font-size:10px;font-weight:700;"
                f"font-variant-numeric:tabular-nums'>"
                f"{pct_label}</span>"
                f"</div>"
                f"</div>"
            )
        return html

    def _period_col(above, below, period_label):
        """Render one period column (WEEK or MONTH) with above + divider + below."""
        # Scale bars independently per side so relative intensity is clear
        above_dists = [abs(r['pct_r'] - 50) for _, _, r in above] or [1]
        below_dists = [abs(r['pct_r'] - 50) for _, _, r in below] or [1]
        max_above = max(above_dists)
        max_below = max(below_dists)

        above_html = _bar_rows(above, pos_c, True,  max_above)
        below_html = _bar_rows(below, neg_c, False, max_below)

        # Section headers
        def _sub_hdr(label, color, count):
            return (
                f"<div style='display:flex;align-items:center;gap:4px;margin-bottom:4px'>"
                f"<span style='color:{color};font-size:9px'>{'▲' if color == pos_c else '▼'}</span>"
                f"<span style='color:#f8fafc;font-size:9px;font-weight:600;"
                f"letter-spacing:0.1em;text-transform:uppercase'>{label}</span>"
                f"<span style='color:{muted};font-size:8px'>({count})</span>"
                f"</div>"
            )

        divider = (
            f"<div style='height:1px;background:{bdr};margin:6px 0'></div>"
            if above and below else ''
        )

        n_rows = len(above) + len(below)

        return (
            f"<div style='background:{bg2};border:1px solid {bdr};"
            f"border-radius:6px;padding:8px 10px;font-family:{FONTS}'>"
            # Period header
            f"<div style='color:{muted};font-size:8px;font-weight:600;"
            f"letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;"
            f"padding-bottom:5px;border-bottom:1px solid {bdr}'>"
            f"PREV {period_label} H/L POSITION</div>"
            # Above mid section
            + (_sub_hdr('ABOVE MID', pos_c, len(above)) + above_html if above else '') +
            divider +
            # Below mid section
            (_sub_hdr('BELOW MID', neg_c, len(below)) + below_html if below else '') +
            f"</div>",
            n_rows
        )

    week_col_html,  w_rows = _period_col(w_above, w_below, 'WEEK')
    month_col_html, m_rows = _period_col(m_above, m_below, 'MONTH')

    # Height: header(24) + sub-hdr(20) + rows(22each) + divider(13) + padding(20)
    max_rows = max(w_rows, m_rows, 1)
    # above + below each have a sub-header; divider if both present
    n_above_max = max(len(w_above), len(m_above))
    n_below_max = max(len(w_below), len(m_below))
    has_both = (n_above_max > 0 and n_below_max > 0)
    height = (
        24                          # period header
        + (20 if n_above_max else 0)  # above sub-header
        + n_above_max * 26
        + (13 if has_both else 0)   # divider
        + (20 if n_below_max else 0)  # below sub-header
        + n_below_max * 26
        + 40                        # padding
    )
    height = max(height, 80)

    html = (
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;"
        f"font-family:{FONTS}'>"
        f"{week_col_html}"
        f"{month_col_html}"
        f"</div>"
    )
    _wrap(html, height)


# ── SVG SPARKLINE ────────────────────────────────────────────────────────────

def _svg_sparkline(data, width=100, height=28, pos_color='#4ade80', neg_color='#f59e0b'):
    if not data or len(data) < 2:
        return ''
    vals = np.array(data)
    mn, mx = vals.min(), vals.max()
    rng = mx - mn if mx != mn else 1
    n = len(vals)
    xs = [round(i * width / (n - 1), 1) for i in range(n)]
    ys = [round(height - 2 - (v - mn) / rng * (height - 4), 1) for v in vals]
    color = pos_color if vals[-1] >= vals[0] else neg_color
    points = ' '.join(f'{x},{y}' for x, y in zip(xs, ys))
    fill_points = f'0,{height} {points} {width},{height}'
    uid = f'sp{abs(hash(tuple(data))) % 99999}'
    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"
        f"<defs><linearGradient id='{uid}' x1='0' y1='0' x2='0' y2='1'>"
        f"<stop offset='0%' stop-color='{color}' stop-opacity='0.3'/>"
        f"<stop offset='100%' stop-color='{color}' stop-opacity='0'/>"
        f"</linearGradient></defs>"
        f"<polygon points='{fill_points}' fill='url(#{uid})'/>"
        f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='1.5' stroke-linejoin='round' stroke-linecap='round'/>"
        f"<circle cx='{xs[-1]}' cy='{ys[-1]}' r='2' fill='{color}'/>"
        f"</svg>"
    )


# ── MARKET STATUS ────────────────────────────────────────────────────────────

# SGX market holidays 2026 (excludes weekends)
_SG_HOLIDAYS = [
    (2026,1,1,'New Year'),  (2026,2,17,'CNY'), (2026,2,18,'CNY'),
    (2026,3,21,'Hari Raya Puasa'), (2026,4,3,'Good Friday'),
    (2026,5,1,'Labour Day'), (2026,5,27,'Hari Raya Haji'),
    (2026,6,1,'Vesak Day'), (2026,8,10,'National Day'),
    (2026,11,9,'Deepavali'), (2026,12,25,'Christmas'),
]
# NYSE market holidays 2026
_US_HOLIDAYS = [
    (2026,1,1,'New Year'), (2026,1,19,'MLK Day'), (2026,2,16,"Presidents'"),
    (2026,4,3,'Good Friday'), (2026,5,25,'Memorial Day'), (2026,6,19,'Juneteenth'),
    (2026,7,3,'Independence'), (2026,9,7,'Labor Day'), (2026,11,26,'Thanksgiving'),
    (2026,12,25,'Christmas'),
]

def _next_holiday(holidays, today):
    from datetime import date
    for y, m, d, name in holidays:
        h = date(y, m, d)
        if h >= today:
            delta = (h - today).days
            return name, h.strftime('%d %b'), delta
    return None, None, None

def _market_status():
    from datetime import date
    now_utc = datetime.now(pytz.utc)
    today = now_utc.astimezone(pytz.timezone('Asia/Singapore')).date()
    results = []
    markets = [
        ('SG',  'Asia/Singapore', 9, 0,  17, 0,  _SG_HOLIDAYS, '#fb7185'),  # rose
        ('US',  'US/Eastern',     9, 30, 16, 0,  _US_HOLIDAYS, '#60a5fa'),  # blue
    ]
    for name, tz_str, oh, om, ch, cm, holidays, dot_color in markets:
        tz = pytz.timezone(tz_str)
        local = now_utc.astimezone(tz)
        is_weekend = local.weekday() >= 5
        open_time = local.replace(hour=oh, minute=om, second=0)
        close_time = local.replace(hour=ch, minute=cm, second=0)
        is_open = not is_weekend and open_time <= local <= close_time
        hol_name, hol_date, hol_days = _next_holiday(holidays, today)
        results.append({
            'name': name, 'time': local.strftime('%H:%M'), 'open': is_open,
            'dot_color': dot_color,
            'hol_name': hol_name, 'hol_date': hol_date, 'hol_days': hol_days,
        })
    return results


# ── RENDER ───────────────────────────────────────────────────────────────────

def _render_market_status_bar():
    markets = _market_status()
    t = get_theme()
    s = _s()
    is_light = t.get('mode') == 'light'
    dots = ''
    for m in markets:
        # Dot color: red for SG, blue for US (always their color, brighter when open)
        dot_c = m['dot_color']
        glow = ''
        pulse = ''
        # Country name: strong text
        nc = ('#0f172a' if is_light else '#f8fafc') if m['open'] else ('#64748b' if is_light else '#f8fafc')
        # Time: always visible
        tc = ('#334155' if is_light else '#f8fafc') if m['open'] else ('#94a3b8' if is_light else '#f8fafc')
        dots += (
            f"<div style='display:flex;align-items:center;gap:4px'>"
            f"<div style='width:6px;height:6px;border-radius:50%;background:{dot_c}'></div>"
            f"<span style='color:{nc};font-size:9px;font-weight:600;letter-spacing:0.06em'>{m['name']}</span>"
            f"<span style='color:{tc};font-size:9px;font-weight:600'>{m['time']}</span>"
            f"</div>"
        )

    # Next holidays
    hols = ''
    for m in markets:
        if m['hol_name']:
            dot_c = m['dot_color']
            hols += (
                f"<div style='display:flex;align-items:center;gap:4px'>"
                f"<div style='width:6px;height:6px;border-radius:2px;background:{dot_c}'></div>"
                f"<span style='color:#e2e8f0;font-size:8px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase'>"
                f"{m['name']} {m['hol_name']} {m['hol_date']}</span>"
                f"</div>"
            )

    html = (
        "<style>@keyframes pulse-dot{0%,100%{opacity:1}50%{opacity:0.4}}</style>"
        f"<div style='display:flex;gap:16px;flex-wrap:wrap;padding:6px 12px;background:{s['bg2']};"
        f"border:1px solid {s['border']};border-radius:4px;align-items:center'>"
        f"<span style='color:#f8fafc;font-size:8px;font-weight:600;letter-spacing:0.1em;"
        f"align-self:center'>MARKETS</span>"
        f"{dots}"
        f"<span style='color:{s['border']};font-size:10px;align-self:center'>│</span>"
        f"{hols}</div>"
    )
    _wrap(html, 36)


def _render_hero_row(data):
    t = get_theme()
    s = _s()
    pos_c, neg_c = t['pos'], t['neg']
    is_light = get_theme().get('mode') == 'light'

    cards = ''
    for sym, cfg in HERO_SYMBOLS.items():
        d = data.get(sym)
        if not d:
            continue
        price, change = d['price'], d['change']
        color = pos_c if change >= 0 else neg_c
        sign = '+' if change >= 0 else ''
        arrow = '&#9650;' if change > 0 else '&#9660;' if change < 0 else '&#8211;'
        price_str = f'{price:{cfg["fmt"]}}'

        if is_light:
            card_bg = f'background:{s["bg2"]};'
            glow = ''
        else:
            card_bg = f'background:linear-gradient(135deg,{s["bg3"]},{s["bg2"]});'
            glow = f'text-shadow:0 0 20px {color}40;'

        # Force high contrast price color
        price_c = '#0f172a' if is_light else '#f8fafc'
        label_c = '#475569' if is_light else '#f8fafc'

        cards += (
            f"<div style='flex:1;min-width:110px;padding:10px 12px;"
            f"{card_bg}"
            f"border:1px solid {s['border']};border-radius:6px;position:relative;overflow:hidden'>"
            f"<div style='position:absolute;top:0;left:0;right:0;height:2px;"
            f"background:linear-gradient(90deg,transparent,{color}40,transparent)'></div>"
            f"<div style='color:{label_c};font-size:8px;font-weight:600;letter-spacing:0.12em;"
            f"text-transform:uppercase;margin-bottom:4px'>{cfg['label']}</div>"
            f"<div style='color:{price_c};font-size:17px;font-weight:700;"
            f"font-variant-numeric:tabular-nums;letter-spacing:-0.02em;"
            f"{glow}'>{price_str}</div>"
            f"<div style='margin-top:3px;display:flex;align-items:center;gap:4px'>"
            f"<span style='color:{color};font-size:10px'>{arrow}</span>"
            f"<span style='color:{color};font-size:11px;font-weight:700'>{sign}{change:.2f}%</span>"
            f"</div></div>"
        )
    html = f"<div style='display:flex;gap:6px;flex-wrap:wrap'>{cards}</div>"
    _wrap(html, 88)


def _render_sparkline_row(spark_data, pulse_data):
    t = get_theme(); s = _s()
    pos_c, neg_c = t['pos'], t['neg']
    cards = ''
    for sym in SPARKLINE_SYMBOLS:
        sdata = spark_data.get(sym)
        pdata = pulse_data.get(sym)
        if not sdata or not pdata:
            continue
        short = clean_symbol(sym)
        change = pdata['change']
        color = pos_c if change >= 0 else neg_c
        sign = '+' if change >= 0 else ''
        svg = _svg_sparkline(sdata, width=80, height=22, pos_color=pos_c, neg_color=neg_c)
        cards += (
            f"<div style='flex:1;min-width:120px;padding:5px 8px;background:{s['card']};"
            f"border:1px solid {s['border']};border-radius:4px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:2px'>"
            f"<span style='color:{s['text2']};font-size:8px;font-weight:600;letter-spacing:0.06em'>{short}</span>"
            f"<span style='color:{color};font-size:8px;font-weight:700'>{sign}{change:.2f}%</span>"
            f"</div>"
            f"{svg}"
            f"</div>"
        )
    html = f"<div style='display:flex;gap:5px;flex-wrap:wrap'>{cards}</div>"
    _wrap(html, 58)

def _render_heatmap_grid(data):
    t = get_theme()
    s = _s()
    pos_c, neg_c = t['pos'], t['neg']
    is_light = get_theme().get('mode') == 'light'

    def _bg(change, pos, neg):
        abs_c = min(abs(change), 5)
        opacity = 0.15 + (abs_c / 5) * 0.55 if is_light else 0.25 + (abs_c / 5) * 0.75
        base = pos if change >= 0 else neg
        r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
        return f'rgba({r},{g},{b},{opacity:.2f})', abs_c

    sectors_html = ''
    active_sectors = 0
    for sector, syms in HEATMAP_SECTORS.items():
        cells = ''
        has_data = False
        for sym in syms:
            d = data.get(sym)
            if not d:
                continue
            has_data = True
            change = d['change']
            name = clean_symbol(sym)
            bg, intensity = _bg(change, pos_c, neg_c)
            # Dynamic text: use high-contrast on strong backgrounds
            if intensity > 3:
                name_c = '#ffffff' if not is_light else '#1e293b'
                val_c = '#ffffff' if not is_light else '#1e293b'
            else:
                name_c = s['hm_txt']
                val_c = pos_c if change >= 0 else neg_c
            sign = '+' if change >= 0 else ''
            cell_border = 'rgba(0,0,0,0.06)' if is_light else 'rgba(255,255,255,0.04)'
            cells += (
                f"<div style='flex:1;min-width:60px;padding:6px 8px;background:{bg};"
                f"border-radius:3px;border:1px solid {cell_border};text-align:center'>"
                f"<div style='color:{name_c};font-size:10px;font-weight:600'>{name}</div>"
                f"<div style='color:{val_c};font-size:11px;font-weight:700;margin-top:1px;"
                f"font-variant-numeric:tabular-nums'>{sign}{change:.2f}%</div>"
                f"</div>"
            )
        if has_data:
            active_sectors += 1
            sectors_html += (
                f"<div style='margin-bottom:6px'>"
                f"<div style='color:{s['muted']};font-size:8px;font-weight:600;letter-spacing:0.1em;"
                f"text-transform:uppercase;margin-bottom:4px;padding-left:2px'>{sector}</div>"
                f"<div style='display:flex;gap:4px;flex-wrap:wrap'>{cells}</div>"
                f"</div>"
            )

    html = (
        f"<div style='padding:10px 12px;background:{s['bg2']};border:1px solid {s['border']};border-radius:6px'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>"
        f"<span style='color:#f8fafc;font-size:9px;font-weight:600;letter-spacing:0.1em;"
        f"text-transform:uppercase'>MARKET HEATMAP</span>"
        f"<span style='color:{s['muted']};font-size:8px'>Day Change %</span>"
        f"</div>"
        f"{sectors_html}</div>"
    )
    _wrap(html, 46 * active_sectors + 50)


def _render_movers(data):
    t = get_theme()
    s = _s()
    pos_c, neg_c = t['pos'], t['neg']
    all_items = [(sym, d['change']) for sym, d in data.items()]
    all_items.sort(key=lambda x: x[1], reverse=True)
    gainers = [(sc, c) for sc, c in all_items if c > 0][:5]
    losers = [(sc, c) for sc, c in all_items if c < 0][-5:]
    losers.reverse()

    def _rows(items, color, is_gain):
        if not items:
            return f"<div style='color:{s['muted']};font-size:10px;padding:8px'>No data</div>"
        max_abs = max(abs(c) for _, c in items) or 1
        html = ''
        for sym, change in items:
            short = clean_symbol(sym)
            bar_pct = max(abs(change) / max_abs * 85, 8)
            sign = '+' if change >= 0 else ''
            align = 'left' if is_gain else 'right'
            grad_dir = '90deg' if is_gain else '270deg'
            html += (
                f"<div style='display:flex;align-items:center;padding:5px 0;gap:6px'>"
                f"<div style='width:45px;flex-shrink:0'>"
                f"<span style='color:{s['text']};font-size:10px;font-weight:600'>{short}</span></div>"
                f"<div style='flex:1;position:relative;height:18px;background:{s['bar_bg']};border-radius:2px;overflow:hidden'>"
                f"<div style='position:absolute;top:0;{align}:0;height:100%;width:{bar_pct}%;"
                f"background:linear-gradient({grad_dir},{color}20,{color}60);border-radius:2px'></div>"
                f"<span style='position:absolute;top:50%;transform:translateY(-50%);{align}:6px;"
                f"color:{color};font-size:10px;font-weight:700;font-variant-numeric:tabular-nums'>"
                f"{sign}{change:.2f}%</span></div></div>"
            )
        return html

    gain_html = _rows(gainers, pos_c, True)
    lose_html = _rows(losers, neg_c, False)
    n_rows = max(len(gainers), len(losers), 1)

    html = (
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;"
        f"padding:10px 12px;background:{s['bg2']};border:1px solid {s['border']};border-radius:6px'>"
        f"<div>"
        f"<div style='color:#f8fafc;font-size:9px;font-weight:600;letter-spacing:0.1em;"
        f"margin-bottom:6px;display:flex;align-items:center;gap:4px'>"
        f"<span style='color:{pos_c};font-size:12px'>&#9650;</span> TOP GAINERS</div>"
        f"{gain_html}</div>"
        f"<div>"
        f"<div style='color:#f8fafc;font-size:9px;font-weight:600;letter-spacing:0.1em;"
        f"margin-bottom:6px;display:flex;align-items:center;gap:4px'>"
        f"<span style='color:{neg_c};font-size:12px'>&#9660;</span> TOP LOSERS</div>"
        f"{lose_html}</div></div>"
    )
    _wrap(html, 28 * n_rows + 50)


def _render_pulse_news():
    from news import fetch_rss_feed
    t = get_theme(); s = _s()
    pos_c = t['pos']

    feeds = [
        ('CNA',            'https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6511'),
        ('Straits Times',  'https://www.straitstimes.com/news/business/rss.xml'),
        ('Bloomberg',      'https://feeds.bloomberg.com/markets/news.rss'),
        ('FT',             'https://www.ft.com/rss/home'),
    ]

    all_items = []
    for name, url in feeds:
        all_items.extend(fetch_rss_feed(name, url))

    all_items.sort(key=lambda x: x.get('sort_key', ''), reverse=True)
    all_items = all_items[:15]
    if not all_items:
        return

    rows = ''
    for i, item in enumerate(all_items):
        bg = s['bg2'] if i % 2 == 0 else s['row_alt']
        rows += (
            f"<div style='padding:4px 10px;background:{bg};border-bottom:1px solid {s['border']}10;"
            f"display:flex;align-items:baseline;gap:8px;font-family:{FONTS};white-space:nowrap;overflow:hidden'>"
            f"<span style='font-size:9px;flex-shrink:0;width:170px;display:flex;gap:6px;align-items:baseline'>"
            f"<span style='color:{pos_c};font-weight:600'>{item['source']}</span>"
            f"<span style='color:{s['muted']}'>{item['date']}</span></span>"
            f"<a href='{item['url']}' target='_blank' style='color:{s['link']};text-decoration:none;"
            f"font-size:10.5px;font-weight:500;overflow:hidden;text-overflow:ellipsis'>{item['title']}</a>"
            f"</div>"
        )

    html = (
        f"<div style='background:{s['bg2']};border:1px solid {s['border']};border-radius:6px;overflow:hidden;"
        f"font-family:{FONTS}'>"
        f"<div style='padding:6px 10px;display:flex;justify-content:space-between;align-items:center;"
        f"border-bottom:1px solid {s['border']}'>"
        f"<span style='color:#f8fafc;font-size:9px;font-weight:600;letter-spacing:0.1em'>LATEST</span>"
        f"<span style='color:{s['muted']};font-size:9px;font-weight:500'>{len(all_items)}</span></div>"
        f"<div style='max-height:160px;overflow-y:auto'>{rows}</div></div>"
    )
    _wrap(html, 190)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def render_pulse_tab(is_mobile):
    with st.spinner('Scanning markets...'):
        data          = _fetch_pulse_batch()
        spark_data    = _fetch_sparklines()
        breakout_data = _fetch_breakout_data()

    if not data:
        st.info('Markets data loading — try refreshing.')
        return

    _render_market_status_bar()
    _render_hero_row(data)
    if spark_data:
        _render_sparkline_row(spark_data, data)

    if is_mobile:
        _render_breakout_panel(breakout_data)
        _render_movers(data)
        _render_pulse_news()
        _render_heatmap_grid(data)
    else:
        # Row: movers (left) | breakout week+month (centre) | news (right)
        col_movers, col_breakout, col_news = st.columns([22, 45, 33])
        with col_movers:
            _render_movers(data)
        with col_breakout:
            _render_breakout_panel(breakout_data)
        with col_news:
            _render_pulse_news()

        _render_heatmap_grid(data)
