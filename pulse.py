import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from collections import OrderedDict
import pytz
import logging
from html import escape as html_escape

from config import FUTURES_GROUPS, THEMES, SYMBOL_NAMES, FONTS, clean_symbol

logger = logging.getLogger(__name__)

# =============================================================================
# THEME
# =============================================================================

def get_theme():
    tn = st.session_state.get('theme', 'Emerald / Amber')
    return THEMES.get(tn, THEMES['Emerald / Amber'])

# =============================================================================
# DATA
# =============================================================================

HERO_SYMBOLS = OrderedDict([
    ('ES=F',     {'label': 'S&P 500',    'fmt': ',.0f'}),
    ('NQ=F',     {'label': 'NASDAQ',     'fmt': ',.0f'}),
    ('BTC-USD',  {'label': 'BITCOIN',    'fmt': ',.0f'}),
    ('GC=F',     {'label': 'GOLD',       'fmt': ',.0f'}),
    ('CL=F',     {'label': 'CRUDE',      'fmt': ',.2f'}),
    ('ZN=F',     {'label': '10Y NOTE',   'fmt': ',.2f'}),
    ('USDSGD=X', {'label': 'USD/SGD',    'fmt': ',.4f'}),
    ('6E=F',     {'label': 'EUR/USD',    'fmt': ',.4f'}),
])

HEATMAP_SECTORS = OrderedDict([
    ('Indices',    ['ES=F', 'NQ=F', 'YM=F', 'RTY=F', 'NKD=F']),
    ('Crypto',     ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD']),
    ('Energy',     ['CL=F', 'NG=F', 'RB=F', 'HO=F']),
    ('Metals',     ['GC=F', 'SI=F', 'PL=F', 'HG=F']),
    ('Grains',     ['ZC=F', 'ZS=F', 'ZW=F', 'ZM=F']),
    ('Softs',      ['SB=F', 'KC=F', 'CC=F', 'CT=F']),
    ('Rates',      ['ZB=F', 'ZN=F', 'ZF=F', 'ZT=F']),
    ('FX',         ['6E=F', '6J=F', '6B=F', '6A=F']),
    ('Singapore',  ['ES3.SI', 'S68.SI']),
])

SPARKLINE_SYMBOLS = ['ES=F', 'BTC-USD', 'GC=F', 'CL=F', 'ZN=F']


@st.cache_data(ttl=180, show_spinner=False)
def _fetch_pulse_batch():
    """Fetch all pulse symbols in one batch — returns dict of {symbol: {price, change, open, high, low}}."""
    all_syms = list(HERO_SYMBOLS.keys())
    for syms in HEATMAP_SECTORS.values():
        all_syms.extend(syms)
    all_syms = list(OrderedDict.fromkeys(all_syms))  # dedupe preserving order

    result = {}

    # Batch download daily data
    try:
        batch = yf.download(all_syms, period='5d', group_by='ticker', threads=True, progress=False)
    except Exception as e:
        logger.warning(f"Pulse batch download failed: {e}")
        batch = pd.DataFrame()

    for sym in all_syms:
        try:
            if batch.empty:
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
            day_open = float(hist['Open'].iloc[-1])
            day_high = float(hist['High'].iloc[-1])
            day_low = float(hist['Low'].iloc[-1])
            change = ((current - prev_close) / prev_close) * 100 if prev_close else 0
            intra_change = ((current - day_open) / day_open) * 100 if day_open else 0

            result[sym] = {
                'price': current,
                'change': round(change, 2),
                'intra': round(intra_change, 2),
                'open': day_open,
                'high': day_high,
                'low': day_low,
                'prev_close': prev_close,
            }
        except Exception as e:
            logger.debug(f"[{sym}] pulse fetch error: {e}")

    return result


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_sparklines():
    """Fetch 30-day close data for sparkline charts."""
    result = {}
    for sym in SPARKLINE_SYMBOLS:
        try:
            hist = yf.Ticker(sym).history(period='1mo')
            if not hist.empty:
                result[sym] = hist['Close'].values.tolist()
        except Exception:
            pass
    return result


# =============================================================================
# SVG SPARKLINE GENERATOR
# =============================================================================

def _svg_sparkline(data, width=120, height=32, pos_color='#4ade80', neg_color='#f59e0b'):
    """Generate an SVG sparkline with gradient fill."""
    if not data or len(data) < 2:
        return ''

    vals = np.array(data)
    mn, mx = vals.min(), vals.max()
    rng = mx - mn if mx != mn else 1

    # Normalize to SVG coordinates
    n = len(vals)
    xs = [round(i * width / (n - 1), 1) for i in range(n)]
    ys = [round(height - 2 - (v - mn) / rng * (height - 4), 1) for v in vals]

    # Color based on start vs end
    color = pos_color if vals[-1] >= vals[0] else neg_color

    # Build path
    points = ' '.join(f'{x},{y}' for x, y in zip(xs, ys))
    fill_points = f'0,{height} {points} {width},{height}'

    uid = f'spark_{hash(tuple(data)) % 99999}'

    return f"""<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' style='display:block'>
        <defs>
            <linearGradient id='{uid}' x1='0' y1='0' x2='0' y2='1'>
                <stop offset='0%' stop-color='{color}' stop-opacity='0.3'/>
                <stop offset='100%' stop-color='{color}' stop-opacity='0'/>
            </linearGradient>
        </defs>
        <polygon points='{fill_points}' fill='url(#{uid})'/>
        <polyline points='{points}' fill='none' stroke='{color}' stroke-width='1.5' stroke-linejoin='round' stroke-linecap='round'/>
        <circle cx='{xs[-1]}' cy='{ys[-1]}' r='2' fill='{color}'/>
    </svg>"""


# =============================================================================
# MARKET STATUS
# =============================================================================

def _market_status():
    """Return market open/close status for major regions."""
    now_utc = datetime.now(pytz.utc)
    results = []

    markets = [
        ('US',        'US/Eastern',      9, 30, 16, 0),
        ('LONDON',    'Europe/London',   8, 0,  16, 30),
        ('EUROPE',    'Europe/Berlin',   9, 0,  17, 30),
        ('TOKYO',     'Asia/Tokyo',      9, 0,  15, 0),
        ('HK',        'Asia/Hong_Kong',  9, 30, 16, 0),
        ('SINGAPORE', 'Asia/Singapore',  9, 0,  17, 0),
    ]

    for name, tz_str, oh, om, ch, cm in markets:
        tz = pytz.timezone(tz_str)
        local = now_utc.astimezone(tz)
        local_time = local.strftime('%H:%M')

        # Check if weekend
        is_weekend = local.weekday() >= 5
        open_time = local.replace(hour=oh, minute=om, second=0)
        close_time = local.replace(hour=ch, minute=cm, second=0)
        is_open = not is_weekend and open_time <= local <= close_time

        results.append({
            'name': name,
            'time': local_time,
            'open': is_open,
        })

    return results


# =============================================================================
# RENDER FUNCTIONS
# =============================================================================

def _render_market_status_bar():
    """Animated market status dots."""
    markets = _market_status()
    t = get_theme()

    dots_html = ''
    for m in markets:
        color = t['pos'] if m['open'] else '#3a3a3a'
        glow = f'0 0 6px {t["pos"]}80' if m['open'] else 'none'
        pulse = 'animation:pulse-dot 2s ease-in-out infinite;' if m['open'] else ''
        dots_html += f"""
            <div style='display:flex;align-items:center;gap:5px'>
                <div style='width:6px;height:6px;border-radius:50%;background:{color};box-shadow:{glow};{pulse}'></div>
                <span style='color:{"#e2e8f0" if m["open"] else "#4a5568"};font-size:9px;font-weight:600;letter-spacing:0.08em'>{m['name']}</span>
                <span style='color:#4a5568;font-size:8px'>{m['time']}</span>
            </div>"""

    html = f"""
    <style>@keyframes pulse-dot {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.4; }} }}</style>
    <div style='display:flex;gap:16px;flex-wrap:wrap;padding:6px 12px;background:#0a0f1a;border:1px solid #1e293b;border-radius:4px;margin-bottom:10px;font-family:{FONTS}'>
        <span style='color:#334155;font-size:8px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;align-self:center'>MARKETS</span>
        {dots_html}
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


def _render_hero_row(data):
    """Big glowing headline numbers."""
    t = get_theme()
    pos_c, neg_c = t['pos'], t['neg']

    cards = ''
    for sym, cfg in HERO_SYMBOLS.items():
        d = data.get(sym)
        if not d:
            continue

        price = d['price']
        change = d['change']
        fmt = cfg['fmt']
        label = cfg['label']

        color = pos_c if change >= 0 else neg_c
        sign = '+' if change >= 0 else ''
        glow = f'text-shadow: 0 0 20px {color}40, 0 0 40px {color}20;'
        arrow = '▲' if change > 0 else '▼' if change < 0 else '–'

        # Format price
        price_str = f'{price:{fmt}}'

        cards += f"""
        <div style='flex:1;min-width:110px;padding:10px 12px;background:linear-gradient(135deg, #0f172a 0%, #0a0f1a 100%);
                    border:1px solid #1e293b;border-radius:6px;position:relative;overflow:hidden'>
            <div style='position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg, transparent, {color}40, transparent)'></div>
            <div style='color:#4a5568;font-size:8px;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:4px'>{label}</div>
            <div style='color:#f1f5f9;font-size:17px;font-weight:700;font-variant-numeric:tabular-nums;letter-spacing:-0.02em;{glow}'>{price_str}</div>
            <div style='margin-top:3px;display:flex;align-items:center;gap:4px'>
                <span style='color:{color};font-size:10px'>{arrow}</span>
                <span style='color:{color};font-size:11px;font-weight:700'>{sign}{change:.2f}%</span>
            </div>
        </div>"""

    html = f"""<div style='display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;font-family:{FONTS}'>{cards}</div>"""
    st.markdown(html, unsafe_allow_html=True)


def _render_heatmap_grid(data):
    """Treemap-style heatmap grid with color-intensity coding."""
    t = get_theme()
    pos_c, neg_c = t['pos'], t['neg']

    def _intensity_color(change, pos, neg):
        """Return color with opacity based on magnitude."""
        abs_c = min(abs(change), 5)  # cap at 5% for color scaling
        opacity = 0.25 + (abs_c / 5) * 0.75  # 0.25 to 1.0
        base = pos if change >= 0 else neg
        # Convert hex to rgba
        r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
        return f'rgba({r},{g},{b},{opacity:.2f})'

    sectors_html = ''
    for sector, syms in HEATMAP_SECTORS.items():
        cells = ''
        for sym in syms:
            d = data.get(sym)
            if not d:
                continue
            change = d['change']
            name = clean_symbol(sym)
            color = _intensity_color(change, pos_c, neg_c)
            text_color = pos_c if change >= 0 else neg_c
            sign = '+' if change >= 0 else ''

            cells += f"""
            <div style='flex:1;min-width:60px;padding:6px 8px;background:{color};border-radius:3px;
                        border:1px solid rgba(255,255,255,0.04);text-align:center;cursor:default;
                        transition:all 0.15s ease'>
                <div style='color:#e2e8f0;font-size:10px;font-weight:600;letter-spacing:0.02em'>{name}</div>
                <div style='color:{text_color};font-size:11px;font-weight:700;margin-top:1px;font-variant-numeric:tabular-nums'>{sign}{change:.2f}%</div>
            </div>"""

        sectors_html += f"""
        <div style='margin-bottom:6px'>
            <div style='color:#334155;font-size:8px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;padding-left:2px'>{sector}</div>
            <div style='display:flex;gap:4px;flex-wrap:wrap'>{cells}</div>
        </div>"""

    html = f"""
    <div style='padding:10px 12px;background:#0a0f1a;border:1px solid #1e293b;border-radius:6px;margin-bottom:10px;font-family:{FONTS}'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>
            <span style='color:#64748b;font-size:9px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase'>MARKET HEATMAP</span>
            <span style='color:#334155;font-size:8px'>Day Change %</span>
        </div>
        {sectors_html}
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


def _render_movers(data):
    """Top gainers and losers with animated bars."""
    t = get_theme()
    pos_c, neg_c = t['pos'], t['neg']

    # Collect all changes
    all_items = []
    for sym, d in data.items():
        all_items.append((sym, d['change'], d['price']))

    all_items.sort(key=lambda x: x[1], reverse=True)
    gainers = [(s, c, p) for s, c, p in all_items if c > 0][:5]
    losers = [(s, c, p) for s, c, p in all_items if c < 0][-5:]
    losers.reverse()  # worst first

    def _mover_col(items, label, color, is_positive=True):
        rows = ''
        if not items:
            return f"<div style='color:#334155;font-size:10px;padding:8px'>No {label.lower()}</div>"

        max_abs = max(abs(c) for _, c, _ in items) or 1
        for sym, change, price in items:
            name = SYMBOL_NAMES.get(sym, clean_symbol(sym))
            short = clean_symbol(sym)
            bar_pct = max(abs(change) / max_abs * 85, 8)
            sign = '+' if change >= 0 else ''
            direction = 'right' if is_positive else 'left'

            rows += f"""
            <div style='display:flex;align-items:center;padding:5px 0;gap:6px'>
                <div style='width:40px;flex-shrink:0'>
                    <span style='color:#e2e8f0;font-size:10px;font-weight:600'>{short}</span>
                </div>
                <div style='flex:1;position:relative;height:18px;background:#0f172a;border-radius:2px;overflow:hidden'>
                    <div style='position:absolute;top:0;{"left" if is_positive else "right"}:0;height:100%;width:{bar_pct}%;
                                background:linear-gradient({"90deg" if is_positive else "270deg"}, {color}20, {color}60);
                                border-radius:2px;transition:width 0.6s ease'></div>
                    <span style='position:absolute;top:50%;transform:translateY(-50%);{"left" if is_positive else "right"}:6px;
                                 color:{color};font-size:10px;font-weight:700;font-variant-numeric:tabular-nums'>{sign}{change:.2f}%</span>
                </div>
            </div>"""

        return f"""
        <div>
            <div style='color:{color};font-size:9px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;
                         display:flex;align-items:center;gap:4px'>
                <span style='font-size:12px'>{'▲' if is_positive else '▼'}</span> {label}
            </div>
            {rows}
        </div>"""

    gainers_html = _mover_col(gainers, 'TOP GAINERS', pos_c, True)
    losers_html = _mover_col(losers, 'TOP LOSERS', neg_c, False)

    html = f"""
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:10px 12px;background:#0a0f1a;
                border:1px solid #1e293b;border-radius:6px;margin-bottom:10px;font-family:{FONTS}'>
        {gainers_html}
        {losers_html}
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


def _render_sparklines(spark_data, pulse_data):
    """Sparkline mini charts for key assets."""
    t = get_theme()
    pos_c, neg_c = t['pos'], t['neg']

    cards = ''
    for sym in SPARKLINE_SYMBOLS:
        sdata = spark_data.get(sym)
        pdata = pulse_data.get(sym)
        if not sdata or not pdata:
            continue

        name = SYMBOL_NAMES.get(sym, clean_symbol(sym))
        short = clean_symbol(sym)
        change = pdata['change']
        color = pos_c if change >= 0 else neg_c
        sign = '+' if change >= 0 else ''
        svg = _svg_sparkline(sdata, width=100, height=28, pos_color=pos_c, neg_color=neg_c)

        cards += f"""
        <div style='flex:1;min-width:140px;padding:8px 10px;background:#0f172a;border:1px solid #1e293b;border-radius:4px'>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px'>
                <span style='color:#94a3b8;font-size:9px;font-weight:600;letter-spacing:0.06em'>{short}</span>
                <span style='color:{color};font-size:9px;font-weight:700'>{sign}{change:.2f}%</span>
            </div>
            {svg}
            <div style='color:#475569;font-size:7px;margin-top:2px;text-align:right'>30 DAY</div>
        </div>"""

    html = f"""
    <div style='display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;font-family:{FONTS}'>
        {cards}
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


def _render_pulse_news():
    """Quick news headlines from RSS (reuses news module)."""
    from news import NEWS_FEEDS, fetch_rss_feed
    t = get_theme()
    pos_c = t['pos']

    all_items = []
    # Pull from Global and Singapore feeds only for speed
    for region in ['Global', 'Singapore']:
        feeds = NEWS_FEEDS.get(region, [])
        for name, url in feeds:
            items = fetch_rss_feed(name, url)
            all_items.extend(items)

    all_items.sort(key=lambda x: x['date'], reverse=True)
    all_items = all_items[:12]

    if not all_items:
        return

    rows = ''
    for i, item in enumerate(all_items):
        bg = '#0a0f1a' if i % 2 == 0 else '#0d1321'
        link = f"<a href='{item['url']}' target='_blank' style='color:#c9d1d9;text-decoration:none;font-size:10px;line-height:1.35'>{item['title']}</a>" if item['url'] else f"<span style='color:#c9d1d9;font-size:10px'>{item['title']}</span>"
        rows += f"""
        <div style='padding:5px 10px;background:{bg};border-bottom:1px solid #1e293b0a'>
            <div>{link}</div>
            <div style='font-size:7px;margin-top:1px'>
                <span style='color:{pos_c};font-weight:600'>{item['source']}</span>
                <span style='color:#334155'> · </span>
                <span style='color:#475569'>{item['date']}</span>
            </div>
        </div>"""

    html = f"""
    <div style='background:#0a0f1a;border:1px solid #1e293b;border-radius:6px;overflow:hidden;font-family:{FONTS}'>
        <div style='padding:6px 10px;display:flex;justify-content:space-between;align-items:center'>
            <span style='color:#64748b;font-size:9px;font-weight:600;letter-spacing:0.1em;text-transform:uppercase'>LATEST</span>
            <span style='color:#334155;font-size:8px'>{len(all_items)} headlines</span>
        </div>
        <div style='max-height:320px;overflow-y:auto'>
            {rows}
        </div>
    </div>"""
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# MAIN RENDER
# =============================================================================

def render_pulse_tab(is_mobile):
    """Render the PULSE landing tab."""

    # Inject pulse-specific CSS
    st.markdown("""
    <style>
        @keyframes shimmer {
            0% { background-position: -200% 0; }
            100% { background-position: 200% 0; }
        }
        @keyframes fade-in {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

    with st.spinner('Scanning markets...'):
        data = _fetch_pulse_batch()
        spark_data = _fetch_sparklines()

    if not data:
        st.markdown(f"<div style='padding:20px;color:#6d6d6d;font-size:11px;font-family:{FONTS}'>Markets data loading — try refreshing in a moment.</div>", unsafe_allow_html=True)
        return

    # Market status bar
    _render_market_status_bar()

    # Hero numbers
    _render_hero_row(data)

    # Sparklines
    if spark_data:
        _render_sparklines(spark_data, data)

    # Layout: heatmap left, movers + news right
    if is_mobile:
        _render_heatmap_grid(data)
        _render_movers(data)
        _render_pulse_news()
    else:
        col_left, col_right = st.columns([55, 45])
        with col_left:
            _render_heatmap_grid(data)
        with col_right:
            _render_movers(data)
            _render_pulse_news()
