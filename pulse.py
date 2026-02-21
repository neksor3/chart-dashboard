import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from collections import OrderedDict
import pytz
import logging

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
    """Fetch all pulse symbols in one batch."""
    all_syms = list(HERO_SYMBOLS.keys())
    for syms in HEATMAP_SECTORS.values():
        all_syms.extend(syms)
    all_syms = list(OrderedDict.fromkeys(all_syms))

    result = {}
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
            change = ((current - prev_close) / prev_close) * 100 if prev_close else 0

            result[sym] = {
                'price': current,
                'change': round(change, 2),
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
# SVG SPARKLINE
# =============================================================================

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
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>"
        f"<defs><linearGradient id='{uid}' x1='0' y1='0' x2='0' y2='1'>"
        f"<stop offset='0%' stop-color='{color}' stop-opacity='0.3'/>"
        f"<stop offset='100%' stop-color='{color}' stop-opacity='0'/>"
        f"</linearGradient></defs>"
        f"<polygon points='{fill_points}' fill='url(#{uid})'/>"
        f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='1.5' stroke-linejoin='round' stroke-linecap='round'/>"
        f"<circle cx='{xs[-1]}' cy='{ys[-1]}' r='2' fill='{color}'/>"
        f"</svg>"
    )


# =============================================================================
# MARKET STATUS
# =============================================================================

def _market_status():
    now_utc = datetime.now(pytz.utc)
    results = []
    markets = [
        ('US',     'US/Eastern',     9, 30, 16, 0),
        ('LONDON', 'Europe/London',  8, 0,  16, 30),
        ('EUROPE', 'Europe/Berlin',  9, 0,  17, 30),
        ('TOKYO',  'Asia/Tokyo',     9, 0,  15, 0),
        ('HK',     'Asia/Hong_Kong', 9, 30, 16, 0),
        ('SGP',    'Asia/Singapore', 9, 0,  17, 0),
    ]
    for name, tz_str, oh, om, ch, cm in markets:
        tz = pytz.timezone(tz_str)
        local = now_utc.astimezone(tz)
        is_weekend = local.weekday() >= 5
        open_time = local.replace(hour=oh, minute=om, second=0)
        close_time = local.replace(hour=ch, minute=cm, second=0)
        is_open = not is_weekend and open_time <= local <= close_time
        results.append({'name': name, 'time': local.strftime('%H:%M'), 'open': is_open})
    return results


# =============================================================================
# RENDER — table-based layouts for Streamlit HTML compatibility
# =============================================================================

def _render_market_status_bar():
    markets = _market_status()
    t = get_theme()
    cells = ""
    for m in markets:
        dot_c = t['pos'] if m['open'] else '#3a3a3a'
        nm_c = '#e2e8f0' if m['open'] else '#4a5568'
        cells += (
            "<td style='padding:4px 8px;white-space:nowrap'>"
            "<span style='display:inline-block;width:6px;height:6px;border-radius:50%;"
            f"background:{dot_c};vertical-align:middle'></span> "
            f"<span style='color:{nm_c};font-size:9px;font-weight:600;letter-spacing:0.08em'>{m['name']}</span> "
            f"<span style='color:#4a5568;font-size:8px'>{m['time']}</span>"
            "</td>"
        )
    st.markdown(
        "<table style='width:100%;background:#0a0f1a;border:1px solid #1e293b;"
        f"border-radius:4px;border-collapse:collapse;margin-bottom:10px;font-family:{FONTS}'>"
        "<tr>"
        "<td style='padding:4px 8px;color:#334155;font-size:8px;font-weight:600;"
        "letter-spacing:0.1em;white-space:nowrap'>MARKETS</td>"
        f"{cells}"
        "</tr></table>",
        unsafe_allow_html=True,
    )


def _render_hero_row(data):
    t = get_theme()
    pos_c, neg_c = t['pos'], t['neg']
    cells = ""
    for sym, cfg in HERO_SYMBOLS.items():
        d = data.get(sym)
        if not d:
            continue
        price = d['price']
        change = d['change']
        color = pos_c if change >= 0 else neg_c
        sign = '+' if change >= 0 else ''
        arrow = '&#9650;' if change > 0 else '&#9660;' if change < 0 else '&#8211;'
        price_str = f'{price:{cfg["fmt"]}}'
        cells += (
            "<td style='padding:10px 12px;background:linear-gradient(135deg,#0f172a,#0a0f1a);"
            "border:1px solid #1e293b;border-radius:6px;vertical-align:top'>"
            f"<div style='color:#4a5568;font-size:8px;font-weight:600;letter-spacing:0.12em;"
            f"text-transform:uppercase;margin-bottom:4px'>{cfg['label']}</div>"
            f"<div style='color:#f1f5f9;font-size:17px;font-weight:700;"
            f"font-variant-numeric:tabular-nums;letter-spacing:-0.02em;"
            f"text-shadow:0 0 20px {color}40'>{price_str}</div>"
            f"<div style='margin-top:3px'>"
            f"<span style='color:{color};font-size:10px'>{arrow}</span> "
            f"<span style='color:{color};font-size:11px;font-weight:700'>{sign}{change:.2f}%</span>"
            "</div></td>"
        )
    st.markdown(
        "<table style='width:100%;border-collapse:separate;border-spacing:6px 0;"
        f"margin-bottom:10px;font-family:{FONTS}'>"
        f"<tr>{cells}</tr></table>",
        unsafe_allow_html=True,
    )


def _render_sparkline_row(spark_data, pulse_data):
    t = get_theme()
    pos_c, neg_c = t['pos'], t['neg']
    cells = ""
    for sym in SPARKLINE_SYMBOLS:
        sdata = spark_data.get(sym)
        pdata = pulse_data.get(sym)
        if not sdata or not pdata:
            continue
        short = clean_symbol(sym)
        change = pdata['change']
        color = pos_c if change >= 0 else neg_c
        sign = '+' if change >= 0 else ''
        svg = _svg_sparkline(sdata, width=100, height=28, pos_color=pos_c, neg_color=neg_c)
        cells += (
            "<td style='padding:8px 10px;background:#0f172a;border:1px solid #1e293b;"
            "border-radius:4px;vertical-align:top'>"
            f"<span style='color:#94a3b8;font-size:9px;font-weight:600;"
            f"letter-spacing:0.06em'>{short}</span> "
            f"<span style='color:{color};font-size:9px;font-weight:700;"
            f"float:right'>{sign}{change:.2f}%</span>"
            f"<div style='margin-top:4px'>{svg}</div>"
            "<div style='color:#475569;font-size:7px;margin-top:2px;text-align:right'>30 DAY</div>"
            "</td>"
        )
    st.markdown(
        "<table style='width:100%;border-collapse:separate;border-spacing:6px 0;"
        f"margin-bottom:10px;font-family:{FONTS}'>"
        f"<tr>{cells}</tr></table>",
        unsafe_allow_html=True,
    )


def _render_heatmap_grid(data):
    t = get_theme()
    pos_c, neg_c = t['pos'], t['neg']

    def _bg(change, pos, neg):
        abs_c = min(abs(change), 5)
        opacity = 0.25 + (abs_c / 5) * 0.75
        base = pos if change >= 0 else neg
        r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
        return f'rgba({r},{g},{b},{opacity:.2f})'

    sectors_html = ""
    for sector, syms in HEATMAP_SECTORS.items():
        cells = ""
        for sym in syms:
            d = data.get(sym)
            if not d:
                continue
            change = d['change']
            name = clean_symbol(sym)
            bg = _bg(change, pos_c, neg_c)
            tc = pos_c if change >= 0 else neg_c
            sign = '+' if change >= 0 else ''
            cells += (
                f"<td style='padding:6px 8px;background:{bg};border-radius:3px;"
                "text-align:center;border:1px solid rgba(255,255,255,0.04)'>"
                f"<div style='color:#e2e8f0;font-size:10px;font-weight:600'>{name}</div>"
                f"<div style='color:{tc};font-size:11px;font-weight:700;margin-top:1px;"
                f"font-variant-numeric:tabular-nums'>{sign}{change:.2f}%</div>"
                "</td>"
            )
        sectors_html += (
            f"<div style='margin-bottom:6px'>"
            f"<div style='color:#334155;font-size:8px;font-weight:600;letter-spacing:0.1em;"
            f"text-transform:uppercase;margin-bottom:4px;padding-left:2px'>{sector}</div>"
            f"<table style='border-collapse:separate;border-spacing:4px 0;width:100%'>"
            f"<tr>{cells}</tr></table></div>"
        )

    st.markdown(
        "<div style='padding:10px 12px;background:#0a0f1a;border:1px solid #1e293b;"
        f"border-radius:6px;margin-bottom:10px;font-family:{FONTS}'>"
        "<div style='margin-bottom:8px'>"
        "<span style='color:#64748b;font-size:9px;font-weight:600;"
        "letter-spacing:0.1em;text-transform:uppercase'>MARKET HEATMAP</span>"
        " <span style='color:#334155;font-size:8px;float:right'>Day Change %</span>"
        "</div>"
        f"{sectors_html}"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_movers(data):
    t = get_theme()
    pos_c, neg_c = t['pos'], t['neg']

    all_items = [(sym, d['change']) for sym, d in data.items()]
    all_items.sort(key=lambda x: x[1], reverse=True)
    gainers = [(s, c) for s, c in all_items if c > 0][:5]
    losers = [(s, c) for s, c in all_items if c < 0][-5:]
    losers.reverse()

    def _rows(items, color, is_gain):
        if not items:
            return "<tr><td style='color:#334155;font-size:10px;padding:8px'>—</td></tr>"
        max_abs = max(abs(c) for _, c in items) or 1
        html = ""
        for sym, change in items:
            short = clean_symbol(sym)
            bar_pct = max(abs(change) / max_abs * 85, 8)
            sign = '+' if change >= 0 else ''
            if is_gain:
                bar = (f"<div style='background:linear-gradient(90deg,{color}20,{color}60);"
                       f"width:{bar_pct}%;height:16px;border-radius:2px'></div>")
            else:
                bar = (f"<div style='background:linear-gradient(270deg,{color}20,{color}60);"
                       f"width:{bar_pct}%;height:16px;border-radius:2px;margin-left:auto'></div>")
            html += (
                "<tr>"
                f"<td style='padding:3px 0;width:50px;color:#e2e8f0;font-size:10px;font-weight:600'>{short}</td>"
                f"<td style='padding:3px 0'>{bar}</td>"
                f"<td style='padding:3px 4px;width:55px;text-align:right;color:{color};"
                f"font-size:10px;font-weight:700;font-variant-numeric:tabular-nums;"
                f"white-space:nowrap'>{sign}{change:.2f}%</td>"
                "</tr>"
            )
        return html

    gain_html = _rows(gainers, pos_c, True)
    lose_html = _rows(losers, neg_c, False)

    st.markdown(
        "<div style='padding:10px 12px;background:#0a0f1a;border:1px solid #1e293b;"
        f"border-radius:6px;margin-bottom:10px;font-family:{FONTS}'>"
        "<table style='width:100%;border-collapse:collapse'><tr>"
        # Gainers
        "<td style='vertical-align:top;width:50%;padding-right:10px'>"
        f"<div style='color:{pos_c};font-size:9px;font-weight:600;"
        "letter-spacing:0.1em;margin-bottom:6px'>&#9650; TOP GAINERS</div>"
        f"<table style='width:100%;border-collapse:collapse'>{gain_html}</table>"
        "</td>"
        # Losers
        "<td style='vertical-align:top;width:50%;padding-left:10px;border-left:1px solid #1e293b'>"
        f"<div style='color:{neg_c};font-size:9px;font-weight:600;"
        "letter-spacing:0.1em;margin-bottom:6px'>&#9660; TOP LOSERS</div>"
        f"<table style='width:100%;border-collapse:collapse'>{lose_html}</table>"
        "</td>"
        "</tr></table></div>",
        unsafe_allow_html=True,
    )


def _render_pulse_news():
    from news import NEWS_FEEDS, fetch_rss_feed
    t = get_theme()
    pos_c = t['pos']

    all_items = []
    for region in ['Global', 'Singapore']:
        feeds = NEWS_FEEDS.get(region, [])
        for name, url in feeds:
            all_items.extend(fetch_rss_feed(name, url))
    all_items.sort(key=lambda x: x['date'], reverse=True)
    all_items = all_items[:12]
    if not all_items:
        return

    rows = ""
    for i, item in enumerate(all_items):
        bg = '#0a0f1a' if i % 2 == 0 else '#0d1321'
        link = (f"<a href='{item['url']}' target='_blank' style='color:#c9d1d9;"
                f"text-decoration:none;font-size:10px;line-height:1.35'>{item['title']}</a>"
                if item['url'] else
                f"<span style='color:#c9d1d9;font-size:10px'>{item['title']}</span>")
        rows += (
            f"<tr style='background:{bg}'>"
            "<td style='padding:5px 10px;border-bottom:1px solid #1e293b10'>"
            f"{link}"
            f"<div style='font-size:7px;margin-top:1px'>"
            f"<span style='color:{pos_c};font-weight:600'>{item['source']}</span> "
            f"<span style='color:#334155'>&#183;</span> "
            f"<span style='color:#475569'>{item['date']}</span>"
            "</div></td></tr>"
        )

    st.markdown(
        "<div style='background:#0a0f1a;border:1px solid #1e293b;"
        f"border-radius:6px;overflow:hidden;font-family:{FONTS}'>"
        "<div style='padding:6px 10px'>"
        "<span style='color:#64748b;font-size:9px;font-weight:600;"
        "letter-spacing:0.1em'>LATEST</span> "
        f"<span style='color:#334155;font-size:8px;float:right'>{len(all_items)} headlines</span>"
        "</div>"
        "<div style='max-height:320px;overflow-y:auto'>"
        f"<table style='width:100%;border-collapse:collapse'>{rows}</table>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN RENDER
# =============================================================================

def render_pulse_tab(is_mobile):
    with st.spinner('Scanning markets...'):
        data = _fetch_pulse_batch()
        spark_data = _fetch_sparklines()

    if not data:
        st.markdown(
            f"<div style='padding:20px;color:#6d6d6d;font-size:11px;"
            f"font-family:{FONTS}'>Markets data loading — try refreshing.</div>",
            unsafe_allow_html=True,
        )
        return

    _render_market_status_bar()
    _render_hero_row(data)

    if spark_data:
        _render_sparkline_row(spark_data, data)

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
