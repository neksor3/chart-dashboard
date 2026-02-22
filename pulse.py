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
            bar_bg=bg3, row_alt='#0d1321', hm_txt=txt,
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
    ('6E=F',     {'label': 'EUR/USD',  'fmt': ',.4f'}),
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

SPARKLINE_SYMBOLS = ['ES=F', 'BTC-USD', 'GC=F', 'CL=F', 'ZN=F']


# ── FETCH ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def _fetch_pulse_batch():
    all_syms = list(HERO_SYMBOLS.keys())
    for syms in HEATMAP_SECTORS.values():
        all_syms.extend(syms)
    all_syms = list(OrderedDict.fromkeys(all_syms))
    result = {}
    try:
        batch = yf.download(all_syms, period='5d', group_by='ticker', threads=True, progress=False)
    except Exception as e:
        logger.warning(f"Pulse batch failed: {e}")
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
            result[sym] = {'price': current, 'change': round(change, 2)}
        except Exception as e:
            logger.debug(f"[{sym}] pulse fetch error: {e}")
    return result


@st.cache_data(ttl=900, show_spinner=False)
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

def _market_status():
    now_utc = datetime.now(pytz.utc)
    results = []
    markets = [
        ('SG',  'Asia/Singapore', 9, 0,  17, 0),
        ('US',  'US/Eastern',     9, 30, 16, 0),
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


# ── RENDER ───────────────────────────────────────────────────────────────────

def _render_market_status_bar():
    markets = _market_status()
    t = get_theme()
    s = _s()
    is_light = t.get('mode') == 'light'
    dots = ''
    for m in markets:
        color = t['pos'] if m['open'] else s['off_dot']
        glow = f'box-shadow:0 0 6px {t["pos"]}80;' if m['open'] else ''
        pulse = 'animation:pulse-dot 2s ease-in-out infinite;' if m['open'] else ''
        # Country name: strong text
        nc = ('#0f172a' if is_light else '#f8fafc') if m['open'] else ('#64748b' if is_light else '#f8fafc')
        # Time: always visible
        tc = ('#334155' if is_light else '#f8fafc') if m['open'] else ('#94a3b8' if is_light else '#f8fafc')
        dots += (
            f"<div style='display:flex;align-items:center;gap:4px'>"
            f"<div style='width:6px;height:6px;border-radius:50%;background:{color};{glow}{pulse}'></div>"
            f"<span style='color:{nc};font-size:9px;font-weight:600;letter-spacing:0.06em'>{m['name']}</span>"
            f"<span style='color:{tc};font-size:9px;font-weight:600'>{m['time']}</span>"
            f"</div>"
        )
    html = (
        "<style>@keyframes pulse-dot{0%,100%{opacity:1}50%{opacity:0.4}}</style>"
        f"<div style='display:flex;gap:16px;flex-wrap:wrap;padding:6px 12px;background:{s['bg2']};"
        f"border:1px solid {s['border']};border-radius:4px'>"
        f"<span style='color:#f8fafc;font-size:8px;font-weight:600;letter-spacing:0.1em;"
        f"align-self:center'>MARKETS</span>"
        f"{dots}</div>"
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
    from news import NEWS_FEEDS, fetch_rss_feed
    t = get_theme(); s = _s()
    pos_c = t['pos']

    # 1/N per region, then per source within region
    max_items = 15
    per_region = max(1, max_items // len(NEWS_FEEDS))
    all_items = []
    for region, feeds in NEWS_FEEDS.items():
        region_items = []
        per_src = max(1, per_region // len(feeds))
        for name, url in feeds:
            region_items.extend(fetch_rss_feed(name, url)[:per_src])
        region_items.sort(key=lambda x: x['date'], reverse=True)
        all_items.extend(region_items[:per_region])
    all_items.sort(key=lambda x: x['date'], reverse=True)
    all_items = all_items[:max_items]
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
        data = _fetch_pulse_batch()
        spark_data = _fetch_sparklines()

    if not data:
        st.info('Markets data loading — try refreshing.')
        return

    _render_market_status_bar()
    _render_hero_row(data)
    if spark_data:
        _render_sparkline_row(spark_data, data)

    if is_mobile:
        _render_movers(data)
        _render_pulse_news()
        _render_heatmap_grid(data)
    else:
        # Movers left, News right — both above fold
        col_left, col_right = st.columns([45, 55])
        with col_left:
            _render_movers(data)
        with col_right:
            _render_pulse_news()

        # Heatmap full-width below fold
        _render_heatmap_grid(data)
