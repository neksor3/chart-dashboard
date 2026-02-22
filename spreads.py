import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from config import FUTURES_GROUPS, THEMES, SYMBOL_NAMES, FONTS, clean_symbol

logger = logging.getLogger(__name__)


# =============================================================================
# THEME HELPER
# =============================================================================

def get_theme():
    name = st.session_state.get('theme', 'Dark')
    return THEMES.get(name, THEMES['Dark'])


# =============================================================================
# SPREAD STATISTICS
# =============================================================================

def _spread_sharpe(returns):
    if len(returns) < 5 or returns.std() == 0: return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(252))

def _spread_sortino(returns):
    if len(returns) < 5: return 0.0
    ann_ret = returns.mean() * 252
    down = returns[returns < 0]
    down_std = np.sqrt(np.mean(down**2)) * np.sqrt(252) if len(down) > 0 else 0
    return float(ann_ret / down_std) if down_std else 0.0

def _spread_drawdowns(returns):
    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    mdd = float(dd.min() * 100)
    add = float(dd[dd < 0].mean() * 100) if (dd < 0).any() else 0.0
    return mdd, add

def _spread_r2(returns):
    if len(returns) < 5: return 0.0
    cum = (1 + returns).cumprod().values
    x = np.arange(len(cum))
    xm, ym = x.mean(), cum.mean()
    ss_xy = np.sum(x * cum) - len(cum) * xm * ym
    ss_xx = np.sum(x * x) - len(cum) * xm * xm
    ss_yy = np.sum(cum * cum) - len(cum) * ym * ym
    denom = ss_xx * ss_yy
    if not denom: return 0.0
    slope = ss_xy / ss_xx if ss_xx else 0
    r2 = float(np.clip((ss_xy ** 2) / denom, 0, 1))
    return r2 if slope > 0 else -r2


# =============================================================================
# DATA FETCHING
# =============================================================================

LOOKBACK_OPTIONS = {
    'YTD': 0, '30 Days': 30, '60 Days': 60,
    '120 Days': 120, '240 Days': 240, '520 Days': 520,
}

@st.cache_data(ttl=900, show_spinner=False)
def fetch_sector_spread_data(sector, lookback_days=0):
    symbols = FUTURES_GROUPS.get(sector, [])
    if not symbols: return None
    if lookback_days == 0:
        start = datetime.now().replace(month=1, day=1).strftime('%Y-%m-%d')
    else:
        start = (datetime.now() - pd.Timedelta(days=int(lookback_days * 1.5))).strftime('%Y-%m-%d')

    data = pd.DataFrame()
    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(start=start)
            if not hist.empty:
                closes = hist['Close'].copy()
                closes.index = closes.index.tz_localize(None) if closes.index.tz else closes.index
                closes.index = closes.index.normalize()
                data[sym] = closes.groupby(closes.index).last()
        except Exception as e:
            logger.debug(f"[{sym}] spread data fetch error: {e}")

    if data.empty or len(data.columns) < 2: return None
    data = data.ffill().dropna()
    if lookback_days > 0 and len(data) > lookback_days:
        data = data.iloc[-lookback_days:]
    if len(data) < 5: return None
    return 100 * (data / data.iloc[0])


# =============================================================================
# SPREAD COMPUTATION
# =============================================================================

def compute_sector_spreads(data):
    if data is None or len(data.columns) < 2: return []

    # Pre-compute returns once per symbol
    returns = {sym: data[sym].pct_change().dropna() for sym in data.columns}
    asset_sharpes = {sym: _spread_sharpe(ret) for sym, ret in returns.items()}
    best_long_sym = max(asset_sharpes, key=asset_sharpes.get)
    best_long_sharpe = asset_sharpes[best_long_sym]

    pairs = []
    for s1, s2 in combinations(data.columns.tolist(), 2):
        spread_ret = (returns[s1] - returns[s2]).dropna()
        sh = _spread_sharpe(spread_ret)
        so = _spread_sortino(spread_ret)

        # Flip so Sharpe is always positive (long the better side)
        if sh < 0:
            spread_ret = -spread_ret
            sh, so = -sh, -so
            s1, s2 = s2, s1

        mdd, add = _spread_drawdowns(spread_ret)
        cum_spread = (1 + spread_ret).cumprod()

        pairs.append({
            'long': s1, 'short': s2,
            'Sharpe': sh, 'Sortino': so,
            'MAR': float(spread_ret.mean() * 252 * 100 / abs(add)) if add != 0 else 0.0,
            'R²': _spread_r2(spread_ret),
            'Tot%': float((cum_spread.iloc[-1] - 1) * 100),
            'Ann%': float(spread_ret.mean() * 252 * 100),
            'Vol%': float(spread_ret.std() * np.sqrt(252) * 100),
            'MDD%': mdd, 'ADD%': add,
            'Corr': float(returns[s1].corr(returns[s2])),
            'Win%': float((spread_ret > 0).sum() / len(spread_ret) * 100) if len(spread_ret) > 0 else 50.0,
            'beats_long': sh > best_long_sharpe,
            # Cum series: data is already rebased to 100
            'cum_long': data[s1], 'cum_short': data[s2],
            'cum_spread': 100 * cum_spread,
        })

    if not pairs: return []

    # Composite ranking across 4 metrics
    n = len(pairs)
    for metric in ['Sharpe', 'Sortino', 'MAR', 'R²']:
        order = sorted(range(n), key=lambda i, m=metric: -pairs[i][m])
        for rank, idx in enumerate(order):
            pairs[idx][f'_{metric}_rank'] = rank + 1
    for p in pairs:
        p['_score'] = np.mean([p[f'_{m}_rank'] for m in ['Sharpe', 'Sortino', 'MAR', 'R²']])
        p['best_long_sym'] = best_long_sym
        p['best_long_sharpe'] = best_long_sharpe

    pairs.sort(key=lambda x: -x['Sharpe'])
    return pairs


# =============================================================================
# SORTING
# =============================================================================

SORT_KEYS = {
    'Composite': ('_score', False), 'Sharpe': ('Sharpe', True),
    'Sortino': ('Sortino', True), 'MAR': ('MAR', True),
    'R²': ('R²', True), 'Total': ('Tot%', True), 'Win Rate': ('Win%', True),
}

def sort_spread_pairs(pairs, sort_key='Composite', ascending=False):
    key, default_desc = SORT_KEYS.get(sort_key, ('Sharpe', True))
    reverse = default_desc if not ascending else not default_desc
    return sorted(pairs, key=lambda x: x.get(key, 0), reverse=reverse)


# =============================================================================
# SHARED DISPLAY HELPERS
# =============================================================================

def _sym_name(symbol):
    return SYMBOL_NAMES.get(symbol, clean_symbol(symbol))


# =============================================================================
# SPREAD TABLE
# =============================================================================

def render_spread_table(pairs, top_n=10):
    show = pairs[:top_n]
    t = get_theme()
    pos_c, neg_c, short_c = t['pos'], t['neg'], t['short']
    _bg3 = t.get('bg3', '#0f172a'); _bdr = t.get('border', '#1e293b')
    _txt2 = t.get('text2', '#94a3b8'); _mut = t.get('muted', '#475569')
    th = f"padding:4px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;"
    td = f"padding:5px 8px;border-bottom:1px solid {_bdr}22;"

    html = f"""<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:6px'><table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>
        <thead style='background:{_bg3}'><tr>
            <th style='{th}text-align:left'>RANK</th>
            <th style='{th}text-align:left'>LONG</th>
            <th style='{th}text-align:left'>SHORT</th>
            <th style='{th}text-align:right'>SHARPE</th>
            <th style='{th}text-align:right'>SORTINO</th>
            <th style='{th}text-align:right'>MAR</th>
            <th style='{th}text-align:right'>R²</th>
            <th style='{th}text-align:right'>WIN%</th>
            <th style='{th}text-align:right'>TOT%</th>
            <th style='{th}text-align:right'>VOL%</th>
            <th style='{th}text-align:right'>MDD%</th>
            <th style='{th}text-align:right'>CORR</th>
            <th style='{th}text-align:center'>vs LONG</th>
        </tr></thead><tbody>"""

    for rank, p in enumerate(show, 1):
        sh_c = pos_c if p['Sharpe'] >= 0 else neg_c
        tot_c = pos_c if p['Tot%'] >= 0 else neg_c
        win_c = pos_c if p['Win%'] >= 55 else (neg_c if p['Win%'] < 45 else _txt2)
        vs = f"<span style='color:{pos_c};font-weight:700'>▲</span>" if p['beats_long'] else f"<span style='color:{_mut}'>—</span>"
        bg = f'linear-gradient(90deg,{pos_c}08,{_bg3},{pos_c}08)' if p['beats_long'] else 'transparent'
        html += f"""<tr style='background:{bg}'>
            <td style='{td}color:{_mut};text-align:left'>{rank}</td>
            <td style='{td}color:{pos_c};font-weight:600;text-align:left'>{_sym_name(p["long"])}</td>
            <td style='{td}color:{short_c};font-weight:600;text-align:left'>{_sym_name(p["short"])}</td>
            <td style='{td}text-align:right'><span style='color:{sh_c};font-weight:700'>{p["Sharpe"]:.2f}</span></td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["Sortino"]:.2f}</td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["MAR"]:.2f}</td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["R²"]:.3f}</td>
            <td style='{td}text-align:right'><span style='color:{win_c};font-weight:600'>{p["Win%"]:.0f}%</span></td>
            <td style='{td}text-align:right'><span style='color:{tot_c};font-weight:600'>{"+" if p["Tot%"] >= 0 else ""}{p["Tot%"]:.1f}%</span></td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["Vol%"]:.1f}%</td>
            <td style='{td}text-align:right;color:{neg_c}'>{p["MDD%"]:.1f}%</td>
            <td style='{td}text-align:right;color:{_txt2}'>{p["Corr"]:.2f}</td>
            <td style='{td}text-align:center'>{vs}</td>
        </tr>"""
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# SPREAD CHARTS
# =============================================================================

def render_spread_charts(pairs, data, mobile=False):
    top_n = min(6, len(pairs))
    if top_n == 0: return

    t = get_theme()
    _pbg = t.get('plot_bg', '#121212'); _grd = t.get('grid', '#1f1f1f')
    _tk = t.get('tick', '#888888'); _mut = t.get('muted', '#475569')

    n_cols = 1 if mobile else min(3, top_n)
    n_rows = (top_n + n_cols - 1) // n_cols

    subtitles = []
    lc, sc = t['long'], t['short']
    for i in range(top_n):
        p = pairs[i]
        subtitles.append(
            f"<span style='color:{lc}'>■</span> {_sym_name(p['long'])}  "
            f"<span style='color:{sc}'>■</span> {_sym_name(p['short'])}  "
            f"<span style='color:#ffffff'>■</span> Spread")
    while len(subtitles) < n_rows * n_cols:
        subtitles.append("")

    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subtitles,
        horizontal_spacing=0.06, vertical_spacing=0.18 if not mobile else 0.08)

    # Date tick positions (shared across all subplots)
    n_ticks = 4; idx_step = max(1, len(data) // n_ticks)
    tick_vals = list(range(0, len(data), idx_step))
    if (len(data) - 1) not in tick_vals:
        tick_vals.append(len(data) - 1)
    tick_text = [data.index[j].strftime('%d %b') for j in tick_vals if j < len(data)]
    tick_vals = tick_vals[:len(tick_text)]

    for i in range(top_n):
        p = pairs[i]
        row, col = i // n_cols + 1, i % n_cols + 1
        axis_idx = (row - 1) * n_cols + col

        for series, color, width, dash, label in [
            (p['cum_long'], t['long'], 1.3, None, 'Long'),
            (p['cum_short'], t['short'], 1.3, None, 'Short'),
            (p['cum_spread'], '#ffffff', 1.5, 'dot', 'Spread'),
        ]:
            fig.add_trace(go.Scatter(
                x=list(range(len(series))), y=series.values, mode='lines',
                line=dict(color=color, width=width, dash=dash, shape='spline', smoothing=1.0),
                showlegend=False, hovertemplate=f'{label}: %{{y:.1f}}<extra></extra>'), row=row, col=col)

        fig.add_hline(y=100, line=dict(color=_grd, width=0.8, dash='dot'), row=row, col=col)

        ax_key = 'xaxis' if axis_idx == 1 else f'xaxis{axis_idx}'
        fig.update_layout(**{ax_key: dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text)})

        fig.add_annotation(
            text=f"<b>{i+1}</b>", x=0.02, y=0.95,
            xref=f"x{'' if axis_idx == 1 else axis_idx} domain",
            yref=f"y{'' if axis_idx == 1 else axis_idx} domain",
            showarrow=False, font=dict(size=12, color=_mut, family=FONTS),
            xanchor='left', yanchor='top')

    for ann in fig['layout']['annotations']:
        if 'domain' not in str(ann.get('xref', '')):
            ann['font'] = dict(size=10, family=FONTS)

    fig.update_layout(
        template='plotly_dark', height=(350 if mobile else 220) * n_rows,
        margin=dict(l=40, r=40, t=45, b=30),
        plot_bgcolor=_pbg, paper_bgcolor=_pbg,
        showlegend=False, hovermode='x unified', font=dict(family=FONTS))
    fig.update_xaxes(gridcolor=_grd, linecolor='rgba(0,0,0,0)',
        tickfont=dict(color=_tk, size=8, family=FONTS), showgrid=False, tickangle=0)
    fig.update_yaxes(gridcolor=_grd, linecolor='rgba(0,0,0,0)',
        tickfont=dict(color=_tk, size=8, family=FONTS), side='right')

    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True, 'displayModeBar': False, 'responsive': True})


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def render_spreads_tab(is_mobile):
    t = get_theme(); pos_c = t['pos']
    _lbl = f"color:#e2e8f0;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"

    # Controls
    if is_mobile:
        col_sec, col_lb, col_sort = st.columns([2, 1, 1])
    else:
        col_sec, col_lb, col_sort, col_dir = st.columns([3, 2, 2, 1])

    with col_sec:
        st.markdown(f"<div style='{_lbl}'>SECTOR</div>", unsafe_allow_html=True)
        sector_names = list(FUTURES_GROUPS.keys())
        spread_sector = st.selectbox("Sector", sector_names,
            index=sector_names.index(st.session_state.get('spread_sector', 'Indices')),
            key='spread_sector_sel', label_visibility='collapsed')
        st.session_state.spread_sector = spread_sector
    with col_lb:
        st.markdown(f"<div style='{_lbl}'>LOOKBACK</div>", unsafe_allow_html=True)
        lookback_label = st.selectbox("Lookback", list(LOOKBACK_OPTIONS.keys()), index=0,
            key='spread_lookback_sel', label_visibility='collapsed')
    with col_sort:
        st.markdown(f"<div style='{_lbl}'>SORT BY</div>", unsafe_allow_html=True)
        sort_key = st.selectbox("Sort", list(SORT_KEYS.keys()), index=0,
            key='spread_sort_sel', label_visibility='collapsed')

    ascending = False
    if not is_mobile:
        with col_dir:
            st.markdown(f"<div style='{_lbl}'>ORDER</div>", unsafe_allow_html=True)
            ascending = st.selectbox("Dir", ['Desc', 'Asc'], index=0,
                key='spread_dir_sel', label_visibility='collapsed') == 'Asc'

    # Fetch and compute
    lookback_days = LOOKBACK_OPTIONS[lookback_label]
    with st.spinner(f'Computing {spread_sector} spreads ({lookback_label})...'):
        data = fetch_sector_spread_data(spread_sector, lookback_days)

    _mut = t.get('muted', '#475569')
    if data is None or len(data.columns) < 2:
        st.markdown(f"<div style='padding:12px;color:{_mut};font-size:11px;font-family:{FONTS}'>Need at least 2 assets with data for spread analysis</div>", unsafe_allow_html=True)
        return

    pairs = compute_sector_spreads(data)
    if not pairs:
        st.markdown(f"<div style='padding:12px;color:{_mut};font-size:11px;font-family:{FONTS}'>No spreads computed</div>", unsafe_allow_html=True)
        return

    sorted_pairs = sort_spread_pairs(pairs, sort_key, ascending)

    # Info bar
    best_sym = pairs[0].get('best_long_sym', '')
    best_sharpe = pairs[0].get('best_long_sharpe', 0)
    n_beats = sum(1 for p in pairs if p['beats_long'])
    _bg3 = t.get('bg3', '#0f172a'); _txt2 = t.get('text2', '#94a3b8')
    st.markdown(f"""
        <div style='padding:5px 10px;background-color:{_bg3};font-family:{FONTS};display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;border-radius:4px'>
            <span style='color:{_mut};font-size:10px'>{len(pairs)} pairs · {data.index[0].strftime('%d %b')} → {data.index[-1].strftime('%d %b %Y')}</span>
            <span style='color:{_txt2};font-size:10px'>
                Best long: <span style='color:{pos_c};font-weight:600'>{_sym_name(best_sym)}</span>
                <span style='color:{_mut}'>Sharpe {best_sharpe:.2f}</span>
                &nbsp;·&nbsp;
                <span style='color:{pos_c if n_beats > 0 else _mut}'>{n_beats} spread{"s" if n_beats != 1 else ""} beat{"s" if n_beats == 1 else ""} it</span>
            </span>
        </div>""", unsafe_allow_html=True)

    render_spread_charts(sorted_pairs, data, mobile=is_mobile)
    render_spread_table(sorted_pairs, top_n=10)
