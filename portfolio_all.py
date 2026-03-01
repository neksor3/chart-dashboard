import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import OrderedDict
import logging

from config import FUTURES_GROUPS, THEMES, SYMBOL_NAMES, FONTS, clean_symbol
from portfolio import (C_POS, C_NEG, C_MUTE, C_BG, C_HDR, C_TXT, C_TXT2, C_BORDER,
                       C_GOLD, C_EW, TH, TD, _short,
                       PERIOD_OPTIONS, fetch_symbol_history, _calc_oos_metrics, _section)

logger = logging.getLogger(__name__)

# =============================================================================
# EW SCAN ENGINE
# =============================================================================

def _compute_ew_returns(symbols, fetch_days=1825):
    """Compute equal-weight buy-and-hold returns for a group."""
    data, valid = fetch_symbol_history(tuple(symbols), days=fetch_days)
    if data is None or len(valid) < 2:
        return None, None
    returns = data.pct_change().dropna()
    n_assets = len(valid)
    eq_w = np.ones(n_assets) / n_assets
    ew_daily = returns.values @ eq_w
    ew_series = pd.Series(ew_daily, index=returns.index)
    return ew_series, valid


def _run_ew_scan(period_days, progress_bar=None):
    """Scan all groups with equal-weight returns."""
    all_results = []
    groups = list(FUTURES_GROUPS.items())

    for i, (gname, syms) in enumerate(groups):
        if progress_bar:
            progress_bar.progress((i + 1) / len(groups), text=f'Scanning: {gname}')
        if len(syms) < 2:
            continue
        try:
            ew_ret, valid = _compute_ew_returns(syms, fetch_days=period_days)
            if ew_ret is None or len(ew_ret) < 20:
                continue
            metrics = _calc_oos_metrics(ew_ret)
            if metrics is None:
                continue
            metrics['group'] = gname
            metrics['n_assets'] = len(valid)
            metrics['ew_returns'] = ew_ret
            all_results.append(metrics)
        except Exception as e:
            logger.warning(f"EW scan error for {gname}: {e}")

    if progress_bar:
        progress_bar.empty()

    return all_results


# =============================================================================
# SORT CONFIG
# =============================================================================

SCAN_SORT_KEYS = {
    'Win Rate': ('win_rate', True),
    'Sharpe': ('sharpe', True),
    'Sortino': ('sortino', True),
    'MAR': ('mar', True),
    'R²': ('r2', True),
    'Total Return': ('total_ret', True),
    'Annual Return': ('ann_ret', True),
}

# =============================================================================
# MAIN RENDER
# =============================================================================

def render_all_tab(is_mobile):
    import portfolio
    theme_name = st.session_state.get('theme', 'Dark')
    theme = THEMES.get(theme_name, THEMES['Dark'])
    portfolio.C_POS = theme['pos']; portfolio.C_NEG = theme['neg']
    pos_c = theme['pos']; neg_c = theme['neg']
    _lbl = f"color:#f8fafc;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"
    _bg3 = theme.get('bg3', '#0f172a'); _mut = theme.get('muted', '#475569')

    # Controls
    if is_mobile:
        col_period, col_sort = st.columns([1, 1])
    else:
        col_period, col_sort, col_btn = st.columns([3, 3, 2])

    with col_period:
        st.markdown(f"<div style='{_lbl}'>PERIOD</div>", unsafe_allow_html=True)
        period_label = st.selectbox("Period", list(PERIOD_OPTIONS.keys()),
                                     index=3, key='portall_period', label_visibility='collapsed')
        period_days = PERIOD_OPTIONS[period_label]
    with col_sort:
        st.markdown(f"<div style='{_lbl}'>RANK BY</div>", unsafe_allow_html=True)
        sort_options = list(SCAN_SORT_KEYS.keys())
        rank_by = st.selectbox("Rank", sort_options, index=0,
                                key='portall_sort', label_visibility='collapsed')

    if is_mobile:
        scan_clicked = st.button('▶  Scan All', key='portall_scan', type='primary')
    else:
        with col_btn:
            st.markdown(f"<div style='{_lbl}'>&nbsp;</div>", unsafe_allow_html=True)
            scan_clicked = st.button('▶  Scan All', key='portall_scan', type='primary')

    if scan_clicked:
        progress = st.progress(0, text='Scanning groups...')
        results = _run_ew_scan(period_days, progress_bar=progress)
        if not results:
            st.warning('No valid groups found')
            return
        st.session_state.portall_results = results
        _render_results(results, theme, rank_by, is_mobile)
    elif 'portall_results' in st.session_state:
        _render_results(st.session_state.portall_results, theme, rank_by, is_mobile)


# =============================================================================
# RENDER RESULTS
# =============================================================================

def _render_results(results, theme, rank_by, is_mobile):
    pos_c = theme['pos']; neg_c = theme['neg']
    _bg3 = theme.get('bg3', '#0f172a'); _bdr = theme.get('border', '#1e293b')
    _txt = theme.get('text', '#e2e8f0'); _txt2 = theme.get('text2', '#94a3b8')
    _mut = theme.get('muted', '#475569')

    # Sort
    sort_key, reverse = SCAN_SORT_KEYS.get(rank_by, ('win_rate', True))
    sorted_results = sorted(results, key=lambda x: x.get(sort_key, 0), reverse=reverse)

    # Table
    th = f"padding:4px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;"
    td = f"padding:5px 8px;border-bottom:1px solid {_bdr}22;"

    html = f"""<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:6px;margin-top:8px'>
    <table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>
        <thead style='background:{_bg3}'><tr>
            <th style='{th}text-align:left'>#</th>
            <th style='{th}text-align:left'>GROUP</th>
            <th style='{th}text-align:right'>ASSETS</th>
            <th style='{th}text-align:right'>WIN%</th>
            <th style='{th}text-align:right'>SHARPE</th>
            <th style='{th}text-align:right'>SORTINO</th>
            <th style='{th}text-align:right'>MAR</th>
            <th style='{th}text-align:right'>R²</th>
            <th style='{th}text-align:right'>TOT%</th>
            <th style='{th}text-align:right'>ANN%</th>
            <th style='{th}text-align:right'>VOL%</th>
            <th style='{th}text-align:right'>MDD%</th>
            <th style='{th}text-align:right'>YTD%</th>
            <th style='{th}text-align:right'>MTD%</th>
        </tr></thead><tbody>"""

    for rank, r in enumerate(sorted_results, 1):
        sh_c = pos_c if r['sharpe'] >= 0 else neg_c
        tot_c = pos_c if r['total_ret'] >= 0 else neg_c
        tot_s = '+' if r['total_ret'] >= 0 else ''
        win_c = pos_c if r['win_rate'] >= 0.55 else (neg_c if r['win_rate'] < 0.45 else _txt2)
        ytd_c = pos_c if r['ytd'] >= 0 else neg_c
        mtd_c = pos_c if r['mtd'] >= 0 else neg_c
        is_top3 = rank <= 3
        bg = f'rgba(74,222,128,0.06)' if is_top3 else 'transparent'
        fw = '700' if is_top3 else '500'
        gc = pos_c if is_top3 else _txt

        html += f"""<tr style='background:{bg}'>
            <td style='{td}color:{_mut}'>{rank}</td>
            <td style='{td}color:{gc};font-weight:{fw}'>{r['group']}</td>
            <td style='{td}text-align:right;color:{_txt2}'>{r['n_assets']}</td>
            <td style='{td}text-align:right'><span style='color:{win_c};font-weight:600'>{r["win_rate"]*100:.1f}%</span></td>
            <td style='{td}text-align:right'><span style='color:{sh_c};font-weight:700'>{r["sharpe"]:.2f}</span></td>
            <td style='{td}text-align:right;color:{_txt2}'>{r["sortino"]:.2f}</td>
            <td style='{td}text-align:right;color:{_txt2}'>{r["mar"]:.2f}</td>
            <td style='{td}text-align:right;color:{_txt2}'>{r["r2"]:.3f}</td>
            <td style='{td}text-align:right'><span style='color:{tot_c};font-weight:600'>{tot_s}{r["total_ret"]*100:.1f}%</span></td>
            <td style='{td}text-align:right;color:{_txt2}'>{r["ann_ret"]*100:.1f}%</td>
            <td style='{td}text-align:right;color:{_txt2}'>{r["ann_vol"]*100:.1f}%</td>
            <td style='{td}text-align:right;color:{neg_c}'>{r["max_dd"]*100:.1f}%</td>
            <td style='{td}text-align:right'><span style='color:{ytd_c};font-weight:600'>{r["ytd"]*100:.1f}%</span></td>
            <td style='{td}text-align:right'><span style='color:{mtd_c}'>{r["mtd"]*100:.1f}%</span></td>
        </tr>"""

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

    # Equity curve charts — batches of 6
    _render_ew_charts(sorted_results, theme, is_mobile)


# =============================================================================
# EW EQUITY CHARTS
# =============================================================================

def _render_ew_charts(sorted_results, theme, is_mobile):
    chart_data = [(r['group'], r['ew_returns']) for r in sorted_results if 'ew_returns' in r]
    if not chart_data:
        return

    _pbg = theme.get('plot_bg', '#121212'); _grd = theme.get('grid', '#1f1f1f')
    _axl = theme.get('axis_line', '#2a2a2a'); _tk = theme.get('tick', '#888888')
    _mut = theme.get('muted', '#475569'); pos_c = theme['pos']

    batch_size = 6
    for batch_start in range(0, len(chart_data), batch_size):
        batch = chart_data[batch_start:batch_start + batch_size]
        n_charts = len(batch)
        n_cols = 1 if is_mobile else min(3, n_charts)
        n_rows = (n_charts + n_cols - 1) // n_cols

        subtitles = [f"<b>{g}</b> · Equal Weight" for g, _ in batch]
        while len(subtitles) < n_rows * n_cols:
            subtitles.append("")

        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=subtitles,
            horizontal_spacing=0.06, vertical_spacing=0.18 if not is_mobile else 0.08)

        for i, (gname, ew_ret) in enumerate(batch):
            row = i // n_cols + 1; col = i % n_cols + 1
            cum = (1 + ew_ret).cumprod() * 100

            fig.add_trace(go.Scatter(
                x=cum.index, y=cum.values,
                mode='lines', line=dict(color=pos_c, width=1.5, shape='spline', smoothing=0.8),
                showlegend=False, hovertemplate='%{y:.1f}<extra></extra>'), row=row, col=col)
            fig.add_hline(y=100, line=dict(color=_grd, width=0.8, dash='dot'), row=row, col=col)

            axis_idx = (row - 1) * n_cols + col
            global_rank = batch_start + i + 1
            fig.add_annotation(
                text=f"<b>{global_rank}</b>", x=0.02, y=0.95,
                xref=f"x{'' if axis_idx == 1 else axis_idx} domain",
                yref=f"y{'' if axis_idx == 1 else axis_idx} domain",
                showarrow=False, font=dict(size=12, color=_mut, family=FONTS),
                xanchor='left', yanchor='top')

        for ann in fig['layout']['annotations']:
            xref_str = str(ann['xref']) if ann['xref'] else ''
            if 'domain' not in xref_str:
                ann['font'] = dict(size=10, family=FONTS)

        chart_h = 350 * n_rows if is_mobile else 220 * n_rows
        fig.update_layout(
            template='plotly_dark', height=chart_h,
            margin=dict(l=40, r=40, t=45, b=30),
            plot_bgcolor=_pbg, paper_bgcolor=_pbg,
            showlegend=False, hovermode='x unified', font=dict(family=FONTS))
        fig.update_xaxes(gridcolor=_grd, linecolor=_axl,
            tickfont=dict(color=_tk, size=8, family=FONTS), showgrid=False, tickangle=0)
        fig.update_yaxes(gridcolor=_grd, linecolor=_axl,
            tickfont=dict(color=_tk, size=8, family=FONTS), side='right')

        st.plotly_chart(fig, use_container_width=True, config={
            'scrollZoom': True, 'displayModeBar': False, 'responsive': True})
