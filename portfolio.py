import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from collections import OrderedDict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from config import FUTURES_GROUPS, THEMES, SYMBOL_NAMES, FONTS, MONO, clean_symbol

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

C_POS = '#60a5fa'; C_NEG = '#fb7185'; C_TXT = '#cccccc'; C_TXT2 = '#9d9d9d'
C_MUTE = '#6d6d6d'; C_BG = '#1e1e1e'; C_HDR = '#16213e'; C_BORDER = '#3c3c3c'
C_GOLD = '#fbbf24'
TH = "padding:3px 6px;border-bottom:1px solid #3a3a3a;color:#8a8a8a;font-weight:500;font-size:9px;text-transform:uppercase;letter-spacing:0.06em;"
TD = "padding:4px 6px;border-bottom:1px solid #2a2a2a;"

def _short(sym):
    return SYMBOL_NAMES.get(sym, sym.replace('=F','').replace('=X','').replace('.SI',''))

# =============================================================================
# PRESETS
# =============================================================================

PRESETS = OrderedDict([
    ('Core 5',     ['IAU', 'VOO', 'VTI', 'SHV', 'BTC-USD']),
    ('Singapore',  ['ES3.SI', 'S68.SI', 'MBH.SI', 'MMS.SI']),
    ('Macro',      ['DBC', 'USO', 'GLD', 'SLV', 'CPER', 'BIL', 'HYG', 'LQD', 'TLT', 'BND', 'EMB', 'EEM', 'SPY', 'BTC-USD', 'ETH-USD']),
    ('Countries',  ['EWA', 'EWZ', 'EWC', 'GXC', 'EWQ', 'EWG', 'EWH', 'PIN', 'EWI', 'EWJ', 'EWM', 'EWW', 'EWS', 'EWY', 'EWP', 'EWT', 'EWU', 'VNM', 'KSA', 'ARGT']),
    ('US Sectors', ['XLB', 'XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLU', 'XLRE', 'SPY']),
])

PORTFOLIO_APPROACHES = OrderedDict([
    ('3mo',              {'windows': OrderedDict([('3mo', 63)]),
                          'blend': {'3mo': 1.0}, 'min_days': 63}),
    ('6mo',              {'windows': OrderedDict([('6mo', 126)]),
                          'blend': {'6mo': 1.0}, 'min_days': 126}),
    ('12mo',             {'windows': OrderedDict([('12mo', 252)]),
                          'blend': {'12mo': 1.0}, 'min_days': 252}),
    ('24mo',             {'windows': OrderedDict([('24mo', 504)]),
                          'blend': {'24mo': 1.0}, 'min_days': 504}),
    ('12mo Avg',         {'windows': OrderedDict([('3mo', 63), ('6mo', 126), ('9mo', 189), ('12mo', 252)]),
                          'blend': {'3mo': 0.25, '6mo': 0.25, '9mo': 0.25, '12mo': 0.25}, 'min_days': 252}),
    ('12mo Recency',     {'windows': OrderedDict([('3mo', 63), ('6mo', 126), ('9mo', 189), ('12mo', 252)]),
                          'blend': {'3mo': 0.40, '6mo': 0.30, '9mo': 0.20, '12mo': 0.10}, 'min_days': 252}),
    ('12mo Inv Recency', {'windows': OrderedDict([('3mo', 63), ('6mo', 126), ('9mo', 189), ('12mo', 252)]),
                          'blend': {'3mo': 0.10, '6mo': 0.20, '9mo': 0.30, '12mo': 0.40}, 'min_days': 252}),
    ('24mo Avg',         {'windows': OrderedDict([('3mo', 63), ('6mo', 126), ('12mo', 252), ('24mo', 504)]),
                          'blend': {'3mo': 0.25, '6mo': 0.25, '12mo': 0.25, '24mo': 0.25}, 'min_days': 504}),
    ('24mo Recency',     {'windows': OrderedDict([('3mo', 63), ('6mo', 126), ('12mo', 252), ('24mo', 504)]),
                          'blend': {'3mo': 0.40, '6mo': 0.30, '12mo': 0.20, '24mo': 0.10}, 'min_days': 504}),
    ('24mo Inv Recency', {'windows': OrderedDict([('3mo', 63), ('6mo', 126), ('12mo', 252), ('24mo', 504)]),
                          'blend': {'3mo': 0.10, '6mo': 0.20, '12mo': 0.30, '24mo': 0.40}, 'min_days': 504}),
])

REBAL_OPTIONS = OrderedDict([
    ('Monthly', 1), ('Quarterly', 3), ('Semi-Annual', 6), ('Annual', 12),
])

PERIOD_OPTIONS = OrderedDict([
    ('1 Year', 365), ('2 Years', 730), ('5 Years', 1825),
    ('10 Years', 3650), ('Max', 9999),
])

SCORE_TO_RANK = {
    'Win Rate': 'win_rate', 'Sharpe': 'sharpe', 'Sortino': 'sortino',
    'MAR': 'mar', 'R²': 'r2', 'Composite': 'sharpe',
}

# =============================================================================
# DATA FETCHING
# =============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def fetch_symbol_history(symbols_tuple, days=1800):
    symbols = list(symbols_tuple)
    if not symbols: return None, []
    start = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
    data = pd.DataFrame(); valid = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(start=start)
            if not hist.empty and len(hist) >= 50:
                closes = hist['Close'].copy()
                closes.index = closes.index.tz_localize(None) if closes.index.tz else closes.index
                closes.index = closes.index.normalize()
                closes = closes.groupby(closes.index).last()
                data[sym] = closes; valid.append(sym)
        except: pass
    if len(valid) < 2: return None, valid
    data = data[valid].ffill().dropna()
    if len(data) < 50: return None, valid
    return data, valid

# =============================================================================
# MC OPTIMIZATION ENGINE
# =============================================================================

def _optimize_window_vectorized(returns_array, n_portfolios, n_assets, max_weight,
                                score_type='Win Rate', min_weight=0.0, allow_short=False):
    if allow_short:
        weights = np.random.randn(n_portfolios, n_assets)
        weights = weights / weights.sum(axis=1, keepdims=True)
        for _ in range(30):
            violated = (weights > max_weight) | (weights < -max_weight)
            if not np.any(violated): break
            weights = np.clip(weights, -max_weight, max_weight)
            weights = weights / weights.sum(axis=1, keepdims=True)
    elif n_assets == 2:
        lo = max(1 - max_weight, min_weight); hi = min(max_weight, 1 - min_weight)
        if lo >= hi: lo, hi = 0.3, 0.7
        w1 = np.random.uniform(lo, hi, n_portfolios)
        weights = np.column_stack([w1, 1 - w1])
    else:
        n_half = n_portfolios // 2
        w_conc = np.random.dirichlet(np.ones(n_assets) * 0.5, n_half)
        w_div = np.random.dirichlet(np.ones(n_assets) * 1.0, n_portfolios - n_half)
        weights = np.vstack([w_conc, w_div])
        for _ in range(30):
            violated = (weights > max_weight) | (weights < min_weight)
            if not np.any(violated): break
            weights = np.maximum(weights, min_weight)
            weights = np.minimum(weights, max_weight)
            weights = weights / weights.sum(axis=1, keepdims=True)

    port_returns = weights @ returns_array.T
    ann_rets = np.mean(port_returns, axis=1) * 252
    ann_vols = np.std(port_returns, axis=1, ddof=1) * np.sqrt(252)

    def _vec_downside_vol(pr):
        neg = np.minimum(pr, 0)
        n_neg = np.maximum(np.sum(pr < 0, axis=1).astype(float), 1)
        return np.sqrt(np.sum(neg**2, axis=1) / n_neg) * np.sqrt(252)

    def _vec_avg_dd(pr):
        cum = np.cumprod(1 + pr, axis=1)
        peak = np.maximum.accumulate(cum, axis=1)
        dd = (cum - peak) / peak
        neg_dd = np.where(dd < 0, dd, 0)
        n_neg = np.maximum(np.sum(dd < 0, axis=1).astype(float), 1)
        return np.sum(neg_dd, axis=1) / n_neg

    if score_type == 'Win Rate':
        scores = np.mean(port_returns > 0, axis=1)
    elif score_type == 'Sortino':
        dv = _vec_downside_vol(port_returns)
        scores = np.where(dv > 0, ann_rets / dv, 0)
    elif score_type == 'MAR':
        avg_dd = _vec_avg_dd(port_returns)
        scores = np.where(avg_dd < 0, ann_rets / np.abs(avg_dd), 0)
    elif score_type == 'R²':
        cum = np.cumprod(1 + port_returns, axis=1)
        n = cum.shape[1]; x = np.arange(n, dtype=float); xm = x.mean()
        ss_xx = np.sum(x * x) - n * xm * xm
        ym = cum.mean(axis=1, keepdims=True)
        ss_xy = (cum @ x) - n * xm * ym.ravel()
        ss_yy = np.sum(cum * cum, axis=1) - n * (ym.ravel() ** 2)
        denom = ss_xx * ss_yy
        scores = np.where(denom > 0, (ss_xy ** 2) / denom, 0)
        slope = np.where(ss_xx > 0, ss_xy / ss_xx, 0)
        scores = np.where(slope > 0, scores, -scores)
    elif score_type == 'Composite':
        sharpes = np.where(ann_vols > 0, ann_rets / ann_vols, 0)
        dv = _vec_downside_vol(port_returns)
        sortinos = np.where(dv > 0, ann_rets / dv, 0)
        win_rates = np.mean(port_returns > 0, axis=1)
        def _rank_pct(a):
            r = a.argsort().argsort().astype(float)
            return r / max(len(r) - 1, 1)
        scores = 0.4 * _rank_pct(sharpes) + 0.3 * _rank_pct(sortinos) + 0.3 * _rank_pct(win_rates)
    else:  # Sharpe
        scores = np.where(ann_vols > 0, ann_rets / ann_vols, 0)

    return weights[np.argmax(scores)]

# =============================================================================
# WALK-FORWARD ENGINE
# =============================================================================

def _optimize_at_rebalance(returns_df, approach, score_type, n_portfolios, mw, mnw=0.0, allow_short=False):
    n_assets = returns_df.shape[1]; data_len = len(returns_df)
    window_weights_list = []; blend_wts = []
    for wname, wdays in approach['windows'].items():
        if data_len >= wdays:
            w_ret = returns_df.iloc[-wdays:]
            best_w = _optimize_window_vectorized(w_ret.values, n_portfolios, n_assets, mw, score_type, mnw, allow_short)
            window_weights_list.append(best_w); blend_wts.append(approach['blend'][wname])
    if not window_weights_list: return None
    blend_wts = np.array(blend_wts); blend_wts /= blend_wts.sum()
    all_w = np.array(window_weights_list)
    opt_w = np.average(all_w, axis=0, weights=blend_wts)
    opt_w /= opt_w.sum()
    return opt_w


def _walk_forward_single(returns_df, approach, score_type, rebal_months,
                         n_portfolios=10000, max_weight=0.50, min_weight=0.0,
                         txn_cost=0.001, allow_short=False):
    n_assets = returns_df.shape[1]; mw = max_weight; mnw = min_weight
    min_is_days = max(approach['windows'].values()); dates = returns_df.index

    if rebal_months == 1: rebal_month_set = set(range(1, 13))
    elif rebal_months == 3: rebal_month_set = {1, 4, 7, 10}
    elif rebal_months == 6: rebal_month_set = {1, 7}
    else: rebal_month_set = {1}

    candidate_dates = []; seen_months = set()
    for d in dates:
        ym = (d.year, d.month)
        if ym not in seen_months and d.month in rebal_month_set:
            seen_months.add(ym); candidate_dates.append(d)

    rebal_dates = []
    for d in candidate_dates:
        idx = dates.get_loc(d)
        if idx >= min_is_days: rebal_dates.append((idx, d))
    if len(rebal_dates) < 2: return None

    oos_segments = []; weight_history = []
    prev_weights = np.ones(n_assets) / n_assets

    for i, (ri, rd) in enumerate(rebal_dates):
        is_data = returns_df.iloc[:ri + 1]
        opt_w = _optimize_at_rebalance(is_data, approach, score_type, n_portfolios, mw, mnw, allow_short)
        if opt_w is None: continue
        oos_start = ri + 1
        oos_end = rebal_dates[i + 1][0] if i + 1 < len(rebal_dates) else len(dates)
        if oos_start >= oos_end: continue
        oos_data = returns_df.iloc[oos_start:oos_end]
        port_oos = oos_data.values @ opt_w
        turnover = np.sum(np.abs(opt_w - prev_weights)) / 2.0
        if txn_cost > 0 and turnover > 0: port_oos[0] -= turnover * txn_cost
        prev_weights = opt_w.copy()
        oos_segments.append(pd.Series(port_oos, index=oos_data.index))
        weight_history.append({'date': rd, 'weights': opt_w.copy(),
            'oos_start': dates[oos_start], 'oos_end': dates[min(oos_end - 1, len(dates) - 1)],
            'oos_days': len(oos_data)})

    if not oos_segments or len(weight_history) < 2: return None
    current_w = _optimize_at_rebalance(returns_df, approach, score_type, n_portfolios, mw, mnw, allow_short)
    if current_w is None: current_w = weight_history[-1]['weights']
    full_oos = pd.concat(oos_segments)
    return {'oos_returns': full_oos, 'weight_history': weight_history,
            'current_weights': current_w, 'last_rebalance': weight_history[-1]['date']}


def _calc_oos_metrics(returns_series):
    r = returns_series.values; n = len(r)
    if n < 5: return None
    cum = np.cumprod(1 + r); total = float(cum[-1] - 1)
    ann_ret = float(np.mean(r) * 252); ann_vol = float(np.std(r, ddof=1) * np.sqrt(252))
    peak = np.maximum.accumulate(cum); dd = (cum - peak) / peak
    max_dd = float(np.min(dd)); avg_dd = float(np.mean(dd[dd < 0])) if np.any(dd < 0) else 0.0
    win_rate = float(np.sum(r > 0) / n)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    neg = np.minimum(r, 0); n_neg = max(np.sum(r < 0), 1)
    down_vol = float(np.sqrt(np.sum(neg**2) / n_neg) * np.sqrt(252))
    sortino = float(ann_ret / down_vol) if down_vol > 0 else 0.0
    mar = float(ann_ret / abs(avg_dd)) if avg_dd != 0 else 0.0
    if n > 2:
        x = np.arange(n, dtype=float); xm, ym = x.mean(), cum.mean()
        ss_xy = np.sum(x * cum) - n * xm * ym
        ss_xx = np.sum(x * x) - n * xm * xm
        ss_yy = np.sum(cum * cum) - n * ym * ym
        slope = ss_xy / ss_xx if ss_xx else 0
        r2 = float(np.clip((ss_xy**2) / (ss_xx * ss_yy), 0, 1)) if (ss_xx * ss_yy) else 0.0
        r2 = r2 if slope > 0 else -r2
    else: r2 = 0.0
    now = pd.Timestamp.now(); idx = returns_series.index
    ytd_mask = idx >= pd.Timestamp(now.year, 1, 1)
    mtd_mask = idx >= pd.Timestamp(now.year, now.month, 1)
    ytd = float(np.prod(1 + r[ytd_mask]) - 1) if ytd_mask.any() else 0.0
    mtd = float(np.prod(1 + r[mtd_mask]) - 1) if mtd_mask.any() else 0.0
    oos_years = (idx[-1] - idx[0]).days / 365.25
    return {'total_ret': total, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
            'max_dd': max_dd, 'avg_dd': avg_dd, 'win_rate': win_rate,
            'sharpe': sharpe, 'sortino': sortino, 'mar': mar, 'r2': r2,
            'ytd': ytd, 'mtd': mtd, 'n_days': n, 'oos_years': round(oos_years, 1)}

# =============================================================================
# GRID SEARCH
# =============================================================================

def run_walkforward_grid(symbols, score_type='Win Rate', rebal_months=3, n_portfolios=10000,
                         fetch_days=1800, max_weight=0.50, min_weight=0.0,
                         txn_cost=0.001, allow_short=False, progress_bar=None):
    data, valid = fetch_symbol_history(tuple(symbols), days=fetch_days)
    if data is None or len(valid) < 2: return None
    returns = data.pct_change().dropna(); n_assets = len(valid)

    # Equal weight benchmark with drift + transaction costs
    eq_w = np.ones(n_assets) / n_assets
    if rebal_months == 1: rebal_month_set = set(range(1, 13))
    elif rebal_months == 3: rebal_month_set = {1, 4, 7, 10}
    elif rebal_months == 6: rebal_month_set = {1, 7}
    else: rebal_month_set = {1}
    dates = returns.index; eq_rebal_set = set(); seen = set()
    for d in dates:
        ym = (d.year, d.month)
        if ym not in seen and d.month in rebal_month_set: seen.add(ym); eq_rebal_set.add(d)
    ret_arr = returns.values; n_days = len(ret_arr); eq_daily = np.zeros(n_days)
    curr_w = eq_w.copy()
    for t in range(n_days):
        eq_daily[t] = curr_w @ ret_arr[t]
        grown = curr_w * (1 + ret_arr[t]); curr_w = grown / grown.sum()
        if t + 1 < n_days and dates[t + 1] in eq_rebal_set:
            turnover = np.sum(np.abs(eq_w - curr_w)) / 2.0
            eq_daily[t] -= turnover * txn_cost; curr_w = eq_w.copy()
    eq_ret = pd.Series(eq_daily, index=returns.index)

    results = OrderedDict()
    approach_list = list(PORTFOLIO_APPROACHES.items())
    for i, (name, approach) in enumerate(approach_list):
        if progress_bar: progress_bar.progress((i + 1) / len(approach_list), text=f'Walk-forward: {name}')
        try:
            wf = _walk_forward_single(returns, approach, score_type, rebal_months,
                                      n_portfolios, max_weight, min_weight, txn_cost, allow_short)
            if wf is not None:
                metrics = _calc_oos_metrics(wf['oos_returns'])
                if metrics is not None:
                    metrics['n_rebalances'] = len(wf['weight_history'])
                    results[name] = {'wf': wf, 'metrics': metrics}
        except Exception as e:
            logger.warning(f"Walk-forward failed for {name}: {e}")

    if not results: return None
    earliest_oos = min(r['wf']['oos_returns'].index[0] for r in results.values())
    eq_trimmed = eq_ret.loc[eq_ret.index >= earliest_oos]
    eq_metrics = _calc_oos_metrics(eq_trimmed)
    return {'results': results, 'symbols': valid, 'returns': returns,
            'eq_returns': eq_trimmed, 'eq_metrics': eq_metrics,
            'score_type': score_type, 'rebal_months': rebal_months, 'txn_cost': txn_cost}

# =============================================================================
# DISPLAY: RANKING TABLE
# =============================================================================

def _fc(v, fmt='f2', neg_is_bad=True):
    if fmt == 'pct': s = f"{v*100:.1f}%"
    elif fmt == 'f3': s = f"{v:.3f}"
    else: s = f"{v:.2f}"
    if neg_is_bad: c = C_POS if v > 0 else (C_NEG if v < 0 else C_TXT)
    else: c = C_NEG
    return f"<span style='color:{c}'>{s}</span>"


def render_ranking_table(grid, rank_by='win_rate'):
    results = grid['results']; eq = grid['eq_metrics']
    lower_better = {'ann_vol', 'max_dd', 'avg_dd'}
    items = [(name, r['metrics']) for name, r in results.items()]
    reverse = rank_by not in lower_better
    items.sort(key=lambda x: x[1].get(rank_by, 0), reverse=reverse)
    best_name = items[0][0] if items else None

    html = f"<div style='background:{C_BG};overflow-x:auto'><table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>"
    html += "<thead><tr>"
    for label, align in [('#','left'),('Approach','left'),('Win%','right'),('Sharpe','right'),
                          ('Sortino','right'),('MAR','right'),('R²','right'),('Total','right'),
                          ('Ann Ret','right'),('Vol','right'),('MaxDD','right'),('YTD','right'),
                          ('OOS','right'),('Rebals','right')]:
        html += f"<th style='{TH}text-align:{align}'>{label}</th>"
    html += "</tr></thead><tbody>"

    for rank, (name, m) in enumerate(items, 1):
        is_best = name == best_name
        bg = 'rgba(96,165,250,0.08)' if is_best else 'transparent'
        badge = f" <span style='color:{C_GOLD};font-size:9px'>★</span>" if is_best else ""
        nc = C_GOLD if is_best else C_TXT; fw = '700' if is_best else '500'
        best_border = f'border-top:2px solid {C_GOLD};border-bottom:2px solid {C_GOLD};' if is_best else ''
        html += f"<tr style='background:{bg};{best_border}'>"
        html += f"<td style='{TD}color:{C_MUTE}'>{rank}</td>"
        html += f"<td style='{TD}color:{nc};font-weight:{fw}'>{name}{badge}</td>"
        html += f"<td style='{TD}text-align:right;font-weight:700'>{_fc(m['win_rate'],'pct')}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['sharpe'])}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['sortino'])}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['mar'])}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['r2'],'f3')}</td>"
        html += f"<td style='{TD}text-align:right;font-weight:600'>{_fc(m['total_ret'],'pct')}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['ann_ret'],'pct')}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['ann_vol'],'pct',False)}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['max_dd'],'pct',False)}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(m['ytd'],'pct')}</td>"
        html += f"<td style='{TD}text-align:right;color:{C_TXT2}'>{m['oos_years']}y</td>"
        html += f"<td style='{TD}text-align:right;color:{C_TXT2}'>{m['n_rebalances']}</td>"
        html += "</tr>"

    # Equal weight row
    if eq:
        html += f"<tr style='background:rgba(255,255,255,0.04);border-top:2px solid {C_GOLD};border-bottom:2px solid {C_GOLD}'>"
        html += f"<td style='{TD}color:{C_MUTE}'>—</td>"
        html += f"<td style='{TD}color:#e2e8f0;font-weight:700'><span style='color:{C_GOLD};font-size:9px'>◆</span> Equal Weight (1/N)</td>"
        html += f"<td style='{TD}text-align:right;font-weight:700'>{_fc(eq['win_rate'],'pct')}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(eq['sharpe'])}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(eq['sortino'])}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(eq['mar'])}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(eq['r2'],'f3')}</td>"
        html += f"<td style='{TD}text-align:right;font-weight:600'>{_fc(eq['total_ret'],'pct')}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(eq['ann_ret'],'pct')}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(eq['ann_vol'],'pct',False)}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(eq['max_dd'],'pct',False)}</td>"
        html += f"<td style='{TD}text-align:right'>{_fc(eq['ytd'],'pct')}</td>"
        html += f"<td style='{TD}text-align:right;color:#e2e8f0'>{eq['oos_years']}y</td>"
        html += f"<td style='{TD}text-align:right;color:{C_TXT2}'>—</td></tr>"

        # Delta row
        if best_name and items:
            best_m = items[0][1]
            higher_better = {'win_rate','sharpe','sortino','mar','r2','total_ret','ann_ret','ytd'}
            html += f"<tr style='background:rgba(251,191,36,0.06)'>"
            html += f"<td style='{TD}color:{C_GOLD};font-size:9px'>Δ</td>"
            html += f"<td style='{TD}color:{C_GOLD};font-size:10px;font-weight:600'>★ vs Equal Weight</td>"
            for key, fmt in [('win_rate','pct'),('sharpe','f2'),('sortino','f2'),('mar','f2'),
                             ('r2','f3'),('total_ret','pct'),('ann_ret','pct'),('ann_vol','pct'),
                             ('max_dd','pct'),('ytd','pct')]:
                bv = best_m[key]; ev = eq[key]; d = bv - ev
                if key in higher_better: good = d > 0
                elif key in {'ann_vol','max_dd'}: good = abs(bv) < abs(ev)
                else: good = d > 0
                c = '#4ade80' if good else '#fb7185'; sign = '+' if d > 0 else ''
                if fmt == 'pct': ds = f"{sign}{d*100:.1f}%"
                elif fmt == 'f3': ds = f"{sign}{d:.3f}"
                else: ds = f"{sign}{d:.2f}"
                if abs(d) < 1e-6: ds = "—"; c = C_MUTE
                html += f"<td style='{TD}text-align:right;color:{c};font-weight:600;font-size:10px'>{ds}</td>"
            html += f"<td style='{TD}text-align:right;color:{C_MUTE}'>—</td>"
            html += f"<td style='{TD}text-align:right;color:{C_MUTE}'>—</td></tr>"

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)
    sorted_names = [name for name, _ in items]
    return best_name, sorted_names

# =============================================================================
# DISPLAY: WEIGHTS TABLE
# =============================================================================

def render_weights_table(grid, approach_name):
    wf = grid['results'][approach_name]['wf']
    syms = grid['symbols']; w = wf['current_weights']
    n_assets = len(syms); eq_w = 1.0 / n_assets
    sorted_idx = np.argsort(-w)

    html = f"<div style='background:{C_BG};overflow-x:auto'><table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>"
    html += f"<thead><tr><th style='{TH}text-align:left'>Asset</th><th style='{TH}text-align:left;width:60px'>Ticker</th>"
    html += f"<th style='{TH}text-align:right'>Weight</th><th style='{TH}text-align:right'>vs EW</th>"
    html += f"<th style='{TH}text-align:left;width:140px'>Allocation</th></tr></thead><tbody>"

    for i in sorted_idx:
        sym = syms[i]; sn = _short(sym); wi = w[i]; delta = wi - eq_w
        if wi < 0: wc = C_NEG
        elif wi > 0.20: wc = C_POS
        elif wi > 0.05: wc = C_TXT
        else: wc = C_MUTE
        dc = '#4ade80' if delta > 0.01 else ('#fb7185' if delta < -0.01 else C_MUTE)
        ds = '+' if delta > 0 else ''
        bar_pct = min(abs(wi) / 0.50 * 100, 100)
        bar_color = C_NEG if wi < 0 else C_POS
        bar = (f"<div style='background:#2a2a2a;border-radius:2px;height:10px;width:100%'>"
               f"<div style='background:{bar_color};border-radius:2px;height:10px;width:{bar_pct:.0f}%'></div></div>")
        html += f"<tr><td style='{TD}color:{wc};font-weight:600'>{sn}</td>"
        html += f"<td style='{TD}color:{C_MUTE};font-size:10px'>{sym}</td>"
        html += f"<td style='{TD}text-align:right;color:{wc};font-weight:700;font-size:12px'>{wi*100:.1f}%</td>"
        html += f"<td style='{TD}text-align:right;color:{dc};font-size:10px'>{ds}{delta*100:.1f}%</td>"
        html += f"<td style='{TD}'>{bar}</td></tr>"

    html += f"<tr style='border-top:2px solid {C_BORDER}'>"
    html += f"<td style='{TD}color:{C_TXT};font-weight:700'>TOTAL</td><td style='{TD}'></td>"
    html += f"<td style='{TD}text-align:right;color:{C_TXT};font-weight:700'>{np.sum(w)*100:.1f}%</td>"
    html += f"<td colspan='2' style='{TD}color:{C_MUTE};font-size:10px'>Optimized on all data through today</td></tr>"
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# DISPLAY: OOS EQUITY CHART
# =============================================================================

def render_oos_chart(grid, approach_name):
    wf = grid['results'][approach_name]['wf']
    oos = wf['oos_returns']; m = grid['results'][approach_name]['metrics']; eq_m = grid['eq_metrics']
    eq_aligned = grid['eq_returns'].loc[grid['eq_returns'].index >= oos.index[0]]
    opt_cum = np.cumprod(1 + oos.values); eq_cum = np.cumprod(1 + eq_aligned.values)
    opt_peak = np.maximum.accumulate(opt_cum); opt_dd = (opt_cum - opt_peak) / opt_peak
    opt_pct = (opt_cum[-1] - 1) * 100; eq_pct = (eq_cum[-1] - 1) * 100

    opt_lbl = (f'{approach_name} ({opt_pct:+.1f}%)  Sharpe {m["sharpe"]:.2f} · Sortino {m["sortino"]:.2f} · '
               f'MAR {m["mar"]:.1f} · Win {m["win_rate"]*100:.0f}% · R² {m["r2"]:.3f}')
    eq_lbl = (f'Equal Weight ({eq_pct:+.1f}%)  Sharpe {eq_m["sharpe"]:.2f} · Sortino {eq_m["sortino"]:.2f} · '
              f'MAR {eq_m["mar"]:.1f} · Win {eq_m["win_rate"]*100:.0f}% · R² {eq_m["r2"]:.3f}')

    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True, vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=oos.index, y=opt_cum, mode='lines', line=dict(color=C_POS, width=2),
        name=opt_lbl, hovertemplate='WF: $%{y:.3f}<extra></extra>'), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq_aligned.index, y=eq_cum, mode='lines',
        line=dict(color='#ffffff', width=1.2, dash='dot'), name=eq_lbl,
        hovertemplate='EW: $%{y:.3f}<extra></extra>'), row=1, col=1)
    for wh in wf['weight_history']:
        if wh['date'] >= oos.index[0]:
            fig.add_vline(x=wh['date'], line=dict(color=C_GOLD, width=0.6, dash='dot'), opacity=0.4, row=1, col=1)
    fig.add_hline(y=1.0, line=dict(color='#333333', width=0.8, dash='dash'), row=1, col=1)

    nr = C_NEG.lstrip('#'); rv, gv, bv = int(nr[:2], 16), int(nr[2:4], 16), int(nr[4:6], 16)
    fig.add_trace(go.Scatter(x=oos.index, y=opt_dd * 100, mode='lines', fill='tozeroy',
        line=dict(color=C_NEG, width=1), fillcolor=f'rgba({rv},{gv},{bv},0.2)',
        name='Drawdown', showlegend=False, hovertemplate='DD: %{y:.1f}%<extra></extra>'), row=2, col=1)

    title = ', '.join(_short(s) for s in grid['symbols'][:5])
    if len(grid['symbols']) > 5: title += f" +{len(grid['symbols'])-5}"
    fig.update_layout(template='plotly_dark', height=400, margin=dict(l=55, r=30, t=35, b=25),
        plot_bgcolor='#0a0a0a', paper_bgcolor='#0a0a0a', showlegend=True,
        legend=dict(x=0.01, y=0.88, bgcolor='rgba(0,0,0,0)', font=dict(size=9, color='#ffffff', family=MONO), borderwidth=0),
        hovermode='x unified', font=dict(family=MONO))
    fig.add_annotation(text=f"<b>{title.upper()}</b>  <span style='font-size:10px;color:{C_GOLD}'>OOS WALK-FORWARD</span>",
        x=0.01, y=0.99, xref='paper', yref='paper', showarrow=False,
        font=dict(size=13, color='#ffffff', family=MONO), xanchor='left', yanchor='top')
    fig.update_xaxes(gridcolor='#1a1a1a', linecolor='#222', tickfont=dict(size=8, color='#555', family=MONO))
    fig.update_yaxes(gridcolor='#1a1a1a', linecolor='#222', tickfont=dict(size=8, color='#555', family=MONO), side='right')
    fig.update_yaxes(tickprefix='$', tickformat='.2f', row=1, col=1)
    fig.update_yaxes(ticksuffix='%', row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'responsive': True})

# =============================================================================
# DISPLAY: MONTHLY RETURNS
# =============================================================================

def render_monthly_table(oos_returns):
    monthly = oos_returns.groupby([oos_returns.index.year, oos_returns.index.month]).apply(
        lambda x: float((1 + x).prod() - 1))
    years = sorted(monthly.index.get_level_values(0).unique())
    mlbl = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    html = f"<div style='background:{C_BG};overflow-x:auto'><table style='border-collapse:collapse;font-family:{FONTS};font-size:11px;width:100%;line-height:1.3'>"
    html += f"<thead><tr><th style='{TH}text-align:left'>Year</th>"
    for m in mlbl: html += f"<th style='{TH}text-align:right'>{m}</th>"
    html += f"<th style='{TH}text-align:right;font-weight:700'>YTD</th></tr></thead><tbody>"
    for yr in years:
        html += f"<tr><td style='{TD}color:{C_TXT};font-weight:700'>{yr}</td>"
        ytd = 1.0
        for mo in range(1, 13):
            if (yr, mo) in monthly.index:
                v = monthly[(yr, mo)]; ytd *= (1 + v)
                c = C_POS if v >= 0 else C_NEG
                html += f"<td style='{TD}text-align:right;color:{c}'>{v*100:.1f}%</td>"
            else: html += f"<td style='{TD}text-align:right;color:{C_MUTE}'>-</td>"
        yv = ytd - 1; yc = C_POS if yv >= 0 else C_NEG
        html += f"<td style='{TD}text-align:right;color:{yc};font-weight:700'>{yv*100:.1f}%</td></tr>"
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# SECTION HEADER HELPER
# =============================================================================

def _section(title, subtitle=''):
    sub = f"<span style='color:{C_MUTE};font-size:10px;margin-left:8px'>{subtitle}</span>" if subtitle else ""
    st.markdown(f"""<div style='margin-top:12px;padding:6px 10px;background:{C_HDR};font-family:{FONTS}'>
        <span style='color:#e2e8f0;font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase'>{title}</span>{sub}
    </div>""", unsafe_allow_html=True)

# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_portfolio_tab(is_mobile):
    global C_POS, C_NEG
    theme_name = st.session_state.get('theme', 'Blue / Rose')
    theme = THEMES.get(theme_name, THEMES['Blue / Rose'])
    C_POS = theme['pos']; C_NEG = theme['neg']
    _lbl = f"color:#e2e8f0;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"

    # Symbols input
    if 'port_sym_input' not in st.session_state:
        st.session_state.port_sym_input = 'IAU, VOO, VTI, SHV, BTC-USD'
    st.markdown(f"<div style='{_lbl}'>SYMBOLS</div>", unsafe_allow_html=True)
    sym_input = st.text_input("Symbols", key='port_sym_input', label_visibility='collapsed',
                               placeholder='Enter symbols: AAPL, MSFT, GOOG')

    # Preset buttons — use on_click callbacks (run before rerender)
    def _set_preset(syms_str):
        st.session_state.port_sym_input = syms_str

    st.markdown(f"<div style='{_lbl};margin-top:4px'>PRESETS</div>", unsafe_allow_html=True)
    preset_cols = st.columns(len(PRESETS) + 1)
    for i, (name, syms) in enumerate(PRESETS.items()):
        with preset_cols[i]:
            st.button(name, key=f'preset_{name}', use_container_width=True,
                      on_click=_set_preset, args=(', '.join(syms),))
    with preset_cols[-1]:
        st.button('Custom', key='preset_custom', use_container_width=True,
                  on_click=_set_preset, args=('',))

    # Controls row 1: Objective + Rebalance + Period + Direction
    if is_mobile:
        c1, c2, c3, c4 = st.columns(4)
    else:
        c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div style='{_lbl}'>OBJECTIVE</div>", unsafe_allow_html=True)
        score = st.selectbox("Objective", ['Win Rate', 'Composite', 'Sharpe', 'Sortino', 'MAR', 'R²'],
                              key='port_score', label_visibility='collapsed')
    with c2:
        st.markdown(f"<div style='{_lbl}'>REBALANCE</div>", unsafe_allow_html=True)
        rebal_label = st.selectbox("Rebalance", list(REBAL_OPTIONS.keys()),
                                    index=1, key='port_rebal', label_visibility='collapsed')
    with c3:
        st.markdown(f"<div style='{_lbl}'>PERIOD</div>", unsafe_allow_html=True)
        period_label = st.selectbox("Period", list(PERIOD_OPTIONS.keys()),
                                     index=3, key='port_period', label_visibility='collapsed')
    with c4:
        st.markdown(f"<div style='{_lbl}'>DIRECTION</div>", unsafe_allow_html=True)
        direction = st.selectbox("Direction", ['Long Only', 'Long/Short'],
                                  key='port_direction', label_visibility='collapsed')

    # Controls row 2: Max Wt + Min Wt + Sims + Cost
    _defaults = {'port_maxwt': '50', 'port_minwt': '0', 'port_sims': '10000', 'port_cost': '0.10'}
    for k, v in _defaults.items():
        if not st.session_state.get(k):  # handles missing AND empty string
            st.session_state[k] = v
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(f"<div style='{_lbl}'>MAX WT %</div>", unsafe_allow_html=True)
        max_wt_str = st.text_input("Max Wt", key='port_maxwt', label_visibility='collapsed')
    with c6:
        st.markdown(f"<div style='{_lbl}'>MIN WT %</div>", unsafe_allow_html=True)
        min_wt_str = st.text_input("Min Wt", key='port_minwt', label_visibility='collapsed')
    with c7:
        st.markdown(f"<div style='{_lbl}'>SIMS</div>", unsafe_allow_html=True)
        sims_str = st.text_input("Sims", key='port_sims', label_visibility='collapsed')
    with c8:
        st.markdown(f"<div style='{_lbl}'>COST %</div>", unsafe_allow_html=True)
        cost_str = st.text_input("Cost", key='port_cost', label_visibility='collapsed')

    # Run button
    run_clicked = st.button('▶  RUN GRID', key='port_run', type='primary', use_container_width=False)

    if not run_clicked and 'port_grid' not in st.session_state:
        st.markdown(f"<div style='padding:20px;color:{C_MUTE};font-size:11px;font-family:{FONTS}'>Configure parameters and click RUN GRID to start walk-forward optimization</div>", unsafe_allow_html=True)
        return

    if run_clicked:
        raw = sym_input.strip()
        if not raw:
            st.warning('Enter symbols to optimize'); return
        symbols = [s.strip().upper() for s in raw.replace(';', ',').split(',') if s.strip()]
        symbols = list(dict.fromkeys(symbols))

        try: max_wt = max(10, min(100, float(max_wt_str))) / 100.0
        except: max_wt = 0.50
        try: min_wt = max(0, min(50, float(min_wt_str))) / 100.0
        except: min_wt = 0.0
        try: n_sims = max(1000, min(100000, int(sims_str)))
        except: n_sims = 10000
        try: txn_cost = max(0, min(5.0, float(cost_str))) / 100.0
        except: txn_cost = 0.001

        allow_short = direction == 'Long/Short'
        rebal = REBAL_OPTIONS[rebal_label]
        fetch_days = PERIOD_OPTIONS[period_label]
        n_syms = len(symbols)
        if min_wt > 0 and min_wt * n_syms > 1.0: min_wt = round(1.0 / n_syms, 4)
        if min_wt >= max_wt: min_wt = 0.0

        progress = st.progress(0, text='Starting walk-forward...')
        grid = run_walkforward_grid(symbols, score_type=score, rebal_months=rebal,
                                     fetch_days=fetch_days, n_portfolios=n_sims,
                                     max_weight=max_wt, min_weight=min_wt,
                                     txn_cost=txn_cost, allow_short=allow_short,
                                     progress_bar=progress)
        progress.empty()

        if not grid or not grid['results']:
            st.warning('Need ≥2 assets with sufficient history for walk-forward'); return

        st.session_state.port_grid = grid
        st.session_state.port_params = {
            'score': score, 'rebal_label': rebal_label, 'period_label': period_label,
            'direction': 'L/S' if allow_short else 'Long',
            'min_wt': min_wt, 'max_wt': max_wt, 'n_sims': n_sims, 'txn_cost': txn_cost,
        }
        # Reset approach selector to winner on new run
        if 'port_view_approach' in st.session_state:
            del st.session_state.port_view_approach

    # Display results
    if 'port_grid' not in st.session_state: return
    grid = st.session_state.port_grid
    params = st.session_state.port_params
    rank_metric = SCORE_TO_RANK.get(params['score'], 'win_rate')
    n_app = len(grid['results'])

    # 1. Approach Ranking
    _section('APPROACH RANKING',
             f"{n_app} lookbacks · {params['rebal_label']} · {params['period_label']} · "
             f"{params['direction']} · wt {params['min_wt']*100:.0f}–{params['max_wt']*100:.0f}% · "
             f"cost {params['txn_cost']*100:.2f}% · {params['n_sims']:,} sims · max {params['score']} · all OOS")
    result = render_ranking_table(grid, rank_metric)
    best_name, sorted_names = result
    if not best_name or not sorted_names: return

    # Approach selector — default to winner, user can pick any
    _lbl = f"color:#e2e8f0;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"
    # Reset to winner if current selection invalid
    cur = st.session_state.get('port_view_approach')
    if cur not in sorted_names:
        st.session_state.port_view_approach = sorted_names[0]
    sel_col, _ = st.columns([3, 5])
    with sel_col:
        st.markdown(f"<div style='{_lbl};margin-top:8px'>VIEW APPROACH</div>", unsafe_allow_html=True)
        selected_approach = st.selectbox("Approach", sorted_names,
                                          key='port_view_approach', label_visibility='collapsed')

    sel = grid['results'][selected_approach]; sm = sel['metrics']; swf = sel['wf']
    is_best = selected_approach == best_name
    star = f"<span style='color:{C_GOLD}'>★</span> " if is_best else ""

    # 2. Summary bar
    st.markdown(f"""<div style='margin-top:12px;padding:5px 10px;background:{C_BG};font-family:{FONTS};
        font-size:10px;color:#8a8a8a;display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px'>
        <span>{star}<b style='color:{C_TXT}'>{selected_approach}</b>
        &nbsp;·&nbsp;{sm['n_days']} OOS days · {sm['oos_years']}y · {sm['n_rebalances']} rebalances</span>
        <span>Win% <b style='color:{C_POS}'>{sm["win_rate"]*100:.1f}%</b>
        &nbsp;Sharpe <b style='color:{C_POS}'>{sm["sharpe"]:.2f}</b>
        &nbsp;Sortino <b style='color:{C_POS}'>{sm["sortino"]:.2f}</b>
        &nbsp;MAR <b style='color:{C_POS}'>{sm["mar"]:.2f}</b></span>
    </div>""", unsafe_allow_html=True)

    # 3. OOS Equity Curve
    _section('OOS EQUITY CURVE', f'{selected_approach} · {params["rebal_label"]} · yellow = rebalance dates')
    render_oos_chart(grid, selected_approach)

    # 4. Monthly Returns (above weights)
    _section('OOS MONTHLY RETURNS', f'{selected_approach} · walk-forward out-of-sample only')
    render_monthly_table(swf['oos_returns'])

    # 5. Current Weights (below monthly)
    _section('CURRENT / NEXT WEIGHTS', f'{selected_approach} · optimized on all data through today · trade these')
    render_weights_table(grid, selected_approach)
