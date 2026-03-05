"""
SANPO — Rates tab
US Treasury yield curve (Treasury.gov XML feed) + SG SGS yield curve (MAS API).
Overlay both curves, show 2s10s spread, SORA vs SOFR.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib.request
import json
import xml.etree.ElementTree as ET
import logging
from datetime import datetime, timedelta

from config import get_theme, FONTS

logger = logging.getLogger(__name__)

_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# =============================================================================
# US TREASURY YIELD CURVE
# =============================================================================

US_TENORS = ['1 Mo', '2 Mo', '3 Mo', '4 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr',
             '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']
US_TENOR_MONTHS = [1, 2, 3, 4, 6, 12, 24, 36, 60, 84, 120, 240, 360]
US_XML_FIELDS = ['d:BC_1MONTH', 'd:BC_2MONTH', 'd:BC_3MONTH', 'd:BC_4MONTH',
                 'd:BC_6MONTH', 'd:BC_1YEAR', 'd:BC_2YEAR', 'd:BC_3YEAR',
                 'd:BC_5YEAR', 'd:BC_7YEAR', 'd:BC_10YEAR', 'd:BC_20YEAR', 'd:BC_30YEAR']


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_us_curve():
    """Fetch latest US Treasury par yield curve from Treasury.gov XML feed."""
    try:
        year = datetime.now().year
        url = (f'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/'
               f'pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value={year}')
        req = urllib.request.Request(url, headers={'User-Agent': _UA})
        resp = urllib.request.urlopen(req, timeout=15)
        raw = resp.read()

        ns = {
            'a': 'http://www.w3.org/2005/Atom',
            'm': 'http://schemas.microsoft.com/ado/2007/08/dataservices/metadata',
            'd': 'http://schemas.microsoft.com/ado/2007/08/dataservices',
        }
        root = ET.fromstring(raw)
        entries = root.findall('.//a:entry', ns)
        if not entries:
            return None, None

        # Get last few entries (most recent dates)
        rows = []
        for entry in entries[-30:]:  # last 30 trading days
            props = entry.find('.//m:properties', ns)
            if props is None:
                continue
            date_el = props.find('d:NEW_DATE', ns)
            if date_el is None:
                continue
            date_str = date_el.text[:10] if date_el.text else ''
            yields = []
            for field in US_XML_FIELDS:
                el = props.find(field, ns)
                val = float(el.text) if el is not None and el.text else None
                yields.append(val)
            rows.append({'date': date_str, 'yields': yields})

        if not rows:
            return None, None

        # Latest curve
        latest = rows[-1]
        latest_date = latest['date']
        latest_yields = latest['yields']

        # Previous day for comparison
        prev_yields = rows[-2]['yields'] if len(rows) >= 2 else None

        # Historical for 2s10s
        hist_2s10s = []
        for r in rows:
            y2 = r['yields'][6]   # 2 Yr index
            y10 = r['yields'][10]  # 10 Yr index
            if y2 is not None and y10 is not None:
                hist_2s10s.append({'date': r['date'], 'spread': y10 - y2})

        return {
            'date': latest_date,
            'tenors': US_TENORS,
            'months': US_TENOR_MONTHS,
            'yields': latest_yields,
            'prev_yields': prev_yields,
            'hist_2s10s': hist_2s10s,
        }, None
    except Exception as e:
        logger.warning(f"US curve fetch error: {e}")
        return None, str(e)


# =============================================================================
# SG SGS YIELD CURVE (MAS API)
# =============================================================================

SG_TENORS = ['6 Mo', '1 Yr', '2 Yr', '5 Yr', '10 Yr', '15 Yr', '20 Yr', '30 Yr']
SG_TENOR_MONTHS = [6, 12, 24, 60, 120, 180, 240, 360]
SG_FIELDS = ['sgs_6m_bid_yield', 'sgs_1y_bid_yield', 'sgs_2y_bid_yield',
             'sgs_5y_bid_yield', 'sgs_10y_bid_yield', 'sgs_15y_bid_yield',
             'sgs_20y_bid_yield', 'sgs_30y_bid_yield']

# SORA resource
SORA_RESOURCE = '9a0bf149-308c-4bd2-832d-76c8e6cb47ed'
# SGS benchmark yields resource
SGS_RESOURCE = '5f2b18a8-0883-4769-a635-879c63d3caac'


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_sg_curve():
    """Fetch latest SGS benchmark yields from MAS API."""
    try:
        url = (f'https://eservices.mas.gov.sg/api/action/datastore/search.json'
               f'?resource_id={SGS_RESOURCE}&limit=30&sort=end_of_day desc')
        req = urllib.request.Request(url, headers={'User-Agent': _UA})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())

        records = data.get('result', {}).get('records', [])
        if not records:
            return None, None

        latest = records[0]
        latest_date = latest.get('end_of_day', '')

        yields = []
        for f in SG_FIELDS:
            val = latest.get(f)
            try:
                yields.append(float(val) if val else None)
            except (ValueError, TypeError):
                yields.append(None)

        prev_yields = None
        if len(records) >= 2:
            prev = records[1]
            prev_yields = []
            for f in SG_FIELDS:
                val = prev.get(f)
                try:
                    prev_yields.append(float(val) if val else None)
                except (ValueError, TypeError):
                    prev_yields.append(None)

        # Historical 2s10s
        hist_2s10s = []
        for r in records:
            y2 = r.get('sgs_2y_bid_yield')
            y10 = r.get('sgs_10y_bid_yield')
            try:
                if y2 and y10:
                    hist_2s10s.append({
                        'date': r.get('end_of_day', ''),
                        'spread': float(y10) - float(y2)
                    })
            except (ValueError, TypeError):
                pass

        return {
            'date': latest_date,
            'tenors': SG_TENORS,
            'months': SG_TENOR_MONTHS,
            'yields': yields,
            'prev_yields': prev_yields,
            'hist_2s10s': list(reversed(hist_2s10s)),
        }, None
    except Exception as e:
        logger.warning(f"SG curve fetch error: {e}")
        return None, str(e)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_sora():
    """Fetch latest SORA from MAS API."""
    try:
        url = (f'https://eservices.mas.gov.sg/api/action/datastore/search.json'
               f'?resource_id={SORA_RESOURCE}&limit=5&sort=end_of_day desc')
        req = urllib.request.Request(url, headers={'User-Agent': _UA})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        records = data.get('result', {}).get('records', [])
        if not records:
            return None
        r = records[0]
        return {
            'date': r.get('end_of_day', ''),
            'sora': r.get('sora', ''),
            'comp_1m': r.get('comp_sora_1m', ''),
            'comp_3m': r.get('comp_sora_3m', ''),
            'comp_6m': r.get('comp_sora_6m', ''),
        }
    except Exception:
        return None


# =============================================================================
# RENDER
# =============================================================================

def _render_curve_chart(us, sg, theme):
    """Overlay US and SG yield curves."""
    pos_c = theme['pos']; neg_c = theme['neg']
    _pbg = theme.get('plot_bg', '#0f1117'); _grd = theme.get('grid', '#1a1f2e')

    fig = go.Figure()

    # US curve
    if us:
        us_x = [m / 12 for m in us['months']]
        us_y = [y for y in us['yields']]
        fig.add_trace(go.Scatter(x=us_x, y=us_y, mode='lines+markers',
            name=f"US Treasury ({us['date']})", line=dict(color='#60a5fa', width=2.5),
            marker=dict(size=6), hovertemplate='%{text}<br>%{y:.2f}%<extra>US</extra>',
            text=us['tenors']))
        # Previous day
        if us.get('prev_yields'):
            fig.add_trace(go.Scatter(x=us_x, y=us['prev_yields'], mode='lines',
                name='US (prev day)', line=dict(color='#60a5fa', width=1, dash='dot'),
                hovertemplate='%{y:.2f}%<extra>US prev</extra>'))

    # SG curve
    if sg:
        sg_x = [m / 12 for m in sg['months']]
        sg_y = [y for y in sg['yields']]
        fig.add_trace(go.Scatter(x=sg_x, y=sg_y, mode='lines+markers',
            name=f"SG SGS ({sg['date']})", line=dict(color=pos_c, width=2.5),
            marker=dict(size=6), hovertemplate='%{text}<br>%{y:.2f}%<extra>SG</extra>',
            text=sg['tenors']))
        if sg.get('prev_yields'):
            fig.add_trace(go.Scatter(x=sg_x, y=sg['prev_yields'], mode='lines',
                name='SG (prev day)', line=dict(color=pos_c, width=1, dash='dot'),
                hovertemplate='%{y:.2f}%<extra>SG prev</extra>'))

    fig.update_layout(template='plotly_dark', height=380,
        margin=dict(l=40, r=20, t=30, b=40), plot_bgcolor=_pbg, paper_bgcolor=_pbg,
        showlegend=True, legend=dict(x=0.01, y=0.99, font=dict(size=10, family=FONTS)),
        hovermode='x unified', font=dict(family=FONTS),
        xaxis_title='Maturity (Years)', yaxis_title='Yield %')
    fig.update_xaxes(gridcolor=_grd, tickfont=dict(color='#888', size=9, family=FONTS),
                     tickvals=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    fig.update_yaxes(gridcolor=_grd, tickfont=dict(color='#888', size=9, family=FONTS), side='right')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def _render_spread_chart(us, sg, theme):
    """US-SG spread at matching tenors + 2s10s for both."""
    pos_c = theme['pos']; neg_c = theme['neg']
    _pbg = theme.get('plot_bg', '#0f1117'); _grd = theme.get('grid', '#1a1f2e')

    fig = make_subplots(rows=1, cols=2, subplot_titles=['US – SG Spread by Tenor', '2s10s Spread'],
                        horizontal_spacing=0.08)

    # Spread by tenor (matching tenors only)
    if us and sg:
        common = [(ut, um, uy, st_, sm, sy)
                  for ut, um, uy in zip(us['tenors'], us['months'], us['yields'])
                  for st_, sm, sy in zip(sg['tenors'], sg['months'], sg['yields'])
                  if um == sm and uy is not None and sy is not None]
        if common:
            labels = [c[0] for c in common]
            spreads = [c[2] - c[5] for c in common]
            colors = ['#60a5fa' if s >= 0 else neg_c for s in spreads]
            fig.add_trace(go.Bar(x=labels, y=spreads, marker_color=colors,
                hovertemplate='%{x}: %{y:.2f}%<extra>US-SG</extra>',
                showlegend=False), row=1, col=1)
            fig.add_hline(y=0, line=dict(color='#ffffff', width=0.5), row=1, col=1)

    # 2s10s
    if us:
        us_2y = us['yields'][6]; us_10y = us['yields'][10]
        if us_2y is not None and us_10y is not None:
            us_2s10s = us_10y - us_2y
            fig.add_trace(go.Bar(x=['US 2s10s'], y=[us_2s10s],
                marker_color='#60a5fa', name='US',
                hovertemplate='%{y:.2f}%<extra></extra>'), row=1, col=2)
    if sg:
        sg_2y = sg['yields'][2]; sg_10y = sg['yields'][4]
        if sg_2y is not None and sg_10y is not None:
            sg_2s10s = sg_10y - sg_2y
            fig.add_trace(go.Bar(x=['SG 2s10s'], y=[sg_2s10s],
                marker_color=pos_c, name='SG',
                hovertemplate='%{y:.2f}%<extra></extra>'), row=1, col=2)
    fig.add_hline(y=0, line=dict(color='#ffffff', width=0.5), row=1, col=2)

    fig.update_layout(template='plotly_dark', height=300,
        margin=dict(l=40, r=20, t=35, b=30), plot_bgcolor=_pbg, paper_bgcolor=_pbg,
        showlegend=True, legend=dict(x=0.85, y=0.99, font=dict(size=10, family=FONTS)),
        font=dict(family=FONTS))
    fig.update_xaxes(gridcolor=_grd, tickfont=dict(color='#888', size=9, family=FONTS))
    fig.update_yaxes(gridcolor=_grd, tickfont=dict(color='#888', size=9, family=FONTS), side='right')
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(size=10, family=FONTS, color='#94a3b8')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def _render_rates_table(us, sg, sora, theme):
    """Compact table showing key rates side by side."""
    pos_c = theme['pos']; neg_c = theme['neg']
    _bg3 = theme.get('bg3', '#0f172a'); _bdr = theme.get('border', '#1e293b')
    _txt = theme.get('text', '#e2e8f0'); _txt2 = theme.get('text2', '#94a3b8')
    _mut = theme.get('muted', '#475569')

    th = f"padding:4px 8px;border-bottom:1px solid {_bdr};color:#f8fafc;font-weight:600;font-size:9px;text-transform:uppercase;letter-spacing:0.05em;"
    td = f"padding:4px 8px;border-bottom:1px solid {_bdr}22;font-size:11px;"

    # Build matching tenor rows
    all_months = sorted(set((us['months'] if us else []) + (sg['months'] if sg else [])))
    us_map = dict(zip(us['months'], zip(us['tenors'], us['yields'], us.get('prev_yields') or [None]*20))) if us else {}
    sg_map = dict(zip(sg['months'], zip(sg['tenors'], sg['yields'], sg.get('prev_yields') or [None]*20))) if sg else {}

    html = f"""<div style='overflow-x:auto;border:1px solid {_bdr};border-radius:4px'>
    <table style='border-collapse:collapse;font-family:{FONTS};width:100%;line-height:1.3'>
    <thead style='background:{_bg3}'><tr>
        <th style='{th}text-align:left'>TENOR</th>
        <th style='{th}text-align:right;color:#60a5fa'>US YIELD</th>
        <th style='{th}text-align:right;color:#60a5fa'>Δ</th>
        <th style='{th}text-align:right;color:{pos_c}'>SG YIELD</th>
        <th style='{th}text-align:right;color:{pos_c}'>Δ</th>
        <th style='{th}text-align:right'>SPREAD</th>
    </tr></thead><tbody>"""

    for m in all_months:
        us_entry = us_map.get(m)
        sg_entry = sg_map.get(m)
        tenor = (us_entry[0] if us_entry else sg_entry[0]) if (us_entry or sg_entry) else f'{m}m'

        us_y = us_entry[1] if us_entry else None
        us_prev = us_entry[2] if us_entry else None
        sg_y = sg_entry[1] if sg_entry else None
        sg_prev = sg_entry[2] if sg_entry else None

        us_str = f'{us_y:.2f}%' if us_y is not None else '—'
        sg_str = f'{sg_y:.2f}%' if sg_y is not None else '—'

        us_d = ''
        if us_y is not None and us_prev is not None:
            d = us_y - us_prev
            c = pos_c if d <= 0 else neg_c
            us_d = f"<span style='color:{c}'>{d:+.2f}</span>"

        sg_d = ''
        if sg_y is not None and sg_prev is not None:
            d = sg_y - sg_prev
            c = pos_c if d <= 0 else neg_c
            sg_d = f"<span style='color:{c}'>{d:+.2f}</span>"

        spread_str = '—'
        if us_y is not None and sg_y is not None:
            sp = us_y - sg_y
            sp_c = '#60a5fa' if sp >= 0 else neg_c
            spread_str = f"<span style='color:{sp_c}'>{sp:+.2f}</span>"

        html += f"""<tr>
            <td style='{td}color:{_txt};font-weight:600'>{tenor}</td>
            <td style='{td}text-align:right;color:#60a5fa'>{us_str}</td>
            <td style='{td}text-align:right'>{us_d}</td>
            <td style='{td}text-align:right;color:{pos_c}'>{sg_str}</td>
            <td style='{td}text-align:right'>{sg_d}</td>
            <td style='{td}text-align:right;font-weight:600'>{spread_str}</td>
        </tr>"""

    html += "</tbody></table></div>"

    # SORA row below table
    if sora:
        sora_val = sora.get('sora', '—')
        c1m = sora.get('comp_1m', '—')
        c3m = sora.get('comp_3m', '—')
        c6m = sora.get('comp_6m', '—')
        html += (f"<div style='margin-top:6px;padding:5px 10px;background:{_bg3};border-radius:4px;"
                 f"font-family:{FONTS};font-size:10px;color:{_txt2}'>"
                 f"SORA ({sora['date']}): "
                 f"O/N <b style='color:{pos_c}'>{sora_val}%</b> · "
                 f"1M <b style='color:{pos_c}'>{c1m}%</b> · "
                 f"3M <b style='color:{pos_c}'>{c3m}%</b> · "
                 f"6M <b style='color:{pos_c}'>{c6m}%</b>"
                 f"</div>")

    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================

def render_rates_tab(is_mobile):
    t = get_theme()
    _mut = t.get('muted', '#475569')
    _lbl = f"color:#f8fafc;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"

    with st.spinner('Fetching rates...'):
        us, us_err = _fetch_us_curve()
        sg, sg_err = _fetch_sg_curve()
        sora = _fetch_sora()

    if not us and not sg:
        st.warning('Could not fetch rate data')
        return

    # Summary bar
    us_10y = us['yields'][10] if us and us['yields'][10] else None
    sg_10y = sg['yields'][4] if sg and sg['yields'][4] else None
    us_2y = us['yields'][6] if us and us['yields'][6] else None
    sg_2y = sg['yields'][2] if sg and sg['yields'][2] else None

    parts = []
    if us:
        parts.append(f"US curve {us['date']}")
    if sg:
        parts.append(f"SG curve {sg['date']}")
    if us_10y:
        parts.append(f"US 10Y <b style='color:#60a5fa'>{us_10y:.2f}%</b>")
    if sg_10y:
        parts.append(f"SG 10Y <b style='color:{t['pos']}'>{sg_10y:.2f}%</b>")
    if us_2y and us_10y:
        us_2s10s = us_10y - us_2y
        c = t['pos'] if us_2s10s >= 0 else t['neg']
        parts.append(f"US 2s10s <b style='color:{c}'>{us_2s10s:+.0f}bp</b>")
    if sg_2y and sg_10y:
        sg_2s10s = sg_10y - sg_2y
        c = t['pos'] if sg_2s10s >= 0 else t['neg']
        parts.append(f"SG 2s10s <b style='color:{c}'>{sg_2s10s:+.0f}bp</b>")

    _bg3 = t.get('bg3', '#0f172a')
    _txt2 = t.get('text2', '#94a3b8')
    st.markdown(f"""<div style='padding:5px 10px;background:{_bg3};font-family:{FONTS};
        font-size:10px;color:{_txt2};border-radius:4px;margin-bottom:6px'>
        {' &nbsp;·&nbsp; '.join(parts)}</div>""", unsafe_allow_html=True)

    # Layout
    if is_mobile:
        _render_curve_chart(us, sg, t)
        _render_spread_chart(us, sg, t)
        _render_rates_table(us, sg, sora, t)
    else:
        left, right = st.columns([6, 4])
        with left:
            tab_curve, tab_spread = st.tabs(['YIELD CURVES', 'SPREADS'])
            with tab_curve:
                _render_curve_chart(us, sg, t)
            with tab_spread:
                _render_spread_chart(us, sg, t)
        with right:
            st.markdown(f"<div style='{_lbl};margin-bottom:4px'>RATES TABLE</div>", unsafe_allow_html=True)
            _render_rates_table(us, sg, sora, t)
