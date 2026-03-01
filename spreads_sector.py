import streamlit as st
from config import FUTURES_GROUPS, THEMES, SYMBOL_NAMES, FONTS, clean_symbol
from spreads import (LOOKBACK_OPTIONS, fetch_sector_spread_data,
                     compute_sector_spreads, sort_spread_pairs,
                     render_spread_table, render_spread_charts)


def render_sector_tab(is_mobile):
    theme_name = st.session_state.get('theme', 'Dark')
    theme = THEMES.get(theme_name, THEMES['Dark'])
    pos_c = theme['pos']
    _lbl = f"color:#e2e8f0;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"
    _bg3 = theme.get('bg3', '#0f172a'); _mut = theme.get('muted', '#475569'); _txt2 = theme.get('text2', '#94a3b8')
    ann_factor = 252

    # Controls
    if is_mobile:
        col_sec, col_lb, col_sort = st.columns([2, 1, 1])
    else:
        col_sec, col_lb, col_sort, col_dir = st.columns([3, 2, 2, 1])

    with col_sec:
        st.markdown(f"<div style='{_lbl}'>SECTOR</div>", unsafe_allow_html=True)
        sector_names = list(FUTURES_GROUPS.keys())
        spread_sector = st.selectbox("Sector", sector_names,
            index=sector_names.index(st.session_state.get('spread_sector', 'Futures')),
            key='spread_sector_sel', label_visibility='collapsed')
        st.session_state.spread_sector = spread_sector
    with col_lb:
        st.markdown(f"<div style='{_lbl}'>LOOKBACK</div>", unsafe_allow_html=True)
        lookback_label = st.selectbox("Lookback", list(LOOKBACK_OPTIONS.keys()), index=0,
            key='spread_lookback_sel', label_visibility='collapsed')
        lookback_days = LOOKBACK_OPTIONS[lookback_label]
    with col_sort:
        st.markdown(f"<div style='{_lbl}'>SORT BY</div>", unsafe_allow_html=True)
        sort_options = ['Composite', 'Sharpe', 'Sortino', 'MAR', 'R²', 'Total', 'Win Rate']
        sort_key = st.selectbox("Sort", sort_options, index=0,
            key='spread_sort_sel', label_visibility='collapsed')
    if not is_mobile:
        with col_dir:
            st.markdown(f"<div style='{_lbl}'>ORDER</div>", unsafe_allow_html=True)
            sort_dir = st.selectbox("Dir", ['Desc', 'Asc'], index=0,
                key='spread_dir_sel', label_visibility='collapsed')
            ascending = sort_dir == 'Asc'
    else:
        ascending = False

    # Fetch and compute
    with st.spinner(f'Computing {spread_sector} spreads ({lookback_label})...'):
        data = fetch_sector_spread_data(spread_sector, lookback_days)

    if data is None or len(data.columns) < 2:
        st.markdown(f"<div style='padding:12px;color:{_mut};font-size:11px;font-family:{FONTS}'>Need at least 2 assets with data for spread analysis</div>", unsafe_allow_html=True)
        return

    pairs = compute_sector_spreads(data, ann_factor)
    if not pairs:
        st.markdown(f"<div style='padding:12px;color:{_mut};font-size:11px;font-family:{FONTS}'>No spreads computed</div>", unsafe_allow_html=True)
        return

    sorted_pairs = sort_spread_pairs(pairs, sort_key, ascending)

    # Info bar
    best_long_sym = pairs[0].get('best_long_sym', '')
    best_long_sharpe = pairs[0].get('best_long_sharpe', 0)
    best_long_name = SYMBOL_NAMES.get(best_long_sym, clean_symbol(best_long_sym))
    n_combos = len(pairs)
    n_beats = sum(1 for p in pairs if p['beats_long'])
    start_date = data.index[0].strftime('%d %b')
    end_date = data.index[-1].strftime('%d %b %Y')

    beats_c = pos_c if n_beats > 0 else _mut
    st.markdown(f"""
        <div style='padding:5px 10px;background-color:{_bg3};font-family:{FONTS};display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;border-radius:4px'>
            <span style='color:{_mut};font-size:10px'>{n_combos} pairs · {start_date} → {end_date}</span>
            <span style='color:{_txt2};font-size:10px'>
                Best long: <span style='color:{pos_c};font-weight:600'>{best_long_name}</span>
                <span style='color:{_mut}'>Sharpe {best_long_sharpe:.2f}</span>
                &nbsp;·&nbsp;
                <span style='color:{beats_c}'>{n_beats} spread{"s" if n_beats != 1 else ""} beat{"s" if n_beats == 1 else ""} it</span>
            </span>
        </div>""", unsafe_allow_html=True)

    # Charts (top 6)
    render_spread_charts(sorted_pairs, data, theme, mobile=is_mobile)

    # Table (top 10)
    render_spread_table(sorted_pairs, theme, top_n=10)
