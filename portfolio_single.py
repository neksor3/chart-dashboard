import streamlit as st
from collections import OrderedDict
from config import FUTURES_GROUPS, THEMES, FONTS
from portfolio import (C_POS, C_NEG, C_MUTE, C_BG, C_TXT, C_TXT2, C_GOLD,
                       REBAL_OPTIONS, PERIOD_OPTIONS, SCORE_TO_RANK,
                       run_walkforward_grid, render_ranking_table,
                       render_weights_table, render_oos_chart,
                       render_monthly_table, _section)


def render_single_tab(is_mobile):
    import portfolio
    theme_name = st.session_state.get('theme', 'Dark')
    theme = THEMES.get(theme_name, THEMES['Dark'])
    portfolio.C_POS = theme['pos']; portfolio.C_NEG = theme['neg']
    _lbl = f"color:#f8fafc;font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;font-family:{FONTS}"

    # Consistent input styling
    st.markdown(f"""<style>
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div[aria-selected] {{
            font-family: {FONTS} !important;
            font-size: 13px !important;
            letter-spacing: 0.01em !important;
        }}
        div[data-baseweb="menu"] li,
        ul[role="listbox"] li {{
            font-family: {FONTS} !important;
            font-size: 13px !important;
        }}
        .stTextInput input {{
            font-family: {FONTS} !important;
            font-size: 13px !important;
            letter-spacing: 0.01em !important;
        }}
        .stTextInput input::placeholder {{
            font-family: {FONTS} !important;
            font-size: 13px !important;
        }}
    </style>""", unsafe_allow_html=True)

    # Portfolio selector
    group_names = ['Custom'] + list(FUTURES_GROUPS.keys())
    if 'port_preset_name' not in st.session_state:
        st.session_state.port_preset_name = 'Custom'
    if 'port_sym_input' not in st.session_state:
        st.session_state.port_sym_input = ''

    def _on_portfolio_change():
        sel = st.session_state.port_selector
        if sel != 'Custom':
            syms = FUTURES_GROUPS.get(sel, [])
            st.session_state.port_sym_input = ', '.join(syms)
        st.session_state.port_preset_name = sel

    p1, p2 = st.columns([1, 4])
    with p1:
        st.markdown(f"<div style='{_lbl}'>PORTFOLIO</div>", unsafe_allow_html=True)
        current_idx = group_names.index(st.session_state.port_preset_name) if st.session_state.port_preset_name in group_names else 0
        st.selectbox("Portfolio", group_names, index=current_idx,
                     key='port_selector', label_visibility='collapsed', on_change=_on_portfolio_change)
    with p2:
        st.markdown(f"<div style='{_lbl}'>SYMBOLS</div>", unsafe_allow_html=True)
        sym_input = st.text_input("Symbols", key='port_sym_input', label_visibility='collapsed',
                                   placeholder='AAPL, MSFT, GOOG, ...')

    # Row 1: Objective, Rebalance, Period, Direction, Sims
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"<div style='{_lbl}'>OBJECTIVE</div>", unsafe_allow_html=True)
        score = st.selectbox("Objective", ['Win Rate', 'Composite', 'Sharpe', 'Sortino', 'MAR', 'R²', 'Total Return'],
                              key='port_score', label_visibility='collapsed')
    with c2:
        st.markdown(f"<div style='{_lbl}'>REBALANCE</div>", unsafe_allow_html=True)
        rebal_label = st.selectbox("Rebalance", list(REBAL_OPTIONS.keys()),
                                    index=2, key='port_rebal', label_visibility='collapsed')
    with c3:
        st.markdown(f"<div style='{_lbl}'>PERIOD</div>", unsafe_allow_html=True)
        period_label = st.selectbox("Period", list(PERIOD_OPTIONS.keys()),
                                     index=3, key='port_period', label_visibility='collapsed')
    with c4:
        st.markdown(f"<div style='{_lbl}'>DIRECTION</div>", unsafe_allow_html=True)
        direction = st.selectbox("Direction", ['Long Only', 'Long/Short'],
                                  key='port_direction', label_visibility='collapsed')
    with c5:
        st.markdown(f"<div style='{_lbl}'>SIMS</div>", unsafe_allow_html=True)
        _defaults = {'port_sims': '10000'}
        for k, v in _defaults.items():
            if k not in st.session_state: st.session_state[k] = v
        if not st.session_state.get('port_sims'): st.session_state['port_sims'] = '10000'
        sims_str = st.text_input("Sims", key='port_sims', label_visibility='collapsed')

    # Row 2: Max Wt, Min Wt, Max Vol, Min Ret, Cost
    _defaults2 = {'port_maxwt': '50', 'port_minwt': '0', 'port_cost': '0.10',
                   'port_maxvol': '', 'port_minret': ''}
    for k, v in _defaults2.items():
        if k not in st.session_state: st.session_state[k] = v
    for k, v in [('port_maxwt','50'),('port_minwt','0'),('port_cost','0.10')]:
        if not st.session_state.get(k): st.session_state[k] = v

    c6, c7, c8, c9, c10 = st.columns(5)
    with c6:
        st.markdown(f"<div style='{_lbl}'>MAX WT %</div>", unsafe_allow_html=True)
        max_wt_str = st.text_input("Max Wt", key='port_maxwt', label_visibility='collapsed')
    with c7:
        st.markdown(f"<div style='{_lbl}'>MIN WT %</div>", unsafe_allow_html=True)
        min_wt_str = st.text_input("Min Wt", key='port_minwt', label_visibility='collapsed')
    with c8:
        st.markdown(f"<div style='{_lbl}'>MAX VOL %</div>", unsafe_allow_html=True)
        max_vol_str = st.text_input("Max Vol", key='port_maxvol', label_visibility='collapsed',
                                     placeholder='e.g. 15')
    with c9:
        st.markdown(f"<div style='{_lbl}'>MIN RET %</div>", unsafe_allow_html=True)
        min_ret_str = st.text_input("Min Ret", key='port_minret', label_visibility='collapsed',
                                     placeholder='e.g. 5')
    with c10:
        st.markdown(f"<div style='{_lbl}'>COST %</div>", unsafe_allow_html=True)
        cost_str = st.text_input("Cost", key='port_cost', label_visibility='collapsed')

    # Optimize button
    run_clicked = st.button('▶  Optimize', key='port_run', type='primary')

    if not run_clicked and 'port_grid' not in st.session_state:
        st.markdown(f"<div style='padding:20px;color:{C_MUTE};font-size:11px;font-family:{FONTS}'>Configure parameters and click OPTIMIZE to start walk-forward optimization</div>", unsafe_allow_html=True)
        return

    if run_clicked:
        raw = sym_input.strip()
        if not raw:
            st.warning('Enter symbols to optimize'); return
        symbols = [s.strip().upper() for s in raw.replace(';', ',').split(',') if s.strip()]
        symbols = list(dict.fromkeys(symbols))

        try: max_wt = max(10, min(100, float(max_wt_str))) / 100.0
        except (ValueError, TypeError): max_wt = 0.50
        try: min_wt = max(0, min(50, float(min_wt_str))) / 100.0
        except (ValueError, TypeError): min_wt = 0.0
        try: n_sims = max(1000, min(100000, int(sims_str)))
        except (ValueError, TypeError): n_sims = 10000
        try: txn_cost = max(0, min(5.0, float(cost_str))) / 100.0
        except (ValueError, TypeError): txn_cost = 0.001
        try: max_vol = float(max_vol_str) / 100.0 if max_vol_str.strip() else None
        except (ValueError, TypeError): max_vol = None
        try: min_ann_ret = float(min_ret_str) / 100.0 if min_ret_str.strip() else None
        except (ValueError, TypeError): min_ann_ret = None

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
                                     progress_bar=progress,
                                     max_vol=max_vol, min_ann_ret=min_ann_ret)
        progress.empty()

        if not grid or not grid['results']:
            st.warning('Need ≥2 assets with sufficient history for walk-forward'); return

        st.session_state.port_grid = grid
        preset_name = st.session_state.get('port_preset_name', 'Custom')
        if preset_name == 'Custom' or not preset_name:
            sym_set = set(symbols)
            for pname, psyms in FUTURES_GROUPS.items():
                if set(psyms) == sym_set:
                    preset_name = pname; break
            else:
                preset_name = 'Portfolio'
        st.session_state.port_params = {
            'score': score, 'rebal_label': rebal_label, 'period_label': period_label,
            'direction': 'L/S' if allow_short else 'Long',
            'min_wt': min_wt, 'max_wt': max_wt, 'n_sims': n_sims, 'txn_cost': txn_cost,
            'max_vol': max_vol, 'min_ann_ret': min_ann_ret,
            'preset_name': preset_name,
        }
        if 'port_view_approach' in st.session_state:
            del st.session_state.port_view_approach

    # Display results
    if 'port_grid' not in st.session_state: return
    grid = st.session_state.port_grid
    params = st.session_state.port_params
    rank_metric = SCORE_TO_RANK.get(params['score'], 'win_rate')
    n_app = len(grid['results'])

    constraints_str = ''
    if params.get('max_vol'): constraints_str += f" · max vol {params['max_vol']*100:.0f}%"
    if params.get('min_ann_ret'): constraints_str += f" · min ret {params['min_ann_ret']*100:.0f}%"
    _section('APPROACH RANKING',
             f"{n_app} lookbacks · {params['rebal_label']} · {params['period_label']} · "
             f"{params['direction']} · wt {params['min_wt']*100:.0f}–{params['max_wt']*100:.0f}% · "
             f"cost {params['txn_cost']*100:.2f}% · {params['n_sims']:,} sims · max {params['score']}{constraints_str} · all OOS")
    result = render_ranking_table(grid, rank_metric)
    best_name, sorted_names = result
    if not best_name or not sorted_names: return

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

    st.markdown(f"""<div style='margin-top:12px;padding:5px 10px;background:{C_BG};font-family:{FONTS};border-radius:4px;
        font-size:10px;color:{C_TXT2};display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px'>
        <span>{star}<b style='color:{C_TXT}'>{selected_approach}</b>
        &nbsp;·&nbsp;{sm['n_days']} OOS days · {sm['oos_years']}y · {sm['n_rebalances']} rebalances</span>
        <span>Win% <b style='color:{portfolio.C_POS}'>{sm["win_rate"]*100:.1f}%</b>
        &nbsp;Sharpe <b style='color:{portfolio.C_POS}'>{sm["sharpe"]:.2f}</b>
        &nbsp;Sortino <b style='color:{portfolio.C_POS}'>{sm["sortino"]:.2f}</b>
        &nbsp;MAR <b style='color:{portfolio.C_POS}'>{sm["mar"]:.2f}</b></span>
    </div>""", unsafe_allow_html=True)

    _section('OOS EQUITY CURVE', f'{selected_approach} · {params["rebal_label"]} · yellow = rebalance dates')
    render_oos_chart(grid, selected_approach)

    _section('OOS MONTHLY RETURNS', f'{selected_approach} · walk-forward out-of-sample only')
    render_monthly_table(swf['oos_returns'])

    _section('CURRENT / NEXT WEIGHTS', f'{selected_approach} · optimized on all data through today · trade these')
    render_weights_table(grid, selected_approach)
