import streamlit as st
import pytz
import re
import logging
import warnings

from config import FONTS, THEMES

# =============================================================================
# SETUP
# =============================================================================

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

st.set_page_config(page_title="SANPO", layout="wide", initial_sidebar_state="collapsed")


def get_theme():
    name = st.session_state.get('theme', 'Dark')
    return THEMES.get(name, THEMES['Dark'])


def _inject_theme_css():
    t = get_theme()
    is_light = t.get('mode') == 'light'
    bg = t.get('bg', '#0f1117'); bg2 = t.get('bg2', '#0a0f1a'); bg3 = t.get('bg3', '#0f172a'); bdr = t.get('border', '#1e293b')
    txt = t.get('text', '#e2e8f0'); txt2 = t.get('text2', '#94a3b8'); muted = t.get('muted', '#475569'); accent = t.get('accent', '#4ade80')
    sb_bg = '#f1f5f9' if is_light else '#1a2744'
    sel_bg = '#f1f5f9' if is_light else '#1a2744'
    sel_c = '#334155' if is_light else '#b0b0b0'
    sel_bdr = '#e2e8f0' if is_light else '#1e293b'
    tab_bdr = bdr
    tab_c = '#64748b' if is_light else '#7a8a9e'
    tab_sel_c = '#0f172a' if is_light else '#f1f5f9'
    radio_bg = bg3 if is_light else '#1a2744'
    radio_bdr = bdr
    btn_bg = bg3 if is_light else '#1a2744'
    btn_c = txt
    st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@500;700&display=swap');
    .stApp {{ background-color: {bg}; font-family: 'Inter', sans-serif; }}
    header[data-testid="stHeader"] {{ background-color: {bg}; }}
    [data-testid="stSidebar"] {{ background-color: {sb_bg}; }}
    .stSelectbox > div > div {{ background-color: {sel_bg}; color: {sel_c}; font-family: 'Inter', sans-serif; border: 1px solid {sel_bdr}; }}
    .stTextInput > div > div > input {{ font-family: 'Inter', sans-serif; }}
    div[data-testid="stHorizontalBlock"] {{ gap: 0.3rem; }}
    .stTabs [data-baseweb="tab-list"] {{ gap: 0; background-color: transparent; padding: 0; border-radius: 0; border-bottom: 1px solid {tab_bdr}; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent; color: {tab_c}; border: none;
        border-bottom: 2px solid transparent;
        padding: 8px 20px; font-size: 12px; font-weight: 600;
        letter-spacing: 0.1em; text-transform: uppercase;
        font-family: 'Inter', sans-serif;
    }}
    .stTabs [aria-selected="true"] {{ background-color: transparent; color: {tab_sel_c}; border-bottom: 2px solid {accent}; font-weight: 700; }}
    .stRadio > div {{ flex-direction: row; gap: 8px; }}
    .stRadio > div > label {{ background-color: {radio_bg}; padding: 4px 12px; border-radius: 3px;
        border: 1px solid {radio_bdr}; color: {sel_c}; font-size: 12px; }}
    div[data-testid="stMarkdownContainer"] p {{ margin-bottom: 0; }}
    .block-container {{ padding-top: 2.5rem; padding-bottom: 0rem; }}
    button[kind="secondary"] {{ background-color: {btn_bg}; color: {btn_c}; border: 1px solid {tab_bdr}; font-family: 'Inter', sans-serif; }}
    .stButton > button {{ font-size: 11px !important; padding: 4px 8px !important; min-height: 30px !important; font-family: 'Inter', sans-serif !important; }}
    @media (max-width: 768px) {{
        .block-container {{ padding: 2.5rem 0.5rem 0 0.5rem !important; }}
        .stButton > button {{ font-size: 9px !important; padding: 2px 4px !important; min-height: 24px !important; }}
    }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    [data-testid="stStatusWidget"] {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)


def _detect_mobile():
    try:
        ua = st.context.headers.get('User-Agent', '')
        return bool(re.search(r'iPhone|Android.*Mobile|Windows Phone', ua, re.I))
    except Exception:
        return False


def main():
    from pulse import render_pulse_tab
    from charts import render_charts_tab
    from spreads import render_spreads_tab
    from portfolio import render_portfolio_tab
    from news import render_news_tab

    # Init session state
    if 'sector' not in st.session_state: st.session_state.sector = 'Indices'
    if 'symbol' not in st.session_state: st.session_state.symbol = 'ES=F'
    if 'chart_type' not in st.session_state: st.session_state.chart_type = 'line'

    st.session_state.theme = 'Dark'

    is_mobile = _detect_mobile()
    est = pytz.timezone('US/Eastern')

    _inject_theme_css()

    # SANPO logo header
    t = get_theme()
    pos_c = t['pos']
    neg_c = t['neg']
    ring_c = '#1e293b'
    title_c = '#f8fafc'

    st.markdown(f"""
        <style>
            @keyframes sanpo-sweep {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
            @keyframes sanpo-blink {{ 0%,100% {{ opacity: 0.9; }} 50% {{ opacity: 0.1; }} }}
            @keyframes sanpo-glow {{ 0%,100% {{ filter: drop-shadow(0 0 3px {pos_c}40); }} 50% {{ filter: drop-shadow(0 0 8px {pos_c}90); }} }}
        </style>
        <div style='display:flex;align-items:center;gap:14px;padding:6px 0'>
            <svg width="44" height="44" viewBox="0 0 40 40" fill="none" style="animation:sanpo-glow 3s ease-in-out infinite">
                <circle cx="20" cy="20" r="18" stroke="{ring_c}" stroke-width="0.8"/>
                <circle cx="20" cy="20" r="12.5" stroke="{ring_c}" stroke-width="0.6"/>
                <circle cx="20" cy="20" r="7" stroke="{ring_c}" stroke-width="0.5"/>
                <circle cx="20" cy="20" r="3" fill="{pos_c}"/>
                <circle cx="20" cy="20" r="5" fill="{pos_c}" opacity="0.15"/>
                <line x1="20" y1="20" x2="20" y2="3" stroke="url(#sanpoSweepG)" stroke-width="1.5" stroke-linecap="round" style="animation:sanpo-sweep 4s linear infinite;transform-origin:20px 20px"/>
                <circle cx="13" cy="9" r="2.2" fill="{pos_c}" style="animation:sanpo-blink 1.8s ease-in-out infinite"/>
                <circle cx="30" cy="13" r="2" fill="{neg_c}" style="animation:sanpo-blink 2.2s ease-in-out infinite 0.4s"/>
                <circle cx="28" cy="29" r="1.8" fill="{pos_c}" style="animation:sanpo-blink 2s ease-in-out infinite 0.9s"/>
                <circle cx="9" cy="25" r="1.7" fill="{neg_c}" style="animation:sanpo-blink 1.6s ease-in-out infinite 1.3s"/>
                <circle cx="25" cy="8" r="1.5" fill="{pos_c}" style="animation:sanpo-blink 2.4s ease-in-out infinite 0.6s"/>
                <circle cx="10" cy="15" r="1.3" fill="{pos_c}" style="animation:sanpo-blink 2.0s ease-in-out infinite 1.7s"/>
                <defs><linearGradient id="sanpoSweepG" x1="20" y1="20" x2="20" y2="3">
                    <stop offset="0%" stop-color="{pos_c}" stop-opacity="0.7"/>
                    <stop offset="100%" stop-color="{pos_c}" stop-opacity="0"/>
                </linearGradient></defs>
            </svg>
            <span style='font-family:Orbitron,sans-serif;font-size:24px;font-weight:700;letter-spacing:0.08em;color:{title_c};line-height:1'>SANPO</span>
        </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab_pulse, tab_charts, tab_spreads, tab_portfolio, tab_news = st.tabs(["PULSE", "CHARTS", "SPREADS", "PORTFOLIO", "NEWS"])

    with tab_pulse:
        render_pulse_tab(is_mobile)

    with tab_charts:
        render_charts_tab(is_mobile, est)

    with tab_spreads:
        render_spreads_tab(is_mobile)

    with tab_portfolio:
        render_portfolio_tab(is_mobile)

    with tab_news:
        render_news_tab(is_mobile)


if __name__ == "__main__":
    main()
