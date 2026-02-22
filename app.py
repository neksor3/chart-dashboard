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
    if 'sector' not in st.session_state: st.session_state.sector = 'Futures'
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
    title_c = '#f8fafc'

    st.markdown(f"""
        <style>
            @keyframes sanpo-sweep {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
            @keyframes sanpo-glow {{ 0%,100% {{ filter: drop-shadow(0 0 3px {pos_c}40); }} 50% {{ filter: drop-shadow(0 0 8px {pos_c}90); }} }}
            @keyframes sanpo-core {{ 0%,100% {{ r: 2.5; }} 6% {{ r: 3.5; }} 18% {{ r: 2.6; }} }}
            @keyframes sanpo-halo {{ 0%,80%,100% {{ opacity: 0.0; }} 6% {{ opacity: 0.30; }} 20% {{ opacity: 0.10; }} 40% {{ opacity: 0.03; }} }}
            @keyframes sanpo-ring {{ 0%,80%,100% {{ opacity: 0.06; }} 8% {{ opacity: 0.50; }} 20% {{ opacity: 0.20; }} 40% {{ opacity: 0.10; }} 60% {{ opacity: 0.06; }} }}
            @keyframes sanpo-dot-sm {{ 0%,80%,100% {{ r: 0.6; opacity: 0; }} 8% {{ r: 1.3; opacity: 0.75; }} 15% {{ r: 1.0; opacity: 0.40; }} 30% {{ r: 0.8; opacity: 0.10; }} 50% {{ r: 0.7; opacity: 0.08; }} 75% {{ r: 0.6; opacity: 0.04; }} }}
            @keyframes sanpo-dot-md {{ 0%,80%,100% {{ r: 0.9; opacity: 0; }} 8% {{ r: 1.8; opacity: 0.80; }} 15% {{ r: 1.5; opacity: 0.45; }} 30% {{ r: 1.2; opacity: 0.12; }} 50% {{ r: 1.0; opacity: 0.08; }} 75% {{ r: 0.9; opacity: 0.04; }} }}
            @keyframes sanpo-dot-lg {{ 0%,80%,100% {{ r: 1.2; opacity: 0; }} 8% {{ r: 2.5; opacity: 0.85; }} 15% {{ r: 2.1; opacity: 0.50; }} 30% {{ r: 1.7; opacity: 0.15; }} 50% {{ r: 1.4; opacity: 0.10; }} 75% {{ r: 1.2; opacity: 0.04; }} }}
        </style>
        <div style='display:flex;align-items:center;gap:14px;padding:6px 0'>
            <svg width="56" height="56" viewBox="0 0 40 40" fill="none" style="animation:sanpo-glow 3s ease-in-out infinite">
                <!-- Rings: pulse outward from core -->
                <circle cx="20" cy="20" r="6"  stroke="#334155" style="animation:sanpo-ring 8s ease-out infinite 0.5s"/>
                <circle cx="20" cy="20" r="12" stroke="#334155" style="animation:sanpo-ring 8s ease-out infinite 1.2s"/>
                <circle cx="20" cy="20" r="18" stroke="#334155" style="animation:sanpo-ring 8s ease-out infinite 2.0s"/>
                <!-- Core: heartbeat pump -->
                <circle cx="20" cy="20" r="4.5" fill="{pos_c}" style="animation:sanpo-halo 8s ease-out infinite 0.0s"/>
                <circle cx="20" cy="20" fill="{pos_c}" style="animation:sanpo-core 8s ease-out infinite 0.0s"/>
                <!-- Sweep line -->
                <line x1="20" y1="20" x2="20" y2="2" stroke="url(#sanpoSweepG)" stroke-width="1.2" stroke-linecap="round" style="animation:sanpo-sweep 4s linear infinite;transform-origin:20px 20px"/>
                <!-- 6 dots: invisible → bloom → linger decay. Tiered by ring. -->
                <!-- Inner ring r≈6 -->
                <circle cx="23.0" cy="14.8" fill="{pos_c}" style="animation:sanpo-dot-sm 8s ease-out infinite 0.7s"/>
                <circle cx="17.0" cy="25.2" fill="{neg_c}" style="animation:sanpo-dot-sm 8s ease-out infinite 0.9s"/>
                <!-- Mid ring r≈12 -->
                <circle cx="31.3" cy="24.1" fill="{pos_c}" style="animation:sanpo-dot-md 8s ease-out infinite 1.5s"/>
                <circle cx="8.7"  cy="15.9" fill="{neg_c}" style="animation:sanpo-dot-md 8s ease-out infinite 1.8s"/>
                <!-- Outer ring r≈18 -->
                <circle cx="35.6" cy="11.0" fill="{pos_c}" style="animation:sanpo-dot-lg 8s ease-out infinite 2.5s"/>
                <circle cx="4.4"  cy="29.0" fill="{neg_c}" style="animation:sanpo-dot-lg 8s ease-out infinite 2.8s"/>
                <defs><linearGradient id="sanpoSweepG" x1="20" y1="20" x2="20" y2="3">
                    <stop offset="0%" stop-color="{pos_c}" stop-opacity="0.6"/>
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

    # Global auto-refresh aligned to :00 :15 :30 :45
    from streamlit.components.v1 import html as st_html
    st_html("""<script>
    (function(){
        var now=new Date(), m=now.getMinutes(), s=now.getSeconds(), ms=now.getMilliseconds();
        var next15=15-m%15;
        var delay=(next15*60-s)*1000-ms;
        if(delay<5000) delay+=900000;
        setTimeout(function(){window.parent.location.reload()}, delay);
    })();
    </script>""", height=0)


if __name__ == "__main__":
    main()
