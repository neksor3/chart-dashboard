"""
SANPO — Research tab
Displays published articles from Substack with links, tags, and descriptions.
Easy to maintain: just add entries to ARTICLES list below.
"""

import streamlit as st
from config import get_theme, FONTS

# =============================================================================
# ARTICLES — add new posts at the TOP of this list
# =============================================================================

ARTICLES = [
    {
        "date": "2026-03-01",
        "title": "Operation Epic Fury: What Just Happened, Why, and What Comes Next",
        "subtitle": "The most consequential military action in the Middle East since 2003, explained.",
        "tags": ["Geopolitics", "Iran", "Middle East"],
        "url": "https://open.substack.com/pub/sanporesearch/p/operation-epic-fury-what-just-happened",
        "read_time": "8 min",
    },
    # ── TEMPLATE — copy this block for each new post ──────────────────────
    # {
    #     "date": "2026-XX-XX",
    #     "title": "Your Title Here",
    #     "subtitle": "One-line description.",
    #     "tags": ["Tag1", "Tag2"],
    #     "url": "https://open.substack.com/pub/sanporesearch/p/your-slug",
    #     "read_time": "5 min",
    # },
]

# =============================================================================
# TAG COLOURS
# =============================================================================

TAG_COLORS = {
    "Geopolitics": ("#60a5fa", "rgba(96,165,250,0.10)"),
    "Iran":        ("#f59e0b", "rgba(245,158,11,0.10)"),
    "Middle East": ("#f59e0b", "rgba(245,158,11,0.10)"),
    "Macro":       ("#60a5fa", "rgba(96,165,250,0.10)"),
    "Gold":        ("#fbbf24", "rgba(251,191,36,0.10)"),
    "Oil":         ("#f59e0b", "rgba(245,158,11,0.10)"),
    "Crypto":      ("#fbbf24", "rgba(251,191,36,0.10)"),
    "Digital Assets": ("#fbbf24", "rgba(251,191,36,0.10)"),
    "Portfolio":   ("#4ade80", "rgba(74,222,128,0.10)"),
    "Quant":       ("#c084fc", "rgba(192,132,252,0.10)"),
    "Singapore":   ("#4ade80", "rgba(74,222,128,0.10)"),
}

DEFAULT_TAG_COLOR = ("#94a3b8", "rgba(148,163,184,0.10)")

# =============================================================================
# RENDER
# =============================================================================

def render():
    theme = get_theme()
    _bg   = theme.get('bg', '#0f1117')
    _bg3  = theme.get('bg3', '#0f172a')
    _bdr  = theme.get('border', '#1e293b')
    _txt  = theme.get('text', '#e2e8f0')
    _txt2 = theme.get('text2', '#94a3b8')
    _mut  = theme.get('muted', '#475569')
    _acc  = theme.get('accent', '#4ade80')

    # ── header ───────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='margin-bottom:24px'>"
        f"<span style='font-family:{FONTS};font-size:11px;font-weight:600;"
        f"letter-spacing:0.12em;text-transform:uppercase;color:{_acc}'>"
        f"RESEARCH</span>"
        f"<div style='font-family:{FONTS};font-size:13px;color:{_txt2};"
        f"margin-top:4px'>Published on "
        f"<a href='https://sanporesearch.substack.com' target='_blank' "
        f"style='color:{_acc};text-decoration:none'>SANPO Research (Substack)</a>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    if not ARTICLES:
        st.info("No articles published yet.")
        return

    # ── article cards ────────────────────────────────────────────────────
    for art in ARTICLES:
        # tags html
        tags_html = ""
        for tag in art.get("tags", []):
            fg, bg = TAG_COLORS.get(tag, DEFAULT_TAG_COLOR)
            tags_html += (
                f"<span style='display:inline-block;font-family:{FONTS};"
                f"font-size:9px;font-weight:600;letter-spacing:0.1em;"
                f"text-transform:uppercase;color:{fg};background:{bg};"
                f"padding:3px 8px;border-radius:3px;margin-right:6px'>"
                f"{tag}</span>"
            )

        # meta line
        date_str = art.get("date", "")
        read_time = art.get("read_time", "")
        meta_parts = []
        if date_str:
            meta_parts.append(date_str)
        if read_time:
            meta_parts.append(read_time)
        meta_html = (
            f"<span style='font-family:{FONTS};font-size:11px;"
            f"color:{_mut}'>{' · '.join(meta_parts)}</span>"
        )

        # card
        card_html = (
            f"<a href='{art['url']}' target='_blank' "
            f"style='text-decoration:none;display:block'>"
            f"<div style='background:{_bg3};border:1px solid {_bdr};"
            f"border-radius:8px;padding:20px 24px;margin-bottom:12px;"
            f"transition:border-color 0.2s'>"
            f"<div style='margin-bottom:10px'>{tags_html}</div>"
            f"<div style='font-family:{FONTS};font-size:17px;font-weight:600;"
            f"color:{_txt};line-height:1.4;margin-bottom:6px'>"
            f"{art['title']}</div>"
            f"<div style='font-family:{FONTS};font-size:13px;color:{_txt2};"
            f"line-height:1.6;margin-bottom:10px'>"
            f"{art.get('subtitle', '')}</div>"
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:center'>"
            f"{meta_html}"
            f"<span style='font-family:{FONTS};font-size:11px;color:{_acc};"
            f"font-weight:500'>Read →</span>"
            f"</div>"
            f"</div></a>"
        )
        st.markdown(card_html, unsafe_allow_html=True)

    # ── footer ───────────────────────────────────────────────────────────
    st.markdown(
        f"<div style='text-align:center;margin-top:32px;padding-top:16px;"
        f"border-top:1px solid {_bdr}'>"
        f"<a href='https://sanporesearch.substack.com' target='_blank' "
        f"style='font-family:{FONTS};font-size:12px;color:{_mut};"
        f"text-decoration:none'>View all posts on Substack →</a></div>",
        unsafe_allow_html=True,
    )
