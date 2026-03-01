"""
SANPO — Research tab
Displays published articles from Substack.
Uses st_html for pixel-perfect rendering matching other SANPO tabs.
"""

import streamlit as st
from streamlit.components.v1 import html as st_html
from config import get_theme, FONTS

# =============================================================================
# ARTICLES — add new posts at the TOP of this list
# =============================================================================

ARTICLES = [
    {
        "date": "01 Mar 2026",
        "title": "Operation Epic Fury: What Just Happened, Why, and What Comes Next",
        "subtitle": "The most consequential military action in the Middle East since 2003, explained.",
        "tags": ["Geopolitics", "Iran", "Middle East"],
        "url": "https://open.substack.com/pub/sanporesearch/p/operation-epic-fury-what-just-happened",
        "read_time": "8 min",
    },
    # ── TEMPLATE — copy, paste, fill ──────────────────────────────────────
    # {
    #     "date": "DD Mon YYYY",
    #     "title": "Your Title Here",
    #     "subtitle": "One-line description.",
    #     "tags": ["Tag1", "Tag2"],
    #     "url": "https://open.substack.com/pub/sanporesearch/p/slug",
    #     "read_time": "5 min",
    # },
]

# =============================================================================
# TAG COLOURS
# =============================================================================

TAG_COLORS = {
    "Geopolitics":    "#60a5fa",
    "Iran":           "#f59e0b",
    "Middle East":    "#f59e0b",
    "Macro":          "#60a5fa",
    "Gold":           "#fbbf24",
    "Oil":            "#f59e0b",
    "Crypto":         "#fbbf24",
    "Digital Assets": "#fbbf24",
    "Portfolio":      "#4ade80",
    "Quant":          "#c084fc",
    "Singapore":      "#4ade80",
    "Energy":         "#f59e0b",
    "Rates":          "#60a5fa",
    "FX":             "#38bdf8",
}

DEFAULT_TAG_COLOR = "#64748b"

# =============================================================================
# RENDER
# =============================================================================

def render():
    t = get_theme()
    bg   = t.get('bg', '#0f1117')
    bg2  = t.get('bg2', '#0a0f1a')
    bg3  = t.get('bg3', '#0f172a')
    bdr  = t.get('border', '#1e293b')
    txt  = t.get('text', '#e2e8f0')
    txt2 = t.get('text2', '#94a3b8')
    mut  = t.get('muted', '#475569')
    acc  = t.get('accent', '#4ade80')
    link = '#c9d1d9'

    # Build article rows
    rows_html = ""
    for i, art in enumerate(ARTICLES):
        # Tags as tiny dots + text
        tags_html = ""
        for tag in art.get("tags", []):
            tc = TAG_COLORS.get(tag, DEFAULT_TAG_COLOR)
            tags_html += (
                f"<span style='display:inline-flex;align-items:center;gap:4px;margin-right:10px'>"
                f"<span style='width:5px;height:5px;border-radius:50%;background:{tc};display:inline-block'></span>"
                f"<span style='font-size:9px;font-weight:600;letter-spacing:0.08em;"
                f"text-transform:uppercase;color:{tc}'>{tag}</span></span>"
            )

        row_bg = bg2 if i % 2 == 0 else bg3

        rows_html += f"""
        <a href="{art['url']}" target="_blank" style="text-decoration:none;display:block">
            <div class="row" style="padding:14px 16px;border-bottom:1px solid {bdr}20;background:{row_bg}">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px">
                    <div style="flex:1;min-width:0">
                        <div style="margin-bottom:5px">{tags_html}</div>
                        <div style="font-size:13px;font-weight:600;color:{txt};line-height:1.4;
                                    margin-bottom:3px">
                            {art['title']}
                        </div>
                        <div style="font-size:11px;color:{txt2};line-height:1.5;
                                    overflow:hidden;text-overflow:ellipsis;white-space:nowrap">
                            {art.get('subtitle', '')}
                        </div>
                    </div>
                    <div style="flex-shrink:0;text-align:right;padding-top:18px">
                        <div style="font-size:9px;color:{mut};white-space:nowrap">{art.get('date', '')}</div>
                        <div style="font-size:9px;color:{mut};margin-top:2px">{art.get('read_time', '')}</div>
                    </div>
                </div>
            </div>
        </a>"""

    n_articles = len(ARTICLES)
    count_text = f"{n_articles} article{'s' if n_articles != 1 else ''}" if ARTICLES else "No articles yet"

    body = f"""
    <div style="max-width:100%">
        <!-- Header bar -->
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:8px 16px;border-bottom:1px solid {bdr}">
            <div style="display:flex;align-items:center;gap:8px">
                <span style="font-size:10px;font-weight:600;letter-spacing:0.1em;
                             text-transform:uppercase;color:{acc}">Research</span>
                <span style="font-size:9px;color:{mut}">·</span>
                <span style="font-size:9px;color:{mut}">{count_text}</span>
            </div>
            <a href="https://sanporesearch.substack.com" target="_blank"
               style="font-size:9px;color:{mut};text-decoration:none;letter-spacing:0.06em;
                      text-transform:uppercase;font-weight:500">
                Substack ↗
            </a>
        </div>

        <!-- Article list -->
        <div style="border:1px solid {bdr};border-radius:4px;overflow:hidden;margin-top:8px">
            {rows_html if ARTICLES else f"<div style='padding:40px;text-align:center;color:{mut};font-size:11px'>No articles published yet.</div>"}
        </div>
    </div>
    """

    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap' rel='stylesheet'>"
        "<style>"
        "* { margin:0; padding:0; box-sizing:border-box; }"
        f"body {{ background:transparent; font-family:{FONTS}; color:{txt}; }}"
        f"a {{ color:{link}; text-decoration:none; }}"
        f".row:hover {{ background:{bdr} !important; }}"
        f"::-webkit-scrollbar {{ width:4px; }}"
        f"::-webkit-scrollbar-track {{ background:{bg2}; }}"
        f"::-webkit-scrollbar-thumb {{ background:{bdr}; border-radius:2px; }}"
        "</style></head><body>"
        f"{body}"
        "</body></html>"
    )

    # Height: header ~40px + per article ~75px + padding
    h = 60 + max(len(ARTICLES), 1) * 80
    st_html(page, height=h)
