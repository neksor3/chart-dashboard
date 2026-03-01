"""
SANPO — Research tab
Auto-pulls published articles from Substack RSS feed.
Rich card layout with excerpts, tag accents, and reading time.
No manual maintenance — publish on Substack, it appears here.
"""

import streamlit as st
from streamlit.components.v1 import html as st_html
import feedparser
import re
import logging
import urllib.request
from html import unescape as html_unescape
from config import get_theme, FONTS

logger = logging.getLogger(__name__)

SUBSTACK_URL = "https://sanporesearch.substack.com"
FEED_URL = f"{SUBSTACK_URL}/feed"
_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'

# =============================================================================
# TAG COLOURS — accent bar + dot color
# =============================================================================

TAG_COLORS = {
    "geopolitics":    "#60a5fa",
    "iran":           "#f59e0b",
    "middle east":    "#f59e0b",
    "macro":          "#60a5fa",
    "gold":           "#fbbf24",
    "oil":            "#f59e0b",
    "energy":         "#f59e0b",
    "crypto":         "#fbbf24",
    "bitcoin":        "#fbbf24",
    "digital assets": "#fbbf24",
    "portfolio":      "#4ade80",
    "quant":          "#c084fc",
    "singapore":      "#4ade80",
    "rates":          "#60a5fa",
    "fx":             "#38bdf8",
    "china":          "#ef4444",
    "trade":          "#60a5fa",
    "markets":        "#4ade80",
}

DEFAULT_TAG_COLOR = "#64748b"

def _get_tag_color(tag):
    return TAG_COLORS.get(tag.lower().strip(), DEFAULT_TAG_COLOR)

def _get_primary_color(tags):
    """Get accent color from first recognized tag."""
    for t in tags:
        c = TAG_COLORS.get(t.lower().strip())
        if c:
            return c
    return DEFAULT_TAG_COLOR

# =============================================================================
# RSS FETCH
# =============================================================================

def _clean_html(raw):
    if not raw:
        return ''
    t = re.sub(r'<[^>]+>', '', raw)
    return re.sub(r'\s+', ' ', html_unescape(t)).strip()

def _estimate_read_time(text):
    words = len(text.split())
    mins = max(1, round(words / 250))
    return f"{mins} min read"

@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_articles():
    try:
        req = urllib.request.Request(FEED_URL, headers={
            'User-Agent': _UA,
            'Accept': 'application/rss+xml, application/xml, text/xml, */*',
        })
        resp = urllib.request.urlopen(req, timeout=15)
        feed = feedparser.parse(resp.read())

        articles = []
        for entry in feed.entries[:20]:
            title = _clean_html(getattr(entry, 'title', ''))
            if not title:
                continue

            link = getattr(entry, 'link', '')
            pub = getattr(entry, 'published', getattr(entry, 'updated', ''))

            # Date
            date_str = ''
            try:
                from email.utils import parsedate_to_datetime
                dt = parsedate_to_datetime(pub)
                date_str = dt.strftime('%d %b %Y')
            except Exception:
                date_str = pub[:16] if pub else ''

            # Tags/categories
            tags = []
            for tag in getattr(entry, 'tags', []):
                t = getattr(tag, 'term', '')
                if t:
                    tags.append(t)

            # Subtitle from description, excerpt from content
            subtitle = _clean_html(getattr(entry, 'summary', ''))
            
            # Full content for excerpt
            content = ''
            if hasattr(entry, 'content'):
                for c in entry.content:
                    content += _clean_html(c.get('value', ''))
            if not content:
                content = subtitle

            # First 2-3 sentences as excerpt (skip if same as subtitle)
            excerpt = ''
            if content:
                sentences = re.split(r'(?<=[.!?])\s+', content)
                excerpt_text = ' '.join(sentences[:3])
                if len(excerpt_text) > 280:
                    excerpt_text = excerpt_text[:277] + '...'
                if excerpt_text != subtitle:
                    excerpt = excerpt_text

            read_time = _estimate_read_time(content) if content else '5 min read'

            articles.append({
                'title': title,
                'subtitle': subtitle,
                'excerpt': excerpt,
                'url': link,
                'date': date_str,
                'tags': tags,
                'read_time': read_time,
            })

        return articles
    except Exception as e:
        logger.warning(f"Research RSS error: {e}")
        return []

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

    articles = _fetch_articles()

    # Build cards
    cards_html = ""
    for i, art in enumerate(articles):
        tags = art.get('tags', [])
        accent = _get_primary_color(tags)

        # Tags row
        tags_html = ""
        for tag in tags:
            tc = _get_tag_color(tag)
            tags_html += (
                f"<span style='display:inline-flex;align-items:center;gap:4px;margin-right:12px'>"
                f"<span style='width:6px;height:6px;border-radius:50%;background:{tc}'></span>"
                f"<span style='font-size:9px;font-weight:600;letter-spacing:0.08em;"
                f"text-transform:uppercase;color:{tc}'>{tag}</span></span>"
            )

        # Excerpt
        excerpt_html = ""
        if art.get('excerpt'):
            excerpt_html = (
                f"<div style='font-size:11px;color:{txt2};line-height:1.6;"
                f"margin-top:8px;padding-top:8px;border-top:1px solid {bdr}30'>"
                f"{art['excerpt']}</div>"
            )

        cards_html += f"""
        <a href="{art['url']}" target="_blank" style="text-decoration:none;display:block">
            <div class="card" style="
                background:{bg3};
                border:1px solid {bdr};
                border-left:3px solid {accent};
                border-radius:6px;
                padding:16px 20px;
                margin-bottom:10px;
                transition:border-color 0.2s, background 0.2s;
            ">
                <!-- Top row: tags + meta -->
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <div>{tags_html}</div>
                    <div style="display:flex;align-items:center;gap:8px;flex-shrink:0">
                        <span style="font-size:9px;color:{mut}">{art.get('read_time', '')}</span>
                        <span style="font-size:9px;color:{mut}">·</span>
                        <span style="font-size:9px;color:{mut}">{art.get('date', '')}</span>
                    </div>
                </div>

                <!-- Title -->
                <div style="font-size:15px;font-weight:600;color:{txt};line-height:1.4">
                    {art['title']}
                </div>

                <!-- Subtitle -->
                <div style="font-size:11.5px;color:{txt2};margin-top:4px;line-height:1.5">
                    {art.get('subtitle', '')}
                </div>

                <!-- Excerpt -->
                {excerpt_html}

                <!-- Read link -->
                <div style="margin-top:10px;font-size:10px;font-weight:600;color:{accent};
                            letter-spacing:0.06em;text-transform:uppercase">
                    Read on Substack →
                </div>
            </div>
        </a>"""

    n = len(articles)
    count_text = f"{n} article{'s' if n != 1 else ''}" if articles else "No articles yet"

    # Empty state
    empty_html = f"""
        <div style="text-align:center;padding:60px 20px">
            <div style="font-size:36px;margin-bottom:16px;opacity:0.3">📡</div>
            <div style="font-size:13px;color:{txt2};margin-bottom:6px">No articles published yet</div>
            <div style="font-size:11px;color:{mut}">Published articles from Substack will appear here automatically</div>
        </div>
    """

    body = f"""
    <div style="max-width:100%">
        <!-- Header -->
        <div style="display:flex;justify-content:space-between;align-items:center;
                    padding:6px 4px 12px 4px;border-bottom:1px solid {bdr}40">
            <div style="display:flex;align-items:center;gap:10px">
                <span style="font-size:10px;font-weight:700;letter-spacing:0.12em;
                             text-transform:uppercase;color:{acc}">Research</span>
                <span style="width:4px;height:4px;border-radius:50%;background:{mut}"></span>
                <span style="font-size:9px;color:{mut}">{count_text}</span>
                <span style="width:4px;height:4px;border-radius:50%;background:{mut}"></span>
                <span style="font-size:9px;color:{mut}">Auto-updated via RSS</span>
            </div>
            <a href="{SUBSTACK_URL}" target="_blank"
               style="font-size:9px;color:{mut};text-decoration:none;letter-spacing:0.06em;
                      text-transform:uppercase;font-weight:600;
                      padding:4px 10px;border:1px solid {bdr};border-radius:3px;
                      transition:border-color 0.2s">
                Substack ↗
            </a>
        </div>

        <!-- Cards -->
        <div style="margin-top:12px">
            {cards_html if articles else empty_html}
        </div>
    </div>
    """

    page = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap' rel='stylesheet'>"
        "<style>"
        "* { margin:0; padding:0; box-sizing:border-box; }"
        f"body {{ background:transparent; font-family:{FONTS}; color:{txt}; }}"
        f"a {{ color:{txt}; text-decoration:none; }}"
        f".card:hover {{ border-color:{acc} !important; background:{bg2} !important; }}"
        f"::-webkit-scrollbar {{ width:4px; }}"
        f"::-webkit-scrollbar-track {{ background:{bg2}; }}"
        f"::-webkit-scrollbar-thumb {{ background:{bdr}; border-radius:2px; }}"
        "</style></head><body>"
        f"{body}"
        "</body></html>"
    )

    # Height: header ~50px + per card ~160px + padding
    h = 70 + max(len(articles), 1) * 170
    h = min(h, 900)  # cap height
    st_html(page, height=h)
