import streamlit as st
import feedparser
import logging
import re
from html import escape as html_escape, unescape as html_unescape
from config import FONTS, THEMES

logger = logging.getLogger(__name__)

def get_theme():
    tn = st.session_state.get('theme', 'Emerald / Amber')
    return THEMES.get(tn, THEMES['Emerald / Amber'])

# =============================================================================
# RSS FEED SOURCES (verified working 19 Feb 2026)
# =============================================================================

NEWS_FEEDS = {
    'Singapore': [
        ('CNA', 'https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6511'),
        ('Straits Times', 'https://www.straitstimes.com/news/business/rss.xml'),
        ('Business Times', 'https://www.businesstimes.com.sg/rss/top-stories'),
    ],
    'Regional': [
        ('SCMP', 'https://www.scmp.com/rss/5/feed'),
        ('Malay Mail', 'https://www.malaymail.com/feed/rss/money'),
        ('The Star', 'https://www.thestar.com.my/rss/Business'),
    ],
    'Global': [
        ('Bloomberg', 'https://feeds.bloomberg.com/markets/news.rss'),
        ('FT', 'https://www.ft.com/rss/home'),
        ('CNBC', 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258'),
        ('BBC Business', 'https://feeds.bbci.co.uk/news/business/rss.xml'),
    ],
    'Tech': [
        ('TechCrunch AI', 'https://techcrunch.com/category/artificial-intelligence/feed/'),
        ('The Verge', 'https://www.theverge.com/rss/index.xml'),
        ('Ars Technica', 'https://feeds.arstechnica.com/arstechnica/technology-lab'),
        ('CoinDesk', 'https://www.coindesk.com/arc/outboundfeeds/rss/'),
        ('STAT News', 'https://www.statnews.com/feed/'),
        ('Endpoints', 'https://endpts.com/feed/'),
        ('Longevity Tech', 'https://www.longevity.technology/feed/'),
    ],
}

# =============================================================================
# FETCH
# =============================================================================

def _clean_text(raw):
    """Strip HTML tags, decode entities, clean whitespace."""
    if not raw:
        return ''
    text = re.sub(r'<[^>]+>', '', raw)       # strip tags
    text = html_unescape(text)               # decode &amp; etc
    text = re.sub(r'\s+', ' ', text).strip() # collapse whitespace
    return text

@st.cache_data(ttl=300, show_spinner=False)
def fetch_rss_feed(name, url):
    """Fetch and parse a single RSS feed. Returns empty list on failure."""
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:15]:
            title = _clean_text(getattr(entry, 'title', ''))
            link = getattr(entry, 'link', '')
            pub = getattr(entry, 'published', getattr(entry, 'updated', ''))

            # Clean summary
            summary_raw = getattr(entry, 'summary', getattr(entry, 'description', ''))
            summary = _clean_text(summary_raw)
            if len(summary) > 120:
                summary = summary[:120].rsplit(' ', 1)[0] + '…'

            # Skip junk (press releases, raw HTML noise)
            if title and any(x in title.lower() for x in ['shareholders are encouraged', 'announces partnership', '<img']):
                continue

            # Parse date
            date_str = ''
            if pub:
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(pub)
                    date_str = dt.strftime('%d %b %H:%M')
                except Exception:
                    # Handle ISO format dates
                    try:
                        clean = pub.replace('T', ' ')[:16]
                        date_str = clean
                    except Exception:
                        date_str = pub[:16] if len(pub) > 16 else pub

            if title:
                items.append({
                    'title': html_escape(title),
                    'url': link,
                    'date': date_str,
                    'source': name,
                    'summary': html_escape(summary) if summary else '',
                })
        return items
    except Exception as e:
        logger.warning(f"RSS feed error [{name}]: {e}")
        return []


# =============================================================================
# RENDER — 4-COLUMN PANEL LAYOUT
# =============================================================================

def render_news_column(region, feeds):
    """Render a single scrollable news column."""
    t = get_theme()
    pos_c = t['pos']

    all_items = []
    for name, url in feeds:
        items = fetch_rss_feed(name, url)
        all_items.extend(items)

    all_items.sort(key=lambda x: x['date'], reverse=True)

    # Header
    html = f"""<div style='padding:6px 10px;background-color:#1a2744;border-radius:4px 4px 0 0;font-family:{FONTS};margin-bottom:0;display:flex;justify-content:space-between;align-items:center'>
        <span style='color:#e2e8f0;font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase'>{region}</span>
        <span style='color:#4a5568;font-size:8px'>{len(all_items)}</span>
    </div>"""

    # Scrollable body
    html += "<div style='background-color:#0f1522;border:1px solid #1e293b;border-top:none;border-radius:0 0 4px 4px;max-height:600px;overflow-y:auto'>"

    if not all_items:
        html += f"<div style='padding:16px;color:#6d6d6d;font-size:10px;font-family:{FONTS};text-align:center'>Feeds loading…</div>"
    else:
        for i, item in enumerate(all_items[:25]):
            bg = '#0f1522' if i % 2 == 0 else '#131b2e'
            title_html = f"<a href='{item['url']}' target='_blank' style='color:#c9d1d9;text-decoration:none;font-size:10.5px;line-height:1.4'>{item['title']}</a>" if item['url'] else f"<span style='color:#c9d1d9;font-size:10.5px'>{item['title']}</span>"

            summary_html = ''
            if item['summary']:
                summary_html = f"<div style='color:#5a6270;font-size:9px;margin-top:2px;line-height:1.3'>{item['summary']}</div>"

            html += f"""<div style='padding:7px 10px;border-bottom:1px solid #1e293b;background:{bg};font-family:{FONTS}'>
                <div>{title_html}</div>
                {summary_html}
                <div style='font-size:8px;margin-top:3px'>
                    <span style='color:{pos_c};font-weight:600'>{item['source']}</span>
                    <span style='color:#4a4a4a;margin:0 3px'>·</span>
                    <span style='color:#6d6d6d'>{item['date']}</span>
                </div>
            </div>"""

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_news_tab(is_mobile):
    """NEWS tab — 4-column scrollable panels."""
    if is_mobile:
        for region, feeds in NEWS_FEEDS.items():
            render_news_column(region, feeds)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    else:
        cols = st.columns(4)
        for col, (region, feeds) in zip(cols, NEWS_FEEDS.items()):
            with col:
                render_news_column(region, feeds)
