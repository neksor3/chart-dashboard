import streamlit as st
import feedparser
import logging
from html import escape as html_escape
from config import FONTS, THEMES

logger = logging.getLogger(__name__)

def get_theme():
    tn = st.session_state.get('theme', 'Emerald / Amber')
    return THEMES.get(tn, THEMES['Emerald / Amber'])

# =============================================================================
# RSS FEED SOURCES
# =============================================================================

NEWS_FEEDS = {
    'Singapore': [
        ('CNA Business', 'https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6511'),
        ('Straits Times', 'https://www.straitstimes.com/news/business/rss.xml'),
        ('MAS', 'https://www.mas.gov.sg/rss/news.xml'),
    ],
    'Hong Kong': [
        ('SCMP Markets', 'https://www.scmp.com/rss/5/feed'),
        ('SCMP HK', 'https://www.scmp.com/rss/2/feed'),
        ('SCMP Economy', 'https://www.scmp.com/rss/92/feed'),
    ],
    'Global Markets': [
        ('CNBC Markets', 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258'),
        ('BBC Business', 'https://feeds.bbci.co.uk/news/business/rss.xml'),
        ('Reuters', 'https://www.rss.app/feeds/v1.1/tHkFMMLQo0kgS2Md.xml'),
    ],
    'Central Banks': [
        ('Fed', 'https://www.federalreserve.gov/feeds/press_all.xml'),
        ('ECB', 'https://www.ecb.europa.eu/rss/press.html'),
        ('BOJ', 'https://www.boj.or.jp/en/rss/whatsnew.xml'),
    ],
}

# =============================================================================
# FETCH + RENDER
# =============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_rss_feed(name, url):
    """Fetch and parse a single RSS feed with error handling."""
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:15]:
            title = getattr(entry, 'title', '')
            link = getattr(entry, 'link', '')
            pub = getattr(entry, 'published', getattr(entry, 'updated', ''))
            date_str = ''
            if pub:
                try:
                    from email.utils import parsedate_to_datetime
                    dt = parsedate_to_datetime(pub)
                    date_str = dt.strftime('%d %b %H:%M')
                except Exception:
                    date_str = pub[:16] if len(pub) > 16 else pub
            if title:
                items.append({
                    'title': html_escape(title),
                    'url': link,
                    'date': date_str,
                    'source': name,
                })
        return items
    except Exception as e:
        logger.warning(f"RSS feed error [{name}]: {e}")
        return []


def render_news_column(region, feeds):
    """Render a single news column for a region."""
    t = get_theme()
    pos_c = t['pos']

    html = f"""<div style='padding:6px 10px;background-color:#1a2744;border-left:2px solid {pos_c};border-radius:4px 4px 0 0;font-family:{FONTS};margin-bottom:0'>
        <span style='color:#e2e8f0;font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase'>{region}</span>
    </div>"""
    html += "<div style='background-color:#0f1522;border:1px solid #1e293b;border-top:none;border-radius:0 0 4px 4px;max-height:700px;overflow-y:auto'>"

    all_items = []
    for name, url in feeds:
        items = fetch_rss_feed(name, url)
        all_items.extend(items)

    all_items.sort(key=lambda x: x['date'], reverse=True)

    if not all_items:
        html += f"<div style='padding:16px;color:#6d6d6d;font-size:11px;font-family:{FONTS}'>No news available — feed may be loading</div>"
    else:
        for item in all_items[:25]:
            title_html = f"<a href='{item['url']}' target='_blank' style='color:#c9d1d9;text-decoration:none;font-size:11px;line-height:1.4'>{item['title']}</a>" if item['url'] else f"<span style='color:#c9d1d9;font-size:11px'>{item['title']}</span>"
            html += f"""<div style='padding:8px 10px;border-bottom:1px solid #1e293b;font-family:{FONTS}'>
                <div>{title_html}</div>
                <div style='font-size:9px;margin-top:3px'>
                    <span style='color:{pos_c}'>{item['source']}</span>
                    <span style='color:#4a4a4a;margin:0 4px'>·</span>
                    <span style='color:#6d6d6d'>{item['date']}</span>
                </div>
            </div>"""

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_news_tab(is_mobile):
    """Full NEWS tab with multi-region RSS feeds."""
    if is_mobile:
        for region, feeds in NEWS_FEEDS.items():
            render_news_column(region, feeds)
    else:
        cols = st.columns(len(NEWS_FEEDS))
        for col, (region, feeds) in zip(cols, NEWS_FEEDS.items()):
            with col:
                render_news_column(region, feeds)
