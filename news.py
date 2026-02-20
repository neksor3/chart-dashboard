import streamlit as st
import feedparser
import logging
import re
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
        ('Business Times', 'https://www.businesstimes.com.sg/rss/top-stories'),
        ('The Edge SG', 'https://www.theedgesingapore.com/rss.xml'),
    ],
    'Regional': [
        ('SCMP Markets', 'https://www.scmp.com/rss/5/feed'),
        ('SCMP Economy', 'https://www.scmp.com/rss/92/feed'),
        ('Nikkei Asia', 'https://asia.nikkei.com/rss/feed/nar'),
        ('Malay Mail', 'https://www.malaymail.com/feed/rss/money'),
        ('The Star Biz', 'https://www.thestar.com.my/rss/Business'),
    ],
    'Global': [
        ('Reuters', 'https://www.rss.app/feeds/v1.1/tHkFMMLQo0kgS2Md.xml'),
        ('Bloomberg', 'https://feeds.bloomberg.com/markets/news.rss'),
        ('Business Insider', 'https://markets.businessinsider.com/rss/news'),
        ('CNBC Markets', 'https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258'),
        ('FT', 'https://www.ft.com/rss/home'),
    ],
    'Tech': [
        ('TechCrunch AI', 'https://techcrunch.com/category/artificial-intelligence/feed/'),
        ('The Verge', 'https://www.theverge.com/rss/index.xml'),
        ('Ars Technica', 'https://feeds.arstechnica.com/arstechnica/technology-lab'),
        ('CoinDesk', 'https://www.coindesk.com/arc/outboundfeeds/rss/'),
        ('The Block', 'https://www.theblock.co/rss.xml'),
        ('STAT News', 'https://www.statnews.com/feed/'),
        ('Endpoints Health', 'https://endpts.com/feed/'),
        ('Longevity Tech', 'https://www.longevity.technology/feed/'),
    ],
}

# =============================================================================
# FETCH
# =============================================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_rss_feed(name, url):
    """Fetch and parse a single RSS feed with error handling."""
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:20]:
            title = getattr(entry, 'title', '')
            link = getattr(entry, 'link', '')
            pub = getattr(entry, 'published', getattr(entry, 'updated', ''))
            # Extract summary
            summary_raw = getattr(entry, 'summary', getattr(entry, 'description', ''))
            summary = ''
            if summary_raw:
                summary = re.sub(r'<[^>]+>', '', summary_raw)
                summary = summary[:140].strip()
                if len(summary_raw) > 140:
                    summary += '…'

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
                    'summary': html_escape(summary) if summary else '',
                })
        return items
    except Exception as e:
        logger.warning(f"RSS feed error [{name}]: {e}")
        return []


# =============================================================================
# RENDER
# =============================================================================

def render_feed_list(feeds):
    """Render a scrollable list of news items from multiple feeds."""
    t = get_theme()
    pos_c = t['pos']

    all_items = []
    for name, url in feeds:
        items = fetch_rss_feed(name, url)
        all_items.extend(items)

    # Sort by date descending
    all_items.sort(key=lambda x: x['date'], reverse=True)

    html = "<div style='background-color:#0f1522;border:1px solid #1e293b;border-radius:4px;max-height:650px;overflow-y:auto'>"

    if not all_items:
        html += f"<div style='padding:20px;color:#6d6d6d;font-size:12px;font-family:{FONTS};text-align:center'>No news available — feeds may be loading or unavailable</div>"
    else:
        for i, item in enumerate(all_items[:40]):
            bg = '#0f1522' if i % 2 == 0 else '#131b2e'
            title_html = f"<a href='{item['url']}' target='_blank' style='color:#e2e8f0;text-decoration:none;font-size:12px;font-weight:500;line-height:1.5'>{item['title']}</a>" if item['url'] else f"<span style='color:#e2e8f0;font-size:12px;font-weight:500'>{item['title']}</span>"

            summary_html = ''
            if item['summary']:
                summary_html = f"<div style='color:#6b7280;font-size:10px;margin-top:3px;line-height:1.4'>{item['summary']}</div>"

            html += f"""<div style='padding:10px 14px;border-bottom:1px solid #1e293b;background:{bg};font-family:{FONTS}'>
                <div>{title_html}</div>
                {summary_html}
                <div style='font-size:9px;margin-top:4px;display:flex;gap:8px;align-items:center'>
                    <span style='color:{pos_c};font-weight:600'>{item['source']}</span>
                    <span style='color:#4a4a4a'>·</span>
                    <span style='color:#6d6d6d'>{item['date']}</span>
                </div>
            </div>"""

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_news_tab(is_mobile):
    """NEWS tab with sub-tabs for each region."""
    categories = list(NEWS_FEEDS.keys())

    sub_tabs = st.tabs([c.upper() for c in categories])

    for tab, cat in zip(sub_tabs, categories):
        with tab:
            feeds = NEWS_FEEDS[cat]
            t = get_theme()
            source_tags = ' '.join(
                f"<span style='display:inline-block;padding:2px 8px;background:#1e293b;border-radius:3px;font-size:9px;color:#8a94a6;font-family:{FONTS};margin:2px'>{name}</span>"
                for name, _ in feeds
            )
            st.markdown(f"<div style='margin-bottom:8px'>{source_tags}</div>", unsafe_allow_html=True)
            render_feed_list(feeds)
