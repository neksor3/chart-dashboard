import streamlit as st
import feedparser
import logging
import re
import urllib.request
from html import escape as html_escape, unescape as html_unescape
from config import FONTS, THEMES

logger = logging.getLogger(__name__)

_UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'

def get_theme():
    tn = st.session_state.get('theme', 'Emerald / Amber')
    return THEMES.get(tn, THEMES['Emerald / Amber'])

NEWS_FEEDS = {
    'Singapore': [
        ('CNA', 'https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml&category=6511'),
        ('Straits Times', 'https://www.straitstimes.com/news/business/rss.xml'),
        ('Business Times', 'https://www.businesstimes.com.sg/rss/top-stories'),
    ],
    'Regional': [
        ('SCMP', 'https://www.scmp.com/rss/5/feed'),
        ('Nikkei Asia', 'https://asia.nikkei.com/rss/feed/nar'),
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

def _clean(raw):
    if not raw: return ''
    t = re.sub(r'<[^>]+>', '', raw)
    return re.sub(r'\s+', ' ', html_unescape(t)).strip()

def _fetch_with_ua(url, timeout=10):
    """Fetch URL with browser user-agent. Returns raw bytes or None."""
    req = urllib.request.Request(url, headers={
        'User-Agent': _UA,
        'Accept': 'application/rss+xml, application/xml, text/xml, */*',
    })
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.read()
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_rss_feed(name, url):
    try:
        # Try with browser user-agent first (needed for Nikkei, etc)
        raw = _fetch_with_ua(url)
        if raw:
            feed = feedparser.parse(raw)
        else:
            feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:20]:
            title = _clean(getattr(entry, 'title', ''))
            link = getattr(entry, 'link', '')
            pub = getattr(entry, 'published', getattr(entry, 'updated', ''))
            if not title or 'shareholders are encouraged' in title.lower():
                continue
            date_str = ''
            if pub:
                try:
                    from email.utils import parsedate_to_datetime
                    date_str = parsedate_to_datetime(pub).strftime('%d %b %H:%M')
                except Exception:
                    date_str = pub[:16]
            items.append({'title': html_escape(title), 'url': link, 'date': date_str, 'source': name})
        return items
    except Exception as e:
        logger.warning(f"RSS error [{name}]: {e}")
        return []

def render_news_column(region, feeds):
    t = get_theme(); pos_c = t['pos']
    all_items = []
    for name, url in feeds:
        all_items.extend(fetch_rss_feed(name, url))
    all_items.sort(key=lambda x: x['date'], reverse=True)

    html = f"""<div style='padding:6px 10px;background-color:#1a2744;border-radius:4px 4px 0 0;font-family:{FONTS};display:flex;justify-content:space-between;align-items:center'>
        <span style='color:#e2e8f0;font-size:11px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase'>{region}</span>
        <span style='color:#4a5568;font-size:8px'>{len(all_items)}</span>
    </div>"""
    html += "<div style='background:#0f1522;border:1px solid #1e293b;border-top:none;border-radius:0 0 4px 4px;max-height:600px;overflow-y:auto'>"
    if not all_items:
        html += f"<div style='padding:16px;color:#6d6d6d;font-size:10px;font-family:{FONTS};text-align:center'>Feeds loading…</div>"
    else:
        for i, item in enumerate(all_items[:30]):
            bg = '#0f1522' if i % 2 == 0 else '#131b2e'
            a = f"<a href='{item['url']}' target='_blank' style='color:#c9d1d9;text-decoration:none;font-size:10.5px;line-height:1.4'>{item['title']}</a>" if item['url'] else f"<span style='color:#c9d1d9;font-size:10.5px'>{item['title']}</span>"
            html += f"""<div style='padding:6px 10px;border-bottom:1px solid #1e293b;background:{bg};font-family:{FONTS}'>
                <div>{a}</div>
                <div style='font-size:8px;margin-top:2px'><span style='color:{pos_c};font-weight:600'>{item['source']}</span> <span style='color:#4a4a4a'>·</span> <span style='color:#6d6d6d'>{item['date']}</span></div>
            </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_news_tab(is_mobile):
    if is_mobile:
        for region, feeds in NEWS_FEEDS.items():
            render_news_column(region, feeds)
    else:
        cols = st.columns(4)
        for col, (region, feeds) in zip(cols, NEWS_FEEDS.items()):
            with col:
                render_news_column(region, feeds)
