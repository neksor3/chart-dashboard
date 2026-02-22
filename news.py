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
    tn = st.session_state.get('theme', 'Dark')
    return THEMES.get(tn, THEMES['Dark'])

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

@st.cache_data(ttl=900, show_spinner=False)
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

def render_news_panel(region, feeds, max_items=20):
    """Render a single news feed with equal source representation."""
    t = get_theme(); pos_c = t['pos']
    _body_bg = t.get('bg2', '#0f1522')
    _bdr = t.get('border', '#1e293b'); _txt = t.get('text', '#e2e8f0')
    _mut = t.get('muted', '#4a5568'); _link_c = t.get('text', '#c9d1d9')
    _row_alt = '#131b2e'

    # Fetch per source, then interleave 1/N
    per_source = {}
    for name, url in feeds:
        per_source[name] = fetch_rss_feed(name, url)
    n_sources = len(per_source)
    per_n = max(1, max_items // n_sources) if n_sources else max_items
    all_items = []
    for name, items in per_source.items():
        all_items.extend(items[:per_n])
    all_items.sort(key=lambda x: x['date'], reverse=True)
    all_items = all_items[:max_items]

    html = f"<div style='background:{_body_bg};border:1px solid {_bdr};border-radius:4px;max-height:75vh;overflow-y:auto'>"
    if not all_items:
        html += f"<div style='padding:16px;color:{_mut};font-size:11px;font-family:{FONTS};text-align:center'>Feeds loading…</div>"
    else:
        _txt2 = t.get('text2', '#94a3b8')
        _accent = t.get('accent', pos_c)
        for i, item in enumerate(all_items):
            bg = _body_bg if i % 2 == 0 else _row_alt
            title_el = f"<a href='{item['url']}' target='_blank' style='color:{_link_c};text-decoration:none;font-size:12px;font-weight:500'>{item['title']}</a>" if item['url'] else f"<span style='color:{_link_c};font-size:12px'>{item['title']}</span>"
            html += (
                f"<div style='padding:4px 10px;border-bottom:1px solid {_bdr}10;font-family:{FONTS};background:{bg};"
                f"display:flex;align-items:baseline;gap:8px;white-space:nowrap;overflow:hidden'>"
                f"<span style='font-size:10px;flex-shrink:0;width:200px;display:flex;gap:6px;align-items:baseline'>"
                f"<span style='color:{_accent};font-weight:600'>{item['source']}</span>"
                f"<span style='color:{_txt2}'>{item['date']}</span></span>"
                f"<span style='overflow:hidden;text-overflow:ellipsis'>{title_el}</span>"
                f"</div>"
            )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def render_news_tab(is_mobile):
    regions = list(NEWS_FEEDS.keys())
    tabs = st.tabs(regions)
    for tab, region in zip(tabs, regions):
        with tab:
            render_news_panel(region, NEWS_FEEDS[region])
