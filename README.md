# Chart Dashboard — Streamlit Deployment Guide

## Files in this folder

```
streamlit-dashboard/
├── app.py                    ← Main dashboard (the only code file)
├── requirements.txt          ← Python packages needed
├── .streamlit/
│   └── config.toml           ← Dark theme settings
└── README.md                 ← This file
```

## Deploy to Streamlit Cloud (free)

### Step 1: Create a GitHub repo (private)
1. Go to https://github.com/new
2. Name it something like `chart-dashboard`
3. Set to **Private**
4. Click "Create repository"

### Step 2: Upload these files
1. Click "uploading an existing file" on the repo page
2. Drag all files from this folder (app.py, requirements.txt, .streamlit/config.toml)
3. Click "Commit changes"

### Step 3: Connect to Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repo → `chart-dashboard`
5. Main file path: `app.py`
6. Click "Deploy"

### Step 4: Share
- You'll get a URL like: `https://your-app-name.streamlit.app`
- Send this to your friend
- They see the dashboard, NOT your code (repo is private)

## Running locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Notes

- Data caches for 2 minutes (configurable via `ttl` in `@st.cache_data`)
- Free Streamlit Cloud apps sleep after inactivity — first load may take 30 seconds
- Yahoo Finance may rate-limit on heavy use; caching helps mitigate this
