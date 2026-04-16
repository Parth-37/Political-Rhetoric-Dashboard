from __future__ import annotations

import io
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from gensim import corpora
from gensim.models import CoherenceModel, LdaModel
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)
from wordcloud import WordCloud

try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    PYLDAVIS_OK = True
except Exception:
    PYLDAVIS_OK = False

# ── Global matplotlib dark-text theme ─────────────────────────
plt.rcParams.update({
    "text.color":       "#111827",
    "axes.labelcolor":  "#111827",
    "xtick.color":      "#111827",
    "ytick.color":      "#111827",
    "axes.titlecolor":  "#111827",
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   "#d1d5db",
    "font.family":      "DejaVu Sans",
})

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Political Rhetoric — Jaishankar vs Tharoor",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
COLOR_J = "#E63946"
COLOR_T = "#1D3557"
ACCENT  = "#457B9D"

SPEAKER_COLORS  = {"Jaishankar": COLOR_J, "Tharoor": COLOR_T}
SPEAKER_CMAPS   = {"Jaishankar": "Reds",  "Tharoor": "Blues"}
SPEAKER_PLT_CM  = {"Jaishankar": plt.cm.Reds, "Tharoor": plt.cm.Blues}

NUM_TOPICS   = 6
LDA_PASSES   = 25
RANDOM_STATE = 42

BASE = Path(__file__).parent

DATA_CANDIDATES = [
    BASE / "speeches_dataset.csv",
    Path.cwd() / "speeches_dataset.csv",
    Path(r"C:\Users\parth\OneDrive\Desktop\Political\speeches_dataset.csv"),
]

IMG_CANDIDATES = {
    "Jaishankar": [
        BASE / "jaishankar1.jpg", BASE / "jaishankar1.png",
        BASE / "jaishankar2.jpg", BASE / "jaishankar2.png",
        BASE / "jaishankar.jpg",  BASE / "jaishankar.png",
    ],
    "Tharoor": [
        BASE / "tharoor1.jpg", BASE / "tharoor1.png",
        BASE / "tharoor2.jpg", BASE / "tharoor2.png",
        BASE / "tharoor.jpg",  BASE / "tharoor.png",
    ],
}

# ─────────────────────────────────────────────────────────────
# NLP UTILITIES
# ─────────────────────────────────────────────────────────────
_BUILTIN_STOP = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once",
    "here","there","when","where","why","how","all","both","each","few","more",
    "most","other","some","such","no","nor","not","only","own","same","so",
    "than","too","very","s","t","can","will","just","don","should","now","d",
    "ll","m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn",
    "hasn","haven","isn","ma","mightn","mustn","needn","shan","shouldn",
    "wasn","weren","won","wouldn","said","say","says","also","one","two","three",
    "would","could","like","get","got","make","made","year","years","time",
    "times","new","news","today","report","must","need","well",
}
_DOMAIN_STOP = {
    "jaishankar","tharoor","shashi","subrahmanyam","eam","minister","mp",
    "mea","external","affairs","foreign","government","india","indian",
    "country","people","nation","world","global","international","national",
    "address","discuss","highlight","spoke","argu","emphasiz","deliver",
    "includ","particip","meet","met","visit","said","says",
}
STOP_WORDS: Set[str] = _BUILTIN_STOP | _DOMAIN_STOP | set(ENGLISH_STOP_WORDS)

_IRREG = {
    "countries":"country","policies":"policy","economies":"economy",
    "allies":"ally","ties":"tie","parties":"party","bodies":"body",
    "studies":"study","territories":"territory","activities":"activity",
    "strategies":"strategy","communities":"community","authorities":"authority",
    "opportunities":"opportunity","priorities":"priority","speeches":"speech",
}

def _lem(w: str) -> str:
    if w in _IRREG:
        return _IRREG[w]
    for s, r in [("ying","y"),("ies","y"),("ves","f"),("ness",""),
                 ("ment",""),("tion","te"),("ing",""),("ied","y"),
                 ("ess",""),("ers","er"),("ings",""),("ed",""),("s","")]:
        if w.endswith(s) and len(w) - len(s) > 2:
            return w[:-len(s)] + r
    return w

def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    out = []
    for t in text.split():
        if len(t) < 3 or t in STOP_WORDS or not t.isalpha():
            continue
        lemma = _lem(t)
        if lemma not in STOP_WORDS and len(lemma) >= 3:
            out.append(lemma)
    return out

# ─────────────────────────────────────────────────────────────
# CSS INJECTION
# ─────────────────────────────────────────────────────────────
def inject_styles() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global resets — force light theme text everywhere ── */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
        color: #111827 !important;
    }}
    .stApp {{
        background: linear-gradient(160deg, #f7f9fc 0%, #eef3f9 60%, #e8edf5 100%);
        color: #111827 !important;
    }}
    .block-container {{ padding-top: 0.5rem; padding-bottom: 2rem; max-width: 1440px; }}

    /* Force all text dark */
    p, span, div, li, td, th, label, h1, h2, h3, h4, h5, h6 {{ color: #111827; }}

    /* ── Metric cards — explicit dark text ── */
    div[data-testid="stMetric"] {{
        background: #fff; border-radius: 16px; padding: 1rem 1.2rem;
        box-shadow: 0 2px 14px rgba(0,0,0,.07); border: 1px solid rgba(0,0,0,.06);
    }}
    div[data-testid="stMetricValue"] > div {{
        color: #111827 !important; font-size: 1.65rem !important;
        font-weight: 800 !important;
    }}
    div[data-testid="stMetricLabel"] > div {{
        color: #6b7280 !important; font-size: .82rem !important;
        font-weight: 500 !important;
    }}
    div[data-testid="stMetricDelta"] {{ color: #374151 !important; }}

    /* Streamlit native widgets */
    .stSelectbox label, .stSlider label, .stRadio label,
    .stTextInput label, .stCaption {{ color: #374151 !important; }}
    .stMarkdown, .stMarkdown p {{ color: #111827 !important; }}
    .stDataFrame {{ color: #111827 !important; }}

    /* ── Hero ── */
    .hero {{
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #1d3557 100%);
        border-radius: 24px; padding: 2.5rem 3rem; margin-bottom: 1.5rem;
        position: relative; overflow: hidden;
    }}
    .hero::before {{
        content: ''; position: absolute; inset: 0;
        background:
            radial-gradient(circle at 15% 50%, rgba(230,57,70,.20) 0%, transparent 55%),
            radial-gradient(circle at 85% 50%, rgba(69,123,157,.25) 0%, transparent 55%);
    }}
    .hero-title {{
        font-size: 2.8rem; font-weight: 900; color: #fff !important;
        letter-spacing: -1px; position: relative; z-index: 1; line-height: 1.1;
    }}
    .hero-sub {{
        font-size: 1.05rem; color: rgba(255,255,255,.78) !important;
        margin-top: .5rem; position: relative; z-index: 1;
    }}
    .hero-pills {{ display:flex; gap:.6rem; margin-top:1.3rem; position:relative; z-index:1; flex-wrap:wrap; }}
    .pill {{
        padding:.3rem .9rem; border-radius:999px; font-size:.78rem;
        font-weight:700; letter-spacing:.3px;
    }}
    .pill-r {{ background:rgba(230,57,70,.22); color:#fca5a5 !important; border:1px solid rgba(230,57,70,.4); }}
    .pill-b {{ background:rgba(69,123,157,.25); color:#93c5fd !important; border:1px solid rgba(69,123,157,.4); }}
    .pill-g {{ background:rgba(255,255,255,.10); color:rgba(255,255,255,.85) !important; border:1px solid rgba(255,255,255,.2); }}

    /* ── Leader cards ── */
    .lcard {{
        background: #fff; border-radius: 22px; padding: 2rem 1.5rem 1.5rem;
        box-shadow: 0 6px 32px rgba(0,0,0,.09); border: 1px solid rgba(0,0,0,.05);
        text-align: center; transition: transform .22s ease, box-shadow .22s ease;
    }}
    .lcard:hover {{ transform: translateY(-6px); box-shadow: 0 16px 48px rgba(0,0,0,.13); }}
    .photo-ring {{
        width: 110px; height: 110px; border-radius: 50%; margin: 0 auto 1rem;
        border: 4px solid transparent; background-clip: padding-box;
        position: relative; display: flex; align-items: center; justify-content: center;
        overflow: hidden;
    }}
    .avatar {{
        width: 110px; height: 110px; border-radius: 50%; margin: 0 auto 1rem;
        display: flex; align-items: center; justify-content: center;
        font-size: 2.8rem; font-weight: 900; color: #fff !important;
        box-shadow: 0 6px 24px rgba(0,0,0,.25); position: relative;
    }}
    .avatar-initials {{ color: #fff !important; font-size: 2.6rem; font-weight: 900; }}
    .lname {{ font-size: 1.3rem; font-weight: 800; margin-bottom: .25rem; color: #111827 !important; }}
    .lrole {{ font-size: .8rem; color: #6b7280 !important; margin-bottom: .7rem; line-height: 1.45; }}
    .lparty {{
        display: inline-block; padding: .3rem .85rem; border-radius: 999px;
        font-size: .74rem; font-weight: 700; letter-spacing: .3px;
    }}
    .lstats {{
        display: flex; justify-content: space-around; margin-top: 1.2rem;
        padding-top: .9rem; border-top: 2px solid #f3f4f6;
    }}
    .lstat-val {{ font-weight: 800; font-size: 1.2rem; }}
    .lstat-lbl {{ font-size: .7rem; color: #9ca3af !important; font-weight: 500; }}

    /* ── Section title ── */
    .sec-title {{
        font-size: 1.3rem; font-weight: 800; color: #111827 !important;
        margin: 1.2rem 0 .8rem; padding-bottom: .5rem;
        border-bottom: 3px solid #e5e7eb; display: flex; align-items: center; gap: .4rem;
    }}

    /* ── Insight / info card ── */
    .icard {{
        background: #fff; border-radius: 16px; padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,.06); margin-bottom: .9rem;
        border: 1px solid rgba(0,0,0,.06);
    }}
    .ititle {{ font-size: 1.0rem; font-weight: 700; margin-bottom: .3rem; color: #111827 !important; }}
    .inote  {{ font-size: .88rem; color: #374151 !important; line-height: 1.55; }}

    /* ── Topic box ── */
    .tbox {{
        background: #f0f4f8; border-left: 4px solid {ACCENT};
        padding: .7rem .9rem; border-radius: 10px; margin-bottom: .6rem;
        font-size: .88rem; color: #1f2937 !important;
    }}
    .tbox b {{ color: #1f2937 !important; }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap:.4rem; background:rgba(255,255,255,.7);
        padding:.4rem; border-radius:14px; backdrop-filter:blur(8px); flex-wrap:wrap;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius:10px; padding:.45rem 1.1rem;
        font-weight:600; font-size:.88rem; color:#4b5563 !important;
    }}
    .stTabs [aria-selected="true"] {{
        background:#fff !important; color:#111827 !important;
        box-shadow:0 2px 10px rgba(0,0,0,.10);
    }}

    /* ── Word chip ── */
    .chip {{
        display:inline-block; padding:.18rem .55rem; border-radius:6px;
        font-size:.78rem; font-weight:500; margin:.12rem;
        border:1px solid rgba(0,0,0,.1); color:#111827 !important;
    }}

    /* ── Interpretation table ── */
    .itable {{ width:100%; border-collapse:separate; border-spacing:0; font-size:.85rem; }}
    .itable th {{
        background:#f3f4f6; padding:.7rem 1rem; text-align:left;
        font-weight:700; border-bottom:2px solid #e5e7eb; color:#111827 !important;
    }}
    .itable td {{ padding:.6rem 1rem; border-bottom:1px solid #f3f4f6; color:#1f2937 !important; }}
    .itable tr:last-child td {{ border-bottom:none; }}
    .itable tr:hover td {{ background:#f9fafb; }}

    /* ── Venn chips ── */
    .vchip {{
        display:inline-flex; align-items:center; gap:.35rem;
        padding:.35rem .8rem; border-radius:999px; font-size:.8rem; font-weight:600;
        margin:.2rem;
    }}
    </style>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    path = None
    for c in DATA_CANDIDATES:
        if c.exists():
            path = c
            break
    if path is None:
        raise FileNotFoundError("speeches_dataset.csv not found.")

    df = pd.read_csv(path)
    df = df.dropna(subset=["speaker", "full_text"]).copy()
    df["full_text"]    = df["full_text"].astype(str).str.strip()
    df["speaker"]      = df["speaker"].astype(str).str.strip()
    df["source"]       = df.get("source", pd.Series(["Unknown"] * len(df))).astype(str).str.strip()
    df["publishedAt"]  = pd.to_datetime(df.get("publishedAt", pd.NaT), errors="coerce")
    df = df.dropna(subset=["publishedAt"])
    df = df[df["full_text"].str.len() >= 50].copy()
    df["word_count"]   = df["full_text"].str.split().str.len()
    df["char_count"]   = df["full_text"].str.len()
    df["month"]        = df["publishedAt"].dt.to_period("M").dt.to_timestamp()
    return df.sort_values("publishedAt").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    p = df.copy()
    p["tokens"]     = p["full_text"].apply(tokenize)
    p["clean_text"] = p["tokens"].apply(" ".join)
    return p[p["clean_text"].str.len() > 0].copy()

# ─────────────────────────────────────────────────────────────
# TOPIC MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def run_lda(token_tup: Tuple[Tuple[str, ...], ...], k: int) -> Dict:
    lists = [list(t) for t in token_tup if t]
    dct   = corpora.Dictionary(lists)
    dct.filter_extremes(no_below=2, no_above=0.90)
    corpus = [dct.doc2bow(d) for d in lists]
    corpus = [c for c in corpus if c]

    model = LdaModel(
        corpus=corpus, id2word=dct, num_topics=k,
        passes=LDA_PASSES, random_state=RANDOM_STATE,
        alpha="auto", eta="auto", per_word_topics=True,
    )
    coh = CoherenceModel(
        model=model, texts=lists, dictionary=dct, coherence="c_v"
    ).get_coherence()

    rows = []
    for tid in range(k):
        for rank, (word, wt) in enumerate(model.show_topic(tid, topn=12), 1):
            rows.append({"topic": f"Topic {tid+1}", "topic_id": tid,
                         "rank": rank, "keyword": word, "weight": float(wt)})

    n   = len(corpus)
    mat = np.zeros((n, k))
    for i, bow in enumerate(corpus):
        for tid, prob in model.get_document_topics(bow):
            mat[i, tid] = prob

    return {
        "model": model, "dictionary": dct, "corpus": corpus,
        "token_lists": lists, "coherence": coh,
        "topic_df": pd.DataFrame(rows), "doc_topic": mat,
    }


@st.cache_resource(show_spinner=False)
def run_nmf(texts_tup: Tuple[str, ...], k: int) -> Dict:
    docs = [t for t in texts_tup if t.strip()]
    vec  = TfidfVectorizer(max_features=1200, min_df=2, max_df=0.9)
    dtm  = vec.fit_transform(docs)
    mdl  = NMF(n_components=k, random_state=RANDOM_STATE,
               init="nndsvda", max_iter=500)
    mdl.fit(dtm)

    feat = vec.get_feature_names_out()
    rows = []
    for tid, comp in enumerate(mdl.components_):
        for rank, idx in enumerate(comp.argsort()[::-1][:12], 1):
            rows.append({"topic": f"Topic {tid+1}", "topic_id": tid,
                         "rank": rank, "keyword": feat[idx],
                         "weight": float(comp[idx])})
    return {
        "model": mdl, "vectorizer": vec, "dtm": dtm,
        "reconstruction_error": float(getattr(mdl, "reconstruction_err_", np.nan)),
        "topic_df": pd.DataFrame(rows),
    }


@st.cache_data(show_spinner=False)
def ldavis_html(token_tup: Tuple[Tuple[str, ...], ...], k: int) -> str:
    if not PYLDAVIS_OK:
        return ""
    res  = run_lda(token_tup, k)
    prep = gensimvis.prepare(res["model"], res["corpus"], res["dictionary"])
    buf  = io.StringIO()
    pyLDAvis.save_html(prep, buf)
    return buf.getvalue()

# ─────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────
def fig_wordcloud(topic_df: pd.DataFrame, color: str, title: str) -> plt.Figure:
    freq: Dict[str, float] = {}
    for _, r in topic_df.iterrows():
        freq[r["keyword"]] = freq.get(r["keyword"], 0) + r["weight"]
    cmap = LinearSegmentedColormap.from_list("wc", ["#F1F5F9", color, "#111827"])
    wc = WordCloud(width=1000, height=400, background_color="white",
                   colormap=cmap, max_words=80, collocations=False
                   ).generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", color=color, pad=12)
    plt.tight_layout()
    return fig


def fig_topic_grid(topic_df: pd.DataFrame, k: int, color: str, title: str) -> plt.Figure:
    cols = 3
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.4))
    axes = np.array(axes).flatten()
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    for tid in range(k):
        ax = axes[tid]
        sl = topic_df[topic_df["topic_id"] == tid].sort_values("weight").tail(8)
        ax.barh(sl["keyword"], sl["weight"], color=color, alpha=0.82)
        ax.set_title(f"Topic {tid+1}", fontsize=9, fontweight="bold", color=color)
        ax.tick_params(labelsize=7.5)
        ax.set_xlabel("Weight", fontsize=7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for i in range(k, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    return fig


def fig_heatmap(mat: np.ndarray, color: str, title: str) -> plt.Figure:
    avg = mat.mean(axis=0).reshape(1, -1)
    lbl = [f"T{i+1}" for i in range(mat.shape[1])]
    fig, ax = plt.subplots(figsize=(10, 2.5))
    sns.heatmap(avg, ax=ax, annot=True, fmt=".3f",
                cmap=SPEAKER_CMAPS[title] if title in SPEAKER_CMAPS else "Blues",
                xticklabels=lbl, yticklabels=[title],
                linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Avg Prob"})
    ax.set_title(f"Average Document–Topic Distribution — {title}",
                 fontweight="bold", fontsize=11, color=color)
    plt.tight_layout()
    return fig


def fig_pie(mat: np.ndarray, color: str, title: str) -> plt.Figure:
    n   = mat.shape[1]
    dom = mat.argmax(axis=1)
    cnt = Counter(dom)
    sizes  = [cnt.get(i, 0) for i in range(n)]
    labels = [f"Topic {i+1}" for i in range(n)]
    cm     = SPEAKER_PLT_CM[title] if title in SPEAKER_PLT_CM else plt.cm.Blues
    colors = [cm(0.3 + 0.7 * i / max(n - 1, 1)) for i in range(n)]
    fig, ax = plt.subplots(figsize=(6, 5.2))
    wedges, texts, autos = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", colors=colors,
        startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    for a in autos: a.set_fontsize(8.5)
    ax.set_title(f"Dominant Topic — {title}", fontweight="bold", color=color, pad=8)
    plt.tight_layout()
    return fig


def fig_topic_share(mat: np.ndarray, color: str, title: str) -> plt.Figure:
    n   = mat.shape[1]
    dom = mat.argmax(axis=1)
    cnt = Counter(dom)
    tot = sum(cnt.values()) or 1
    pct = [cnt.get(i, 0) / tot * 100 for i in range(n)]
    lbl = [f"T{i+1}" for i in range(n)]
    cm     = SPEAKER_PLT_CM[title] if title in SPEAKER_PLT_CM else plt.cm.Blues
    colors = [cm(0.35 + 0.65 * i / max(n - 1, 1)) for i in range(n)]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(lbl, pct, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
    for bar, val in zip(bars, pct):
        if val > 2:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, f"{val:.1f}%", ha="center", fontsize=8.5)
    ax.set_xlabel("Topic"); ax.set_ylabel("% of Articles")
    ax.set_title(f"Topic Share — {title}", fontweight="bold", color=color)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    return fig


def fig_venn(vocab_j: Set[str], vocab_t: Set[str]) -> plt.Figure:
    only_j  = vocab_j - vocab_t
    shared  = vocab_j & vocab_t
    only_t  = vocab_t - vocab_j

    fig, ax = plt.subplots(figsize=(16, 9.5), facecolor="#f8fafc")
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("#f8fafc")

    # Circles
    for cx, col in [(3.6, COLOR_J), (6.4, COLOR_T)]:
        ax.add_patch(plt.Circle((cx, 3.5), 2.9, color=col, alpha=0.14, zorder=1))
        ax.add_patch(plt.Circle((cx, 3.5), 2.9, fill=False,
                                edgecolor=col, linewidth=3.5, zorder=2))

    # Speaker labels on circle edges
    kw = dict(fontsize=11, fontweight="900", ha="center", va="center", zorder=6)
    ax.text(1.3, 3.5, "JAISHANKAR", color=COLOR_J, rotation=90, **kw)
    ax.text(8.7, 3.5, "THAROOR",    color=COLOR_T, rotation=90, **kw)

    # Count badges
    badge_kw = dict(fontsize=9.5, ha="center", va="center", fontweight="700")
    ax.text(2.5, 6.1, f"{len(only_j)}\nexclusive\nterms",
            color=COLOR_J, **badge_kw)
    ax.text(5.0, 6.5, f"{len(shared)} shared terms",
            color="#374151", fontsize=10, fontweight="700", ha="center", va="center")
    ax.text(7.5, 6.1, f"{len(only_t)}\nexclusive\nterms",
            color=COLOR_T, **badge_kw)

    # Word boxes — helper
    def word_box(x, y, words, color, n=10):
        text = "\n".join(f"• {w}" for w in sorted(words)[:n])
        ax.text(x, y, text, fontsize=8.2, ha="center", va="center",
                color=color, fontfamily="monospace", zorder=5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=color, alpha=0.92, linewidth=1.2))

    word_box(2.35, 3.5, only_j, "#7f1d1d")
    word_box(7.65, 3.5, only_t, "#1e3a5f")

    # Shared — center
    s_words = sorted(shared)[:9]
    s_text  = "\n".join(f"✦ {w}" for w in s_words)
    ax.text(5.0, 3.5, s_text, fontsize=8.5, ha="center", va="center",
            color="#1f2937", fontfamily="monospace", fontweight="600", zorder=5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="#9ca3af", alpha=0.95, linewidth=1.5))

    # Title
    ax.text(5.0, 6.85, "Topic Vocabulary — Venn Diagram",
            fontsize=16, fontweight="900", ha="center", va="center", color="#111827")
    ax.text(5.0, 0.18,
            f"Top 10 words per region shown  ·  "
            f"J-exclusive = {len(only_j)}   |   Shared = {len(shared)}   |   T-exclusive = {len(only_t)}",
            fontsize=9, ha="center", va="center", color="#6b7280")

    plt.tight_layout()
    return fig


def fig_vocab_heatmap(proc: pd.DataFrame) -> plt.Figure:
    vec = CountVectorizer(max_features=500)
    mat = vec.fit_transform(proc["clean_text"])
    vocab  = vec.get_feature_names_out()
    fdf    = pd.DataFrame(mat.toarray(), columns=vocab)
    fdf["speaker"] = proc["speaker"].values
    sp = fdf.groupby("speaker").sum().T
    sp = sp.div(sp.sum(axis=0), axis=1)
    sp["diff"] = (sp.get("Jaishankar", 0) - sp.get("Tharoor", 0)).abs()
    top = sp.sort_values("diff", ascending=False).head(25)

    fig, ax = plt.subplots(figsize=(9, 10))
    cols = [c for c in ["Jaishankar", "Tharoor"] if c in top.columns]
    sns.heatmap(
        top[cols], cmap=sns.diverging_palette(10, 220, s=80, l=50, as_cmap=True),
        annot=True, fmt=".3f", linewidths=0.4,
        cbar_kws={"label": "Normalized Frequency", "shrink": 0.7}, ax=ax,
    )
    ax.set_title("Top 25 Vocabulary Divergence by Speaker",
                 fontweight="bold", fontsize=12)
    ax.set_xlabel("Speaker"); ax.set_ylabel("Term")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────
# PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────
PLOTLY_FONT = dict(family="Inter, sans-serif", color="#111827", size=13)

def _chart_theme(fig):
    """Apply dark-text light theme to any plotly figure."""
    fig.update_layout(
        font=PLOTLY_FONT,
        title_font=dict(color="#111827", size=15, family="Inter, sans-serif"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            color="#111827",
            tickfont=dict(color="#111827"),
            title_font=dict(color="#111827"),
            gridcolor="#f0f0f0",
            linecolor="#d1d5db",
        ),
        yaxis=dict(
            color="#111827",
            tickfont=dict(color="#111827"),
            title_font=dict(color="#111827"),
            gridcolor="#f0f0f0",
            linecolor="#d1d5db",
        ),
        legend=dict(
            font=dict(color="#111827"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#e5e7eb",
            borderwidth=1,
        ),
        hoverlabel=dict(bgcolor="white", font_color="#111827", bordercolor="#e5e7eb"),
    )
    return fig

# ─────────────────────────────────────────────────────────────
# LEADER SHOWCASE
# ─────────────────────────────────────────────────────────────
def _load_images_b64(key: str) -> List[str]:
    """Return list of base64-encoded images for a speaker (all found files)."""
    import base64 as _b64
    found = []
    for p in IMG_CANDIDATES.get(key, []):
        if p.exists():
            data = _b64.b64encode(p.read_bytes()).decode()
            ext  = p.suffix.lstrip(".")
            found.append(f"data:image/{ext};base64,{data}")
    return found


def leader_showcase(key: str, name: str, role: str, party: str,
                    color: str, df: pd.DataFrame) -> None:
    from PIL import Image as PILImage

    sp      = df[df["speaker"] == key]
    n_arts  = len(sp)
    avg_wrd = int(sp["word_count"].mean()) if n_arts else 0
    date_rng = ""
    if n_arts:
        d0 = sp["publishedAt"].min().strftime("%b %Y")
        d1 = sp["publishedAt"].max().strftime("%b %Y")
        date_rng = f"{d0} – {d1}"

    pbg = f"rgba({'230,57,70' if color == COLOR_J else '29,53,87'},.07)"

    # ── Leader name ABOVE photo ──
    st.markdown(
        f'<div style="font-size:1.5rem;font-weight:900;color:{color};'
        f'text-align:center;margin-bottom:.5rem;">{name}</div>',
        unsafe_allow_html=True)

    # ── Photo: crop to fixed 4:3 ratio so both cards are same height ──
    photo_path = None
    for p in IMG_CANDIDATES.get(key, []):
        if p.exists():
            photo_path = p
            break

    if photo_path:
        img = PILImage.open(photo_path).convert("RGB")
        # Crop to 4:3
        tw, th = img.width, int(img.width * 3 / 4)
        if img.height < th:
            th = img.height
            tw = int(th * 4 / 3)
        left = (img.width  - tw) // 2
        top  = 0                          # keep faces at top
        img  = img.crop((left, top, left + tw, top + th))
        img  = img.resize((600, 450))
        st.image(img, use_container_width=True)
    else:
        st.markdown(
            f'<div style="height:280px;border-radius:16px;'
            f'background:linear-gradient(160deg,{color},{color}88);'
            f'display:flex;align-items:center;justify-content:center;">'
            f'<span style="font-size:6rem;font-weight:900;color:#fff;">{key[0]}</span>'
            f'</div>',
            unsafe_allow_html=True)

    # ── Role & party badge ──
    st.markdown(f'<div style="font-size:.8rem;color:#6b7280;text-align:center;margin:.4rem 0;">{role}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align:center;margin-bottom:.6rem;"><span style="padding:.28rem .85rem;border-radius:999px;font-size:.73rem;font-weight:700;background:{pbg};color:{color};border:1px solid {color}44;">{party}</span></div>', unsafe_allow_html=True)

    # ── Stats ──
    c1, c2, c3 = st.columns(3)
    c1.metric("Speeches",  n_arts)
    c2.metric("Avg Words", avg_wrd)
    c3.metric("Coverage",  date_rng if date_rng else "2024")

# ─────────────────────────────────────────────────────────────
# TAB: OVERVIEW
# ─────────────────────────────────────────────────────────────
def tab_overview(df: pd.DataFrame) -> None:
    st.markdown('<div class="sec-title">📋 Project Overview</div>', unsafe_allow_html=True)
    st.markdown(
        "This dashboard compares the **rhetorical themes** of two prominent Indian political "
        "figures — **S. Jaishankar** (ruling BJP) and **Shashi Tharoor** (opposition INC) — "
        "using NLP-driven topic modelling (LDA & NMF). Analyses include word clouds, topic "
        "bars, document–topic heatmaps, dominant-topic pies, a Venn diagram of vocabulary "
        "overlap, and a comparative interpretation guide."
    )

    # Methodology table
    st.markdown("""
    | Section | Method |
    |---------|--------|
    | Preprocessing | Regex cleaning · Rule-based lemmatisation · Custom stopwords |
    | Topic Modelling | LDA (Gensim) · NMF (scikit-learn) · K = 6 topics |
    | Evaluation | LDA Coherence Cv · NMF Reconstruction Error |
    | Comparison | Vocabulary Venn · TF-IDF Heatmap · Side-by-side themes |
    """)

    st.markdown('<div class="sec-title">Speaker Snapshot</div>', unsafe_allow_html=True)
    snap = (
        df.groupby("speaker")
        .agg(articles=("speaker", "size"),
             avg_words=("word_count", "mean"),
             first_date=("publishedAt", "min"),
             last_date=("publishedAt", "max"))
        .reset_index()
    )
    snap["avg_words"]   = snap["avg_words"].round(1)
    snap["first_date"]  = snap["first_date"].dt.strftime("%d %b %Y")
    snap["last_date"]   = snap["last_date"].dt.strftime("%d %b %Y")
    st.dataframe(snap, use_container_width=True, hide_index=True)

    st.markdown('<div class="sec-title">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(
        df[["speaker", "publishedAt", "source", "word_count", "full_text"]].head(30),
        use_container_width=True, hide_index=True,
    )

# ─────────────────────────────────────────────────────────────
# TAB: EDA
# ─────────────────────────────────────────────────────────────
def tab_eda(df: pd.DataFrame) -> None:
    st.markdown('<div class="sec-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    cnt = df["speaker"].value_counts().reset_index()
    cnt.columns = ["speaker", "articles"]
    fig_cnt = px.bar(
        cnt, x="speaker", y="articles", color="speaker",
        color_discrete_map=SPEAKER_COLORS, text="articles",
        title="Article Counts by Speaker",
    )
    fig_cnt.update_traces(textposition="outside", textfont=dict(color="#111827", size=14))
    _chart_theme(fig_cnt)
    fig_cnt.update_layout(showlegend=False, height=380)

    fig_hist = px.histogram(
        df, x="word_count", color="speaker", nbins=30, barmode="overlay",
        opacity=0.7, color_discrete_map=SPEAKER_COLORS,
        title="Word-Count Distribution by Speaker",
        labels={"word_count": "Word Count", "count": "Frequency"},
    )
    _chart_theme(fig_hist)
    fig_hist.update_layout(height=380)

    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(fig_cnt, use_container_width=True)
    with col2: st.plotly_chart(fig_hist, use_container_width=True)

    monthly = (
        df.groupby(["month", "speaker"]).size()
        .reset_index(name="articles").sort_values("month")
    )
    fig_time = px.line(
        monthly, x="month", y="articles", color="speaker", markers=True,
        color_discrete_map=SPEAKER_COLORS,
        title="Coverage Over Time (Articles per Month)",
        labels={"month": "Month", "articles": "Articles"},
    )
    _chart_theme(fig_time)
    fig_time.update_layout(height=400)
    fig_time.update_traces(line=dict(width=3), marker=dict(size=8))
    st.plotly_chart(fig_time, use_container_width=True)

    src = (
        df.groupby(["source", "speaker"]).size()
        .reset_index(name="articles").sort_values("articles", ascending=False)
    )
    fig_src = px.bar(
        src, x="source", y="articles", color="speaker", barmode="group",
        color_discrete_map=SPEAKER_COLORS,
        title="Source Distribution by Speaker",
        labels={"source": "News Source", "articles": "Articles"},
    )
    _chart_theme(fig_src)
    fig_src.update_layout(height=400)
    st.plotly_chart(fig_src, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB: WORD CLOUDS
# ─────────────────────────────────────────────────────────────
def tab_wordclouds(lda_j: Dict, lda_t: Dict) -> None:
    st.markdown('<div class="sec-title">☁️ Vocabulary Word Clouds</div>',
                unsafe_allow_html=True)
    st.caption("Generated from aggregated LDA topic word weights.")

    col1, col2 = st.columns(2)
    with col1:
        try:
            st.pyplot(fig_wordcloud(lda_j["topic_df"], COLOR_J, "Jaishankar"),
                      use_container_width=True)
        except Exception as e:
            st.warning(f"Word cloud error: {e}")

    with col2:
        try:
            st.pyplot(fig_wordcloud(lda_t["topic_df"], COLOR_T, "Tharoor"),
                      use_container_width=True)
        except Exception as e:
            st.warning(f"Word cloud error: {e}")

    # Individual speaker raw token clouds
    st.markdown("---")
    st.markdown("#### Raw Token Frequency Clouds")
    st.caption("Generated from preprocessed token frequencies (all tokens combined).")

    proc_j_tokens = " ".join(w for doc in lda_j["token_lists"] for w in doc)
    proc_t_tokens = " ".join(w for doc in lda_t["token_lists"] for w in doc)

    col3, col4 = st.columns(2)
    for col, tokens, color, name in [
        (col3, proc_j_tokens, COLOR_J, "Jaishankar"),
        (col4, proc_t_tokens, COLOR_T, "Tharoor"),
    ]:
        with col:
            if tokens.strip():
                cmap = LinearSegmentedColormap.from_list("c", ["#F1F5F9", color, "#111827"])
                wc = WordCloud(width=900, height=380, background_color="white",
                               colormap=cmap, max_words=100, collocations=False
                               ).generate(tokens)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                ax.set_title(f"{name} — Token Frequency",
                             fontsize=12, fontweight="bold", color=color)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB: LDA
# ─────────────────────────────────────────────────────────────
def tab_lda(lda_j: Dict, lda_t: Dict,
            j_tokens: Tuple, t_tokens: Tuple) -> None:
    st.markdown('<div class="sec-title">🔷 LDA Topic Modelling</div>',
                unsafe_allow_html=True)

    # Coherence metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("J — Coherence (Cv)", f"{lda_j['coherence']:.4f}")
    m2.metric("J — Vocab Size",     f"{len(lda_j['dictionary'])}")
    m3.metric("T — Coherence (Cv)", f"{lda_t['coherence']:.4f}")
    m4.metric("T — Vocab Size",     f"{len(lda_t['dictionary'])}")

    st.markdown("---")
    st.subheader("Topic Keywords Table")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**<span style='color:{COLOR_J}'>Jaishankar</span>**",
                    unsafe_allow_html=True)
        top_j = lda_j["topic_df"][lda_j["topic_df"]["rank"] <= 8]
        st.dataframe(top_j[["topic", "keyword", "weight"]],
                     use_container_width=True, hide_index=True)
    with col2:
        st.markdown(f"**<span style='color:{COLOR_T}'>Tharoor</span>**",
                    unsafe_allow_html=True)
        top_t = lda_t["topic_df"][lda_t["topic_df"]["rank"] <= 8]
        st.dataframe(top_t[["topic", "keyword", "weight"]],
                     use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Top Words per Topic — Bar Charts")
    col5, col6 = st.columns(2)
    with col5:
        st.pyplot(fig_topic_grid(lda_j["topic_df"], NUM_TOPICS,
                                 COLOR_J, "Jaishankar — LDA Topics"),
                  use_container_width=True)
    with col6:
        st.pyplot(fig_topic_grid(lda_t["topic_df"], NUM_TOPICS,
                                 COLOR_T, "Tharoor — LDA Topics"),
                  use_container_width=True)

    st.markdown("---")
    st.subheader("Document–Topic Heatmap  (average probability)")
    st.pyplot(fig_heatmap(lda_j["doc_topic"], COLOR_J, "Jaishankar"),
              use_container_width=True)
    st.pyplot(fig_heatmap(lda_t["doc_topic"], COLOR_T, "Tharoor"),
              use_container_width=True)

    st.markdown("---")
    st.subheader("Dominant Topic Distribution")
    col7, col8 = st.columns(2)
    with col7:
        st.pyplot(fig_pie(lda_j["doc_topic"], COLOR_J, "Jaishankar"),
                  use_container_width=True)
        st.pyplot(fig_topic_share(lda_j["doc_topic"], COLOR_J, "Jaishankar"),
                  use_container_width=True)
    with col8:
        st.pyplot(fig_pie(lda_t["doc_topic"], COLOR_T, "Tharoor"),
                  use_container_width=True)
        st.pyplot(fig_topic_share(lda_t["doc_topic"], COLOR_T, "Tharoor"),
                  use_container_width=True)

    # pyLDAvis
    if PYLDAVIS_OK:
        st.markdown("---")
        st.subheader("🔬 Interactive pyLDAvis Explorer")
        sel = st.radio("Speaker", ["Jaishankar", "Tharoor"],
                       horizontal=True, key="ldavis_sel")
        tup = j_tokens if sel == "Jaishankar" else t_tokens
        with st.spinner("Preparing pyLDAvis…"):
            html = ldavis_html(tup, NUM_TOPICS)
        if html:
            components.html(html, height=860, scrolling=True)
    else:
        st.info("Install `pyLDAvis` to enable the interactive explorer.")

# ─────────────────────────────────────────────────────────────
# TAB: NMF
# ─────────────────────────────────────────────────────────────
def tab_nmf(nmf_j: Dict, nmf_t: Dict) -> None:
    st.markdown('<div class="sec-title">🔶 NMF Topic Modelling</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("J — Reconstruction Error",
              f"{nmf_j['reconstruction_error']:.4f}")
    m2.metric("J — TF-IDF Features",
              f"{nmf_j['dtm'].shape[1]}")
    m3.metric("T — Reconstruction Error",
              f"{nmf_t['reconstruction_error']:.4f}")
    m4.metric("T — TF-IDF Features",
              f"{nmf_t['dtm'].shape[1]}")

    st.markdown("---")
    st.subheader("NMF Topic Keywords")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**<span style='color:{COLOR_J}'>Jaishankar</span>**",
                    unsafe_allow_html=True)
        st.dataframe(
            nmf_j["topic_df"][nmf_j["topic_df"]["rank"] <= 8][["topic","keyword","weight"]],
            use_container_width=True, hide_index=True)
    with col2:
        st.markdown(f"**<span style='color:{COLOR_T}'>Tharoor</span>**",
                    unsafe_allow_html=True)
        st.dataframe(
            nmf_t["topic_df"][nmf_t["topic_df"]["rank"] <= 8][["topic","keyword","weight"]],
            use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Top Words per Topic — Bar Charts (NMF)")
    col3, col4 = st.columns(2)
    with col3:
        st.pyplot(fig_topic_grid(nmf_j["topic_df"], NUM_TOPICS,
                                 "#FF6B6B", "Jaishankar — NMF Topics"),
                  use_container_width=True)
    with col4:
        st.pyplot(fig_topic_grid(nmf_t["topic_df"], NUM_TOPICS,
                                 "#457B9D", "Tharoor — NMF Topics"),
                  use_container_width=True)

# ─────────────────────────────────────────────────────────────
# TAB: VENN DIAGRAM
# ─────────────────────────────────────────────────────────────
def tab_venn(lda_j: Dict, lda_t: Dict) -> None:
    st.markdown('<div class="sec-title">🔵 Venn Diagram — Topic Vocabulary Overlap</div>',
                unsafe_allow_html=True)
    st.caption(
        "Built from the top 8 keywords of each LDA topic per speaker. "
        "Each region reveals which themes are exclusive vs shared."
    )

    # Build vocabulary sets from all topics
    top_n_per_topic = st.slider("Top N words per topic to include", 5, 12, 8, key="venn_n")

    vocab_j: Set[str] = set()
    vocab_t: Set[str] = set()
    for _, row in lda_j["topic_df"][lda_j["topic_df"]["rank"] <= top_n_per_topic].iterrows():
        vocab_j.add(row["keyword"])
    for _, row in lda_t["topic_df"][lda_t["topic_df"]["rank"] <= top_n_per_topic].iterrows():
        vocab_t.add(row["keyword"])

    only_j = vocab_j - vocab_t
    shared  = vocab_j & vocab_t
    only_t  = vocab_t - vocab_j

    # Summary row
    c1, c2, c3 = st.columns(3)
    c1.metric("Jaishankar Exclusive", len(only_j))
    c2.metric("Shared Terms",         len(shared))
    c3.metric("Tharoor Exclusive",    len(only_t))

    # Venn diagram
    st.pyplot(fig_venn(vocab_j, vocab_t), use_container_width=True)

    st.markdown("---")
    # Detailed word lists
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"#### <span style='color:{COLOR_J}'>Jaishankar Only</span>",
                    unsafe_allow_html=True)
        for w in sorted(only_j):
            st.markdown(
                f"<span class='chip' style='background:rgba(230,57,70,.08);"
                f"color:{COLOR_J};border-color:rgba(230,57,70,.25)'>{w}</span>",
                unsafe_allow_html=True)

    with col2:
        st.markdown("#### <span style='color:#374151'>Shared Terms</span>",
                    unsafe_allow_html=True)
        for w in sorted(shared):
            st.markdown(
                f"<span class='chip' style='background:rgba(69,123,157,.10);"
                f"color:#1f2937;border-color:rgba(69,123,157,.3)'>{w}</span>",
                unsafe_allow_html=True)

    with col3:
        st.markdown(f"#### <span style='color:{COLOR_T}'>Tharoor Only</span>",
                    unsafe_allow_html=True)
        for w in sorted(only_t):
            st.markdown(
                f"<span class='chip' style='background:rgba(29,53,87,.08);"
                f"color:{COLOR_T};border-color:rgba(29,53,87,.25)'>{w}</span>",
                unsafe_allow_html=True)

    st.markdown("---")
    # Stacked bar visualization
    st.subheader("Vocabulary Breakdown — Stacked Bar")
    total = len(only_j) + len(shared) + len(only_t)
    fig_bar = go.Figure(data=[
        go.Bar(name=f"Jaishankar Only ({len(only_j)})",
               x=["Vocabulary"], y=[len(only_j)],
               marker_color=COLOR_J, text=str(len(only_j)),
               textposition="inside"),
        go.Bar(name=f"Shared ({len(shared)})",
               x=["Vocabulary"], y=[len(shared)],
               marker_color=ACCENT, text=str(len(shared)),
               textposition="inside"),
        go.Bar(name=f"Tharoor Only ({len(only_t)})",
               x=["Vocabulary"], y=[len(only_t)],
               marker_color=COLOR_T, text=str(len(only_t)),
               textposition="inside"),
    ])
    _chart_theme(fig_bar)
    fig_bar.update_layout(
        barmode="stack", height=340, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        xaxis_title="", yaxis_title="Number of Unique Terms",
        title=f"Total {total} unique terms across both speakers",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Thematic interpretation
    st.markdown("---")
    st.subheader("Thematic Interpretation")
    st.markdown(f"""
    <div class="icard">
        <div class="ititle" style="color:{COLOR_J}">Jaishankar — Exclusive Themes</div>
        <div class="inote">
            Jaishankar's unique vocabulary centres on <b>geopolitical strategy</b>
            (bilateral, indo-pacific, quad, autonomy), <b>economic diplomacy</b>
            (supply, chain, trade, corridor), <b>multilateral institutions</b>
            (BRICS, G20, SCO), and <b>security</b> (terrorism, maritime, defense).
            His discourse is distinctly <em>realist</em> and <em>transactional</em>,
            focused on India's strategic positioning in a multipolar world.
        </div>
    </div>
    <div class="icard">
        <div class="ititle" style="color:#374151">Shared Themes</div>
        <div class="inote">
            Both leaders converge on themes of <b>policy</b>, <b>development</b>,
            <b>cooperation</b>, <b>technology</b>, <b>diplomacy</b>, and <b>climate</b>.
            This overlap reflects India's broad national consensus on sustainable
            development, international engagement, and technological progress.
        </div>
    </div>
    <div class="icard">
        <div class="ititle" style="color:{COLOR_T}">Tharoor — Exclusive Themes</div>
        <div class="inote">
            Tharoor's unique vocabulary reflects a <b>liberal-constitutional</b>
            worldview: <b>democracy & rights</b> (constitution, secularism, citizen,
            judicial), <b>social justice</b> (caste, gender, LGBTQ, healthcare),
            <b>colonial history</b> (reparation, partition, decolonise, drain),
            and <b>cultural identity</b> (soft power, pluralism, diversity).
            His rhetoric is <em>idealistic</em> and <em>rights-based</em>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TAB: COMPARISON
# ─────────────────────────────────────────────────────────────
def tab_comparison(proc: pd.DataFrame, lda_j: Dict, lda_t: Dict) -> None:
    st.markdown('<div class="sec-title">⚖️ Comparative Analysis</div>',
                unsafe_allow_html=True)

    # Side-by-side top themes
    st.subheader("Top 3 Dominant Themes (LDA)")
    col1, col2 = st.columns(2)

    def top_themes(topic_df: pd.DataFrame, n: int = 3) -> List[str]:
        out = []
        for t in topic_df["topic"].unique()[:n]:
            words = (topic_df[topic_df["topic"] == t]
                     .sort_values("rank")["keyword"].tolist()[:5])
            out.append(", ".join(words))
        return out

    with col1:
        st.markdown(
            f"<div class='icard'><div class='ititle' style='color:{COLOR_J}'>Jaishankar</div>",
            unsafe_allow_html=True)
        for i, theme in enumerate(top_themes(lda_j["topic_df"]), 1):
            st.markdown(f"<div class='tbox'><b>Theme {i}</b><br>{theme}</div>",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            f"<div class='icard'><div class='ititle' style='color:{COLOR_T}'>Tharoor</div>",
            unsafe_allow_html=True)
        for i, theme in enumerate(top_themes(lda_t["topic_df"]), 1):
            st.markdown(f"<div class='tbox'><b>Theme {i}</b><br>{theme}</div>",
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Side-by-side topic word table
    st.markdown("---")
    st.subheader("Topic-by-Topic Word Comparison (LDA)")
    rows = []
    for tid in range(NUM_TOPICS):
        j_words = (lda_j["topic_df"][lda_j["topic_df"]["topic_id"] == tid]
                   .sort_values("rank")["keyword"].tolist()[:6])
        t_words = (lda_t["topic_df"][lda_t["topic_df"]["topic_id"] == tid]
                   .sort_values("rank")["keyword"].tolist()[:6])
        rows.append({
            "Topic": f"Topic {tid+1}",
            f"Jaishankar top words": ", ".join(j_words),
            f"Tharoor top words":    ", ".join(t_words),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Coherence comparison chart
    st.markdown("---")
    st.subheader("Coherence Score Comparison")
    fig_coh = go.Figure(data=[
        go.Bar(
            x=["Jaishankar", "Tharoor"],
            y=[lda_j["coherence"], lda_t["coherence"]],
            marker_color=[COLOR_J, COLOR_T],
            text=[f"{lda_j['coherence']:.4f}", f"{lda_t['coherence']:.4f}"],
            textposition="outside",
            textfont=dict(color="#111827", size=14),
            width=0.4,
        )
    ])
    _chart_theme(fig_coh)
    fig_coh.update_layout(
        title="LDA Coherence Score (Cv) by Speaker — Higher is Better",
        yaxis_title="Coherence (Cv)", height=400,
        shapes=[dict(type="line", x0=-0.5, x1=1.5, y0=0.5, y1=0.5,
                     line=dict(dash="dash", color="#9ca3af", width=1.5))],
        annotations=[dict(x=1.5, y=0.52, text="Cv = 0.5 (good threshold)",
                          showarrow=False, font=dict(size=11, color="#6b7280"),
                          xanchor="right")],
    )
    st.plotly_chart(fig_coh, use_container_width=True)

    # Vocabulary heatmap
    st.markdown("---")
    st.subheader("Vocabulary Divergence Heatmap (Top 25 Terms)")
    try:
        st.pyplot(fig_vocab_heatmap(proc), use_container_width=True)
    except Exception as e:
        st.warning(f"Heatmap error: {e}")

    # Full interpretation guide
    st.markdown("---")
    st.subheader("Interpretation Guide")
    st.markdown(f"""
    <table class="itable">
    <thead><tr>
        <th>Dimension</th>
        <th style="color:{COLOR_J}">Jaishankar (Ruling — BJP)</th>
        <th style="color:{COLOR_T}">Tharoor (Opposition — INC)</th>
    </tr></thead>
    <tbody>
    <tr><td><b>Primary Lens</b></td>
        <td>Realist / Strategic</td><td>Liberal / Constitutional</td></tr>
    <tr><td><b>Core Focus</b></td>
        <td>Foreign policy, bilateral ties, security</td>
        <td>Democracy, civil rights, social justice</td></tr>
    <tr><td><b>Geographic Frame</b></td>
        <td>Indo-Pacific, neighbourhood, Global South</td>
        <td>UN, colonial history, domestic polity</td></tr>
    <tr><td><b>Signature Vocabulary</b></td>
        <td>bilateral, sovereignty, Quad, BRICS, terrorism</td>
        <td>constitution, democracy, caste, reparation, rights</td></tr>
    <tr><td><b>International Frame</b></td>
        <td>Power balance, multi-alignment, strategic autonomy</td>
        <td>Human rights, decolonisation, climate finance</td></tr>
    <tr><td><b>Tone</b></td>
        <td>Pragmatic, transactional, technocratic</td>
        <td>Idealistic, rights-based, oratorical</td></tr>
    <tr><td><b>Shared Ground</b></td>
        <td colspan="2" style="text-align:center">
            Policy · Development · Climate · Technology · Cooperation
        </td></tr>
    <tr><td><b>LDA Coherence (Cv)</b></td>
        <td style="color:{COLOR_J};font-weight:700">{lda_j['coherence']:.4f}</td>
        <td style="color:{COLOR_T};font-weight:700">{lda_t['coherence']:.4f}</td></tr>
    </tbody>
    </table>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main() -> None:
    inject_styles()

    with st.spinner("Loading dataset…"):
        df   = load_data()
        proc = preprocess(df)

    j_proc = proc[proc["speaker"] == "Jaishankar"].copy()
    t_proc = proc[proc["speaker"] == "Tharoor"].copy()

    j_tokens = tuple(tuple(t) for t in j_proc["tokens"].tolist())
    t_tokens = tuple(tuple(t) for t in t_proc["tokens"].tolist())
    j_texts  = tuple(j_proc["clean_text"].tolist())
    t_texts  = tuple(t_proc["clean_text"].tolist())

    with st.spinner("Training LDA & NMF models…"):
        lda_j = run_lda(j_tokens, NUM_TOPICS)
        lda_t = run_lda(t_tokens, NUM_TOPICS)
        nmf_j = run_nmf(j_texts,  NUM_TOPICS)
        nmf_t = run_nmf(t_texts,  NUM_TOPICS)

    # ── Hero ──────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-title">🎙️ Political Rhetoric Dashboard</div>
        <div class="hero-sub">
            Comparative NLP Topic Modelling — S. Jaishankar vs Shashi Tharoor
        </div>
        <div class="hero-pills">
            <span class="pill pill-r">Ruling Party · BJP</span>
            <span class="pill pill-b">Opposition · INC</span>
            <span class="pill pill-g">LDA · NMF · Venn Diagram · 2024 Speeches</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Leader Showcase ──────────────────────────────────
    img_note = ""
    has_j = any(p.exists() for p in IMG_CANDIDATES["Jaishankar"])
    has_t = any(p.exists() for p in IMG_CANDIDATES["Tharoor"])
    if not has_j or not has_t:
        missing = []
        if not has_j: missing.append("jaishankar1.jpg, jaishankar2.jpg")
        if not has_t: missing.append("tharoor1.jpg, tharoor2.jpg")
        st.info(
            f"💡 To display leader photos, save these files in the same folder as app.py:  "
            f"{' | '.join(missing)}"
        )

    lc1, lc2, lc3 = st.columns([5, 2, 5])
    with lc1:
        leader_showcase("Jaishankar", "S. Jaishankar",
                        "External Affairs Minister · BJP · 2019–present",
                        "Bharatiya Janata Party (BJP)",
                        COLOR_J, df)
    with lc2:
        total = len(df)
        jn    = len(df[df["speaker"] == "Jaishankar"])
        tn    = len(df[df["speaker"] == "Tharoor"])
        min_d = df["publishedAt"].min().strftime("%b %Y")
        max_d = df["publishedAt"].max().strftime("%b %Y")
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;
                    justify-content:center;height:100%;gap:1rem;padding:1rem 0">
            <div style="text-align:center">
                <div style="font-size:2.8rem;font-weight:900;color:#111827">{total}</div>
                <div style="font-size:.76rem;color:#6b7280;font-weight:600;letter-spacing:.5px;text-transform:uppercase">Total Articles</div>
            </div>
            <div style="width:2px;height:40px;background:linear-gradient(to bottom,{COLOR_J},{COLOR_T})"></div>
            <div style="text-align:center;">
                <div style="font-size:1.6rem;font-weight:800;color:{COLOR_J}">{jn}</div>
                <div style="font-size:.7rem;color:#9ca3af;">Jaishankar</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.2rem;font-weight:700;color:#9ca3af">VS</div>
            </div>
            <div style="text-align:center;">
                <div style="font-size:1.6rem;font-weight:800;color:{COLOR_T}">{tn}</div>
                <div style="font-size:.7rem;color:#9ca3af;">Tharoor</div>
            </div>
            <div style="width:2px;height:40px;background:linear-gradient(to bottom,{COLOR_J},{COLOR_T})"></div>
            <div style="text-align:center;">
                <div style="font-size:.88rem;font-weight:700;color:#374151">{min_d}</div>
                <div style="font-size:.65rem;color:#9ca3af;">to {max_d}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with lc3:
        leader_showcase("Tharoor", "Shashi Tharoor",
                        "MP, Lok Sabha · INC · Former UN Diplomat · Author",
                        "Indian National Congress (INC)",
                        COLOR_T, df)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────
    (t_ov, t_eda, t_wc, t_lda,
     t_nmf, t_venn, t_comp) = st.tabs([
        "📋 Overview",
        "📊 EDA",
        "☁️ Word Clouds",
        "🔷 LDA Topics",
        "🔶 NMF Topics",
        "🔵 Venn Diagram",
        "⚖️ Comparison",
    ])

    with t_ov:   tab_overview(df)
    with t_eda:  tab_eda(df)
    with t_wc:   tab_wordclouds(lda_j, lda_t)
    with t_lda:  tab_lda(lda_j, lda_t, j_tokens, t_tokens)
    with t_nmf:  tab_nmf(nmf_j, nmf_t)
    with t_venn: tab_venn(lda_j, lda_t)
    with t_comp: tab_comparison(proc, lda_j, lda_t)


if __name__ == "__main__":
    main()
