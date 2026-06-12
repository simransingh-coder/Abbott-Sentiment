import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime
import plotly.express as px

st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="😊",
    layout="wide"
)

# ── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f5f7ff; }
.hero { text-align:center; padding: 2rem 0 1rem; }
.hero h1 { font-size: 2.6rem; font-weight: 800; color: #1E2761; margin:0; }
.hero p  { font-size: 1.1rem; color: #555; margin-top:.4rem; }
.kpi-row { display:flex; gap:1rem; margin: 1.2rem 0; }
.kpi { flex:1; border-radius:12px; padding:1.2rem; text-align:center; color:#fff; }
.kpi .num { font-size:2rem; font-weight:800; line-height:1; }
.kpi .lbl { font-size:.85rem; margin-top:.3rem; opacity:.9; }
.kpi.total  { background: linear-gradient(135deg,#1E2761,#4a5db5); }
.kpi.pos    { background: linear-gradient(135deg,#2e7d32,#66bb6a); }
.kpi.neg    { background: linear-gradient(135deg,#c62828,#ef5350); }
.kpi.neu    { background: linear-gradient(135deg,#555,#9e9e9e); }
.tip-box { background:#e8f4fd; border-left:4px solid #1565c0;
           border-radius:8px; padding:.9rem 1rem; margin:1rem 0; font-size:.92rem; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
LANGUAGES = [
    "English","Spanish","French","German","Italian","Portuguese",
    "Hindi","Bengali","Tamil","Telugu","Marathi","Urdu",
    "Chinese (Simplified)","Chinese (Traditional)","Japanese","Korean",
    "Arabic","Turkish","Russian","Dutch","Polish","Greek",
    "Swedish","Norwegian","Danish","Finnish","Hebrew",
    "Indonesian","Malay","Thai","Vietnamese","Swahili",
]

def load_file(f) -> pd.DataFrame | None:
    ext = f.name.rsplit(".", 1)[-1].lower()
    try:
        if ext in ("xlsx", "xls"):
            return pd.read_excel(f)
        for enc in ("utf-8", "utf-8-sig", "latin-1", "iso-8859-1", "cp1252"):
            try:
                f.seek(0)
                return pd.read_csv(f, encoding=enc)
            except (UnicodeDecodeError, Exception):
                continue
        st.error("Could not decode the CSV. Try saving it as UTF-8.")
    except Exception as e:
        st.error(f"Error loading file: {e}")
    return None


def build_prompt(comments: list[str], brand: str, post_type: str) -> str:
    numbered = "\n".join(f"[{i}] {str(c).strip()[:300]}" for i, c in enumerate(comments))
    n = len(comments)
    return f"""You are a professional social-media analyst. Analyze the sentiment of the {n} comments below.

CONTEXT  →  Brand: {brand} | Post type: {post_type}

━━━ MULTILINGUAL RULES ━━━
• Comments may be in ANY language (Hindi, Spanish, Arabic, French, etc.)
• Analyze each comment in its ORIGINAL language — do NOT translate first
• Sentiment labels (Positive / Negative / Neutral) are universal
• Report every "topic" in ENGLISH regardless of source language

━━━ COMMENTS ━━━
{numbered}

━━━ TASK ━━━
For every comment return:
  index      – same integer as the [index] above
  language   – detected language (e.g. "Hindi", "Spanish", "English")
  sentiment  – exactly one of: Positive | Negative | Neutral
  topic      – the single best-fit topic IN ENGLISH from this list:
               Product Quality · Price/Value · Taste/Flavor · Convenience ·
               Ingredients/Nutrition · Availability · Customer Service ·
               Packaging · Shipping · Brand Trust · Competitor Comparison ·
               General Feedback

Also produce a summary block.

━━━ OUTPUT FORMAT ━━━
Return ONLY valid JSON — no markdown fences, no prose before or after.

{{
  "comments": [
    {{"index": 0, "language": "English", "sentiment": "Positive", "topic": "Product Quality"}},
    {{"index": 1, "language": "Hindi",   "sentiment": "Negative", "topic": "Price/Value"}}
  ],
  "summary": {{
    "total_analyzed": {n},
    "positive_count": 0, "negative_count": 0, "neutral_count": 0,
    "positive_pct": 0.0,  "negative_pct": 0.0,  "neutral_pct": 0.0,
    "language_distribution": {{"English": 0, "Hindi": 0}},
    "positive_topics":  [{{"topic": "Product Quality", "count": 0}}],
    "negative_topics":  [{{"topic": "Price/Value",     "count": 0}}],
    "neutral_topics":   [{{"topic": "General Feedback","count": 0}}],
    "key_insights": ["Insight 1", "Insight 2", "Insight 3"]
  }}
}}

CRITICAL: your response must contain exactly {n} objects in the "comments" array."""


def parse_response(raw: str) -> dict | None:
    cleaned = raw.strip()
    # strip markdown fences if present
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[-1] if cleaned.count("```") >= 2 else cleaned
        cleaned = cleaned.replace("json", "", 1).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        st.error(f"JSON parse error: {e}\n\nMake sure you copied the **entire** response from Claude.")
        return None


def merge_results(df: pd.DataFrame, results: dict) -> pd.DataFrame:
    lookup = {r["index"]: r for r in results.get("comments", [])}
    out = df.copy()
    out["AI_Sentiment"] = [lookup.get(i, {}).get("sentiment", "—") for i in range(len(df))]
    out["AI_Topic"]     = [lookup.get(i, {}).get("topic",     "—") for i in range(len(df))]
    out["AI_Language"]  = [lookup.get(i, {}).get("language",  "—") for i in range(len(df))]
    return out


def show_dashboard(results: dict):
    s = results["summary"]
    total = s.get("total_analyzed", 0)
    pos   = s.get("positive_count", 0);  pos_pct = s.get("positive_pct", 0)
    neg   = s.get("negative_count", 0);  neg_pct = s.get("negative_pct", 0)
    neu   = s.get("neutral_count",  0);  neu_pct = s.get("neutral_pct",  0)

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi total"><div class="num">{total:,}</div><div class="lbl">Total Comments</div></div>
      <div class="kpi pos"><div class="num">{pos:,}</div><div class="lbl">Positive&nbsp;&nbsp;{pos_pct:.1f}%</div></div>
      <div class="kpi neg"><div class="num">{neg:,}</div><div class="lbl">Negative&nbsp;&nbsp;{neg_pct:.1f}%</div></div>
      <div class="kpi neu"><div class="num">{neu:,}</div><div class="lbl">Neutral&nbsp;&nbsp;{neu_pct:.1f}%</div></div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Sentiment split")
        fig = px.pie(
            values=[pos, neg, neu],
            names=["Positive", "Negative", "Neutral"],
            color_discrete_map={"Positive":"#4caf50","Negative":"#f44336","Neutral":"#9e9e9e"},
            hole=0.45,
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(margin=dict(t=10,b=10), height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Language distribution")
        lang_dist = s.get("language_distribution", {})
        if lang_dist:
            ldf = (pd.DataFrame(lang_dist.items(), columns=["Language","Count"])
                     .sort_values("Count", ascending=False).head(12))
            fig2 = px.bar(ldf, x="Language", y="Count",
                          color="Count", color_continuous_scale="Blues", text="Count")
            fig2.update_traces(textposition="outside")
            fig2.update_layout(margin=dict(t=10,b=10), height=340,
                               coloraxis_showscale=False, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Language distribution not returned by Claude.")

    # Topics
    st.subheader("Top topics by sentiment")
    tc1, tc2, tc3 = st.columns(3)
    def _topic_list(col, emoji, label, key):
        with col:
            st.markdown(f"**{emoji} {label}**")
            items = s.get(key, [])
            if items:
                for t in items[:6]:
                    st.markdown(f"- {t.get('topic','?')} ({t.get('count',0)})")
            else:
                st.caption("None")
    _topic_list(tc1, "✅","Positive","positive_topics")
    _topic_list(tc2, "❌","Negative","negative_topics")
    _topic_list(tc3, "⚪","Neutral", "neutral_topics")

    # Insights
    insights = s.get("key_insights", [])
    if insights:
        st.subheader("💡 Key insights")
        for i, ins in enumerate(insights, 1):
            st.markdown(f"{i}. {ins}")


# ── App layout ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>😊 Multilingual Sentiment Analysis</h1>
  <p>Analyze comments in any language — no translation needed</p>
</div>""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🌍 Supported languages")
    with st.expander("View all"):
        for lang in LANGUAGES:
            st.markdown(f"- {lang}")
    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown("""
1. Upload Excel / CSV  
2. Pick the comment column  
3. Click **Generate prompt**  
4. Paste prompt into Claude  
5. Copy Claude's JSON back  
6. Click **Process response**  
7. Explore & download!
""")

# Tabs
tab_upload, tab_results, tab_guide = st.tabs(["📤 Upload & Analyze", "📊 Results", "📖 Guide"])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("#### Step 1 — Upload your comments file")
    uploaded = st.file_uploader(
        "Excel (.xlsx / .xls) or CSV — any language, any encoding",
        type=["xlsx","xls","csv"]
    )

    if uploaded:
        df = load_file(uploaded)
        if df is not None:
            st.success(f"✅ Loaded **{len(df):,} rows** · {len(df.columns)} columns")
            with st.expander("Preview (first 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)

            st.markdown("#### Step 2 — Configure")
            col_a, col_b = st.columns(2)
            with col_a:
                text_col = st.selectbox("Comment column *", df.columns.tolist())
            with col_b:
                max_comments = st.slider("Max comments to send Claude", 50, 300, 200, 25,
                    help="Claude handles ~300 comments well in one call. For larger files, run in batches.")

            col_c, col_d = st.columns(2)
            with col_c:
                brand = st.text_input("Brand name", "Brand")
            with col_d:
                post_type = st.text_input("Post type", "All posts")

            # sample preview
            if text_col:
                samples = df[text_col].dropna().astype(str).head(4).tolist()
                st.markdown("**Sample comments from selected column:**")
                for s_text in samples:
                    st.markdown(f"> {s_text[:180]}")

            st.markdown("#### Step 3 — Generate prompt")
            if st.button("🚀 Generate Claude prompt", type="primary", use_container_width=True):
                comments = df[text_col].fillna("").astype(str).tolist()[:max_comments]
                prompt = build_prompt(comments, brand, post_type)
                st.session_state["prompt"]   = prompt
                st.session_state["src_df"]   = df
                st.session_state["text_col"] = text_col
                st.session_state["n_sent"]   = len(comments)
                st.success(f"Prompt ready — covers **{len(comments):,} comments**")

            if "prompt" in st.session_state:
                st.markdown("#### Step 4 — Copy prompt → paste in Claude → paste response back")
                with st.expander("📋 Claude prompt (click to expand & copy)", expanded=True):
                    st.code(st.session_state["prompt"], language="text")
                    st.download_button(
                        "⬇️ Download prompt as .txt",
                        data=st.session_state["prompt"],
                        file_name=f"sentiment_prompt_{datetime.now():%Y%m%d_%H%M%S}.txt",
                    )

                st.markdown("""
<div class="tip-box">
<b>📌 Next step:</b><br>
Copy the prompt above → open <a href="https://claude.ai" target="_blank">claude.ai</a>
→ paste & send → copy Claude's entire JSON reply → paste below → click <b>Process response</b>.
</div>""", unsafe_allow_html=True)

                raw_response = st.text_area(
                    "Paste Claude's JSON response here",
                    height=260,
                    placeholder='{"comments":[...],"summary":{...}}'
                )

                if st.button("✨ Process response", use_container_width=True, type="primary"):
                    if raw_response.strip():
                        results = parse_response(raw_response)
                        if results:
                            st.session_state["results"] = results
                            st.success("✅ Results processed! Open the **Results** tab.")
                            st.balloons()
                    else:
                        st.warning("Please paste Claude's response first.")

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab_results:
    if "results" not in st.session_state:
        st.info("👆 Complete the analysis in the Upload tab first.")
    else:
        results = st.session_state["results"]
        show_dashboard(results)

        st.markdown("---")
        st.subheader("📋 Detailed results")

        merged = merge_results(st.session_state["src_df"], results)

        # Filters
        f1, f2, f3 = st.columns(3)
        with f1:
            sent_opts = merged["AI_Sentiment"].unique().tolist()
            sent_filter = st.multiselect("Sentiment", sent_opts, default=sent_opts)
        with f2:
            lang_opts = merged["AI_Language"].unique().tolist()
            lang_filter = st.multiselect("Language", lang_opts, default=lang_opts)
        with f3:
            topic_opts = merged["AI_Topic"].unique().tolist()
            topic_filter = st.multiselect("Topic", topic_opts, default=topic_opts)

        view = merged[
            merged["AI_Sentiment"].isin(sent_filter) &
            merged["AI_Language"].isin(lang_filter) &
            merged["AI_Topic"].isin(topic_filter)
        ]
        st.caption(f"Showing {len(view):,} of {len(merged):,} rows")
        st.dataframe(view, use_container_width=True, height=380)

        # Downloads
        st.markdown("### ⬇️ Download results")
        d1, d2, d3 = st.columns(3)
        with d1:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as w:
                view.to_excel(w, index=False, sheet_name="Sentiment")
                s = results["summary"]
                pd.DataFrame({
                    "Metric":     ["Total","Positive","Negative","Neutral"],
                    "Count":      [s.get("total_analyzed",0), s.get("positive_count",0),
                                   s.get("negative_count",0), s.get("neutral_count",0)],
                    "Percentage": [100, s.get("positive_pct",0),
                                   s.get("negative_pct",0), s.get("neutral_pct",0)],
                }).to_excel(w, index=False, sheet_name="Summary")
            st.download_button("📥 Excel", buf.getvalue(),
                file_name=f"sentiment_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
        with d2:
            st.download_button("📥 CSV", view.to_csv(index=False),
                file_name=f"sentiment_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv", use_container_width=True)
        with d3:
            st.download_button("📥 JSON summary", json.dumps(results["summary"], indent=2),
                file_name=f"sentiment_summary_{datetime.now():%Y%m%d_%H%M%S}.json",
                mime="application/json", use_container_width=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab_guide:
    st.markdown("""
## 📖 Quick-start guide

### Supported file formats
| Format | Extension | Notes |
|--------|-----------|-------|
| Excel  | .xlsx / .xls | All sheets, multi-language cells |
| CSV    | .csv | Auto-detects encoding (UTF-8, Latin-1, etc.) |

### Multilingual support
Claude reads each comment in its original language — no pre-translation step needed.
Topics are always reported in **English** so your pivot tables stay consistent.

**Tested languages include:** English, Spanish, Hindi, Arabic, French, German,
Chinese, Japanese, Korean, Portuguese, Russian, and many more.

### Tips for best results
- **Batch size:** 150–200 comments per call gives the best accuracy/speed balance.
- **Context matters:** Filling in the brand name and post type improves topic labeling.
- **Check a sample:** After processing, spot-check 10–15 rows to verify accuracy.
- **Large files:** For 500+ comments, run two or three separate analyses and combine the Excel downloads.

### Understanding the outputs
| Column | Meaning |
|--------|---------|
| `AI_Sentiment` | Positive / Negative / Neutral |
| `AI_Topic` | Main topic discussed (in English) |
| `AI_Language` | Language Claude detected for that comment |

### Troubleshooting
| Symptom | Fix |
|---------|-----|
| "JSON parse error" | Make sure you copied **all** of Claude's response, including the outer `{}` |
| Wrong language detected | Provide more context in the Brand/Post-type fields |
| Encoding error on CSV | Re-save your file as UTF-8 in Excel (Save As → CSV UTF-8) |
| Fewer rows than expected | Increase the *Max comments* slider and re-generate the prompt |
""")
