"""
FaithCheck -- AI Answer Faithfulness Verifier
==============================================
An MVP tool that decomposes an AI-generated answer into individual claims,
verifies each claim against a source document using semantic retrieval +
cross-encoder NLI, and visualizes the result as a color-coded faithfulness
report.

Run with:  streamlit run app.py
"""

import html
import json

import streamlit as st

from verifier import CROSS_ENCODER_OPTIONS, DEFAULT_CROSS_ENCODER_LABEL, verify_answer

st.set_page_config(page_title="FaithCheck", page_icon="§", layout="wide")

# ---------------------------------------------------------------------------
# Design tokens -- dark premium theme
# ---------------------------------------------------------------------------
INK = "#F2F4F8"
INK_SOFT = "#C7CCD6"
MUTED = "#8B93A3"
PAPER = "#0B0D11"
PAPER_RAISED = "#14171D"
HAIRLINE = "#262B34"
ACCENT = "#C9A227"
ACCENT_SOFT = "#8A7020"
GREEN = "#33C08A"
GREEN_BG = "#0F241C"
RED = "#EA5A5A"
RED_BG = "#2A1416"
AMBER = "#E0A542"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif;
        color: {INK_SOFT};
    }}

    .stApp {{
        background:
            radial-gradient(circle at 15% 0%, rgba(201,162,39,0.07) 0%, rgba(201,162,39,0) 40%),
            radial-gradient(circle at 85% 100%, rgba(51,192,138,0.05) 0%, rgba(51,192,138,0) 40%),
            {PAPER};
    }}

    /* ---- Masthead ------------------------------------------------- */
    .fc-masthead {{
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-bottom: 1px solid {HAIRLINE};
        padding-bottom: 18px;
        margin-bottom: 6px;
        position: relative;
    }}
    .fc-masthead::after {{
        content: "";
        position: absolute;
        left: 0;
        bottom: -1px;
        width: 64px;
        height: 2px;
        background: linear-gradient(90deg, {ACCENT}, transparent);
    }}
    .fc-masthead h1 {{
        font-family: 'Source Serif 4', serif;
        font-weight: 700;
        font-size: 2.3em;
        color: {INK};
        margin: 0;
        letter-spacing: -0.01em;
    }}
    .fc-masthead .fc-tag {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 0.85em;
        color: {MUTED};
        text-align: right;
        max-width: 320px;
        line-height: 1.4;
    }}
    .fc-subline {{
        color: {MUTED};
        font-size: 0.95em;
        margin: 12px 0 30px 0;
    }}

    /* ---- Section labels -------------------------------------------- */
    .fc-exhibit-label {{
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        font-size: 1.05em;
        color: {ACCENT};
        margin-bottom: 2px;
        letter-spacing: 0.01em;
    }}
    .fc-exhibit-sub {{
        font-size: 0.82em;
        color: {MUTED};
        margin-bottom: 8px;
    }}

    /* ---- Form controls ----------------------------------------------- */
    .stTextArea textarea {{
        background-color: {PAPER_RAISED} !important;
        border: 1px solid {HAIRLINE} !important;
        border-radius: 6px !important;
        color: {INK} !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        box-shadow: inset 0 1px 2px rgba(0,0,0,0.35) !important;
    }}
    .stTextArea textarea::placeholder {{
        color: {MUTED} !important;
        opacity: 0.7;
    }}
    .stTextArea textarea:focus {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 0 1px {ACCENT}, 0 0 12px rgba(201,162,39,0.15) !important;
    }}
    .stTextArea label, .stSlider label, .stSelectbox label, .stCheckbox label {{
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: {INK} !important;
        font-weight: 500 !important;
        font-size: 0.9em !important;
    }}
    .stSelectbox div[data-baseweb="select"] > div {{
        background-color: {PAPER_RAISED} !important;
        border-color: {HAIRLINE} !important;
        color: {INK} !important;
    }}

    div.stButton > button, div.stFormSubmitButton > button {{
        background: linear-gradient(180deg, {ACCENT} 0%, {ACCENT_SOFT} 100%);
        color: {PAPER};
        border: 1px solid {ACCENT};
        border-radius: 6px;
        font-weight: 600;
        padding: 0.6em 1.8em;
        font-family: 'IBM Plex Sans', sans-serif;
        letter-spacing: 0.01em;
        box-shadow: 0 4px 14px rgba(201,162,39,0.2);
        transition: filter 0.15s ease, box-shadow 0.15s ease;
    }}
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {{
        filter: brightness(1.08);
        box-shadow: 0 6px 18px rgba(201,162,39,0.3);
        color: {PAPER};
        border-color: {ACCENT};
    }}

    /* ---- Expanders (advanced settings, claim rows) ------------------- */
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {{
        font-family: 'IBM Plex Sans', sans-serif !important;
        background-color: transparent !important;
        border: none !important;
        color: {INK} !important;
    }}
    [data-testid="stExpander"] {{
        border: 1px solid {HAIRLINE} !important;
        border-radius: 8px !important;
        background-color: {PAPER_RAISED} !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }}

    /* ---- Verdict stamp ------------------------------------------------ */
    .fc-stamp {{
        display: inline-block;
        border: 1px solid var(--stamp-color);
        color: var(--stamp-color);
        padding: 26px 52px;
        text-align: center;
        border-radius: 10px;
        background: linear-gradient(180deg, {PAPER_RAISED} 0%, {PAPER} 100%);
        box-shadow: 0 0 0 1px rgba(255,255,255,0.02), 0 10px 30px rgba(0,0,0,0.45);
    }}
    .fc-stamp .fc-stamp-label {{
        font-size: 0.78em;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {MUTED};
        margin-bottom: 8px;
    }}
    .fc-stamp .fc-stamp-score {{
        font-family: 'Source Serif 4', serif;
        font-weight: 700;
        font-size: 3.2em;
        line-height: 1;
        color: var(--stamp-color);
        text-shadow: 0 0 24px color-mix(in srgb, var(--stamp-color) 35%, transparent);
    }}
    .fc-stamp .fc-stamp-detail {{
        font-size: 0.82em;
        color: {MUTED};
        margin-top: 8px;
    }}

    /* ---- Highlighted answer ------------------------------------------- */
    .claim-supported {{
        background-color: {GREEN_BG};
        border-left: 3px solid {GREEN};
        padding: 6px 12px;
        border-radius: 4px;
        margin-bottom: 6px;
        display: block;
        font-size: 0.95em;
        color: {INK_SOFT};
    }}
    .claim-hallucinated {{
        background-color: {RED_BG};
        border-left: 3px solid {RED};
        padding: 6px 12px;
        border-radius: 4px;
        margin-bottom: 6px;
        display: block;
        font-size: 0.95em;
        color: {INK_SOFT};
    }}
    .fc-badge {{
        display: inline-block;
        font-size: 0.72em;
        font-weight: 600;
        padding: 2px 9px;
        border-radius: 3px;
        margin-right: 8px;
        vertical-align: middle;
        letter-spacing: 0.02em;
    }}
    .fc-badge-supported {{ background-color: {GREEN_BG}; color: {GREEN}; border: 1px solid {GREEN}; }}
    .fc-badge-hallucinated {{ background-color: {RED_BG}; color: {RED}; border: 1px solid {RED}; }}

    /* ---- Section rule --------------------------------------------- */
    .fc-rule {{
        border: none;
        border-top: 1px solid {HAIRLINE};
        margin: 32px 0 24px 0;
    }}
    .fc-section-title {{
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        font-size: 1.35em;
        color: {INK};
        margin-bottom: 4px;
    }}
    .fc-section-sub {{
        font-size: 0.85em;
        color: {MUTED};
        margin-bottom: 16px;
    }}

    /* ---- Ledger row (claim breakdown) ------------------------------ */
    .fc-ledger-row {{
        border-top: 1px solid {HAIRLINE};
        padding: 12px 2px;
    }}
    .fc-ledger-row:last-child {{
        border-bottom: 1px solid {HAIRLINE};
    }}

    /* ---- Blockquote (evidence) -- dark premium touch ---------------- */
    [data-testid="stMarkdownContainer"] blockquote {{
        border-left: 3px solid {ACCENT};
        background-color: {PAPER_RAISED};
        color: {INK_SOFT};
        padding: 8px 14px;
        border-radius: 4px;
    }}

    /* ---- Code block (JSON export) ------------------------------------ */
    .stCodeBlock, pre, code {{
        background-color: {PAPER_RAISED} !important;
        border: 1px solid {HAIRLINE} !important;
        border-radius: 6px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Masthead
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="fc-masthead">
        <h1>FaithCheck</h1>
        <div class="fc-tag">Faithfulness verification for AI-generated answers</div>
    </div>
    <div class="fc-subline">
        Every claim in the AI answer is checked against the source document and
        marked supported or flagged, with the evidence shown alongside it.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
with st.form("inputs"):
    query = st.text_area(
        "Query (optional -- the question the AI was asked)",
        height=80,
        placeholder="e.g. What is the university's late-submission policy?",
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="fc-exhibit-label">Exhibit A</div>', unsafe_allow_html=True)
        st.markdown('<div class="fc-exhibit-sub">Source document -- the ground truth reference</div>', unsafe_allow_html=True)
        source_doc = st.text_area(
            "Source document",
            height=280,
            placeholder="Paste the policy document, contract, or reference text here...",
            label_visibility="collapsed",
        )
    with col2:
        st.markdown('<div class="fc-exhibit-label">Exhibit B</div>', unsafe_allow_html=True)
        st.markdown('<div class="fc-exhibit-sub">AI answer -- the text being verified</div>', unsafe_allow_html=True)
        ai_answer = st.text_area(
            "AI answer",
            height=280,
            placeholder="Paste the AI-generated answer here...",
            label_visibility="collapsed",
        )

    with st.expander("Advanced settings", expanded=False):
        acc_col1, acc_col2 = st.columns(2)
        with acc_col1:
            model_label = st.selectbox(
                "NLI model",
                options=list(CROSS_ENCODER_OPTIONS.keys()),
                index=list(CROSS_ENCODER_OPTIONS.keys()).index(DEFAULT_CROSS_ENCODER_LABEL),
                help="DeBERTa-v3-base is noticeably more accurate at genuine entailment/"
                "contradiction judgments than MiniLM, at the cost of a larger download "
                "(~370MB) and slower inference. Use MiniLM only if you need speed over accuracy.",
            )
            chunk_size = st.slider(
                "Source chunk size (sentences per evidence unit)",
                min_value=1,
                max_value=4,
                value=2,
                help="Chunks overlap (sliding window) so evidence spanning a boundary "
                "isn't missed. Smaller chunks are more precise; larger chunks add context.",
            )
        with acc_col2:
            top_k = st.slider(
                "Evidence candidates retrieved per claim (top-K)",
                min_value=1,
                max_value=8,
                value=4,
                help="Higher K widens the search for supporting evidence in longer documents, "
                "at the cost of more NLI calls per claim (slower).",
            )
            threshold = st.slider(
                "Entailment threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help="Minimum entailment probability required to call a claim supported. "
                "Raise this to be stricter; lower it to be more lenient.",
            )
        numeric_guard = st.checkbox(
            "Enable numeric/date consistency guard",
            value=True,
            help="Flags a claim when it states a number, percentage, date, or amount that "
            "never appears anywhere in the retrieved evidence -- NLI models alone frequently "
            "miss this kind of substitution error.",
        )

    submitted = st.form_submit_button("Analyze faithfulness", type="primary")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if submitted:
    if not source_doc.strip() or not ai_answer.strip():
        st.warning("Please provide both a source document and an AI answer.")
        st.stop()

    with st.spinner("Verifying claims against the source..."):
        results, score = verify_answer(
            source_doc,
            ai_answer,
            sentences_per_chunk=chunk_size,
            top_k_evidence=top_k,
            entailment_threshold=threshold,
            cross_encoder_name=CROSS_ENCODER_OPTIONS[model_label],
            enable_numeric_guard=numeric_guard,
        )

    if not results:
        st.info("No claims could be extracted from the AI answer.")
        st.stop()

    supported_n = sum(1 for r in results if r.label == "supported")
    total_n = len(results)

    # --- Verdict stamp ---------------------------------------------------
    if score >= 80:
        stamp_color = GREEN
    elif score >= 50:
        stamp_color = AMBER
    else:
        stamp_color = RED

    st.markdown('<hr class="fc-rule" />', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="text-align:center; margin-bottom: 8px;">
            <div class="fc-stamp" style="--stamp-color:{stamp_color};">
                <div class="fc-stamp-label">Faithfulness score</div>
                <div class="fc-stamp-score">{score}%</div>
                <div class="fc-stamp-detail">{supported_n} of {total_n} claims supported</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Highlighted answer ----------------------------------------------
    st.markdown('<hr class="fc-rule" />', unsafe_allow_html=True)
    st.markdown('<div class="fc-section-title">Highlighted answer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fc-section-sub">Exhibit B, marked sentence by sentence against Exhibit A.</div>',
        unsafe_allow_html=True,
    )
    highlighted_html = ""
    for r in results:
        css_class = "claim-supported" if r.label == "supported" else "claim-hallucinated"
        badge_class = "fc-badge-supported" if r.label == "supported" else "fc-badge-hallucinated"
        badge_text = "Supported" if r.label == "supported" else "Flagged"
        highlighted_html += (
            f'<span class="{css_class}">'
            f'<span class="fc-badge {badge_class}">{badge_text}</span>'
            f"{html.escape(r.claim)}</span>"
        )
    st.markdown(highlighted_html, unsafe_allow_html=True)

    # --- Claim-by-claim breakdown -----------------------------------------
    st.markdown('<hr class="fc-rule" />', unsafe_allow_html=True)
    st.markdown('<div class="fc-section-title">Claim-by-claim breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fc-section-sub">Each claim with its verdict, confidence, and the evidence used to decide it.</div>',
        unsafe_allow_html=True,
    )
    for i, r in enumerate(results, start=1):
        badge_class = "fc-badge-supported" if r.label == "supported" else "fc-badge-hallucinated"
        badge_text = "Supported" if r.label == "supported" else "Flagged"
        preview = r.claim[:90] + ("..." if len(r.claim) > 90 else "")
        with st.expander(f"{i}.  {preview}"):
            st.markdown(
                f'<span class="fc-badge {badge_class}">{badge_text}</span>'
                f'<span style="color:{MUTED}; font-size:0.85em;">NLI label: {r.nli_label} &nbsp;&middot;&nbsp; '
                f'Confidence: {r.confidence:.0%}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Claim:** {r.claim}")
            if r.nli_label == "numeric_mismatch":
                st.markdown(
                    f"**Numeric guard triggered** -- claim states "
                    f"`{', '.join(sorted(r.numeric_flag))}` which does not appear anywhere "
                    f"in the retrieved evidence, even though the model alone judged it as entailment."
                )
            st.markdown("**Best matching evidence from Exhibit A:**")
            st.markdown(f"> {r.best_evidence}")

    # --- Raw data / export -----------------------------------------------
    st.markdown('<hr class="fc-rule" />', unsafe_allow_html=True)
    with st.expander("Export as JSON"):
        export = {
            "query": query,
            "faithfulness_score": score,
            "claims": [
                {
                    "claim": r.claim,
                    "label": r.label,
                    "nli_label": r.nli_label,
                    "confidence": r.confidence,
                    "best_evidence": r.best_evidence,
                    "numeric_mismatch": sorted(r.numeric_flag) if r.numeric_flag else None,
                }
                for r in results
            ],
        }
        st.code(json.dumps(export, indent=2), language="json")
else:
    st.markdown('<hr class="fc-rule" />', unsafe_allow_html=True)
    st.info("Fill in Exhibit A and Exhibit B above, then click **Analyze faithfulness**.")