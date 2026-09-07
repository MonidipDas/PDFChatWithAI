import asyncio
import json
import os
import sys
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pdfchat.config import configure_api_key, validate_api_key
from pdfchat.embeddings import create_vector_store
from pdfchat.pdf_processing import extract_text_from_pdf
from pdfchat.qa import get_answer
from pdfchat.memory import load_memory, clear_memory
from evals import (
    TEST_QUESTIONS,
    SAMPLE_DOCUMENT_TEXT,
    METRIC_KEYS,
    evaluate_single_question,
    evaluate_live_query,
    aggregate_results,
    keyword_fact_check,
    llm_judge,
    save_results,
)
from pdfchat.qa import format_context


def ensure_event_loop() -> None:
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def apply_custom_css():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to bottom right, #1e1e2f, #252542);
            color: #ffffff;
            font-family: 'Inter', sans-serif;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3b3b58;
            color: #fca311 !important;
        }
        .stButton>button {
            background-color: #fca311;
            color: #14213d;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #e5980b;
            transform: scale(1.05);
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            border: 1px solid #4a4a6a;
            background-color: #2e2e48;
            color: #ffffff;
        }
        h1, h2, h3 {
            color: #fca311;
        }
        </style>
    """, unsafe_allow_html=True)


def _init_live_evals():
    """Initialize the live evaluations list in session state."""
    if "live_evals" not in st.session_state:
        st.session_state["live_evals"] = []


def tab_chat(retriever):
    st.subheader("Chat with PDF")
    _init_live_evals()

    question = st.text_input("Ask a question about the PDF:", key="chat_input")
    if question:
        # --- Retrieve + Answer (measure latency) ---
        start_time = time.perf_counter()

        docs = retriever.invoke(question)
        context = format_context(docs)
        answer = get_answer(question, retriever)

        latency = round(time.perf_counter() - start_time, 2)

        st.markdown(f"**You:** {question}")
        st.info(f"**AI:** {answer}")
        st.caption(f"Response time: {latency:.2f}s")

        # --- Run LLM-judge evaluation in the background ---
        with st.spinner("Evaluating answer quality..."):
            eval_result = evaluate_live_query(question, answer, context, latency)

        # Store in session state
        st.session_state["live_evals"].append(eval_result)

        # Show inline quality indicator
        judge = eval_result["llm_judge"]
        if judge.get("reasoning") and judge["reasoning"] != "LLM judge call failed":
            avg_score = sum(judge.get(k, 0) for k in METRIC_KEYS) / len(METRIC_KEYS)
            if avg_score >= 4:
                st.success(f"Quality Score: {avg_score:.1f}/5  --  {judge.get('reasoning', '')}")
            elif avg_score >= 3:
                st.warning(f"Quality Score: {avg_score:.1f}/5  --  {judge.get('reasoning', '')}")
            else:
                st.error(f"Quality Score: {avg_score:.1f}/5  --  {judge.get('reasoning', '')}")
        else:
            st.warning("Could not evaluate answer quality.")


def tab_memory():
    st.subheader("🧠 Agentic Memory (Context)")
    st.write("This tab displays the conversation history that the AI is actively using to retain context.")
    history = load_memory()
    
    if not history:
        st.write("No memory stored yet.")
    else:
        for idx, item in enumerate(history):
            with st.expander(f"Interaction {idx + 1}: {item['question']}", expanded=False):
                st.write("**Answer:**")
                st.write(item['answer'])
                
    if st.button("Clear Memory"):
        clear_memory()
        st.success("Memory cleared!")
        st.rerun()


def _load_cached_results():
    """Load previously saved eval results from JSON if available."""
    filepath = os.path.join(os.path.dirname(__file__), "eval_results.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("per_question", []), data.get("summary", {})
    return None, None


def _render_eval_charts(results, agg):
    """Render the evaluation charts and detailed results."""

    # ── Top-level KPI metrics ──────────────────────────────────────────
    st.markdown("---")
    kpi_cols = st.columns(4)
    overall_scores = [v for v in agg["avg_metrics"].values() if v > 0]
    overall = round(sum(overall_scores) / len(overall_scores), 2) if overall_scores else 0

    kpi_cols[0].metric("Overall Quality", f"{overall}/5")
    kpi_cols[1].metric("Keyword Fact Score", f"{agg['avg_keyword_fact_score']:.0%}")
    kpi_cols[2].metric("Avg Latency", f"{agg['avg_latency_seconds']:.2f}s")
    kpi_cols[3].metric("Questions Evaluated", agg["total_questions"])

    st.markdown("---")

    # ── Row 1: Radar chart + Pass-rate bar chart ──────────────────────
    col_radar, col_pass = st.columns(2)

    with col_radar:
        st.markdown("##### LLM-Judge Average Scores")
        labels = [k.replace("_", " ").title() for k in METRIC_KEYS]
        values = [agg["avg_metrics"].get(k, 0) for k in METRIC_KEYS]
        # Close the radar polygon
        labels_closed = labels + [labels[0]]
        values_closed = values + [values[0]]

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(252, 163, 17, 0.25)",
            line=dict(color="#fca311", width=2),
            marker=dict(size=6, color="#fca311"),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(color="#aaa")),
                angularaxis=dict(tickfont=dict(color="white")),
            ),
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=60, r=60, t=30, b=30),
            height=350,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_pass:
        st.markdown("##### Pass Rate by Metric (score >= 4/5)")
        pass_df = pd.DataFrame({
            "Metric": [k.replace("_", " ").title() for k in METRIC_KEYS],
            "Pass Rate (%)": [agg["pass_rates_pct"].get(k, 0) for k in METRIC_KEYS],
        })
        fig_pass = px.bar(
            pass_df, x="Metric", y="Pass Rate (%)",
            color="Pass Rate (%)",
            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
            range_color=[0, 100],
            text_auto=".0f",
        )
        fig_pass.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(range=[0, 110]),
            margin=dict(l=40, r=20, t=30, b=30),
            height=350,
        )
        st.plotly_chart(fig_pass, use_container_width=True)

    # ── Row 2: Per-question scores heatmap + Latency chart ────────────
    col_heat, col_lat = st.columns(2)

    with col_heat:
        st.markdown("##### Per-Question LLM-Judge Scores")
        q_labels = [f"Q{i+1}" for i in range(len(results))]
        metric_labels = [k.replace("_", " ").title() for k in METRIC_KEYS]
        z_data = []
        for r in results:
            row = [r["llm_judge"].get(k, 0) for k in METRIC_KEYS]
            z_data.append(row)

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_data,
            x=metric_labels,
            y=q_labels,
            colorscale=[[0, "#2c2c54"], [0.5, "#f39c12"], [1, "#2ecc71"]],
            zmin=0, zmax=5,
            text=[[str(v) for v in row] for row in z_data],
            texttemplate="%{text}",
            textfont=dict(size=13, color="white"),
            showscale=True,
            colorbar=dict(title="Score", tickvals=[0, 1, 2, 3, 4, 5]),
        ))
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=40, r=20, t=30, b=30),
            height=350,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_lat:
        st.markdown("##### Response Latency per Question")
        lat_df = pd.DataFrame({
            "Question": [f"Q{i+1}" for i in range(len(results))],
            "Latency (s)": [r["latency_seconds"] for r in results],
        })
        fig_lat = px.bar(
            lat_df, x="Question", y="Latency (s)",
            color="Latency (s)",
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            text_auto=".2f",
        )
        fig_lat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=40, r=20, t=30, b=30),
            height=350,
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    # ── Detailed per-question breakdown ───────────────────────────────
    st.markdown("---")
    st.markdown("##### Detailed Per-Question Results")

    for idx, r in enumerate(results):
        kw = r["keyword_check"]
        judge = r["llm_judge"]
        kw_pct = f"{kw['score']:.0%}"
        if kw.get("is_unanswerable"):
            kw_label = "Hedging detected" if kw.get("hedging_detected") else "No hedging"
        else:
            kw_label = f"{kw_pct} ({', '.join(kw.get('matched', []))} matched)"

        with st.expander(f"Q{idx+1} [{r['category']}]: {r['question']}", expanded=False):
            st.markdown(f"**Answer:** {r['answer']}")
            st.markdown(f"**Latency:** {r['latency_seconds']:.2f}s")
            st.markdown(f"**Keyword Check:** {kw_label}")

            if judge.get("reasoning") and judge["reasoning"] != "LLM judge call failed":
                score_parts = [f"**{k.replace('_', ' ').title()}:** {judge[k]}/5" for k in METRIC_KEYS]
                st.markdown(" | ".join(score_parts))
                st.caption(f"Reasoning: {judge['reasoning']}")
            else:
                st.warning("LLM judge evaluation failed for this question.")


def _render_live_query_charts(live_results):
    """Render charts specifically for live user query evaluations."""

    # --- KPI row ---
    st.markdown("---")
    total = len(live_results)
    all_scores = []
    for r in live_results:
        j = r["llm_judge"]
        scores = [j.get(k, 0) for k in METRIC_KEYS if j.get(k, 0) > 0]
        if scores:
            all_scores.append(sum(scores) / len(scores))

    avg_quality = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0
    avg_latency = round(sum(r["latency_seconds"] for r in live_results) / total, 2)
    high_quality = sum(1 for s in all_scores if s >= 4)

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Queries Evaluated", total)
    kpi_cols[1].metric("Avg Quality", f"{avg_quality:.2f}/5")
    kpi_cols[2].metric("Avg Latency", f"{avg_latency:.2f}s")
    kpi_cols[3].metric("High Quality (>=4/5)", f"{high_quality}/{total}")

    st.markdown("---")

    # --- Row 1: Quality trend + Metric breakdown ---
    col_trend, col_breakdown = st.columns(2)

    with col_trend:
        st.markdown("##### Quality Score Trend")
        trend_data = []
        for i, r in enumerate(live_results):
            j = r["llm_judge"]
            scores = [j.get(k, 0) for k in METRIC_KEYS if j.get(k, 0) > 0]
            avg = round(sum(scores) / len(scores), 2) if scores else 0
            trend_data.append({"Query #": i + 1, "Quality Score": avg})

        trend_df = pd.DataFrame(trend_data)
        fig_trend = px.line(
            trend_df, x="Query #", y="Quality Score",
            markers=True,
            color_discrete_sequence=["#fca311"],
        )
        fig_trend.add_hline(y=4, line_dash="dash", line_color="#2ecc71",
                            annotation_text="Pass threshold")
        fig_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            yaxis=dict(range=[0, 5.5]),
            margin=dict(l=40, r=20, t=30, b=30),
            height=350,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_breakdown:
        st.markdown("##### Average Score by Metric")
        metric_avgs = {}
        for k in METRIC_KEYS:
            vals = [r["llm_judge"].get(k, 0) for r in live_results if r["llm_judge"].get(k, 0) > 0]
            metric_avgs[k] = round(sum(vals) / len(vals), 2) if vals else 0

        breakdown_df = pd.DataFrame({
            "Metric": [k.replace("_", " ").title() for k in METRIC_KEYS],
            "Avg Score": [metric_avgs[k] for k in METRIC_KEYS],
        })
        fig_breakdown = px.bar(
            breakdown_df, x="Metric", y="Avg Score",
            color="Avg Score",
            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
            range_color=[0, 5],
            text_auto=".2f",
        )
        fig_breakdown.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(range=[0, 5.5]),
            margin=dict(l=40, r=20, t=30, b=30),
            height=350,
        )
        st.plotly_chart(fig_breakdown, use_container_width=True)

    # --- Row 2: Latency trend + Per-query heatmap ---
    col_lat, col_heat = st.columns(2)

    with col_lat:
        st.markdown("##### Latency Trend")
        lat_trend = pd.DataFrame({
            "Query #": [i + 1 for i in range(total)],
            "Latency (s)": [r["latency_seconds"] for r in live_results],
        })
        fig_lat_trend = px.bar(
            lat_trend, x="Query #", y="Latency (s)",
            color="Latency (s)",
            color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
            text_auto=".2f",
        )
        fig_lat_trend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            coloraxis_showscale=False,
            margin=dict(l=40, r=20, t=30, b=30),
            height=350,
        )
        st.plotly_chart(fig_lat_trend, use_container_width=True)

    with col_heat:
        st.markdown("##### Per-Query Score Breakdown")
        q_labels = [f"Q{i+1}" for i in range(total)]
        metric_labels = [k.replace("_", " ").title() for k in METRIC_KEYS]
        z_data = [[r["llm_judge"].get(k, 0) for k in METRIC_KEYS] for r in live_results]

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_data, x=metric_labels, y=q_labels,
            colorscale=[[0, "#2c2c54"], [0.5, "#f39c12"], [1, "#2ecc71"]],
            zmin=0, zmax=5,
            text=[[str(v) for v in row] for row in z_data],
            texttemplate="%{text}",
            textfont=dict(size=13, color="white"),
            showscale=True,
            colorbar=dict(title="Score", tickvals=[0, 1, 2, 3, 4, 5]),
        ))
        fig_heat.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            margin=dict(l=40, r=20, t=30, b=30),
            height=350,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # --- Detailed per-query breakdown ---
    st.markdown("---")
    st.markdown("##### Query History")
    for idx, r in enumerate(reversed(live_results)):
        q_num = total - idx
        judge = r["llm_judge"]
        scores = [judge.get(k, 0) for k in METRIC_KEYS if judge.get(k, 0) > 0]
        avg = round(sum(scores) / len(scores), 2) if scores else 0
        ts = r.get("timestamp", "")

        with st.expander(f"Q{q_num}: {r['question'][:80]}  |  Score: {avg}/5  |  {r['latency_seconds']:.2f}s", expanded=False):
            st.markdown(f"**Answer:** {r['answer']}")
            st.markdown(f"**Latency:** {r['latency_seconds']:.2f}s")
            if ts:
                st.caption(f"Evaluated at: {ts}")
            if judge.get("reasoning") and judge["reasoning"] != "LLM judge call failed":
                score_parts = [f"**{k.replace('_', ' ').title()}:** {judge[k]}/5" for k in METRIC_KEYS]
                st.markdown(" | ".join(score_parts))
                st.caption(f"Reasoning: {judge['reasoning']}")
            else:
                st.warning("LLM judge evaluation failed for this query.")


def tab_dashboard():
    st.subheader("LLM Evaluation Dashboard")
    _init_live_evals()

    live_results = st.session_state.get("live_evals", [])

    # --- Section selector ---
    section = st.radio(
        "View",
        ["Live User Queries", "Sample Benchmark"],
        horizontal=True,
        help="'Live User Queries' shows evaluations of real questions asked in the Chat tab. "
             "'Sample Benchmark' runs a fixed set of test questions.",
    )

    if section == "Live User Queries":
        st.write("Quality metrics for real user queries, evaluated live by an LLM judge.")

        if live_results:
            if st.button("Clear Live Evaluation History"):
                st.session_state["live_evals"] = []
                st.rerun()
            _render_live_query_charts(live_results)
        else:
            st.info(
                "No live evaluations yet. Go to the **Chat** tab and ask questions -- "
                "each answer will be automatically evaluated and shown here."
            )

    else:  # Sample Benchmark
        st.write("Run a benchmark evaluation of the RAG pipeline on predefined sample questions.")

        cached_results, cached_agg = _load_cached_results()

        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            run_eval = st.button("Run Sample Benchmark", type="primary")

        if run_eval:
            with st.status("Running LLM Evaluation...", expanded=True) as status:
                st.write("Building hybrid retriever + cross-encoder reranker...")
                retriever = create_vector_store(SAMPLE_DOCUMENT_TEXT)

                results = []
                progress_bar = st.progress(0, text="Evaluating questions...")

                for idx, test in enumerate(TEST_QUESTIONS):
                    st.write(f"Evaluating Q{idx+1}/{len(TEST_QUESTIONS)}: {test['question'][:60]}...")
                    result = evaluate_single_question(test, retriever)
                    results.append(result)
                    progress_bar.progress(
                        (idx + 1) / len(TEST_QUESTIONS),
                        text=f"Evaluated {idx+1}/{len(TEST_QUESTIONS)} questions",
                    )

                agg = aggregate_results(results)
                save_results(results, agg)
                status.update(label="Evaluation complete!", state="complete", expanded=False)

            st.session_state["eval_results"] = results
            st.session_state["eval_agg"] = agg

        display_results = st.session_state.get("eval_results")
        display_agg = st.session_state.get("eval_agg")

        if display_results is None and cached_results is not None:
            display_results = cached_results
            display_agg = cached_agg
            with col_status:
                st.info("Showing previously saved results. Click 'Run Sample Benchmark' for fresh results.")

        if display_results and display_agg:
            _render_eval_charts(display_results, display_agg)
        elif not run_eval:
            st.info("No benchmark results found. Click 'Run Sample Benchmark' to start.")


def main() -> None:
    ensure_event_loop()
    st.set_page_config(page_title="Agentic PDF Chat", layout="wide", initial_sidebar_state="expanded")
    apply_custom_css()
    
    st.title("📚 Agentic PDF Explorer")

    with st.sidebar:
        st.header("⚙️ Configuration")
        try:
            configure_api_key()
            validate_api_key()
            st.success("✅ Groq API Authenticated")
        except Exception as exc:
            st.error(f"❌ API Error: {exc}")
            return
            
        st.divider()
        st.subheader("📄 Document Upload")
        uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_pdf is None:
        st.info("👈 Please upload a PDF in the sidebar to begin.")
        return

    with st.spinner("Processing PDF with Hybrid Search and Reranking..."):
        try:
            # We would ideally cache this retriever creation in Streamlit session state
            # but for simplicity we keep it here.
            text = extract_text_from_pdf(uploaded_pdf)
            retriever = create_vector_store(text)
        except Exception as exc:
            st.error(f"Error processing PDF: {exc}")
            return

    # Create Tabs
    tab1, tab2, tab3 = st.tabs(["Chat", "Agentic Memory", "LLM Evaluation"])
    
    with tab1:
        tab_chat(retriever)
        
    with tab2:
        tab_memory()
        
    with tab3:
        tab_dashboard()


if __name__ == "__main__":
    main()
