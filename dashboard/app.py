"""Streamlit dashboard for ReturnShield AI."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.case_builder import CaseBuilder
from src.metrics import MetricsSummary, compute_metrics, format_metrics_report
from src.models import (
    AccountProfile,
    FraudType,
    OrderItem,
    PipelineResult,
    ReturnRequest,
    ShipmentEvent,
)
from src.nova_client import NovaClient
from src.orchestrator import ReturnIntegrityPipeline

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.set_page_config(
    page_title="ReturnShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_pipeline() -> ReturnIntegrityPipeline:
    return ReturnIntegrityPipeline(NovaClient())


@st.cache_resource
def get_case_builder() -> CaseBuilder:
    return CaseBuilder(NovaClient())


def generate_audio_player(text: str) -> None:
    """Generate audio using Polly and display an inline player."""
    try:
        pipeline = get_pipeline()
        audio_bytes = pipeline.nova.text_to_speech(text)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")
        else:
            st.warning("Audio generation unavailable (check AWS credentials)")
    except Exception as e:
        st.error(f"Audio generation failed: {e}")
        with st.expander("View Text"):
            st.text(text)


def load_synthetic_cases() -> tuple[list[ReturnRequest], dict[str, FraudType], list[dict]]:
    cases_path = DATA_DIR / "synthetic_cases.json"
    with open(cases_path, encoding="utf-8") as f:
        raw_cases = json.load(f)

    requests, ground_truth = [], {}
    for case in raw_cases:
        account = AccountProfile(**case["account"])
        item = OrderItem(**case["item"])
        events = [ShipmentEvent(**e) for e in case.get("shipment_events", [])]
        req = ReturnRequest(
            order_id=case["order_id"],
            account=account,
            item=item,
            return_reason=case["return_reason"],
            return_submitted_at=datetime.fromisoformat(case["return_submitted_at"]),
            order_delivered_at=datetime.fromisoformat(case["order_delivered_at"]),
            shipment_events=events,
            return_images=case.get("return_images", []),
            customer_message=case.get("customer_message"),
        )
        requests.append(req)
        ground_truth[req.return_id] = FraudType(case["ground_truth"])

    return requests, ground_truth, raw_cases


def render_header() -> None:
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem 0;">
            <h1>🛡️ ReturnShield AI</h1>
            <p style="font-size:1.2rem; color: #666;">
                Multi-Agent Fraud Detection &mdash; Powered by Amazon Nova
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_cards(summary: MetricsSummary) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{summary.precision:.1%}", help="Flagged returns that are actual fraud")
    c2.metric("Recall", f"{summary.recall:.1%}", help="Actual fraud cases detected")
    c3.metric("F1 Score", f"{summary.f1_score:.1%}", help="Harmonic mean of precision and recall")
    c4.metric("False Positive Rate", f"{summary.false_positive_rate:.1%}", help="Legitimate returns incorrectly flagged")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Mean Latency", f"{summary.mean_latency_ms:.0f} ms")
    c6.metric("Loss Prevention", f"{summary.loss_prevention_rate:.1%}")
    c7.metric("Value at Risk", f"${summary.total_value_at_risk:,.0f}")
    c8.metric("Value Protected", f"${summary.value_protected:,.0f}")


def render_distributions(summary: MetricsSummary) -> None:
    col1, col2, col3 = st.columns(3)

    with col1:
        if summary.risk_distribution:
            fig = px.pie(
                names=list(summary.risk_distribution.keys()),
                values=list(summary.risk_distribution.values()),
                title="Risk Level Distribution",
                color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c", "#c0392b"],
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if summary.action_distribution:
            fig = px.bar(
                x=list(summary.action_distribution.keys()),
                y=list(summary.action_distribution.values()),
                title="Action Distribution",
                color=list(summary.action_distribution.keys()),
                color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c", "#c0392b"],
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        if summary.fraud_type_distribution:
            fig = px.bar(
                x=list(summary.fraud_type_distribution.values()),
                y=list(summary.fraud_type_distribution.keys()),
                orientation="h",
                title="Fraud Type Distribution",
                color_discrete_sequence=["#3498db"],
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


def render_single_result(result: PipelineResult, ground_truth: FraudType | None = None) -> None:
    """Render a full pipeline result with all agent tabs."""
    req = result.intake.normalized_request
    pol = result.policy
    act = result.action

    risk_color = {
        "low": "green", "medium": "orange", "high": "red", "critical": "darkred"
    }.get(pol.risk_level.value, "gray")

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        f"**Risk Score:** <span style='color:{risk_color}'>{pol.risk_score:.3f}</span>",
        unsafe_allow_html=True,
    )
    m2.markdown(
        f"**Risk Level:** <span style='color:{risk_color}'>{pol.risk_level.value.upper()}</span>",
        unsafe_allow_html=True,
    )
    m3.markdown(f"**Fraud Type:** {pol.most_likely_fraud_type.value}")
    if ground_truth:
        is_correct = (
            (pol.most_likely_fraud_type == FraudType.LEGITIMATE)
            == (ground_truth == FraudType.LEGITIMATE)
        )
        m4.markdown(f"**Verdict:** {'Correct' if is_correct else 'Incorrect'}")
    else:
        m4.markdown(f"**Confidence:** {pol.confidence:.0%}")

    st.markdown(f"**Item:** {req.item.title} | **Price:** ${req.item.price:.2f} | "
                f"**Latency:** {result.total_processing_time_ms:.0f} ms")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Evidence", "Behavior", "Policy", "Action", "Audit",
    ])

    with tab1:
        evidence_text = result.evidence.evidence_summary or "No anomalies detected."
        st.text(evidence_text)
        if st.button("🔊 Play Evidence", key=f"ev_{result.return_id}"):
            generate_audio_player(evidence_text)
        if result.evidence.signals:
            for sig in result.evidence.signals:
                st.markdown(f"- **{sig.signal_type}**: {sig.description} (conf: {sig.confidence:.0%})")

    with tab2:
        st.text(result.behavior.behavior_summary)
        if st.button("🔊 Play Behavior", key=f"bh_{result.return_id}"):
            generate_audio_player(result.behavior.behavior_summary)
        flags = []
        if result.behavior.return_velocity_flag:
            flags.append("Return Velocity")
        if result.behavior.address_cluster_flag:
            flags.append("Address Cluster")
        if result.behavior.category_concentration_flag:
            flags.append("Category Concentration")
        if result.behavior.value_pattern_flag:
            flags.append("Value Pattern")
        if result.behavior.timing_anomaly_flag:
            flags.append("Timing Anomaly")
        if flags:
            st.warning(f"Behavioral flags: {', '.join(flags)}")

    with tab3:
        st.markdown(f"**Reasoning:** {pol.policy_reasoning}")
        if st.button("🔊 Play Reasoning", key=f"pr_{result.return_id}"):
            generate_audio_player(pol.policy_reasoning)
        if pol.fraud_type_probabilities:
            fig = px.bar(
                x=list(pol.fraud_type_probabilities.values()),
                y=list(pol.fraud_type_probabilities.keys()),
                orientation="h",
                title="Fraud Type Probabilities",
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.info(f"**Recommended Action:** {act.recommended_action.value}")
        st.markdown("**Customer Message:**")
        st.text_area(
            "msg", act.customer_message, height=100, disabled=True,
            key=f"cm_{result.return_id}", label_visibility="collapsed",
        )
        if st.button("🔊 Play Customer Message", key=f"cmsg_{result.return_id}"):
            generate_audio_player(act.customer_message)

        st.markdown("**Internal Notes:**")
        st.text_area(
            "notes", act.internal_notes, height=100, disabled=True,
            key=f"in_{result.return_id}", label_visibility="collapsed",
        )

        if act.voice_escalation_script:
            st.markdown("---")
            st.markdown("**Nova Sonic Voice Escalation Script:**")
            st.text_area(
                "script", act.voice_escalation_script, height=250, disabled=True,
                key=f"vs_{result.return_id}", label_visibility="collapsed",
            )
            if st.button("🔊 Play Voice Script", key=f"vsplay_{result.return_id}", type="primary"):
                with st.spinner("Generating audio..."):
                    generate_audio_player(act.voice_escalation_script)
            if act.escalation_reason:
                st.warning(f"Escalation Reason: {act.escalation_reason}")

    with tab5:
        st.text(result.audit.decision_explanation)
        if st.button("🔊 Play Audit Report", key=f"au_{result.return_id}"):
            generate_audio_player(result.audit.decision_explanation)


# ---------------------------------------------------------------------------
# Live Case Analysis Tab
# ---------------------------------------------------------------------------

EXAMPLE_SCENARIOS = [
    "Damaged box of iPhone 15 Pro Max, returned same day after delivery",
    "Customer returned a MacBook Air but the serial number doesn't match the one we shipped",
    "Empty box return on a PS5 - package weighs almost nothing",
    "Customer claims AirPods Pro never arrived but tracking shows handed to resident",
    "Long-time customer returning a defective Samsung tablet with dead pixels after 5 days",
    "Suspicious account with 4 addresses returned a Sony camera the day after delivery",
    "Customer bought and returned 22 headphones in the past year, now returning another pair",
    "Counterfeit swap on a Nintendo Switch - returned item looks like a cheap clone",
]


def render_live_analysis() -> None:
    st.subheader("Live Case Analysis")
    st.markdown(
        "Describe **any** return scenario in plain language. Amazon Nova will parse it "
        "into a structured case and run the full 6-agent fraud detection pipeline."
    )

    col_input, col_examples = st.columns([3, 2])

    with col_examples:
        st.markdown("**Try an example:**")
        for i, example in enumerate(EXAMPLE_SCENARIOS):
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state["live_description"] = example

    with col_input:
        description = st.text_area(
            "Describe the return scenario",
            value=st.session_state.get("live_description", ""),
            height=120,
            placeholder="e.g. Damaged box of iPhone 15 Pro, returned same day...",
        )

        uploaded_files = st.file_uploader(
            "Upload return photos (optional)",
            type=["jpg", "jpeg", "png", "gif", "webp"],
            accept_multiple_files=True,
            key="live_images",
        )

        analyze_clicked = st.button(
            "🔍 Analyze with Nova", type="primary", use_container_width=True,
            disabled=not description.strip(),
        )

    if analyze_clicked and description.strip():
        with st.spinner("Nova is parsing your scenario and running the 6-agent pipeline..."):
            builder = get_case_builder()
            pipeline = get_pipeline()

            request = builder.build_from_text(description)

            if uploaded_files:
                image_data = []
                for uf in uploaded_files:
                    ext = uf.name.rsplit(".", 1)[-1].lower()
                    fmt = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                           "gif": "gif", "webp": "webp"}.get(ext, "jpeg")
                    image_data.append((uf.getvalue(), fmt))
                request.return_image_data = image_data

            result = pipeline.process(request)
            st.session_state["live_result"] = result
            st.session_state["live_request"] = request

    if "live_result" in st.session_state:
        result = st.session_state["live_result"]
        request = st.session_state["live_request"]

        st.markdown("---")
        st.markdown(f"### Analysis: {request.item.title}")

        with st.expander("Parsed Case Data (from Nova)", expanded=False):
            st.json({
                "order_id": request.order_id,
                "item": {
                    "title": request.item.title,
                    "price": request.item.price,
                    "category": request.item.category,
                    "weight_grams": request.item.weight_grams,
                    "serial_number": request.item.serial_number,
                },
                "account": {
                    "account_id": request.account.account_id,
                    "return_rate": request.account.return_rate,
                    "account_age_days": request.account.account_age_days,
                    "previous_fraud_flags": request.account.previous_fraud_flags,
                    "linked_addresses": request.account.linked_addresses,
                },
                "return_reason": request.return_reason,
                "customer_message": request.customer_message,
            })

        render_single_result(result)

        if result.intake.initial_flags:
            st.markdown("**Initial Flags:**")
            for flag in result.intake.initial_flags:
                st.markdown(f"- {flag}")


# ---------------------------------------------------------------------------
# Benchmark Suite Tab
# ---------------------------------------------------------------------------

def render_benchmark() -> None:
    st.subheader("Benchmark Suite")
    st.markdown("Run the 8 synthetic test cases through the pipeline to measure accuracy and performance.")

    pipeline = get_pipeline()
    requests, ground_truth, raw_cases = load_synthetic_cases()

    col_run, col_upload = st.columns([1, 2])

    with col_upload:
        uploaded_files = st.file_uploader(
            "Upload return images (attach to a case below)",
            type=["jpg", "jpeg", "png", "gif", "webp"],
            accept_multiple_files=True,
            key="bench_images",
        )
        if uploaded_files and raw_cases:
            case_idx = st.selectbox(
                "Attach images to case",
                options=list(range(len(requests))),
                format_func=lambda i: raw_cases[i].get("scenario_name", f"Case {i+1}"),
                key="bench_case_sel",
            )
            if case_idx is not None:
                image_data = []
                for uf in uploaded_files:
                    ext = uf.name.rsplit(".", 1)[-1].lower()
                    fmt = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                           "gif": "gif", "webp": "webp"}.get(ext, "jpeg")
                    image_data.append((uf.getvalue(), fmt))
                requests[case_idx].return_image_data = image_data
                st.success(f"{len(uploaded_files)} image(s) attached to "
                           f"{raw_cases[case_idx].get('scenario_name', 'case')}")

    with col_run:
        if st.button("Run Full Pipeline", type="primary", use_container_width=True):
            with st.spinner("Processing all 8 return cases..."):
                results = pipeline.process_batch(requests)
                st.session_state["bench_results"] = results
                st.session_state["bench_gt"] = ground_truth
                st.session_state["bench_raw"] = raw_cases

    if "bench_results" not in st.session_state:
        st.info("Click **Run Full Pipeline** to process all test cases and see performance metrics.")
        return

    results = st.session_state["bench_results"]
    ground_truth = st.session_state["bench_gt"]
    raw_cases = st.session_state["bench_raw"]

    summary = compute_metrics(results, ground_truth)

    st.markdown("---")
    st.markdown("#### Performance KPIs")
    render_kpi_cards(summary)

    st.markdown("---")
    st.markdown("#### Distributions")
    render_distributions(summary)

    st.markdown("---")
    st.markdown("#### Individual Case Analysis")

    for i, result in enumerate(results):
        gt = ground_truth.get(result.return_id, FraudType.LEGITIMATE)
        raw = raw_cases[i] if i < len(raw_cases) else {}
        with st.expander(
            f"{raw.get('scenario_name', 'Case')} | Risk: {result.policy.risk_score:.2f} | "
            f"{result.policy.risk_level.value.upper()}",
            expanded=False,
        ):
            render_single_result(result, ground_truth=gt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    render_header()

    live_tab, bench_tab = st.tabs(["🔍 Live Case Analysis", "📊 Benchmark Suite"])

    with live_tab:
        render_live_analysis()

    with bench_tab:
        render_benchmark()


if __name__ == "__main__":
    main()
