"""Streamlit presentation app for the SubRep quarter-plan demo.

Run from the repository root:

    streamlit run demo/streamlit_subrep_demo.py

The app is a visual layer over real pipeline artifacts. It can run
``demo.run_full_pipeline`` from the sidebar, then reads the generated admission
report, SkillLibrary JSON, MeTTa certificates, and trained/stub MDN outputs.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Box low's precision lowered by casting to float32.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Box high's precision lowered by casting to float32.*",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo.run_full_pipeline import run_pipeline
from utils.subrep_demo_data import (
    CERTIFICATE_PATH,
    LIBRARY_PATH,
    MDN_CHECKPOINT_PATH,
    REPORT_PATH,
    build_failed_skill_rejection_probe,
    build_mdn_selection_trace,
    load_demo_artifacts,
    support_geometry_feasible,
)


def main() -> None:
    st.set_page_config(
        page_title="SubRep Demo",
        page_icon="SR",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    with st.sidebar:
        st.title("SubRep Demo")
        st.divider()
        st.write("**Artifacts**")
        st.code("\n".join([str(REPORT_PATH), str(LIBRARY_PATH), str(CERTIFICATE_PATH)]))
        run_clicked = st.button("Run / Re-run Pipeline", type="primary", width="stretch")
        refresh_clicked = st.button("Refresh Artifacts", width="stretch")
        play_clicked = st.button("Play Story Animation", width="stretch")
        st.caption("The app reads real outputs. Training is not run live.")

    if run_clicked:
        _run_pipeline_in_app()

    if refresh_clicked:
        st.cache_data.clear()
        st.rerun()

    artifacts = _cached_artifacts()
    report = artifacts.report
    selection_trace = _cached_selection_trace()

    _hero(report, artifacts.store_synced)
    _animated_story(play_clicked)
    _real_rollout_demo()
    _proof_snapshot(artifacts, report, selection_trace)
    _failed_skill_gate_demo()
    _admission_and_library(artifacts, report)
    _mdn_and_zero_shot(report, selection_trace)
    _motive_shift_explorer(artifacts.skill_rows)
    _audit_artifacts(selection_trace)


@st.cache_data(show_spinner=False)
def _cached_artifacts():
    return load_demo_artifacts()


@st.cache_data(show_spinner=False)
def _cached_selection_trace() -> dict[str, Any]:
    return build_mdn_selection_trace()


def _run_pipeline_in_app() -> None:
    st.info("Running the real SubRep pipeline. This may take a few seconds.")
    buffer = io.StringIO()
    with st.spinner("Executing skills, certifying, storing certificates, and running MDN selection..."):
        with contextlib.redirect_stdout(buffer):
            stats = run_pipeline()
    st.cache_data.clear()
    st.success(
        f"Pipeline complete: {stats['admitted']} admitted, "
        f"{stats['rejected']} rejected, library size {stats['library_size']}."
    )
    with st.expander("Pipeline log"):
        st.code(buffer.getvalue()[-8000:])


def _hero(report: dict[str, Any], store_synced: bool) -> None:
    source = report.get("mdn_source", "not loaded")
    source_label = "trained MDN" if source == "trained_checkpoint" else str(source)
    sync_label = "store/library synced" if store_synced else "store/library needs check"

    st.markdown(
        f"""
        <div class="hero">
          <div>
            <div class="eyebrow">MO-LunarLander Prototype</div>
            <h1>SubRep certifies skills before reuse.</h1>
            <p>
              This demo shows the full story in one place:
              execute candidate skills, compute improvements, pass/fail CDS/PDS,
              store certificates, and reuse admitted skills with {source_label}.
            </p>
          </div>
          <div class="hero-badges">
            <span>{_esc(source_label)}</span>
            <span>{_esc(sync_label)}</span>
            <span>no retraining during reuse</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _animated_story(play_clicked: bool) -> None:
    st.subheader("1. Watch The Mechanism")
    cols = st.columns([1.2, 0.8], gap="large")
    with cols[0]:
        st.markdown(_lander_animation_html(), unsafe_allow_html=True)
    with cols[1]:
        placeholder = st.empty()
        progress = st.progress(0)
        steps = [
            ("Execute", "A pilot policy runs a candidate skill in MO-LunarLander."),
            ("Measure", "The pipeline computes payoff improvement and motive improvement."),
            ("Certify", "CDS/PDS gates decide if reuse is mathematically safe."),
            ("Store", "Only admitted certificates enter MeTTa and the SkillLibrary."),
            ("Select", "The MDN outputs alpha weights and support geometry."),
            ("Reuse", "The library returns only admissible skills. No retraining."),
        ]

        if play_clicked:
            for index, (title, text) in enumerate(steps, start=1):
                placeholder.markdown(_story_card(title, text, active=True), unsafe_allow_html=True)
                progress.progress(index / len(steps))
                time.sleep(0.55)
        else:
            placeholder.markdown(
                _story_card("Ready", "Click Play Story Animation in the sidebar for the video walkthrough.", active=False),
                unsafe_allow_html=True,
            )
            progress.progress(0)


def _real_rollout_demo() -> None:
    st.subheader("2. See A Real Skill Execution")
    st.caption(
        "This optional player renders an actual MO-LunarLander rollout. "
        "It uses the PPO pilot checkpoint when available and the same 2-objective reward mapping used by SubRep."
    )
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        frame_slot = st.empty()
        st.info("Click the button to play a short real rollout inside the app.")
    with right:
        st.markdown("**What this visual proves**")
        st.write("The lander movement is the skill execution side of the story.")
        st.write("The certification and reuse proof still comes from the pipeline artifacts below.")
        play_real = st.button("Play Real Lander Rollout", width="stretch")

    if not play_real:
        return

    metric_slots = st.columns(4)
    try:
        with st.spinner("Rendering a real rollout..."):
            result = _run_visual_rollout(frame_slot)
        metric_slots[0].metric("Steps", result["steps"])
        metric_slots[1].metric("Payoff", f"{result['payoff']:.2f}")
        metric_slots[2].metric("Motive 0", f"{result['motive_0']:.2f}")
        metric_slots[3].metric("Motive 1", f"{result['motive_1']:.2f}")
        st.success(
            f"Rollout finished by {result['end_reason']}. "
            "These same payoff/motive summaries are what SubRep compares against the baseline during certification."
        )
    except Exception as exc:
        st.warning(
            "Could not render the live rollout in this environment. "
            "The rest of the demo still works from saved pipeline artifacts."
        )
        st.caption(str(exc))


def _run_visual_rollout(frame_slot, *, max_steps: int = 240, frame_stride: int = 4) -> dict[str, Any]:
    from env.lunar_lander_wrapper import SubRepEnv
    from pilot.rl_pilot import RLPilot

    env = SubRepEnv(seed=42, render_mode="rgb_array")
    pilot = None
    if Path("models/pilot_ppo.pt").exists():
        pilot = RLPilot.load("models/pilot_ppo.pt", map_location="cpu")

    obs, _ = env.reset(seed=42)
    total_payoff = 0.0
    motive_totals = np.zeros(2, dtype=np.float32)
    discount = 1.0
    end_reason = "max_steps"

    try:
        first_frame = env.env.render()
        if first_frame is not None:
            frame_slot.image(first_frame, caption="Initial state", width="stretch")

        for step in range(1, max_steps + 1):
            if pilot is not None:
                action = pilot.predict(obs, deterministic=True, return_probability=False)
            else:
                action = env.env.action_space.sample()

            obs, reward_vec, terminated, truncated, info = env.step(int(action))
            reward_vec = np.asarray(reward_vec, dtype=np.float32)
            total_payoff += discount * float(np.sum(reward_vec))
            motive_totals += discount * reward_vec

            if step % frame_stride == 0 or terminated or truncated:
                frame = env.env.render()
                if frame is not None:
                    frame_slot.image(
                        frame,
                        caption=(
                            f"Step {step} | payoff {total_payoff:.2f} | "
                            f"motives [{motive_totals[0]:.2f}, {motive_totals[1]:.2f}]"
                        ),
                        width="stretch",
                    )
                    time.sleep(0.025)

            if terminated:
                end_reason = "terminated"
                if info.get("landing_success"):
                    end_reason = "successful landing"
                elif info.get("crashed"):
                    end_reason = "crash"
                break
            if truncated:
                end_reason = "truncated"
                break

            discount *= 0.99

        return {
            "steps": step,
            "payoff": float(total_payoff),
            "motive_0": float(motive_totals[0]),
            "motive_1": float(motive_totals[1]),
            "end_reason": end_reason,
        }
    finally:
        env.close()


def _proof_snapshot(artifacts, report: dict[str, Any], selection_trace: dict[str, Any]) -> None:
    st.subheader("3.Proof Snapshot")
    total = int(report.get("total_attempted", 0) or 0)
    admitted = int(report.get("admitted", 0) or 0)
    rejected = int(report.get("rejected", 0) or 0)
    library_count = int(artifacts.library.get("skill_count", len(artifacts.skill_rows)) or 0)
    failed_probe = build_failed_skill_rejection_probe()
    safety_ok = artifacts.store_synced and bool(failed_probe["blocked"])
    reuse_rate = _selection_reuse_rate(selection_trace)

    cols = st.columns(3)
    cols[0].metric("Safety Compliance", "100%" if safety_ok else "check")
    cols[1].metric("Reuse Rate", "-" if reuse_rate is None else f"{reuse_rate:.0f}%")
    cols[2].metric("Certified Skills", library_count)
    st.caption(
        f"Detailed audit: {total} attempted, {admitted} admitted, {rejected} rejected, "
        f"{artifacts.certificate_count} MeTTa certificates. The raw CDS/PDS counts are kept in the report below."
    )


def _failed_skill_gate_demo() -> None:
    st.subheader("4. Live Unsafe-Skill Block")
    probe = build_failed_skill_rejection_probe()
    left, right = st.columns([0.62, 0.38], gap="large")

    with left:
        st.markdown("**Deliberately unsafe candidate**")
        st.write(
            "This tiny live check uses the real CDS/PDS gates. "
            "The candidate worsens the worst motive more than the payoff can compensate, "
            "so it never reaches certificate storage or the SkillLibrary."
        )
        st.code(
            "\n".join(
                [
                    f"delta_r = {probe['delta_r']}",
                    f"delta_n = {probe['delta_n']}",
                    "CDS: delta_r + min(delta_n) >= 0",
                    "PDS: delta_r + min(delta_n) >= -epsilon",
                ]
            )
        )

    with right:
        st.metric("CDS Gate", "fail" if not probe["cds_pass"] else "pass")
        st.metric("PDS Gate", "fail" if not probe["pds_pass"] else "pass")
        if probe["blocked"]:
            st.success("Blocked from certificate store and library.")
        else:
            st.error("Unsafe candidate was not blocked.")
        st.caption(probe["reason"])


def _admission_and_library(artifacts, report: dict[str, Any]) -> None:
    st.subheader("5. Admission Report And Certified Library")
    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        st.markdown("**Example admitted certificate**")
        admitted = report.get("example_admitted_skill")
        if admitted:
            st.json(admitted, expanded=False)
        else:
            st.warning("No admitted skill example found. Run the pipeline first.")

        st.markdown("**Rejected-skill failure reasons**")
        reasons = report.get("failure_reasons", {}) or {}
        if reasons:
            st.table([{"reason": key, "count": value} for key, value in reasons.items()])
        else:
            st.success("No rejected skills in the latest run. The invariant still holds: rejected skills would be blocked before storage.")

    with right:
        st.markdown("**Certified SkillLibrary**")
        if artifacts.skill_rows:
            st.dataframe(
                artifacts.skill_rows,
                width="stretch",
                hide_index=True,
                column_config={
                    "skill_id": "Skill",
                    "gate_type": "Gate",
                    "region": "Region",
                    "delta_r": st.column_config.NumberColumn("Delta r", format="%.3f"),
                    "delta_n_0": st.column_config.NumberColumn("Motive 0", format="%.3f"),
                    "delta_n_1": st.column_config.NumberColumn("Motive 1", format="%.3f"),
                    "margin": st.column_config.NumberColumn("Margin", format="%.3f"),
                },
            )
        else:
            st.warning("No serialized library found. Run the pipeline first.")


def _mdn_and_zero_shot(report: dict[str, Any], selection_trace: dict[str, Any]) -> None:
    st.subheader("6. Trained MDN And Zero-Shot Reuse")
    source = selection_trace.get("mdn_source") or report.get("mdn_source", "unknown")
    support_values = report.get("support_values")
    feasible = bool(report.get("support_geometry_feasible", support_geometry_feasible(support_values)))

    cols = st.columns(4)
    cols[0].metric("MDN Source", "trained" if source == "trained_checkpoint" else source)
    cols[1].metric("Alpha", _vector_text(report.get("alpha_values")))
    cols[2].metric("Weights", _vector_text(report.get("derived_weights")))
    cols[3].metric("Support feasible", "yes" if feasible else "check")

    st.info(
        "Zero-shot here means selection changes under the current context/weights, "
        "but the skill library and trained MDN are frozen. No new policy training occurs."
    )

    decisions = selection_trace.get("decisions", [])
    if decisions:
        st.dataframe(
            decisions,
            width="stretch",
            hide_index=True,
            column_config={
                "observation": "Context",
                "selected_skill_id": "Selected Skill",
                "score": st.column_config.NumberColumn("Score", format="%.3f"),
                "weight_0": st.column_config.NumberColumn("Weight 0", format="%.3f"),
                "weight_1": st.column_config.NumberColumn("Weight 1", format="%.3f"),
                "support_0": st.column_config.NumberColumn("Support 0", format="%.3f"),
                "support_1": st.column_config.NumberColumn("Support 1", format="%.3f"),
                "admissible_count": "Admissible",
            },
        )
    else:
        st.warning(selection_trace.get("error", "No MDN selection trace available."))


def _motive_shift_explorer(skill_rows: list[dict[str, Any]]) -> None:
    st.subheader("7. Motive-Shift Explorer")
    st.caption("slide the motive priority and watch the certified-library ranking update without retraining.")
    if not skill_rows:
        st.warning("Run the pipeline first to populate the certified skill library.")
        return

    safety_weight = st.slider("Motive 0 priority", 0.0, 1.0, 0.5, 0.05)
    fuel_weight = 1.0 - safety_weight
    ranked = []
    for row in skill_rows:
        delta_r = float(row.get("delta_r") or 0.0)
        delta_n_0 = float(row.get("delta_n_0") or 0.0)
        delta_n_1 = float(row.get("delta_n_1") or 0.0)
        score = delta_r + safety_weight * delta_n_0 + fuel_weight * delta_n_1
        ranked.append(
            {
                "skill_id": row["skill_id"],
                "gate": row["gate_type"],
                "region": row["region"],
                "score": score,
                "delta_r": delta_r,
                "weighted_motive": safety_weight * delta_n_0 + fuel_weight * delta_n_1,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["skill_id"]))

    cols = st.columns([0.35, 0.65], gap="large")
    with cols[0]:
        st.metric("Motive 0", f"{safety_weight:.2f}")
        st.metric("Motive 1", f"{fuel_weight:.2f}")
        st.success(f"Top reusable skill: {ranked[0]['skill_id']}")
    with cols[1]:
        st.dataframe(
            ranked[:6],
            width="stretch",
            hide_index=True,
            column_config={
                "score": st.column_config.NumberColumn("Score", format="%.3f"),
                "delta_r": st.column_config.NumberColumn("Delta r", format="%.3f"),
                "weighted_motive": st.column_config.NumberColumn("Weighted motive", format="%.3f"),
            },
        )


def _audit_artifacts(selection_trace: dict[str, Any]) -> None:
    st.subheader("8. Reproducible Artifacts")
    cols = st.columns(4)
    cols[0].code(str(REPORT_PATH))
    cols[1].code(str(LIBRARY_PATH))
    cols[2].code(str(CERTIFICATE_PATH))
    cols[3].code(str(MDN_CHECKPOINT_PATH))

    if "error" in selection_trace:
        st.caption(f"Selection trace note: {selection_trace['error']}")


def _story_card(title: str, text: str, *, active: bool) -> str:
    css_class = "story-card active" if active else "story-card"
    return f"""
    <div class="{css_class}">
      <div class="story-title">{_esc(title)}</div>
      <div class="story-text">{_esc(text)}</div>
    </div>
    """


def _vector_text(values: Any) -> str:
    if not isinstance(values, list):
        return "-"
    return "[" + ", ".join(f"{float(v):.2f}" for v in values[:2]) + "]"


def _selection_reuse_rate(selection_trace: dict[str, Any]) -> float | None:
    decisions = selection_trace.get("decisions", [])
    if not decisions:
        return None
    selected = sum(1 for decision in decisions if decision.get("status") == "selected")
    return 100.0 * selected / len(decisions)


def _lander_animation_html() -> str:
    return """
    <style>
      body { margin:0; background:#0b1220; font-family:Inter,Arial,sans-serif; color:white; }
      .scene { position:relative; height:280px; overflow:hidden; border-radius:18px;
        background: linear-gradient(180deg, #111c33 0%, #162a42 58%, #263238 59%, #1f2933 100%);
      }
      .stars { position:absolute; inset:0; background:
        radial-gradient(circle at 16% 18%, #fff 0 1px, transparent 2px),
        radial-gradient(circle at 48% 24%, #fff 0 1px, transparent 2px),
        radial-gradient(circle at 74% 14%, #fff 0 1px, transparent 2px),
        radial-gradient(circle at 86% 34%, #fff 0 1px, transparent 2px);
        opacity:.8;
      }
      .lander { position:absolute; left:17%; top:20px; width:82px; height:58px;
        animation: descend 5s ease-in-out infinite; }
      .body { position:absolute; left:20px; top:8px; width:42px; height:34px;
        border:3px solid #e6f0ff; border-radius:10px; background:#284b63; }
      .leg { position:absolute; top:39px; width:30px; height:3px; background:#e6f0ff; }
      .leg.left { left:5px; transform:rotate(-28deg); }
      .leg.right { right:5px; transform:rotate(28deg); }
      .flame { position:absolute; left:34px; top:44px; width:14px; height:30px;
        border-radius:50%; background:#ffb703; animation:flicker .25s alternate infinite; }
      .gate { position:absolute; top:62px; left:48%; width:180px; padding:14px;
        border-radius:14px; background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.22);
        animation:pulse 2s ease-in-out infinite; }
      .gate h3 { margin:0 0 6px; font-size:18px; }
      .gate p { margin:0; color:#cbd5e1; font-size:13px; }
      .library { position:absolute; right:26px; bottom:72px; width:168px; min-height:46px;
        border-radius:14px; background:#e7f7ed; color:#14532d; padding:10px 12px; font-weight:700;
        display:flex; align-items:center; justify-content:center; text-align:center; z-index:2; }
      .skill { position:absolute; left:44%; top:155px; width:72px; height:32px;
        border-radius:999px; background:#80ed99; color:#0b3d20; text-align:center;
        line-height:32px; font-weight:800; animation:store 5s ease-in-out infinite; z-index:3; }
      .ground { position:absolute; left:0; right:0; bottom:0; height:55px; background:#111827; }
      @keyframes descend { 0%{transform:translateY(0)} 50%{transform:translateY(105px)} 100%{transform:translateY(0)} }
      @keyframes flicker { from{transform:scaleY(.72); opacity:.75} to{transform:scaleY(1.08); opacity:1} }
      @keyframes pulse { 0%,100%{box-shadow:0 0 0 rgba(128,237,153,0)} 50%{box-shadow:0 0 28px rgba(128,237,153,.35)} }
      @keyframes store { 0%,45%{opacity:0; transform:translateX(0)} 55%{opacity:1} 100%{opacity:1; transform:translateX(220px) translateY(-8px)} }
    </style>
    <div class="scene">
      <div class="stars"></div>
      <div class="lander">
        <div class="body"></div><div class="leg left"></div><div class="leg right"></div><div class="flame"></div>
      </div>
      <div class="gate"><h3>CDS/PDS Gate</h3><p>Only safe skills are certified.</p></div>
      <div class="skill">Skill</div>
      <div class="library">Certified Library</div>
      <div class="ground"></div>
    </div>
    """


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1.6rem; max-width: 1240px; }
          .hero {
            display:flex; justify-content:space-between; gap:22px; align-items:flex-end;
            padding:28px; border-radius:22px; color:white;
            background:linear-gradient(135deg,#101828 0%,#1d3557 58%,#14532d 100%);
            margin-bottom:22px;
          }
          .hero h1 { font-size: 2.4rem; margin: 0 0 .55rem; }
          .hero p { color:#dbe4f0; max-width:760px; margin:0; font-size:1.05rem; }
          .eyebrow { color:#9ee6b1; text-transform:uppercase; font-size:.78rem; letter-spacing:.08em; font-weight:800; margin-bottom:.45rem; }
          .hero-badges { display:flex; flex-direction:column; gap:8px; min-width:210px; }
          .hero-badges span {
            background:rgba(255,255,255,.13); border:1px solid rgba(255,255,255,.20);
            border-radius:999px; padding:8px 12px; font-weight:700; text-align:center;
          }
          .story-card {
            border:1px solid #d0d5dd; border-radius:16px; padding:20px;
            background:#fff; min-height:160px;
          }
          .story-card.active { border-color:#16a34a; box-shadow:0 12px 32px rgba(22,163,74,.18); }
          .story-title { font-size:1.45rem; font-weight:800; margin-bottom:8px; color:#101828; }
          .story-text { color:#475467; font-size:1.05rem; }
          @media (max-width: 900px) {
            .hero { flex-direction:column; align-items:flex-start; }
            .hero-badges { width:100%; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _esc(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
