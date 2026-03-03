from datetime import timedelta
from xml.etree import ElementTree

import altair as alt
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def parse_time(time_str: str | None) -> timedelta | None:
    """Parse a LiveSplit time string like '00:01:23.4567890' into a timedelta."""
    if not time_str or not time_str.strip():
        return None
    parts = time_str.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return None
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)

def format_time(td: timedelta | None) -> str:
    """Format a timedelta as H:MM:SS.mmm, or '-' for None."""
    if td is None:
        return "-"
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:06.3f}"


def format_time_lss(td: timedelta) -> str:
    """Format a timedelta to LiveSplit's HH:MM:SS.FFFFFFF format (7 decimal places)."""
    total_microseconds = td / timedelta(microseconds=1)

    hours = int(total_microseconds // (3600 * 1_000_000))
    minutes = int((total_microseconds % (3600 * 1_000_000)) // (60 * 1_000_000))
    seconds = int((total_microseconds % (60 * 1_000_000)) // 1_000_000)
    microseconds = int(total_microseconds % 1_000_000)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{microseconds:06d}0" # Pad to 7 decimal places
    


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def parse_lss(file_bytes: bytes) -> tuple[dict, list[dict]]:
    """Parse a .lss file and return (run_info, segments).

    run_info: dict with keys 'game', 'category', 'attempts'
    segments: list of dicts with keys 'name', 'pb_split', 'best_segment'
    """
    root = ElementTree.fromstring(file_bytes)

    game = root.findtext("GameName", default="Unknown")
    category = root.findtext("CategoryName", default="Unknown")

    attempt_count_el = root.find("AttemptCount")
    attempts = int(attempt_count_el.text) if attempt_count_el is not None and attempt_count_el.text else 0

    run_info = {"game": game, "category": category, "attempts": attempts}

    segments = []
    for seg_el in root.iter("Segment"):
        name = seg_el.findtext("Name", default="?")

        # Personal best cumulative split time
        pb_split = None
        split_times = seg_el.find("SplitTimes")
        if split_times is not None:
            for st_el in split_times.findall("SplitTime"):
                if st_el.get("name") == "Personal Best":
                    pb_split = parse_time(st_el.findtext("RealTime"))
                    break

        # Best segment time
        best_seg_el = seg_el.find("BestSegmentTime")
        best_segment = None
        if best_seg_el is not None:
            best_segment = parse_time(best_seg_el.findtext("RealTime"))

        # Segment history (per-attempt individual durations)
        history: list[dict] = []
        hist_el = seg_el.find("SegmentHistory")
        if hist_el is not None:
            for time_el in hist_el.findall("Time"):
                dur = parse_time(time_el.findtext("RealTime"))
                if dur is not None:
                    attempt_id = int(time_el.get("id", "0"))
                    history.append({"attempt_id": attempt_id, "duration": dur})

        segments.append({
            "name": name,
            "pb_split": pb_split,
            "best_segment": best_segment,
            "history": history,
        })

    return run_info, segments


# ---------------------------------------------------------------------------
# Segment duration computation
# ---------------------------------------------------------------------------

def compute_segment_durations(segments: list[dict]) -> list[timedelta | None]:
    """Compute per-segment PB durations from cumulative PB splits."""
    durations: list[timedelta | None] = []
    prev: timedelta | None = None
    for seg in segments:
        cur = seg["pb_split"]
        if cur is not None and prev is not None:
            durations.append(cur - prev)
        elif cur is not None and prev is None and len(durations) == 0:
            durations.append(cur)  # first segment
        else:
            durations.append(None)
        prev = cur
    return durations


# ---------------------------------------------------------------------------
# History data pipeline
# ---------------------------------------------------------------------------

def rank_splits_for_training(hist_df: pd.DataFrame) -> pd.DataFrame | None:
    """Return a DataFrame of segments ranked by inconsistency (most inconsistent first).

    Parameters
    ----------
    hist_df : DataFrame with columns "Segment" and "Duration (s)"

    Returns
    -------
    DataFrame with columns "Segment", "Std Dev (s)" and "CV (%)", sorted by CV descending,
    or None if there is not enough data.
    """
    grouped = hist_df.groupby("Segment")["Duration (s)"]
    stats = pd.DataFrame({
        "Std Dev (s)": grouped.std(),
        "Mean (s)": grouped.mean(),
    }).reset_index()
    stats = stats.dropna(subset=["Std Dev (s)"])
    stats["CV (%)"] = (stats["Std Dev (s)"] / stats["Mean (s)"]) * 100
    stats = stats.drop(columns=["Mean (s)"])
    stats = stats.sort_values(by="CV (%)", ascending=False)
    return stats


def build_history_df(segments: list[dict], n_attempts: int) -> pd.DataFrame:
    """Build a DataFrame of segment durations from the last *n_attempts* attempts."""
    rows: list[dict] = []
    all_ids: set[int] = set()
    for seg in segments:
        for h in seg["history"]:
            all_ids.add(h["attempt_id"])

    # Keep only the most recent n attempt IDs (consistent window across segments)
    recent_ids = sorted(all_ids)[-n_attempts:]
    recent_set = set(recent_ids)

    for seg in segments:
        for h in seg["history"]:
            if h["attempt_id"] in recent_set:
                rows.append({
                    "Segment": seg["name"],
                    "Duration (s)": h["duration"].total_seconds(),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Balanced time goal
# ---------------------------------------------------------------------------

def compute_sum_of_best(segments: list[dict]) -> timedelta | None:
    """Sum all best segment times. Returns None if any segment is missing."""
    total = timedelta()
    for seg in segments:
        if seg["best_segment"] is None:
            return None
        total += seg["best_segment"]
    return total


def compute_balanced_segments(
    segments: list[dict],
    durations: list[timedelta | None],
    hist_df: pd.DataFrame,
    aggressiveness: float,
) -> list[timedelta | None]:
    """Compute balanced goal durations using boxplot whisker ranges.

    For each segment, defines a range from best_segment (all-time) to the
    upper whisker of the recent-history boxplot.  A single ratio is applied
    uniformly across all segments.

    The aggressiveness slider scales from 0 (= current PB pace) to
    1 (= Sum of Best).

    Returns a list of balanced timedelta per segment, or None where data
    is missing.
    """
    # --- Upper whisker per segment from recent history ---
    whiskers: dict[str, float] = {}
    for seg_name, group in hist_df.groupby("Segment"):
        data = group["Duration (s)"]
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        within_fence = data[data <= upper_fence]
        whiskers[seg_name] = within_fence.max()

    # --- Per-segment ranges: (best_seconds, upper_whisker_seconds) ---
    ranges: list[tuple[float, float] | None] = []
    sob_seconds = 0.0
    pb_seconds = 0.0

    for seg, dur in zip(segments, durations):
        best = seg["best_segment"]
        if best is None or dur is None:
            ranges.append(None)
            continue
        best_s = best.total_seconds()
        upper_s = max(whiskers.get(seg["name"], dur.total_seconds()), best_s)
        ranges.append((best_s, upper_s))
        sob_seconds += best_s
        pb_seconds += dur.total_seconds()

    # --- Compute ratio_pb: the ratio where Σ goals = PB ---
    total_range = sum((r[1] - r[0]) for r in ranges if r is not None)
    ratio_pb = (pb_seconds - sob_seconds) / total_range if total_range > 0 else 0.0

    # aggressiveness 0 → PB pace (ratio_pb), 1 → SoB (ratio 0)
    actual_ratio = ratio_pb * (1.0 - aggressiveness)

    # --- Per-segment balanced goals ---
    result: list[timedelta | None] = []
    for r in ranges:
        if r is None:
            result.append(None)
        else:
            best_s, upper_s = r
            result.append(timedelta(seconds=best_s + actual_ratio * (upper_s - best_s)))
    return result


def inject_balanced_comparison(
    file_bytes: bytes,
    cumulative_splits: list[timedelta | None],
    comparison_name: str = "Balanced Goal",
) -> bytes:
    """Inject balanced goal cumulative splits into .lss XML as a custom comparison.

    Adds (or replaces) a SplitTime entry in each Segment's SplitTimes element.
    Returns the modified XML as bytes.
    """
    root = ElementTree.fromstring(file_bytes)

    for seg_el, cum in zip(root.iter("Segment"), cumulative_splits):
        split_times = seg_el.find("SplitTimes")
        if split_times is None:
            split_times = ElementTree.SubElement(seg_el, "SplitTimes")

        # Remove any existing entry with the same comparison name (idempotent)
        for existing in split_times.findall("SplitTime"):
            if existing.get("name") == comparison_name:
                split_times.remove(existing)

        if cum is None:
            continue

        st_el = ElementTree.SubElement(split_times, "SplitTime")
        st_el.set("name", comparison_name)
        rt_el = ElementTree.SubElement(st_el, "RealTime")
        rt_el.text = format_time_lss(cum)

    ElementTree.indent(root, space="  ")
    return ElementTree.tostring(root, encoding="unicode", xml_declaration=True).encode("utf-8")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="LiveSplit Stats")
st.title("LiveSplit Stats")

uploaded = st.file_uploader("Upload a .lss file", type=["lss"])

if uploaded is not None:
    try:
        run_info, segments = parse_lss(uploaded.getvalue())
    except ElementTree.ParseError:
        st.error("Failed to parse the uploaded file. Make sure it is a valid .lss (XML) file.")
        st.stop()

    st.subheader(f"{run_info['game']} — {run_info['category']}")
    st.caption(f"Attempts: {run_info['attempts']}")

    durations = compute_segment_durations(segments)

    rows = []
    for seg, dur in zip(segments, durations):
        rows.append({
            "Segment": seg["name"],
            "PB Split": format_time(seg["pb_split"]),
            "PB Segment Duration": format_time(dur),
            "Best Segment": format_time(seg["best_segment"]),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width='stretch')

    # ------------------------------------------------------------------
    # Split consistency boxplot
    # ------------------------------------------------------------------
    all_attempt_ids: set[int] = set()
    for seg in segments:
        for h in seg["history"]:
            all_attempt_ids.add(h["attempt_id"])
    total_attempts = len(all_attempt_ids)

    if total_attempts >= 2:
        st.subheader("Split Consistency")

        n_attempts = st.slider(
            "Recent attempts to include",
            min_value=2,
            max_value=total_attempts,
            value=min(20, total_attempts),
        )

        hist_df = build_history_df(segments, n_attempts)

        if not hist_df.empty:
            segment_order = [seg["name"] for seg in segments]
            chart = alt.Chart(hist_df).mark_boxplot().encode(
                alt.X("Segment", sort=segment_order),
                alt.Y("Duration (s)"),
            )
            st.altair_chart(chart, width='stretch')

            # ----------------------------------------------------------
            # Top 5 splits that need training
            # ----------------------------------------------------------
            training_targets = rank_splits_for_training(hist_df)
            if training_targets is not None and not training_targets.empty:
                st.markdown("#### Splits to Train")
                st.dataframe(
                    training_targets.head(5),
                    hide_index=True,
                    width='stretch',
                )
        else:
            st.info("No segment history data available for the selected attempts.")

    # ------------------------------------------------------------------
    # Balanced Time Goal
    # ------------------------------------------------------------------
    sob = compute_sum_of_best(segments)
    pb_total = segments[-1]["pb_split"] if segments and segments[-1]["pb_split"] else None

    if sob is not None and pb_total is not None and total_attempts >= 2:
        st.subheader("Balanced Time Goal")

        bal_col1, bal_col2 = st.columns(2)
        with bal_col1:
            n_bal_attempts = st.slider(
                "Recent attempts to include",
                min_value=2,
                max_value=total_attempts,
                value=min(20, total_attempts),
                key="balanced_attempts",
            )
        with bal_col2:
            aggressiveness = st.slider(
                "Aggressiveness (0 = PB, 1 = Sum of Best)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
            )

        bal_hist_df = build_history_df(segments, n_bal_attempts)

        if not bal_hist_df.empty:
            balanced = compute_balanced_segments(
                segments, durations, bal_hist_df, aggressiveness,
            )

            if balanced and any(b is not None for b in balanced):
                cumulative = timedelta()
                goal_rows = []
                for seg, dur, bal in zip(segments, durations, balanced):
                    if bal is not None:
                        cumulative += bal
                    goal_rows.append({
                        "Segment": seg["name"],
                        "Best Segment": format_time(seg["best_segment"]),
                        "PB Segment": format_time(dur),
                        "Balanced Segment": format_time(bal),
                        "Balanced Split": format_time(
                            cumulative if bal is not None else None
                        ),
                    })

                st.metric("Balanced Goal Time", format_time(cumulative))
                met_col1, met_col2 = st.columns(2)
                met_col1.metric("Sum of Best", format_time(sob))
                met_col2.metric("Personal Best", format_time(pb_total))

                st.dataframe(
                    pd.DataFrame(goal_rows), hide_index=True, width='stretch',
                )

                # Build cumulative split list for export
                cum = timedelta()
                cumulative_splits: list[timedelta | None] = []
                for bal in balanced:
                    if bal is not None:
                        cum += bal
                        cumulative_splits.append(cum)
                    else:
                        cumulative_splits.append(None)

                modified_lss = inject_balanced_comparison(
                    uploaded.getvalue(), cumulative_splits,
                )

                base_name = uploaded.name.rsplit(".", 1)[0]
                st.download_button(
                    label="Download .lss with Balanced Goal",
                    data=modified_lss,
                    file_name=f"{base_name}_balanced.lss",
                    mime="application/xml",
                )
