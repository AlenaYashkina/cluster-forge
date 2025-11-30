import json
import os
import shutil
import subprocess
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict

import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError
from PIL.Image import Resampling
from streamlit.runtime.scriptrunner import get_script_run_ctx

def _do_rerun() -> None:
    # Use stable rerun API; avoid manually raising RerunException to prevent internal crashes.
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.session_state["manual_queue_rerun"] = True
        st.stop()


def _ensure_streamlit_runner() -> None:
    """
    If the script is invoked via plain `python manual_cluster.py`, there is no
    Streamlit ScriptRunContext, which triggers noisy warnings. Relaunch the
    script through `streamlit run` to bootstrap the runtime properly.
    """
    try:
        ctx = get_script_run_ctx()
    except Exception:
        ctx = None

    is_running_with_streamlit = getattr(st, "_is_running_with_streamlit", False)
    if callable(is_running_with_streamlit):
        is_running_with_streamlit = is_running_with_streamlit()

    if ctx is None and not is_running_with_streamlit:
        script_path = Path(__file__).resolve()
        cmd = [sys.executable, "-m", "streamlit", "run", str(script_path)]
        subprocess.run(cmd, check=False)
        sys.exit(0)


IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
STATE_DIRNAME = ".manual_cluster"
LOG_NAME = "actions.jsonl"
CLUSTER_STATE_DIRNAME = ".clustering"
CLUSTER_STATE_FILE = "state.json"
LOOKALIKES: Dict[str, str] = {
    "\u0430": "a",
    "\u0432": "b",
    "\u0435": "e",
    "\u043a": "k",
    "\u043c": "m",
    "\u043d": "h",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0442": "t",
    "\u0445": "x",
    "\u0443": "y",
    "\u0451": "e",
}



def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def list_unlabeled_images(root: Path, known_clusters: Optional[Set[str]] = None) -> List[str]:
    known_tokens = {_canonical_token(c) for c in known_clusters} if known_clusters else set()
    result: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        pdir = Path(dirpath)
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".") and (_canonical_token(d) not in known_tokens)
        ]
        for fname in filenames:
            fp = pdir / fname
            if fp.is_file() and is_image_file(fp):
                result.append(str(fp))
    result.sort()
    return result


def ensure_cluster_dir(img_path: Path, cluster: str) -> Path:
    cluster = cluster.strip()
    base = img_path.parent / cluster
    base.mkdir(parents=True, exist_ok=True)
    return base


def move_into_cluster(img_path: Path, cluster: str) -> Path:
    dst_dir = ensure_cluster_dir(img_path, cluster)
    candidate = dst_dir / img_path.name
    if candidate.exists():
        stem, ext = img_path.stem, img_path.suffix
        idx = 1
        while True:
            alt = dst_dir / f"{stem}__dup{idx}{ext}"
            if not alt.exists():
                candidate = alt
                break
            idx += 1
    shutil.move(str(img_path), str(candidate))
    return candidate


def append_log(root: Path, entry: Dict[str, str]) -> None:
    log_dir = root / STATE_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_NAME
    with log_path.open("a", encoding="utf-8") as fh:
        json.dump(entry, fh, ensure_ascii=False)
        fh.write("\n")


def load_history(root: Path) -> List[Dict[str, str]]:
    log_path = root / STATE_DIRNAME / LOG_NAME
    if not log_path.exists():
        return []
    entries: List[Dict[str, str]] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def load_main_state(root: Path) -> Optional[Dict[str, Any]]:
    state_path = root / CLUSTER_STATE_DIRNAME / CLUSTER_STATE_FILE
    if not state_path.exists():
        return None
    try:
        with state_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def clusters_from_state(state: Optional[Dict[str, Any]]) -> Set[str]:
    if not state:
        return set()
    names: Set[str] = set()
    for key in ("counts", "clusters"):
        for cname in (state.get(key) or {}):
            if cname:
                names.add(str(cname))
    return names


def clusters_from_history(entries: List[Dict[str, str]]) -> Set[str]:
    names: Set[str] = set()
    for entry in entries:
        cname = entry.get("cluster")
        if cname and cname not in {"__deleted__"}:
            names.add(cname)
    return names


def gather_known_clusters(
    history_entries: List[Dict[str, str]], main_state: Optional[Dict[str, Any]]
) -> Set[str]:
    names: Set[str] = set(st.session_state.get("manual_clusters", set()))
    names.update(clusters_from_state(main_state))
    names.update(clusters_from_history(history_entries))
    return names


def _canonical_token(token: str) -> str:
    token = unicodedata.normalize("NFKC", token).casefold()
    return "".join(LOOKALIKES.get(ch, ch) for ch in token)


def render_cluster_buttons(
    clusters: List[str],
    per_row: int = 4,
    on_click: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """
    Render cluster choices as a grid of buttons instead of a dropdown.
    Returns the currently selected cluster; optionally triggers on_click immediately.
    """
    if "selected_cluster_choice" not in st.session_state:
        st.session_state["selected_cluster_choice"] = None

    selected = st.session_state.get("selected_cluster_choice")
    if not clusters:
        st.info("No clusters yet. Create one below to start labeling.")
        return None

    for row_start in range(0, len(clusters), per_row):
        row = clusters[row_start : row_start + per_row]
        cols = st.columns(per_row, gap="small")
        for col, name in zip(cols, row):
            label = f"* {name}" if name == selected else name
            key_token = _canonical_token(name).replace(" ", "_")
            if col.button(
                label,
                key=f"cluster_btn_{row_start}_{key_token}",
                width='stretch',
            ):
                selected = name
                st.session_state["selected_cluster_choice"] = name
                if on_click:
                    on_click(name)
    return selected


def render_cluster_buttons_custom(
    clusters: List[str],
    on_click: Optional[Callable[[str], None]] = None,
) -> Optional[str]:
    """
    Render clusters in a custom, human-friendly order.

    Rows (if present) in this order:
      1) "Требуется ВГГ", "Требуется очистка от снега", "Требуется УО"
      2) "ВГГ", "УО"
      3) "Лопата", "Щётка", "Тряпка"
    Remaining clusters follow after these rows, laid out in rows of up to 5.
    Returns the selected cluster (and triggers on_click immediately when a chip is pressed).
    """
    if "selected_cluster_choice" not in st.session_state:
        st.session_state["selected_cluster_choice"] = None

    selected = st.session_state.get("selected_cluster_choice")
    if not clusters:
        st.info("No clusters yet. Create one below to start labeling.")
        return None

    # map canonical token -> original name for robust matching
    token_map = { _canonical_token(n): n for n in clusters }

    desired_rows = [
        ["Требуется ВГГ", "Требуется очистка от снега", "Требуется УО"],
        ["ВГГ", "УО"],
        ["Лопата", "Щётка", "Тряпка"],
    ]

    used = set()
    rows: List[List[str]] = []
    for dr in desired_rows:
        row_items: List[str] = []
        for name in dr:
            tok = _canonical_token(name)
            if tok in token_map:
                row_items.append(token_map[tok])
                used.add(token_map[tok])
        if row_items:
            rows.append(row_items)

    # remaining clusters (not yet used) appended afterwards
    # ensure 'Норм' does not appear in the upper cluster grid; it is rendered in the final action row
    norm_tok = _canonical_token("Норм")
    if norm_tok in token_map:
        used.add(token_map[norm_tok])

    remaining = [c for c in clusters if c not in used]
    if remaining:
        # chunk remaining into rows of up to 5
        per_row = 5
        for i in range(0, len(remaining), per_row):
            rows.append(remaining[i : i + per_row])

    # render rows
    row_start = 0
    for row in rows:
        cols = st.columns(len(row), gap="small")
        for col, name in zip(cols, row):
            label = f"* {name}" if name == selected else name
            key_token = _canonical_token(name).replace(" ", "_")
            if col.button(label, key=f"cluster_btn_custom_{row_start}_{key_token}", width='stretch'):
                selected = name
                st.session_state["selected_cluster_choice"] = name
                if on_click:
                    on_click(name)
        row_start += len(row)

    return selected


def _match_child_by_name(parent: Path, target: str) -> Optional[Path]:
    try:
        entries = list(parent.iterdir())
    except (OSError, PermissionError):
        return None
    target_key = _canonical_token(target)
    for entry in entries:
        if _canonical_token(entry.name) == target_key:
            return entry
    return None


def resolve_input_path(raw: str) -> Optional[Path]:
    clean = raw.strip().strip("\"'")
    if not clean:
        return None
    try:
        candidate = Path(clean).expanduser()
    except (RuntimeError, OSError):
        return None
    if candidate.exists():
        return candidate

    parts = candidate.parts
    if not parts:
        return None
    current = Path(parts[0])
    if not current.exists():
        return None
    for part in parts[1:]:
        next_path = current / part
        if next_path.exists():
            current = next_path
            continue
        match = _match_child_by_name(current, part)
        if match is None:
            return None
        current = match
    return current if current.exists() else None


def load_image_preview(path: Path, preview_h: int) -> Image.Image:
    try:
        img = Image.open(path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as err:
        raise UnidentifiedImageError(f"Cannot open image: {path}") from err
    w, h = img.size
    if h == 0:
        h = 1
    scale = preview_h / float(h)
    resized = img.resize((max(1, int(w * scale)), preview_h), Resampling.LANCZOS)
    return resized


_ensure_streamlit_runner()

st.set_page_config(page_title="Manual Cluster Helper", layout="wide")

st.session_state.setdefault("manual_queue", [])
st.session_state.setdefault("manual_clusters", set())
st.session_state.setdefault("manual_root", "")
st.session_state.setdefault("selected_cluster_choice", None)

st.markdown("### Settings")
default_root = st.session_state.get("manual_root") or str(Path(r"C:\Users\Alena\Downloads\3) M16"))
controls_col1, controls_col2, controls_col3 = st.columns([6, 1.2, 1.2], gap="medium")
with controls_col1:
    root_input = st.text_input("Root folder", value=default_root, label_visibility="visible")
preview_h = 620
with controls_col2:
    if st.button("Rescan images", width='stretch'):
        new_root = resolve_input_path(root_input)
        if new_root is None or not new_root.is_dir():
            st.warning("Cannot access provided root folder.")
        else:
            new_history = load_history(new_root)
            new_state = load_main_state(new_root)
            new_known = gather_known_clusters(new_history, new_state)
            st.session_state["manual_clusters"] = new_known
            st.session_state["manual_queue"] = list_unlabeled_images(new_root, new_known)
            st.session_state["manual_root"] = str(new_root)
            st.session_state["selected_cluster_choice"] = None
            _do_rerun()
with controls_col3:
    if st.button("Shuffle queue", width='stretch'):
        import random

        random.shuffle(st.session_state["manual_queue"])

root_path = resolve_input_path(root_input)
if root_path is None or not root_path.is_dir():
    st.error("Root folder does not exist.")
    st.stop()

if st.session_state.get("manual_root") != str(root_path):
    st.session_state["manual_queue"] = []
    st.session_state["manual_clusters"] = set()
    st.session_state["manual_root"] = str(root_path)
    st.session_state["selected_cluster_choice"] = None

# Load existing progress and cluster names (both from this helper and the main app state)
history_entries = load_history(root_path)
main_state = load_main_state(root_path)

known_clusters = gather_known_clusters(history_entries, main_state)
st.session_state["manual_clusters"] = known_clusters
if (
    st.session_state.get("selected_cluster_choice")
    and st.session_state["selected_cluster_choice"] not in known_clusters
):
    st.session_state["selected_cluster_choice"] = None

if not st.session_state["manual_queue"]:
    st.session_state["manual_queue"] = list_unlabeled_images(root_path, known_clusters)

queue = st.session_state["manual_queue"]
if not queue:
    st.success("No unlabeled images. Press Rescan if you added new files.")
    st.stop()

manual_total = sum(1 for e in history_entries if e.get("cluster") not in {"__deleted__", None})
deleted_total = sum(1 for e in history_entries if e.get("cluster") == "__deleted__")
cluster_counts = defaultdict(int)
for entry in history_entries:
    cname = entry.get("cluster")
    if cname and cname not in {"__deleted__"}:
        cluster_counts[cname] += 1

state_counts = (main_state or {}).get("counts", {})
state_manual_total = (main_state or {}).get("manual_total", 0)
state_deleted_total = (main_state or {}).get("deleted_total", 0)
state_auto_total = (main_state or {}).get("auto_total", 0)
processed_total = manual_total + deleted_total
total_seen = processed_total + len(queue)
processed_pct = (processed_total / total_seen * 100.0) if total_seen else 0.0

cur_path = Path(queue[0])
try:
    preview = load_image_preview(cur_path, preview_h)
except UnidentifiedImageError as err:
    st.warning(f"Cannot open {cur_path}: {err}")
    queue.pop(0)
    _do_rerun()

col_img, col_form = st.columns([3, 2], gap="large")

with col_img:
    st.image(preview, width='stretch')
    st.code(str(cur_path), language="text")

with col_form:
    st.markdown("### Cluster assignment")
    st.markdown(
        """
        <style>
        :root {
            --chip-bg: #1f2430;
            --chip-border: #343b4c;
            --chip-text: #f2f4f9;
        }
        .cluster-pane {
            max-width: 520px;
            margin: 0 auto;
            text-align: center;
            padding: 0.2rem 0.4rem;
        }
        .cluster-pane .cluster-buttons-marker {
            padding: 0.2rem 0.2rem 0;
            margin: 0 auto 0.4rem;
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            justify-content: center;
        }
        .cluster-pane .stButton button {
            background: var(--chip-bg);
            color: var(--chip-text);
            border: 1px solid var(--chip-border);
            border-radius: 8px;
            padding: 0.1rem 0.38rem;
            font-size: clamp(7.5px, 0.75vw, 9.5px) !important;
            font-weight: 600;
            letter-spacing: 0.01px;
            box-shadow: none;
            white-space: nowrap; /* prevent ugly wraps */
            word-break: keep-all; /* avoid breaking words for cyrillic/latin */
            max-width: 220px;
            min-width: 82px;
            width: auto;
            line-height: 0.96;
            overflow: hidden;
            text-overflow: ellipsis; /* show ellipsis instead of wrapping */
        }
        /* Global override to ensure buttons stay on one line even outside cluster-pane */
        .stButton > button {
            font-size: 8.5px !important;
            padding: 0.1rem 0.38rem !important;
            white-space: nowrap !important;
            word-break: keep-all !important;
            min-width: 82px !important;
            max-width: 220px !important;
            line-height: 1 !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }
        .cluster-pane .stButton button:hover {
            background: #273040;
            border-color: #46526a;
        }
        .cluster-pane .stButton button:focus {
            outline: 2px solid #8896b8;
            border-color: #5c6c8f;
        }
        div[data-baseweb="input"], textarea[data-baseweb="textarea"] {
            max-width: 480px;
            margin-left: auto;
            margin-right: auto;
        }
        .stButton button {
            max-width: 480px;
            margin-left: auto;
            margin-right: auto;
        }
        .chip-help {
            font-size: 12px;
            color: #aeb4c3;
            margin-top: 0.1rem;
            margin-bottom: 0.35rem;
        }
        div[data-baseweb="input"] input,
        textarea, textarea[data-baseweb="textarea"],
        .stButton button, label, .stMarkdown p, .stTextInput label {
            font-size: 11px !important;
        }
        .smaller-buttons button {
            font-size: 11px !important;
        }
        div[data-testid="stImage"] img {
            max-height: 620px;
            object-fit: contain;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.container().markdown('<div class="cluster-pane">', unsafe_allow_html=True)
    st.markdown('<div class="cluster-buttons-marker"></div>', unsafe_allow_html=True)
    new_name_key = f"new_cluster_{cur_path}"

    def do_assign(cluster_choice: str) -> None:
        cluster_choice = cluster_choice.strip()
        if not cluster_choice:
            st.warning("Enter or select a cluster name.")
            st.stop()
        try:
            dst = move_into_cluster(cur_path, cluster_choice)
        except (OSError, shutil.Error) as move_err:
            st.error(f"Assign error: {move_err}")
            st.stop()

        st.session_state["manual_clusters"].add(cluster_choice)
        st.session_state["selected_cluster_choice"] = None
        append_log(
            root_path,
            {
                "ts": datetime.now().isoformat(),
                "src": str(cur_path),
                "dst": str(dst),
                "cluster": cluster_choice,
                "note": "",
            },
        )
        queue.pop(0)
        _do_rerun()

    def on_new_cluster_enter() -> None:
        new_val = st.session_state.get(new_name_key, "").strip()
        if not new_val:
            return
        do_assign(new_val)

    options = sorted(known_clusters)
    # Use custom renderer: desired rows are handled in render_cluster_buttons_custom
    selected = render_cluster_buttons_custom(
        options,
        on_click=lambda name: do_assign(name),
    )
    st.caption("Click a cluster to assign instantly.")

    # Final action row: 'Норм', 'Delete', 'Skip' (in this order)
    col_norm, col_delete, col_skip = st.columns([1, 1, 1])
    with col_norm:
        norm_name = None
        norm_tok = _canonical_token("Норм")
        opts_map = { _canonical_token(n): n for n in options }
        if norm_tok in opts_map:
            norm_name = opts_map[norm_tok]
        if norm_name:
            btn_norm = st.button(norm_name, use_container_width=True)
            if btn_norm:
                do_assign(norm_name)
        else:
            # placeholder when not present
            st.button("Норм", use_container_width=True)
    with col_delete:
        btn_delete = st.button("Delete", type="secondary", use_container_width=True)
    with col_skip:
        btn_skip = st.button("Skip > queue end", use_container_width=True)

    st.markdown("#### New cluster or rename")
    new_name = st.text_input(
        "New cluster name (optional)",
        key=new_name_key,
        placeholder="Enter a name to create & assign",
        on_change=on_new_cluster_enter,
    )
    st.caption("Type a name and press Enter to create & apply.")

    st.markdown("</div>", unsafe_allow_html=True)

# Note: explicit Assign button removed; clicking a cluster chip assigns instantly,
# and typing a new cluster + Enter will create & assign via on_new_cluster_enter.

if btn_delete:
    try:
        cur_path.unlink()
    except (OSError, PermissionError) as delete_err:
        st.error(f"Delete error: {delete_err}")
    else:
        st.session_state["selected_cluster_choice"] = None
        append_log(
            root_path,
            {
                "ts": datetime.now().isoformat(),
                "src": str(cur_path),
                "dst": "",
                "cluster": "__deleted__",
                "note": "",
            },
        )
        queue.pop(0)
        _do_rerun()

if btn_skip:
    queue.append(queue.pop(0))
    st.session_state["selected_cluster_choice"] = None
    _do_rerun()

st.divider()
st.markdown("### Summary")
summary_df = pd.DataFrame(
    [
        {
            "Manual labeled": manual_total,
            "Deleted": deleted_total,
            "Unlabeled queue": len(queue),
            "Total processed": processed_total,
            "Processed %": round(processed_pct, 3),
        }
    ]
)
st.table(summary_df)

if main_state:
    st.markdown("### Current state (.clustering)")
    state_df = pd.DataFrame(
        [
            {
                "Clusters": len(state_counts),
                "Manual total": state_manual_total,
                "Auto total": state_auto_total,
                "Deleted total": state_deleted_total,
            }
        ]
    )
    st.table(state_df)

st.markdown("### Cluster stats (actions log)")
if cluster_counts:
    cluster_df = pd.DataFrame(
        [{"Cluster": name, "Labeled": count} for name, count in sorted(cluster_counts.items())]
    )
    st.table(cluster_df)
else:
    st.info("No clusters labeled yet.")

if state_counts:
    st.markdown("### Cluster stats (.clustering)")
    cluster_df_state = pd.DataFrame(
        [{"Cluster": name, "Labeled": count} for name, count in sorted(state_counts.items())]
    )
    st.table(cluster_df_state)
