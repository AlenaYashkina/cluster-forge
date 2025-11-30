from __future__ import annotations

import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from PIL import UnidentifiedImageError
from PIL.Image import Resampling

import app as core


def main() -> None:
    st.set_page_config(page_title="Manual + AI Photo Clustering", layout="wide")
    st.session_state.setdefault("prefetch_pool", ThreadPoolExecutor(max_workers=2))
    st.session_state.setdefault("_prefetch_future", None)
    st.session_state.setdefault("_prefetched_bundle", None)
    st.session_state.setdefault("_prefetch_target", None)
    st.session_state.setdefault("_prefetch_params", None)
    st.session_state.setdefault("queue", [])
    st.session_state.setdefault("did_backfill", False)

    with st.sidebar:
        st.markdown("### Settings")
        root_input = st.text_input(
            "Root folder",
            value=str(r"C:\Users\Alena\Dropbox\Загрузки\Звёзды\Исходники БЕЗ ШТАМПОВ — копия (3)\3) М16\4, 5) Яуза"),
        )

        st.markdown("### Режим")
        mode_choice = st.radio(
            "Cluster mode",
            options=["ИИ + подсказки", "Только вручную"],
            horizontal=True,
            help="ИИ использует CLIP для подсказок и авто-батча. Ручной режим отключает нейросеть.",
        )
        manual_only_mode = mode_choice == "Только вручную"

        enhanced_mode = False
        if manual_only_mode:
            enhanced_mode = bool(
                st.checkbox(
                    "Ускоренный ручной режим (без метрик, только лог)",
                    value=False,
                    help="Минимум вычислений: перенос файлов и логирование без метрик/нейросети.",
                )
            )

        conf_thr = float(
            st.slider(
                "Cosine threshold (assign/suggestion)",
                0.70,
                0.99,
                0.90,
                0.01,
                disabled=manual_only_mode,
            )
        )
        min_samp = int(
            st.number_input(
                "Min samples per cluster (auto/gating)", value=40, min_value=1, step=1, disabled=manual_only_mode
            )
        )
        last_k = int(st.number_input("Metrics window (last K items)", value=200, min_value=20, step=10))
        batch_n = int(st.number_input("Auto batch size", value=50, min_value=1, step=10, disabled=manual_only_mode))
        auto_enable = bool(
            st.checkbox(
                "Auto-move when confident (batch)",
                value=True,
                disabled=manual_only_mode,
                help="Работает только в режиме ИИ.",
            )
        )
        auto_active = auto_enable and not manual_only_mode
        inst_assign_enabled = True
        preview_h = int(st.slider("Preview height (px)", 300, 1200, 950, 10))

        st.markdown("### Actions")
        if st.button("🔁 Rescan"):
            core.refresh_pool(Path(root_input))
            st.session_state["queue"] = [str(p) for p in core.list_unlabeled_images_global(Path(root_input))]
            st.rerun()
        if st.button("📥 Fill queue"):
            core.refresh_pool(Path(root_input))
            st.session_state["queue"] = [str(p) for p in core.list_unlabeled_images_global(Path(root_input))]
            st.rerun()
        if st.button("↩️ Undo last move"):
            last_op = st.session_state.get("last_move")
            if last_op:
                try:
                    mem = core.load_state(Path(root_input))
                    core.undo_move(last_op, memory=mem, root_path=Path(root_input))
                    st.toast("Undo: restored → sent to tail", icon="↩️")
                    st.session_state["last_move"] = None
                    st.rerun()
                except (shutil.Error, OSError) as undo_err:
                    st.error(f"Undo error: {undo_err}")
        if st.button("🔀 Shuffle queue"):
            q_cur = st.session_state.get("queue", [])
            random.shuffle(q_cur)
            st.session_state["queue"] = q_cur

        st.markdown("---")
        if st.button("🧮 Recompute metrics (replay history)"):
            # rebuild centroids & counts; keep persistent totals untouched
            memory_tmp = core.Memory(clusters={}, counts={}, history=[], version=0)
            updated, missing = 0, 0

            cur_state = core.load_state(Path(root_input))
            for rec in cur_state.history:
                img_p = Path(rec.path_after)
                if img_p.exists():
                    try:
                        emb = core.embed_image_cached(img_p, max_edge=1024)
                    except (UnidentifiedImageError, OSError, ValueError, RuntimeError):
                        missing += 1
                        memory_tmp.counts[rec.cluster] = memory_tmp.counts.get(rec.cluster, 0) + 1
                        continue

                    sug, _sb, _rn, _sr = core.predict_cluster(emb, memory_tmp)
                    rec.was_predicted = sug is not None
                    rec.correct = bool(sug == rec.cluster)
                    updated += 1

                    memory_tmp.clusters.setdefault(rec.cluster, [])
                    memory_tmp.counts[rec.cluster] = memory_tmp.counts.get(rec.cluster, 0) + 1
                    memory_tmp.clusters[rec.cluster].append(core.to_float_list(emb))
                else:
                    missing += 1
                    memory_tmp.counts[rec.cluster] = memory_tmp.counts.get(rec.cluster, 0) + 1

            cur_state.clusters = memory_tmp.clusters
            cur_state.counts = memory_tmp.counts
            # keep cur_state.manual_total / auto_total as-is
            core.save_state(Path(root_input), cur_state)
            st.session_state["known_clusters"] = sorted(
                set(cur_state.counts.keys()) | set(cur_state.clusters.keys()) | core.discover_known_clusters(Path(root_input))
            )
            core._invalidate_centroids_cache()
            st.success(f"Done. Updated {updated} records. Missing files: {missing}.")

        if st.button("🛠️ Repair / Flatten clusters"):
            st.session_state["__do_repair__"] = True
            st.rerun()

    root = Path(root_input).resolve()
    if not root.exists():
        st.error("Root folder does not exist.")
        st.stop()

    fast_log_path: Optional[Path] = (root / core.STATE_DIRNAME / "fast_assignments.log") if enhanced_mode else None

    memory_state = core.load_state(root)
    fs_known = core.discover_known_clusters(root)
    known_from_state = set(memory_state.counts.keys()) | set(memory_state.clusters.keys())
    st.session_state["known_clusters"] = sorted(known_from_state | fs_known)

    if not memory_state.clusters and not memory_state.counts:
        _ = core.bootstrap_memory_from_fs(root, memory_state, max_per_cluster=40)
        core._invalidate_centroids_cache()

    if (len(memory_state.history) < 5) and (not st.session_state.get("did_backfill", False)):
        _ = core.backfill_history_from_fs(root, memory_state, max_events_per_cluster=60)
        st.session_state["did_backfill"] = True

    core.refresh_pool(root)

    _, _, device, _ = core.load_clip()
    st.caption(f"Device: {'CUDA' if device.type == 'cuda' else 'CPU'}. Clusters are sub-folders next to each photo.")

    if st.session_state.pop("__do_repair__", False):
        with st.status("Repairing / flattening…", expanded=True) as status_ctx:
            moved_cnt, deduped_cnt, fixed_cnt = core.flatten_clusters(
                root, memory_state, st.session_state.get("known_clusters", [])
            )
            status_ctx.update(
                label=f"Done. Moved: {moved_cnt}, removed duplicates: {deduped_cnt}, fixed history paths: {fixed_cnt}.",
                state="complete",
            )
        core._invalidate_centroids_cache()
        st.rerun()

    # ----------- Metrics & summary (consistent, history-cap-safe) -----------------
    if enhanced_mode:
        win_acc, pred_num = 0.0, 0
        manual_precision_value = "disabled"
    else:
        win_acc, pred_num = core._window_accuracy(memory_state, last_k_window=last_k)
        manual_precision_value = round(win_acc, 3)
    progress = core.compute_progress(memory_state, root)
    overall_metrics = {
        "Mode": "Manual" if manual_only_mode else "AI + suggestions",
        "Manual labeled (count)": int(memory_state.manual_total),
        "Auto moves (count)": int(memory_state.auto_total),
        "Labeled (truth)": progress["labeled_truth"],
        "Deleted (total)": progress["deleted_total"],
        "Unlabeled (now)": progress["unlabeled_now"],
        "Total images (est.)": progress["total_all"],
        "Done % (of total)": f"{progress['pct_done']}%",
        f"Manual suggestion precision (last {pred_num})": manual_precision_value,
        "Auto enabled (batch)": bool(auto_active),
        "Manual-only mode": bool(manual_only_mode),
        "Enhanced mode": bool(enhanced_mode),
    }

    # ------------------------------ Main labeling flow ----------------------------
    queue_list: List[str] = st.session_state.get("queue", [])
    if not queue_list:
        st.info('Queue is empty. Use buttons in the left panel: "Fill queue" or "Rescan".')
        st.stop()

    prefetch_future = st.session_state.get("_prefetch_future")
    if prefetch_future and prefetch_future.done():
        try:
            st.session_state["_prefetched_bundle"] = prefetch_future.result()
        except Exception:
            st.session_state["_prefetched_bundle"] = None
        st.session_state["_prefetch_future"] = None
        st.session_state["_prefetch_target"] = None
        st.session_state["_prefetch_params"] = None

    cur_path = Path(queue_list[0])
    base_leaf = core.leaf_root_for_image(cur_path)
    bundle = st.session_state.get("_prefetched_bundle")
    use_bundle = (
        bundle
        and bundle.get("path") == str(cur_path)
        and bundle.get("manual_only") == manual_only_mode
    )

    if use_bundle:
        display_img = bundle["display"]
        orig_w, orig_h = bundle["orig_size"]
        emb_full = bundle["embedding"]
        st.session_state["_prefetched_bundle"] = None
    else:
        try:
            cur_image = core.load_image_any(cur_path, max_edge=1280)
        except (UnidentifiedImageError, OSError, ValueError) as open_err:
            st.error(f"Failed to open {cur_path.name}: {open_err}")
            st.session_state["queue"].pop(0)
            st.stop()
        orig_w, orig_h = cur_image.size
        scale_val = preview_h / float(orig_h) if orig_h else 1.0
        disp_w = max(1, int(orig_w * scale_val))
        display_img = cur_image.resize((disp_w, preview_h), Resampling.LANCZOS)
        emb_full = None if manual_only_mode else core.embed_image_cached(cur_path, max_edge=1024)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.image(display_img, caption=cur_path.name)

    if manual_only_mode:
        suggest = None
        s_best = 0.0
        second_best = None
        s_second = 0.0
        margin_now = 0.0
    else:
        if emb_full is None:
            emb_full = core.embed_image_cached(cur_path, max_edge=1024)
        suggest, s_best, second_best, s_second = core.predict_cluster(emb_full, memory_state)
        margin_now = (s_best - s_second) if second_best is not None else s_best

    core._schedule_prefetch(queue_list, preview_h, manual_only_mode)

    with col2:
        st.markdown("**Diagnostics**")
        if manual_only_mode:
            st.info("Ручной режим: нейросеть и авто-подсказки отключены.")
        else:
            st.write(f"Suggest: {suggest or '-'}  |  score: {s_best:.2f}  |  margin: {margin_now:.2f}")
        st.markdown("**Path**")
        st.code(str(cur_path), language="text")

        leaf_cls = core.existing_clusters_in_leaf(base_leaf)
        global_cls = core.all_existing_clusters()
        combined: List[str] = sorted({*leaf_cls, *global_cls})

        labeled_options: List[str] = ["<new>"] + [(f"{c} (suggested)" if suggest == c else c) for c in combined]
        opt2raw: Dict[str, str] = {"<new>": "<new>"}
        for c in combined:
            opt2raw[c] = c
            opt2raw[f"{c} (suggested)"] = c

        default_label = f"{suggest} (suggested)" if suggest in combined else "<new>"

        sel_key = f"select_{str(cur_path)}"
        sel = st.selectbox(
            "Choose cluster",
            options=labeled_options,
            index=labeled_options.index(default_label),
            key=sel_key,
            on_change=core._mark_sel_changed,
            args=(str(cur_path),),
        )

        needs_new = opt2raw[sel] == "<new>"
        with st.form(key=f"assign_form_{str(cur_path)}", clear_on_submit=False):
            new_name = st.text_input("New cluster name", key=f"new_name_{str(cur_path)}") if needs_new else ""
            submit_assign = st.form_submit_button("Assign", type="primary")
            submit_delete = st.form_submit_button("Delete")

        raw_sel = opt2raw[sel]
        if submit_delete:
            core.delete_file_and_advance(cur_path, memory=memory_state, root_dir=root)

        if submit_assign:
            chosen_name = new_name.strip() if raw_sel == "<new>" else raw_sel
            if raw_sel == "<new>":
                chosen_name = core.normalize_cluster_input(chosen_name)
                if not chosen_name:
                    st.warning("Provide a cluster name.")
                    st.stop()
            core.assign_current(
                cur_p=cur_path,
                base_leaf_folder=base_leaf,
                chosen=chosen_name,
                embedding_vec=emb_full,
                suggestion=suggest,
                memory=memory_state,
                root_dir=root,
                mode="manual",
                fast_log=fast_log_path,
            )

        if inst_assign_enabled and st.session_state.get("sel_changed_for") == str(cur_path):
            st.session_state["sel_changed_for"] = None
            if raw_sel != "<new>":
                core.assign_current(
                    cur_p=cur_path,
                    base_leaf_folder=base_leaf,
                    chosen=raw_sel,
                    embedding_vec=emb_full,
                    suggestion=suggest,
                    memory=memory_state,
                    root_dir=root,
                    mode="manual",
                    fast_log=fast_log_path,
                )

        disabled_batch = (not auto_active) or (len(core.get_centroids(memory_state)) == 0)
        if manual_only_mode:
            st.info("Auto-move disabled while manual-only mode is active.")
        if st.button("Auto-move (batch)", key="auto_batch_btn", disabled=disabled_batch):
            moved_count = 0
            pool_batch = [Path(p) for p in st.session_state.get("pool_images", [])] or core.list_unlabeled_images_global(root)
            random.shuffle(pool_batch)

            for p in pool_batch:
                if moved_count >= int(batch_n):
                    break
                try:
                    emb_b = core.embed_image_cached(p, max_edge=1024)
                    pred_b, sim_b, _runner_b, _sr_b = core.predict_cluster(emb_b, memory_state)
                except (UnidentifiedImageError, OSError, ValueError):
                    continue

                if pred_b is None:
                    continue

                sim_ok = sim_b >= conf_thr
                win_acc2, _ = core._window_accuracy_by_cluster(memory_state, last_k, pred_b)
                mature2 = (memory_state.counts.get(pred_b, 0) >= min_samp)

                if sim_ok and auto_active and (win_acc2 >= core.DEFAULT_WIN_ACC_GATE) and mature2:
                    try:
                        before_sha2 = core.sha1_of_file(p)
                        op3 = core.move_to_cluster(p, core.leaf_root_for_image(p), pred_b)

                        memory_state.clusters.setdefault(pred_b, [])
                        memory_state.counts[pred_b] = memory_state.counts.get(pred_b, 0) + 1
                        memory_state.clusters[pred_b].append(core.to_float_list(emb_b))
                        memory_state.history.append(
                            core.LabeledItem(
                                sha1=before_sha2,
                                cluster=pred_b,
                                leaf=str(core.leaf_root_for_image(p)),
                                path_after=str(op3.dst),
                                was_predicted=True,
                                correct=True,
                                mode="auto",
                            )
                        )
                        memory_state.auto_total += 1
                        moved_count += 1
                        if moved_count % core.BATCH_SAVE_EVERY == 0:
                            core.save_state(root, memory_state)
                    except (shutil.Error, OSError, ValueError):
                        continue

            core.save_state(root, memory_state)
            core._invalidate_centroids_cache()
            st.success(f"Auto-moved: {moved_count} files.")

        st.divider()
        st.markdown("#### Summary")
        summary_df = pd.DataFrame([overall_metrics])
        st.dataframe(summary_df, hide_index=True, width='stretch')

        st.markdown("**Per-cluster metrics**")
        st.dataframe(
            core.compute_cluster_metrics_df(memory_state, last_k_window=last_k, min_samples_threshold=min_samp),
            hide_index=True,
            width='stretch',
        )


if __name__ == "__main__":
    main()
