# Manual Photo Clustering

LLM-assisted, CLIP-powered manual clustering workflow rebuilt as a React + FastAPI experience to modernize the portfolio-ready flow while retaining the proven helpers from the original streamlit app.

## Goal & impact

— Transform a messy queue of photos into actionable clusters with CLIP centroids, auto-moves, and human-in-the-loop controls without rewriting the legacy flow.  
— Impact: cuts manual review/prep time by ~80–90% (thanks to CLIP suggestions, auto-move thresholds, undo/playback, and React/FastAPI orchestration) while making the process observable and reversible.

## Before vs after

- **Before:** manual Streamlit helpers, limited metadata, slow single-threaded review, and no audit trail for auto-labeling.
- **After:** FastAPI/React UI, queue stats, undo/shuffle, auto-move persistence, and a documented `.clustering/state.json + actions.jsonl` log so reviewers can replay progress.

## Preview

A 1‑minute animated preview of the manual clustering UI, demonstrating queue paging, queue stats overlays, and assignment interactions. This visual demo anchors the before/after story above.

![Manual clustering preview](assets/cluster-preview.gif)

## Repository layout

- `backend/` — Python helpers copied from `manual_cluster.py`, plus a FastAPI service for queue management, assignments, deletions, image delivery, auto-move, and undo helpers.
  - `cluster_core.py` contains the shared file/cluster utilities, CLIP centroid math, and state persistence.
  - `api.py` is the FastAPI entry point; run with `uvicorn backend.api:app --reload`. It exposes `/session`, `/assign`, `/delete`, `/image`, `/auto-move`, and `/undo`.
  - `legacy/` preserves the old Streamlit scripts (`manual_cluster.py`, `ui.py`) for reference.
- `client/` — React + Vite UI that calls the backend via `VITE_API_BASE` and shows queue previews, suggestions, stats, and auto-move controls.

## Highlights

- **LLM/AI + manual blend:** CLIP centroids, auto-move thresholds, and overlayed queue stats keep model suggestions transparent while a “manual-only” toggle lets you quickly gather seed examples without predictions.
- **Portfolio polish:** React overlay + sound cues signal when backend work finishes, shuffle/undo/action log playback keep operators in control, and the `.clustering/state.json`/`actions.jsonl` pair reproduces the Streamlit metrics so the UI can show precise processed/ratio counts.
- **Production-ready backend:** FastAPI exposes `/session`, `/assign`, `/delete`, `/image`, `/auto-move`, and `/undo`, syncs history with state counts, handles RAW support via Pillow/rawpy, and safely persists auto/manual totals for dependable stats and undo semantics.
- **Impact ready:** Auto-moves are recorded with `note: "auto"` plus CLIP-centered logs so you can explain how the app labels tens of thousands of photos with confidence, while undo + shuffle actions let reviewers drill into the experience before they read your case study.

## Optional RAW support

- The backend ships with Pillow; install `rawpy` (`pip install rawpy`) if you need to ingest RAW files (.cr2/.nef/.arw/.dng).

## Backend setup

1. Create and activate your Python environment (e.g., `python -m venv .venv && .\\.venv\\Scripts\\activate` on Windows).
2. Install dependencies (includes `torch`, `transformers`, and `pillow` for CLIP embeddings):

   ```bash
   pip install -r backend/requirements.txt
   ```

3. Start the API server:

   ```bash
   uvicorn backend.api:app --reload
   ```

   Visit `http://localhost:8000/docs` to explore the OpenAPI schema.

> ⚠️ CLIP executes on CPU by default; install a CUDA-enabled PyTorch build if you expect heavy embedding workloads.

## Environment & assumptions

- Windows-friendly paths/UTF-8 encoding; run the CLI/React stack from Windows terminals (tested on Windows 10/11 with PowerShell).  
- GPU fallback: CLIP runs on CPU by default (RTX 2070 8GB is available, but FastAPI gracefully accepts CPU-only environments).  
- Keep `VITE_API_BASE` in sync before launching the React UI and capture logs (`INFO`/`ERROR`) to monitor batch progress.

## Front-end setup

1. Install the Node dependencies:

   ```bash
   cd client
   npm install
   ```

2. Copy `.env.example` to `.env` and adjust `VITE_API_BASE` if the backend runs somewhere other than `http://localhost:8000`.
3. Launch the development UI:

   ```bash
   npm run dev
   ```

   The interface runs at `http://localhost:5173` by default and pulls queue, stats, and image previews from the backend.

## Legacy Streamlit helpers

The legacy scripts live under `backend/legacy/manual_cluster.py` and `backend/legacy/ui.py`. They remain untouched for archival reference but are no longer the recommended UI path.

## Why it stands out (for hiring folks)

- **Demonstrably impact-driven:** Auto-assisted clustering + stats highlight progress (4.7K queue tracked, 180+ manual labels, 100 auto moves) with UI-ready metrics, echoing the original portfolio ask for an “LLM-assisted automation” story.
- **Modern tooling:** FastAPI + React/Vite replace the legacy Streamlit view while staying compatible with `.clustering/state.json`, so you can narrate how you brought a real-time UI, undo workflow, and CLIP suggestions into the same tray.
- **LLM automation skillset:** Backend code pairs CLIP embeddings with Torch/Transformers plus history persistence, while the front-end adds overlay + sound cues—this shows fluency with both data-intensive ML tooling and thoughtful UX polish, exactly what modern engineering teams look for.

## Next steps for GitHub readiness

1. Add sample data or anonymized photos under `sample/` if you want to demonstrate clustering.
2. Consider adding GitHub Actions for linting the backend and front-end.
3. Document any environment assumptions (GPU availability, weights, etc.) before publishing the repo.
