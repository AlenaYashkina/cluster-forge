# Cluster Forge — CLIP‑assisted image clustering for photo reports

`cluster-forge` is an **interactive image clustering tool** that sits between raw site photos and the final photo‑report automation pipeline.

It was built to speed up preparation of maintenance photo reports for architectural / festive lighting: thousands of field photos must be grouped by **object**, **issue type**, and **fix stage** before they go into automated PowerPoint generation.

This tool combines:

- **CLIP‑based auto‑suggestions** for likely clusters  
- A **web UI** for fast manual review and corrections  
- **Physical re‑organization of files on disk**, so downstream scripts see a clean, normalized folder tree

It’s used together with the separate project **Photo Report Generator** (automatic PPTX reports).


## Filesystem API safety (important)

The FastAPI backend is intended for **local use only**, never a public internet deployment. Set `CLUSTER_FORGE_WORKSPACE` to an existing folder containing disposable copies of your images before starting the API. Requests outside that workspace are rejected. Set `CLUSTER_FORGE_ORIGIN` to the exact frontend origin (default: `http://localhost:5173`).

`GET /session` is read-only with respect to image movement; use the explicit `POST /auto-move` action for automatic sorting. The API has no user authentication; workspace restrictions are a defense-in-depth measure, not permission to expose it publicly. Backup original photos, then test assigning, undoing and symlink cases before using any valuable files.

## What it does

Given a root folder with photos, `cluster-forge` lets you:

- Load all images from a tree of subfolders into an **interactive queue**
- See one photo at a time in a React UI and **assign it to a cluster** (e.g.  
  `1_Выявлено`, `2_Устранено`, `Снег`, `Мусор`, etc.)
- Let **CLIP** propose a cluster when it’s confident enough (threshold is configurable):
  - if similarity ≥ threshold → the image is auto‑assigned
  - otherwise it is sent to the manual queue
- **Move files on disk** into per‑cluster subfolders, preserving useful path fragments  
  (construction ID, date, issue type, stage)
- Mark images as:
  - **accepted** (assigned to a cluster)
  - **trash / delete later**
  - **“maybe later”** (send to the end of the queue)
- **Undo the last move** if you mis‑clicked
- Keep a **history log (NDJSON)** with every decision, so you can audit and recompute stats

The end result is a **clean folder structure** where each cluster has its own directory and all images are consistently grouped for downstream automation (Photo Report Generator).


## How it works (high level)

### 1. Backend (Python + FastAPI + CLIP)

- Uses `transformers.CLIPModel` and `CLIPProcessor` to extract image embeddings, not OpenCLIP text-prompt zero-shot classification.
- A person labels representative images first. The system stores embeddings for manually assigned examples when **manual-only mode is off**.
- New images are compared with the stored cluster centroids using similarity. High-similarity matches may be moved automatically; ambiguous items remain in the manual review queue.
- Similarity scores and thresholds are **not model accuracy or validated confidence percentages**. No CLIP weights are retrained.
- File moves and decisions are recorded under `.manual_cluster/` and clustering state under `.clustering/`.
- The FastAPI API exposes explicit assign, delete, skip, shuffle, auto-move and undo actions.

### 2. Frontend (React + Vite)

- Single‑page app that talks to the backend via REST (API URL is set in `.env`)
- Lets the user:
  - set root path
  - toggle **manual‑only mode** (skip CLIP suggestions)
  - shuffle the queue
  - quickly assign / skip / undo
- Shows error messages from the backend (invalid root, missing files, etc.)
- Built with React + Vite (JSX), with lint configuration. See the actual client source for the current API wrapper.


## Typical workflow

1. **Prepare a root folder** with project photos  
   (paths still contain construction IDs, dates and raw “issue” labels).
2. **Start the backend** (example):

   ```bash
   # create venv and install dependencies
   pip install -r backend/requirements.txt

   # run FastAPI app (from repo root)
   # Set CLUSTER_FORGE_WORKSPACE to an existing folder of disposable photo copies first.
   # Windows PowerShell example:
   # $env:CLUSTER_FORGE_WORKSPACE = "C:\\photo-demo"
   python -m uvicorn api:app --host 127.0.0.1 --port 8000
   ```

3. **Configure the frontend** (from the `client/` folder):

   ```bash
   cd client
   cp .env.example .env
   # the supplied .env.example contains:
   VITE_API_BASE=http://localhost:8000
   ```

4. **Run the frontend** (still in `client/`):

   ```bash
   npm install
   npm run dev
   ```

5. Open the shown URL (usually `http://localhost:5173`), select a folder **inside the configured `CLUSTER_FORGE_WORKSPACE`**,  
   and start assigning clusters. CLIP can suggest or auto-route sufficiently similar images once labeled examples exist;
   the rest you review manually.

When you’re done, the folder tree is already grouped by cluster, and you can feed it
directly into the **Photo Report Generator** project.


## Tech highlights

- **Python** (3.x), **FastAPI**, **Pydantic**
- **Hugging Face Transformers CLIP** for image similarity against labeled examples
- **React**, **Vite**, **JSX**
- File‑system heavy logic with `pathlib` and incremental **NDJSON logging**
- Focus on **human‑in‑the‑loop ML tooling**: the model helps, but the final decision is always yours.


## Repository layout

```text
cluster-forge/
├── api.py                 # FastAPI entry point
├── backend/
│   ├── api.py             # REST handlers
│   └── cluster_core.py    # Image embeddings, queue and file operations
├── client/
│   ├── package.json
│   ├── vite.config.js
│   └── src/               # React JSX interface
├── assets/
│   └── cluster-preview.gif
├── tests/                 # Filesystem safety tests
└── README.md
```

## Preview

An animated preview of the manual clustering UI, demonstrating queue paging, queue stats overlays, and assignment interactions. This visual demo anchors the before/after story above.

![Manual clustering preview](assets/cluster-preview.gif)

## Relationship to other projects

`cluster-forge` is a **support tool** for the separate project:

- **Photo Report Generator** — automatic PPTX reports for lighting maintenance  
  (stamped photos, grouped by object/issue/stage).

Together they form a small pipeline:

> raw photos → `cluster-forge` (interactive clustering) →  
> normalized folders → Photo Report Generator (PPTX decks)


## About the author

Built by **Alena Yashkina** — workflow automation practitioner using AI-assisted coding.  
Portfolio and contact links:

- GitHub: https://github.com/AlenaYashkina
- LinkedIn: https://www.linkedin.com/in/alena-yashkina

