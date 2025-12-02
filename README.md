# Cluster Forge — CLIP‑assisted image clustering for photo reports

`cluster-forge` is an **interactive image clustering tool** that sits between raw site photos and the final photo‑report automation pipeline.

It was built to speed up preparation of maintenance photo reports for architectural / festive lighting: thousands of field photos must be grouped by **object**, **issue type**, and **fix stage** before they go into automated PowerPoint generation.

This tool combines:

- **CLIP‑based auto‑suggestions** for likely clusters  
- A **web UI** for fast manual review and corrections  
- **Physical re‑organization of files on disk**, so downstream scripts see a clean, normalized folder tree

It’s used together with the separate project **Photo Report Generator** (automatic PPTX reports).


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

- Uses **OpenCLIP** to encode:
  - images from the root folder
  - text prompts that describe each cluster label
- For each image:
  - computes cosine similarity to each label prompt
  - picks the best match and compares it to a **confidence threshold**
- Exposes a **FastAPI** backend with endpoints like:
  - get next item in queue
  - assign to a cluster
  - move to trash / later
  - undo last assignment
  - compute basic stats from the history log
- Physically **moves files on disk** into cluster folders using `pathlib`, while
  updating an NDJSON history log in `.cluster_state/`.

### 2. Frontend (React + Vite)

- Single‑page app that talks to the backend via REST (API URL is set in `.env`)
- Lets the user:
  - set root path
  - toggle **manual‑only mode** (skip CLIP suggestions)
  - shuffle the queue
  - quickly assign / skip / undo
- Shows error messages from the backend (invalid root, missing files, etc.)
- Built with React + Vite (TypeScript), Axios for API calls, ESLint for static checks.


## Typical workflow

1. **Prepare a root folder** with project photos  
   (paths still contain construction IDs, dates and raw “issue” labels).
2. **Start the backend** (example):

   ```bash
   # create venv and install dependencies
   pip install -r requirements.txt

   # run FastAPI app
   uvicorn api:app --reload --port 8000
   ```

3. **Configure the frontend**:

   ```bash
   cp .env.example .env
   # set VITE_API_URL to the backend URL, e.g.
   VITE_API_URL=http://localhost:8000
   ```

4. **Run the frontend**:

   ```bash
   npm install
   npm run dev
   ```

5. Open the shown URL (usually `http://localhost:5173`), set the root folder,  
   and start assigning clusters. CLIP will auto‑label everything it’s confident about;
   the rest you review manually.

When you’re done, the folder tree is already grouped by cluster, and you can feed it
directly into the **Photo Report Generator** project.


## Tech highlights

- **Python** (3.x), **FastAPI**, **Pydantic**
- **OpenCLIP** for zero‑shot image classification
- **React**, **Vite**, **TypeScript**
- File‑system heavy logic with `pathlib` and incremental **NDJSON logging**
- Focus on **human‑in‑the‑loop ML tooling**: the model helps, but the final decision is always yours.


## Repository layout (simplified)

```text
cluster-forge/
├── api.py              # FastAPI app (REST API for the UI)
├── cluster_core.py     # CLIP model, queue logic, file moves, logging
├── cluster-preview.gif # Short demo of the UI
├── client/ (or repo root)
│   ├── package.json    # React/Vite frontend
│   ├── vite.config.js
│   ├── src/…           # React components & API client
│   └── .env.example    # VITE_API_URL and other client settings
└── README.md           # This file
```

(Exact paths may differ slightly depending on how you clone / place the backend code.)


## Preview

A 1‑minute animated preview of the manual clustering UI, demonstrating queue paging, queue stats overlays, and assignment interactions. This visual demo anchors the before/after story above.

![Manual clustering preview](assets/cluster-preview.gif)

## Screenshot

![Manual clustering screenshot](assets/cluster-preview.gif)


## Relationship to other projects

`cluster-forge` is a **support tool** for the separate project:

- **Photo Report Generator** — automatic PPTX reports for lighting maintenance  
  (stamped photos, grouped by object/issue/stage).

Together they form a small pipeline:

> raw photos → `cluster-forge` (interactive clustering) →  
> normalized folders → Photo Report Generator (PPTX decks)


## About the author

Built by **Alena Yashkina** — lighting engineer turned AI‑automation developer.  
Portfolio and contact links:

- GitHub: https://github.com/AlenaYashkina
- LinkedIn: https://www.linkedin.com/in/alena-yashkina-a9994a35a/

