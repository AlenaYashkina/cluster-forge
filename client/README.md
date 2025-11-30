# React UI for Manual Cluster Helper

This folder hosts the React + Vite front-end that talks to the Python backend. The UI exposes the queue, cluster buttons and actions via familiar React components.

## Setup

```bash
cd client
npm install
```

## Environment

Copy `.env.example` to `.env` and update `VITE_API_BASE` if the backend runs on a non-default host/port (defaults to `http://localhost:8000`).

## Development

```bash
npm run dev
```

The dev server hosts the interface at `http://localhost:5173` by default. It will forward API calls to the backend URL defined in `VITE_API_BASE`.

## Production Build

```bash
npm run build
```

After building, `npm run preview` can be used to sanity check the generated static bundle.
