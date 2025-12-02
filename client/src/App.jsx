import { useCallback, useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";
const QUEUE_CHUNK_SIZE = 60;

const formatNumber = (value) => {
  return new Intl.NumberFormat("en-US").format(value);
};

function App() {
  const [rootInput, setRootInput] = useState(() => {
    return localStorage.getItem("manualRoot") ?? "";
  });
  const [session, setSession] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [actionMessage, setActionMessage] = useState("");
  const [newCluster, setNewCluster] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const [manualOnly, setManualOnly] = useState(false);
  const [autoThreshold, setAutoThreshold] = useState(0.9);
  const [autoMinSamples, setAutoMinSamples] = useState(5);
  const [isAutoMoving, setIsAutoMoving] = useState(false);
  const [isUndoing, setIsUndoing] = useState(false);
  const [isShuffling, setIsShuffling] = useState(false);
  const [isSkipping, setIsSkipping] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [overlayVisible, setOverlayVisible] = useState(false);
  const [overlayMessage, setOverlayMessage] = useState("");

  const currentImage = useMemo(
    () => (session?.queue?.length ? session.queue[0] : null),
    [session]
  );

  const playSignal = useCallback(() => {
    try {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      const ctx = new AudioContext();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = 620;
      gain.gain.value = 0.12;
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.35);
      osc.stop(ctx.currentTime + 0.4);
    } catch (err) {
      // ignore autoplay restrictions
    }
  }, []);

  const runWithOverlay = useCallback(
    async (message, action, { soundOnSuccess = false } = {}) => {
      setOverlayMessage(message);
      setOverlayVisible(true);
      let result;
      let succeeded = false;
      try {
        result = await action();
        succeeded = true;
        return result;
      } finally {
        setOverlayVisible(false);
        if (soundOnSuccess && succeeded) {
          playSignal();
        }
      }
    },
    [playSignal]
  );

  const fetchSession = useCallback(async () => {
    if (!rootInput) {
      setSession(null);
      return;
    }
    setIsLoading(true);
    setErrorMessage("");
    try {
      const json = await runWithOverlay(
        "Loading queue...",
        async () => {
          const params = new URLSearchParams({ root: rootInput });
          if (manualOnly) {
            params.set("manual_only", "1");
          }
          params.set("queue_limit", QUEUE_CHUNK_SIZE.toString());
          const response = await fetch(`${API_BASE}/session?${params.toString()}`);
          if (!response.ok) {
            throw new Error(await response.text());
          }
          return response.json();
        },
        { soundOnSuccess: false }
      );
      setSession(json);
      setActionMessage("");
      const autoMoved = json?.auto_move?.moved?.length ?? 0;
      if (autoMoved) {
        setActionMessage(`Auto-moved ${autoMoved} file${autoMoved === 1 ? "" : "s"}.`);
        playSignal();
      } else if (json?.auto_move?.errors?.length) {
        setErrorMessage(json.auto_move.errors.join(" "));
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setIsLoading(false);
    }
  }, [rootInput, manualOnly]);

  useEffect(() => {
    localStorage.setItem("manualRoot", rootInput);
  }, [rootInput]);

  useEffect(() => {
    fetchSession();
  }, [fetchSession, refreshKey]);

  const reloadSession = () => setRefreshKey((value) => value + 1);

  const assignCluster = async (cluster) => {
    if (!currentImage || !session) {
      return;
    }
    if (isLoading) {
      return;
    }
    const payload = {
      root: session.root,
      image_path: currentImage,
      cluster,
      manual_only: manualOnly,
    };
    try {
      await runWithOverlay(
        "Assigning photo...",
        async () => {
          const response = await fetch(`${API_BASE}/assign`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
          });
          if (!response.ok) {
            throw new Error(await response.text());
          }
        },
        { soundOnSuccess: true }
      );
      setActionMessage(`Assigned to ${cluster}`);
      setNewCluster("");
      reloadSession();
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Assign failed");
    }
  };

  const deleteImage = async () => {
    if (!currentImage || !session) {
      return;
    }
    if (isLoading) {
      return;
    }
    const payload = {
      root: session.root,
      image_path: currentImage,
    };
    try {
      await runWithOverlay(
        "Deleting photo...",
        async () => {
          const response = await fetch(`${API_BASE}/delete`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
          });
          if (!response.ok) {
            throw new Error(await response.text());
          }
        },
        { soundOnSuccess: true }
      );
      setActionMessage("Deleted image");
      reloadSession();
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Delete failed");
    }
  };

  const skipImage = useCallback(async () => {
    if (!session) {
      return;
    }
    setIsSkipping(true);
    setErrorMessage("");
    try {
      await runWithOverlay(
        "Skipping photo...",
        async () => {
          const response = await fetch(`${API_BASE}/queue/skip`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ root: session.root }),
          });
          if (!response.ok) {
            throw new Error(await response.text());
          }
        }
      );
      setActionMessage("Skipped image");
      reloadSession();
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Skip failed");
    } finally {
      setIsSkipping(false);
    }
  }, [session, runWithOverlay, reloadSession]);

  const loadMoreQueue = useCallback(async () => {
    if (!session?.queue_has_more || isLoadingMore) {
      return;
    }
    setIsLoadingMore(true);
    setErrorMessage("");
    try {
      const offset = session.queue.length;
      const limit = session.queue_limit ?? QUEUE_CHUNK_SIZE;
      const params = new URLSearchParams({
        root: session.root,
        offset: offset.toString(),
        limit: limit.toString(),
      });
      const response = await fetch(`${API_BASE}/queue?${params.toString()}`);
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = await response.json();
      const added = Array.isArray(data.queue) ? data.queue.length : 0;
      setSession((prev) => {
        if (!prev) {
          return prev;
        }
        return {
          ...prev,
          queue: [...prev.queue, ...(data.queue ?? [])],
          queue_total: data.queue_total,
          queue_offset: data.queue_offset,
          queue_limit: data.queue_limit,
          queue_has_more: data.queue_has_more,
        };
      });
      if (added) {
        setActionMessage(`Loaded ${added} more queue item${added === 1 ? "" : "s"}.`);
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Failed to load queue");
    } finally {
      setIsLoadingMore(false);
    }
  }, [session, isLoadingMore]);

  const handleNewCluster = async (event) => {
    event.preventDefault();
    const value = newCluster.trim();
    if (value) {
      await assignCluster(value);
    }
  };

  const shuffleQueue = useCallback(async () => {
    if (!session) {
      return;
    }
    setIsShuffling(true);
    setErrorMessage("");
    try {
      await runWithOverlay(
        "Shuffling queue...",
        async () => {
          const response = await fetch(`${API_BASE}/queue/shuffle`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ root: session.root }),
          });
          if (!response.ok) {
            throw new Error(await response.text());
          }
        },
        { soundOnSuccess: true }
      );
      setActionMessage("Queue shuffled");
      reloadSession();
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Shuffle failed");
    } finally {
      setIsShuffling(false);
    }
  }, [session, runWithOverlay, reloadSession]);

  const triggerAutoMove = async () => {
    if (!session || isAutoMoving || !currentImage) {
      return;
    }
    if (manualOnly) {
      return;
    }
    setIsAutoMoving(true);
    setErrorMessage("");
    try {
      const data = await runWithOverlay(
        "Auto-moving photo...",
        async () => {
          const response = await fetch(`${API_BASE}/auto-move`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              root: session.root,
              threshold: autoThreshold,
              min_samples: autoMinSamples,
              image_path: currentImage,
            }),
          });
          if (!response.ok) {
            throw new Error(await response.text());
          }
          return response.json();
        },
        { soundOnSuccess: false }
      );
      const result = data?.result ?? {};
      const moved = result?.moved?.length ?? 0;
      const errors = result?.errors ?? [];
      if (moved) {
        setActionMessage(`Auto-moved ${moved} file${moved === 1 ? "" : "s"}.`);
        playSignal();
      } else if (errors.length) {
        setErrorMessage(errors.join(" "));
      } else {
        setActionMessage("Auto-move skipped (no eligible candidate).");
      }
      reloadSession();
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Auto-move failed");
    } finally {
      setIsAutoMoving(false);
    }
  };

  const undoLastMove = async () => {
    if (!session || isUndoing) {
      return;
    }
    setIsUndoing(true);
    setErrorMessage("");
    try {
      const response = await fetch(`${API_BASE}/undo`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ root: session.root }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = await response.json();
      const restored = data?.result?.restored_path;
      setActionMessage(
        restored ? `Undo restored ${restored}` : "Undo applied to the last move"
      );
      reloadSession();
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Undo failed");
    } finally {
      setIsUndoing(false);
    }
  };

  const stats = session?.stats;
  const knownClusters = session?.known_clusters ?? [];

  const previewUrl =
    currentImage && session
      ? `${API_BASE}/image?${new URLSearchParams({
          root: session.root,
          path: currentImage,
        }).toString()}`
      : null;

  return (
    <main className="app-shell">
      {overlayVisible && (
        <div className="action-overlay">
          <div className="action-overlay__content">
            <div className="action-spinner" />
            <p>{overlayMessage || "Working..."}</p>
          </div>
        </div>
      )}
      <header className="header">
        <div>
          <h1>Manual Cluster Helper</h1>
          <p>Keep the existing CLIP-powered clustering but control it through a React web UI.</p>
        </div>
        <div className="input-row">
          <label htmlFor="root-input">Root folder</label>
          <input
            id="root-input"
            value={rootInput}
            onChange={(event) => setRootInput(event.target.value)}
            placeholder="C:/path/to/photos"
          />
          <button type="button" onClick={reloadSession} disabled={!rootInput}>
            Refresh
          </button>
        </div>
        <div className="queue-actions">
          <button
            type="button"
            onClick={shuffleQueue}
            disabled={!session?.queue?.length || isShuffling}
          >
            {isShuffling ? "Shuffling queue…" : "Shuffle queue"}
          </button>
          <button
            type="button"
            onClick={undoLastMove}
            disabled={!session || isUndoing}
          >
            {isUndoing ? "Undoing…" : "Undo last move"}
          </button>
          {session?.queue_has_more && (
            <button
              type="button"
              onClick={loadMoreQueue}
              disabled={isLoadingMore}
            >
              {isLoadingMore ? "Loading queue…" : "Load more queue"}
            </button>
          )}
          {session && (
            <span className="queue-progress">
              Loaded {session.queue.length} / {session.queue_total ?? session.queue.length}
            </span>
          )}
        </div>
        <label className="manual-toggle">
          <input
            type="checkbox"
            checked={manualOnly}
            onChange={(event) => setManualOnly(event.target.checked)}
          />
          Manual-only mode (skip CLIP suggestions + auto-move)
        </label>
      </header>

      {errorMessage && <div className="alert error">{errorMessage}</div>}

      {isLoading && <div className="alert neutral">Loading queue...</div>}

      {!rootInput && <div className="alert neutral">Provide a root path to load images.</div>}

      {session && (
        <>
          {actionMessage && <div className="alert success">{actionMessage}</div>}

          <section className="preview-panel">
            <div className="preview-image">
              {previewUrl ? (
                <img src={previewUrl} alt="Current queue item" loading="lazy" />
              ) : (
                <div className="preview-placeholder">No image available</div>
              )}
            </div>
            <div className="preview-details">
              <p>
                <strong>Currently labeling</strong>
              </p>
              <code>{currentImage ?? "Queue is empty"}</code>
              {!manualOnly && session?.suggestion && (
                <div className="suggestion-panel">
                  <p>
                    <strong>Suggested cluster:</strong>{" "}
                    {session.suggestion.cluster} (
                    {session.suggestion.score ?? "–"})
                  </p>
                  <p className="suggestion-meta">
                    {session.suggestion.second_cluster
                      ? `Also: ${session.suggestion.second_cluster} (${session.suggestion.second_score})`
                      : "No secondary candidate yet."}{" "}
                    • Margin: {session.suggestion.margin ?? 0}
                  </p>
                </div>
              )}
              <div className="clustering-grid">
                {knownClusters.length ? (
                  knownClusters.map((clusterName) => (
                    <button
                      type="button"
                      className="cluster-chip"
                      key={clusterName}
                      onClick={() => assignCluster(clusterName)}
                    >
                      {clusterName}
                    </button>
                  ))
                ) : (
                  <p className="muted">No clusters defined yet.</p>
                )}
              </div>
              <form className="new-cluster-form" onSubmit={handleNewCluster}>
                <input
                  value={newCluster}
                  onChange={(event) => setNewCluster(event.target.value)}
                  placeholder="Create & assign"
                />
                <button type="submit" disabled={!newCluster.trim()}>
                  Assign new
                </button>
              </form>
              <div className="inline-actions">
                <button type="button" className="secondary" onClick={deleteImage}>
                  Delete
                </button>
                 <button
                   type="button"
                   className="secondary"
                   onClick={skipImage}
                   disabled={isSkipping}
                 >
                   {isSkipping ? "Skipping…" : "Skip"}
                 </button>
              </div>
              {!manualOnly ? (
                <div className="auto-controls">
                  <div className="auto-controls__row">
                    <label htmlFor="threshold-input">Threshold</label>
                    <input
                      id="threshold-input"
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      value={autoThreshold}
                      onChange={(event) => {
                        const value = parseFloat(event.target.value);
                        setAutoThreshold(
                          Number.isNaN(value) ? 0 : Math.min(1, Math.max(0, value))
                        );
                      }}
                    />
                  </div>
                  <div className="auto-controls__row">
                    <label htmlFor="min-samples-input">Min samples</label>
                    <input
                      id="min-samples-input"
                      type="number"
                      min="1"
                      step="1"
                      value={autoMinSamples}
                      onChange={(event) => {
                        const value = parseInt(event.target.value, 10);
                        setAutoMinSamples(value > 0 ? value : 1);
                      }}
                    />
                  </div>
                  <button
                    type="button"
                    className="auto-controls__button"
                    onClick={triggerAutoMove}
                    disabled={isAutoMoving || !session?.queue?.length}
                  >
                    {isAutoMoving ? "Auto-moving…" : "Auto-move"}
                  </button>
                </div>
              ) : (
                <div className="manual-mode-note">
                  Manual-only mode is enabled; CLIP suggestions and auto-move are paused.
                </div>
              )}
            </div>
          </section>

          <section className="stats-panel">
            <article>
              <h2>Progress</h2>
              <dl>
                <div>
                  <dt>Manual labeled</dt>
                  <dd>{formatNumber(stats.manual_labeled ?? 0)}</dd>
                </div>
                <div>
                  <dt>Deleted</dt>
                  <dd>{formatNumber(stats.deleted ?? 0)}</dd>
                </div>
                <div>
                  <dt>Queue</dt>
                  <dd>{formatNumber(stats.queue_length ?? 0)}</dd>
                </div>
                <div>
                  <dt>Processed %</dt>
                  <dd>{stats.processed_pct ?? 0}%</dd>
                </div>
              </dl>
            </article>
            <article>
              <h2>Cluster breakdown</h2>
              <div className="counts-list">
                {Object.entries(stats.cluster_counts ?? {}).map(([name, count]) => (
                  <div key={name} className="counts-list__row">
                    <span>{name}</span>
                    <span>{formatNumber(count)}</span>
                  </div>
                ))}
              </div>
            </article>
            <article>
              <h2>.clustering snapshot</h2>
              <dl>
                <div>
                  <dt>Clusters</dt>
                  <dd>{stats.state_summary?.clusters ?? 0}</dd>
                </div>
                <div>
                  <dt>Auto total</dt>
                  <dd>{formatNumber(stats.state_summary?.auto_total ?? 0)}</dd>
                </div>
                <div>
                  <dt>Manual total</dt>
                  <dd>{formatNumber(stats.state_summary?.manual_total ?? 0)}</dd>
                </div>
              </dl>
            </article>
          </section>
        </>
      )}
    </main>
  );
}

export default App;
