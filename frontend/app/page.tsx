"use client";

import { ChangeEvent, FormEvent, useCallback, useEffect, useMemo, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Telemetry = {
  utilization_percent?: number | null;
  memory_total_bytes?: number | null;
  memory_used_bytes?: number | null;
  memory_free_bytes?: number | null;
  temperature_c?: number | null;
};

type Device = {
  backend: string;
  index: number | null;
  id: string;
  name: string;
  vendor: string;
  device_type: string;
  healthy: boolean;
  experimental?: boolean;
  telemetry?: Telemetry | null;
};

type Worker = {
  worker_id: string;
  backend?: string;
  status: string;
  healthy: boolean;
  devices: Device[];
  active_jobs: number;
  engines: string[];
};

type AcceleratorInventory = {
  selected_policy: string;
  priority: string[];
  backends: Array<{ backend: string; available: boolean; experimental: boolean; devices: Device[] }>;
  workers: Worker[];
};

type Compatibility = {
  combinations: Array<{
    engine: string;
    backend: string;
    worker_id: string;
    devices: Device[];
    precisions: string[];
    experimental: boolean;
    available: boolean;
  }>;
};

type Validation = {
  valid: boolean;
  backend?: string;
  precision?: string;
  worker_id?: string;
  issues: Array<{ code: string; message: string }>;
};

type Output = { format: string; size_bytes: number; download_url: string };
type Job = {
  id: string;
  filename: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  stage: string;
  progress: number;
  parameters: Record<string, unknown>;
  device_used?: string;
  gpu_name?: string;
  engine_requested: string;
  engine_used?: string;
  device_requested: { policy?: string; backend?: string; device_index?: number; allow_fallback?: boolean };
  backend_used?: string;
  runtime_device_used?: string;
  precision_requested: string;
  precision_used?: string;
  fallback_used: boolean;
  fallback_reason?: string;
  worker_id?: string;
  error_code?: string;
  error?: string;
  runtime_seconds?: number;
  outputs: Output[];
};

const outputOptions = ["json", "txt", "srt", "vtt", "tsv"];
const modelOptions = ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en", "large", "large-v2", "large-v3", "turbo"];

function bytes(value: number) {
  if (!value) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB"];
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
  return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

export default function Home() {
  const [inventory, setInventory] = useState<AcceleratorInventory>({ selected_policy: "auto", priority: [], backends: [], workers: [] });
  const [compatibility, setCompatibility] = useState<Compatibility>({ combinations: [] });
  const [validation, setValidation] = useState<Validation | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);

  const [model, setModel] = useState("base");
  const [engine, setEngine] = useState("openai-whisper");
  const [language, setLanguage] = useState("");
  const [task, setTask] = useState("transcribe");
  const [devicePolicy, setDevicePolicy] = useState("auto");
  const [backend, setBackend] = useState("");
  const [deviceIndex, setDeviceIndex] = useState("");
  const [precision, setPrecision] = useState("auto");
  const [allowFallback, setAllowFallback] = useState(true);
  const [temperature, setTemperature] = useState(0);
  const [beamSize, setBeamSize] = useState("");
  const [bestOf, setBestOf] = useState("");
  const [patience, setPatience] = useState("");
  const [initialPrompt, setInitialPrompt] = useState("");
  const [wordTimestamps, setWordTimestamps] = useState(true);
  const [conditionPrevious, setConditionPrevious] = useState(true);
  const [threads, setThreads] = useState("");
  const [formats, setFormats] = useState(["json", "txt", "srt"]);

  const refresh = useCallback(async () => {
    const [inventoryResult, compatibilityResult, jobsResult] = await Promise.allSettled([
      fetch(`${API_URL}/system/accelerators`, { cache: "no-store" }).then((response) => response.json()),
      fetch(`${API_URL}/system/compatibility`, { cache: "no-store" }).then((response) => response.json()),
      fetch(`${API_URL}/jobs?limit=50`, { cache: "no-store" }).then((response) => response.json()),
    ]);
    if (inventoryResult.status === "fulfilled") setInventory(inventoryResult.value);
    if (compatibilityResult.status === "fulfilled") setCompatibility(compatibilityResult.value);
    if (jobsResult.status === "fulfilled") setJobs(jobsResult.value.items || []);
  }, []);

  useEffect(() => {
    refresh();
    const interval = window.setInterval(refresh, 5000);
    return () => window.clearInterval(interval);
  }, [refresh]);

  useEffect(() => {
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const response = await fetch(`${API_URL}/jobs/validate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          signal: controller.signal,
          body: JSON.stringify({
            engine,
            model,
            device: {
              policy: devicePolicy,
              backend: devicePolicy === "manual" ? backend || null : null,
              device_index: devicePolicy === "manual" && deviceIndex !== "" ? Number(deviceIndex) : null,
              allow_fallback: allowFallback,
            },
            precision,
          }),
        });
        if (response.ok) {
          setValidation(await response.json());
        } else {
          setValidation({
            valid: false,
            issues: [{ code: "VALIDATION_UNAVAILABLE", message: `Validation API returned HTTP ${response.status}.` }],
          });
        }
      } catch (reason) {
        if (!(reason instanceof DOMException && reason.name === "AbortError")) {
          setValidation({
            valid: false,
            issues: [{ code: "VALIDATION_UNAVAILABLE", message: `Could not reach the validation API at ${API_URL}.` }],
          });
        }
      }
    }, 200);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [engine, model, devicePolicy, backend, deviceIndex, precision, allowFallback]);

  const activeJobs = useMemo(() => jobs.filter((job) => ["queued", "running"].includes(job.status)).length, [jobs]);
  const healthyWorkers = inventory.workers.filter((worker) => worker.healthy);
  const activeWorker = healthyWorkers[0];
  const activeDevice = activeWorker?.devices[0];
  const telemetry = activeDevice?.telemetry;
  const compatibleBackends = useMemo(
    () => Array.from(new Set(compatibility.combinations.filter((item) => item.engine === engine && item.available).map((item) => item.backend))),
    [compatibility, engine],
  );
  const selectedCombination = compatibility.combinations.find((item) => item.engine === engine && item.backend === backend && item.available);
  const precisionOptions = Array.from(new Set(selectedCombination?.precisions || ["float16", "float32"]));
  const selectableDevices = selectedCombination?.devices || [];

  function addFiles(incoming: FileList | File[]) {
    const allowed = /\.(mp3|wav|flac|ogg|m4a|aac|mp4|mkv|mov|avi|webm)$/i;
    setFiles((current) => {
      const merged = [...current, ...Array.from(incoming).filter((file) => allowed.test(file.name))];
      return merged.filter((file, index) => merged.findIndex((item) => item.name === file.name && item.size === file.size) === index).slice(0, 100);
    });
  }

  function onFiles(event: ChangeEvent<HTMLInputElement>) {
    if (event.target.files) addFiles(event.target.files);
  }

  function toggleFormat(format: string) {
    setFormats((current) => current.includes(format) ? current.filter((item) => item !== format) : [...current, format]);
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    if (!files.length) return setError("Select at least one audio or video file.");
    if (!formats.length) return setError("Select at least one output format.");
    if (task === "translate" && model === "turbo") return setError("Turbo does not reliably support translation. Choose another model.");
    if (!validation?.valid) return setError(validation?.issues?.[0]?.message || "No compatible worker is available for this request.");
    setSubmitting(true);
    setError("");
    try {
      const form = new FormData();
      files.forEach((file) => form.append("files", file));
      const uploadResponse = await fetch(`${API_URL}/uploads`, { method: "POST", body: form });
      if (!uploadResponse.ok) throw new Error((await uploadResponse.text()) || "Upload failed");
      const uploads = await uploadResponse.json();
      const parameters: Record<string, unknown> = {
        model, engine, language: language || null, task,
        device: devicePolicy === "manual" ? backend : "auto",
        precision, allow_fallback: allowFallback,
        fp16: precision === "float16" ? true : null, temperature,
        beam_size: beamSize ? Number(beamSize) : null,
        best_of: temperature > 0 && bestOf ? Number(bestOf) : null,
        patience: patience ? Number(patience) : null,
        initial_prompt: initialPrompt || null,
        condition_on_previous_text: conditionPrevious,
        word_timestamps: wordTimestamps,
        threads: backend === "cpu" && threads ? Number(threads) : null,
        output_formats: formats,
      };
      const jobResponse = await fetch(`${API_URL}/jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          upload_ids: uploads.map((upload: { id: string }) => upload.id),
          engine,
          model,
          device: {
            policy: devicePolicy,
            backend: devicePolicy === "manual" ? backend : null,
            device_index: devicePolicy === "manual" && deviceIndex !== "" ? Number(deviceIndex) : null,
            allow_fallback: allowFallback,
          },
          precision,
          parameters,
        }),
      });
      if (!jobResponse.ok) throw new Error((await jobResponse.text()) || "Could not create jobs");
      setFiles([]);
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Request failed");
    } finally {
      setSubmitting(false);
    }
  }

  async function cancel(jobId: string) {
    await fetch(`${API_URL}/jobs/${jobId}/cancel`, { method: "POST" });
    await refresh();
  }

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#"><span className="mark">W</span><span>Whisper Docker</span></a>
        <div className={`connection ${healthyWorkers.length ? "online" : "offline"}`}>
          <span className="pulse" /> {healthyWorkers.length ? `${healthyWorkers.length} worker ready` : "No worker"}
        </div>
      </header>

      <section className="hero shell">
        <div>
          <p className="eyebrow">MULTI-ACCELERATOR CONTROL ROOM</p>
          <h1>Turn long recordings into<br /><em>clean, useful text.</em></h1>
          <p className="lede">A private Whisper pipeline for CUDA, ROCm, Intel, Apple Silicon, and CPU workers. Queue media, choose a compatible runtime, and keep every result local.</p>
        </div>
        <div className="hero-orbit" aria-hidden="true"><div className="core">{activeWorker?.backend?.toUpperCase() || "AUTO"}</div></div>
      </section>

      <section className="metrics shell">
        <article><span>Backend</span><strong>{activeWorker?.backend?.toUpperCase() || "Offline"}</strong><small>{activeWorker ? `${activeWorker.worker_id} · ${activeWorker.status}` : "Awaiting a fresh heartbeat"}</small></article>
        <article><span>Device</span><strong>{activeDevice?.name || "Not detected"}</strong><small>{activeDevice ? `${activeDevice.vendor} · ${activeDevice.device_type}` : "Start a backend worker"}</small></article>
        <article><span>Telemetry</span><strong>{telemetry?.memory_free_bytes != null ? `${bytes(telemetry.memory_free_bytes)} free` : "Unavailable"}</strong><small>{telemetry?.temperature_c != null ? `${telemetry.temperature_c}°C${telemetry.utilization_percent != null ? ` · ${telemetry.utilization_percent}% utilization` : ""}` : "Metric not exposed by runtime"}</small></article>
        <article><span>Queue</span><strong>{activeJobs} active</strong><small>{jobs.length} recent jobs</small></article>
      </section>

      {!healthyWorkers.length && (
        <aside className="warning shell"><b>Jobs are safely paused.</b> No fresh compatible worker heartbeat is available. Start the appropriate accelerator profile after its host preflight passes.</aside>
      )}

      <div className="workspace shell">
        <form className="panel composer" onSubmit={submit}>
          <div className="panel-heading"><div><p className="kicker">NEW BATCH</p><h2>Configure transcription</h2></div><span className="step">01</span></div>

          <label
            className={`dropzone ${dragging ? "dragging" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => { event.preventDefault(); setDragging(false); addFiles(event.dataTransfer.files); }}
          >
            <input type="file" multiple accept="audio/*,video/*,.mkv" onChange={onFiles} />
            <span className="upload-icon">↑</span>
            <b>Drop media here</b>
            <small>or browse · up to 100 audio/video files</small>
          </label>

          {files.length > 0 && <div className="file-list">{files.map((file, index) => <div key={`${file.name}-${file.size}`}><span>{file.name}<small>{bytes(file.size)}</small></span><button type="button" onClick={() => setFiles((items) => items.filter((_, itemIndex) => itemIndex !== index))}>×</button></div>)}</div>}

          <div className="form-grid">
            <label>Engine<select value={engine} onChange={(event) => setEngine(event.target.value)}><option value="openai-whisper">OpenAI Whisper</option></select></label>
            <label>Model<select value={model} onChange={(event) => setModel(event.target.value)}>{modelOptions.map((item) => <option key={item}>{item}</option>)}</select></label>
            <label>Language<input value={language} onChange={(event) => setLanguage(event.target.value)} placeholder="Auto detect" /></label>
            <label>Task<select value={task} onChange={(event) => setTask(event.target.value)}><option value="transcribe">Transcribe</option><option value="translate">Translate to English</option></select></label>
            <label>Device policy<select value={devicePolicy} onChange={(event) => { setDevicePolicy(event.target.value); setDeviceIndex(""); }}><option value="auto">Auto select</option><option value="manual">Manual</option></select></label>
            <label>Backend<select value={backend} disabled={devicePolicy === "auto"} onChange={(event) => { setBackend(event.target.value); setDeviceIndex(""); setPrecision("auto"); }}><option value="">Select backend</option>{inventory.priority.map((item) => <option key={item} value={item} disabled={!compatibleBackends.includes(item)}>{item.toUpperCase()}{!compatibleBackends.includes(item) ? " · unavailable" : ""}</option>)}</select></label>
            <label>Device<select value={deviceIndex} disabled={devicePolicy === "auto" || !selectableDevices.length} onChange={(event) => setDeviceIndex(event.target.value)}><option value="">Default device</option>{selectableDevices.map((item) => <option key={item.id} value={item.index ?? ""}>{item.name}</option>)}</select></label>
            <label>Precision<select value={precision} onChange={(event) => setPrecision(event.target.value)}><option value="auto">Auto</option>{precisionOptions.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
          </div>

          <div className="switches execution"><label><input type="checkbox" checked={allowFallback} onChange={(event) => setAllowFallback(event.target.checked)} /> Allow explicit fallback</label><span className={validation?.valid ? "valid" : "invalid"}>{validation?.valid ? `Will use ${validation.backend} · ${validation.precision}` : validation?.issues?.[0]?.message || "Checking compatibility…"}</span></div>

          <fieldset><legend>Output formats</legend><div className="checks">{outputOptions.map((format) => <label key={format}><input type="checkbox" checked={formats.includes(format)} onChange={() => toggleFormat(format)} /><span>{format.toUpperCase()}</span></label>)}</div></fieldset>

          <details>
            <summary>Advanced decoding</summary>
            <div className="form-grid advanced">
              <label>Temperature <output>{temperature.toFixed(1)}</output><input type="range" min="0" max="1" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} /></label>
              <label>Beam size<input type="number" min="1" value={beamSize} onChange={(event) => setBeamSize(event.target.value)} placeholder="Automatic" /></label>
              <label>Best of<input type="number" min="1" value={bestOf} onChange={(event) => setBestOf(event.target.value)} disabled={temperature === 0} placeholder="Temperature > 0" /></label>
              <label>Patience<input type="number" min="0" step="0.1" value={patience} onChange={(event) => setPatience(event.target.value)} placeholder="Default" /></label>
              {(backend === "cpu" || devicePolicy === "auto") && <label>CPU threads<input type="number" min="1" value={threads} onChange={(event) => setThreads(event.target.value)} placeholder="Used only on CPU" /></label>}
              <label className="span-2">Initial prompt<textarea value={initialPrompt} onChange={(event) => setInitialPrompt(event.target.value)} rows={3} placeholder="Vocabulary, names, or context to guide Whisper…" /></label>
            </div>
            <div className="switches"><label><input type="checkbox" checked={wordTimestamps} onChange={(event) => setWordTimestamps(event.target.checked)} /> Word timestamps</label><label><input type="checkbox" checked={conditionPrevious} onChange={(event) => setConditionPrevious(event.target.checked)} /> Condition on previous text</label></div>
          </details>

          {error && <p className="error">{error}</p>}
          <button className="primary" disabled={submitting || !files.length}>{submitting ? "Uploading and queuing…" : `Queue ${files.length || ""} ${files.length === 1 ? "file" : "files"}`}</button>
        </form>

        <section className="panel jobs">
          <div className="panel-heading"><div><p className="kicker">ACTIVITY</p><h2>Recent jobs</h2></div><button className="quiet" onClick={refresh}>Refresh</button></div>
          {!jobs.length && <div className="empty"><span>◎</span><h3>The queue is clear</h3><p>Your transcription jobs will appear here.</p></div>}
          <div className="job-list">
            {jobs.map((job) => (
              <article className="job" key={job.id}>
                <div className="job-main"><div className={`status-dot ${job.status}`} /><div className="job-name"><b>{job.filename}</b><span>{job.stage.replaceAll("_", " ")} · {(job.parameters.model as string) || "base"} · requested {job.device_requested?.backend || "auto"} → {job.backend_used || "pending"}{job.precision_used ? ` · ${job.precision_used}` : ""}</span></div><span className={`badge ${job.status}`}>{job.status}</span></div>
                <div className="progress"><i style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
                <div className="job-foot"><span>{Math.round(job.progress * 100)}%{job.runtime_seconds ? ` · ${job.runtime_seconds.toFixed(1)}s` : ""}{job.gpu_name ? ` · ${job.gpu_name}` : ""}</span><div>{["queued", "running"].includes(job.status) && <button onClick={() => cancel(job.id)}>Cancel</button>}{job.outputs.map((output) => <a key={output.format} href={`${API_URL}${output.download_url}`}>{output.format.toUpperCase()}</a>)}</div></div>
                {job.error && <p className="job-error">{job.error}</p>}
                {job.fallback_used && <p className="job-fallback">Fallback: {job.fallback_reason || "A compatible worker was selected."}</p>}
              </article>
            ))}
          </div>
        </section>
      </div>

      <footer className="shell"><span>WHISPER DOCKER · LOCAL FIRST</span><span>capability routed · no privileged workers</span></footer>
    </main>
  );
}
