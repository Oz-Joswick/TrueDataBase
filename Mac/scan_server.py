"""
face_system/mac/scan_server.py
───────────────────────────────
HTTPS server for camera-based face identification.
Runs as its own process (spawned by main.py).

Access from any device on the LAN:
    https://<host-ip>:8766

The browser will warn about the self-signed certificate —
click Advanced → Proceed to continue. This is expected for
self-signed certs and is safe on a trusted local network.
"""

import json
import os
import socket
import ssl
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import httpx
import numpy as np

from config import (
    PI_URL,
    PI_IDENTIFY_THRESHOLD,
    SCAN_SERVER_PORT,
    SCAN_SERVER_HOST,
)
from detector import init_worker_model, detect


# ── Network helpers ────────────────────────────────────────────────────────────

def _local_ip() -> str:
    """Best-guess LAN IP of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# ── Self-signed TLS cert ───────────────────────────────────────────────────────

def _generate_cert(local_ip: str):
    """
    Generate a self-signed TLS certificate covering localhost and local_ip.
    Saves to the system temp directory and returns (cert_path, key_path).
    Regenerated on every server start so the SAN always matches the current IP.
    Raises ImportError if 'cryptography' is not installed.
    """
    import ipaddress
    from datetime import datetime, timezone, timedelta
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "face_system")])

    san_entries = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    if local_ip not in ("127.0.0.1", "localhost"):
        san_entries.append(x509.IPAddress(ipaddress.IPv4Address(local_ip)))

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(key, hashes.SHA256())
    )

    tmp   = tempfile.gettempdir()
    cpath = os.path.join(tmp, "face_system_scan.crt")
    kpath = os.path.join(tmp, "face_system_scan.key")

    with open(cpath, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(kpath, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))

    return cpath, kpath


# ── Camera page HTML ───────────────────────────────────────────────────────────

_CAMERA_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
<title>face_system &middot; scan</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0d1117; --surface: #161b22; --surface2: #1f2937;
    --border: #30363d; --text: #e6edf3; --muted: #7d8590;
    --green: #3fb950; --yellow: #d29922; --red: #f85149;
    --blue: #58a6ff; --purple: #bc8cff;
    --font: 'Segoe UI', system-ui, sans-serif;
    --mono: 'Cascadia Code', Consolas, monospace;
  }
  body {
    background: var(--bg); color: var(--text);
    font-family: var(--font); min-height: 100svh;
    display: flex; flex-direction: column; align-items: center;
  }
  header {
    width: 100%; background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 14px 20px; display: flex; align-items: center; gap: 10px;
    position: sticky; top: 0; z-index: 10;
  }
  .logo { font-family: var(--mono); font-size: 15px; font-weight: 700; }
  .sub  { color: var(--muted); font-size: 12px; }
  #app  { width: 100%; max-width: 520px; padding: 20px 16px;
          display: flex; flex-direction: column; gap: 16px; }

  /* Camera box */
  .camera-box {
    position: relative; background: #000; border-radius: 12px;
    overflow: hidden; border: 1px solid var(--border);
    aspect-ratio: 4 / 3;
  }
  #video, #canvas {
    width: 100%; height: 100%; object-fit: cover; display: block;
    border-radius: 12px;
  }
  #canvas { display: none; position: absolute; top: 0; left: 0; }
  #cam-error {
    display: none; position: absolute; inset: 0;
    background: var(--surface); flex-direction: column;
    align-items: center; justify-content: center;
    gap: 12px; text-align: center; padding: 24px;
  }
  #cam-error.show { display: flex; }
  #cam-error .icon { font-size: 40px; }
  #cam-error p { font-size: 12px; color: var(--muted); line-height: 1.6; }

  /* File-input fallback mode */
  #file-placeholder {
    display: none; position: absolute; inset: 0;
    background: var(--surface); flex-direction: column;
    align-items: center; justify-content: center;
    gap: 14px; text-align: center; padding: 24px; cursor: pointer;
    transition: background 0.15s;
  }
  #file-placeholder:hover { background: var(--surface2); }
  #file-placeholder.show  { display: flex; }
  #file-placeholder .fp-icon { font-size: 44px; }
  #file-placeholder .fp-label { font-size: 15px; font-weight: 600; }
  #file-placeholder .fp-sub   { font-size: 11px; color: var(--muted); line-height: 1.5; }

  /* Snap button */
  #snap-btn {
    width: 100%; padding: 15px; border-radius: 10px;
    background: var(--blue); color: #000;
    border: none; font-size: 15px; font-weight: 700;
    cursor: pointer; display: flex; align-items: center;
    justify-content: center; gap: 10px;
    transition: opacity 0.15s, transform 0.1s;
    font-family: var(--font);
  }
  #snap-btn:active { transform: scale(0.98); }
  #snap-btn:disabled { opacity: 0.45; cursor: not-allowed; }
  #snap-btn.busy { background: var(--surface2); color: var(--muted);
                   border: 1px solid var(--border); }
  .spinner {
    width: 18px; height: 18px; border: 2px solid var(--border);
    border-top-color: var(--blue); border-radius: 50%;
    animation: spin 0.7s linear infinite; display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Results */
  #results { display: flex; flex-direction: column; gap: 10px; }
  .result-header {
    display: flex; align-items: center; gap: 8px;
    font-size: 11px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600;
  }
  .result-header hr { flex: 1; border: none; border-top: 1px solid var(--border); }

  .face-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .face-card.known   { border-color: rgba(63,185,80,0.45); }
  .face-card.unknown { border-color: var(--border); }
  .face-card.error   { border-color: rgba(248,81,73,0.3); }

  .face-top { display: flex; align-items: center; gap: 12px; }
  .face-avatar {
    width: 44px; height: 44px; border-radius: 50%;
    background: var(--surface2); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-size: 15px; font-weight: 700; font-family: var(--mono);
    color: var(--muted); flex-shrink: 0;
  }
  .face-avatar.known   { color: var(--green); border-color: rgba(63,185,80,0.45); }
  .face-avatar.error   { color: var(--red);   border-color: rgba(248,81,73,0.3); }
  .face-info { flex: 1; min-width: 0; }
  .face-name {
    font-size: 16px; font-weight: 700; font-family: var(--mono);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .face-name.known   { color: var(--green); }
  .face-name.unknown { color: var(--muted); }
  .face-name.error   { color: var(--red); }
  .face-sub { font-size: 11px; color: var(--muted); margin-top: 3px;
              display: flex; align-items: center; gap: 6px; }

  .badge {
    display: inline-block; font-size: 9px; padding: 2px 6px;
    border-radius: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.4px;
  }
  .badge.known   { background: rgba(63,185,80,0.15);    color: var(--green); }
  .badge.unknown { background: rgba(125,133,144,0.12);  color: var(--muted); }
  .badge.error   { background: rgba(248,81,73,0.12);    color: var(--red); }

  .conf-row { display: flex; align-items: center; gap: 10px; }
  .conf-bar { flex: 1; height: 6px; background: var(--surface2); border-radius: 3px; overflow: hidden; }
  .conf-fill { height: 100%; border-radius: 3px; transition: width 0.6s ease; }
  .conf-fill.known   { background: var(--green); }
  .conf-fill.unknown { background: var(--muted); }
  .conf-fill.error   { background: var(--red);   }
  .conf-pct { font-size: 14px; font-weight: 700; font-family: var(--mono);
              min-width: 44px; text-align: right; }
  .conf-pct.known   { color: var(--green); }
  .conf-pct.unknown { color: var(--muted); }
  .conf-pct.error   { color: var(--red); }

  #no-faces {
    text-align: center; padding: 28px; color: var(--muted);
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; font-size: 13px; line-height: 1.5;
  }
  #again-btn {
    display: none; width: 100%; padding: 13px; border-radius: 10px;
    background: var(--surface2); color: var(--text);
    border: 1px solid var(--border); font-size: 14px; font-weight: 600;
    cursor: pointer; font-family: var(--font);
  }
  #again-btn:hover { border-color: var(--blue); color: var(--blue); }
  #again-btn.show { display: block; }

  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* ── Status bar ─────────────────────────────────────────── */
  #status-bar {
    width: 100%; padding: 8px 20px; font-size: 12px;
    display: flex; align-items: center; gap: 8px;
    border-bottom: 1px solid var(--border);
    transition: background 0.4s;
  }
  #status-bar.online  { background: rgba(63,185,80,0.07); }
  #status-bar.offline { background: rgba(248,81,73,0.07); }
  #status-bar.warning { background: rgba(210,153,34,0.07); }
  #status-bar.checking{ background: transparent; }
  #status-dot {
    width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0;
    background: var(--muted); transition: background 0.3s;
  }
  #status-dot.online  { background: var(--green); }
  #status-dot.offline { background: var(--red);    animation: pulse 2s infinite; }
  #status-dot.warning { background: var(--yellow); animation: pulse 1.5s infinite; }
  #status-text { flex: 1; color: var(--muted); }
  #status-bar.online  #status-text { color: var(--green); }
  #status-bar.offline #status-text { color: var(--red); }
  #status-bar.warning #status-text { color: var(--yellow); }

  /* ── Warning toasts ─────────────────────────────────────── */
  #warn-container {
    position: fixed; top: 110px; left: 50%; transform: translateX(-50%);
    width: calc(100% - 32px); max-width: 500px;
    display: flex; flex-direction: column; gap: 8px;
    z-index: 500; pointer-events: none;
  }
  .w-toast {
    background: var(--surface2); border-radius: 8px;
    padding: 11px 14px; font-size: 12px; line-height: 1.5;
    display: flex; align-items: flex-start; gap: 10px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.5);
    pointer-events: auto;
    animation: wSlide 0.25s ease;
  }
  .w-toast.warn  { border-left: 3px solid var(--yellow); }
  .w-toast.error { border-left: 3px solid var(--red); }
  .w-toast.info  { border-left: 3px solid var(--blue); }
  .w-icon  { flex-shrink: 0; font-size: 15px; }
  .w-body  { flex: 1; color: var(--text); }
  .w-title { font-weight: 700; margin-bottom: 2px; }
  .w-msg   { color: var(--muted); font-size: 11px; }
  .w-close { flex-shrink: 0; color: var(--muted); cursor: pointer; font-size: 18px; line-height: 1; }
  .w-close:hover { color: var(--text); }
  @keyframes wSlide {
    from { opacity: 0; transform: translateY(-10px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ── Disconnect overlay ─────────────────────────────────── */
  #disconnect-overlay {
    display: none; position: fixed; inset: 0;
    background: rgba(13,17,23,0.96); z-index: 999;
    flex-direction: column; align-items: center;
    justify-content: center; gap: 16px;
    text-align: center; padding: 40px;
  }
  #disconnect-overlay.show { display: flex; }
  #disconnect-overlay .d-icon { font-size: 52px; }
  #disconnect-overlay h2 { font-size: 22px; color: var(--red); font-weight: 700; }
  #disconnect-overlay p  { color: var(--muted); font-size: 13px; line-height: 1.7; max-width: 320px; }
  #reload-btn {
    background: var(--blue); color: #000; border: none;
    border-radius: 8px; padding: 13px 28px; font-size: 14px;
    font-weight: 700; cursor: pointer; font-family: var(--font);
    margin-top: 4px;
  }
</style>
</head>
<body>

<header>
  <span class="logo">face_system</span>
  <span class="sub">&middot; scan</span>
</header>

<!-- Status bar -->
<div id="status-bar" class="checking">
  <div id="status-dot"></div>
  <span id="status-text">Checking connection...</span>
</div>

<!-- Warning toasts -->
<div id="warn-container"></div>

<!-- Disconnect overlay -->
<div id="disconnect-overlay">
  <div class="d-icon">&#9889;</div>
  <h2>Connection Lost</h2>
  <p>The scan server stopped responding.<br>Make sure face_system is still running on the host machine, then reload.</p>
  <button id="reload-btn" onclick="location.reload()">Reload Page</button>
</div>

<div id="app">

  <div class="camera-box">
    <video id="video" autoplay playsinline muted></video>
    <canvas id="canvas"></canvas>
    <div id="cam-error">
      <div class="icon">&#128247;</div>
      <p id="cam-msg">Allow camera access in your browser, then reload.</p>
    </div>
    <!-- Tappable overlay used in file-input fallback mode -->
    <div id="file-placeholder" onclick="triggerFilePick()">
      <div class="fp-icon">&#128247;</div>
      <div class="fp-label">Tap to take a photo</div>
      <div class="fp-sub">Opens your device camera or photo library</div>
    </div>
  </div>

  <!-- Hidden file input — works on iOS/Android over plain HTTP -->
  <input type="file" id="file-input" accept="image/*" capture="user" style="display:none">

  <button id="snap-btn" onclick="snap()">
    <span id="snap-dot">&#11044;</span>
    <span id="snap-label">Snap Photo</span>
    <div class="spinner" id="spinner"></div>
  </button>

  <div id="results"></div>
  <button id="again-btn" onclick="again()">&#8635; Scan Again</button>

</div>

<script>
const video       = document.getElementById('video');
const canvas      = document.getElementById('canvas');
const snapBtn     = document.getElementById('snap-btn');
const results     = document.getElementById('results');
const againBtn    = document.getElementById('again-btn');
const camErr      = document.getElementById('cam-error');
const camMsg      = document.getElementById('cam-msg');
const fileInput   = document.getElementById('file-input');
const filePlaceholder = document.getElementById('file-placeholder');

// Track which mode we're in
let _fileMode = false;

// ── File input fallback (iOS / HTTP) ──────────────────────────

function triggerFilePick() {
  fileInput.click();
}

fileInput.addEventListener('change', () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  const img  = new Image();
  img.onload = () => {
    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext('2d').drawImage(img, 0, 0);
    URL.revokeObjectURL(url);

    // Show captured image, hide placeholder
    canvas.style.display      = 'block';
    filePlaceholder.classList.remove('show');

    // Auto-submit
    submitCanvas();
  };
  img.src = url;

  // Reset so same file can be picked again after "Scan Again"
  fileInput.value = '';
});

function _enterFileMode() {
  _fileMode = true;
  video.style.display = 'none';
  filePlaceholder.classList.add('show');
  snapBtn.querySelector('#snap-label').textContent = 'Take / Choose Photo';
  snapBtn.querySelector('#snap-dot').textContent   = '&#128247;';
}

// ── getUserMedia (desktop / HTTPS) ────────────────────────────

async function startCamera() {
  // getUserMedia requires secure context (HTTPS or localhost)
  const secureCtx = location.protocol === 'https:' || location.hostname === 'localhost';
  const hasApi    = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

  if (!secureCtx || !hasApi) {
    _enterFileMode();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    video.style.display = 'block';
  } catch (err) {
    // getUserMedia failed — fall back to file input silently
    _enterFileMode();
    if (err.name === 'NotAllowedError') {
      showWarn('Camera Denied',
        'Camera permission was denied. You can still tap "Take / Choose Photo" to use your device camera.',
        'warn', 8000);
    } else if (err.name === 'NotFoundError') {
      showWarn('No Camera Found',
        'No camera detected. You can still choose a photo from your library.', 'info', 6000);
    }
  }
}

// ── Snap (live mode) ──────────────────────────────────────────

async function snap() {
  if (_fileMode) {
    // In file mode: trigger the picker; submitCanvas() is called from the change handler
    triggerFilePick();
    return;
  }

  // Live video mode: capture current frame
  const ctx = canvas.getContext('2d');
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.style.display = 'block';
  video.style.display  = 'none';
  submitCanvas();
}

// ── Shared submit logic ───────────────────────────────────────

async function submitCanvas() {
  setBusy(true);
  results.innerHTML = '';
  againBtn.classList.remove('show');

  if (_piOnline === false) {
    showWarn('Pi Offline', 'Results will show errors — identification requires the Pi to be online.', 'error', 6000);
  } else if (_dbPeople === 0) {
    showWarn('Empty Database', 'No people are registered yet. Faces will show as Unknown.', 'warn', 6000);
  }

  canvas.toBlob(async (blob) => {
    try {
      const ctrl = new AbortController();
      const tid  = setTimeout(() => ctrl.abort(), 30000);
      const resp = await fetch('/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'image/jpeg' },
        body: blob,
        signal: ctrl.signal,
      });
      clearTimeout(tid);
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ error: 'Server error ' + resp.status }));
        throw new Error(err.error || 'Server error ' + resp.status);
      }
      renderResults(await resp.json());
    } catch (e) {
      if (e.name === 'AbortError') {
        results.innerHTML = `<div id="no-faces" style="color:var(--red)">&#9888; Request timed out — server may be overloaded.</div>`;
        showWarn('Timeout', 'The scan request took too long. Try again.', 'error');
      } else if (!navigator.onLine || e.message.includes('fetch')) {
        document.getElementById('disconnect-overlay').classList.add('show');
      } else {
        results.innerHTML = `<div id="no-faces" style="color:var(--red)">&#9888; ${esc(e.message)}</div>`;
        showWarn('Scan Error', e.message, 'error');
      }
    } finally {
      setBusy(false);
      againBtn.classList.add('show');
    }
  }, 'image/jpeg', 0.92);
}

function renderResults(data) {
  if (!data.faces || data.faces.length === 0) {
    results.innerHTML = '<div id="no-faces">No faces detected in this photo.<br>'
      + '<small>Try better lighting or move closer to the camera.</small></div>';
    return;
  }

  const header = `<div class="result-header">
    <hr>
    ${data.faces.length} face${data.faces.length !== 1 ? 's' : ''} detected
    <hr>
  </div>`;

  const cards = data.faces.map(f => {
    const type     = f.type || 'unknown';
    const label    = f.label || 'Unknown';
    const initials = label.replace(/[^A-Za-z0-9]/g, '').substring(0, 2).toUpperCase() || '?';
    const pct      = f.confidence != null ? Math.round(f.confidence * 100) : null;
    const pctStr   = pct != null ? pct + '%' : '\u2014';
    const fillW    = pct != null ? pct : 0;
    const pid      = f.person_id ? `<span style="font-size:10px;color:var(--muted)">ID&nbsp;${f.person_id}</span>` : '';

    return `<div class="face-card ${type}">
      <div class="face-top">
        <div class="face-avatar ${type}">${esc(initials)}</div>
        <div class="face-info">
          <div class="face-name ${type}">${esc(label)}</div>
          <div class="face-sub">
            <span class="badge ${type}">${type}</span>
            ${pid}
          </div>
        </div>
      </div>
      <div class="conf-row">
        <div class="conf-bar">
          <div class="conf-fill ${type}" style="width:${fillW}%"></div>
        </div>
        <div class="conf-pct ${type}">${pctStr}</div>
      </div>
    </div>`;
  }).join('');

  results.innerHTML = header + cards;
}

function again() {
  canvas.style.display = 'none';
  results.innerHTML    = '';
  againBtn.classList.remove('show');
  if (_fileMode) {
    filePlaceholder.classList.add('show');
  } else {
    video.style.display = 'block';
  }
}

function setBusy(on) {
  snapBtn.disabled = on;
  snapBtn.classList.toggle('busy', on);
  document.getElementById('spinner').style.display = on ? 'block' : 'none';
  document.getElementById('snap-dot').style.display = on ? 'none' : '';
  document.getElementById('snap-label').textContent = on ? 'Identifying...' : 'Snap Photo';
}

function esc(s) {
  return String(s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── Status / warning system ────────────────────────────────

let _piOnline    = null;   // last known Pi state
let _dbPeople    = null;   // last known DB count
let _serverAlive = true;   // scan server reachable
let _failStreak  = 0;      // consecutive /status failures

async function checkStatus() {
  try {
    const ctrl = new AbortController();
    const tid  = setTimeout(() => ctrl.abort(), 5000);
    const resp = await fetch('/status', { signal: ctrl.signal });
    clearTimeout(tid);

    if (!resp.ok) throw new Error('server ' + resp.status);

    _failStreak  = 0;
    _serverAlive = true;
    document.getElementById('disconnect-overlay').classList.remove('show');

    const d = await resp.json();
    _applyStatus(d);
  } catch (e) {
    _failStreak++;
    if (_failStreak >= 2) {
      // Two consecutive failures → show disconnect overlay
      document.getElementById('disconnect-overlay').classList.add('show');
      snapBtn.disabled = true;
    } else {
      setStatusBar('offline', 'Scan server unreachable — retrying...');
    }
  }
}

function _applyStatus(d) {
  if (!d.pi_online) {
    const reason = d.pi_error || 'unreachable';
    setStatusBar('offline', `Pi offline (${reason}) — identification unavailable`);

    // Only toast once per offline event
    if (_piOnline !== false) {
      showWarn('Pi Offline',
        'The database server is unreachable. Face identification will fail until it reconnects.',
        'error', 10000);
    }
    _piOnline  = false;
    snapBtn.disabled = false;   // let them try; scan result will say Pi offline
  } else if (d.db_people === 0) {
    setStatusBar('warning', 'Pi online \u00b7 database is empty — no people registered yet');

    if (_dbPeople !== 0) {
      showWarn('Empty Database',
        'No people have been added to the database. Register faces using the dashboard first.',
        'warn', 10000);
    }
    _piOnline  = true;
    _dbPeople  = 0;
    snapBtn.disabled = false;
  } else {
    setStatusBar('online', `Pi online \u00b7 ${d.db_people} people in database`);

    // Recover from previous offline warning
    if (_piOnline === false) {
      showWarn('Pi Reconnected', 'Face identification is available again.', 'info', 4000);
    }
    if (_dbPeople === 0 && d.db_people > 0) {
      showWarn('Database Ready', `${d.db_people} people loaded.`, 'info', 4000);
    }
    _piOnline  = true;
    _dbPeople  = d.db_people;
    snapBtn.disabled = false;
  }
}

function setStatusBar(level, text) {
  document.getElementById('status-bar').className = level;
  document.getElementById('status-dot').className = level;
  document.getElementById('status-text').textContent = text;
}

function showWarn(title, msg, level = 'warn', duration = 7000) {
  const icons = { warn: '&#9888;', error: '&#128683;', info: '&#8505;' };
  const el    = document.createElement('div');
  el.className = `w-toast ${level}`;
  el.innerHTML = `
    <span class="w-icon">${icons[level] || icons.info}</span>
    <div class="w-body">
      <div class="w-title">${esc(title)}</div>
      <div class="w-msg">${esc(msg)}</div>
    </div>
    <span class="w-close" onclick="this.parentElement.remove()">&times;</span>`;
  document.getElementById('warn-container').appendChild(el);
  if (duration > 0) setTimeout(() => el.remove(), duration);
}

// Check on load and then every 25 seconds
checkStatus();
setInterval(checkStatus, 25000);

// Also re-check whenever the page becomes visible again (tab switch / phone unlock)
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) checkStatus();
});

startCamera();
</script>
</body>
</html>
"""


# ── HTTP request handler ───────────────────────────────────────────────────────

class _ScanHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"  [scan] {self.address_string()} — {fmt % args}")

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._html(_CAMERA_HTML)
        elif self.path == "/health":
            self._json({"ok": True})
        elif self.path == "/status":
            try:
                with httpx.Client(timeout=3.0) as client:
                    resp = client.get(f"{PI_URL}/status")
                    if resp.status_code == 200:
                        d = resp.json()
                        self._json({
                            "pi_online":    True,
                            "db_people":    d.get("total_people", 0),
                            "needs_review": d.get("needs_review", 0),
                            "ping_ms":      None,
                        })
                    else:
                        self._json({"pi_online": False, "pi_error": f"HTTP {resp.status_code}", "db_people": 0})
            except httpx.ConnectError:
                self._json({"pi_online": False, "pi_error": "unreachable", "db_people": 0})
            except httpx.TimeoutException:
                self._json({"pi_online": False, "pi_error": "timeout", "db_people": 0})
            except Exception as e:
                self._json({"pi_online": False, "pi_error": str(e), "db_people": 0})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path != "/scan":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._json({"error": "Empty body"}, 400)
            return
        if length > 15_000_000:     # 15 MB safety cap
            self._json({"error": "Image too large"}, 413)
            return

        data = self.rfile.read(length)

        # Decode image
        arr   = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            self._json({"error": "Could not decode image"}, 400)
            return

        # Detect faces (blocking — fine for per-request use)
        faces = detect(frame)
        if not faces:
            self._json({"faces_found": 0, "faces": []})
            return

        # Identify each detected face against the Pi
        matches = []
        try:
            with httpx.Client(timeout=5.0) as client:
                for face in faces:
                    try:
                        resp = client.post(
                            f"{PI_URL}/identify",
                            json={
                                "embedding":  face["embedding"],
                                "threshold":  PI_IDENTIFY_THRESHOLD,
                            },
                        )
                        if resp.status_code == 200:
                            m = resp.json()
                            if m.get("name") and m["name"] != "Unknown":
                                matches.append({
                                    "label":      m["name"],
                                    "confidence": m.get("confidence"),
                                    "type":       "known",
                                    "person_id":  m.get("person_id"),
                                })
                            else:
                                matches.append({
                                    "label":      "Unknown",
                                    "confidence": None,
                                    "type":       "unknown",
                                    "person_id":  None,
                                })
                        else:
                            matches.append({
                                "label": f"Pi error {resp.status_code}",
                                "confidence": None, "type": "error",
                            })
                    except httpx.TimeoutException:
                        matches.append({
                            "label": "Pi timeout", "confidence": None, "type": "error",
                        })
                    except httpx.ConnectError:
                        matches.append({
                            "label": "Pi offline", "confidence": None, "type": "error",
                        })
        except Exception as e:
            self._json({"error": str(e)}, 500)
            return

        self._json({"faces_found": len(matches), "faces": matches})


# ── Entry point ────────────────────────────────────────────────────────────────

def run_scan_server():
    """
    Called by main.py as the target of a dedicated Process.
    Loads InsightFace, starts the HTTPS server, serves forever.
    """
    local_ip = _local_ip()

    print("  [scan] loading InsightFace model...")
    init_worker_model()

    # Try to set up HTTPS with a self-signed cert
    use_https = False
    try:
        cert_path, key_path = _generate_cert(local_ip)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert_path, key_path)
        use_https = True
        print("  [scan] SSL cert ready")
    except ImportError:
        print("  [scan] 'cryptography' not installed — serving plain HTTP")
        print("  [scan] camera access will only work on localhost")
        ctx = None

    server = HTTPServer((SCAN_SERVER_HOST, SCAN_SERVER_PORT), _ScanHandler)

    if use_https:
        server.socket = ctx.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    else:
        scheme = "http"

    url = f"{scheme}://{local_ip}:{SCAN_SERVER_PORT}"
    print(f"  [scan] ready — {url}")
    print(f"  [scan] (accept the certificate warning on first visit)")

    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        server.server_close()
