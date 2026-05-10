/**
 * SignEase — Unified Sign Language Detection
 * Frontend Controller
 */

// ── State ──
let currentMode = 'ASL';
let isStreaming  = false;
let pollInterval = null;
let lastPrediction = '';

// ── DOM Refs ──
const videoFeed      = document.getElementById('video-feed');
const predictionChar = document.getElementById('prediction-value');
const sentenceText   = document.getElementById('sentence-text');
const charCount      = document.getElementById('char-count');
const modeBtnASL     = document.getElementById('mode-asl');
const modeBtnISL     = document.getElementById('mode-isl');
const modeBadge      = document.getElementById('mode-badge');
const streamStatus   = document.getElementById('stream-status');
const streamDot      = document.getElementById('stream-dot');
const gestureGuide   = document.getElementById('gesture-guide');
const islKeysInfo    = document.getElementById('isl-keys-info');
const streamPill     = document.getElementById('stream-status-pill');

// ── Toast ──
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const iconMap = { success: '✓', error: '✗', info: 'i' };
    const icon = document.createElement('div');
    icon.className = 'toast-icon';
    icon.textContent = iconMap[type] || 'i';

    const text = document.createElement('span');
    text.textContent = message;

    toast.appendChild(icon);
    toast.appendChild(text);
    container.appendChild(toast);

    setTimeout(() => toast.remove(), 3000);
}

// ── Mode Switching ──
function switchMode(mode) {
    if (mode === currentMode) return;

    // Show loading overlay on video
    const videoWrapper = document.querySelector('.video-wrapper');
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.id = 'loading-overlay';
    overlay.innerHTML = '<div class="spinner"></div>';
    videoWrapper.appendChild(overlay);

    fetch('/switch_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode })
    })
        .then(r => r.json())
        .then(data => {
            if (data.status === 'success') {
                currentMode = data.mode;
                updateModeUI();
                restartStream();
                showToast(`Switched to ${currentMode} mode`, 'success');
            }
        })
        .catch(() => showToast('Failed to switch mode', 'error'))
        .finally(() => {
            const ol = document.getElementById('loading-overlay');
            if (ol) ol.remove();
        });
}

function updateModeUI() {
    // Navbar pill buttons
    modeBtnASL.classList.toggle('active', currentMode === 'ASL');
    modeBtnISL.classList.toggle('active', currentMode === 'ISL');

    // Badge in video panel
    modeBadge.textContent = currentMode;
    modeBadge.className = `badge badge-mode${currentMode === 'ISL' ? ' isl-active' : ''}`;

    // Gesture guide + ISL keys
    if (currentMode === 'ASL') {
        gestureGuide.src = '/static/images/asl_signs.jpeg';
        gestureGuide.alt = 'ASL Gesture Reference';
        islKeysInfo.style.display = 'none';
    } else {
        gestureGuide.src = '/static/images/isl_gestures.png';
        gestureGuide.alt = 'ISL Gesture Reference';
        islKeysInfo.style.display = 'flex';
    }
}

// ── Stream ──
function startStream() {
    videoFeed.src = '/video_feed?' + Date.now();
    isStreaming = true;
    setStreamUI(true);
    startPolling();
}

function stopStream() {
    fetch('/stop', { method: 'POST' })
        .then(r => r.json())
        .then(() => {
            isStreaming = false;
            setStreamUI(false);
            stopPolling();
            showToast('Stream paused', 'info');
        });
}

function restartStream() {
    stopPolling();
    setTimeout(() => {
        videoFeed.src = '/video_feed?' + Date.now();
        isStreaming = true;
        setStreamUI(true);
        startPolling();
    }, 800);
}

function setStreamUI(live) {
    streamStatus.textContent = live ? 'Live' : 'Paused';
    streamDot.className = `status-dot ${live ? 'active' : 'inactive'}`;

    // Update navbar toggle icon
    const icon = document.getElementById('stream-icon');
    if (icon) {
        icon.innerHTML = live
            ? '<circle cx="12" cy="12" r="10"/><rect x="9" y="8" width="2" height="8"/><rect x="13" y="8" width="2" height="8"/>' // pause
            : '<circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/>';  // play
    }
}

function toggleStream() {
    isStreaming ? stopStream() : startStream();
}

// ── Polling ──
function startPolling() {
    stopPolling();
    pollInterval = setInterval(fetchData, 250);
}

function stopPolling() {
    if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
}

function fetchData() {
    fetch('/get_data')
        .then(r => r.json())
        .then(data => {
            // Prediction
            const pred = data.prediction || '';
            if (pred !== lastPrediction) {
                predictionChar.textContent = pred || '—';
                predictionChar.classList.remove('pop');
                void predictionChar.offsetWidth;
                predictionChar.classList.add('pop');
                lastPrediction = pred;
            }

            // Sentence
            const sentence = data.sentence || '';
            if (sentence) {
                sentenceText.innerHTML = sentence + '<span class="cursor"></span>';
            } else {
                sentenceText.innerHTML = '<span class="placeholder">Start signing to build a sentence...</span>';
            }

            // Char count
            if (charCount) {
                const len = sentence.length;
                charCount.textContent = len === 0 ? '0 chars' : `${len} char${len !== 1 ? 's' : ''}`;
            }
        })
        .catch(() => {});
}

// ── Controls ──
function clearSentence() {
    fetch('/clear_sentence', { method: 'POST' })
        .then(r => r.json())
        .then(() => {
            sentenceText.innerHTML = '<span class="placeholder">Start signing to build a sentence...</span>';
            predictionChar.textContent = '—';
            if (charCount) charCount.textContent = '0 chars';
            showToast('Sentence cleared', 'success');
        });
}

function speakText() {
    const btn = document.getElementById('speak-btn');
    if (btn && btn.disabled) return;

    if (btn) {
        btn.disabled = true;
        btn.classList.add('speaking');
        const iconEl = btn.querySelector('svg');
        if (iconEl) iconEl.style.opacity = '0.5';
    }

    fetch('/speak', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.status === 'success') {
                showToast('Speaking...', 'info');
            } else {
                showToast(data.message || 'Nothing to speak', 'error');
            }
        })
        .catch(() => showToast('TTS unavailable', 'error'))
        .finally(() => {
            if (btn) {
                // Keep disabled briefly so the speech subprocess can start
                setTimeout(() => {
                    btn.disabled = false;
                    btn.classList.remove('speaking');
                    const iconEl = btn.querySelector('svg');
                    if (iconEl) iconEl.style.opacity = '';
                }, 600);
            }
        });
}

function deleteLast() {
    fetch('/delete_last', { method: 'POST' })
        .then(r => r.json())
        .then(() => showToast('Deleted last character', 'info'));
}

function addSpace() {
    fetch('/add_space', { method: 'POST' })
        .then(r => r.json())
        .then(() => showToast('Space added', 'info'));
}

function correctWithAI() {
    const btn = document.getElementById('ai-btn');
    if (btn) { btn.disabled = true; }
    showToast('Correcting with NVIDIA NIM...', 'info');

    fetch('/correct', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.status === 'success') {
                sentenceText.innerHTML = data.corrected + '<span class="cursor"></span>';
                if (charCount) {
                    const l = data.corrected.length;
                    charCount.textContent = `${l} char${l !== 1 ? 's' : ''}`;
                }
                showToast(`"${data.original}" → "${data.corrected}"`, 'success');
            } else {
                showToast(data.message || 'Correction failed', 'error');
            }
        })
        .catch(() => showToast('AI correction unavailable', 'error'))
        .finally(() => { if (btn) btn.disabled = false; });
}

// ── Save Session ──
function saveSession() {
    const btn = document.getElementById('save-btn');
    if (btn) { btn.disabled = true; }
    showToast('Saving session...', 'info');

    fetch('/save_session', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.status === 'success') {
                showToast('✓ Session saved to history!', 'success');
                loadHistory(); // Refresh history panel
            } else {
                showToast(data.message || 'Save failed', 'error');
            }
        })
        .catch(() => showToast('Failed to save session', 'error'))
        .finally(() => { if (btn) btn.disabled = false; });
}

// ── Load History ──
function loadHistory() {
    const list = document.getElementById('history-list');
    if (!list) return;

    fetch('/user_history')
        .then(r => r.json())
        .then(data => {
            list.innerHTML = '';
            if (!data.items || data.items.length === 0) {
                list.innerHTML = '<div class="history-empty" id="history-empty">No saved sessions yet.<br>Sign something and click <strong>Save Session</strong>!</div>';
                return;
            }
            data.items.forEach(item => {
                const div = document.createElement('div');
                div.className = 'history-item';
                const modeBadgeClass = item.mode === 'ISL' ? 'history-badge-isl' : 'history-badge-asl';
                div.innerHTML = `
                    <div class="history-item-top">
                        <span class="history-badge ${modeBadgeClass}">${item.mode}</span>
                        <span class="history-time">${item.saved_at}</span>
                    </div>
                    <div class="history-sentence">${item.sentence}</div>
                `;
                list.appendChild(div);
            });
        })
        .catch(() => {
            const list = document.getElementById('history-list');
            if (list) list.innerHTML = '<div class="history-empty">Could not load history.</div>';
        });
}

// ── Model Status ──
function setStatusChip(id, dotId, ok) {
    const chip = document.getElementById(id);
    const dot  = document.getElementById(dotId);

    if (chip) {
        chip.textContent = ok ? 'Ready' : 'Error';
        chip.className = `status-chip ${ok ? 'ok' : 'err'}`;
    }
    if (dot) {
        dot.className = `status-dot-sm ${ok ? 'ok' : 'err'}`;
    }
}

function fetchModelStatus() {
    fetch('/model_status')
        .then(r => r.json())
        .then(data => {
            setStatusChip('asl-model-status', 'asl-dot', data.asl_model);
            setStatusChip('isl-model-status', 'isl-dot', data.isl_model);
            setStatusChip('tts-status',        'tts-dot', data.tts_engine);
            setStatusChip('nvidia-status',     'nvidia-dot', data.nvidia_nim);
        });
}

// ── Gesture guide ──
function openGestureGuide() {
    window.open(gestureGuide.src, '_blank');
}

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
    updateModeUI();
    startStream();
    fetchModelStatus();
    loadHistory();
});
