/* Lingua — main.js */

const srcText     = document.getElementById('srcText');
const srcLang     = document.getElementById('srcLang');
const tgtLang     = document.getElementById('tgtLang');
const output      = document.getElementById('output');
const charCount   = document.getElementById('charCount');
const translateBtn= document.getElementById('translateBtn');
const clearBtn    = document.getElementById('clearBtn');
const swapBtn     = document.getElementById('swapBtn');
const copyBtn     = document.getElementById('copyBtn');
const errorBar    = document.getElementById('errorBar');

let lastTranslation = '';

// ── Char counter ──────────────────────────────────────────────────────────────
srcText.addEventListener('input', () => {
  const n = srcText.value.length;
  charCount.textContent = `${n} / 5000`;
  charCount.style.color = n > 4500 ? '#b85c38' : '';
});

// ── Swap ──────────────────────────────────────────────────────────────────────
swapBtn.addEventListener('click', () => {
  const tmp = srcLang.value;
  srcLang.value = tgtLang.value;
  tgtLang.value = tmp;

  if (lastTranslation) {
    srcText.value = lastTranslation;
    srcText.dispatchEvent(new Event('input'));
    clearOutput();
  }
});

// ── Clear ─────────────────────────────────────────────────────────────────────
clearBtn.addEventListener('click', () => {
  srcText.value = '';
  srcText.dispatchEvent(new Event('input'));
  clearOutput();
  hideError();
});

// ── Output helpers ────────────────────────────────────────────────────────────
function clearOutput() {
  lastTranslation = '';
  output.innerHTML = '<span class="placeholder">Translation will appear here…</span>';
}

function showLoading() {
  output.innerHTML = `
    <div class="shimmer">
      <div class="shimmer-line"></div>
      <div class="shimmer-line"></div>
      <div class="shimmer-line"></div>
    </div>`;
}

function setOutput(text) {
  lastTranslation = text;
  output.textContent = text;
}

function showError(msg) {
  errorBar.textContent = '⚠ ' + msg;
  errorBar.classList.add('visible');
}

function hideError() {
  errorBar.classList.remove('visible');
}

// ── Copy ──────────────────────────────────────────────────────────────────────
copyBtn.addEventListener('click', async () => {
  if (!lastTranslation) return;
  try {
    await navigator.clipboard.writeText(lastTranslation);
    copyBtn.classList.add('copied');
    copyBtn.innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="20 6 9 17 4 12"/>
      </svg> Copied`;
    setTimeout(() => {
      copyBtn.classList.remove('copied');
      copyBtn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
        </svg> Copy`;
    }, 2200);
  } catch {
    showError('Clipboard access denied.');
  }
});

// ── Translate ─────────────────────────────────────────────────────────────────
async function doTranslate() {
  const text = srcText.value.trim();
  if (!text) { showError('Please enter some text to translate.'); return; }

  hideError();
  showLoading();
  translateBtn.disabled = true;

  try {
    const res = await fetch('/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, src: srcLang.value, tgt: tgtLang.value }),
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || `Server error ${res.status}`);
      clearOutput();
    } else {
      setOutput(data.translation);
    }
  } catch (err) {
    showError('Network error — is the server running?');
    clearOutput();
  } finally {
    translateBtn.disabled = false;
  }
}

translateBtn.addEventListener('click', doTranslate);

// Ctrl/Cmd + Enter shortcut
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') doTranslate();
});
