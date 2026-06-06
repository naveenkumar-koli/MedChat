/**
 * MedChat — Frontend Controller
 * Three-panel RAG chatbot UI:
 *  - Left:   Knowledge base docs + PDF upload + evaluation metrics
 *  - Center: Chat with citations, follow-ups, pipeline animation
 *  - Right:  Retrieved chunks visualization (ranked by cross-encoder score)
 */

// ── Toast Notifications ───────────────────────────────────────────────────
const toastContainer = (() => {
    const el = document.createElement("div");
    el.id = "toast-container";
    document.body.appendChild(el);
    return el;
})();

function showToast(msg, type = "info", duration = 3500) {
    const t = document.createElement("div");
    t.className = `toast ${type}`;
    t.innerHTML = `<i class="fas fa-${type === "success" ? "check-circle" : type === "error" ? "circle-xmark" : "circle-info"}"></i> ${msg}`;
    toastContainer.appendChild(t);
    setTimeout(() => t.remove(), duration);
}

// ── Dark / Light Theme Toggle ──────────────────────────────────────────────
(function initTheme() {
    const root    = document.documentElement;
    const savedTheme = localStorage.getItem("medchat-theme") || "dark";
    root.setAttribute("data-theme", savedTheme);

    function applyThemeIcon(theme) {
        const icon = document.getElementById("theme-icon");
        if (!icon) return;
        icon.className = theme === "light" ? "fas fa-moon" : "fas fa-sun";
    }

    applyThemeIcon(savedTheme);

    document.addEventListener("DOMContentLoaded", () => {
        applyThemeIcon(root.getAttribute("data-theme"));
        const btn = document.getElementById("theme-toggle-btn");
        if (!btn) return;
        btn.addEventListener("click", () => {
            const current = root.getAttribute("data-theme") || "dark";
            const next    = current === "dark" ? "light" : "dark";
            root.setAttribute("data-theme", next);
            localStorage.setItem("medchat-theme", next);
            applyThemeIcon(next);
            showToast(next === "light" ? "☀️ Light mode on" : "🌙 Dark mode on", "info", 1500);
        });
    });
})();



// ── Simple Markdown renderer ──────────────────────────────────────────────
function renderMarkdown(text) {
    return text
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<em>$1</em>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/^## (.+)$/gm, "<h2>$1</h2>")
        .replace(/^### (.+)$/gm, "<h3>$1</h3>")
        .replace(/^#{1} (.+)$/gm, "<h2>$1</h2>")
        .replace(/^\d+\.\s+(.+)$/gm, "<li>$1</li>")
        .replace(/^[-•]\s+(.+)$/gm, "<li>$1</li>")
        .replace(/\n{2,}/g, "</p><p>")
        .replace(/^(?!<[hlu])(.+)$/gm, "$1")
        .replace(/(<li>[\s\S]+?<\/li>)+/g, m => `<ul>${m}</ul>`)
        // Citation links like [filename.pdf, p.42]
        .replace(/\[([^\]]+\.pdf[^\]]*)\]/g,
            '<span class="citation-inline" title="$1">📄 $1</span>');
}

// ── Pipeline Stage Animator ───────────────────────────────────────────────
class PipelineAnimator {
    constructor() {
        this.badges = {
            retrieve: document.getElementById("pb-retrieve"),
            rerank:   document.getElementById("pb-rerank"),
            generate: document.getElementById("pb-generate"),
        };
        this.stages = {
            retrieve: document.getElementById("stage-retrieve"),
            rerank:   document.getElementById("stage-rerank"),
            generate: document.getElementById("stage-generate"),
        };
    }

    reset() {
        Object.values(this.badges).forEach(b => b.classList.remove("active", "done"));
        Object.values(this.stages).forEach(s => s.classList.remove("active", "done"));
    }

    activate(stage) {
        this.reset();
        const seq = ["retrieve", "rerank", "generate"];
        const idx = seq.indexOf(stage);
        seq.forEach((s, i) => {
            if (i < idx) {
                this.badges[s]?.classList.add("done");
                this.stages[s]?.classList.add("done");
            } else if (i === idx) {
                this.badges[s]?.classList.add("active");
                this.stages[s]?.classList.add("active");
            }
        });
    }

    complete() {
        Object.values(this.badges).forEach(b => { b.classList.remove("active"); b.classList.add("done"); });
        Object.values(this.stages).forEach(s => { s.classList.remove("active"); s.classList.add("done"); });
    }
}

// ── Retrieval Panel ───────────────────────────────────────────────────────
class RetrievalPanel {
    constructor() {
        this.panel = document.getElementById("retrieval-panel");
        this.body  = this.panel.querySelector(".retrieval-body");
        document.getElementById("close-retrieval-btn").addEventListener("click", () => this.close());
    }

    open(chunks) {
        this.body.innerHTML = "";
        if (!chunks || chunks.length === 0) {
            this.body.innerHTML = `<div class="empty-state"><i class="fas fa-magnifying-glass-chart"></i><p>No chunks retrieved</p></div>`;
            return;
        }
        chunks.forEach(chunk => {
            const card = document.createElement("div");
            card.className = "chunk-card";

            // Normalize CE score to 0–1 range for bar (CE can go -10 to +10)
            const ceNorm = Math.max(0, Math.min(1, (chunk.rerank_score + 5) / 10));
            const annPct = Math.round(Math.max(0, chunk.ann_score) * 100);

            card.innerHTML = `
                <div class="chunk-rank">${chunk.rank}</div>
                <div class="chunk-source">
                    <i class="fas fa-file-pdf" style="color:var(--accent-red);font-size:10px"></i>
                    <span class="chunk-file">${chunk.filename}</span>
                    <span class="chunk-page">p.${chunk.page}</span>
                </div>
                <div class="chunk-text">${chunk.text}</div>
                <div class="chunk-scores">
                    <span class="score-pill score-ann" title="ANN cosine score">ANN: ${chunk.ann_score?.toFixed(3) ?? '—'}</span>
                    <span class="score-pill score-ce"  title="Cross-encoder score">CE: ${chunk.rerank_score?.toFixed(3) ?? '—'}</span>
                </div>
                <div class="score-bar-wrap">
                    <div class="score-bar-track">
                        <div class="score-bar-fill" style="width:${Math.round(ceNorm*100)}%"></div>
                    </div>
                </div>`;
            this.body.appendChild(card);
        });
        this.panel.classList.remove("collapsed");
    }

    close() { this.panel.classList.add("collapsed"); }
}

// ── Document Sidebar ──────────────────────────────────────────────────────
class DocumentSidebar {
    constructor() {
        this.docList   = document.getElementById("doc-list");
        this.docsCount = document.getElementById("docs-count");
        document.getElementById("refresh-docs-btn").addEventListener("click", () => this.load());
        document.getElementById("docs-panel-btn").addEventListener("click", () => this.load());
    }

    async load() {
        try {
            const resp = await fetch("/api/documents");
            const data = await resp.json();
            this.render(data.documents || []);
        } catch (e) {
            console.error("Failed to load documents", e);
        }
    }

    render(docs) {
        this.docsCount.textContent = docs.length;
        if (docs.length === 0) {
            this.docList.innerHTML = `<div class="empty-state"><i class="fas fa-file-pdf"></i><p>No documents indexed yet.<br>Upload PDFs to get started.</p></div>`;
            return;
        }
        this.docList.innerHTML = docs.map(d => `
            <div class="doc-item">
                <div class="doc-name" title="${d.filename}">
                    <i class="fas fa-file-pdf" style="color:var(--accent-red);margin-right:5px"></i>${d.filename}
                </div>
                <div class="doc-meta">
                    <span class="doc-chip">${d.total_pages} pages</span>
                    <span class="doc-chip">${d.total_chunks} chunks</span>
                </div>
            </div>`).join("");
    }
}

// ── PDF Uploader ──────────────────────────────────────────────────────────
class PdfUploader {
    constructor(sidebar) {
        this.sidebar  = sidebar;
        this.drop     = document.getElementById("upload-drop");
        this.progress = document.getElementById("upload-progress");
        this.bar      = document.getElementById("progress-bar");
        this.label    = document.getElementById("progress-label");
        this.detail   = document.getElementById("progress-detail");
        this.fileInput= document.getElementById("pdf-file-input");

        this.fileInput.addEventListener("change", e => {
            if (e.target.files[0]) this.upload(e.target.files[0]);
        });

        this.drop.addEventListener("dragover", e => { e.preventDefault(); this.drop.classList.add("drag-over"); });
        this.drop.addEventListener("dragleave", () => this.drop.classList.remove("drag-over"));
        this.drop.addEventListener("drop", e => {
            e.preventDefault();
            this.drop.classList.remove("drag-over");
            const file = e.dataTransfer.files[0];
            if (file?.type === "application/pdf") this.upload(file);
            else showToast("Only PDF files are supported", "error");
        });
    }

    async upload(file) {
        this.drop.style.display = "none";
        this.progress.style.display = "block";
        this.label.textContent = `Uploading ${file.name}…`;
        this.detail.textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB`;

        // Animate progress bar (fake progress during server processing)
        let pct = 0;
        const interval = setInterval(() => {
            pct = Math.min(pct + Math.random() * 8, 90);
            this.bar.style.width = `${pct}%`;
        }, 400);

        const formData = new FormData();
        formData.append("pdf", file);

        try {
            const resp = await fetch("/api/ingest", { method: "POST", body: formData });
            clearInterval(interval);
            const data = await resp.json();

            if (data.error) throw new Error(data.error);

            this.bar.style.width = "100%";
            this.label.textContent = "✅ Ingestion complete";
            this.detail.textContent = `${data.total_vectors?.toLocaleString()} total vectors indexed in ${data.elapsed_s}s`;
            showToast(`${file.name} indexed successfully!`, "success");
            setTimeout(() => {
                this.drop.style.display = "block";
                this.progress.style.display = "none";
                this.bar.style.width = "0%";
            }, 3000);
            this.sidebar.load();
        } catch (err) {
            clearInterval(interval);
            this.label.textContent = "❌ Ingestion failed";
            this.detail.textContent = err.message;
            showToast(`Upload failed: ${err.message}`, "error");
            setTimeout(() => {
                this.drop.style.display = "block";
                this.progress.style.display = "none";
                this.bar.style.width = "0%";
            }, 4000);
        }
    }
}

// ── Metrics Display ───────────────────────────────────────────────────────
class MetricsDisplay {
    constructor() {
        this.pill     = document.getElementById("metrics-pill");
        this.p50El    = document.getElementById("m-p50");
        this.p95El    = document.getElementById("m-p95");
        this.queryEl  = document.getElementById("m-queries");
        this.barWrap  = document.getElementById("latency-bar-wrap");
        this.barFill  = document.getElementById("latency-bar-fill");
        this.barVal   = document.getElementById("latency-bar-val");
        this.metricLatency = document.getElementById("metric-latency");
        this.poll();
        setInterval(() => this.poll(), 15000);
    }

    async poll() {
        try {
            const resp = await fetch("/api/metrics");
            const data = await resp.json();
            this.update(data);
        } catch(e) { /* ignore */ }
    }

    update(data) {
        const lat = data.latency || {};
        const p50 = lat.p50_s;
        const p95 = lat.p95_s;
        this.queryEl.textContent = data.query_count || 0;
        if (p50 !== undefined) {
            this.p50El.textContent = `${p50}s`;
            this.p95El.textContent = `${p95}s`;
            const cls = p95 <= 3 ? "good" : p95 <= 5 ? "warn" : "bad";
            this.pill.className = `metrics-pill ${cls}`;
            this.metricLatency.textContent = `p95: ${p95}s`;
        }
    }

    showLastQuery(timing) {
        const total = timing?.total_s;
        if (!total) return;
        this.barWrap.style.display = "block";
        const pct = Math.min(100, (total / 5) * 100);
        this.barFill.style.width = `${pct}%`;
        this.barFill.className = `latency-bar-fill ${total > 5 ? "warn" : ""}`;
        this.barVal.textContent = `${total}s`;
        this.metricLatency.textContent = `Last: ${total}s`;
        const cls = total <= 3 ? "good" : total <= 5 ? "warn" : "bad";
        this.pill.className = `metrics-pill ${cls}`;
    }
}

// ── Chat Controller ───────────────────────────────────────────────────────
class ChatController {
    constructor(pipeline, retrieval, metrics) {
        this.pipeline  = pipeline;
        this.retrieval = retrieval;
        this.metrics   = metrics;

        this.msgContainer = document.getElementById("chat-messages");
        this.typingBar    = document.getElementById("typing-bar");
        this.input        = document.getElementById("user-input");
        this.sendBtn      = document.getElementById("send-btn");
        this.welcome      = document.getElementById("welcome-screen");

        this.recognition  = null;
        this.isListening  = false;

        this.bindEvents();
        this.initVoice();
    }

    bindEvents() {
        this.sendBtn.addEventListener("click", () => this.send());
        this.input.addEventListener("keydown", e => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); this.send(); }
        });
        this.input.addEventListener("input", () => {
            this.input.style.height = "auto";
            this.input.style.height = Math.min(this.input.scrollHeight, 120) + "px";
        });
        document.getElementById("voice-btn").addEventListener("click", () => this.toggleVoice());
    }

    initVoice() {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) return;
        this.recognition = new SR();
        this.recognition.lang = "en-US";
        this.recognition.interimResults = false;
        this.recognition.onresult = e => {
            this.input.value = e.results[0][0].transcript;
            this.stopVoice();
        };
        this.recognition.onerror = () => this.stopVoice();
        this.recognition.onend   = () => this.stopVoice();
    }

    toggleVoice() {
        if (this.isListening) this.stopVoice();
        else this.startVoice();
    }

    startVoice() {
        if (!this.recognition) { showToast("Voice input not supported in this browser", "error"); return; }
        this.isListening = true;
        document.getElementById("voice-btn").classList.add("recording");
        this.recognition.start();
    }

    stopVoice() {
        this.isListening = false;
        document.getElementById("voice-btn").classList.remove("recording");
        try { this.recognition?.stop(); } catch(e) {}
    }

    async send() {
        const question = this.input.value.trim();
        if (!question) return;

        this.input.value = "";
        this.input.style.height = "auto";
        this.sendBtn.disabled = true;

        if (this.welcome) { this.welcome.remove(); this.welcome = null; }

        this.appendUserMsg(question);
        this.showTyping();
        this.pipeline.reset();

        try {
            // Simulate pipeline stages visually
            this.pipeline.activate("retrieve");
            await this.sleep(400);
            this.pipeline.activate("rerank");

            const resp = await fetch("/api/ask", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
            });
            const data = await resp.json();

            if (data.error) throw new Error(data.error);

            this.pipeline.activate("generate");
            await this.sleep(200);
            this.pipeline.complete();

            this.hideTyping();
            this.appendBotMsg(data);
            this.retrieval.open(data.retrieved_chunks);
            this.metrics.showLastQuery(data.timing);
            this.metrics.poll();
        } catch (err) {
            this.pipeline.reset();
            this.hideTyping();
            this.appendErrorMsg(err.message);
            showToast(`Error: ${err.message}`, "error");
        } finally {
            this.sendBtn.disabled = false;
            this.input.focus();
        }
    }

    sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

    appendUserMsg(text) {
        const row = document.createElement("div");
        row.className = "msg-row user";
        row.innerHTML = `
            <div class="avatar user"><i class="fas fa-user"></i></div>
            <div class="bubble user">${text}</div>`;
        this.msgContainer.appendChild(row);
        this.scrollBottom();
    }

    appendBotMsg(data) {
        const row = document.createElement("div");
        row.className = "msg-row bot";

        // Build citations strip
        const citeHTML = (data.citations || []).map(c => `
            <span class="citation-badge" title="${c.preview || ''}">
                <i class="fas fa-file-pdf"></i>
                ${c.filename} p.${c.page}
                <span style="opacity:0.6;font-size:9px">CE:${c.rerank_score?.toFixed(2) ?? '?'}</span>
            </span>`).join("");

        // Extract follow-up questions from answer
        const followups = this.extractFollowups(data.answer);

        const timingStr = data.timing
            ? `Retrieve ${data.timing.retrieve_s}s · Rerank ${data.timing.rerank_s}s · Generate ${data.timing.generate_s}s · Total ${data.timing.total_s}s`
            : "";

        row.innerHTML = `
            <div class="avatar bot"><i class="fas fa-brain"></i></div>
            <div class="bubble bot">
                <div class="bubble-content">${renderMarkdown(data.answer)}</div>
                ${citeHTML ? `<div class="citation-strip">${citeHTML}</div>` : ""}
                ${timingStr ? `<div class="timing-badge"><i class="fas fa-stopwatch"></i> ${timingStr}</div>` : ""}
                ${followups.length ? `<div class="followups">${followups.map(q =>
                    `<button class="followup-btn" onclick="fillQuestion(${JSON.stringify(q)})">${q}</button>`
                ).join("")}</div>` : ""}
            </div>`;
        this.msgContainer.appendChild(row);
        this.scrollBottom();
    }

    appendErrorMsg(msg) {
        const row = document.createElement("div");
        row.className = "msg-row bot";
        row.innerHTML = `
            <div class="avatar bot" style="background:var(--accent-red)"><i class="fas fa-triangle-exclamation"></i></div>
            <div class="bubble bot" style="border-color:rgba(239,68,68,0.3)">
                ❌ <strong>Error:</strong> ${msg}<br><small style="color:var(--text-muted)">Ensure the Flask server is running and the vector index is built.</small>
            </div>`;
        this.msgContainer.appendChild(row);
        this.scrollBottom();
    }

    extractFollowups(text) {
        const matches = [];
        const lines = text.split("\n");
        let inFollowup = false;
        for (const line of lines) {
            if (/suggested follow.up/i.test(line)) { inFollowup = true; continue; }
            if (inFollowup) {
                const m = line.match(/^\d+\.\s+(.+\?)/);
                if (m) matches.push(m[1].trim());
                if (matches.length >= 3) break;
            }
        }
        return matches;
    }

    showTyping() {
        this.typingBar.style.display = "flex";
        ["stage-retrieve","stage-rerank","stage-generate"].forEach(id => {
            const el = document.getElementById(id);
            el.classList.remove("done");
            el.classList.add("active");
        });
    }

    hideTyping() { this.typingBar.style.display = "none"; }

    scrollBottom() {
        this.msgContainer.scrollTop = this.msgContainer.scrollHeight;
    }
}

// ── Global helper for starter questions ──────────────────────────────────
function fillQuestion(q) {
    const input = document.getElementById("user-input");
    input.value = q;
    input.dispatchEvent(new Event("input"));
    input.focus();
    // Auto-send
    setTimeout(() => window.chatApp?.send(), 100);
}

// ── Boot ──────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    const pipeline  = new PipelineAnimator();
    const retrieval = new RetrievalPanel();
    const metrics   = new MetricsDisplay();
    const sidebar   = new DocumentSidebar();
    const uploader  = new PdfUploader(sidebar);
    const chat      = new ChatController(pipeline, retrieval, metrics);

    window.chatApp = chat;

    // Load docs on startup
    sidebar.load();

    // Toggle upload zone (already visible in sidebar, just scroll to it)
    document.getElementById("upload-toggle-btn").addEventListener("click", () => {
        const up = document.getElementById("upload-zone");
        up.scrollIntoView({ behavior: "smooth" });
    });
});