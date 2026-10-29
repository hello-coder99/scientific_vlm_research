/**
 * Scientific Chart Verifier - Interactive Frontend Application
 * Pure Vanilla JavaScript (Zero External Frontend Dependencies)
 */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------------------------------
    // State Management
    // -------------------------------------------------------------------------
    const state = {
        scenarios: [],
        selectedScenarioId: "chart_001",
        customImageFile: null,
        tolerance: 0.03,
        lastVerificationResult: null,
        currentImageView: "overlay", // "overlay" or "original"
    };

    // -------------------------------------------------------------------------
    // DOM Elements
    // -------------------------------------------------------------------------
    const dom = {
        // Tabs
        navTabs: document.querySelectorAll(".nav-tab"),
        tabPanes: document.querySelectorAll(".tab-pane"),

        // Scenarios
        scenarioChipsContainer: document.getElementById("scenarioChipsContainer"),
        chartSourceBadge: document.getElementById("chartSourceBadge"),

        // Upload / Dropzone
        dropzoneContainer: document.getElementById("dropzoneContainer"),
        fileInput: document.getElementById("chartImageFileInput"),
        dropzoneEmpty: document.getElementById("dropzoneEmpty"),
        dropzonePreview: document.getElementById("dropzonePreview"),
        previewImageTag: document.getElementById("previewImageTag"),
        previewFilename: document.getElementById("previewFilename"),
        clearImageBtn: document.getElementById("clearImageBtn"),

        // Tolerance Slider
        toleranceSlider: document.getElementById("toleranceSlider"),
        toleranceValBadge: document.getElementById("toleranceValBadge"),

        // JSON Editor
        predictionJsonEditor: document.getElementById("predictionJsonEditor"),
        formatJsonBtn: document.getElementById("formatJsonBtn"),
        resetJsonBtn: document.getElementById("resetJsonBtn"),
        jsonStatusText: document.getElementById("jsonStatusText"),
        jsonStatusIndicator: document.querySelector(".status-indicator"),

        // Run Verification
        runVerificationBtn: document.getElementById("runVerificationBtn"),
        runSpinner: document.getElementById("runSpinner"),
        quickTestRunBtn: document.getElementById("quickTestRunBtn"),

        // Results HUD
        emptyResultsState: document.getElementById("emptyResultsState"),
        resultsWrapper: document.getElementById("resultsWrapper"),
        hudGroundingScore: document.getElementById("hudGroundingScore"),
        countSupported: document.getElementById("countSupported"),
        countContradicted: document.getElementById("countContradicted"),
        countFabricated: document.getElementById("countFabricated"),
        countOmitted: document.getElementById("countOmitted"),

        // Viewer Stage
        displayMainImg: document.getElementById("displayMainImg"),
        viewOverlayBtn: document.getElementById("viewOverlayBtn"),
        viewOriginalBtn: document.getElementById("viewOriginalBtn"),
        viewHtmlCertLink: document.getElementById("viewHtmlCertLink"),
        downloadOverlayBtn: document.getElementById("downloadOverlayBtn"),
        exportJsonReportBtn: document.getElementById("exportJsonReportBtn"),

        // Proof Cards & Tables
        proofCardsGrid: document.getElementById("proofCardsGrid"),
        totalPointsBadge: document.getElementById("totalPointsBadge"),
        textAuditTbody: document.getElementById("textAuditTbody"),
        textGroundingRateBadge: document.getElementById("textGroundingRateBadge"),
        calibMetricsGrid: document.getElementById("calibMetricsGrid"),

        // Live LLM Extractor
        liveModelSelect: document.getElementById("liveModelSelect"),
        extractVlmBtn: document.getElementById("extractVlmBtn"),
        extractSpinner: document.getElementById("extractSpinner"),
        llmExtractionStatus: document.getElementById("llmExtractionStatus"),
        extractionModeBadge: document.getElementById("extractionModeBadge"),
        extractionNoteText: document.getElementById("extractionNoteText"),

        // Benchmark Fleet Tab
        runFleetBenchmarkBtn: document.getElementById("runFleetBenchmarkBtn"),
        fleetSpinner: document.getElementById("fleetSpinner"),
        benchmarkTbody: document.getElementById("benchmarkTbody"),
    };

    // -------------------------------------------------------------------------
    // Initialization
    // -------------------------------------------------------------------------
    async function init() {
        setupTabs();
        setupDropzone();
        setupSlider();
        setupJsonEditor();
        setupActionButtons();
        await loadScenarios();
    }

    // -------------------------------------------------------------------------
    // Tab Navigation
    // -------------------------------------------------------------------------
    function setupTabs() {
        dom.navTabs.forEach(tabBtn => {
            tabBtn.addEventListener("click", () => {
                const targetTabId = tabBtn.getAttribute("data-tab");

                dom.navTabs.forEach(b => {
                    b.classList.remove("active");
                    b.setAttribute("aria-selected", "false");
                });
                dom.tabPanes.forEach(pane => pane.classList.remove("active"));

                tabBtn.classList.add("active");
                tabBtn.setAttribute("aria-selected", "true");
                const targetPane = document.getElementById(targetTabId);
                if (targetPane) targetPane.classList.add("active");

                // Auto-load benchmark table if navigating to benchmark tab
                if (targetTabId === "tab-benchmark" && dom.benchmarkTbody.children.length === 0) {
                    runFleetBenchmark();
                }
            });
        });
    }

    // -------------------------------------------------------------------------
    // Benchmark Scenarios Loading
    // -------------------------------------------------------------------------
    async function loadScenarios() {
        try {
            const resp = await fetch("/api/scenarios");
            if (!resp.ok) throw new Error("Failed to load benchmark scenarios");
            state.scenarios = await resp.json();
            renderScenarioChips();

            // Select initial scenario
            if (state.scenarios.length > 0) {
                selectScenario(state.scenarios[0].id);
            }
        } catch (err) {
            console.error("Scenario load error:", err);
            dom.scenarioChipsContainer.innerHTML = `<div style="color: var(--color-contradicted); padding: 0.5rem;">Failed to load benchmark scenarios. Ensure FastAPI server is running.</div>`;
        }
    }

    function renderScenarioChips() {
        dom.scenarioChipsContainer.innerHTML = "";
        state.scenarios.forEach(sc => {
            const chip = document.createElement("div");
            chip.className = `scenario-chip ${sc.id === state.selectedScenarioId ? "selected" : ""}`;
            chip.id = `scenarioChip_${sc.id}`;

            let tagClass = "red";
            if (sc.id === "chart_002") tagClass = "magenta";
            if (sc.id === "chart_005") tagClass = "cyan";

            chip.innerHTML = `
                <div class="chip-top">
                    <span class="chip-id">${sc.id}</span>
                    <span class="chip-tag ${tagClass}">${sc.category}</span>
                </div>
                <div class="chip-title">${sc.title}</div>
                <div class="chip-desc">${sc.description}</div>
            `;

            chip.addEventListener("click", () => selectScenario(sc.id));
            dom.scenarioChipsContainer.appendChild(chip);
        });
    }

    function selectScenario(scenarioId) {
        state.selectedScenarioId = scenarioId;
        state.customImageFile = null;

        // Update active chip styling
        document.querySelectorAll(".scenario-chip").forEach(c => c.classList.remove("selected"));
        const activeChip = document.getElementById(`scenarioChip_${scenarioId}`);
        if (activeChip) activeChip.classList.add("selected");

        const sc = state.scenarios.find(s => s.id === scenarioId);
        if (!sc) return;

        // Update badge
        dom.chartSourceBadge.textContent = `Benchmark: ${sc.id}`;
        dom.chartSourceBadge.className = "badge badge-accent";

        // Update dropzone preview with scenario image
        dom.previewImageTag.src = sc.image_url;
        dom.previewFilename.textContent = `${sc.id}.png (${sc.title})`;
        dom.dropzoneEmpty.classList.add("hidden");
        dom.dropzonePreview.classList.remove("hidden");

        // Populate JSON editor
        dom.predictionJsonEditor.value = JSON.stringify(sc.prediction, null, 4);
        validateJson();
    }

    // -------------------------------------------------------------------------
    // Dropzone & File Upload
    // -------------------------------------------------------------------------
    function setupDropzone() {
        const dz = dom.dropzoneContainer;
        const fi = dom.fileInput;

        dz.addEventListener("click", (e) => {
            if (e.target !== dom.clearImageBtn) {
                fi.click();
            }
        });

        fi.addEventListener("change", (e) => {
            if (fi.files && fi.files[0]) {
                handleUserImageFile(fi.files[0]);
            }
        });

        ["dragenter", "dragover"].forEach(evtName => {
            dz.addEventListener(evtName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dz.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach(evtName => {
            dz.addEventListener(evtName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dz.classList.remove("dragover");
            });
        });

        dz.addEventListener("drop", (e) => {
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleUserImageFile(e.dataTransfer.files[0]);
            }
        });

        dom.clearImageBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            resetToCurrentScenario();
        });
    }

    function handleUserImageFile(file) {
        state.customImageFile = file;
        state.selectedScenarioId = null;

        document.querySelectorAll(".scenario-chip").forEach(c => c.classList.remove("selected"));
        dom.chartSourceBadge.textContent = "Custom Upload";
        dom.chartSourceBadge.className = "badge badge-green";

        const reader = new FileReader();
        reader.onload = (e) => {
            dom.previewImageTag.src = e.target.result;
            dom.previewFilename.textContent = file.name;
            dom.dropzoneEmpty.classList.add("hidden");
            dom.dropzonePreview.classList.remove("hidden");
        };
        reader.readAsDataURL(file);
    }

    function resetToCurrentScenario() {
        state.customImageFile = null;
        dom.fileInput.value = "";
        if (state.scenarios.length > 0) {
            selectScenario(state.scenarios[0].id);
        } else {
            dom.dropzoneEmpty.classList.remove("hidden");
            dom.dropzonePreview.classList.add("hidden");
            dom.chartSourceBadge.textContent = "No Chart Selected";
        }
    }

    // -------------------------------------------------------------------------
    // Slider Controls
    // -------------------------------------------------------------------------
    function setupSlider() {
        dom.toleranceSlider.addEventListener("input", (e) => {
            const val = parseFloat(e.target.value);
            state.tolerance = val / 100.0;
            dom.toleranceValBadge.textContent = `${val.toFixed(1)}% Axis Span`;
        });
    }

    // -------------------------------------------------------------------------
    // JSON Editor Controls
    // -------------------------------------------------------------------------
    function setupJsonEditor() {
        dom.predictionJsonEditor.addEventListener("input", validateJson);

        dom.formatJsonBtn.addEventListener("click", () => {
            try {
                const parsed = JSON.parse(dom.predictionJsonEditor.value);
                dom.predictionJsonEditor.value = JSON.stringify(parsed, null, 4);
                validateJson();
            } catch (e) {
                alert("Cannot format: Invalid JSON syntax.");
            }
        });

        dom.resetJsonBtn.addEventListener("click", () => {
            if (state.selectedScenarioId) {
                const sc = state.scenarios.find(s => s.id === state.selectedScenarioId);
                if (sc) {
                    dom.predictionJsonEditor.value = JSON.stringify(sc.prediction, null, 4);
                    validateJson();
                }
            } else {
                dom.predictionJsonEditor.value = JSON.stringify({
                    chart_id: "custom_chart",
                    title: "Chart Title",
                    x_axis: "X-Axis",
                    y_axis: "Y-Axis",
                    points: [
                        { x: 1.0, y: 10.0 },
                        { x: 2.0, y: 20.0 }
                    ]
                }, null, 4);
                validateJson();
            }
        });
    }

    function validateJson() {
        try {
            const content = dom.predictionJsonEditor.value.trim();
            if (!content) {
                dom.jsonStatusText.textContent = "Empty JSON payload";
                dom.jsonStatusIndicator.className = "status-indicator invalid";
                return false;
            }
            JSON.parse(content);
            dom.jsonStatusText.textContent = "Valid JSON schema";
            dom.jsonStatusIndicator.className = "status-indicator valid";
            return true;
        } catch (e) {
            dom.jsonStatusText.textContent = `Syntax Error: ${e.message.slice(0, 32)}...`;
            dom.jsonStatusIndicator.className = "status-indicator invalid";
            return false;
        }
    }

    // -------------------------------------------------------------------------
    // Action Buttons & Verification Execution
    // -------------------------------------------------------------------------
    function setupActionButtons() {
        dom.runVerificationBtn.addEventListener("click", runVerification);
        if (dom.extractVlmBtn) {
            dom.extractVlmBtn.addEventListener("click", extractViaLlm);
        }
        if (dom.quickTestRunBtn) {
            dom.quickTestRunBtn.addEventListener("click", () => {
                selectScenario("chart_001");
                runVerification();
            });
        }

        // Overlay vs Original Image Toggles
        dom.viewOverlayBtn.addEventListener("click", () => switchImageView("overlay"));
        dom.viewOriginalBtn.addEventListener("click", () => switchImageView("original"));

        // Download Overlay
        dom.downloadOverlayBtn.addEventListener("click", () => {
            if (!state.lastVerificationResult || !state.lastVerificationResult.overlay_image_base64) return;
            const a = document.createElement("a");
            a.href = state.lastVerificationResult.overlay_image_base64;
            a.download = `${state.lastVerificationResult.chart_id}_verified.png`;
            a.click();
        });

        // Export JSON Report
        dom.exportJsonReportBtn.addEventListener("click", () => {
            if (!state.lastVerificationResult) return;
            const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(state.lastVerificationResult, null, 2));
            const a = document.createElement("a");
            a.href = dataStr;
            a.download = `${state.lastVerificationResult.chart_id}_verification_verdict.json`;
            a.click();
        });

        // Fleet Benchmark Live Button
        dom.runFleetBenchmarkBtn.addEventListener("click", runFleetBenchmark);
    }

    // -------------------------------------------------------------------------
    // Live LLM Extractor Execution
    // -------------------------------------------------------------------------
    async function extractViaLlm() {
        const modelChoice = dom.liveModelSelect ? dom.liveModelSelect.value : "gemini";
        if (dom.extractSpinner) dom.extractSpinner.classList.remove("hidden");
        if (dom.extractVlmBtn) dom.extractVlmBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append("model_choice", modelChoice);
            if (state.customImageFile) {
                formData.append("image_file", state.customImageFile);
            } else {
                formData.append("chart_id", state.selectedScenarioId || "chart_001");
            }

            const resp = await fetch("/api/extract_vlm", {
                method: "POST",
                body: formData
            });

            if (!resp.ok) {
                const errJson = await resp.json().catch(() => ({}));
                throw new Error(errJson.detail || `HTTP ${resp.status}`);
            }

            const data = await resp.json();

            // Populate prediction JSON editor with extracted structured data
            if (data.prediction) {
                const pred = { ...data.prediction };
                delete pred._meta;
                dom.predictionJsonEditor.value = JSON.stringify(pred, null, 2);
                validateJson();
            }

            // Display Live Origin Status Bar
            if (dom.llmExtractionStatus) {
                dom.llmExtractionStatus.classList.remove("hidden");
                if (data.is_genuine) {
                    dom.extractionModeBadge.textContent = "🟢 GENUINE (Live API)";
                    dom.extractionModeBadge.className = "status-mode-badge genuine";
                } else {
                    dom.extractionModeBadge.textContent = "🟣 SIMULATED / RANDOM (Fallback)";
                    dom.extractionModeBadge.className = "status-mode-badge simulated";
                }
                dom.extractionNoteText.textContent = data.status_note || "Extracted structured data";
            }
        } catch (err) {
            console.error("VLM Extraction error:", err);
            if (dom.llmExtractionStatus) {
                dom.llmExtractionStatus.classList.remove("hidden");
                dom.extractionModeBadge.textContent = "⚠️ EXTRACTION FAILED";
                dom.extractionModeBadge.className = "status-mode-badge simulated";
                dom.extractionNoteText.textContent = err.message;
            }
        } finally {
            if (dom.extractSpinner) dom.extractSpinner.classList.add("hidden");
            if (dom.extractVlmBtn) dom.extractVlmBtn.disabled = false;
        }
    }

    function switchImageView(viewMode) {
        state.currentImageView = viewMode;
        if (!state.lastVerificationResult) return;

        if (viewMode === "overlay") {
            dom.displayMainImg.src = state.lastVerificationResult.overlay_image_base64;
            dom.viewOverlayBtn.classList.add("active");
            dom.viewOriginalBtn.classList.remove("active");
        } else {
            dom.displayMainImg.src = state.lastVerificationResult.original_image_base64;
            dom.viewOriginalBtn.classList.add("active");
            dom.viewOverlayBtn.classList.remove("active");
        }
    }

    async function runVerification() {
        if (!validateJson()) {
            alert("Please fix the JSON syntax before running verification.");
            return;
        }

        // Show spinner & disable button
        dom.runSpinner.classList.remove("hidden");
        dom.runVerificationBtn.disabled = true;

        const formData = new FormData();
        formData.append("tolerance", state.tolerance);
        formData.append("prediction_json", dom.predictionJsonEditor.value);

        if (state.customImageFile) {
            formData.append("image_file", state.customImageFile);
        } else if (state.selectedScenarioId) {
            formData.append("chart_id", state.selectedScenarioId);
        } else {
            alert("Please select a benchmark chart or upload an image.");
            dom.runSpinner.classList.add("hidden");
            dom.runVerificationBtn.disabled = false;
            return;
        }

        try {
            const resp = await fetch("/api/verify", {
                method: "POST",
                body: formData
            });

            if (!resp.ok) {
                const errData = await resp.json();
                throw new Error(errData.detail || "Verification failed");
            }

            const data = await resp.json();
            state.lastVerificationResult = data;
            renderVerificationResults(data);
        } catch (err) {
            console.error("Verification error:", err);
            alert(`Verification Error: ${err.message}`);
        } finally {
            dom.runSpinner.classList.add("hidden");
            dom.runVerificationBtn.disabled = false;
        }
    }

    // -------------------------------------------------------------------------
    // Render Results in Diagnostics HUD
    // -------------------------------------------------------------------------
    function renderVerificationResults(data) {
        dom.emptyResultsState.classList.add("hidden");
        dom.resultsWrapper.classList.remove("hidden");

        // 1. HUD Scores
        const groundingScore = (data.metrics.overall_grounding_score || 0) * 100;
        dom.hudGroundingScore.textContent = `${groundingScore.toFixed(1)}%`;

        dom.countSupported.textContent = data.counts.supported || 0;
        dom.countContradicted.textContent = data.counts.contradicted || 0;
        dom.countFabricated.textContent = data.counts.fabricated || 0;
        dom.countOmitted.textContent = data.counts.omitted || 0;

        // 2. Display Overlay
        dom.displayMainImg.src = data.overlay_image_base64;
        dom.viewOverlayBtn.classList.add("active");
        dom.viewOriginalBtn.classList.remove("active");
        dom.viewHtmlCertLink.href = data.certificate_url;

        // 3. Proof Cards Grid
        dom.proofCardsGrid.innerHTML = "";
        dom.totalPointsBadge.textContent = `${data.points.length} Claims Audited`;

        data.points.forEach((pt, idx) => {
            const card = document.createElement("div");
            card.className = "proof-card";

            const cropData = data.proof_crops && data.proof_crops[idx] ? data.proof_crops[idx] : null;
            const imgSrc = cropData ? cropData.base64 : "";

            let statusClass = "supported";
            let statusText = "SUPPORTED";
            if (pt.status === "VALUE_CONTRADICTED") {
                statusClass = "contradicted";
                statusText = "VALUE CONTRADICTED";
            } else if (pt.status === "FABRICATED_POINT") {
                statusClass = "fabricated";
                statusText = "FABRICATED POINT";
            } else if (pt.status === "OUT_OF_BOUNDS") {
                statusClass = "contradicted";
                statusText = "OUT OF BOUNDS";
            }

            const errorText = pt.discrepancy !== null ? `Δ = ${pt.discrepancy.toFixed(2)} (${((pt.relative_error || 0) * 100).toFixed(1)}%)` : "";

            card.innerHTML = `
                <img class="proof-crop-img" src="${imgSrc}" alt="Crop x=${pt.x}">
                <div class="proof-meta">
                    <span class="proof-status-badge ${statusClass}">${statusText}</span>
                    <div class="proof-coords">x: ${pt.x} | Claimed y: ${pt.claimed_y}</div>
                    ${pt.observed_y !== null ? `<div style="font-size: 0.72rem; color: var(--color-supported);">Observed mark: y=${pt.observed_y.toFixed(2)}</div>` : `<div style="font-size: 0.72rem; color: var(--color-fabricated);">No marker on canvas</div>`}
                    ${errorText ? `<div class="proof-delta">${errorText}</div>` : ""}
                    <div class="proof-explanation">${pt.explanation}</div>
                </div>
            `;
            dom.proofCardsGrid.appendChild(card);
        });

        // 4. Semantic Text Grounding Table
        dom.textAuditTbody.innerHTML = "";
        const textScore = (data.metrics.text_grounding_score || 0) * 100;
        dom.textGroundingRateBadge.textContent = `Text Grounding: ${textScore.toFixed(1)}%`;

        for (const [field, tr] of Object.entries(data.text_verification || {})) {
            const row = document.createElement("tr");
            let badgeClass = "badge-green";
            if (tr.status === "CONTRADICTED") badgeClass = "chip-tag red";
            if (tr.status === "UNGROUNDED") badgeClass = "chip-tag red";

            row.innerHTML = `
                <td><strong>${field}</strong></td>
                <td><code style="color: #cbd5e1;">${tr.claimed_text}</code></td>
                <td><code style="color: #38bdf8;">${tr.observed_text || "None"}</code></td>
                <td>${(tr.similarity * 100).toFixed(0)}%</td>
                <td><span class="badge ${badgeClass}">${tr.status}</span></td>
            `;
            dom.textAuditTbody.appendChild(row);
        }

        // 5. Affine Calibration Diagnostics
        dom.calibMetricsGrid.innerHTML = `
            <div class="calib-stat-box">
                <span class="calib-stat-label">X-Axis Linear Fit (R²)</span>
                <span class="calib-stat-val">${data.calibration.x_axis.r_squared.toFixed(5)}</span>
            </div>
            <div class="calib-stat-box">
                <span class="calib-stat-label">Y-Axis Linear Fit (R²)</span>
                <span class="calib-stat-val">${data.calibration.y_axis.r_squared.toFixed(5)}</span>
            </div>
            <div class="calib-stat-box">
                <span class="calib-stat-label">Inlier Ticks Snap</span>
                <span class="calib-stat-val">X: ${data.calibration.x_axis.inlier_count} | Y: ${data.calibration.y_axis.inlier_count}</span>
            </div>
            <div class="calib-stat-box">
                <span class="calib-stat-label">Canvas Dimensions</span>
                <span class="calib-stat-val">${data.calibration.plot_box.width} × ${data.calibration.plot_box.height} px</span>
            </div>
        `;
    }

    // -------------------------------------------------------------------------
    // VLM Benchmark Fleet Live Execution
    // -------------------------------------------------------------------------
    async function runFleetBenchmark() {
        dom.fleetSpinner.classList.remove("hidden");
        dom.runFleetBenchmarkBtn.disabled = true;

        try {
            const resp = await fetch("/api/benchmark");
            if (!resp.ok) throw new Error("Benchmark execution failed");
            const models = await resp.json();

            // Sort by MRI descending
            models.sort((a, b) => b.model_reliability_index - a.model_reliability_index);

            dom.benchmarkTbody.innerHTML = "";
            models.forEach((m, rank) => {
                const tr = document.createElement("tr");

                let rankBadge = `#${rank + 1}`;
                if (rank === 0) rankBadge = "🥇 #1";
                if (rank === 1) rankBadge = "🥈 #2";
                if (rank === 2) rankBadge = "🥉 #3";

                let mriColor = "var(--color-supported)";
                if (m.model_reliability_index < 85) mriColor = "#f59e0b";
                if (m.model_reliability_index < 80) mriColor = "var(--color-contradicted)";

                const originBadge = m.is_genuine
                    ? `<span class="mode-badge-genuine">🟢 GENUINE (Live API)</span>`
                    : `<span class="mode-badge-simulated">🟣 SIMULATED / RANDOM</span>`;

                tr.innerHTML = `
                    <td><strong>${rankBadge}</strong></td>
                    <td><strong>${m.model_name}</strong></td>
                    <td>${originBadge}</td>
                    <td>${m.total_points}</td>
                    <td><span style="color: var(--color-supported); font-weight: 600;">${(m.support_rate * 100).toFixed(1)}%</span></td>
                    <td><span style="color: var(--color-contradicted); font-weight: 600;">${(m.hallucination_rate * 100).toFixed(1)}%</span></td>
                    <td>${m.fabricated_points}</td>
                    <td>${m.omitted_points}</td>
                    <td><strong style="color: ${mriColor}; font-size: 1rem;">${m.model_reliability_index.toFixed(1)} / 100</strong></td>
                    <td><div class="table-diag-note">${m.status_note || "Evaluation completed"}</div></td>
                `;
                dom.benchmarkTbody.appendChild(tr);
            });
        } catch (err) {
            console.error("Benchmark error:", err);
            dom.benchmarkTbody.innerHTML = `<tr><td colspan="10" style="color: var(--color-contradicted); text-align: center;">Error running benchmark: ${err.message}</td></tr>`;
        } finally {
            dom.fleetSpinner.classList.add("hidden");
            dom.runFleetBenchmarkBtn.disabled = false;
        }
    }

    // Launch!
    init();
});
