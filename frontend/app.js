// Image Classification System - Frontend Logic
document.addEventListener("DOMContentLoaded", () => {
    // API Base URL - Automatically handles both local server and separate frontend port
    const API_BASE_URL = window.location.origin.includes("8000") 
        ? "" 
        : "http://127.0.0.1:8000";

    const STORAGE_KEY = "cifar10_classification_history";
    const MAX_HISTORY_ITEMS = 12;

    // DOM Elements - Upload & Preview
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const dropContent = document.getElementById("dropContent");
    const previewArea = document.getElementById("previewArea");
    const imagePreview = document.getElementById("imagePreview");
    const previewFileName = document.getElementById("previewFileName");
    const previewFileSize = document.getElementById("previewFileSize");
    const removeImageBtn = document.getElementById("removeImageBtn");
    const predictBtn = document.getElementById("predictBtn");
    const resetBtn = document.getElementById("resetBtn");
    const predictBtnText = document.getElementById("predictBtnText");
    const predictSpinner = document.getElementById("predictSpinner");

    // DOM Elements - Alerts
    const errorBanner = document.getElementById("errorBanner");
    const errorMessage = document.getElementById("errorMessage");
    const closeErrorBtn = document.getElementById("closeErrorBtn");

    // DOM Elements - Results
    const resultsSection = document.getElementById("resultsSection");
    const emptyResultState = document.getElementById("emptyResultState");
    const loadingState = document.getElementById("loadingState");
    const predictionContent = document.getElementById("predictionContent");
    const predictedClassName = document.getElementById("predictedClassName");
    const topConfidenceTag = document.getElementById("topConfidenceTag");
    const topConfidenceBar = document.getElementById("topConfidenceBar");
    const top5List = document.getElementById("top5List");
    const inferenceBadge = document.getElementById("inferenceBadge");

    // DOM Elements - History
    const historyGrid = document.getElementById("historyGrid");
    const emptyHistoryState = document.getElementById("emptyHistoryState");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    const historyCount = document.getElementById("historyCount");

    // State
    let currentFile = null;
    let currentPreviewDataUrl = null;

    // Initialize application
    init();

    function init() {
        setupEventListeners();
        loadHistory();
        fetchModelInfo();
    }

    function setupEventListeners() {
        // Drag & Drop Events
        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.add("drag-over");
            });
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropZone.classList.remove("drag-over");
            });
        });

        dropZone.addEventListener("drop", (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                handleFileSelection(files[0]);
            }
        });

        // File Input Change
        fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelection(e.target.files[0]);
            }
        });

        // Action Buttons
        removeImageBtn.addEventListener("click", resetUploadState);
        resetBtn.addEventListener("click", resetAll);
        predictBtn.addEventListener("click", handlePredict);
        closeErrorBtn.addEventListener("click", hideError);
        clearHistoryBtn.addEventListener("click", handleClearHistory);
    }

    // --- File Handling & Validation ---
    function handleFileSelection(file) {
        hideError();

        if (!file) return;

        // 1. Validate file extension & MIME type
        const validExtensions = [".jpg", ".jpeg", ".png"];
        const fileExt = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
        const validTypes = ["image/jpeg", "image/png", "image/jpg"];

        if (!validExtensions.includes(fileExt) && !validTypes.includes(file.type)) {
            showError(`Unsupported file format "${file.name}". Please upload a JPEG or PNG image.`);
            return;
        }

        // 2. Validate file size (10MB limit)
        const maxSizeBytes = 10 * 1024 * 1024;
        if (file.size > maxSizeBytes) {
            const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
            showError(`File size (${sizeMB} MB) exceeds the 10 MB maximum upload limit.`);
            return;
        }

        // 3. Set current file and generate preview
        currentFile = file;

        const reader = new FileReader();
        reader.onload = (e) => {
            currentPreviewDataUrl = e.target.result;
            imagePreview.src = currentPreviewDataUrl;
            previewFileName.textContent = file.name;
            previewFileSize.textContent = formatBytes(file.size);

            dropContent.classList.add("hidden");
            previewArea.classList.remove("hidden");
            predictBtn.disabled = false;
        };

        reader.onerror = () => {
            showError("Failed to read the selected file. Please try another image.");
            resetUploadState();
        };

        reader.readAsDataURL(file);
    }

    function resetUploadState() {
        currentFile = null;
        currentPreviewDataUrl = null;
        fileInput.value = "";
        imagePreview.src = "";
        previewArea.classList.add("hidden");
        dropContent.classList.remove("hidden");
        predictBtn.disabled = true;
    }

    function resetAll() {
        resetUploadState();
        hideError();
        emptyResultState.classList.remove("hidden");
        loadingState.classList.add("hidden");
        predictionContent.classList.add("hidden");
        inferenceBadge.classList.add("hidden");
    }

    // --- Prediction Workflow ---
    async function handlePredict() {
        if (!currentFile) {
            showError("Please select or drop an image first.");
            return;
        }

        hideError();
        setLoading(true);

        const formData = new FormData();
        formData.append("file", currentFile);

        try {
            const response = await fetch(`${API_BASE_URL}/api/predict`, {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                // Backend returned an error response
                const errorMsg = data.detail || `Server returned error status ${response.status}`;
                showError(errorMsg);
                showEmptyResults();
                return;
            }

            // Success: Display prediction results
            displayPredictionResults(data);

            // Save to recent history
            savePredictionToHistory({
                id: Date.now().toString(),
                fileName: currentFile.name,
                predictedClass: data.predicted_class,
                confidencePercentage: data.confidence_percentage,
                inferenceTimeMs: data.inference_time_ms,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                thumbnail: currentPreviewDataUrl
            });

        } catch (error) {
            console.error("[Prediction Error]", error);
            showError(`Unable to connect to the classification server at ${API_BASE_URL}. Please ensure the FastAPI backend is running.`);
            showEmptyResults();
        } finally {
            setLoading(false);
        }
    }

    function displayPredictionResults(data) {
        emptyResultState.classList.add("hidden");
        loadingState.classList.add("hidden");
        predictionContent.classList.remove("hidden");

        // Top Prediction
        predictedClassName.textContent = data.predicted_class;
        topConfidenceTag.textContent = `${data.confidence_percentage.toFixed(2)}%`;
        topConfidenceBar.style.width = `${Math.min(data.confidence_percentage, 100)}%`;

        // Inference Latency
        inferenceBadge.textContent = `⚡ ${data.inference_time_ms.toFixed(1)} ms`;
        inferenceBadge.classList.remove("hidden");

        // Top 5 Breakdown
        top5List.innerHTML = "";
        data.top_5.forEach((item, index) => {
            const itemEl = document.createElement("div");
            itemEl.className = "top5-item";

            const isTop = index === 0;
            const barWidth = Math.min(item.percentage, 100);

            itemEl.innerHTML = `
                <div class="top5-item-header">
                    <span class="top5-class-name">${index + 1}. ${item.class_name} ${isTop ? '★' : ''}</span>
                    <span class="top5-percentage">${item.percentage.toFixed(2)}%</span>
                </div>
                <div class="top5-bar-track">
                    <div class="top5-bar-fill" style="width: ${barWidth}%;"></div>
                </div>
            `;
            top5List.appendChild(itemEl);
        });
    }

    function showEmptyResults() {
        emptyResultState.classList.remove("hidden");
        loadingState.classList.add("hidden");
        predictionContent.classList.add("hidden");
        inferenceBadge.classList.add("hidden");
    }

    function setLoading(isLoading) {
        if (isLoading) {
            predictBtn.disabled = true;
            resetBtn.disabled = true;
            predictSpinner.classList.remove("hidden");
            predictBtnText.textContent = "Classifying...";
            emptyResultState.classList.add("hidden");
            predictionContent.classList.add("hidden");
            loadingState.classList.remove("hidden");
        } else {
            predictBtn.disabled = !currentFile;
            resetBtn.disabled = false;
            predictSpinner.classList.add("hidden");
            predictBtnText.textContent = "Classify Image";
            loadingState.classList.add("hidden");
        }
    }

    // --- Error Notifications ---
    function showError(msg) {
        errorMessage.textContent = msg;
        errorBanner.classList.remove("hidden");
        errorBanner.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function hideError() {
        errorBanner.classList.add("hidden");
        errorMessage.textContent = "";
    }

    // --- History Management (localStorage) ---
    function loadHistory() {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            const history = raw ? JSON.parse(raw) : [];
            renderHistory(history);
        } catch (e) {
            console.error("Failed to load history from localStorage:", e);
            renderHistory([]);
        }
    }

    function savePredictionToHistory(item) {
        try {
            const raw = localStorage.getItem(STORAGE_KEY);
            let history = raw ? JSON.parse(raw) : [];

            // Add new item to beginning
            history.unshift(item);

            // Keep only latest items
            if (history.length > MAX_HISTORY_ITEMS) {
                history = history.slice(0, MAX_HISTORY_ITEMS);
            }

            localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
            renderHistory(history);
        } catch (e) {
            console.warn("Failed to persist history to localStorage (possibly storage quota exceeded):", e);
        }
    }

    function renderHistory(history) {
        historyCount.textContent = `${history.length} saved`;

        if (!history || history.length === 0) {
            emptyHistoryState.classList.remove("hidden");
            historyGrid.classList.add("hidden");
            clearHistoryBtn.disabled = true;
            return;
        }

        emptyHistoryState.classList.add("hidden");
        historyGrid.classList.remove("hidden");
        clearHistoryBtn.disabled = false;

        historyGrid.innerHTML = "";
        history.forEach(item => {
            const card = document.createElement("div");
            card.className = "history-card-item";

            card.innerHTML = `
                <div class="history-img-wrapper">
                    <img src="${item.thumbnail}" alt="${item.predictedClass}" class="history-thumb">
                </div>
                <div class="history-details">
                    <div class="history-class">
                        <span>${item.predictedClass}</span>
                        <span class="history-conf">${item.confidencePercentage.toFixed(1)}%</span>
                    </div>
                    <div class="history-meta">
                        <span>${item.timestamp}</span>
                        <span>⚡ ${item.inferenceTimeMs.toFixed(0)} ms</span>
                    </div>
                </div>
            `;

            // Clicking a history card previews the image
            card.style.cursor = "pointer";
            card.title = `Click to view ${item.fileName}`;
            card.addEventListener("click", () => {
                imagePreview.src = item.thumbnail;
                previewFileName.textContent = item.fileName;
                previewFileSize.textContent = "(from history)";
                dropContent.classList.add("hidden");
                previewArea.classList.remove("hidden");
            });

            historyGrid.appendChild(card);
        });
    }

    function handleClearHistory() {
        try {
            localStorage.removeItem(STORAGE_KEY);
            renderHistory([]);
        } catch (e) {
            console.error("Failed to clear localStorage:", e);
        }
    }

    // --- Backend Model Info Fetch ---
    async function fetchModelInfo() {
        try {
            const res = await fetch(`${API_BASE_URL}/api/model/info`);
            if (res.ok) {
                const info = await res.json();
                const container = document.getElementById("modelInfoPills");
                if (container && info) {
                    container.innerHTML = `
                        <div class="pill"><strong>Architecture:</strong> ${info.architecture}</div>
                        <div class="pill"><strong>Dataset:</strong> ${info.dataset} (${info.number_of_classes} classes)</div>
                        <div class="pill"><strong>Top-1 Acc:</strong> ${info.top_1_accuracy}</div>
                        <div class="pill"><strong>Top-5 Acc:</strong> ${info.top_5_accuracy}</div>
                        <div class="pill pill-latency"><strong>Status:</strong> ${info.status} (${info.device})</div>
                    `;
                }
            }
        } catch (e) {
            // Non-blocking: background info fetch
            console.log("Could not reach /api/model/info yet; using defaults.");
        }
    }

    // Utility: Format file size
    function formatBytes(bytes) {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
    }
});
