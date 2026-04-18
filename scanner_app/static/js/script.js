const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultsArea = document.getElementById('results-area');
const loading = document.getElementById('loading');
const originalImg = document.getElementById('original-img');
const resultImg = document.getElementById('result-img');
const detectionList = document.getElementById('detection-list');
const sliderRange = document.getElementById('slider-range');
const beforeContainer = document.getElementById('before-container');
const totalCostEl = document.getElementById('total-cost');

// Camera Elements
const openCameraBtn = document.getElementById('open-camera-btn');
const cameraModal = document.getElementById('camera-modal');
const cameraPreview = document.getElementById('camera-preview');
const captureBtn = document.getElementById('capture-btn');
const closeCameraBtn = document.getElementById('close-camera-btn');
const captureCanvas = document.getElementById('capture-canvas');

// State
let stream = null;
const COST_MAP = {
    'scratch': 2000,
    'dent': 5500,
    'crack': 3500,
    'glass_damage': 8500,
    'lamp_damage': 4000,
    'deformation': 15000
};

let sessionData = {
    totalScans: 0,
    totalDamages: 0
};

// Setup interactions
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        handleUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleUpload(e.target.files[0]);
    }
});

// Camera Operations
openCameraBtn.addEventListener('click', async (e) => {
    e.stopPropagation();
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' },
            audio: false
        });
        cameraPreview.srcObject = stream;
        cameraModal.classList.remove('hidden');
    } catch (err) {
        console.error(err);
        alert('Could not access camera. Please check permissions.');
    }
});

closeCameraBtn.addEventListener('click', () => {
    stopCamera();
    cameraModal.classList.add('hidden');
});

captureBtn.addEventListener('click', () => {
    const context = captureCanvas.getContext('2d');
    captureCanvas.width = cameraPreview.videoWidth;
    captureCanvas.height = cameraPreview.videoHeight;
    context.drawImage(cameraPreview, 0, 0, captureCanvas.width, captureCanvas.height);
    
    captureCanvas.toBlob((blob) => {
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });
        stopCamera();
        cameraModal.classList.add('hidden');
        handleUpload(file);
    }, 'image/jpeg', 0.9);
});

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

// Comparison Slider Logic
sliderRange.addEventListener('input', (e) => {
    const value = e.target.value;
    beforeContainer.style.width = value + '%';
});

function handleUpload(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    // UI state
    dropZone.classList.add('hidden');
    resultsArea.classList.remove('hidden');
    loading.classList.remove('hidden');
    detectionList.innerHTML = '';
    sliderRange.value = 50;
    beforeContainer.style.width = '50%';
    totalCostEl.innerText = '₹0';
    
    // Preview original
    const reader = new FileReader();
    reader.onload = (e) => {
        originalImg.src = e.target.result;
    };
    reader.readAsDataURL(file);

    // Prepare upload
    const formData = new FormData();
    formData.append('image', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loading.classList.add('hidden');
        if (data.error) {
            alert(data.error);
            resetApp();
            return;
        }

        // Store state & Calculate Costs
        sessionData.totalScans++;
        sessionData.totalDamages += data.detections.length;
        
        let totalCost = 0;
        data.detections.forEach(det => {
            const cost = COST_MAP[det.class] || 2500;
            totalCost += cost;
        });

        // Update UI
        updateStatsUI();
        animateValue(totalCostEl, 0, totalCost, 1000);

        // Show result
        resultImg.src = data.result_url + '?t=' + new Date().getTime();
        
        // Show detections
        if (data.detections.length === 0) {
            detectionList.innerHTML = '<p style="color: var(--text-dim)">No damages detected. The vehicle appears clean!</p>';
        } else {
            data.detections.forEach(det => {
                const item = document.createElement('div');
                item.className = 'detection-item';
                item.innerHTML = `
                    <span class="det-name">${det.class.replace('_', ' ').toUpperCase()}</span>
                    <span class="det-conf">${det.confidence}% Confidence</span>
                `;
                detectionList.appendChild(item);
            });
        }
    })
    .catch(err => {
        console.error(err);
        alert('Analysis failed. Please try again.');
        resetApp();
    });
}

function updateStatsUI() {
    document.getElementById('total-scans').innerText = sessionData.totalScans;
    document.getElementById('total-damages').innerText = sessionData.totalDamages;
}

function animateValue(obj, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const currentVal = Math.floor(progress * (end - start) + start);
        obj.innerHTML = `₹${currentVal.toLocaleString('en-IN')}`;
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

function resetApp() {
    resultsArea.classList.add('hidden');
    dropZone.classList.remove('hidden');
    fileInput.value = '';
    originalImg.src = '';
    resultImg.src = '';
}
