const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultsArea = document.getElementById('results-area');
const loading = document.getElementById('loading');
const originalImg = document.getElementById('original-img');
const resultImg = document.getElementById('result-img');
const detectionList = document.getElementById('detection-list');
const sliderRange = document.getElementById('slider-range');
const beforeContainer = document.getElementById('before-container');
const pdfBtn = document.getElementById('pdf-btn');

// Session State
let sessionData = {
    totalScans: 0,
    totalDamages: 0,
    currentFilename: '',
    currentDetections: []
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

        // Store state
        sessionData.totalScans++;
        sessionData.totalDamages += data.detections.length;
        sessionData.currentFilename = data.result_url.split('/').pop().replace('detected_', '');
        sessionData.currentDetections = data.detections;
        updateStatsUI();

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

// PDF Generation
pdfBtn.addEventListener('click', () => {
    if (!sessionData.currentFilename) return;

    pdfBtn.disabled = true;
    const originalText = pdfBtn.innerHTML;
    pdfBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Generating...';

    // Ensure we send a clean filename without query params
    const cleanFilename = sessionData.currentFilename.split('?')[0];

    fetch('/generate-pdf', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            filename: cleanFilename,
            detections: sessionData.currentDetections
        })
    })
    .then(async response => {
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Server error');
        }
        return response.blob();
    })
    .then(blob => {
        if (blob.type !== 'application/pdf') {
            throw new Error('Received invalid file format');
        }
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Damage_Report_${cleanFilename.split('.')[0]}.pdf`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        
        pdfBtn.disabled = false;
        pdfBtn.innerHTML = originalText;
    })
    .catch(err => {
        console.error(err);
        alert('Failed to generate PDF: ' + err.message);
        pdfBtn.disabled = false;
        pdfBtn.innerHTML = originalText;
    });
});

function resetApp() {
    resultsArea.classList.add('hidden');
    dropZone.classList.remove('hidden');
    fileInput.value = '';
    originalImg.src = '';
    resultImg.src = '';
}
