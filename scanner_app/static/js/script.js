const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const resultsArea = document.getElementById('results-area');
const loading = document.getElementById('loading');
const originalImg = document.getElementById('original-img');
const resultImg = document.getElementById('result-img');
const detectionList = document.getElementById('detection-list');

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

        // Show result
        resultImg.src = data.result_url + '?t=' + new Date().getTime(); // Prevent caching
        
        // Show detections
        if (data.detections.length === 0) {
            detectionList.innerHTML = '<p style="color: var(--text-dim)">No damages detected. The vehicle appears clean!</p>';
        } else {
            data.detections.forEach(det => {
                const item = document.createElement('div');
                item.className = 'detection-item';
                item.innerHTML = `
                    <span class="det-name">${det.class.replace('_', ' ')}</span>
                    <span class="det-conf">Confidence: ${det.confidence}%</span>
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

function resetApp() {
    resultsArea.classList.add('hidden');
    dropZone.classList.remove('hidden');
    fileInput.value = '';
    originalImg.src = '';
    resultImg.src = '';
}
