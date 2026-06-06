document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const previewSection = document.getElementById('preview-section');
    const previewImage = document.getElementById('preview-image');
    const btnReselect = document.getElementById('btn-reselect');
    const btnPredict = document.getElementById('btn-predict');
    const loadingIndicator = document.getElementById('loading-indicator');
    const resultSection = document.getElementById('result-section');
    const predictionText = document.getElementById('prediction-text');

    let currentFile = null;

    // Click to upload
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Drag and drop events
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Reselect button
    btnReselect.addEventListener('click', () => {
        resetUI();
        fileInput.click();
    });

    // Predict button
    btnPredict.addEventListener('click', async () => {
        if (!currentFile) return;

        // Show loading
        previewSection.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');
        resultSection.classList.add('hidden');

        // Create form data
        const formData = new FormData();
        formData.append('image', currentFile);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            loadingIndicator.classList.add('hidden');
            resultSection.classList.remove('hidden');
            previewSection.classList.remove('hidden');
            btnPredict.classList.add('hidden'); // Hide predict button after success

            if (data.success) {
                predictionText.textContent = data.prediction;
                predictionText.style.color = '#fff';
            } else {
                predictionText.textContent = 'LỖI: ' + data.error;
                predictionText.style.color = '#ef4444';
                predictionText.style.fontSize = '1.5rem';
                predictionText.style.textShadow = 'none';
            }
        } catch (error) {
            console.error('Error:', error);
            loadingIndicator.classList.add('hidden');
            resultSection.classList.remove('hidden');
            previewSection.classList.remove('hidden');
            btnPredict.classList.add('hidden');
            
            predictionText.textContent = 'Lỗi kết nối máy chủ';
            predictionText.style.color = '#ef4444';
            predictionText.style.fontSize = '1.5rem';
            predictionText.style.textShadow = 'none';
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Vui lòng chọn một file ảnh hợp lệ!');
            return;
        }

        currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadZone.classList.add('hidden');
            previewSection.classList.remove('hidden');
            resultSection.classList.add('hidden');
            btnPredict.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    }

    function resetUI() {
        currentFile = null;
        fileInput.value = '';
        uploadZone.classList.remove('hidden');
        previewSection.classList.add('hidden');
        resultSection.classList.add('hidden');
        loadingIndicator.classList.add('hidden');
        
        predictionText.textContent = '';
        predictionText.style.color = '#fff';
        predictionText.style.fontSize = '3rem';
    }
});
