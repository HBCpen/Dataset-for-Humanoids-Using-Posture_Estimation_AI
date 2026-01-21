// Pose Estimation Pipeline - Web Demo JavaScript

document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const uploadZone = document.getElementById('uploadZone');
    const uploadButton = document.getElementById('uploadButton');
    const fileInput = document.getElementById('fileInput');
    const progressSection = document.getElementById('progress');
    const progressBar = document.getElementById('progressBar');
    const progressFilename = document.getElementById('progressFilename');
    const progressStatus = document.getElementById('progressStatus');
    const resultsSection = document.getElementById('results');
    const uploadSection = document.getElementById('upload');
    const resetButton = document.getElementById('resetButton');
    const statsGrid = document.getElementById('statsGrid');
    const previewContainer = document.getElementById('previewContainer');
    const downloadLinks = document.getElementById('downloadLinks');

    // State
    let currentFile = null;

    // Event Listeners
    uploadButton.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    resetButton.addEventListener('click', resetUI);

    // File handling
    async function handleFile(file) {
        currentFile = file;

        // Show progress section
        uploadSection.style.display = 'none';
        progressSection.style.display = 'block';
        resultsSection.style.display = 'none';

        progressFilename.textContent = file.name;
        progressBar.style.width = '0%';
        progressStatus.textContent = 'ファイルをアップロード中...';

        try {
            // Step 1: Upload file
            progressBar.style.width = '20%';
            const uploadResult = await uploadFile(file);

            if (!uploadResult.success) {
                throw new Error(uploadResult.error);
            }

            // Step 2: Process file
            progressBar.style.width = '40%';
            progressStatus.textContent = '姿勢推定を実行中...';

            const processResult = await processFile(uploadResult.filename);

            if (!processResult.success) {
                throw new Error(processResult.error);
            }

            // Step 3: Show results
            progressBar.style.width = '100%';
            progressStatus.textContent = '完了!';

            setTimeout(() => {
                showResults(processResult.result);
            }, 500);

        } catch (error) {
            progressStatus.textContent = `エラー: ${error.message}`;
            progressBar.style.background = 'var(--color-error)';

            setTimeout(() => {
                resetUI();
            }, 3000);
        }
    }

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        return await response.json();
    }

    async function processFile(filename) {
        const response = await fetch('/api/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filename })
        });

        return await response.json();
    }

    function showResults(result) {
        progressSection.style.display = 'none';
        resultsSection.style.display = 'block';

        // Populate stats
        const stats = [
            { value: result.total_frames || '—', label: '総フレーム数' },
            { value: result.frames_with_pose || '—', label: '検出フレーム' },
            { value: result.detection_rate ? `${(result.detection_rate * 100).toFixed(1)}%` : '—', label: '検出率' },
            { value: result.pose_detected !== undefined ? (result.pose_detected ? '✓' : '✗') : '—', label: 'ポーズ検出' }
        ];

        statsGrid.innerHTML = stats.map(stat => `
            <div class="stat-item">
                <div class="stat-value">${stat.value}</div>
                <div class="stat-label">${stat.label}</div>
            </div>
        `).join('');

        // Preview
        if (result.output_video) {
            previewContainer.innerHTML = `
                <video controls>
                    <source src="/api/results/${result.output_video.split('/').pop()}" type="video/mp4">
                </video>
            `;
        } else if (result.output_image) {
            previewContainer.innerHTML = `
                <img src="/api/results/${result.output_image.split('/').pop()}" alt="Result">
            `;
        } else {
            previewContainer.innerHTML = '<p style="color: var(--color-text-muted);">プレビューなし</p>';
        }

        // Download links
        const downloads = [];

        if (result.output_json) {
            downloads.push({
                url: `/api/results/${result.output_json.split('/').pop()}`,
                label: 'JSONデータ',
                icon: '{ }'
            });
        }

        if (result.output_video) {
            downloads.push({
                url: `/api/results/${result.output_video.split('/').pop()}`,
                label: '骨格動画',
                icon: '▶'
            });
        }

        downloadLinks.innerHTML = downloads.map(dl => `
            <a href="${dl.url}" class="download-link" download>
                <span style="font-family: monospace; font-weight: bold;">${dl.icon}</span>
                ${dl.label}
            </a>
        `).join('');
    }

    function resetUI() {
        uploadSection.style.display = 'block';
        progressSection.style.display = 'none';
        resultsSection.style.display = 'none';
        progressBar.style.width = '0%';
        progressBar.style.background = 'var(--color-accent-gradient)';
        fileInput.value = '';
        currentFile = null;
    }

    // Smooth scroll for nav links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
});
