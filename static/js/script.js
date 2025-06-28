function openTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
    document.getElementById(tabName).classList.add('active');
    document.querySelector(`.tab[onclick="openTab('${tabName}')"]`).classList.add('active');
}

function previewImage(fileInputId, previewId) {
    const fileInput = document.getElementById(fileInputId);
    const preview = document.getElementById(previewId);
    preview.innerHTML = '';

    if (fileInput.files && fileInput.files[0]) {
        const url = URL.createObjectURL(fileInput.files[0]);
        const img = document.createElement('img');
        img.src = url;
        img.alt = 'Pratinjau Gambar';
        img.style.maxWidth = '100%';
        preview.appendChild(img);
    }
}

async function embedMessage() {
    const fileInput = document.getElementById('embedFile');
    const message = document.getElementById('message').value;
    const output = document.getElementById('embedOutput');
    const progress = document.getElementById('progress');
    const progressBar = document.querySelector('.progress-bar');

    if (!fileInput.files[0] || !message) {
        output.innerHTML = '<p class="error">Pilih gambar dan masukkan pesan!</p>';
        return;
    }

    output.innerHTML = '';
    progress.classList.add('visible');
    progressBar.style.width = '0%';

    let progressValue = 0;
    const progressInterval = setInterval(() => {
        progressValue += 10;
        progressBar.style.width = `${progressValue}%`;
        if (progressValue >= 80) clearInterval(progressInterval);
    }, 200);

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('message', message);

    try {
        const response = await fetch('/embed', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        setTimeout(() => progress.classList.remove('visible'), 500);

        if (response.ok) {
            output.innerHTML = `<p>${result.message}</p><a href="${result.file}" download>⬇️ Unduh Gambar Stego</a>`;
        } else {
            output.innerHTML = `<p class="error">${result.error}</p>`;
        }
    } catch (error) {
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        setTimeout(() => progress.classList.remove('visible'), 500);
        output.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
}

async function extractMessage() {
    const fileInput = document.getElementById('extractFile');
    const output = document.getElementById('extractOutput');
    const progress = document.getElementById('progress');
    const progressBar = document.querySelector('.progress-bar');

    if (!fileInput.files[0]) {
        output.innerHTML = '<p class="error">Pilih gambar stego!</p>';
        return;
    }

    output.innerHTML = '';
    progress.classList.add('visible');
    progressBar.style.width = '0%';

    let progressValue = 0;
    const progressInterval = setInterval(() => {
        progressValue += 10;
        progressBar.style.width = `${progressValue}%`;
        if (progressValue >= 80) clearInterval(progressInterval);
    }, 200);

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/extract', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();

        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        setTimeout(() => progress.classList.remove('visible'), 500);

        if (response.ok && !result.message.startsWith('Error') && !result.message.startsWith('Tidak ditemukan')) {
            output.innerHTML = `<p>✅ Pesan Tersembunyi: <br><span style="font-weight: bold;">${result.message}</span></p>`;
        } else {
            output.innerHTML = `<p class="error">${result.message}</p>`;
        }
    } catch (error) {
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        setTimeout(() => progress.classList.remove('visible'), 500);
        output.innerHTML = `<p class="error">Error: ${error.message}</p>`;
    }
}