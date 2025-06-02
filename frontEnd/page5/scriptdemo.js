document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const imagePlaceholder = document.getElementById('imagePlaceholder');
    const uploadedImage = document.getElementById('uploadedImage');
    const uploadLabel = document.querySelector('.upload-label');
    const resultDisplay = document.getElementById('resultDisplay');
    const detectButton = document.getElementById('detectButton');

    imageUpload.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedImage.src = e.target.result;
                uploadedImage.style.display = 'block';
                uploadLabel.style.display = 'none'; // Sembunyikan teks "Pilih Gambar +"
                resultDisplay.textContent = ''; // Kosongkan hasil sebelumnya
            }
            reader.readAsDataURL(file);
        }
    });

    detectButton.addEventListener('click', function() {
        if (uploadedImage.src === '#' || uploadedImage.style.display === 'none') {
            resultDisplay.textContent = 'Silakan unggah gambar terlebih dahulu.';
            resultDisplay.style.fontWeight = 'bold'
            resultDisplay.style.fontSize = '1.2em';
            resultDisplay.style.color = 'black';
            return;
        }

        // --- Simulasi Proses Deteksi ---
        resultDisplay.textContent = 'Mendeteksi...';

        // Ganti ini dengan logika pemanggilan API ke backend model ML Anda
        setTimeout(() => {
            const randomNumber = Math.random();
            if (randomNumber > 0.5) {
                resultDisplay.textContent = 'Terindikasi Kanker Otak.';
                resultDisplay.style.color = 'red';
                resultDisplay.style.fontWeight = 'bold';
                resultDisplay.style.fontSize = '1.2em';
            } else {
                resultDisplay.textContent = 'Tidak Terindikasi Kanker Otak.';
                resultDisplay.style.color = 'green';
                resultDisplay.style.fontWeight = 'bold';
                resultDisplay.style.fontSize = '1.2em'; 
            }
        }, 2000); // Simulasi waktu proses 2 detik
        // --- Akhir Simulasi ---
    });

    // Untuk fungsionalitas klik pada area placeholder gambar
    imagePlaceholder.addEventListener('click', () => {
        if (uploadedImage.style.display === 'none' || uploadedImage.src === '#') {
            imageUpload.click();
        }
    });
});