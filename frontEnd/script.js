const form = document.getElementById("uploadForm");
const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
const resultDiv = document.getElementById("result");

imageInput.addEventListener("change", function () {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        previewImage.hidden = false;

        reader.onload = function (e) {
            previewImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
});

form.addEventListener("submit", async function (e) {
    e.preventDefault();

    const file = imageInput.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("image", file);

    resultDiv.textContent = "🔄 Processing...";

    try {
        const response = await fetch('http://localhost:5001/predict', {
            method: 'POST',
            body: formData  // Gunakan formData langsung
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        if (data.prediction) {
            resultDiv.textContent = `🧪 Prediction: ${data.prediction.toUpperCase()}`;
        } else {
            resultDiv.textContent = `❌ Error: ${data.error}`;
        }
    } catch (err) {
        resultDiv.textContent = `❌ Failed to connect to server`;
        console.error(err);
    }
});