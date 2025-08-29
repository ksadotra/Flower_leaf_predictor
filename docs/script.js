document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const predictButton = document.getElementById('predictButton');
    const resultText = document.getElementById('resultText');
    const segmentedImageContainer = document.getElementById('segmentedImageContainer');
    const segmentedImage = document.getElementById('segmentedImage');

    // THIS IS THE ONLY LINE YOU WILL EDIT LATER
    const BACKEND_URL = 'https://flower-leaf-predictor-mjuj.onrender.com';

    if (imageUpload) {
        imageUpload.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';
                    resultText.textContent = 'Image loaded. Click "Predict".';
                    segmentedImageContainer.style.display = 'none';
                };
                reader.readAsDataURL(file);
            }
        });
    }

    if (predictButton) {
        predictButton.addEventListener('click', async () => {
            const file = imageUpload.files[0];
            if (!file) {
                alert('Please upload an image first!');
                return;
            }

            resultText.textContent = 'Analyzing image and predicting...';
            segmentedImageContainer.style.display = 'none';
            
            const formData = new FormData();
            formData.append('image', file);

            try {
                if (BACKEND_URL.includes('https://flower-leaf-predictor-mjuj.onrender.com')) {
                    alert('Please update the BACKEND_URL in script.js with your live server URL first!');
                    resultText.textContent = 'Backend URL not configured.';
                    return;
                }

                const response = await fetch(BACKEND_URL, {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    let predictionMessage = `Prediction: ${data.prediction} (${data.confidence.toFixed(2)}%)`;
                    if (data.warning) {
                        predictionMessage += `\n(${data.warning})`;
                        resultText.style.color = 'orange';
                    } else {
                        resultText.style.color = '#333';
                    }
                    resultText.textContent = predictionMessage;

                    if (data.segmented_image) {
                        segmentedImage.src = `data:image/png;base64,${data.segmented_image}`;
                        segmentedImage.style.display = 'block';
                        segmentedImageContainer.style.display = 'block';
                    }
                } else {
                    resultText.textContent = `Error: ${data.error || 'Prediction failed'}`;
                    resultText.style.color = 'red';
                }
            } catch (error) {
                console.error('Error during prediction:', error);
                resultText.textContent = 'Failed to connect to the backend. Please check the console.';
                resultText.style.color = 'red';
            }
        });
    }
});
