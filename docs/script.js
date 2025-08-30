document.addEventListener('DOMContentLoaded', () => {
    // --- CONFIGURATION ---
    // IMPORTANT: Replace this with the actual URL of your deployed Flask backend.
    const API_URL = 'https://flower-leaf-predictor-backend.onrender.com/predict'; // Example for local development

    // --- DOM Element References ---
    const imageUploadInput = document.getElementById('imageUpload');
    const predictButton = document.getElementById('predictButton');
    const imagePreview = document.getElementById('imagePreview');
    const resultText = document.getElementById('resultText');
    const imagePreviewContainer = document.getElementById('imagePreviewContainer');
    const predictionResultContainer = document.getElementById('predictionResult');
    const uploadLabel = document.querySelector('.upload-label span');

    let selectedFile = null;

    // --- EVENT LISTENERS ---

    /**
     * Handles the file selection event. It reads the selected image file,
     * displays its preview, and prepares it for upload.
     */
    imageUploadInput.addEventListener('change', (event) => {
        const files = event.target.files;
        if (files && files.length > 0) {
            selectedFile = files[0];

            // Use FileReader to display the image preview
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreviewContainer.style.display = 'block'; // Show the preview container
            };
            reader.readAsDataURL(selectedFile);

            // Update UI elements
            uploadLabel.textContent = selectedFile.name; // Show the filename
            predictionResultContainer.style.display = 'none'; // Hide previous results
            resultText.textContent = '';
        }
    });

    /**
     * Handles the click event for the 'Predict' button. It sends the
     * selected image to the backend API and displays the result.
     */
    predictButton.addEventListener('click', async () => {
        if (!selectedFile) {
            alert('Please upload an image first.');
            return;
        }

        // Set UI to a loading state
        predictButton.disabled = true;
        predictButton.textContent = 'Analyzing...';
        resultText.textContent = '';
        predictionResultContainer.style.display = 'none';

        // Prepare the form data to send the image
        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                // Handle server-side errors (e.g., 500 Internal Server Error)
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const result = await response.json();

            // Display the result or an error from the API
            if (result.error) {
                resultText.textContent = `Error: ${result.error}`;
            } else {
                resultText.textContent = `Result: ${result.prediction}`;
            }

        } catch (error) {
            console.error('Error during prediction:', error);
            resultText.textContent = 'An error occurred. Please try again.';
        } finally {
            // Restore the UI from the loading state
            predictionResultContainer.style.display = 'block'; // Show the result container
            predictButton.disabled = false;
            predictButton.textContent = 'Predict';
        }
    });
});
