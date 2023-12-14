function previewImage() {
    const input = document.getElementById('imageInput');
    const errorMessages = document.getElementById('errorMessagesPredict');
    const preview = document.getElementById('imagePreview');

    errorMessages.innerHTML = ''; 

    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function (e) {
            preview.style.backgroundImage = "url('" + e.target.result + "')";
        };
        reader.readAsDataURL(input.files[0]);
    }
}

function trainModel() {
    const errorMessages = document.getElementById('errorMessagestrain');
    const loadingSpinner = document.getElementById('loadingSpinner');
    errorMessages.innerHTML = '';

    // Get the selected model from the dropdown
    const selectedModel = document.getElementById('modelSelect').value;

    // Show loading spinner
    loadingSpinner.style.display = 'block';

    // Disable buttons during training
    document.getElementById('predictButton').disabled = true;
    document.getElementById('trainButton').disabled = true;
    document.getElementById('liveButton').disabled = true;

    // Fetch to the '/train' route with the selected model
    fetch('/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_choice: selectedModel }),  // Pass the selected model
    })
    .then(response => response.json())
    .then(data => {
        if (!data.success) {
            errorMessages.innerHTML = `<p class="alert alert-danger text-white font-weight-bold" role="alert">Error: ${data.message}</p>`;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        errorMessages.innerHTML = `<p class="alert alert-danger text-white font-weight-bold" role="alert">Error: ${error.message}</p>`;
    })
    .finally(() => {
        // Hide loading spinner and enable buttons
        loadingSpinner.style.display = 'none';
        document.getElementById('predictButton').disabled = false;
        document.getElementById('trainButton').disabled = false;
        document.getElementById('liveButton').disabled = false;
    });
}


function predict() {
    const input = document.getElementById('imageInput');
    const file = input.files[0];
    const errorMessages = document.getElementById('errorMessagesPredict');
    const modelChoice = document.getElementById('modelSelect2').value;

    errorMessages.innerHTML = '';

    if (!file) {
        errorMessages.innerHTML = '<p class="alert alert-danger text-white font-weight-bold" role="alert">Please choose an image.</p>';
        return;
    }

    const formData = new FormData();
    formData.append('image', file);
    formData.append('model_choice', modelChoice);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        const predictionsDiv = document.getElementById('predictions');

        predictionsDiv.style.display ="block";
        predictionsDiv.innerHTML = `<h4 class="text-white">Predictions:</h4>`;
        i=0;
        for (const [classLabel, probability] of Object.entries(data.predictions)) {
            i++;
            i ===7? predictionsDiv.innerHTML += `<span>${classLabel}: ${((probability.toFixed(2)))+"%"}</span> ` : predictionsDiv.innerHTML += `<span>${classLabel}: ${((probability.toFixed(2)))+"%"} | </span> `;
        }
    })
    .catch(error => console.error('Error:', error));
}

function liveModel(){


    // Get the selected model from the dropdown
    const selectedModel = document.getElementById('modelSelect3').value;


    // Disable buttons during training
    document.getElementById('predictButton').disabled = true;
    document.getElementById('trainButton').disabled = true;
    document.getElementById('liveButton').disabled = true;

    // Fetch to the '/train' route with the selected model
    fetch('/live', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_choice: selectedModel }),  // Pass the selected model
    })
    .then(response => response.json())
    .then(data => {
        if (!data.success) {
            console.log('Error:',data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    })
    .finally(() => {

        document.getElementById('predictButton').disabled = false;
        document.getElementById('trainButton').disabled = false;
        document.getElementById('liveButton').disabled = false;
    });


}




function disableButtons() {
    const predictButton = document.getElementById('predictButton');
    const trainButton = document.getElementById('trainButton');
    document.getElementById('liveButton').disabled = true;
    
    predictButton.disabled = true;
    trainButton.disabled = true;
}

function enableButtons() {
    const predictButton = document.getElementById('predictButton');
    const trainButton = document.getElementById('trainButton');
    document.getElementById('liveButton').disabled = false;
    predictButton.disabled = false;
    trainButton.disabled = false;
}
