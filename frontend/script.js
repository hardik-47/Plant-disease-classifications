// Show preview when image is selected
document.getElementById("imageInput").addEventListener("change", function() {
    let file = this.files[0];
    if (file) {
        let preview = document.getElementById("preview");
        preview.src = URL.createObjectURL(file);
        preview.style.display = "block";
    }
});

async function uploadImage() {
    let fileInput = document.getElementById("imageInput");
    let file = fileInput.files[0];

    if (!file) {
        alert("Please select an image first!");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    // Change the URL to your backend's deployed link when live
    let response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        body: formData
    });

    let result = await response.json();

    document.getElementById("result").innerHTML =
        `<p><strong>Disease:</strong> ${result.prediction}</p>
         <p><strong>Confidence:</strong> ${result.confidence}%</p>`;
}
