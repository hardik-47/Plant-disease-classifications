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

    document.getElementById("result").innerHTML = "<p>Processing...</p>";

    try {
        let response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Server error: " + response.statusText);
        }
        
        let result = await response.json();

        console.log(result);


        document.getElementById("result").innerHTML =
            `<p><strong>Disease:</strong> ${result.prediction}</p>
             <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(2)}%</p>`;
    } catch (error) {
        document.getElementById("result").innerHTML =
            `<p style="color:red;">Error: ${error.message}</p>`;
    }
}
