const dragArea = document.getElementById("drag-area");
const input = document.getElementById("image-input");
const selectedFileText = document.getElementById("selected-file");
const form = document.getElementById("upload-form");
const loader = document.getElementById("loader");

dragArea.addEventListener("click", (e) => {
    if (e.target !== input) {
        input.click();
    }
});

input.addEventListener("click", (e) => {
    e.stopPropagation();
});

input.addEventListener("change", () => {
    if (input.files.length > 0) {
        selectedFileText.textContent = "Selected: " + input.files[0].name;
    } else {
        selectedFileText.textContent = "No image selected";
    }
});

dragArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dragArea.classList.add("active");
});

dragArea.addEventListener("dragleave", () => {
    dragArea.classList.remove("active");
});

dragArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dragArea.classList.remove("active");

    const file = e.dataTransfer.files[0];

    if (file) {
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        input.files = dataTransfer.files;
        selectedFileText.textContent = "Selected: " + file.name;
    }
});

form.addEventListener("submit", () => {
    loader.classList.remove("hidden");
});