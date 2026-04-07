function startDetection() {
    const messages = [
        "Analyzing image structure...",
        "Detecting copy-move regions...",
        "Detecting splicing artifacts..."
    ];

    let i = 0;
    const status = document.getElementById("status");

    const interval = setInterval(() => {
        status.innerText = messages[i];
        i++;

        if (i === messages.length) {
            clearInterval(interval);
            document.getElementById("loader").style.display = "none";
            document.getElementById("result").style.display = "block";
        }
    }, 1200);
}
