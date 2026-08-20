// Elements
const video = document.getElementById("webcam");
const statusText = document.getElementById("status-text");
const drowsyAlert = document.getElementById("drowsy-alert");

// Start webcam stream
async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
        });
        video.srcObject = stream;
        video.play();
    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Unable to access your webcam.");
    }
}

// Convert video frame to Base64
function captureFrame() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return canvas.toDataURL("image/jpeg");
}

// Send frame to backend
async function sendFrame() {
    if (video.videoWidth === 0) return; // Wait for video

    const frame = captureFrame();

    try {
        const response = await fetch("http://127.0.0.1:5000/detect", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ image: frame }),
        });

        const data = await response.json();
        updateUI(data);

    } catch (err) {
        console.error("Error sending frame:", err);
    }
}

// UI Updates
function updateUI(data) {
    const { pitch, yaw, roll, drowsy } = data;

    statusText.innerHTML = `
        <b>Pitch:</b> ${pitch}°<br>
        <b>Yaw:</b> ${yaw}°<br>
        <b>Roll:</b> ${roll}°
    `;

    if (drowsy === true) {
        drowsyAlert.style.display = "block";
        drowsyAlert.innerText = "⚠️ Drowsiness Detected!";
    } else {
        drowsyAlert.style.display = "none";
    }

    // Optional: Move cursor using yaw axis
    // const screenX = window.innerWidth / 2 + yaw * 15;
    // const screenY = window.innerHeight / 2 + pitch * 15;
    // window.scrollTo(screenX, screenY);
}

// Start periodic data sending every 200ms
setInterval(sendFrame, 200);

// Init
startWebcam();
