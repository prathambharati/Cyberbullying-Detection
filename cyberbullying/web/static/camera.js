// Used by the face login and account pages. Turns on the webcam, takes three
// quick pictures and posts them with the rest of the form.
(function () {
  const FRAMES = 3;
  const GAP_MS = 250;

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  function snapshot(video) {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
  }

  function explain(error) {
    if (error.name === "NotAllowedError") {
      return "The browser blocked the camera. Allow it from the address bar and try again.";
    }
    if (error.name === "NotFoundError") {
      return "No camera was found.";
    }
    return "Something went wrong: " + error.message;
  }

  document.querySelectorAll("[data-camera-form]").forEach((form) => {
    const video = form.querySelector("video");
    const status = form.querySelector(".status");
    const button = form.querySelector("button[type=submit]");
    let stream = null;

    async function startCamera() {
      if (stream) return;
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
      await sleep(700); // let the camera settle its exposure
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!form.reportValidity()) return;
      button.disabled = true;
      try {
        status.textContent = "Turning on the camera...";
        await startCamera();
        status.textContent = "Hold still...";
        const data = new FormData(form);
        for (let i = 0; i < FRAMES; i++) {
          data.append("frames", await snapshot(video), `frame${i}.jpg`);
          await sleep(GAP_MS);
        }
        status.textContent = "Checking...";
        const response = await fetch(form.dataset.endpoint, {
          method: "POST",
          body: data,
          headers: { "X-CSRF-Token": data.get("csrf_token") },
        });
        const isJson = (response.headers.get("content-type") || "").includes("application/json");
        if (!isJson) {
          status.textContent = `The server said ${response.status}. Reload the page and try again.`;
          return;
        }
        const reply = await response.json();
        if (reply.ok && reply.redirect) {
          window.location.assign(reply.redirect);
          return;
        }
        status.textContent = reply.message;
      } catch (error) {
        status.textContent = explain(error);
      } finally {
        button.disabled = false;
      }
    });
  });
})();
