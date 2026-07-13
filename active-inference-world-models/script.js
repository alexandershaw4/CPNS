const video = document.getElementById("world-model-video");

document.querySelectorAll(".chapter").forEach((button) => {
  button.addEventListener("click", () => {
    if (!video) return;
    const time = Number(button.dataset.time || 0);
    video.currentTime = time;
    video.play().catch(() => {});
    video.scrollIntoView({ behavior: "smooth", block: "center" });
  });
});

const copyButton = document.getElementById("copy-citation");
const citation = document.getElementById("citation-text");
const toast = document.getElementById("copy-toast");

function showToast(message) {
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 2200);
}

if (copyButton && citation) {
  copyButton.addEventListener("click", async () => {
    const text = citation.innerText.trim();
    try {
      await navigator.clipboard.writeText(text);
      showToast("Citation copied.");
    } catch (error) {
      const area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
      showToast("Citation copied.");
    }
  });
}
