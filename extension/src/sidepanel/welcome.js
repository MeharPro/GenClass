const state = document.getElementById("micState");
async function show() {
  try {
    const p = await navigator.permissions.query({ name: "microphone" });
    state.textContent = p.state === "granted" ? "Microphone allowed ✓" : p.state === "denied" ? "Blocked: allow it in the site settings for this extension" : "";
  } catch { /* ignore */ }
}
document.getElementById("grant").addEventListener("click", async () => {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ audio: true });
    s.getTracks().forEach((t) => t.stop());
  } catch (e) {
    state.textContent = `Not allowed: ${e.message}`;
  }
  show();
});
show();
