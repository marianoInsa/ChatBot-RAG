// --- ELEMENTOS ---
const tabRegister = document.getElementById("tab-register");
const tabClients = document.getElementById("tab-clients");
const tabUpload = document.getElementById("tab-upload");
const tabChat = document.getElementById("tab-chat");

const panelRegister = document.getElementById("panel-register");
const panelClients = document.getElementById("panel-clients");
const panelUpload = document.getElementById("panel-upload");
const panelChat = document.getElementById("panel-chat");

const registerBtn = document.getElementById("register-btn");
const registerResult = document.getElementById("register-result");

const refreshClientsBtn = document.getElementById("refresh-clients-btn");
const clientsList = document.getElementById("clients-list");

const uploadClientId = document.getElementById("upload-client-id");
const pdfFiles = document.getElementById("pdf-files");
const urlsInput = document.getElementById("urls-input");
const uploadBtn = document.getElementById("upload-btn");
const uploadResult = document.getElementById("upload-result");

const chatClientId = document.getElementById("chat-client-id");
const modelSelect = document.getElementById("model-select");
const apiKeyContainer = document.getElementById("api-key-container");
const apiKeyInput = document.getElementById("api-key-input");
const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

let currentApiKey = "";

function getStorageKey(model) {
  return `api_key_${model}`;
}

// --- TABS ---
function showPanel(panelId) {
  [panelRegister, panelClients, panelUpload, panelChat].forEach((p) => p.classList.remove("active"));
  [tabRegister, tabClients, tabUpload, tabChat].forEach((t) => t.classList.remove("active"));
  document.getElementById(`panel-${panelId}`).classList.add("active");
  document.getElementById(`tab-${panelId}`).classList.add("active");
}

tabRegister.addEventListener("click", () => showPanel("register"));
tabClients.addEventListener("click", () => {
  showPanel("clients");
  loadClients();
});
tabUpload.addEventListener("click", () => showPanel("upload"));
tabChat.addEventListener("click", () => showPanel("chat"));

// --- REGISTRO ---
registerBtn.addEventListener("click", async () => {
  registerResult.classList.add("hidden");
  try {
    const res = await fetch("/api/clients/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Error");
    registerResult.classList.remove("hidden");
    registerResult.innerHTML = `
      <p><strong>Cliente registrado</strong></p>
      <p><code>client_id: ${data.client_id}</code></p>
      <p>Copia este ID para cargar documentos y chatear.</p>
    `;
    uploadClientId.value = data.client_id;
    chatClientId.value = data.client_id;
  } catch (e) {
    registerResult.classList.remove("hidden");
    registerResult.innerHTML = `<p class="error">Error: ${e.message}</p>`;
  }
});

// --- LISTA DE CLIENTES ---
async function loadClients() {
  clientsList.innerHTML = "<p>Cargando...</p>";
  try {
    const res = await fetch("/api/admin/clients");
    const clients = await res.json();
    if (!res.ok) throw new Error(clients.detail || "Error");
    if (!clients.length) {
      clientsList.innerHTML = "<p>No hay clientes registrados.</p>";
      return;
    }
    clientsList.innerHTML = clients
      .map(
        (c) => `
      <div class="client-card">
        <p><strong>${c.client_id}</strong></p>
        <p>Docs: ${c.stats?.documents_count ?? 0} | Chunks: ${c.stats?.chunks_count ?? 0}</p>
        <div class="client-actions">
          <button class="link-btn use-client-btn" data-id="${c.client_id}">Usar</button>
          <button class="link-btn delete-client-btn" data-id="${c.client_id}">Eliminar</button>
        </div>
      </div>
    `
      )
      .join("");
    clientsList.querySelectorAll(".use-client-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        uploadClientId.value = btn.dataset.id;
        chatClientId.value = btn.dataset.id;
        showPanel("upload");
      });
    });
    clientsList.querySelectorAll(".delete-client-btn").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (!confirm("¿Eliminar este cliente?")) return;
        try {
          await fetch(`/api/admin/clients/${btn.dataset.id}`, { method: "DELETE" });
          loadClients();
        } catch (e) {
          alert("Error: " + e.message);
        }
      });
    });
  } catch (e) {
    clientsList.innerHTML = `<p class="error">Error: ${e.message}</p>`;
  }
}

refreshClientsBtn.addEventListener("click", loadClients);

// --- CARGA DE DOCUMENTOS ---
uploadBtn.addEventListener("click", async () => {
  const clientId = uploadClientId.value.trim();
  if (!clientId) {
    alert("Ingresa un client_id");
    return;
  }
  uploadResult.classList.add("hidden");
  const formData = new FormData();
  const files = pdfFiles.files;
  if (files?.length) {
    for (let i = 0; i < files.length; i++) formData.append("files", files[i]);
  }
  const urlLines = urlsInput.value.split("\n").filter((u) => u.trim());
  formData.append("urls", JSON.stringify(urlLines));
  try {
    const res = await fetch(`/api/clients/${clientId}/documents/upload`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    uploadResult.classList.remove("hidden");
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    uploadResult.innerHTML = `<p class="success">${data.message}</p>`;
    if (data.errors?.length) uploadResult.innerHTML += `<p class="warn">Avisos: ${data.errors.join(", ")}</p>`;
  } catch (e) {
    uploadResult.classList.remove("hidden");
    uploadResult.innerHTML = `<p class="error">Error: ${e.message}</p>`;
  }
});

// --- CHAT ---
modelSelect.addEventListener("change", () => {
  const model = modelSelect.value;
  if (model === "ollama") {
    apiKeyContainer.classList.add("hidden");
  } else {
    apiKeyContainer.classList.remove("hidden");
    apiKeyInput.value = localStorage.getItem(getStorageKey(model)) || "";
  }
});

function addMessage(text, className) {
  const msg = document.createElement("div");
  msg.className = `message ${className}`;
  msg.textContent = text;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});

async function sendMessage() {
  const text = userInput.value.trim();
  const clientId = chatClientId.value.trim();
  const model = modelSelect.value;
  if (!text) return;
  if (!clientId) {
    alert("Ingresa un client_id");
    return;
  }
  if (model !== "ollama" && !apiKeyInput.value.trim()) {
    alert("Ingresa la API Key para este modelo");
    return;
  }
  currentApiKey = apiKeyInput.value.trim();
  if (model !== "ollama") localStorage.setItem(getStorageKey(model), currentApiKey);

  addMessage(`Tú: ${text}`, "user");
  userInput.value = "";

  try {
    const res = await fetch(`/api/clients/${clientId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: text,
        model_provider: model,
        api_key: currentApiKey,
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    addMessage(`Bot: ${data.response}`, "bot");
  } catch (e) {
    addMessage(`Error: ${e.message}`, "bot");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  modelSelect.dispatchEvent(new Event("change"));
});
