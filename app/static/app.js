const configSection = document.getElementById("config-section");
const chatSection = document.getElementById("chat-section");

const modelSelect = document.getElementById("model-select");
const apiKeyContainer = document.getElementById("api-key-container");
const apiKeyInput = document.getElementById("api-key-input");
const startChatBtn = document.getElementById("start-chat-btn");

const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const changeModelBtn = document.getElementById("change-model-btn");

let currentApiKey = "";

// Obtener el nombre de la clave en localStorage
function getStorageKey(model) {
  return `api_key_${model}`;
}

// --- CONFIGURACIÓN ---

// Mostrar/Ocultar input y Cargar Key guardada
modelSelect.addEventListener("change", () => {
  const model = modelSelect.value;
  
  if (model === "ollama") {
    apiKeyContainer.classList.add("hidden");
  } else {
    apiKeyContainer.classList.remove("hidden");
    // Busco si ya hay una key para este modelo 
    const savedKey = localStorage.getItem(getStorageKey(model));
    apiKeyInput.value = savedKey || ""; 
  }
});

// Botón "Comenzar a Chatear"
startChatBtn.addEventListener("click", () => {
  const model = modelSelect.value;
  const key = apiKeyInput.value.trim();

  // Validación
  if (model !== "ollama" && !key) {
    alert("Por favor, ingresa una API Key para usar este modelo.");
    return;
  }

  // Guardo la key
  currentApiKey = key;

  if (model !== "ollama") {
    localStorage.setItem(getStorageKey(model), key);
  }

  // Cambia de pantalla
  configSection.classList.add("hidden");
  chatSection.classList.remove("hidden");
  
  if(chatBox.innerHTML === "") {
      addMessage(`Sistema: Has iniciado chat con ${model}`, "bot");
  }
});

// Botón "Cambiar de Chat Model"
changeModelBtn.addEventListener("click", () => {
  chatSection.classList.add("hidden");
  configSection.classList.remove("hidden");
  
  currentApiKey = "";
  chatBox.innerHTML = "";
  // Dispara el evento para que verifique si hay key guardada
  modelSelect.dispatchEvent(new Event('change'));
});

// --- CHAT ---

function addMessage(text, className) {
  const message = document.createElement("div");
  message.className = `message ${className}`;
  message.textContent = text;
  chatBox.appendChild(message);
  chatBox.scrollTop = chatBox.scrollHeight;
}

async function sendMessage() {
  const text = userInput.value.trim();
  const selectedModel = modelSelect.value;

  if (!text) return;

  addMessage(`Tú: ${text}`, "user");
  userInput.value = "";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: text,
        model_provider: selectedModel,
        api_key: currentApiKey
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Error en la respuesta del servidor");
    }

    const data = await response.json();
    addMessage(`Bot: ${data.response}`, "bot");
    
  } catch (error) {
    console.error(error);
    addMessage(`❌ Error: ${error.message}`, "bot");
  }
}

sendBtn.addEventListener("click", sendMessage);

userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});

// Inicialización
document.addEventListener("DOMContentLoaded", () => {
    modelSelect.dispatchEvent(new Event('change'));
});