const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const modelSelect = document.getElementById("model-select");

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
      }),
    });

    if (!response.ok) {
      throw new Error("Error en la respuesta del servidor");
    }

    const data = await response.json();
    addMessage(`Bot (${selectedModel}): ${data.response}`, "bot");
  } catch (error) {
    console.error(error);
    addMessage("❌ Error al conectar con el servidor", "bot");
  }
}

sendBtn.addEventListener("click", sendMessage);

userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});
