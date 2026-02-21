import os
import requests
import streamlit as st

# CONFIGURACIÓN

API_URL = os.getenv("API_URL", "http://localhost:8000")
CHAT_ENDPOINT = f"{API_URL}/api/chat"

MODELS = {
    "groq": "Groq",
    "gemini": "Google Gemini",
    "ollama": "Ollama (local)",
}

st.set_page_config(
    page_title="Asistente Virtual | Hermanos Jota",
    page_icon="🛋️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ESTILOS

st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    border-right: 1px solid #1e293b;
}

/* Mensajes del chat */
[data-testid="stChatMessage"] {
    background-color: #1e293b;
    border-radius: 12px;
    margin-bottom: 8px;
}

/* Input de chat */
[data-testid="stChatInput"] textarea {
    background-color: #1e293b !important;
}

/* Links */
a {
    color: #3b82f6 !important;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# ESTADO DE SESIÓN

if "chat_started" not in st.session_state:
    st.session_state.chat_started = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = "groq"
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# SIDEBAR

with st.sidebar:
    st.markdown("<h1 style='font-size:2.5rem; margin-bottom:0;'>🛋️</h1>", unsafe_allow_html=True)

    st.title("Asistente Virtual")
    st.caption("**Hermanos Jota**")

    st.markdown("""
    ChatBot con arquitectura **RAG**, aumentado con documentos internos
    sobre la mueblería **Hermanos Jota** y su sitio web.
    """)

    st.markdown(
        "🌐 [Web de la mueblería](https://hermanos-jota-flame.vercel.app/)  \n"
        "📂 [Repo del proyecto](https://github.com/marianoInsa/ChatBot-RAG)"
    )

    st.markdown("Creado por **Mariano Insaurralde**")

    st.divider()

    # CONFIGURACIÓN DEL MODELO
    st.subheader("⚙️ Configuración")

    selected_model = st.selectbox(
        "Modelo de lenguaje",
        options=list(MODELS.keys()),
        format_func=lambda k: MODELS[k],
        index=list(MODELS.keys()).index(st.session_state.model),
        key="model_select",
    )

    api_key_input = ""
    if selected_model != "ollama":
        api_key_input = st.text_input(
            "API Key",
            type="password",
            placeholder="Ej: gsk_... o AIza...",
            value=st.session_state.api_key if st.session_state.model == selected_model else "",
            help="Requerida para Groq y Google Gemini. Se usa solo en esta sesión.",
        )

    if st.button("Iniciar chat", type="primary", use_container_width=True):
        if selected_model != "ollama" and not api_key_input.strip():
            st.error("Ingresá una API Key para usar este modelo.")
        else:
            st.session_state.model = selected_model
            st.session_state.api_key = api_key_input.strip()
            st.session_state.chat_started = True
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": f"¡Hola! Soy el asistente de **Hermanos Jota**. "
                               f"Estoy usando **{MODELS[selected_model]}**. ¿En qué puedo ayudarte?",
                }
            ]
            st.rerun()

    if st.session_state.chat_started:
        st.divider()
        if st.button("🔄 Cambiar modelo", use_container_width=True):
            st.session_state.chat_started = False
            st.session_state.messages = []
            st.rerun()

# ÁREA PRINCIPAL DEL CHAT

if not st.session_state.chat_started:
    # Pantalla de bienvenida
    st.markdown("<br>" * 4, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<h2 style='text-align:center; color:#3b82f6;'>🛋️ Asistente Virtual</h2>"
            "<p style='text-align:center; color:#94a3b8;'>"
            "Seleccioná un modelo en el panel izquierdo y presioná <strong>Iniciar chat</strong> para comenzar."
            "</p>",
            unsafe_allow_html=True,
        )
else:
    # Historial de mensajes
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input del usuario
    if prompt := st.chat_input("Escribí tu pregunta..."):
        # Mostrar mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Llamada a la API
        with st.chat_message("assistant"):
            with st.spinner("Consultando..."):
                try:
                    response = requests.post(
                        CHAT_ENDPOINT,
                        json={
                            "question": prompt,
                            "model_provider": st.session_state.model,
                            "api_key": st.session_state.api_key,
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    answer = response.json().get("response", "Sin respuesta.")
                except requests.exceptions.ConnectionError:
                    answer = "❌ No se pudo conectar con la API. Verificá que el servidor esté corriendo."
                except requests.exceptions.Timeout:
                    answer = "❌ La solicitud tardó demasiado. Intentá de nuevo."
                except requests.exceptions.HTTPError as e:
                    detail = e.response.json().get("detail", str(e)) if e.response else str(e)
                    answer = f"❌ Error del servidor: {detail}"
                except Exception as e:
                    answer = f"❌ Error inesperado: {e}"

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
