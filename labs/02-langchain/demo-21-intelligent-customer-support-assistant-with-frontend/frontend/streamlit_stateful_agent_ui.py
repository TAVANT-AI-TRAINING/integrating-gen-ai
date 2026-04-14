import streamlit as st
import requests

# -------------------------------
# Page Title
# -------------------------------
st.title("Smart Assistant Chat")

# -------------------------------
# Backend Configuration
# -------------------------------
BACKEND_BASE_URL = "http://localhost:8000"

# -------------------------------
# Helper function to get new session ID from backend
# -------------------------------
def get_new_session_id():
    """Get a new session ID from the backend."""
    try:
        response = requests.get(f"{BACKEND_BASE_URL}/new-session", timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("session_id")
    except Exception as e:
        st.error(f"Failed to create new session: {e}")
        return None

# -------------------------------
# Initialize session state
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your smart assistant. How can I help you today?"}]
if "session_id" not in st.session_state:
    session_id = get_new_session_id()
    if session_id:
        st.session_state.session_id = session_id
    else:
        st.error("Failed to initialize session. Please refresh the page.")
        st.stop()

# -------------------------------
# New Session Button
# -------------------------------
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("New Session", help="Start a new conversation"):
        # Clear messages and get new session ID
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your smart assistant. How can I help you today?"}]
        session_id = get_new_session_id()
        if session_id:
            st.session_state.session_id = session_id
            st.rerun()
        else:
            st.error("Failed to create new session. Please try again.")

with col2:
    st.write(f"**Session ID:** `{st.session_state.session_id}`")

# -------------------------------
# Display chat history
# -------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------------------
# Input area
# -------------------------------
if prompt := st.chat_input("Ask me about your order, location, or any query..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---------------------------
    # Call the backend FastAPI
    # ---------------------------
    backend_url = f"{BACKEND_BASE_URL}/route_query"
    try:
        with st.spinner("Thinking..."):
            payload = {
                "query": prompt,
                "session_id": st.session_state.session_id
            }
            response = requests.post(backend_url, json=payload, timeout=60)
            response.raise_for_status()

            data = response.json()
            answer = data.get("response", "No response received.")
    except Exception as e:
        answer = f"Error: {e}"

    # ---------------------------
    # Show assistant message
    # ---------------------------
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
