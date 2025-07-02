import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

# --- Session State Setup ---
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Functions ---
def start_new_chat(initial_message: str):
    res = requests.post(f"{API_URL}/chats/", json={"initial_question": user_input})
    data = res.json()
    chat_id = data["chat_id"]
    assistant_response = data["response"]["content"]

    st.session_state.chat_id = chat_id
    st.session_state.messages = [
    {"role": "user", "content": user_input},
    {"role": "assistant", "content": assistant_response}
]

def fetch_chat_history(chat_id: str):
    
    res = requests.get(f"{API_URL}/chats/{chat_id}/history")
    if res.status_code == 200:
        st.session_state.chat_history = res.json()

def send_message(content: str):
    if not st.session_state.chat_id:
        start_new_chat(content)
    else:
        res = requests.post(
            f"{API_URL}/chats/{st.session_state.chat_id}/message",
            json={"content": content}
        )
        if res.status_code == 200:
            fetch_chat_history(st.session_state.chat_id)

def delete_chat(chat_id: str):
    requests.delete(f"{API_URL}/chats/{chat_id}")
    if chat_id == st.session_state.chat_id:
        st.session_state.chat_id = None
        st.session_state.chat_history = []

# --- Sidebar: Chat History ---
st.sidebar.title("📚 Chat History")
chats = requests.get(f"{API_URL}/chats/list").json()
for chat in chats:
    if st.sidebar.button(chat["title"], key=chat["chat_id"]):
        st.session_state.chat_id = chat["chat_id"]
        fetch_chat_history(chat["chat_id"])

    st.sidebar.button("🗑️", key=f"del-{chat['chat_id']}", on_click=delete_chat, args=(chat["chat_id"],))

if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_id = None
    st.session_state.chat_history = []

# --- Main Chat Interface ---
st.title("📖 Sophia: Nigerian Education Chatbot")

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Ask about Nigerian universities, JAMB, or courses...")
if user_input:
    send_message(user_input)
