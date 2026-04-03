import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="MCP Agent", layout="centered")
st.title(" MCP Agent")
st.caption(" Weather · Web Search · Filesystem")

#Sidebar: available tools 
with st.sidebar:
    st.header("Available Tools")
    try:
        res = requests.get(f"{API_URL}/tools", timeout=3)
        if res.status_code == 200:
            tools = res.json()["tools"]
            for tool in tools:
                with st.expander(f" {tool['name']}"):
                    st.write(tool["description"])
        else:
            st.warning("Could not load tools.")
    except Exception:
        st.error("API not reachable. Is FastAPI running?")

    st.divider()
    if st.button(" Clear Chat"):
        st.session_state.messages = []
        st.rerun()

#Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session"

#chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#chat input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "message": prompt,
                        "thread_id": st.session_state.thread_id
                    },
                    timeout=60
                )
                if res.status_code == 200:
                    reply = res.json()["response"]
                else:
                    reply = f"Error {res.status_code}: {res.json().get('detail', 'Unknown error')}"
            except Exception as e:
                reply = f"Could not reach API: {e}"

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})