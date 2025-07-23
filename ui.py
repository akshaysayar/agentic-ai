import streamlit as st
import streamlit.components.v1 as components

import requests

FASTAPI_RAG_ENDPOINT = "http://localhost:8000/rag"

st.set_page_config(page_title="Lawyer's Chatbot", page_icon="💬")
st.title("🦅 Il(l)eagle Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hello! How can I assist you today?", "sources": []}
    ]

if "suggested_prompts" not in st.session_state:
    st.session_state["suggested_prompts"] = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown("#### Sources:")
            for source in msg["sources"]:
                # Supports link if present, else plain text
                if isinstance(source, dict) and "title" in source and "url" in source:
                    st.markdown(f"- [{source['title']}]({source['url']})")
                else:
                    st.markdown(f"- {source}")

# Function to handle user input or suggested prompt
def process_user_input(prompt_text):
    st.session_state.messages.append({"role": "user", "content": prompt_text})

    chat_history_for_api = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]

    try:
        with st.spinner("Thinking..."):
            response = requests.post(
                FASTAPI_RAG_ENDPOINT,
                json={"query": prompt_text, "chat_history": chat_history_for_api},
                timeout=10,
            )
            response.raise_for_status()
            rag_output = response.json()

        bot_response = rag_output.get("answer", "⚠️ No response from RAG.")
        suggested_prompts = rag_output.get("suggested_prompts", [])
        sources = rag_output.get("sources", [])

    except Exception as e:
        bot_response = f"⚠️ Error communicating with backend: {e}"
        suggested_prompts = []
        sources = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_response,
        "sources": sources
    })
    st.session_state["suggested_prompts"] = suggested_prompts

    with st.chat_message("assistant"):
        st.markdown(bot_response)

        if sources:
            st.markdown("# Sources:")
            for source in sources:
                st.markdown(f"- {source}")


    # # Display assistant message immediately
    # with st.chat_message("assistant"):
    #     st.markdown(bot_response)
    #     if sources:
    #         st.markdown("#### Sources:")
    #         for source in sources:
    #             if isinstance(source, dict) and "title" in source and "url" in source:
    #                 st.markdown(f"- [{source['title']}]({source['url']})")
    #             else:
    #                 st.markdown(f"- {source}")

# Suggested prompt buttons
if st.session_state["suggested_prompts"]:
    st.markdown("---")
    st.markdown("**What would you like to do next?**")
    cols = st.columns(len(st.session_state["suggested_prompts"]))
    for i, prompt_text in enumerate(st.session_state["suggested_prompts"]):
        with cols[i]:
            if st.button(prompt_text, key=f"suggested_{i}"):
                process_user_input(prompt_text)
                st.rerun()

# Main chat input
if user_input := st.chat_input("Type your message here..."):
    process_user_input(user_input)
    st.rerun()
