import streamlit as st
from chatbot import chat_build

st.set_page_config(page_title="Mariam RAG Chatbot", page_icon="💬")

st.title("💬 Mariam's Chatbot")

# Cache the chain so it loads once
@st.cache_resource
def load_chain():
    return chat_build()

chain = load_chain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input box
if prompt := st.chat_input("Ask a question about Mariam..."):

    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(prompt)
            st.markdown(response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})