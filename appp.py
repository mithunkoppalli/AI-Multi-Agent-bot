import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import pdfplumber

from langchain_groq import ChatGroq
from multiagent import app as langgraph_app


# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="AI Multi-Agent Chatbot", layout="wide")


# -----------------------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_id" not in st.session_state:
    st.session_state.chat_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if "selected_chat" not in st.session_state:
    st.session_state.selected_chat = None


# -----------------------------------------------------------
# SESSION CONTEXT BUILDER
# -----------------------------------------------------------
def build_context(messages, limit=6):
    context = []
    for msg in messages[-limit:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        context.append(f"{role}: {msg['content']}")
    return "\n".join(context)


# -----------------------------------------------------------
# CHAT PERSISTENCE (SAVE / LOAD)
# -----------------------------------------------------------
CHAT_DIR = Path("chats")
CHAT_DIR.mkdir(exist_ok=True)

def save_chat(chat_id, messages):
    with open(CHAT_DIR / f"{chat_id}.json", "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def load_chat(chat_id):
    file = CHAT_DIR / f"{chat_id}.json"
    if file.exists():
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def list_chats():
    return sorted([f.stem for f in CHAT_DIR.glob("*.json")], reverse=True)


# -----------------------------------------------------------
# DIRECT LLM FOR PDF SUMMARY
# -----------------------------------------------------------
direct_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key="YOUR_GROQ_API_KEY",
    temperature=0
)

def summarize_large_pdf(pdf_text):
    CHUNK_SIZE = 600
    chunks = [pdf_text[i:i + CHUNK_SIZE] for i in range(0, len(pdf_text), CHUNK_SIZE)]

    summaries = []
    for chunk in chunks:
        res = direct_llm.invoke([
            {"role": "user", "content": "Summarize this part briefly:\n" + chunk}
        ])
        summaries.append(res.content)

    while len(summaries) > 3:
        new_batch = []
        for i in range(0, len(summaries), 3):
            res = direct_llm.invoke([
                {"role": "user", "content": "Combine these summaries:\n" + "\n".join(summaries[i:i+3])}
            ])
            new_batch.append(res.content)
        summaries = new_batch

    final = direct_llm.invoke([
        {"role": "user", "content": "Give a final summary:\n" + "\n".join(summaries)}
    ])

    return final.content


# -----------------------------------------------------------
# LAYOUT
# -----------------------------------------------------------
left, right = st.columns([1, 2])


# -----------------------------------------------------------
# LEFT SIDE — PREVIOUS CHATS + HISTORY + PDF
# -----------------------------------------------------------
with left:
    st.subheader("Previous Chats")

    chat_ids = list_chats()

    if chat_ids:
        st.session_state.selected_chat = st.selectbox(
            "Select a chat",
            chat_ids,
            index=0 if st.session_state.selected_chat is None else chat_ids.index(st.session_state.selected_chat)
            if st.session_state.selected_chat in chat_ids else 0
        )

        if st.button("Load Chat"):
            st.session_state.chat_id = st.session_state.selected_chat
            st.session_state.messages = load_chat(st.session_state.selected_chat)
    else:
        st.info("No previous chats available")

    st.write("---")

    if st.button("New Chat"):
        st.session_state.messages = []
        st.session_state.chat_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    st.write(f"Chat ID: {st.session_state.chat_id}")
    st.write("---")

    st.subheader("Chat History")

    if st.session_state.messages:
        for msg in st.session_state.messages:
            preview = msg["content"][:120] + "..." if len(msg["content"]) > 120 else msg["content"]
            st.markdown(
                f"<div style='background:#f0f2f6;padding:6px;border-radius:6px;margin-bottom:4px;'>"
                f"{preview}</div>",
                unsafe_allow_html=True
            )
            st.caption(msg["timestamp"])
    else:
        st.info("No messages yet")

    st.write("---")

    st.subheader("Upload PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_pdf:
        pdf_text = ""
        with pdfplumber.open(uploaded_pdf) as pdf:
            for page in pdf.pages:
                pdf_text += page.extract_text() or ""

        if st.button("Summarize PDF"):
            summary = summarize_large_pdf(pdf_text)
            st.session_state.messages.append({
                "role": "assistant",
                "content": summary,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            save_chat(st.session_state.chat_id, st.session_state.messages)


# -----------------------------------------------------------
# RIGHT SIDE — INPUT + RESPONSE
# -----------------------------------------------------------
with right:
    st.title("AI Multi-Agent Chatbot")

    user_input = st.chat_input("Type your question...")

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

        context = build_context(st.session_state.messages)

        result = langgraph_app.invoke({
            "user_input": f"Conversation so far:\n{context}\n\nCurrent question:\n{user_input}",
            "plan": [],
            "research": [],
            "final_answer": ""
        })

        urls = []
        for item in result.get("research", []):
            if "Source:" in item:
                urls.append(item.split("Source:")[1].split("\n")[0].strip())

        url_text = "\n\nSources:\n" + "\n".join(urls) if urls else ""

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["final_answer"] + url_text,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

        save_chat(st.session_state.chat_id, st.session_state.messages)

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.markdown(st.session_state.messages[-1]["content"], unsafe_allow_html=True)
        st.caption(st.session_state.messages[-1]["timestamp"])
