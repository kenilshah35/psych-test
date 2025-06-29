import os
import tempfile
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import tiktoken
import openai
import time

# Configure page
st.set_page_config(page_title="Therapy Chatbot", page_icon="💬")

# Constants
MODEL_NAME = "llama-3.3-70b-versatile"
API_KEY = os.getenv("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"
MAX_TRANSCRIPT_TOKENS = 9000
TOKEN_LIMIT_PER_MINUTE = 12000

# Ensure API key is set
if not API_KEY:
    st.error("Please set the GROQ_API_KEY environment variable.")
    st.stop()

client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Initialize session state for transcripts and chat
if "transcripts_texts" not in st.session_state:
    st.session_state.transcripts_texts = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "token_count" not in st.session_state:
    st.session_state.token_count = 0
if "token_reset_time" not in st.session_state:
    st.session_state.token_reset_time = time.time()

# Initialize tokenizer for counting tokens
try:
    tokenizer = tiktoken.encoding_for_model(MODEL_NAME)
except Exception:
    tokenizer = tiktoken.get_encoding("p50k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def enforce_token_limit(texts: list, max_tokens: int) -> list:
    combined = " ".join(texts)
    total = count_tokens(combined)
    while total > max_tokens and texts:
        texts.pop(0)
        combined = " ".join(texts)
        total = count_tokens(combined)
    return texts

# Added helper to count tokens in messages
def count_message_tokens(messages: list) -> int:
    return sum(count_tokens(msg["content"]) for msg in messages)

def extract_text(uploaded_file) -> str:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".txt":
        raw = uploaded_file.read()
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin1")
    elif ext == ".pdf":
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    elif ext == ".docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.read())
            tmp.flush()
            doc = Document(tmp.name)
        os.unlink(tmp.name)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return ""

def call_groqcloud_api(messages: list) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages
    )
    return response.choices[0].message.content

# Sidebar: File upload for transcripts
st.sidebar.title("Upload Transcripts")
uploaded_files = st.sidebar.file_uploader(
    "Upload txt, pdf, or docx files",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True
)
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.processed_files:
            text = extract_text(uploaded_file)
            if text:
                st.session_state.transcripts_texts.append(text)
            st.session_state.processed_files.add(uploaded_file.name)
    st.session_state.transcripts_texts = enforce_token_limit(
        st.session_state.transcripts_texts, MAX_TRANSCRIPT_TOKENS
    )

# Token usage tracking: reset every minute and display
t_current = time.time()
if t_current - st.session_state.token_reset_time >= 60:
    st.session_state.token_count = 0
    st.session_state.token_reset_time = t_current

# Calculate transcript token usage
transcript_tokens = 0
if st.session_state.transcripts_texts:
    transcript_str = "\n\n".join(st.session_state.transcripts_texts)
    transcript_tokens = count_tokens(transcript_str)

st.sidebar.title("Token Usage")
st.sidebar.metric("Tokens this minute", st.session_state.token_count, f"Limit: {TOKEN_LIMIT_PER_MINUTE}")
st.sidebar.metric("Transcript tokens", transcript_tokens, f"Max: {MAX_TRANSCRIPT_TOKENS}")

# Main: Chat interface
st.title("Therapy Session Chatbot")

# Display existing chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Prepare initial system messages
system_messages = [
    {"role": "system", "content": "You are a helpful therapy assistant. You have background transcripts that you should use to inform your responses. Do not reference or reveal the transcripts directly to the user."}
]
if st.session_state.transcripts_texts:
    transcript_str = "\n\n".join(st.session_state.transcripts_texts)
    system_messages.append({"role": "system", "content": f"Background transcripts (for internal use only):\n{transcript_str}"})

# Handle user input and generate response
if user_input := st.chat_input("Your message"):
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    # Display the user message immediately
    st.chat_message("user").write(user_input)
    # Remove oldest chats if context token usage exceeds threshold
    while count_message_tokens(system_messages + st.session_state.chat_history) > 11500 and st.session_state.chat_history:
        st.session_state.chat_history.pop(0)
    # Prepare messages and update token usage counter
    full_messages = system_messages + st.session_state.chat_history
    num_tokens = count_message_tokens(full_messages)
    st.session_state.token_count += num_tokens
    # Generate response
    with st.spinner("Generating response..."):
        assistant_response = call_groqcloud_api(full_messages)
    # Append assistant message and display
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    st.chat_message("assistant").write(assistant_response)
