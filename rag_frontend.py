import base64
import re
from concurrent.futures import ThreadPoolExecutor
import uuid
import hashlib
from stt import speech_to_text
from tts import txt_to_speech

import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from memory_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata
)


USER_ID = "default_user"  # long-term memory isi user ke naam se store hoti hai


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def clean_for_speech(text, max_chars=1500):
    """Markdown hata kar bolne layak plain text banata hai."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # code blocks
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)       # [text](url) -> text
    text = re.sub(r"[*_`#>|~]+", " ", text)                    # markdown symbols
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


# ---------------------- Streaming TTS helpers ----------------------
@st.cache_resource
def get_tts_pool():
    return ThreadPoolExecutor(max_workers=2)


_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")


def split_sentences(buffer):
    """(poore ho chuke sentences, adhoora bacha hua hissa) lautata hai."""
    parts = _SENT_SPLIT.split(buffer)
    return parts[:-1], parts[-1]


# Browser (parent window) mein ek chhota audio queue player, taaki sentences
# ek ke baad ek bajein, overlap na hon.
_TTS_BOOT = (
    "window.__ttsQ=[];window.__ttsCur=null;"
    "window.__ttsNext=function(){if(window.__ttsCur)return;"
    "var s=window.__ttsQ.shift();if(!s)return;"
    "var a=new Audio(s);window.__ttsCur=a;"
    "var d=function(){window.__ttsCur=null;window.__ttsNext();};"
    "a.onended=d;a.onerror=d;a.play().catch(d);};"
)

_TTS_QUEUE_HTML = """<script>
(function () {
  var w = window.parent;
  if (!w.__ttsNext) { w.eval("__BOOT__"); }
  w.__ttsQ.push("data:audio/wav;base64,__B64__");
  w.__ttsNext();
})();
</script>"""

_TTS_STOP_HTML = """<script>
(function () {
  var w = window.parent;
  w.__ttsQ = [];
  if (w.__ttsCur) { try { w.__ttsCur.pause(); } catch (e) {} w.__ttsCur = null; }
})();
</script>"""


def queue_audio(wav_bytes):
    b64 = base64.b64encode(wav_bytes).decode()
    html = _TTS_QUEUE_HTML.replace("__BOOT__", _TTS_BOOT).replace("__B64__", b64)
    components.html(html, height=0)


def stop_audio():
    components.html(_TTS_STOP_HTML, height=0)


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
_saved_meta = thread_document_metadata(thread_key)
if _saved_meta and not thread_docs:
    thread_docs[_saved_meta.get("filename", "document.pdf")] = _saved_meta
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("Pdf chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.toggle("🔊 Voice (TTS)", key="tts_enabled", value=False)

# ---- Voice input (STT): mic se bolkar poochho ----
STT_LANGUAGES = {"Hindi": "hi-IN", "English": "en-IN"}
stt_lang_name = st.sidebar.selectbox("🎤 Select language ", list(STT_LANGUAGES), index=0)
mic_audio = (
    st.sidebar.audio_input("🎤 ask by speaking(max 30 sec)")
    if hasattr(st, "audio_input")
    else None
)

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread_id in threads:
        if st.sidebar.button(str(thread_id), key=f"side-thread-{thread_id}"):
            selected_thread = thread_id

# ============================ Main Layout ========================
st.title("Chatbot Creted by Ritesh")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

# Mic se aayi nayi recording ko text banakar chat input ki tarah use karo.
# Hash se pata chalta hai ki yahi recording pehle process ho chuki hai ya nahi
# (widget har rerun par purani recording pakde rehta hai).
if mic_audio is not None:
    audio_bytes = mic_audio.getvalue()
    audio_hash = hashlib.md5(audio_bytes).hexdigest()
    if audio_hash != st.session_state.get("last_audio_hash"):
        st.session_state["last_audio_hash"] = audio_hash
        try:
            with st.spinner("Awaaz sun raha hoon..."):
                spoken_text = speech_to_text(
                    audio_bytes, language_code=STT_LANGUAGES[stt_lang_name]
                )
            if spoken_text:
                user_input = spoken_text
            else:
                st.warning("Kuch sunai nahi diya, dobara boliye.")
        except Exception as e:
            st.error(f"STT fail hua: {e}")

if user_input:
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key, "user_id": USER_ID},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, meta in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # sirf chat_node ka jawab dikhao (remember_node ka JSON nahi)
                elif(
                    isinstance(message_chunk, AIMessage)
                    and meta.get("langgraph_node") == "chat_node"
                    and isinstance(message_chunk.content, str)
                ):
                    yield message_chunk.content

        tts_on = bool(st.session_state.get("tts_enabled"))
        text_box = None
        ai_message = ""

        for chunk in ai_only_stream():
            if text_box is None:
                text_box = st.empty()
            ai_message += chunk
            text_box.markdown(ai_message + "▌")

        if text_box is not None:
            text_box.markdown(ai_message)

        # TTS ON ho to poora jawab ek saath, ek hi baar API ko bhejo
        if tts_on and ai_message.strip():
            try:
                with st.spinner("🔊 Awaaz ban rahi hai..."):
                    audio_bytes = txt_to_speech(
                        clean_for_speech(ai_message, max_chars=5000)
                    )
                st.audio(audio_bytes, format="audio/wav", autoplay=True)
            except Exception as e:
                st.warning(f"TTS fail hua: {e}")


        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content:
            temp_messages.append({"role": "assistant", "content": msg.content})
        # ToolMessage aur khaali tool-call wale AI messages skip
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
    