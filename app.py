import os
import sys
import tempfile
from pathlib import Path
from google import genai
from google.genai import types
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="NoteExtract.ai",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="collapsed",
)

COLORS = ["#00FFB2", "#7C6FFF", "#FF6B9D", "#FFB347", "#00D4FF", "#FF6B6B", "#A8FF78"]

STEPS = [
    ("ingest",     "Ingesting input"),
    ("transcribe", "Transcribing with Whisper"),
    ("chunk",      "Chunking transcript"),
    ("embed",      "Embedding chunks"),
    ("llm",        "Extracting notes with Gemini"),
]

_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#0A0A0F;color:#E0E0F0}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding-top:2rem;padding-bottom:2rem;max-width:900px}
.stTabs [data-baseweb="tab-list"]{background:#12121A;border-radius:10px 10px 0 0;border:1px solid #1E1E2E;border-bottom:none}
.stTabs [data-baseweb="tab"]{background:transparent;color:#555;font-family:'Space Mono',monospace;font-size:12px;font-weight:700;letter-spacing:1px;padding:14px 28px}
.stTabs [aria-selected="true"]{background:#00FFB212!important;color:#00FFB2!important;border-bottom:2px solid #00FFB2!important}
.stTabs [data-baseweb="tab-panel"]{background:#12121A;border:1px solid #1E1E2E;border-top:none;border-radius:0 0 10px 10px;padding:1.5rem}
.stTextInput input{background:#0A0A0F!important;border:1px solid #1E1E2E!important;border-radius:8px!important;color:#C0C0D8!important}
.stButton>button{background:linear-gradient(135deg,#00FFB2CC,#00FFB2)!important;color:#0A0A0F!important;border:none!important;border-radius:8px!important;font-family:'Space Mono',monospace!important;font-weight:700!important;font-size:13px!important;letter-spacing:1px!important;width:100%}
.stButton>button:disabled{opacity:0.3!important}
.status-box{background:#12121A;border:1px solid #1E1E2E;border-radius:10px;padding:1.2rem 1.5rem;margin:1rem 0}
.step{display:flex;align-items:center;gap:10px;padding:5px 0;color:#555;font-family:'Space Mono',monospace;font-size:12px}
.step.done{color:#00FFB2}.step.active{color:#fff}.step.err{color:#FF6B6B}
.result-header{background:#12121A;border:1px solid #1E1E2E;border-radius:10px;padding:1.5rem;margin-bottom:1rem}
.result-title{font-size:22px;font-weight:600;color:#fff;margin-bottom:0.5rem}
.result-summary{color:#8888AA;line-height:1.7;font-size:14px}
.sec-label{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#444;margin:1.5rem 0 0.75rem}
.note-card{border-left:3px solid;padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:10px}
.note-ts{font-family:'Space Mono',monospace;font-size:11px;margin-bottom:4px}
.note-heading{font-weight:600;font-size:14px;color:#fff;margin-bottom:4px}
.note-content{color:#A0A0C0;font-size:13px;line-height:1.65}
.action-item{display:flex;align-items:flex-start;gap:12px;padding:10px 14px;background:#12121A;border:1px solid #1E1E2E;border-radius:8px;margin-bottom:8px;font-size:14px;color:#D0D0E8}
.action-bullet{color:#00FFB2;font-family:'Space Mono',monospace;font-size:14px;flex-shrink:0}
.concept-tag{display:inline-block;padding:3px 12px;border-radius:4px;font-family:'Space Mono',monospace;font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin:3px}
.stDownloadButton>button{background:transparent!important;border:1px solid #1E1E2E!important;border-radius:6px!important;color:#666!important;font-family:'Space Mono',monospace!important;font-size:11px!important}
.stDownloadButton>button:hover{border-color:#00FFB244!important;color:#00FFB2!important}
.chat-wrap{margin-top:0.5rem}
.chat-msg{display:flex;gap:10px;margin-bottom:14px;align-items:flex-start}
.chat-msg.user{flex-direction:row-reverse}
.chat-avatar{width:30px;height:30px;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;font-family:'Space Mono',monospace;font-weight:700}
.chat-avatar.user{background:#00FFB215;border:1px solid #00FFB244;color:#00FFB2}
.chat-avatar.bot{background:#7C6FFF15;border:1px solid #7C6FFF44;color:#7C6FFF}
.chat-bubble{padding:10px 14px;border-radius:10px;font-size:13px;line-height:1.65;max-width:80%}
.chat-bubble.user{background:#00FFB210;border:1px solid #00FFB233;color:#E0E0F0;border-radius:10px 2px 10px 10px}
.chat-bubble.bot{background:#12121A;border:1px solid #1E1E2E;color:#A0A0C0;border-radius:2px 10px 10px 10px}
</style>"""


def init_state():
    """Initialise all session_state keys on first run."""
    defaults = {
        "input_type":    None,
        "input_data":    None,
        "ready":         False,
        "running":       False,
        "result":        None,
        "store":         None,   # VectorStore kept in memory for Q&A
        "chat_history":  [],     # list of {"role": "user"|"assistant", "text": str}
        "error":         None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def status_html(done: list, active: str = None, err: str = None) -> str:
    h = '<div class="status-box">'
    for k, label in STEPS:
        if k in done:
            h += f'<div class="step done">&#10003;&nbsp; {label}</div>'
        elif err and k == active:
            h += f'<div class="step err">&#10007;&nbsp; {label} &mdash; {err}</div>'
        elif k == active:
            h += f'<div class="step active">&#8635;&nbsp; {label}...</div>'
        else:
            h += f'<div class="step">&middot;&nbsp; {label}</div>'
    return h + '</div>'


def run_pipeline(itype: str, idata, wmodel: str, sbox):
    """Execute the full pipeline, updating sbox with live status."""
    done = []
    tmp  = tempfile.mkdtemp()

    # from any working directory
    project_root = str(Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    def upd(active=None, err=None):
        sbox.markdown(status_html(done, active, err), unsafe_allow_html=True)

    try:
        segs  = []
        title = ""
        upd("ingest")

        if itype == "video":
            from ingestion.local_video import extract_audio
            apath, meta = extract_audio(idata, output_dir=tmp)
            title = meta.title
            done.append("ingest")
            upd("transcribe")
            from transcription.whisper_engine import transcribe
            segs = transcribe(apath, model_size=wmodel)
            done.append("transcribe")

        elif itype == "youtube":
            from ingestion.youtube import download_audio
            apath, title = download_audio(idata, output_dir=tmp)
            done.append("ingest")
            upd("transcribe")
            from transcription.whisper_engine import transcribe
            segs = transcribe(apath, model_size=wmodel)
            done.append("transcribe")

        elif itype == "file":
            fname, fbytes = idata
            tp = Path(tmp) / fname
            tp.write_bytes(fbytes)
            from ingestion.file_loader import load_file
            segs  = load_file(str(tp))
            title = Path(fname).stem
            done.append("ingest")
            done.append("transcribe")

        if not segs:
            raise ValueError("No transcript segments found in this input.")

        upd("chunk")
        from processing.chunker import chunk_segments
        chunks = chunk_segments(segs, window_words=250, overlap_words=50)
        done.append("chunk")

        upd("embed")
        from processing.vector_store import VectorStore
        store = VectorStore()
        store.add_chunks(chunks)
        done.append("embed")

        upd("llm")
        from llm.gemini_extractor import extract_notes
        result = extract_notes(store, video_title=title)
        done.append("llm")
        upd()

        return result, store, None

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"\n[Pipeline ERROR]\n{tb}")
        upd(active=None, err=str(e))
        return None, None, str(e)


def answer_question(question: str, store, chat_history: list) -> str:

    # Retrieve relevant chunks from this session's store
    chunks = store.query(question, top_k=6)
    if not chunks:
        return "I couldn't find relevant information in the extracted notes to answer that."

    context = "\n\n".join(
        f"{c.get('timestamp', '')} {c['text']}".strip()
        for c in chunks
    )

    # Build conversation history string
    history_str = ""
    if chat_history:
        lines = []
        for msg in chat_history[-6:]:   # last 3 turns (6 messages)
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['text']}")
        history_str = "\n".join(lines) + "\n\n"

    prompt = f"""You are a helpful assistant that answers questions ONLY based on the provided video transcript excerpts below.
If the answer is not found in the excerpts, say "I don't have enough information in the extracted notes to answer that."
Do not use any outside knowledge. Be concise and direct.

Transcript excerpts:
{context}

{history_str}User: {question}
Assistant:"""

    api_key = os.getenv("GEMINI_API_KEY")
    client  = genai.Client(api_key=api_key)
    model   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()


def render_chat():
    """Render the Q&A chat section below the extracted notes."""
    st.markdown('<div class="sec-label">Ask About These Notes</div>', unsafe_allow_html=True)
    st.caption("Questions are answered strictly from the extracted content — no outside knowledge.")

    # Render existing messages
    if st.session_state.chat_history:
        st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history:
            role  = msg["role"]
            text  = msg["text"]
            avatar = "you" if role == "user" else "ai"
            st.markdown(
                f'<div class="chat-msg {role}">'
                f'<div class="chat-avatar {role if role == "user" else "bot"}">'
                f'{"Y" if role == "user" else "AI"}</div>'
                f'<div class="chat-bubble {"user" if role == "user" else "bot"}">{text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # Input box
    question = st.chat_input("Ask a question about the video...")
    if question:
        st.session_state.chat_history.append({"role": "user", "text": question})
        with st.spinner("Thinking..."):
            answer = answer_question(
                question,
                st.session_state.store,
                st.session_state.chat_history[:-1],  # history before this question
            )
        st.session_state.chat_history.append({"role": "assistant", "text": answer})
        st.rerun()



def render_results(result):
    import json
    from datetime import datetime

    # Build Markdown in memory
    def _slug(t): return "".join(c if c.isalnum() or c in "-_" else "_" for c in t)[:60]
    slug = _slug(result.title)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_lines = [
        f"# {result.title}",
        f"\n> *Generated by NoteExtract.ai — {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        "## Summary\n", result.summary,
        "\n## Key Concepts\n",
        " · ".join(f"`{c}`" for c in result.key_concepts),
        "\n## Notes\n",
    ]
    for note in result.notes:
        ts_part = f"**{note.timestamp}** — " if note.timestamp else ""
        md_lines.append(f"### {ts_part}{note.heading}\n")
        md_lines.append(f"{note.content}\n")
    md_lines += ["\n## Action Items\n", *[f"- [ ] {item}" for item in result.action_items]]
    md_content = "\n".join(md_lines)

    json_content = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)

    st.markdown(
        f'<div class="result-header">'
        f'<div class="result-title">{result.title}</div>'
        f'<div class="result-summary">{result.summary}</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec-label">Key Concepts</div>', unsafe_allow_html=True)
    tags = "".join(
        f'<span class="concept-tag" style="color:{COLORS[i%len(COLORS)]};'
        f'background:{COLORS[i%len(COLORS)]}11;border:1px solid {COLORS[i%len(COLORS)]}33">{c}</span>'
        for i, c in enumerate(result.key_concepts)
    )
    st.markdown(tags, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">Notes</div>', unsafe_allow_html=True)
    for i, note in enumerate(result.notes):
        color = COLORS[i % len(COLORS)]
        ts = (
            f'<div class="note-ts" style="color:{color}">{note.timestamp}</div>'
            if note.timestamp else ""
        )
        st.markdown(
            f'<div class="note-card" style="border-color:{color};background:{color}08">'
            f'{ts}<div class="note-heading">{note.heading}</div>'
            f'<div class="note-content">{note.content}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sec-label">Action Items</div>', unsafe_allow_html=True)
    for item in result.action_items:
        st.markdown(
            f'<div class="action-item">'
            f'<span class="action-bullet">&#9658;</span><span>{item}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sec-label">Export</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download Markdown", data=md_content,
                           file_name=f"{slug}_{ts}.md", mime="text/markdown",
                           use_container_width=True)
    with c2:
        st.download_button("Download JSON", data=json_content,
                           file_name=f"{slug}_{ts}.json", mime="application/json",
                           use_container_width=True)


def main():
    init_state()

    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div style="display:flex;align-items:center;gap:14px;margin-bottom:2rem;'
        'padding-bottom:1.5rem;border-bottom:1px solid #1E1E2E">'
        '<div style="width:42px;height:42px;background:#00FFB210;border:1px solid '
        '#00FFB244;border-radius:10px;font-size:22px;display:flex;align-items:center;'
        'justify-content:center">&#9672;</div>'
        '<div><div style="font-family:\'Space Mono\',monospace;font-size:18px;'
        'font-weight:700;color:#fff">NoteExtract'
        '<span style="color:#00FFB2">.ai</span></div>'
        '<div style="font-family:\'Space Mono\',monospace;font-size:10px;color:#444;'
        'letter-spacing:2px;text-transform:uppercase">Video &#8594; Structured Notes'
        '</div></div></div>',
        unsafe_allow_html=True,
    )

    if not os.getenv("GEMINI_API_KEY"):
        st.error(
            "**GEMINI_API_KEY not set.**\n\n"
            "Get a free key at [aistudio.google.com](https://aistudio.google.com)\n\n"
            "Then add to your `.env` file: `GEMINI_API_KEY=your_key_here`"
        )
        st.stop()

    with st.sidebar:
        st.markdown("### Settings")
        wmodel = st.selectbox(
            "Whisper Model",
            ["tiny", "base", "small", "medium", "large"],
            index=1,
            help="Larger = more accurate but slower.",
        )
        st.caption(f"Selected: **{wmodel}**\nRuns fully offline.")
        st.markdown("---")
        st.markdown("**Free tier limits**")
        st.caption("Gemini: 1,500 req/day · 15/min\nWhisper: unlimited (local)\nEmbeddings: unlimited (local)")


    t1, t2, t3 = st.tabs(["VIDEO FILE", "YOUTUBE URL", "TRANSCRIPT FILE"])

    with t1:
        st.markdown("##### Upload a video file")
        st.caption("Supported: .mp4  .mkv  .mov  .avi  .webm  .m4v  .flv")
        uv = st.file_uploader(
            "video",
            type=["mp4", "mkv", "mov", "avi", "webm", "m4v", "flv"],
            label_visibility="collapsed",
            key="vu",
        )
        if uv is not None:
            # Save to a real temp file so FFmpeg can read it by path
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uv.name).suffix)
            tf.write(uv.read())
            tf.flush()
            tf.close()
            st.session_state.input_type = "video"
            st.session_state.input_data = tf.name
            st.session_state.ready      = True
            st.success(f"Ready: {uv.name}  ({uv.size // 1024:,} KB)")

    with t2:
        st.markdown("##### Paste a YouTube URL")
        url = st.text_input(
            "url",
            placeholder="https://www.youtube.com/watch?v=...",
            label_visibility="collapsed",
            key="yt_url",
        )
        if url and url.startswith("http"):
            st.session_state.input_type = "youtube"
            st.session_state.input_data = url
            st.session_state.ready      = True

    with t3:
        st.markdown("##### Upload a transcript file")
        st.caption("Supported: .txt (optional [MM:SS] timestamps)  .srt  .vtt")
        uf = st.file_uploader(
            "transcript",
            type=["txt", "srt", "vtt"],
            label_visibility="collapsed",
            key="fu",
        )
        if uf is not None:
            st.session_state.input_type = "file"
            st.session_state.input_data = (uf.name, uf.read())
            st.session_state.ready      = True
            st.success(f"Ready: {uf.name}  ({uf.size:,} bytes)")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "EXTRACT NOTES",
        disabled=not st.session_state.ready,
        use_container_width=True,
    ):
        st.session_state.result       = None
        st.session_state.store        = None
        st.session_state.chat_history = []     # reset chat on new extraction
        st.session_state.error        = None
        st.session_state.running      = True

        sbox = st.empty()
        result, store, err = run_pipeline(
            st.session_state.input_type,
            st.session_state.input_data,
            wmodel,
            sbox,
        )

        st.session_state.running = False
        st.session_state.result  = result
        st.session_state.store   = store
        st.session_state.error   = err

    if st.session_state.result is not None:
        st.markdown("---")
        render_results(st.session_state.result)
        st.markdown("---")
        render_chat()


if __name__ == "__main__":
    main()