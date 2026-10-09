import asyncio
import json
import os
import time
from pathlib import Path

import inngest
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    # Path(...).name strips any directory parts, so "../../x.pdf" can't escape uploads/
    file_path = uploads_dir / Path(file.name).name
    file_path.write_bytes(file.getbuffer())
    return file_path


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


async def send_rag_query_event(question: str, top_k: int) -> str:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return result[0]


def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url, params={"includeOutput": "true"}, timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_run_output(
    event_id: str,
    timeout_s: float = 300.0,
    poll_interval_s: float = 1.5,
) -> dict:
    start = time.time()
    last_run: dict = {}

    while time.time() - start < timeout_s:
        runs = fetch_runs(event_id)

        if runs:
            run = runs[0]
            last_run = run
            status = run.get("status", "")

            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Inngest function run {status}: {run}")

            if status == "Completed":
                output = run.get("output")

                if isinstance(output, str):
                    try:
                        output = json.loads(output)
                    except json.JSONDecodeError:
                        output = None

                if isinstance(output, dict):
                    return output
                # Status says Completed but there's no output yet
                # (ended_at is still null), so keep polling.

        time.sleep(poll_interval_s)

    st.warning("Timed out waiting for the run's output. Last response from Inngest:")
    st.json(last_run)
    raise TimeoutError("Timed out waiting for Inngest to produce an output.")


# ---------------------------------------------------------------- Ingest UI
st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    # Streamlit reruns this script on every interaction (including the question
    # form below), and the uploader keeps returning the same file. Without this
    # guard the PDF would be re-ingested on every rerun.
    file_key = f"{uploaded.name}:{uploaded.size}"

    if st.session_state.get("ingested_key") != file_key:
        try:
            with st.spinner("Uploading and triggering ingestion..."):
                path = save_uploaded_pdf(uploaded)
                asyncio.run(send_rag_ingest_event(path))
            st.session_state["ingested_key"] = file_key
            st.success(f"Triggered ingestion for: {path.name}")
            st.caption("You can upload another PDF if you like.")
        except Exception as exc:
            st.error(f"Could not trigger ingestion: {exc}")
    else:
        st.caption(f"{uploaded.name} was already sent for ingestion.")

# ----------------------------------------------------------------- Query UI
st.divider()
st.title("Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input(
        "How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1
    )
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    try:
        with st.spinner("Sending event and generating answer..."):
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
            output = wait_for_run_output(event_id)
    except Exception as exc:
        st.error(f"Something went wrong while answering: {exc}")
    else:
        answer = output.get("answer", "")
        sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")
