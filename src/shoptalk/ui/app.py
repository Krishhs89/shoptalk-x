"""
ShopTalk-X Streamlit frontend. Talks to the FastAPI backend over HTTP only
(no direct model loading here) -- matches the "Streamlit frontend hitting
the REST API" architecture from the problem statement, and lets the UI run
on a laptop with zero GPU while the API runs wherever the models live.

Run: streamlit run src/shoptalk/ui/app.py
     (or docker compose up ui, see docker-compose.yml)
"""
import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("SHOPTALK_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("SHOPTALK_API_KEY")
# LLM generation is the dominant cost of a search request and varies wildly
# by hardware -- seconds on GPU, tens of minutes on CPU-only (see
# results/day6_latency_report.md: 16.8 min for one request on a CPU-only
# Mac). A short fixed timeout here would abort perfectly good in-flight
# requests, so this defaults generously and is tunable via env var.
SEARCH_TIMEOUT_S = int(os.environ.get("SHOPTALK_SEARCH_TIMEOUT_S", "1800"))


def _headers():
    return {"X-API-Key": API_KEY} if API_KEY else {}


@st.cache_data(ttl=10)
def check_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return resp.json() if resp.ok else None
    except requests.RequestException:
        return None


@st.cache_data
def load_catalog_preview():
    """Best-effort local read of products.parquet, purely for UI convenience
    (populating the verification tab's item picker). The API call path never
    depends on this -- if it's missing (e.g. UI deployed separately from
    data), the tab just falls back to a plain text input."""
    try:
        from shoptalk.config import load_config

        cfg = load_config()
        return pd.read_parquet(f"{cfg['data']['processed_dir']}/products.parquet")
    except Exception:
        return None


def escape_markdown_dollars(text: str) -> str:
    """st.markdown() treats paired $...$ as inline LaTeX (Streamlit's math
    support) -- fine for the isolated single-$ prices in render_product_hits,
    but this assistant's free-text answers routinely mention two or more
    prices in one message (e.g. "Option A at $20 and Option B at $30"), and
    the whole span between the two $ signs then gets parsed as a LaTeX
    formula instead of shown as text. Escaping $ preserves any other
    markdown the LLM used (bold, lists) while preventing that."""
    return text.replace("$", "\\$")


def render_llm_text(text: str):
    st.markdown(escape_markdown_dollars(text))


def render_product_hits(hits: list):
    if not hits:
        st.caption("No products retrieved.")
        return
    for hit in hits:
        cols = st.columns([3, 1, 1, 1])
        cols[0].markdown(f"**{hit['item_name']}**  \n`{hit['item_id']}`")
        cols[1].caption(hit["category"])
        cols[2].caption(f"${hit['price_usd']:.2f}")
        cols[3].caption(f"score {hit['rerank_score']:.2f}")


def _transcribe_voice_clip(voice_clip) -> str:
    """voice_clip: an st.audio_input UploadedFile (WAV bytes). Lazy-imports
    the voice module so the UI doesn't hard-depend on faster-whisper (a
    stretch/optional extra, requirements/voice.txt) just to load."""
    import tempfile

    from shoptalk.voice.transcribe import transcribe

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(voice_clip.getvalue())
        tmp_path = tmp.name
    try:
        return transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)


def send_feedback(request_id: str, rating: int):
    try:
        requests.post(
            f"{API_BASE_URL}/feedback",
            json={"request_id": request_id, "rating": rating},
            headers=_headers(),
            timeout=5,
        )
        st.toast("Thanks for the feedback!")
    except requests.RequestException as e:
        st.toast(f"Feedback failed to send: {e}")


@st.cache_data(ttl=15)
def fetch_conversation_list(user_name: str):
    """Cached briefly (15s) so the sidebar doesn't hit the API on every
    Streamlit rerun (which happens after every widget interaction), while
    still picking up new conversations soon after they're created."""
    try:
        resp = requests.get(
            f"{API_BASE_URL}/conversations", params={"user_name": user_name}, headers=_headers(), timeout=5
        )
        return resp.json()["conversations"] if resp.ok else []
    except requests.RequestException:
        return []


def fetch_conversation_detail(session_id: str):
    try:
        resp = requests.get(f"{API_BASE_URL}/conversations/{session_id}", headers=_headers(), timeout=5)
        return resp.json() if resp.ok else None
    except requests.RequestException:
        return None


def _resume_conversation(session_id: str):
    detail = fetch_conversation_detail(session_id)
    if detail is None:
        st.toast("Couldn't load that conversation.")
        return
    st.session_state.history = [{"role": t["role"], "content": t["content"]} for t in detail["history"]]
    st.session_state.session_id = detail["session_id"]
    st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
    st.rerun()


def chat_search_tab():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of {"role", "content", "hits"?, "request_id"?}
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "uploader_key" not in st.session_state:
        # st.file_uploader/st.audio_input retain their value across reruns
        # until their `key` changes -- without this, uploading one photo
        # would silently keep re-searching that same stale photo on every
        # later turn, even after the user switched back to typing plain
        # text questions. Bumped after every successful search below.
        st.session_state.uploader_key = 0

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            render_llm_text(turn["content"])
            if turn.get("hits"):
                render_product_hits(turn["hits"])
            if turn.get("request_id"):
                col1, col2 = st.columns([1, 1])
                if col1.button("👍", key=f"up_{turn['request_id']}"):
                    send_feedback(turn["request_id"], 1)
                if col2.button("👎", key=f"down_{turn['request_id']}"):
                    send_feedback(turn["request_id"], -1)

    st.divider()
    col_img, col_voice = st.columns(2)
    with col_img:
        uploaded_image = st.file_uploader(
            "Search by photo (optional)", type=["jpg", "jpeg", "png"],
            key=f"image_uploader_{st.session_state.uploader_key}",
        )
        if uploaded_image:
            st.image(uploaded_image, caption="query photo", width=200)
    with col_voice:
        voice_clip = st.audio_input(
            "Or ask by voice (optional)", key=f"voice_uploader_{st.session_state.uploader_key}"
        )

    query = st.chat_input("Ask ShopTalk-X for a product...")

    if not query and voice_clip is not None:
        with st.spinner("Transcribing..."):
            query = _transcribe_voice_clip(voice_clip)
        if not query:
            st.warning("Couldn't make out any speech in that clip -- try again or type your question.")
            return
        st.caption(f"Heard: {query!r}")

    if not query and uploaded_image is None:
        return
    if not query:
        query = "(searching by uploaded photo)"

    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        render_llm_text(query)

    with st.chat_message("assistant"):
        spinner_msg = (
            "Searching the catalog... (the LLM step can take a while on "
            "CPU-only hardware -- see docs/USAGE_WALKTHROUGH.md)"
        )
        with st.spinner(spinner_msg):
            try:
                user_name = st.session_state.get("user_name", "").strip() or None
                if uploaded_image is not None:
                    files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                    params = {"session_id": st.session_state.session_id} if st.session_state.session_id else {}
                    if user_name:
                        params["user_name"] = user_name
                    resp = requests.post(
                        f"{API_BASE_URL}/search/image", files=files, params=params,
                        headers=_headers(), timeout=SEARCH_TIMEOUT_S,
                    )
                else:
                    resp = requests.post(
                        f"{API_BASE_URL}/search/text",
                        json={
                            "query": query,
                            "session_id": st.session_state.session_id,
                            "user_name": user_name,
                            "stream": False,
                        },
                        headers=_headers(), timeout=SEARCH_TIMEOUT_S,
                    )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                st.error(f"Search failed: {e}")
                return

        st.session_state.session_id = data["session_id"]
        render_llm_text(data["answer"])
        render_product_hits(data["hits"])
        latency = data["latency"]
        st.caption(
            f"stage1 {latency['stage1_ms']:.0f}ms · rerank {latency['rerank_ms']:.0f}ms · "
            f"llm {latency['llm_ms']:.0f}ms · total {latency['total_ms']:.0f}ms"
        )

    st.session_state.history.append(
        {"role": "assistant", "content": data["answer"], "hits": data["hits"], "request_id": data["request_id"]}
    )
    st.session_state.uploader_key += 1  # reset photo/voice widgets so the next turn starts clean
    fetch_conversation_list.clear()  # this turn just changed the sidebar list -- don't wait out the ttl
    st.rerun()


def verification_tab():
    st.subheader("Order verification")
    st.caption("Upload a photo of a received item and confirm it matches what was ordered.")

    catalog = load_catalog_preview()
    if catalog is not None and len(catalog) > 0:
        options = catalog[["item_id", "item_name"]].apply(lambda r: f"{r['item_id']} — {r['item_name'][:60]}", axis=1)
        choice = st.selectbox("Ordered item", options)
        order_item_id = choice.split(" — ")[0]
    else:
        order_item_id = st.text_input("Ordered item_id")

    photo = st.file_uploader("Received item photo", type=["jpg", "jpeg", "png"], key="verify_photo")
    if photo:
        st.image(photo, caption="received item", width=200)

    if st.button("Verify") and order_item_id and photo:
        with st.spinner("Comparing photo to catalog..."):
            try:
                files = {"file": (photo.name, photo.getvalue(), photo.type)}
                resp = requests.post(
                    f"{API_BASE_URL}/verify", files=files, params={"order_item_id": order_item_id},
                    headers=_headers(), timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.RequestException as e:
                st.error(f"Verification failed: {e}")
                return

        badge = {"match": "🟢", "mismatch": "🔴", "suspect": "🟡"}[result["verdict"]]
        st.markdown(f"### {badge} {result['verdict'].upper()}")
        st.caption(f"confidence={result['confidence']:.3f}  threshold={result['threshold']:.3f}")
        if result["verdict"] == "suspect":
            st.warning("Low-confidence result -- routed to human review, not an automatic accusation.")


def main():
    st.set_page_config(page_title="ShopTalk-X", page_icon="🛍️", layout="centered")
    st.title("🛍️ ShopTalk-X")

    health = check_health()
    with st.sidebar:
        st.markdown(f"**API:** `{API_BASE_URL}`")
        if health and health["status"] == "ok":
            st.success("API online")
            st.json(health["models_loaded"])
        elif health:
            st.warning("API online, but the LLM (Ollama) is unreachable -- searches will fail at the answer step")
            st.json(health["models_loaded"])
        else:
            st.error("API unreachable -- start the FastAPI service first")

        st.divider()
        st.session_state.user_name = st.text_input(
            "Your name", value=st.session_state.get("user_name", ""),
            placeholder="e.g. krishna",
            help="Used to save and look up your past conversations -- not an account, just a label.",
        )

        if st.button("New conversation"):
            st.session_state.history = []
            st.session_state.session_id = None
            st.session_state.uploader_key = st.session_state.get("uploader_key", 0) + 1
            st.rerun()

        st.divider()
        st.markdown("**Your past conversations**")
        user_name = st.session_state.user_name.strip()
        if not user_name:
            st.caption("Enter your name above to see and resume past conversations.")
        else:
            conversations = fetch_conversation_list(user_name)
            if not conversations:
                st.caption("No past conversations yet -- start one below.")
            for convo in conversations:
                is_current = convo["session_id"] == st.session_state.get("session_id")
                label = f"{'▶ ' if is_current else ''}{convo['preview']} ({convo['turn_count']} turns)"
                if st.button(label, key=f"resume_{convo['session_id']}", disabled=is_current, use_container_width=True):
                    _resume_conversation(convo["session_id"])

    tab1, tab2 = st.tabs(["Chat search", "Verify order"])
    with tab1:
        chat_search_tab()
    with tab2:
        verification_tab()


if __name__ == "__main__":
    main()
