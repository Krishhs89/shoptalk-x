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


def chat_search_tab():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of {"role", "content", "hits"?, "request_id"?}
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
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
        uploaded_image = st.file_uploader("Search by photo (optional)", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            st.image(uploaded_image, caption="query photo", width=200)
    with col_voice:
        voice_clip = st.audio_input("Or ask by voice (optional)")

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
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching the catalog..."):
            try:
                if uploaded_image is not None:
                    files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                    params = {"session_id": st.session_state.session_id} if st.session_state.session_id else {}
                    resp = requests.post(
                        f"{API_BASE_URL}/search/image", files=files, params=params,
                        headers=_headers(), timeout=60,
                    )
                else:
                    resp = requests.post(
                        f"{API_BASE_URL}/search/text",
                        json={"query": query, "session_id": st.session_state.session_id, "stream": False},
                        headers=_headers(), timeout=60,
                    )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                st.error(f"Search failed: {e}")
                return

        st.session_state.session_id = data["session_id"]
        st.markdown(data["answer"])
        render_product_hits(data["hits"])
        latency = data["latency"]
        st.caption(
            f"stage1 {latency['stage1_ms']:.0f}ms · rerank {latency['rerank_ms']:.0f}ms · "
            f"llm {latency['llm_ms']:.0f}ms · total {latency['total_ms']:.0f}ms"
        )

    st.session_state.history.append(
        {"role": "assistant", "content": data["answer"], "hits": data["hits"], "request_id": data["request_id"]}
    )
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
        if health:
            st.success("API online")
            st.json(health["models_loaded"])
        else:
            st.error("API unreachable -- start the FastAPI service first")
        if st.button("New conversation"):
            st.session_state.history = []
            st.session_state.session_id = None
            st.rerun()

    tab1, tab2 = st.tabs(["Chat search", "Verify order"])
    with tab1:
        chat_search_tab()
    with tab2:
        verification_tab()


if __name__ == "__main__":
    main()
