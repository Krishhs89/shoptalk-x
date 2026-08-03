# Model Card: CLIP Image Embeddings + BLIP Captioning

## Intended use
**CLIP** (`open_clip`, `ViT-B-32` / `laion2b_s34b_b79k`, LAION): embeds
catalog images and user-uploaded query photos into a shared vector space for
photo-as-query ANN search (`shoptalk.embeddings.embed_image`,
`shoptalk.retrieval.image_search`). Also used as the pairwise similarity
signal for the verification head (see `verification.md`).

**BLIP** (`Salesforce/blip-image-captioning-base`): generates text captions
from product images two ways —
1. **Offline batch** (`shoptalk.data.caption_images`): captions every
   catalog product's main image, appending the caption to the text
   `document` field so text search picks up visual attributes (pattern,
   material appearance) sellers often omit from bullet points.
2. **Query-time** (`shoptalk.retrieval.image_search`): captions the
   *uploaded query photo* to manufacture a pseudo-text query for reranking
   (see `reranker.md`).

## Training data
Both used off-the-shelf, not fine-tuned. CLIP was trained on LAION-2B;
BLIP's captioning checkpoint on its original pretraining corpora (COCO +
web image-text pairs). Neither has seen ABO data during training — all
ABO-specific adaptation happens downstream (fine-tuned text embeddings,
trained verification head), not in these two models.

## Eval results
- **CLIP retrieval correctness**: validated end-to-end with a real product
  photo as query — recovered the exact source product at cosine similarity
  1.000, with plausible near-neighbors (other phone cases, other shoes)
  filling out the rest of the top-K.
- **BLIP caption quality**: qualitatively reviewed on the smoke-test sample
  — captions were specific and accurate (e.g. "a pair of brown sued shoes",
  "samsung galaxy s9 tempered screen protector"). No formal caption-quality
  metric (BLEU/CIDEr) computed — out of scope without human-written
  reference captions for this catalog.

## Limitations / failure modes
- **BLIP repetition-loop bug (found and fixed)**: greedy decoding on some
  images degenerated into repeating a single token
  ("sailor sailor sailor..."). Fixed with `repetition_penalty=1.5,
  no_repeat_ngram_size=3` in both captioning call sites; verified zero
  degenerate captions afterward on the validation sample. If you see this
  failure mode return on new data, it's a decoding-parameter issue, not a
  model-loading bug.
- **Whole-image embedding, no cropping**: CLIP embeds the entire uploaded
  photo. For cluttered "in-the-wild" photos (product + background clutter,
  multiple objects), embedding quality degrades — this is exactly the gap a
  YOLO detect-and-crop preprocessing step (design doc §3.3a) would close;
  not implemented in this build (see submission notes on scope).
- **Two separate embedding spaces**: CLIP's image-embedding space and BGE's
  text-embedding space are NOT comparable — never compute cosine similarity
  between a CLIP vector and a BGE vector directly. They're stored in
  separate Chroma collections (`shoptalk_products` vs `shoptalk_images`) for
  exactly this reason.
