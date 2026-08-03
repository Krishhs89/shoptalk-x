"""
Day-3 image-as-query search: upload a photo -> CLIP-embed -> ANN over the
image collection -> rerank.

The stage-2 cross-encoder (rerank.py) is text-only, and an uploaded photo has
no natural-language query to pair it with -- so we caption the QUERY IMAGE
itself with BLIP to get a pseudo-text query, then reuse the exact same
cross-encoder reranker from Day 2 against each candidate's (caption-augmented)
document text. This keeps "ANN -> rerank" as a two-stage pipeline for photo
queries too, per the execution plan, without a separate image reranker model.

Usage:
  python -m shoptalk.retrieval.image_search --image path/to/photo.jpg
"""
import argparse
import sys

import chromadb
import open_clip
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from shoptalk.config import load_config
from shoptalk.retrieval.rerank import rerank

_clip_model = None
_clip_preprocess = None
_clip_device = None
_blip_processor = None
_blip_model = None
_image_collection = None


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_clip(clcfg: dict):
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is None:
        _clip_device = _resolve_device(clcfg["device"])
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            clcfg["model_name"], pretrained=clcfg["pretrained"], device=_clip_device
        )
        _clip_model.eval()
    return _clip_model, _clip_preprocess, _clip_device


def _get_blip(ccfg: dict):
    global _blip_processor, _blip_model
    if _blip_model is None:
        device = _resolve_device(ccfg["device"])
        _blip_processor = BlipProcessor.from_pretrained(ccfg["model"])
        _blip_model = BlipForConditionalGeneration.from_pretrained(ccfg["model"]).to(device)
        _blip_model.eval()
    return _blip_processor, _blip_model


def _get_image_collection(clcfg: dict):
    global _image_collection
    if _image_collection is None:
        client = chromadb.PersistentClient(path=clcfg["chroma_dir"])
        _image_collection = client.get_collection(clcfg["collection_name"])
    return _image_collection


def caption_query_image(image_path: str, cfg: dict) -> str:
    ccfg = cfg["captioning"]
    processor, model = _get_blip(ccfg)
    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=ccfg["max_new_tokens"],
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
        )
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


def clip_ann_search(image_path: str, top_k: int, cfg: dict) -> list:
    clcfg = cfg["clip"]
    model, preprocess, device = _get_clip(clcfg)
    collection = _get_image_collection(clcfg)

    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    results = collection.query(
        query_embeddings=[embedding.cpu().numpy()[0].tolist()],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "item_id": results["ids"][0][i],
                "score": 1 - results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i],
            }
        )
    return hits


def image_search(image_path: str, stage1_k: int = None, top_k: int = None, cfg: dict = None) -> dict:
    """Returns {"pseudo_query": str, "hits": [...]}."""
    cfg = cfg or load_config()
    rcfg = cfg["retrieval"]
    stage1_k = stage1_k or rcfg["image_stage1_k"]
    top_k = top_k or rcfg["top_k"]

    candidates = clip_ann_search(image_path, stage1_k, cfg)
    pseudo_query = caption_query_image(image_path, cfg)
    hits = rerank(pseudo_query, candidates, top_k=top_k, cfg=cfg)
    return {"pseudo_query": pseudo_query, "hits": hits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stage1-k", type=int, default=None)
    args = parser.parse_args()

    result = image_search(args.image, stage1_k=args.stage1_k, top_k=args.top_k)
    print(f"query image: {args.image}")
    print(f"BLIP pseudo-query (used for rerank): {result['pseudo_query']!r}\n")
    for rank, hit in enumerate(result["hits"], 1):
        md = hit["metadata"]
        print(
            f"{rank:2d}. [rerank={hit['rerank_score']:.3f} clip={hit['stage1_score']:.3f}] "
            f"{md['item_name'][:65]!r} ({md['category']}, ${md['price_usd']:.2f}) -- id={hit['item_id']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
