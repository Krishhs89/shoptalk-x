"""Maps this catalog's product categories/names onto the 80 classes a
COCO-pretrained YOLO checkpoint actually knows how to detect.

The catalog (~10k Amazon-style products across ~50 fine-grained categories
like CELLULAR_PHONE_CASE, FINERING, HARDWARE_HANDLE) and COCO's 80 generic
household/street object classes barely overlap -- most catalog items have no
COCO equivalent at all. `resolve_coco_class` is deliberately conservative:
no match returns None rather than guessing, so `count.py` can report
"unsupported" instead of silently fabricating a count for a class the
detector was never trained to recognize.
"""
from typing import Optional

# Direct category -> COCO class, for categories with an unambiguous match.
CATEGORY_TO_COCO_CLASS = {
    "CHAIR": "chair",
    "STOOL_SEATING": "chair",
    "SOFA": "couch",
    "TABLE": "dining table",
    "HANDBAG": "handbag",
}

# item_name substring -> COCO class, checked when the category itself
# doesn't map (e.g. GROCERY covers dozens of COCO-relevant foods and
# non-food items alike, so it's matched by name, not as a whole category).
# Ordered longest-phrase-first so "wine glass" is tried before "glass"
# would ever be (it isn't a COCO class on its own, but the principle holds
# for any future two-word class added here).
KEYWORD_TO_COCO_CLASS = {
    "wine glass": "wine glass",
    "teddy bear": "teddy bear",
    "hair drier": "hair drier",
    "hair dryer": "hair drier",
    "cell phone": "cell phone",
    "sports ball": "sports ball",
    "tennis racket": "tennis racket",
    "baseball bat": "baseball bat",
    "baseball glove": "baseball glove",
    "potted plant": "potted plant",
    "toothbrush": "toothbrush",
    "backpack": "backpack",
    "umbrella": "umbrella",
    "suitcase": "suitcase",
    "scissors": "scissors",
    "banana": "banana",
    "sandwich": "sandwich",
    "broccoli": "broccoli",
    "carrot": "carrot",
    "hot dog": "hot dog",
    "donut": "donut",
    "cake": "cake",
    "bottle": "bottle",
    "bowl": "bowl",
    "vase": "vase",
    "clock": "clock",
    "mug": "cup",
    "cup": "cup",
    "book": "book",
    "orange": "orange",
    "apple": "apple",
    "pizza": "pizza",
    "kite": "kite",
    "skateboard": "skateboard",
    "surfboard": "surfboard",
    "frisbee": "frisbee",
    "toaster": "toaster",
    "microwave": "microwave",
    "tie": "tie",
}


def resolve_coco_class(item_name: str, category: str) -> Optional[str]:
    """Returns the COCO class name this product should map to for counting,
    or None if the pretrained detector has no relevant class -- callers must
    treat None as "can't validate this product's quantity", not as a
    reason to fall back to a guess."""
    if category in CATEGORY_TO_COCO_CLASS:
        return CATEGORY_TO_COCO_CLASS[category]

    name_lower = item_name.lower()
    for keyword, coco_class in KEYWORD_TO_COCO_CLASS.items():
        if keyword in name_lower:
            return coco_class

    return None
