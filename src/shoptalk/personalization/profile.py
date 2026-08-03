"""User profile vectors: a recency-decayed average of a user's interacted
product embeddings (design doc §3.2c)."""
import numpy as np
import pandas as pd


def build_user_profile(user_id: str, interactions_df: pd.DataFrame, item_embeddings: dict, decay: float):
    """item_embeddings: {item_id: np.ndarray}. Returns a normalized profile
    vector, or None if the user has no interactions with embedded items."""
    user_rows = interactions_df[interactions_df["user_id"] == user_id].sort_values("timestamp")
    embeddings, weights = [], []
    n = len(user_rows)
    for i, (_, row) in enumerate(user_rows.iterrows()):
        emb = item_embeddings.get(row["item_id"])
        if emb is None:
            continue
        embeddings.append(emb)
        weights.append(decay ** (n - 1 - i))  # most recent interaction gets weight decay^0 = 1

    if not embeddings:
        return None

    weights = np.array(weights)
    profile = np.average(np.stack(embeddings), axis=0, weights=weights)
    norm = np.linalg.norm(profile)
    return profile / norm if norm > 0 else profile
