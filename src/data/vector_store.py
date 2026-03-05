"""ChromaDB knowledge base — 3 collections for the learning loop.

Collections:
- confirmed_images: labeled images for Vision Agent few-shot prompting
- interaction_patterns: guidance outcomes for Decision Agent
- correction_patterns: model errors for Vision Agent warnings

See docs/architecture.md section 3 (Knowledge Base Growth).
"""

from pathlib import Path
from typing import Any

import chromadb

from src.config import CHROMA_PATH


class KnowledgeBase:
    """Stores and retrieves learning signals across 3 ChromaDB collections.

    See docs/architecture.md section 3.
    """

    def __init__(self, persist_dir: Path | None = None) -> None:
        path = persist_dir or CHROMA_PATH
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))

        self.confirmed_images = self.client.get_or_create_collection(
            name="confirmed_images",
            metadata={"hnsw:space": "cosine"},
        )
        self.interaction_patterns = self.client.get_or_create_collection(
            name="interaction_patterns",
            metadata={"hnsw:space": "cosine"},
        )
        self.correction_patterns = self.client.get_or_create_collection(
            name="correction_patterns",
            metadata={"hnsw:space": "cosine"},
        )

    # -- Confirmed images (Vision Agent few-shot) ------------------------------

    def add_confirmed_image(
        self,
        image_id: str,
        description: str,
        device_type: str,
        confirmed_fields: str,
        image_path: str = "",
    ) -> None:
        """Store a confirmed image for few-shot retrieval by the Vision Agent."""
        self.confirmed_images.upsert(
            ids=[image_id],
            documents=[description],
            metadatas=[
                {
                    "device_type": device_type,
                    "confirmed_fields": confirmed_fields,
                    "image_path": image_path,
                }
            ],
        )

    def find_similar_images(
        self, description: str, n_results: int = 3, max_distance: float = 0.4
    ) -> list[dict[str, Any]]:
        """Find confirmed images similar to the given description.

        Only returns results within max_distance (cosine distance, 0=identical,
        1=orthogonal). This prevents injecting irrelevant examples into prompts
        when the knowledge base has no genuinely similar entries.
        """
        if self.confirmed_images.count() == 0:
            return []
        results = self.confirmed_images.query(
            query_texts=[description],
            n_results=min(n_results, self.confirmed_images.count()),
        )
        return self._unpack_results(results, max_distance=max_distance)

    # -- Interaction patterns (Decision Agent guidance) -------------------------

    def add_interaction_pattern(
        self,
        interaction_id: str,
        situation_description: str,
        guidance_text: str,
        outcome: str,
        turns_to_success: int,
        device_type: str = "unknown",
        primary_issue: str = "",
        effectiveness_rate: float = 0.0,
    ) -> None:
        """Store a guidance outcome for the Decision Agent."""
        self.interaction_patterns.upsert(
            ids=[interaction_id],
            documents=[situation_description],
            metadatas=[
                {
                    "guidance_text": guidance_text,
                    "outcome": outcome,
                    "turns_to_success": turns_to_success,
                    "device_type": device_type,
                    "primary_issue": primary_issue,
                    "effectiveness_rate": effectiveness_rate,
                }
            ],
        )

    def find_similar_interactions(
        self, description: str, n_results: int = 5, max_distance: float = 0.4
    ) -> list[dict[str, Any]]:
        """Find past interactions similar to the current situation.

        Only returns results within max_distance to avoid polluting the
        Decision Agent's context with irrelevant guidance.
        """
        if self.interaction_patterns.count() == 0:
            return []
        results = self.interaction_patterns.query(
            query_texts=[description],
            n_results=min(n_results, self.interaction_patterns.count()),
        )
        return self._unpack_results(results, max_distance=max_distance)

    # -- Correction patterns (Vision Agent error warnings) ---------------------

    def add_correction_pattern(
        self,
        correction_id: str,
        error_description: str,
        device_type: str,
        field_name: str,
        original_value: str,
        corrected_value: str,
    ) -> None:
        """Store a model error for Vision Agent error warnings."""
        self.correction_patterns.upsert(
            ids=[correction_id],
            documents=[error_description],
            metadatas=[
                {
                    "device_type": device_type,
                    "field_name": field_name,
                    "original_value": original_value,
                    "corrected_value": corrected_value,
                }
            ],
        )

    def find_similar_corrections(
        self, description: str, n_results: int = 3, max_distance: float = 0.4
    ) -> list[dict[str, Any]]:
        """Find past correction patterns similar to the current extraction.

        Only returns results within max_distance to avoid warning about
        errors from unrelated device types.
        """
        if self.correction_patterns.count() == 0:
            return []
        results = self.correction_patterns.query(
            query_texts=[description],
            n_results=min(n_results, self.correction_patterns.count()),
        )
        return self._unpack_results(results, max_distance=max_distance)

    # -- Stats -----------------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        """Return counts across all collections."""
        return {
            "confirmed_images": self.confirmed_images.count(),
            "interaction_patterns": self.interaction_patterns.count(),
            "correction_patterns": self.correction_patterns.count(),
        }

    # -- Internal helpers ------------------------------------------------------

    @staticmethod
    def _unpack_results(
        results: dict, max_distance: float = 1.0
    ) -> list[dict[str, Any]]:
        """Convert ChromaDB query results into a flat list of dicts.

        Filters out results beyond max_distance (cosine distance). This
        prevents returning irrelevant matches when the knowledge base has
        no genuinely similar entries — ChromaDB always returns top-N
        regardless of distance.
        """
        items = []
        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            if distance > max_distance:
                continue
            items.append(
                {
                    "id": doc_id,
                    "document": results["documents"][0][i],
                    "distance": distance,
                    **results["metadatas"][0][i],
                }
            )
        return items
