"""Build civic_shout_emails v1.

Run manually:
    python -m entities.civic_shout_emails.create_cache_v1

Pulls all vectorized Civic Shout emails from Mongo, fetches their dense
vectors from Pinecone, computes fresh sparse vectors via Pinecone Inference,
and stores the result via cache.put(1, df).
"""

from __future__ import annotations

import asyncio
import logging

import polars as pl
from pinecone import Pinecone

from entities.civic_shout_emails.cache import cache
from libs.clients.mongo import FindOptions
from libs.clients.pinecone import PINECONE_SPARSE_ENGLISH_V0, SparseEmbedding
from libs.container import Container

logger = logging.getLogger(__name__)

CIVIC_SHOUT_ORG_ID = "civic_shout"
PINECONE_DENSE_INDEX = "production-content-routing-dense"
PINECONE_DENSE_NAMESPACE = "civic_shout_vv1"
SPARSE_BATCH_SIZE = 96


async def _fetch_dense_vectors(
    api_key: str,
    email_ids: list[int],
) -> dict[int, list[float]]:
    """Fetch dense vectors from Pinecone by email_id."""
    pc = Pinecone(api_key=api_key)
    index_info = pc.describe_index(PINECONE_DENSE_INDEX)
    index = pc.Index(PINECONE_DENSE_INDEX, host=index_info.host)

    dense_map: dict[int, list[float]] = {}
    batch_size = 1000
    for i in range(0, len(email_ids), batch_size):
        batch = email_ids[i : i + batch_size]
        str_ids = [str(eid) for eid in batch]
        result = index.fetch(ids=str_ids, namespace=PINECONE_DENSE_NAMESPACE)
        vectors = result.get("vectors", {})
        for str_id, vec_data in vectors.items():
            eid = int(str_id)
            dense_map[eid] = vec_data["values"]
        logger.info(
            "Fetched dense vectors %d/%d", min(i + batch_size, len(email_ids)), len(email_ids)
        )

    return dense_map


async def _compute_sparse_vectors(
    container: Container,
    texts: list[str],
) -> list[SparseEmbedding]:
    """Compute sparse vectors via Pinecone Inference in batches."""
    return await container.pinecone.embed_sparse(
        texts,
        model=PINECONE_SPARSE_ENGLISH_V0,
        input_type="passage",
        batch_size=SPARSE_BATCH_SIZE,
    )


async def main() -> None:
    async with Container() as c:
        async with c.mongo_db("content_routing") as mongo:
            docs = await mongo.find(
                "content_routing.emails",
                FindOptions(
                    filter={"organization_id": CIVIC_SHOUT_ORG_ID, "vectorized": True},
                    projection={
                        "email_id": 1,
                        "organization_id": 1,
                        "sent_at": 1,
                        "body_plaintext": 1,
                        "_id": 0,
                    },
                    batch_size=5000,
                    cost_key="civic_shout_emails_v1_build",
                ),
            )

        logger.info("Fetched %d emails from Mongo", len(docs))

        email_ids = [int(d["email_id"]) for d in docs]
        texts = [d.get("body_plaintext") or "" for d in docs]

        dense_map = await _fetch_dense_vectors(
            api_key=c.settings.pinecone_api_key.get_secret_value(),
            email_ids=email_ids,
        )
        logger.info("Dense vectors fetched: %d/%d", len(dense_map), len(email_ids))

        sparse_embeddings = await _compute_sparse_vectors(c, texts)
        logger.info("Sparse vectors computed: %d", len(sparse_embeddings))

        rows = []
        for doc, sparse in zip(docs, sparse_embeddings, strict=True):
            eid = int(doc["email_id"])
            dense = dense_map.get(eid)
            rows.append(
                {
                    "email_id": eid,
                    "organization_id": doc["organization_id"],
                    "date_sent": doc.get("sent_at"),
                    "body_plaintext": doc.get("body_plaintext") or "",
                    "dense_vector": dense,
                    "sparse_indices": sparse.indices,
                    "sparse_values": sparse.values,
                }
            )

        df = pl.DataFrame(
            {
                "email_id": pl.Series([r["email_id"] for r in rows], dtype=pl.Int64),
                "organization_id": pl.Series([r["organization_id"] for r in rows], dtype=pl.String),
                "date_sent": pl.Series([r["date_sent"] for r in rows]).cast(
                    pl.Datetime("us", "UTC")
                ),
                "body_plaintext": pl.Series([r["body_plaintext"] for r in rows], dtype=pl.String),
                "dense_vector": pl.Series(
                    [r["dense_vector"] for r in rows],
                    dtype=pl.List(pl.Float32),
                ),
                "sparse_vector": pl.Series(
                    [{"indices": r["sparse_indices"], "values": r["sparse_values"]} for r in rows],
                    dtype=pl.Struct(
                        {
                            "indices": pl.List(pl.Int64),
                            "values": pl.List(pl.Float32),
                        }
                    ),
                ),
            }
        )

        logger.info("Built DataFrame: %d rows, %d cols", df.height, df.width)
        cache.put(1, df)
        logger.info("Saved civic_shout_emails v1 (%d rows)", df.height)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
