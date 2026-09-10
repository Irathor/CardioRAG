import numpy as np

from cardiorag.embeddings.embedder import resolve_device


def test_embed_returns_correct_shape(embedder):
    result = embedder.embed(["cardiac MRI", "myocardial fibrosis", "deep learning"])

    assert result.vectors.shape == (3, embedder.dimension)
    assert result.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert result.dimension == embedder.dimension


def test_embed_empty_list_returns_empty_array(embedder):
    result = embedder.embed([])

    assert result.vectors.shape == (0, embedder.dimension)
    assert result.encode_seconds == 0.0


def test_embed_normalizes_vectors_to_unit_length(embedder):
    result = embedder.embed(["some text", "some other text about hearts"])

    norms = np.linalg.norm(result.vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    assert result.normalized is True


def test_semantically_similar_texts_score_higher_than_unrelated(embedder):
    # Sanity check that the embedding actually captures meaning, not just
    # producing well-formed vectors: this is what retrieval will depend on.
    anchor = embedder.embed(["cardiac MRI assessment of the heart"]).vectors[0]
    related = embedder.embed(["magnetic resonance imaging of cardiac function"]).vectors[0]
    unrelated = embedder.embed(["stock market prices rose sharply today"]).vectors[0]

    # Vectors are unit-normalized, so dot product is cosine similarity.
    sim_related = float(np.dot(anchor, related))
    sim_unrelated = float(np.dot(anchor, unrelated))

    assert sim_related > sim_unrelated


def test_resolve_device_explicit_value_passes_through_without_torch_check():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
