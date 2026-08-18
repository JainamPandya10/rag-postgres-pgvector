from src.chunking import chunk_text


def test_chunk_text_basic():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=30, overlap=5)

    assert len(chunks) > 1
    # every chunk should have at most chunk_size words
    for c in chunks:
        assert len(c.split()) <= 30


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_overlap():
    text = " ".join(str(i) for i in range(20))
    chunks = chunk_text(text, chunk_size=10, overlap=3)
    # last 3 words of chunk 1 should equal first 3 words of chunk 2
    assert chunks[0].split()[-3:] == chunks[1].split()[:3]
