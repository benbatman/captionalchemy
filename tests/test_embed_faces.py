import os
import json
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from captionalchemy.tools.cv.embed_known_faces import embed_faces


@pytest.fixture
def face_entries():
    """Two dummy face entries."""
    return [
        {"name": "Alice", "image_path": "path/to/alice.jpg"},
        {"name": "Bob", "image_path": "path/to/bob.jpg"},
    ]


def test_embed_faces_empty_json():
    with pytest.raises(ValueError):
        embed_faces("", "output_embeddings.json")


@patch("captionalchemy.tools.cv.embed_known_faces.FaceAnalysis")
@patch("captionalchemy.tools.cv.embed_known_faces.cv2.imread")
@patch("torch.cuda.is_available", return_value=False)
@patch("torch.backends.mps.is_available", return_value=False)
def test_embed_faces_success(
    mock_mps, mock_cuda, mock_imread, mock_face_analysis, temp_dir, face_entries
):
    """Test successful face embedding process."""
    input_path = os.path.join(temp_dir, "known_faces.json")
    output_path = os.path.join(temp_dir, "output_embeddings.json")

    # Write sample known-faces JSON
    for entry in face_entries:
        # Create dummy image files
        img_path = os.path.join(temp_dir, f"{entry['name']}.jpg")
        open(img_path, "wb").close()
        entry["image_path"] = img_path
    with open(input_path, "w") as f:
        json.dump(face_entries, f)

    # Mock cv2.imread to return a dummy array for any path
    mock_imread.return_value = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    # Mock FaceAnalysis to return a dummy embedding and get() method
    mock_app = MagicMock()
    # Each call returns a single face with a known embedding
    face = MagicMock()
    face.embedding = np.random.rand(512).astype(np.float32)
    mock_app.get.return_value = [face]
    mock_face_analysis.return_value = mock_app

    # Run
    embed_faces(input_path, output_path)

    # Read output and verify
    with open(output_path, "r") as f:
        data = json.load(f)

    # Should have two entries
    assert len(data) == 2
    names = [entry["name"] for entry in data]
    assert set(names) == {"Alice", "Bob"}

    for d in data:
        emb = d["embedding"]
        assert isinstance(emb, list)
        assert len(emb) == 512
