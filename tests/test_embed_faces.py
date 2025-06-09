import unittest
import tempfile
import os
import json
import numpy as np
from unittest.mock import patch, MagicMock

from captionalchemy.tools.cv.embed_known_faces import embed_faces


class TestEmbedFacesFunctionality(unittest.TestCase):
    def setUp(self):
        # Two dummy face entries
        self.entries = [
            {"name": "Alice", "image_path": "path/to/alice.jpg"},
            {"name": "Bob", "image_path": "path/to/bob.jpg"},
        ]

    def test_embed_faces_empty_json(self):
        with self.assertRaises(ValueError):
            embed_faces("", "output_embeddings.json")

    @patch("captionalchemy.tools.cv.embed_known_faces.FaceAnalysis")
    @patch("captionalchemy.tools.cv.embed_known_faces.cv2.imread")
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_embed_faces_success(
        self, mock_mps, mock_cuda, mock_imread, mock_face_analysis
    ):
        # Prepare temporary JSON input and output paths
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "known_faces.json")
            output_path = os.path.join(tmpdir, "output_embeddings.json")

            # Write sample known-faces JSON
            for entry in self.entries:
                # Create dummy image files
                img_path = os.path.join(tmpdir, f"{entry['name']}.jpg")
                open(img_path, "wb").close()
                entry["image_path"] = img_path
            with open(input_path, "w") as f:
                json.dump(self.entries, f)

        # Mock cv2.imread to return a dummy array for any path
        mock_imread.return_value = np.random.randint(
            0, 255, (100, 100, 3), dtype=np.uint8
        )

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
        self.assertEqual(len(data), 2)
        names = [entry["name"] for entry in data]
        self.assertCountEqual(names, ["Alice", "Bob"])
        for d in data:
            emb = d["embedding"]
            self.assertIsInstance(emb, list)
            self.assertEqual(len(emb), 512)


if __name__ == "__main__":
    unittest.main()
