import json
import logging

import insightface
from insightface.app import FaceAnalysis
import cv2
import torch


def embed_faces(
    known_faces_json: str, output_embeddings_json: str = "embed_faces.json"
) -> None:
    """
    JSON should be structured as:
    [
        {
            "name": "Person Name",
            "image_path": "path/to/image.jpg"
        },
        ...
    ]
    This function reads the JSON file,
        loads the images,
        and extracts face embeddings using the InsightFace model.

    Args:
        known_faces_json (str): Path to the JSON file containing known faces.

    Returns:
        dict: A dictionary with names as keys and their corresponding face embeddings as values.
    """
    logger = logging.getLogger(__name__)
    if not known_faces_json:
        raise ValueError("known_faces.json must be provided.")
    try:
        with open(known_faces_json, "r") as f:
            known_list = json.load(f)

    except FileNotFoundError as e:
        logger.error(f"Error loading known faces JSON: {e}")
        raise FileNotFoundError(
            f"Could not find the known faces JSON file: {known_faces_json}"
        ) from e

    insightface.model_zoo.get_model
    provider = (
        "CUDAExecutionProvider"
        if torch.cuda.is_available()
        else (
            "CoreMLExecutionProvider"
            if torch.backends.mps.is_available()
            else "CPUExecutionProvider"
        )
    )
    ctx_id = 0 if provider != "CPUExecutionProvider" else -1
    # Initialize models
    app = FaceAnalysis(
        providers=[provider],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    embeddings = []
    for entry in known_list:
        name = entry["name"]
        image_path = entry["image_path"]

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logger.warning(f"Could not read image {image_path}. Skipping.")
            continue

        faces = app.get(img)
        if not faces:
            logger.warning(f"No faces detected in image {image_path}. Skipping.")
            continue

        embedding = faces[0].embedding
        embeddings.append({"name": name, "embedding": embedding.tolist()})
    # Save embeddings to JSON
    with open(output_embeddings_json, "w") as f:
        json.dump(embeddings, f, indent=4)
    logger.info(f"Embeddings saved to {output_embeddings_json}")
