from langchain_huggingface import HuggingFaceEmbeddings

import logging
from app.config.config import get_settings

logger = logging.getLogger(__name__)

hugging_face_embeddings = HuggingFaceEmbeddings(model_name=get_settings().hugging_face_embeddings_model_name)