import os
import warnings
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# suppress noisy third-party warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*use_fast.*")
warnings.filterwarnings("ignore", message=".*slow image processor.*")

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# Explicit opt-in; the bundled stdio web tools remain the default.
WEB_TOOLS_PROVIDER: str = os.getenv("WEB_TOOLS_PROVIDER", "local").strip().lower()

CHROMA_DIR = Path("./chroma_db")
TEXT_COLLECTION = "text_docs"
IMAGE_COLLECTION = "image_docs"

# sentence-transformers model for text (384-dim)
TEXT_EMBED_MODEL = "all-MiniLM-L6-v2"

# CLIP model for images + cross-modal text queries (512-dim)
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

CLAUDE_MODEL = "claude-sonnet-4-6"

# retrieval
TOP_K = 5

# text chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# supported file extensions
TEXT_EXTENSIONS = {".txt", ".md", ".rst"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}
