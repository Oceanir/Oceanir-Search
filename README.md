# Oceanir Search

Universal semantic search for text, code, and images. Built with Zig + ONNX.

## Features

- **Text/Code Search** - MiniLM embeddings (~230ms)
- **Image Embedding** - CLIP vision (~700ms)
- **Hybrid Search** - BM25 + vector with local reranking
- **Multi-format** - Supports 25+ code languages, PNG/JPG/GIF/WebP images

## Installation

```bash
# Install ONNX Runtime
brew install onnxruntime

# Build
zig build -Doptimize=ReleaseFast

# Add to path
cp zig-out/bin/oceanir-search /usr/local/bin/
```

## Usage

```bash
# Index a directory
oceanir-search index ./src

# Search
oceanir-search "authentication flow"

# Test image embedding
oceanir-search embed-image photo.png

# Status
oceanir-search status
```

## Performance

| Operation | Time |
|-----------|------|
| Text search | ~230ms |
| Image embed | ~700ms |
| Index (8 files) | ~4s |

## Architecture

```
src/
├── main.zig        # CLI and search logic
├── embeddings.zig  # Text embeddings (MiniLM)
├── vision.zig      # Image embeddings (CLIP)
├── onnx.zig        # ONNX Runtime bindings
├── tokenizer.zig   # WordPiece tokenizer
├── store.zig       # Index storage
├── scanner.zig     # File discovery
├── bm25.zig        # BM25 scoring
└── vector.zig      # Vector operations
```

## Models

Downloaded automatically on first use:
- **MiniLM-L6-v2** (22M params) - Text embeddings
- **CLIP ViT-base** (150M params) - Image embeddings

## License

MIT
