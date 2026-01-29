# Oceanir Search

Universal semantic search for text, code, and images.

## Features

- **Semantic Search** - Natural language queries, not just keywords
- **Hybrid Scoring** - BM25 + vector embeddings for accuracy
- **Fast** - MiniLM embeddings (~230ms), CLIP images (~700ms)
- **Multi-format** - 25+ code languages, PNG/JPG/GIF/WebP images

## Installation

```bash
# Homebrew (macOS)
brew tap oceanir/tap
brew install oceanir-search

# Or build from source
brew install onnxruntime
zig build -Doptimize=ReleaseFast
```

## Usage

### CLI

```bash
# Index a directory
oceanir-search index ./src

# Search
oceanir-search "authentication flow"

# Status
oceanir-search status
```

### API

```bash
# Start API server
cd mcp && npm run api

# Search
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "error handling"}'
```

### MCP Server (Claude/Cursor)

Add to your MCP config:

```json
{
  "mcpServers": {
    "oceanir-search": {
      "command": "node",
      "args": ["/path/to/oceanir-search/mcp/dist/mcp.js"]
    }
  }
}
```

### Claude Code Skill

Copy `skill.md` to your Claude Code skills directory.

## API Tiers

| Feature | Free | Pro ($19/mo) | Enterprise |
|---------|------|--------------|------------|
| Searches/day | 100 | Unlimited | Unlimited |
| Max results | 5 | 25 | 100 |
| Image search | - | ✓ | ✓ |
| Self-hosted | - | - | ✓ |

## Architecture

```
oceanir-search/
├── src/              # Zig core
│   ├── main.zig      # CLI & search
│   ├── embeddings.zig # MiniLM
│   ├── vision.zig    # CLIP
│   └── ...
├── mcp/              # TypeScript
│   ├── src/api.ts    # REST API
│   └── src/mcp.ts    # MCP server
└── skill.md          # Claude skill
```

## Models

Auto-downloaded on first use:
- **MiniLM-L6-v2** (22M) - Text embeddings
- **CLIP ViT-base** (150M) - Image embeddings

## License

Proprietary - See LICENSE file.
Commercial licensing: contact@oceanir.ai
