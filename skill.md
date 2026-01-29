# Oceanir Search Skill

<command-name>oceanir</command-name>

<description>
Semantic search for code and text. Search using natural language - finds similar content even without exact matches.
</description>

<usage>
/oceanir <query>           Search the codebase
/oceanir index [path]      Index a directory
/oceanir status            Show index status
</usage>

<instructions>
When the user invokes /oceanir:

1. **For search queries** (/oceanir <query>):
   - Run `oceanir-search "<query>"` using Bash
   - Display results with file paths, line numbers, and relevance scores
   - Offer to read the top result if relevant

2. **For indexing** (/oceanir index [path]):
   - Run `oceanir-search index <path>` using Bash
   - Show indexing progress and completion stats

3. **For status** (/oceanir status):
   - Run `oceanir-search status` using Bash
   - Display index statistics

Example queries:
- "authentication error handling"
- "database connection"
- "API rate limiting"
- "user login flow"

The search uses MiniLM embeddings (384-dim) with hybrid BM25+vector scoring.
Results show: file:startLine-endLine (score%)
</instructions>

<examples>
User: /oceanir authentication flow
Action: Run `oceanir-search "authentication flow"` and display results

User: /oceanir index ./src
Action: Run `oceanir-search index ./src` and show progress

User: /oceanir status
Action: Run `oceanir-search status` and display stats
</examples>
