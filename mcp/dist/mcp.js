#!/usr/bin/env node
/**
 * Oceanir Search MCP Server
 *
 * Provides semantic search capabilities to Claude, Cursor, and other MCP clients.
 * Uses persistent daemon for fast query response times.
 */
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema, } from '@modelcontextprotocol/sdk/types.js';
import { spawn, execSync } from 'child_process';
import { existsSync } from 'fs';
import { homedir } from 'os';
import path from 'path';
import { createInterface } from 'readline';
// Find oceanir-search binary
const findBinary = () => {
    const locations = [
        path.join(homedir(), '.oceanir', 'bin', 'oceanir-search'),
        '/usr/local/bin/oceanir-search',
        path.join(process.cwd(), '..', 'zig-out', 'bin', 'oceanir-search'),
        path.join(__dirname, '..', '..', 'zig-out', 'bin', 'oceanir-search'),
    ];
    for (const loc of locations) {
        if (existsSync(loc))
            return loc;
    }
    try {
        execSync('which oceanir-search', { encoding: 'utf-8' });
        return 'oceanir-search';
    }
    catch {
        throw new Error('oceanir-search binary not found. Install with: brew tap oceanir/tap && brew install oceanir-search');
    }
};
// Daemon process manager
class SearchDaemon {
    process = null;
    readline = null;
    responseQueue = [];
    ready = false;
    binary;
    constructor() {
        this.binary = findBinary();
    }
    async start() {
        if (this.process)
            return;
        return new Promise((resolve, reject) => {
            this.process = spawn(this.binary, ['serve'], {
                stdio: ['pipe', 'pipe', 'pipe'],
            });
            this.process.stderr?.on('data', (data) => {
                const msg = data.toString();
                console.error('[daemon]', msg.trim());
                if (msg.includes('Ready.')) {
                    this.ready = true;
                    resolve();
                }
            });
            this.readline = createInterface({
                input: this.process.stdout,
                crlfDelay: Infinity,
            });
            this.readline.on('line', (line) => {
                const callback = this.responseQueue.shift();
                if (callback) {
                    callback(line);
                }
            });
            this.process.on('error', (err) => {
                console.error('[daemon] Process error:', err);
                reject(err);
            });
            this.process.on('exit', (code) => {
                console.error('[daemon] Process exited with code:', code);
                this.process = null;
                this.ready = false;
            });
            // Timeout for startup
            setTimeout(() => {
                if (!this.ready) {
                    reject(new Error('Daemon startup timeout'));
                }
            }, 60000);
        });
    }
    async search(query) {
        if (!this.ready || !this.process) {
            await this.start();
        }
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Search timeout'));
            }, 30000);
            this.responseQueue.push((response) => {
                clearTimeout(timeout);
                try {
                    resolve(JSON.parse(response));
                }
                catch {
                    resolve({ error: 'Invalid response', raw: response });
                }
            });
            this.process.stdin.write(JSON.stringify({ query }) + '\n');
        });
    }
    stop() {
        if (this.process) {
            this.process.stdin?.write('quit\n');
            this.process.kill();
            this.process = null;
            this.ready = false;
        }
    }
}
// Global daemon instance
const daemon = new SearchDaemon();
// Create MCP server
const server = new Server({
    name: 'oceanir-search',
    version: '1.0.0',
}, {
    capabilities: {
        tools: {},
    },
});
// Define tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
        tools: [
            {
                name: 'semantic_search',
                description: 'Search code and text using natural language. Finds semantically similar content even without exact keyword matches.',
                inputSchema: {
                    type: 'object',
                    properties: {
                        query: {
                            type: 'string',
                            description: 'Natural language search query (e.g., "authentication error handling", "database connection setup")',
                        },
                        path: {
                            type: 'string',
                            description: 'Directory to search in (default: current directory)',
                        },
                        limit: {
                            type: 'number',
                            description: 'Maximum number of results (default: 10)',
                        },
                    },
                    required: ['query'],
                },
            },
            {
                name: 'index_directory',
                description: 'Index a directory for semantic search. Required before searching.',
                inputSchema: {
                    type: 'object',
                    properties: {
                        path: {
                            type: 'string',
                            description: 'Directory to index (default: current directory)',
                        },
                    },
                },
            },
            {
                name: 'search_status',
                description: 'Check the status of the search index.',
                inputSchema: {
                    type: 'object',
                    properties: {},
                },
            },
            {
                name: 'embed_image',
                description: 'Generate embeddings for an image file (Pro feature).',
                inputSchema: {
                    type: 'object',
                    properties: {
                        path: {
                            type: 'string',
                            description: 'Path to the image file',
                        },
                    },
                    required: ['path'],
                },
            },
        ],
    };
});
// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    try {
        const binary = findBinary();
        switch (name) {
            case 'semantic_search': {
                const query = args.query;
                // Use daemon for fast search
                const result = await daemon.search(query);
                if (result.error) {
                    return {
                        content: [{ type: 'text', text: `Error: ${result.error}` }],
                        isError: true,
                    };
                }
                // Format results nicely
                const formatted = [`Search: "${result.query}" (${result.time_ms}ms)\n`];
                for (const r of result.results || []) {
                    formatted.push(`  ${r.file}:${r.lines[0]}-${r.lines[1]} (${r.score}%)`);
                }
                return {
                    content: [{ type: 'text', text: formatted.join('\n') }],
                };
            }
            case 'index_directory': {
                const indexPath = args.path || '.';
                const result = execSync(`${binary} index "${indexPath}"`, {
                    encoding: 'utf-8',
                    timeout: 300000,
                    cwd: indexPath !== '.' ? indexPath : undefined,
                });
                return {
                    content: [{ type: 'text', text: result }],
                };
            }
            case 'search_status': {
                const result = execSync(`${binary} status`, {
                    encoding: 'utf-8',
                    timeout: 5000,
                });
                return {
                    content: [{ type: 'text', text: result }],
                };
            }
            case 'embed_image': {
                const imagePath = args.path;
                const result = execSync(`${binary} embed-image "${imagePath}"`, {
                    encoding: 'utf-8',
                    timeout: 30000,
                });
                return {
                    content: [{ type: 'text', text: result }],
                };
            }
            default:
                throw new Error(`Unknown tool: ${name}`);
        }
    }
    catch (error) {
        return {
            content: [
                {
                    type: 'text',
                    text: `Error: ${error instanceof Error ? error.message : String(error)}`,
                },
            ],
            isError: true,
        };
    }
});
// Cleanup on exit
process.on('SIGINT', () => {
    daemon.stop();
    process.exit(0);
});
process.on('SIGTERM', () => {
    daemon.stop();
    process.exit(0);
});
// Start server
async function main() {
    // Pre-start daemon for faster first query
    console.error('Starting Oceanir Search daemon...');
    await daemon.start();
    console.error('Daemon ready');
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error('Oceanir Search MCP server running');
}
main().catch(console.error);
