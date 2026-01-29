#!/usr/bin/env node
/**
 * Oceanir Search MCP Server
 *
 * Provides semantic search capabilities to Claude, Cursor, and other MCP clients.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { execSync } from 'child_process';
import { existsSync } from 'fs';
import { homedir } from 'os';
import path from 'path';

// Find oceanir-search binary
const findBinary = (): string => {
  const locations = [
    path.join(homedir(), '.oceanir', 'bin', 'oceanir-search'),
    '/usr/local/bin/oceanir-search',
    path.join(process.cwd(), '..', 'zig-out', 'bin', 'oceanir-search'),
    path.join(__dirname, '..', '..', 'zig-out', 'bin', 'oceanir-search'),
  ];

  for (const loc of locations) {
    if (existsSync(loc)) return loc;
  }

  try {
    execSync('which oceanir-search', { encoding: 'utf-8' });
    return 'oceanir-search';
  } catch {
    throw new Error('oceanir-search binary not found. Install with: brew tap oceanir/tap && brew install oceanir-search');
  }
};

// Create MCP server
const server = new Server(
  {
    name: 'oceanir-search',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

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
        const query = (args as { query: string; path?: string; limit?: number }).query;
        const searchPath = (args as { path?: string }).path || '.';

        const result = execSync(`${binary} search "${query.replace(/"/g, '\\"')}" "${searchPath}"`, {
          encoding: 'utf-8',
          timeout: 30000,
          cwd: searchPath !== '.' ? searchPath : undefined,
        });

        return {
          content: [
            {
              type: 'text',
              text: result,
            },
          ],
        };
      }

      case 'index_directory': {
        const indexPath = (args as { path?: string }).path || '.';

        const result = execSync(`${binary} index "${indexPath}"`, {
          encoding: 'utf-8',
          timeout: 300000,
          cwd: indexPath !== '.' ? indexPath : undefined,
        });

        return {
          content: [
            {
              type: 'text',
              text: result,
            },
          ],
        };
      }

      case 'search_status': {
        const result = execSync(`${binary} status`, {
          encoding: 'utf-8',
          timeout: 5000,
        });

        return {
          content: [
            {
              type: 'text',
              text: result,
            },
          ],
        };
      }

      case 'embed_image': {
        const imagePath = (args as { path: string }).path;

        const result = execSync(`${binary} embed-image "${imagePath}"`, {
          encoding: 'utf-8',
          timeout: 30000,
        });

        return {
          content: [
            {
              type: 'text',
              text: result,
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
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

// Start server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Oceanir Search MCP server running');
}

main().catch(console.error);
