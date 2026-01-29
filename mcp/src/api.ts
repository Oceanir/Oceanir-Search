#!/usr/bin/env node
/**
 * Oceanir Search API Server
 *
 * Tiers:
 * - Free: 100 searches/day, 5 results max
 * - Pro ($19/mo): Unlimited searches, 25 results, image search
 * - Enterprise: Self-hosted, custom limits, priority support
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { RateLimiterMemory } from 'rate-limiter-flexible';
import jwt from 'jsonwebtoken';
import { execSync, spawn } from 'child_process';
import { existsSync } from 'fs';
import { homedir } from 'os';
import path from 'path';

const app = express();
const PORT = process.env.PORT || 3000;
const JWT_SECRET = process.env.JWT_SECRET || 'oceanir-dev-secret-change-in-prod';

// Tier definitions
const TIERS = {
  free: {
    maxResults: 5,
    dailyLimit: 100,
    imageSearch: false,
    apiAccess: true,
  },
  pro: {
    maxResults: 25,
    dailyLimit: -1, // unlimited
    imageSearch: true,
    apiAccess: true,
  },
  enterprise: {
    maxResults: 100,
    dailyLimit: -1,
    imageSearch: true,
    apiAccess: true,
    selfHosted: true,
  }
};

// Rate limiters per tier
const freeLimiter = new RateLimiterMemory({
  points: 100,
  duration: 86400, // 24 hours
});

const proLimiter = new RateLimiterMemory({
  points: 10000,
  duration: 86400,
});

// Middleware
app.use(helmet());
app.use(cors());
app.use(express.json());

// Auth middleware
interface AuthRequest extends express.Request {
  user?: { id: string; tier: keyof typeof TIERS; email: string };
}

const authenticate = (req: AuthRequest, res: express.Response, next: express.NextFunction) => {
  const authHeader = req.headers.authorization;

  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    // Anonymous = free tier
    req.user = { id: 'anonymous', tier: 'free', email: '' };
    return next();
  }

  try {
    const token = authHeader.split(' ')[1];
    const decoded = jwt.verify(token, JWT_SECRET) as { id: string; tier: keyof typeof TIERS; email: string };
    req.user = decoded;
    next();
  } catch {
    res.status(401).json({ error: 'Invalid token' });
  }
};

// Rate limit middleware
const rateLimit = async (req: AuthRequest, res: express.Response, next: express.NextFunction) => {
  const tier = req.user?.tier || 'free';
  const userId = req.user?.id || req.ip || 'unknown';

  try {
    if (tier === 'free') {
      await freeLimiter.consume(userId);
    } else {
      await proLimiter.consume(userId);
    }
    next();
  } catch {
    res.status(429).json({
      error: 'Rate limit exceeded',
      upgrade: 'https://oceanir.ai/pricing'
    });
  }
};

// Find oceanir-search binary
const findBinary = (): string => {
  const locations = [
    path.join(homedir(), '.oceanir', 'bin', 'oceanir-search'),
    '/usr/local/bin/oceanir-search',
    path.join(process.cwd(), '..', 'zig-out', 'bin', 'oceanir-search'),
  ];

  for (const loc of locations) {
    if (existsSync(loc)) return loc;
  }

  // Try PATH
  try {
    execSync('which oceanir-search', { encoding: 'utf-8' });
    return 'oceanir-search';
  } catch {
    throw new Error('oceanir-search binary not found');
  }
};

// Routes
app.get('/', (req, res) => {
  res.json({
    name: 'Oceanir Search API',
    version: '1.0.0',
    docs: 'https://oceanir.ai/docs',
    endpoints: {
      'POST /search': 'Semantic search',
      'POST /index': 'Index a directory (Pro+)',
      'GET /status': 'Index status',
    },
    tiers: TIERS,
  });
});

app.post('/search', authenticate, rateLimit, async (req: AuthRequest, res) => {
  const { query, path: searchPath = '.', limit } = req.body;

  if (!query) {
    return res.status(400).json({ error: 'Query required' });
  }

  const tier = TIERS[req.user?.tier || 'free'];
  const maxResults = Math.min(limit || tier.maxResults, tier.maxResults);

  try {
    const binary = findBinary();
    const result = execSync(`${binary} search "${query}" "${searchPath}"`, {
      encoding: 'utf-8',
      timeout: 30000,
    });

    // Parse results
    const lines = result.split('\n').filter(l => l.trim());
    const results = lines
      .filter(l => /^\s*\d+\./.test(l))
      .slice(0, maxResults)
      .map(line => {
        const match = line.match(/(\d+)\.\s+(.+?):(\d+)-(\d+)\s+\((\d+)%\)/);
        if (match) {
          return {
            file: match[2],
            startLine: parseInt(match[3]),
            endLine: parseInt(match[4]),
            score: parseInt(match[5]) / 100,
          };
        }
        return null;
      })
      .filter(Boolean);

    res.json({
      query,
      results,
      tier: req.user?.tier || 'free',
      remaining: tier.dailyLimit > 0 ? 'check /usage' : 'unlimited',
    });
  } catch (error) {
    res.status(500).json({ error: 'Search failed', details: String(error) });
  }
});

app.post('/index', authenticate, rateLimit, async (req: AuthRequest, res) => {
  const tier = req.user?.tier || 'free';

  if (tier === 'free') {
    return res.status(403).json({
      error: 'Indexing requires Pro tier',
      upgrade: 'https://oceanir.ai/pricing'
    });
  }

  const { path: indexPath = '.' } = req.body;

  try {
    const binary = findBinary();
    const result = execSync(`${binary} index "${indexPath}"`, {
      encoding: 'utf-8',
      timeout: 300000, // 5 min
    });

    res.json({ success: true, output: result });
  } catch (error) {
    res.status(500).json({ error: 'Index failed', details: String(error) });
  }
});

app.get('/status', authenticate, async (req: AuthRequest, res) => {
  try {
    const binary = findBinary();
    const result = execSync(`${binary} status`, { encoding: 'utf-8' });
    res.json({ status: result.trim() });
  } catch (error) {
    res.json({ status: 'No index found' });
  }
});

app.get('/usage', authenticate, async (req: AuthRequest, res) => {
  const userId = req.user?.id || 'anonymous';
  const tier = req.user?.tier || 'free';

  try {
    const limiter = tier === 'free' ? freeLimiter : proLimiter;
    const result = await limiter.get(userId);

    res.json({
      tier,
      used: result?.consumedPoints || 0,
      limit: TIERS[tier].dailyLimit,
      remaining: TIERS[tier].dailyLimit > 0
        ? TIERS[tier].dailyLimit - (result?.consumedPoints || 0)
        : 'unlimited',
      resetsAt: new Date(Date.now() + (result?.msBeforeNext || 0)).toISOString(),
    });
  } catch {
    res.json({ tier, used: 0, limit: TIERS[tier].dailyLimit });
  }
});

// Stripe webhook for subscriptions (placeholder)
app.post('/webhook/stripe', express.raw({ type: 'application/json' }), (req, res) => {
  // TODO: Handle Stripe webhooks for subscription management
  res.json({ received: true });
});

// Start server
app.listen(PORT, () => {
  console.log(`
  ╔═══════════════════════════════════════════╗
  ║         Oceanir Search API v1.0.0         ║
  ╠═══════════════════════════════════════════╣
  ║  http://localhost:${PORT}                    ║
  ║                                           ║
  ║  Endpoints:                               ║
  ║    POST /search   - Semantic search       ║
  ║    POST /index    - Index directory       ║
  ║    GET  /status   - Index status          ║
  ║    GET  /usage    - API usage stats       ║
  ╚═══════════════════════════════════════════╝
  `);
});

export { app, TIERS };
