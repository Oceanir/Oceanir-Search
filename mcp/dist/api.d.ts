#!/usr/bin/env node
/**
 * Oceanir Search API Server
 *
 * Tiers:
 * - Free: 100 searches/day, 5 results max
 * - Pro ($19/mo): Unlimited searches, 25 results, image search
 * - Enterprise: Self-hosted, custom limits, priority support
 */
declare const app: import("express-serve-static-core").Express;
declare const TIERS: {
    free: {
        maxResults: number;
        dailyLimit: number;
        imageSearch: boolean;
        apiAccess: boolean;
    };
    pro: {
        maxResults: number;
        dailyLimit: number;
        imageSearch: boolean;
        apiAccess: boolean;
    };
    enterprise: {
        maxResults: number;
        dailyLimit: number;
        imageSearch: boolean;
        apiAccess: boolean;
        selfHosted: boolean;
    };
};
export { app, TIERS };
