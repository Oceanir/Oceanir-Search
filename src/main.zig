const std = @import("std");
const vector = @import("vector.zig");
const bm25 = @import("bm25.zig");
const store = @import("store.zig");
const scanner = @import("scanner.zig");
const embeddings = @import("embeddings.zig");
const vision = @import("vision.zig");

const print = std.debug.print;

const VERSION = "1.0.0";

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Ensure embedders are cleaned up before exit
    defer embeddings.deinitGlobal();
    defer vision.deinitGlobal();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        printUsage();
        return;
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "search") or std.mem.eql(u8, command, "s")) {
        if (args.len < 3) {
            print("Error: search requires a query\n", .{});
            return;
        }
        try runSearch(allocator, args[2], if (args.len > 3) args[3] else ".");
    } else if (std.mem.eql(u8, command, "index") or std.mem.eql(u8, command, "i")) {
        const path = if (args.len > 2) args[2] else ".";
        try runIndex(allocator, path);
    } else if (std.mem.eql(u8, command, "status")) {
        try runStatus(allocator);
    } else if (std.mem.eql(u8, command, "embed-image")) {
        if (args.len < 3) {
            print("Usage: oceanir-search embed-image <image_path>\n", .{});
            return;
        }
        try testImageEmbed(allocator, args[2]);
    } else if (std.mem.eql(u8, command, "serve") or std.mem.eql(u8, command, "daemon")) {
        try runServer(allocator);
    } else if (std.mem.eql(u8, command, "--version") or std.mem.eql(u8, command, "-v")) {
        print("Oceanir Search v{s}\n", .{VERSION});
        print("Universal semantic search - text, code, images\n", .{});
    } else if (std.mem.eql(u8, command, "--help") or std.mem.eql(u8, command, "-h")) {
        printUsage();
    } else {
        // Treat as search query
        try runSearch(allocator, command, ".");
    }
}

fn printUsage() void {
    print(
        \\
        \\   ___                      _        ____                      _
        \\  / _ \\  ___ ___  __ _ _ __ (_)_ __  / ___|  ___  __ _ _ __ ___| |__
        \\ | | | |/ __/ _ \\/ _` | '_ \\| | '__| \\___ \\ / _ \\/ _` | '__/ __| '_ \\
        \\ | |_| | (_|  __/ (_| | | | | | |     ___) |  __/ (_| | | | (__| | | |
        \\  \\___/ \\___\\___|\\__,_|_| |_|_|_|    |____/ \\___|\\__,_|_|  \\___|_| |_|
        \\
        \\  Universal semantic search - text, code, images
        \\
        \\USAGE:
        \\    oceanir-search <command> [options]
        \\    oceanir-search <query>       Direct search
        \\
        \\COMMANDS:
        \\    search, s <query>    Search indexed files
        \\    index, i [path]      Index files in directory
        \\    status               Show index status
        \\    serve                Start daemon (keeps model loaded)
        \\    embed-image <path>   Test image embedding
        \\
        \\OPTIONS:
        \\    -h, --help           Show this help
        \\    -v, --version        Show version
        \\
        \\EXAMPLES:
        \\    oceanir-search "error handling"
        \\    oceanir-search index ./src
        \\    oceanir-search embed-image photo.png
        \\
    , .{});
}

fn runSearch(allocator: std.mem.Allocator, query: []const u8, path: []const u8) !void {
    _ = path;

    const start_time = std.time.nanoTimestamp();

    // Load store
    var idx = store.Store.load(allocator) catch |err| {
        if (err == error.FileNotFound) {
            print("No index found. Run: oceanir-search index <path>\n", .{});
            return;
        }
        return err;
    };
    defer idx.deinit();

    if (idx.chunks.count() == 0) {
        print("Index is empty. Run: oceanir-search index <path>\n", .{});
        return;
    }

    // Generate query embedding (uses search_query prefix for nomic)
    const query_embedding = try embeddings.embedQuery(allocator, query);
    defer allocator.free(query_embedding);

    // Hybrid search: BM25 + vector
    var results = try hybridSearch(allocator, &idx, query, query_embedding, 10);
    defer results.deinit(allocator);

    // Local reranking for better precision
    localRerank(query, &results);

    const end_time = std.time.nanoTimestamp();
    const elapsed_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;

    print("\n  Oceanir Search: \"{s}\"\n", .{query});
    print("  Found {d} matches in {d:.1}ms\n\n", .{ results.items.len, elapsed_ms });

    // Print results
    for (results.items, 0..) |result, i| {
        const score_pct: u32 = @intFromFloat(result.score * 100);
        print("  {d}. {s}:{d}-{d} ({d}%)\n", .{
            i + 1,
            result.chunk.file_path,
            result.chunk.start_line,
            result.chunk.end_line,
            score_pct,
        });
    }
    print("\n", .{});
}

const BATCH_SIZE: usize = 64; // Larger batches for throughput
const NUM_WORKERS: usize = 4; // Parallel embedding workers

fn runIndex(allocator: std.mem.Allocator, path: []const u8) !void {
    const start_time = std.time.nanoTimestamp();

    print("\n  Oceanir Search - Indexing {s}\n", .{path});

    // Scan files
    var files = try scanner.scan(allocator, path);
    defer {
        for (files.items) |f| {
            allocator.free(f.path);
            allocator.free(f.content);
        }
        files.deinit(allocator);
    }

    print("  Found {d} files\n", .{files.items.len});

    // Create store
    var idx = store.Store.init(allocator);
    defer idx.deinit();

    // Collect all chunks first
    var all_chunks: std.ArrayListUnmanaged(struct {
        file_path: []const u8,
        content: []const u8,
        start_line: usize,
        end_line: usize,
    }) = .{};
    defer all_chunks.deinit(allocator);

    var indexed: usize = 0;
    var image_count: usize = 0;

    for (files.items) |file| {
        if (file.file_type == .image) {
            const img_embedding = vision.embedImage(allocator, file.path) catch |err| {
                std.debug.print("  Skipping image {s}: {}\n", .{ file.path, err });
                continue;
            };
            allocator.free(img_embedding);
            image_count += 1;
            continue;
        }

        const chunks = try chunkFile(allocator, file);
        defer allocator.free(chunks);

        for (chunks) |chunk| {
            try all_chunks.append(allocator, .{
                .file_path = file.path,
                .content = chunk.content,
                .start_line = chunk.start_line,
                .end_line = chunk.end_line,
            });
        }
        indexed += 1;
    }

    const total_chunks = all_chunks.items.len;
    print("  Embedding {d} chunks (batch: {d}, workers: {d})...\n", .{ total_chunks, BATCH_SIZE, NUM_WORKERS });

    // Process in large batches
    var processed: usize = 0;
    while (processed < total_chunks) {
        const batch_end = @min(processed + BATCH_SIZE, total_chunks);
        const batch = all_chunks.items[processed..batch_end];

        var texts = try allocator.alloc([]const u8, batch.len);
        defer allocator.free(texts);
        for (batch, 0..) |chunk, i| {
            texts[i] = chunk.content;
        }

        const batch_embeddings = try embeddings.embedBatch(allocator, texts);
        defer {
            for (batch_embeddings) |emb| allocator.free(emb);
            allocator.free(batch_embeddings);
        }

        for (batch, 0..) |chunk, i| {
            const emb_copy = try allocator.alloc(f32, batch_embeddings[i].len);
            @memcpy(emb_copy, batch_embeddings[i]);

            try idx.addChunk(.{
                .id = try generateId(allocator, chunk.file_path, chunk.start_line),
                .file_path = try allocator.dupe(u8, chunk.file_path),
                .content = try allocator.dupe(u8, chunk.content),
                .start_line = chunk.start_line,
                .end_line = chunk.end_line,
                .embedding = emb_copy,
            });
        }

        processed = batch_end;
        const pct = (processed * 100) / total_chunks;
        print("  [{d}%] {d}/{d} chunks\r", .{ pct, processed, total_chunks });
    }
    print("\n", .{});

    if (image_count > 0) {
        print("  Found {d} images (vision indexing coming soon)\n", .{image_count});
    }

    idx.updateBm25Stats();
    try idx.save();

    const end_time = std.time.nanoTimestamp();
    const elapsed_ms = @as(f64, @floatFromInt(end_time - start_time)) / 1_000_000.0;
    const chunks_per_sec = @as(f64, @floatFromInt(total_chunks)) / (elapsed_ms / 1000.0);

    print("  Indexed {d} files, {d} chunks in {d:.0}ms ({d:.1} chunks/sec)\n\n", .{ indexed, total_chunks, elapsed_ms, chunks_per_sec });
}

fn runStatus(allocator: std.mem.Allocator) !void {
    var idx = store.Store.load(allocator) catch |err| {
        if (err == error.FileNotFound) {
            print("\n  No Oceanir Search index found.\n  Run: oceanir-search index <path>\n\n", .{});
            return;
        }
        return err;
    };
    defer idx.deinit();

    print("\n  Oceanir Search Index Status\n", .{});
    print("  --------------------------\n", .{});
    print("  Total chunks: {d}\n", .{idx.chunks.count()});
    print("  Embedding dim: {d}\n\n", .{embeddings.EMBEDDING_DIM});
}

fn testImageEmbed(allocator: std.mem.Allocator, image_path: []const u8) !void {
    const start = std.time.nanoTimestamp();

    print("\n  Embedding image: {s}\n", .{image_path});

    const emb = try vision.embedImage(allocator, image_path);
    defer allocator.free(emb);

    const elapsed = @as(f64, @floatFromInt(std.time.nanoTimestamp() - start)) / 1_000_000.0;

    print("  Dimension: {d}\n", .{emb.len});
    print("  First 5 values: [{d:.4}, {d:.4}, {d:.4}, {d:.4}, {d:.4}]\n", .{
        emb[0], emb[1], emb[2], emb[3], emb[4],
    });
    print("  Time: {d:.1}ms\n\n", .{elapsed});
}

const SearchResult = struct {
    chunk: *store.Chunk,
    score: f32,
    bm25_score: f32,
    vector_score: f32,
};

fn hybridSearch(
    allocator: std.mem.Allocator,
    idx: *store.Store,
    query: []const u8,
    query_embedding: []const f32,
    limit: usize,
) !std.ArrayListUnmanaged(SearchResult) {
    var results: std.ArrayListUnmanaged(SearchResult) = .{};

    // Score all chunks
    var it = idx.chunks.iterator();
    while (it.next()) |entry| {
        const chunk = entry.value_ptr;

        // Vector similarity (cosine returns -1 to 1, but for normalized vectors it's 0 to 1)
        const vec_score = vector.cosineSimilarity(query_embedding, chunk.embedding);

        // BM25 score
        const bm25_raw = bm25.score(query, chunk.content, &idx.idf, idx.avg_doc_len);
        const bm25_norm = normalizeBm25(bm25_raw);

        // Combined score (35% BM25, 65% vector)
        // Apply scaling to spread scores better
        const raw_combined = 0.35 * bm25_norm + 0.65 * vec_score;
        // Scale to 0.3-1.0 range for better visual differentiation
        const combined = 0.3 + raw_combined * 0.7;

        try results.append(allocator, .{
            .chunk = chunk,
            .score = combined,
            .bm25_score = bm25_raw,
            .vector_score = vec_score,
        });
    }

    // Sort by score descending
    std.mem.sort(SearchResult, results.items, {}, struct {
        fn lessThan(_: void, a: SearchResult, b: SearchResult) bool {
            return a.score > b.score;
        }
    }.lessThan);

    // Truncate to limit
    if (results.items.len > limit) {
        results.shrinkRetainingCapacity(limit);
    }

    return results;
}

fn normalizeBm25(s: f32) f32 {
    // Better normalization with wider range
    // BM25 scores typically range 0-20, map to 0-1
    const scaled = @min(s / 10.0, 1.0);
    return scaled;
}

/// Local reranking: boost scores based on exact matches
fn localRerank(query: []const u8, results: *std.ArrayListUnmanaged(SearchResult)) void {
    const query_lower = blk: {
        var buf: [512]u8 = undefined;
        const len = @min(query.len, 512);
        for (query[0..len], 0..) |c, i| {
            buf[i] = std.ascii.toLower(c);
        }
        break :blk buf[0..len];
    };

    for (results.items) |*result| {
        var boost: f32 = 0;
        const content = result.chunk.content;

        // 1. Exact phrase match (big boost)
        if (containsIgnoreCase(content, query)) {
            boost += 0.2;
        }

        // 2. Individual term matches with position weighting
        var terms = std.mem.tokenizeScalar(u8, query_lower, ' ');
        var matched: f32 = 0;
        var total: f32 = 0;
        while (terms.next()) |term| {
            total += 1;
            if (containsIgnoreCase(content, term)) {
                matched += 1;
                // Bonus if term appears in first 200 chars (likely function signature)
                if (content.len >= term.len) {
                    const prefix_len = @min(content.len, 200);
                    if (containsIgnoreCase(content[0..prefix_len], term)) {
                        boost += 0.03;
                    }
                }
            }
        }

        // Term coverage boost
        if (total > 0) {
            boost += (matched / total) * 0.15;
        }

        // 3. File path relevance
        var path_terms = std.mem.tokenizeScalar(u8, query_lower, ' ');
        while (path_terms.next()) |term| {
            if (containsIgnoreCase(result.chunk.file_path, term)) {
                boost += 0.05;
            }
        }

        result.score = @min(result.score + boost, 1.0);
    }

    // Re-sort after boosting
    std.mem.sort(SearchResult, results.items, {}, struct {
        fn lessThan(_: void, a: SearchResult, b: SearchResult) bool {
            return a.score > b.score;
        }
    }.lessThan);
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0 or haystack.len < needle.len) return false;

    var i: usize = 0;
    while (i <= haystack.len - needle.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i..][0..needle.len], needle)) {
            return true;
        }
    }
    return false;
}

const TempChunk = struct {
    content: []const u8,
    start_line: usize,
    end_line: usize,
};

fn chunkFile(allocator: std.mem.Allocator, file: scanner.ScannedFile) ![]TempChunk {
    var chunks: std.ArrayListUnmanaged(TempChunk) = .{};

    // Simple chunking: 50 lines per chunk
    const chunk_size: usize = 50;

    var start: usize = 0;
    var line_start: usize = 0;
    var current_line: usize = 1;

    for (file.content, 0..) |c, i| {
        if (c == '\n') {
            current_line += 1;

            if (current_line - line_start >= chunk_size or i == file.content.len - 1) {
                try chunks.append(allocator, .{
                    .content = file.content[start..i],
                    .start_line = line_start + 1,
                    .end_line = current_line,
                });

                start = i + 1;
                line_start = current_line;
            }
        }
    }

    return chunks.toOwnedSlice(allocator);
}

fn generateId(allocator: std.mem.Allocator, path: []const u8, line: usize) ![]u8 {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(path);
    hasher.update(std.mem.asBytes(&line));
    const hash = hasher.finalResult();

    return try std.fmt.allocPrint(allocator, "{s}", .{std.fmt.bytesToHex(hash[0..8], .lower)});
}

fn runServer(allocator: std.mem.Allocator) !void {
    const stdin = std.fs.File.stdin();
    const stdout = std.fs.File.stdout();

    // Pre-load embedder (this is the slow part we want to do once)
    print("Oceanir Search daemon starting...\n", .{});
    print("Loading embedding model...\n", .{});

    const init_start = std.time.nanoTimestamp();
    const warmup_emb = try embeddings.embed(allocator, "warmup"); // Force model load
    allocator.free(warmup_emb);
    const init_time = @as(f64, @floatFromInt(std.time.nanoTimestamp() - init_start)) / 1_000_000.0;

    print("Model loaded in {d:.0}ms\n", .{init_time});
    print("Ready. Send JSON queries to stdin.\n", .{});
    print("Format: {{\"query\": \"your search\"}}\n", .{});
    print("Send 'quit' or Ctrl+D to exit.\n\n", .{});

    // Load store once
    var idx = store.Store.load(allocator) catch |err| {
        if (err == error.FileNotFound) {
            print("{{\"error\": \"No index found. Run: oceanir-search index <path>\"}}\n", .{});
            return;
        }
        return err;
    };
    defer idx.deinit();

    // Read loop - manual line buffering
    var line_buf: [4096]u8 = undefined;
    var line_len: usize = 0;

    while (true) {
        // Read one byte at a time until newline
        var byte_buf: [1]u8 = undefined;
        const bytes_read = stdin.read(&byte_buf) catch |err| {
            print("{{\"error\": \"Read error: {}\"}}\n", .{err});
            break;
        };

        if (bytes_read == 0) break; // EOF

        if (byte_buf[0] == '\n') {
            const input = std.mem.trim(u8, line_buf[0..line_len], " \t\r\n");
            line_len = 0; // Reset for next line

            if (input.len == 0) continue;

            // Check for quit command
            if (std.mem.eql(u8, input, "quit") or std.mem.eql(u8, input, "exit")) {
                _ = stdout.write("{\"status\": \"goodbye\"}\n") catch {};
                break;
            }

            // Parse JSON query
            const query = parseQuery(input) orelse {
                _ = stdout.write("{\"error\": \"Invalid JSON\"}\n") catch {};
                continue;
            };

            // Execute search
            const start_time = std.time.nanoTimestamp();

            const query_embedding = embeddings.embedQuery(allocator, query) catch {
                _ = stdout.write("{\"error\": \"Embedding failed\"}\n") catch {};
                continue;
            };
            defer allocator.free(query_embedding);

            var results = hybridSearch(allocator, &idx, query, query_embedding, 10) catch {
                _ = stdout.write("{\"error\": \"Search failed\"}\n") catch {};
                continue;
            };
            defer results.deinit(allocator);

            localRerank(query, &results);

            const elapsed_ms = @as(f64, @floatFromInt(std.time.nanoTimestamp() - start_time)) / 1_000_000.0;

            // Output JSON results
            var json_buf: [8192]u8 = undefined;
            var json_stream = std.io.fixedBufferStream(&json_buf);
            const json_writer = json_stream.writer();

            json_writer.print("{{\"query\": \"{s}\", \"time_ms\": {d:.1}, \"results\": [", .{ query, elapsed_ms }) catch continue;

            for (results.items, 0..) |result, i| {
                if (i > 0) json_writer.writeAll(",") catch continue;
                const score_pct: u32 = @intFromFloat(result.score * 100);
                json_writer.print("{{\"file\": \"{s}\", \"lines\": [{d}, {d}], \"score\": {d}}}", .{
                    result.chunk.file_path,
                    result.chunk.start_line,
                    result.chunk.end_line,
                    score_pct,
                }) catch continue;
            }

            json_writer.writeAll("]}}\n") catch continue;
            _ = stdout.write(json_stream.getWritten()) catch continue;
        } else {
            if (line_len < line_buf.len) {
                line_buf[line_len] = byte_buf[0];
                line_len += 1;
            }
        }
    }
}

fn parseQuery(json: []const u8) ?[]const u8 {
    // Simple JSON parsing for {"query": "..."}
    const query_key = "\"query\"";
    const key_pos = std.mem.indexOf(u8, json, query_key) orelse return null;
    const after_key = json[key_pos + query_key.len ..];

    // Find colon
    const colon_pos = std.mem.indexOf(u8, after_key, ":") orelse return null;
    const after_colon = std.mem.trimLeft(u8, after_key[colon_pos + 1 ..], " \t");

    // Find opening quote
    if (after_colon.len == 0 or after_colon[0] != '"') return null;
    const value_start = after_colon[1..];

    // Find closing quote (handle escapes simply)
    var i: usize = 0;
    while (i < value_start.len) : (i += 1) {
        if (value_start[i] == '\\' and i + 1 < value_start.len) {
            i += 1; // Skip escaped char
        } else if (value_start[i] == '"') {
            return value_start[0..i];
        }
    }
    return null;
}

test "basic search" {
    // TODO: Add tests
}
