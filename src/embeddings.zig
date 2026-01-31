const std = @import("std");
const onnx = @import("onnx.zig");
const tokenizer = @import("tokenizer.zig");

// ============================================================================
// Oceanir Search - Neural Embeddings
// ============================================================================
// MiniLM-L6-v2 (22M params) for fast, high-quality text embeddings.
// Falls back to SimHash if ONNX unavailable.
// ============================================================================

/// Embedding dimension (384 for MiniLM)
pub const EMBEDDING_DIM: usize = 384;

/// Maximum sequence length for tokenizer
const MAX_SEQ_LEN: usize = 512;

/// Global embedder (initialized lazily)
var global_embedder: ?*Embedder = null;

// ============================================================================
// Neural Embedder (MiniLM - 22M params, fast)
// ============================================================================
/// Model URLs for auto-download
const MODEL_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx";
const VOCAB_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt";

// Fast model (smaller, 2x faster) - use OCEANIR_FAST=1 env var
const FAST_MODEL_URL = "https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/resolve/main/onnx/model.onnx";
const FAST_VOCAB_URL = "https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/resolve/main/vocab.txt";

fn useFastModel() bool {
    const env = std.posix.getenv("OCEANIR_FAST");
    return env != null and std.mem.eql(u8, env.?, "1");
}

pub const Embedder = struct {
    allocator: std.mem.Allocator,
    session: onnx.OnnxSession,
    tok: tokenizer.Tokenizer,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const home = std.posix.getenv("HOME") orelse "/tmp";

        // Check for fast mode
        const fast = useFastModel();
        const model_dir = if (fast) "models-fast" else "models";

        // Try paths in order: ~/.oceanir/models[-fast], then HuggingFace cache
        const oceanir_model = try std.fmt.allocPrint(allocator, "{s}/.oceanir/{s}/model.onnx", .{ home, model_dir });
        defer allocator.free(oceanir_model);
        const oceanir_vocab = try std.fmt.allocPrint(allocator, "{s}/.oceanir/{s}/vocab.txt", .{ home, model_dir });
        defer allocator.free(oceanir_vocab);

        if (fast) {
            std.debug.print("  [FAST MODE] Using paraphrase-MiniLM-L3-v2\n", .{});
        }

        const hf_model = try std.fmt.allocPrint(
            allocator,
            "{s}/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/onnx/model.onnx",
            .{home},
        );
        defer allocator.free(hf_model);
        const hf_vocab = try std.fmt.allocPrint(
            allocator,
            "{s}/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/vocab.txt",
            .{home},
        );
        defer allocator.free(hf_vocab);

        // Check which path exists
        var model_path: []const u8 = undefined;
        var vocab_path: []const u8 = undefined;

        if (fileExists(oceanir_model)) {
            model_path = oceanir_model;
            vocab_path = oceanir_vocab;
        } else if (fileExists(hf_model)) {
            model_path = hf_model;
            vocab_path = hf_vocab;
        } else {
            // Model not found - print download instructions
            const model_name = if (fast) "paraphrase-MiniLM-L3-v2 (fast)" else "all-MiniLM-L6-v2";
            const model_size = if (fast) "~66MB" else "~90MB";
            std.debug.print("\n", .{});
            std.debug.print("╔════════════════════════════════════════════════════════════╗\n", .{});
            std.debug.print("║  Oceanir Search - First Run Setup                          ║\n", .{});
            std.debug.print("╠════════════════════════════════════════════════════════════╣\n", .{});
            std.debug.print("║  Downloading embedding model ({s}, one-time)...       ║\n", .{model_size});
            std.debug.print("║  Model: {s}                      ║\n", .{model_name});
            std.debug.print("╚════════════════════════════════════════════════════════════╝\n", .{});
            std.debug.print("\n", .{});

            // Create directory and download
            try downloadModels(allocator, home, fast);
            model_path = oceanir_model;
            vocab_path = oceanir_vocab;
        }

        self.* = .{
            .allocator = allocator,
            .session = try onnx.OnnxSession.init(allocator, model_path),
            .tok = try tokenizer.Tokenizer.init(allocator, vocab_path),
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.session.deinit();
        self.tok.deinit();
        self.allocator.destroy(self);
    }

    pub fn embed(self: *Self, text: []const u8) ![]f32 {
        var tokens = try self.tok.encode(text, MAX_SEQ_LEN);
        defer tokens.deinit(self.allocator);
        return try self.session.embed(tokens.input_ids, tokens.attention_mask, tokens.token_type_ids);
    }

    /// Batch embed multiple texts at once (faster than individual calls)
    pub fn embedBatch(self: *Self, texts: []const []const u8) ![][]f32 {
        if (texts.len == 0) return &[_][]f32{};

        // Tokenize all texts (serial - tokenizer not thread-safe)
        var all_tokens = try self.allocator.alloc(tokenizer.TokenizedInput, texts.len);
        defer {
            for (all_tokens) |*t| t.deinit(self.allocator);
            self.allocator.free(all_tokens);
        }

        // Find max length for padding
        var max_len: usize = 0;
        for (texts, 0..) |text, i| {
            all_tokens[i] = try self.tok.encode(text, MAX_SEQ_LEN);
            max_len = @max(max_len, all_tokens[i].input_ids.len);
        }

        // Pad all sequences to max_len
        var batch_input_ids = try self.allocator.alloc([]const i64, texts.len);
        defer self.allocator.free(batch_input_ids);
        var batch_attention_mask = try self.allocator.alloc([]const i64, texts.len);
        defer self.allocator.free(batch_attention_mask);
        var batch_token_type_ids = try self.allocator.alloc([]const i64, texts.len);
        defer self.allocator.free(batch_token_type_ids);

        var padded_buffers = try self.allocator.alloc([]i64, texts.len * 3);
        defer {
            for (padded_buffers) |buf| self.allocator.free(buf);
            self.allocator.free(padded_buffers);
        }

        for (0..texts.len) |i| {
            const tokens_i = &all_tokens[i];

            // Pad input_ids with 0
            padded_buffers[i * 3] = try self.allocator.alloc(i64, max_len);
            @memcpy(padded_buffers[i * 3][0..tokens_i.input_ids.len], tokens_i.input_ids);
            @memset(padded_buffers[i * 3][tokens_i.input_ids.len..], 0);
            batch_input_ids[i] = padded_buffers[i * 3];

            // Pad attention_mask with 0
            padded_buffers[i * 3 + 1] = try self.allocator.alloc(i64, max_len);
            @memcpy(padded_buffers[i * 3 + 1][0..tokens_i.attention_mask.len], tokens_i.attention_mask);
            @memset(padded_buffers[i * 3 + 1][tokens_i.attention_mask.len..], 0);
            batch_attention_mask[i] = padded_buffers[i * 3 + 1];

            // Pad token_type_ids with 0
            padded_buffers[i * 3 + 2] = try self.allocator.alloc(i64, max_len);
            @memcpy(padded_buffers[i * 3 + 2][0..tokens_i.token_type_ids.len], tokens_i.token_type_ids);
            @memset(padded_buffers[i * 3 + 2][tokens_i.token_type_ids.len..], 0);
            batch_token_type_ids[i] = padded_buffers[i * 3 + 2];
        }

        return try self.session.embedBatch(batch_input_ids, batch_attention_mask, batch_token_type_ids);
    }
};

// ============================================================================
// Public API
// ============================================================================

/// Generate embedding for text
pub fn embed(allocator: std.mem.Allocator, text: []const u8) ![]f32 {
    if (global_embedder == null) {
        global_embedder = try Embedder.init(allocator);
    }
    return global_embedder.?.embed(text);
}

/// Generate embedding for query (same as embed for MiniLM)
pub fn embedQuery(allocator: std.mem.Allocator, text: []const u8) ![]f32 {
    return embed(allocator, text);
}

/// Batch embed multiple texts (~10x faster for indexing)
pub fn embedBatch(allocator: std.mem.Allocator, texts: []const []const u8) ![][]f32 {
    if (global_embedder == null) {
        global_embedder = try Embedder.init(allocator);
    }
    return global_embedder.?.embedBatch(texts);
}

/// Cleanup embedder (call before exit)
pub fn deinitGlobal() void {
    if (global_embedder) |e| {
        e.deinit();
        global_embedder = null;
    }
}

/// Fallback SimHash embedding (used if ONNX unavailable)
fn embedFallback(allocator: std.mem.Allocator, text: []const u8) ![]f32 {
    const embedding = try allocator.alloc(f32, EMBEDDING_DIM);
    @memset(embedding, 0);

    if (text.len == 0) {
        normalizeOrUniform(embedding);
        return embedding;
    }

    const NGRAM_SIZES = [_]usize{ 2, 3, 4, 5 };
    for (NGRAM_SIZES) |n| {
        if (text.len < n) continue;
        var i: usize = 0;
        while (i <= text.len - n) : (i += 1) {
            const ngram = text[i .. i + n];
            var hasher = std.hash.XxHash64.init(0);
            hasher.update(ngram);
            const idx = hasher.final() % EMBEDDING_DIM;
            embedding[idx] += @floatFromInt(n);
        }
    }

    for (embedding) |*v| {
        v.* = std.math.tanh(v.* * 0.1);
    }
    normalizeOrUniform(embedding);

    return embedding;
}

fn normalizeOrUniform(v: []f32) void {
    var norm: f32 = 0;
    for (v) |val| {
        norm += val * val;
    }
    norm = @sqrt(norm);

    if (norm > 0.0001) {
        for (v) |*val| {
            val.* /= norm;
        }
    } else {
        const uniform: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(v.len)));
        for (v) |*val| {
            val.* = uniform;
        }
    }
}

// ============================================================================
// Model Download Helpers
// ============================================================================

fn fileExists(path: []const u8) bool {
    std.fs.cwd().access(path, .{}) catch return false;
    return true;
}

fn downloadModels(allocator: std.mem.Allocator, home: []const u8, fast: bool) !void {
    const dir_name = if (fast) "models-fast" else "models";
    const models_dir = try std.fmt.allocPrint(allocator, "{s}/.oceanir/{s}", .{ home, dir_name });
    defer allocator.free(models_dir);

    std.fs.cwd().makePath(models_dir) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };

    const model_path = try std.fmt.allocPrint(allocator, "{s}/model.onnx", .{models_dir});
    defer allocator.free(model_path);
    const vocab_path = try std.fmt.allocPrint(allocator, "{s}/vocab.txt", .{models_dir});
    defer allocator.free(vocab_path);

    const model_url = if (fast) FAST_MODEL_URL else MODEL_URL;
    const vocab_url = if (fast) FAST_VOCAB_URL else VOCAB_URL;

    std.debug.print("  Downloading model.onnx...\n", .{});
    _ = try runCurl(allocator, model_url, model_path);

    std.debug.print("  Downloading vocab.txt...\n", .{});
    _ = try runCurl(allocator, vocab_url, vocab_path);

    std.debug.print("  ✓ Models downloaded to {s}\n\n", .{models_dir});
}

fn runCurl(allocator: std.mem.Allocator, url: []const u8, output: []const u8) !void {
    const url_z = try allocator.dupeZ(u8, url);
    defer allocator.free(url_z);
    const output_z = try allocator.dupeZ(u8, output);
    defer allocator.free(output_z);

    var child = std.process.Child.init(&[_][]const u8{
        "curl", "-fsSL", "-o", output_z, url_z,
    }, allocator);
    child.stderr_behavior = .Inherit;
    child.stdout_behavior = .Inherit;

    _ = try child.spawnAndWait();
}

test "embedding dimension" {
    const allocator = std.testing.allocator;
    const emb = try embedFallback(allocator, "test text");
    defer allocator.free(emb);
    try std.testing.expectEqual(EMBEDDING_DIM, emb.len);
}
