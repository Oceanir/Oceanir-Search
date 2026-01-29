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
pub const Embedder = struct {
    allocator: std.mem.Allocator,
    session: onnx.OnnxSession,
    tok: tokenizer.Tokenizer,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const home = std.posix.getenv("HOME") orelse "/tmp";

        const model_path = try std.fmt.allocPrint(
            allocator,
            "{s}/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/onnx/model.onnx",
            .{home},
        );
        defer allocator.free(model_path);

        const vocab_path = try std.fmt.allocPrint(
            allocator,
            "{s}/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf/vocab.txt",
            .{home},
        );
        defer allocator.free(vocab_path);

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

test "embedding dimension" {
    const allocator = std.testing.allocator;
    const emb = try embedFallback(allocator, "test text");
    defer allocator.free(emb);
    try std.testing.expectEqual(EMBEDDING_DIM, emb.len);
}
