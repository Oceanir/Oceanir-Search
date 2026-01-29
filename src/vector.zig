const std = @import("std");

/// Compute cosine similarity between two vectors
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len or a.len == 0) return 0;

    var dot_product: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    // SIMD-friendly loop (Zig will auto-vectorize)
    for (a, b) |ai, bi| {
        dot_product += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0) return 0;

    return dot_product / denom;
}

/// Compute euclidean distance between two vectors
pub fn euclideanDistance(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return std.math.inf(f32);

    var sum: f32 = 0;
    for (a, b) |ai, bi| {
        const diff = ai - bi;
        sum += diff * diff;
    }

    return @sqrt(sum);
}

/// Normalize a vector in-place to unit length
pub fn normalize(v: []f32) void {
    var norm: f32 = 0;
    for (v) |vi| {
        norm += vi * vi;
    }
    norm = @sqrt(norm);

    if (norm > 0) {
        for (v) |*vi| {
            vi.* /= norm;
        }
    }
}

/// Dot product of two vectors
pub fn dot(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return 0;

    var sum: f32 = 0;
    for (a, b) |ai, bi| {
        sum += ai * bi;
    }
    return sum;
}

test "cosine similarity" {
    const a = [_]f32{ 1, 0, 0 };
    const b = [_]f32{ 1, 0, 0 };
    const c = [_]f32{ 0, 1, 0 };

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cosineSimilarity(&a, &b), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cosineSimilarity(&a, &c), 0.0001);
}

test "normalize" {
    var v = [_]f32{ 3, 4 };
    normalize(&v);

    try std.testing.expectApproxEqAbs(@as(f32, 0.6), v[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), v[1], 0.0001);
}
