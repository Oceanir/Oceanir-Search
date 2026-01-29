const std = @import("std");

/// BM25 parameters
const k1: f32 = 1.2;
const b_param: f32 = 0.75;

/// Compute BM25 score for a query against a document
pub fn score(
    query: []const u8,
    document: []const u8,
    idf: *const std.StringHashMap(f32),
    avg_doc_len: f32,
) f32 {
    const doc_len: f32 = @floatFromInt(document.len);

    // Tokenize query
    var query_terms = std.mem.tokenizeScalar(u8, query, ' ');

    var total_score: f32 = 0;

    while (query_terms.next()) |term| {
        // Count occurrences of term in document
        const tf = countOccurrences(document, term);
        if (tf == 0) continue;

        // Get IDF (default to 0 if not found)
        const term_idf = idf.get(term) orelse 0;

        // BM25 formula
        const tf_f: f32 = @floatFromInt(tf);
        const numerator = tf_f * (k1 + 1.0);
        const denominator = tf_f + k1 * (1.0 - b_param + b_param * doc_len / avg_doc_len);

        total_score += term_idf * (numerator / denominator);
    }

    return total_score;
}

/// Count occurrences of a substring (case-insensitive)
fn countOccurrences(haystack: []const u8, needle: []const u8) usize {
    if (needle.len == 0 or haystack.len < needle.len) return 0;

    var count: usize = 0;
    var i: usize = 0;

    while (i <= haystack.len - needle.len) {
        if (std.ascii.eqlIgnoreCase(haystack[i..][0..needle.len], needle)) {
            count += 1;
            i += needle.len;
        } else {
            i += 1;
        }
    }

    return count;
}

/// Compute IDF values for a corpus
pub fn computeIdf(
    allocator: std.mem.Allocator,
    documents: []const []const u8,
) !std.StringHashMap(f32) {
    var term_doc_freq = std.StringHashMap(usize).init(allocator);
    defer term_doc_freq.deinit();

    const total_docs: f32 = @floatFromInt(documents.len);

    // Count document frequency for each term
    for (documents) |doc| {
        var seen = std.StringHashMap(void).init(allocator);
        defer seen.deinit();

        var tokens = std.mem.tokenizeScalar(u8, doc, ' ');
        while (tokens.next()) |token| {
            if (seen.contains(token)) continue;
            try seen.put(token, {});

            const entry = try term_doc_freq.getOrPut(token);
            if (!entry.found_existing) {
                entry.value_ptr.* = 0;
            }
            entry.value_ptr.* += 1;
        }
    }

    // Compute IDF
    var idf_map = std.StringHashMap(f32).init(allocator);
    var it = term_doc_freq.iterator();
    while (it.next()) |entry| {
        const df: f32 = @floatFromInt(entry.value_ptr.*);
        const idf_value = @log((total_docs - df + 0.5) / (df + 0.5) + 1.0);
        try idf_map.put(entry.key_ptr.*, idf_value);
    }

    return idf_map;
}

/// Extract code-aware terms (handles CamelCase and snake_case)
pub fn extractCodeTerms(allocator: std.mem.Allocator, text: []const u8) !std.ArrayListUnmanaged([]const u8) {
    var terms: std.ArrayListUnmanaged([]const u8) = .{};

    var i: usize = 0;
    var term_start: usize = 0;

    while (i < text.len) : (i += 1) {
        const c = text[i];

        // Split on whitespace and punctuation
        if (std.ascii.isWhitespace(c) or c == '_' or c == '.' or c == '(' or c == ')') {
            if (i > term_start) {
                try terms.append(allocator, text[term_start..i]);
            }
            term_start = i + 1;
            continue;
        }

        // CamelCase split: lowercase followed by uppercase
        if (i > 0 and i > term_start) {
            const prev = text[i - 1];
            if (std.ascii.isLower(prev) and std.ascii.isUpper(c)) {
                try terms.append(allocator, text[term_start..i]);
                term_start = i;
            }
        }
    }

    // Add final term
    if (text.len > term_start) {
        try terms.append(allocator, text[term_start..]);
    }

    return terms;
}

test "count occurrences" {
    try std.testing.expectEqual(@as(usize, 2), countOccurrences("hello world hello", "hello"));
    try std.testing.expectEqual(@as(usize, 0), countOccurrences("hello world", "foo"));
}

test "extract code terms" {
    const allocator = std.testing.allocator;
    var terms = try extractCodeTerms(allocator, "getUserName");
    defer terms.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), terms.items.len);
    try std.testing.expectEqualStrings("get", terms.items[0]);
    try std.testing.expectEqualStrings("User", terms.items[1]);
    try std.testing.expectEqualStrings("Name", terms.items[2]);
}
