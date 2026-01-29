const std = @import("std");

/// WordPiece tokenizer for BERT-style models
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: std.StringHashMap(i64),
    id_to_token: std.AutoHashMap(i64, []const u8),
    unk_token_id: i64,
    cls_token_id: i64,
    sep_token_id: i64,
    pad_token_id: i64,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, vocab_path: []const u8) !Self {
        var self = Self{
            .allocator = allocator,
            .vocab = std.StringHashMap(i64).init(allocator),
            .id_to_token = std.AutoHashMap(i64, []const u8).init(allocator),
            .unk_token_id = 100,
            .cls_token_id = 101,
            .sep_token_id = 102,
            .pad_token_id = 0,
        };

        // Load vocabulary
        const file = try std.fs.openFileAbsolute(vocab_path, .{});
        defer file.close();

        const content = try file.readToEndAlloc(allocator, 10 * 1024 * 1024);
        defer allocator.free(content);

        var line_num: i64 = 0;
        var lines = std.mem.splitScalar(u8, content, '\n');
        while (lines.next()) |line| {
            if (line.len == 0) continue;
            const token = try allocator.dupe(u8, line);
            try self.vocab.put(token, line_num);
            try self.id_to_token.put(line_num, token);
            line_num += 1;
        }

        // Get special token IDs
        if (self.vocab.get("[UNK]")) |id| self.unk_token_id = id;
        if (self.vocab.get("[CLS]")) |id| self.cls_token_id = id;
        if (self.vocab.get("[SEP]")) |id| self.sep_token_id = id;
        if (self.vocab.get("[PAD]")) |id| self.pad_token_id = id;

        return self;
    }

    pub fn deinit(self: *Self) void {
        var it = self.id_to_token.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.vocab.deinit();
        self.id_to_token.deinit();
    }

    /// Tokenize text into token IDs
    pub fn encode(self: *Self, text: []const u8, max_length: usize) !TokenizedInput {
        var input_ids = try self.allocator.alloc(i64, max_length);
        var attention_mask = try self.allocator.alloc(i64, max_length);
        const token_type_ids = try self.allocator.alloc(i64, max_length);

        // Initialize with padding
        @memset(input_ids, self.pad_token_id);
        @memset(attention_mask, 0);
        @memset(token_type_ids, 0);

        // Add [CLS] token
        var pos: usize = 0;
        input_ids[pos] = self.cls_token_id;
        attention_mask[pos] = 1;
        pos += 1;

        // Tokenize text using WordPiece
        var words = std.mem.tokenizeAny(u8, text, " \t\n\r");
        while (words.next()) |word| {
            if (pos >= max_length - 1) break; // Leave room for [SEP]

            const word_tokens = try self.tokenizeWord(word);
            defer self.allocator.free(word_tokens);

            for (word_tokens) |token_id| {
                if (pos >= max_length - 1) break;
                input_ids[pos] = token_id;
                attention_mask[pos] = 1;
                pos += 1;
            }
        }

        // Add [SEP] token
        if (pos < max_length) {
            input_ids[pos] = self.sep_token_id;
            attention_mask[pos] = 1;
            pos += 1;
        }

        return .{
            .input_ids = input_ids,
            .attention_mask = attention_mask,
            .token_type_ids = token_type_ids,
            .length = pos,
        };
    }

    /// WordPiece tokenization of a single word
    fn tokenizeWord(self: *Self, word: []const u8) ![]i64 {
        var tokens: std.ArrayListUnmanaged(i64) = .{};

        // Lowercase the word
        var lower_buf: [256]u8 = undefined;
        const lower_len = @min(word.len, 255);
        for (word[0..lower_len], 0..) |c, i| {
            lower_buf[i] = std.ascii.toLower(c);
        }
        const lower_word = lower_buf[0..lower_len];

        // Try whole word first
        if (self.vocab.get(lower_word)) |id| {
            try tokens.append(self.allocator, id);
            return try tokens.toOwnedSlice(self.allocator);
        }

        // WordPiece: break into subwords
        var start: usize = 0;
        while (start < lower_word.len) {
            var end = lower_word.len;
            var found = false;

            while (start < end) {
                var subword: []const u8 = undefined;
                if (start > 0) {
                    // Add ## prefix for continuation
                    var buf: [258]u8 = undefined;
                    buf[0] = '#';
                    buf[1] = '#';
                    const sub_len = end - start;
                    @memcpy(buf[2 .. 2 + sub_len], lower_word[start..end]);
                    subword = buf[0 .. 2 + sub_len];
                } else {
                    subword = lower_word[start..end];
                }

                if (self.vocab.get(subword)) |id| {
                    try tokens.append(self.allocator, id);
                    found = true;
                    start = end;
                    break;
                }
                end -= 1;
            }

            if (!found) {
                // Unknown token
                try tokens.append(self.allocator, self.unk_token_id);
                start += 1;
            }
        }

        return try tokens.toOwnedSlice(self.allocator);
    }
};

pub const TokenizedInput = struct {
    input_ids: []i64,
    attention_mask: []i64,
    token_type_ids: []i64,
    length: usize,

    pub fn deinit(self: *TokenizedInput, allocator: std.mem.Allocator) void {
        allocator.free(self.input_ids);
        allocator.free(self.attention_mask);
        allocator.free(self.token_type_ids);
    }
};

test "tokenizer basic" {
    // Test would require vocab file
}
