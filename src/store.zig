const std = @import("std");

pub const ContentType = enum { text, image };

pub const Chunk = struct {
    id: []u8,
    file_path: []u8,
    content: []u8,
    start_line: usize,
    end_line: usize,
    embedding: []f32,
    content_type: ContentType = .text,

    pub fn deinit(self: *Chunk, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.file_path);
        allocator.free(self.content);
        allocator.free(self.embedding);
    }
};

pub const IndexedFile = struct {
    path: []u8,
    hash: [32]u8,
    chunk_ids: std.ArrayListUnmanaged([]u8),

    pub fn deinit(self: *IndexedFile, allocator: std.mem.Allocator) void {
        allocator.free(self.path);
        for (self.chunk_ids.items) |id| {
            allocator.free(id);
        }
        self.chunk_ids.deinit(allocator);
    }
};

pub const Store = struct {
    allocator: std.mem.Allocator,
    files: std.StringHashMap(IndexedFile),
    chunks: std.StringHashMap(Chunk),
    idf: std.StringHashMap(f32),
    avg_doc_len: f32,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .files = std.StringHashMap(IndexedFile).init(allocator),
            .chunks = std.StringHashMap(Chunk).init(allocator),
            .idf = std.StringHashMap(f32).init(allocator),
            .avg_doc_len = 500,
        };
    }

    pub fn deinit(self: *Self) void {
        // Free chunks
        var chunk_it = self.chunks.iterator();
        while (chunk_it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.chunks.deinit();

        // Free files
        var file_it = self.files.iterator();
        while (file_it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
        }
        self.files.deinit();

        self.idf.deinit();
    }

    pub fn addChunk(self: *Self, chunk: Chunk) !void {
        try self.chunks.put(chunk.id, chunk);
    }

    pub fn updateBm25Stats(self: *Self) void {
        if (self.chunks.count() == 0) return;

        // Compute average document length
        var total_len: usize = 0;
        var chunk_it = self.chunks.iterator();
        while (chunk_it.next()) |entry| {
            total_len += entry.value_ptr.content.len;
        }
        self.avg_doc_len = @as(f32, @floatFromInt(total_len)) / @as(f32, @floatFromInt(self.chunks.count()));

        // TODO: Compute IDF properly
    }

    /// Save store to disk (simplified binary format)
    pub fn save(self: *Self) !void {
        const path = try getStorePath(self.allocator);
        defer self.allocator.free(path);

        // Build buffer with all data
        var buffer: std.ArrayListUnmanaged(u8) = .{};
        defer buffer.deinit(self.allocator);

        // Write header
        try buffer.appendSlice(self.allocator, "SGREP\x00\x01\x00"); // Magic + version

        // Write chunk count
        try appendU32(self.allocator, &buffer, @intCast(self.chunks.count()));

        // Write each chunk
        var chunk_it = self.chunks.iterator();
        while (chunk_it.next()) |entry| {
            const chunk = entry.value_ptr;

            // Write lengths
            try appendU32(self.allocator, &buffer, @intCast(chunk.id.len));
            try appendU32(self.allocator, &buffer, @intCast(chunk.file_path.len));
            try appendU32(self.allocator, &buffer, @intCast(chunk.content.len));
            try appendU32(self.allocator, &buffer, @intCast(chunk.embedding.len));

            // Write data
            try buffer.appendSlice(self.allocator, chunk.id);
            try buffer.appendSlice(self.allocator, chunk.file_path);
            try buffer.appendSlice(self.allocator, chunk.content);
            try appendU32(self.allocator, &buffer, @intCast(chunk.start_line));
            try appendU32(self.allocator, &buffer, @intCast(chunk.end_line));

            // Write embedding
            for (chunk.embedding) |v| {
                try buffer.appendSlice(self.allocator, std.mem.asBytes(&v));
            }
        }

        // Write to file
        const file = try std.fs.createFileAbsolute(path, .{});
        defer file.close();
        try file.writeAll(buffer.items);
    }

    /// Load store from disk
    pub fn load(allocator: std.mem.Allocator) !Self {
        const path = try getStorePath(allocator);
        defer allocator.free(path);

        const file = std.fs.openFileAbsolute(path, .{}) catch |err| {
            if (err == error.FileNotFound) return error.FileNotFound;
            return err;
        };
        defer file.close();

        // Read entire file
        const stat = try file.stat();
        const data = try allocator.alloc(u8, stat.size);
        defer allocator.free(data);
        _ = try file.readAll(data);

        var pos: usize = 0;

        // Read and verify header
        if (data.len < 8) return error.InvalidFormat;
        if (!std.mem.eql(u8, data[0..5], "SGREP")) {
            return error.InvalidFormat;
        }
        pos = 8;

        var self = Self.init(allocator);
        errdefer self.deinit();

        // Read chunk count
        const chunk_count = readU32(data, &pos);

        // Read chunks
        for (0..chunk_count) |_| {
            const id_len = readU32(data, &pos);
            const path_len = readU32(data, &pos);
            const content_len = readU32(data, &pos);
            const embedding_len = readU32(data, &pos);

            const id = try allocator.dupe(u8, data[pos .. pos + id_len]);
            pos += id_len;
            errdefer allocator.free(id);

            const file_path = try allocator.dupe(u8, data[pos .. pos + path_len]);
            pos += path_len;
            errdefer allocator.free(file_path);

            const content = try allocator.dupe(u8, data[pos .. pos + content_len]);
            pos += content_len;
            errdefer allocator.free(content);

            const start_line = readU32(data, &pos);
            const end_line = readU32(data, &pos);

            const embedding = try allocator.alloc(f32, embedding_len);
            errdefer allocator.free(embedding);
            for (embedding) |*v| {
                const bytes = data[pos..][0..4];
                v.* = @bitCast(bytes.*);
                pos += 4;
            }

            try self.chunks.put(id, .{
                .id = id,
                .file_path = file_path,
                .content = content,
                .start_line = start_line,
                .end_line = end_line,
                .embedding = embedding,
            });
        }

        self.updateBm25Stats();
        return self;
    }
};

fn appendU32(allocator: std.mem.Allocator, buffer: *std.ArrayListUnmanaged(u8), value: u32) !void {
    try buffer.appendSlice(allocator, std.mem.asBytes(&value));
}

fn readU32(data: []const u8, pos: *usize) u32 {
    const bytes = data[pos.*..][0..4];
    pos.* += 4;
    return std.mem.readInt(u32, bytes, .little);
}

fn getStorePath(allocator: std.mem.Allocator) ![]u8 {
    // Get home directory
    const home = std.posix.getenv("HOME") orelse "/tmp";

    // Create .oceanir directory if needed (separate from sgrep)
    const oceanir_dir = try std.fmt.allocPrint(allocator, "{s}/.oceanir", .{home});
    defer allocator.free(oceanir_dir);

    std.fs.makeDirAbsolute(oceanir_dir) catch |err| {
        if (err != error.PathAlreadyExists) return err;
    };

    // Generate project-specific store name
    const cwd = std.fs.cwd();
    var cwd_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const cwd_path = try cwd.realpath(".", &cwd_path_buf);

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(cwd_path);
    const hash = hasher.finalResult();

    return try std.fmt.allocPrint(
        allocator,
        "{s}/.oceanir/project_{s}.store.bin",
        .{ home, std.fmt.bytesToHex(hash[0..8], .lower) },
    );
}

test "store roundtrip" {
    const allocator = std.testing.allocator;

    var store = Store.init(allocator);
    defer store.deinit();

    // Add a chunk
    const embedding = try allocator.alloc(f32, 3);
    embedding[0] = 1.0;
    embedding[1] = 2.0;
    embedding[2] = 3.0;

    try store.addChunk(.{
        .id = try allocator.dupe(u8, "test-id"),
        .file_path = try allocator.dupe(u8, "test.txt"),
        .content = try allocator.dupe(u8, "hello world"),
        .start_line = 1,
        .end_line = 5,
        .embedding = embedding,
    });

    try std.testing.expectEqual(@as(usize, 1), store.chunks.count());
}
