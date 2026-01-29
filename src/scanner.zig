const std = @import("std");

pub const FileType = enum { text, image };

pub const ScannedFile = struct {
    path: []u8,
    content: []u8, // Empty for images
    file_type: FileType,
};

/// Supported text file extensions
const text_extensions = [_][]const u8{
    ".zig",  ".rs",   ".go",   ".py",   ".js",
    ".ts",   ".tsx",  ".jsx",  ".c",    ".h",
    ".cpp",  ".hpp",  ".java", ".rb",   ".swift",
    ".kt",   ".scala", ".lua", ".sh",   ".md",
    ".json", ".yaml", ".yml",  ".toml",
};

/// Supported image extensions
const image_extensions = [_][]const u8{
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
};

/// Directories to skip
const skip_dirs = [_][]const u8{
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "target",
    "build",
    "dist",
    ".next",
    "__pycache__",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
};

/// Scan a directory for code files
pub fn scan(allocator: std.mem.Allocator, path: []const u8) !std.ArrayListUnmanaged(ScannedFile) {
    var files: std.ArrayListUnmanaged(ScannedFile) = .{};
    errdefer {
        for (files.items) |f| {
            allocator.free(f.path);
            allocator.free(f.content);
        }
        files.deinit(allocator);
    }

    var dir = std.fs.cwd().openDir(path, .{ .iterate = true }) catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print("Directory not found: {s}\n", .{path});
            return files;
        }
        return err;
    };
    defer dir.close();

    try scanDir(allocator, dir, path, &files);

    return files;
}

fn scanDir(
    allocator: std.mem.Allocator,
    dir: std.fs.Dir,
    base_path: []const u8,
    files: *std.ArrayListUnmanaged(ScannedFile),
) !void {
    var walker = dir.iterate();

    while (try walker.next()) |entry| {
        // Skip hidden files
        if (entry.name[0] == '.') continue;

        const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ base_path, entry.name });
        errdefer allocator.free(full_path);

        switch (entry.kind) {
            .directory => {
                // Skip certain directories
                if (shouldSkipDir(entry.name)) {
                    allocator.free(full_path);
                    continue;
                }

                var subdir = dir.openDir(entry.name, .{ .iterate = true }) catch {
                    allocator.free(full_path);
                    continue;
                };
                defer subdir.close();

                try scanDir(allocator, subdir, full_path, files);
                allocator.free(full_path);
            },
            .file => {
                // Check extension
                const file_type = getFileType(entry.name) orelse {
                    allocator.free(full_path);
                    continue;
                };

                if (file_type == .image) {
                    // For images, just store the path (no content)
                    try files.append(allocator, .{
                        .path = full_path,
                        .content = try allocator.alloc(u8, 0),
                        .file_type = .image,
                    });
                } else {
                    // Read text file content
                    const content = dir.readFileAlloc(allocator, entry.name, 10 * 1024 * 1024) catch {
                        allocator.free(full_path);
                        continue;
                    };

                    // Skip binary files (check for null bytes in first 1KB)
                    const check_len = @min(content.len, 1024);
                    var is_binary = false;
                    for (content[0..check_len]) |c| {
                        if (c == 0) {
                            is_binary = true;
                            break;
                        }
                    }

                    if (is_binary) {
                        allocator.free(content);
                        allocator.free(full_path);
                        continue;
                    }

                    try files.append(allocator, .{
                        .path = full_path,
                        .content = content,
                        .file_type = .text,
                    });
                }
            },
            else => {
                allocator.free(full_path);
            },
        }
    }
}

fn shouldSkipDir(name: []const u8) bool {
    for (skip_dirs) |skip| {
        if (std.mem.eql(u8, name, skip)) return true;
    }
    return false;
}

fn getFileType(name: []const u8) ?FileType {
    for (text_extensions) |ext| {
        if (std.mem.endsWith(u8, name, ext)) return .text;
    }
    for (image_extensions) |ext| {
        if (std.mem.endsWith(u8, name, ext)) return .image;
    }
    return null;
}

/// Get file extension
pub fn getExtension(path: []const u8) ?[]const u8 {
    const idx = std.mem.lastIndexOf(u8, path, ".") orelse return null;
    return path[idx..];
}

/// Detect language from file extension
pub fn detectLanguage(path: []const u8) []const u8 {
    const ext = getExtension(path) orelse return "unknown";

    if (std.mem.eql(u8, ext, ".zig")) return "zig";
    if (std.mem.eql(u8, ext, ".rs")) return "rust";
    if (std.mem.eql(u8, ext, ".go")) return "go";
    if (std.mem.eql(u8, ext, ".py")) return "python";
    if (std.mem.eql(u8, ext, ".js")) return "javascript";
    if (std.mem.eql(u8, ext, ".ts")) return "typescript";
    if (std.mem.eql(u8, ext, ".tsx")) return "typescript";
    if (std.mem.eql(u8, ext, ".jsx")) return "javascript";
    if (std.mem.eql(u8, ext, ".c")) return "c";
    if (std.mem.eql(u8, ext, ".h")) return "c";
    if (std.mem.eql(u8, ext, ".cpp")) return "cpp";
    if (std.mem.eql(u8, ext, ".java")) return "java";
    if (std.mem.eql(u8, ext, ".rb")) return "ruby";
    if (std.mem.eql(u8, ext, ".swift")) return "swift";
    if (std.mem.eql(u8, ext, ".kt")) return "kotlin";
    if (std.mem.eql(u8, ext, ".md")) return "markdown";
    if (std.mem.eql(u8, ext, ".json")) return "json";
    if (std.mem.eql(u8, ext, ".yaml") or std.mem.eql(u8, ext, ".yml")) return "yaml";
    if (std.mem.eql(u8, ext, ".toml")) return "toml";

    return "unknown";
}

test "file type detection" {
    try std.testing.expect(getFileType("main.zig") == .text);
    try std.testing.expect(getFileType("lib.rs") == .text);
    try std.testing.expect(getFileType("image.png") == .image);
    try std.testing.expect(getFileType("binary") == null);
}

test "language detection" {
    try std.testing.expectEqualStrings("zig", detectLanguage("src/main.zig"));
    try std.testing.expectEqualStrings("rust", detectLanguage("lib.rs"));
    try std.testing.expectEqualStrings("typescript", detectLanguage("app.tsx"));
}
