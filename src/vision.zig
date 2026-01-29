const std = @import("std");
const onnx = @import("onnx.zig");

const c = @cImport({
    @cInclude("stb_image.h");
});

// ============================================================================
// Oceanir Search - Vision Embeddings (CLIP)
// ============================================================================
// Uses CLIP ViT-base (150M params) for image embeddings.
// ============================================================================

/// CLIP embedding dimension
pub const CLIP_DIM: usize = 512;

/// CLIP input size
const CLIP_SIZE: usize = 224;

/// Supported image extensions
pub const IMAGE_EXTENSIONS = [_][]const u8{ ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp" };

/// Check if a file is an image
pub fn isImage(path: []const u8) bool {
    const lower_path = blk: {
        var buf: [1024]u8 = undefined;
        const len = @min(path.len, 1024);
        for (path[0..len], 0..) |ch, i| {
            buf[i] = std.ascii.toLower(ch);
        }
        break :blk buf[0..len];
    };

    for (IMAGE_EXTENSIONS) |ext| {
        if (std.mem.endsWith(u8, lower_path, ext)) {
            return true;
        }
    }
    return false;
}

/// Global CLIP embedder
var clip_embedder: ?*ClipEmbedder = null;

/// CLIP Vision Embedder
pub const ClipEmbedder = struct {
    allocator: std.mem.Allocator,
    vision_session: onnx.OnnxSession,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const home = std.posix.getenv("HOME") orelse "/tmp";

        // CLIP vision encoder (fp16 for good speed/quality balance)
        const vision_path = try std.fmt.allocPrint(
            allocator,
            "{s}/.cache/oceanir/clip/onnx/vision_model_fp16.onnx",
            .{home},
        );
        defer allocator.free(vision_path);

        self.* = .{
            .allocator = allocator,
            .vision_session = onnx.OnnxSession.init(allocator, vision_path) catch {
                std.debug.print("CLIP model not found. Run: oceanir-search download-models\n", .{});
                return error.ClipModelNotFound;
            },
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.vision_session.deinit();
        self.allocator.destroy(self);
    }

    /// Embed an image file
    pub fn embedImage(self: *Self, image_path: []const u8) ![]f32 {
        // Load and preprocess image
        const pixels = try loadAndPreprocessImage(self.allocator, image_path);
        defer self.allocator.free(pixels);

        // Run vision encoder
        return try self.runVisionEncoder(pixels);
    }

    fn runVisionEncoder(self: *Self, pixels: []const f32) ![]f32 {
        // CLIP vision expects pixel_values: [batch, channels, height, width] = [1, 3, 224, 224]
        const batch_size: usize = 1;
        const channels: usize = 3;
        const height: usize = CLIP_SIZE;
        const width: usize = CLIP_SIZE;

        const input_shape = [_]i64{
            @intCast(batch_size),
            @intCast(channels),
            @intCast(height),
            @intCast(width),
        };

        // Create input tensor
        var input_tensor: ?*onnx.c.OrtValue = null;
        var status = self.vision_session.api.*.CreateTensorWithDataAsOrtValue.?(
            self.vision_session.memory_info,
            @constCast(@ptrCast(pixels.ptr)),
            pixels.len * @sizeOf(f32),
            &input_shape,
            4,
            onnx.c.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensor,
        );
        if (status != null) {
            self.vision_session.api.*.ReleaseStatus.?(status);
            return error.TensorCreationError;
        }
        defer self.vision_session.api.*.ReleaseValue.?(input_tensor);

        // Input/output names for CLIP vision
        const input_names = [_][*c]const u8{"pixel_values"};
        const output_names = [_][*c]const u8{"image_embeds"};
        const inputs = [_]?*const onnx.c.OrtValue{input_tensor};

        // Run inference
        var output_tensor: ?*onnx.c.OrtValue = null;
        status = self.vision_session.api.*.Run.?(
            self.vision_session.session,
            null,
            &input_names,
            &inputs,
            1,
            &output_names,
            1,
            @ptrCast(&output_tensor),
        );
        if (status != null) {
            const msg = self.vision_session.api.*.GetErrorMessage.?(status);
            std.debug.print("CLIP Run Error: {s}\n", .{msg});
            self.vision_session.api.*.ReleaseStatus.?(status);
            return error.InferenceError;
        }
        defer self.vision_session.api.*.ReleaseValue.?(output_tensor);

        // Get output data
        var output_data: ?*f32 = null;
        status = self.vision_session.api.*.GetTensorMutableData.?(output_tensor, @ptrCast(&output_data));
        if (status != null) {
            self.vision_session.api.*.ReleaseStatus.?(status);
            return error.OutputError;
        }

        // Copy and normalize embedding
        const result = try self.allocator.alloc(f32, CLIP_DIM);
        const src_ptr: [*]f32 = @ptrCast(output_data.?);

        // Copy
        for (0..CLIP_DIM) |i| {
            result[i] = src_ptr[i];
        }

        // L2 normalize
        var norm: f32 = 0;
        for (result) |v| {
            norm += v * v;
        }
        norm = @sqrt(norm);
        if (norm > 0) {
            for (result) |*v| {
                v.* /= norm;
            }
        }

        return result;
    }
};

/// Load and preprocess image to CLIP format
fn loadAndPreprocessImage(allocator: std.mem.Allocator, path: []const u8) ![]f32 {
    // Need null-terminated path for C
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    // Load image with stb_image
    var width: c_int = 0;
    var height: c_int = 0;
    var channels: c_int = 0;

    const img_data = c.stbi_load(path_z.ptr, &width, &height, &channels, 3);
    if (img_data == null) {
        return error.ImageLoadFailed;
    }
    defer c.stbi_image_free(img_data);

    // Allocate output tensor [1, 3, 224, 224]
    const output_size = 1 * 3 * CLIP_SIZE * CLIP_SIZE;
    const pixels = try allocator.alloc(f32, output_size);
    errdefer allocator.free(pixels);

    // CLIP normalization constants
    const mean = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
    const std_dev = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

    // Resize and normalize to CLIP format
    // Simple bilinear resize
    const w: usize = @intCast(width);
    const h: usize = @intCast(height);

    for (0..CLIP_SIZE) |y| {
        for (0..CLIP_SIZE) |x| {
            // Map to source coordinates
            const src_x = (x * w) / CLIP_SIZE;
            const src_y = (y * h) / CLIP_SIZE;
            const src_idx = (src_y * w + src_x) * 3;

            // Get RGB values and normalize
            for (0..3) |ch| {
                const pixel_val: f32 = @as(f32, @floatFromInt(img_data[src_idx + ch])) / 255.0;
                const normalized = (pixel_val - mean[ch]) / std_dev[ch];

                // Output format: [batch, channel, height, width]
                const out_idx = ch * CLIP_SIZE * CLIP_SIZE + y * CLIP_SIZE + x;
                pixels[out_idx] = normalized;
            }
        }
    }

    return pixels;
}

/// Get or create CLIP embedder
pub fn getClipEmbedder(allocator: std.mem.Allocator) !*ClipEmbedder {
    if (clip_embedder) |e| {
        return e;
    }
    clip_embedder = try ClipEmbedder.init(allocator);
    return clip_embedder.?;
}

/// Cleanup CLIP embedder
pub fn deinitGlobal() void {
    if (clip_embedder) |e| {
        e.deinit();
        clip_embedder = null;
    }
}

/// Embed an image file
pub fn embedImage(allocator: std.mem.Allocator, path: []const u8) ![]f32 {
    const embedder = try getClipEmbedder(allocator);
    return embedder.embedImage(path);
}

test "is image" {
    try std.testing.expect(isImage("photo.png"));
    try std.testing.expect(isImage("PHOTO.PNG"));
    try std.testing.expect(isImage("test.jpg"));
    try std.testing.expect(!isImage("code.zig"));
    try std.testing.expect(!isImage("readme.md"));
}
