const std = @import("std");
const builtin = @import("builtin");
pub const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

/// ONNX Runtime wrapper for embedding inference
pub const OnnxSession = struct {
    allocator: std.mem.Allocator,
    lib: std.DynLib,
    api: *const c.OrtApi,
    env: *c.OrtEnv,
    session: *c.OrtSession,
    memory_info: *c.OrtMemoryInfo,

    const Self = @This();

    const OrtGetApiBaseFn = *const fn () callconv(.c) ?*c.OrtApiBase;
    const AppendCoreMLFn = *const fn (*c.OrtSessionOptions, u32) callconv(.c) ?*c.OrtStatus;

    const OrtRuntime = struct {
        lib: std.DynLib,
        api: *const c.OrtApi,
    };

    fn openOnnxRuntimeLib() !std.DynLib {
        if (std.posix.getenv("ONNXRUNTIME_LIB_PATH")) |path| {
            // path is [:0]const u8, coerce to []const u8 for DynLib.open
            std.debug.print("  Loading ONNX Runtime from: {s}\n", .{path});
            return std.DynLib.open(@as([]const u8, path));
        }

        // Try ~/.oceanir/lib first (ARM64 with CoreML), then system paths
        if (std.posix.getenv("HOME")) |home| {
            var buf: [512]u8 = undefined;
            const oceanir_path = std.fmt.bufPrint(&buf, "{s}/.oceanir/lib/libonnxruntime.dylib", .{home}) catch null;
            if (oceanir_path) |path| {
                if (std.DynLib.open(path)) |lib| {
                    std.debug.print("  Loading ONNX Runtime from: {s}\n", .{path});
                    return lib;
                } else |_| {}
            }
        }

        const candidates = switch (builtin.os.tag) {
            .macos, .tvos, .watchos, .ios, .visionos => &[_][]const u8{
                "/opt/homebrew/lib/libonnxruntime.dylib",
                "/usr/local/lib/libonnxruntime.dylib",
                "/usr/lib/libonnxruntime.dylib",
            },
            .linux => &[_][]const u8{
                "/usr/local/lib/libonnxruntime.so",
                "/usr/lib/libonnxruntime.so",
            },
            else => &[_][]const u8{},
        };

        for (candidates) |path| {
            if (std.DynLib.open(path)) |lib| {
                std.debug.print("  Loading ONNX Runtime from: {s}\n", .{path});
                return lib;
            } else |_| {}
        }

        return error.OnnxRuntimeNotFound;
    }

    fn loadOrtRuntime() !OrtRuntime {
        var lib = try openOnnxRuntimeLib();
        errdefer lib.close();

        const get_api_base = lib.lookup(OrtGetApiBaseFn, "OrtGetApiBase") orelse {
            return error.OnnxSymbolNotFound;
        };

        const api_base = get_api_base() orelse return error.OnnxApiError;
        const api = api_base.*.GetApi.?(c.ORT_API_VERSION);
        if (api == null) return error.OnnxApiError;

        return .{ .lib = lib, .api = api };
    }

    fn tryEnableCoreML(lib: *std.DynLib, api: *const c.OrtApi, session_options: ?*c.OrtSessionOptions) bool {
        // Allow disabling CoreML via env var (OCEANIR_NO_GPU=1)
        if (std.posix.getenv("OCEANIR_NO_GPU")) |v| {
            if (std.mem.eql(u8, v, "1")) {
                return false;
            }
        }

        const append = lib.lookup(AppendCoreMLFn, "OrtSessionOptionsAppendExecutionProvider_CoreML") orelse {
            return false;
        };

        // Use COREML_FLAG_CREATE_MLPROGRAM (0x010) for best performance on Apple Silicon.
        const coreml_flags: u32 = 0x010;
        const status = append(session_options.?, coreml_flags);
        if (status == null) {
            std.debug.print("  [GPU] CoreML/ANE acceleration enabled\n", .{});
            return true;
        }

        api.*.ReleaseStatus.?(status);
        return false;
    }

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
        // Load ONNX Runtime dynamically (avoids link-time dylib parsing issues on ARM)
        var ort = try loadOrtRuntime();
        errdefer ort.lib.close();
        const api = ort.api;

        // Create environment
        var env: ?*c.OrtEnv = null;
        var status = api.*.CreateEnv.?(c.ORT_LOGGING_LEVEL_WARNING, "oceanir", &env);
        if (status != null) {
            api.*.ReleaseStatus.?(status);
            return error.OnnxEnvError;
        }

        // Create session options
        var session_options: ?*c.OrtSessionOptions = null;
        status = api.*.CreateSessionOptions.?(&session_options);
        if (status != null) {
            api.*.ReleaseStatus.?(status);
            api.*.ReleaseEnv.?(env);
            return error.OnnxSessionOptionsError;
        }
        defer api.*.ReleaseSessionOptions.?(session_options);

        // Maximum CPU parallelism - 0 means use all available cores
        _ = api.*.SetIntraOpNumThreads.?(session_options, 0);
        _ = api.*.SetInterOpNumThreads.?(session_options, 0);
        _ = api.*.SetSessionGraphOptimizationLevel.?(session_options, c.ORT_ENABLE_ALL);

        // Try CoreML (if available), otherwise fall back to CPU.
        if (!tryEnableCoreML(&ort.lib, api, session_options)) {
            std.debug.print("  [CPU] Using multi-threaded CPU inference\n", .{});
        }

        // Create session - need null-terminated path
        const path_z = try allocator.dupeZ(u8, model_path);
        defer allocator.free(path_z);

        var session: ?*c.OrtSession = null;
        status = api.*.CreateSession.?(env, path_z.ptr, session_options, &session);
        if (status != null) {
            const msg = api.*.GetErrorMessage.?(status);
            std.debug.print("ONNX Error: {s}\n", .{msg});
            api.*.ReleaseStatus.?(status);
            api.*.ReleaseEnv.?(env);
            return error.OnnxSessionError;
        }

        // Create memory info for CPU
        var memory_info: ?*c.OrtMemoryInfo = null;
        status = api.*.CreateCpuMemoryInfo.?(c.OrtArenaAllocator, c.OrtMemTypeDefault, &memory_info);
        if (status != null) {
            api.*.ReleaseStatus.?(status);
            api.*.ReleaseSession.?(session);
            api.*.ReleaseEnv.?(env);
            return error.OnnxMemoryInfoError;
        }

        return .{
            .allocator = allocator,
            .lib = ort.lib,
            .api = api,
            .env = env.?,
            .session = session.?,
            .memory_info = memory_info.?,
        };
    }

    pub fn deinit(self: *Self) void {
        self.api.*.ReleaseMemoryInfo.?(self.memory_info);
        self.api.*.ReleaseSession.?(self.session);
        self.api.*.ReleaseEnv.?(self.env);
        self.lib.close();
    }

    /// Run inference and get embeddings with mean pooling
    pub fn embed(self: *Self, input_ids: []const i64, attention_mask: []const i64, token_type_ids: []const i64) ![]f32 {
        const batch_size: usize = 1;
        const seq_len: usize = input_ids.len;

        // Input shapes
        const input_shape = [_]i64{ @intCast(batch_size), @intCast(seq_len) };

        // Create input tensors
        var input_ids_tensor: ?*c.OrtValue = null;
        var status = self.api.*.CreateTensorWithDataAsOrtValue.?(
            self.memory_info,
            @constCast(@ptrCast(input_ids.ptr)),
            input_ids.len * @sizeOf(i64),
            &input_shape,
            2,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &input_ids_tensor,
        );
        if (status != null) {
            self.api.*.ReleaseStatus.?(status);
            return error.TensorCreationError;
        }
        defer self.api.*.ReleaseValue.?(input_ids_tensor);

        var attention_mask_tensor: ?*c.OrtValue = null;
        status = self.api.*.CreateTensorWithDataAsOrtValue.?(
            self.memory_info,
            @constCast(@ptrCast(attention_mask.ptr)),
            attention_mask.len * @sizeOf(i64),
            &input_shape,
            2,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &attention_mask_tensor,
        );
        if (status != null) {
            self.api.*.ReleaseStatus.?(status);
            return error.TensorCreationError;
        }
        defer self.api.*.ReleaseValue.?(attention_mask_tensor);

        var token_type_ids_tensor: ?*c.OrtValue = null;
        status = self.api.*.CreateTensorWithDataAsOrtValue.?(
            self.memory_info,
            @constCast(@ptrCast(token_type_ids.ptr)),
            token_type_ids.len * @sizeOf(i64),
            &input_shape,
            2,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &token_type_ids_tensor,
        );
        if (status != null) {
            self.api.*.ReleaseStatus.?(status);
            return error.TensorCreationError;
        }
        defer self.api.*.ReleaseValue.?(token_type_ids_tensor);

        // Input/output names - nomic model outputs last_hidden_state
        const input_names = [_][*c]const u8{ "input_ids", "attention_mask", "token_type_ids" };
        const output_names = [_][*c]const u8{"last_hidden_state"};
        const inputs = [_]?*const c.OrtValue{ input_ids_tensor, attention_mask_tensor, token_type_ids_tensor };

        // Run inference
        var output_tensor: ?*c.OrtValue = null;
        status = self.api.*.Run.?(
            self.session,
            null, // run options
            &input_names,
            &inputs,
            3,
            &output_names,
            1,
            @ptrCast(&output_tensor),
        );
        if (status != null) {
            const msg = self.api.*.GetErrorMessage.?(status);
            std.debug.print("ONNX Run Error: {s}\n", .{msg});
            self.api.*.ReleaseStatus.?(status);
            return error.InferenceError;
        }
        defer self.api.*.ReleaseValue.?(output_tensor);

        // Get output data - shape is [batch, seq_len, hidden_dim]
        var output_data: ?*f32 = null;
        status = self.api.*.GetTensorMutableData.?(output_tensor, @ptrCast(&output_data));
        if (status != null) {
            self.api.*.ReleaseStatus.?(status);
            return error.OutputError;
        }

        // Get output shape
        var type_info: ?*c.OrtTensorTypeAndShapeInfo = null;
        status = self.api.*.GetTensorTypeAndShape.?(output_tensor, &type_info);
        if (status != null) {
            self.api.*.ReleaseStatus.?(status);
            return error.ShapeError;
        }
        defer self.api.*.ReleaseTensorTypeAndShapeInfo.?(type_info);

        var dims_count: usize = 0;
        _ = self.api.*.GetDimensionsCount.?(type_info, &dims_count);

        var dims: [4]i64 = undefined;
        _ = self.api.*.GetDimensions.?(type_info, &dims, dims_count);

        // dims should be [1, seq_len, hidden_dim]
        const output_seq_len: usize = @intCast(dims[1]);
        const hidden_dim: usize = @intCast(dims[2]);

        // Mean pooling over sequence dimension (with attention mask)
        const result = try self.allocator.alloc(f32, hidden_dim);
        @memset(result, 0);

        const src_ptr: [*]f32 = @ptrCast(output_data.?);
        var valid_tokens: f32 = 0;

        for (0..output_seq_len) |s| {
            if (attention_mask[s] == 1) {
                valid_tokens += 1;
                for (0..hidden_dim) |h| {
                    result[h] += src_ptr[s * hidden_dim + h];
                }
            }
        }

        // Divide by number of valid tokens
        if (valid_tokens > 0) {
            for (result) |*v| {
                v.* /= valid_tokens;
            }
        }

        // L2 normalize the embedding
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

    /// Batched embedding - process multiple texts at once for ~10x speedup
    pub fn embedBatch(
        self: *Self,
        batch_input_ids: []const []const i64,
        batch_attention_mask: []const []const i64,
        batch_token_type_ids: []const []const i64,
    ) ![][]f32 {
        const batch_size = batch_input_ids.len;
        if (batch_size == 0) return &[_][]f32{};

        const seq_len = batch_input_ids[0].len;

        // Flatten batch into contiguous arrays
        const total_elements = batch_size * seq_len;
        const flat_input_ids = try self.allocator.alloc(i64, total_elements);
        defer self.allocator.free(flat_input_ids);
        const flat_attention_mask = try self.allocator.alloc(i64, total_elements);
        defer self.allocator.free(flat_attention_mask);
        const flat_token_type_ids = try self.allocator.alloc(i64, total_elements);
        defer self.allocator.free(flat_token_type_ids);

        for (0..batch_size) |b| {
            const offset = b * seq_len;
            @memcpy(flat_input_ids[offset..][0..seq_len], batch_input_ids[b]);
            @memcpy(flat_attention_mask[offset..][0..seq_len], batch_attention_mask[b]);
            @memcpy(flat_token_type_ids[offset..][0..seq_len], batch_token_type_ids[b]);
        }

        // Input shape: [batch_size, seq_len]
        const input_shape = [_]i64{ @intCast(batch_size), @intCast(seq_len) };

        // Create input tensors
        var input_ids_tensor: ?*c.OrtValue = null;
        var status = self.api.*.CreateTensorWithDataAsOrtValue.?(
            self.memory_info,
            @ptrCast(flat_input_ids.ptr),
            total_elements * @sizeOf(i64),
            &input_shape,
            2,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &input_ids_tensor,
        );
        if (status != null) {
            self.api.*.ReleaseStatus.?(status);
            return error.TensorCreationError;
        }
        defer self.api.*.ReleaseValue.?(input_ids_tensor);

        var attention_mask_tensor: ?*c.OrtValue = null;
        status = self.api.*.CreateTensorWithDataAsOrtValue.?(
            self.memory_info,
            @ptrCast(flat_attention_mask.ptr),
            total_elements * @sizeOf(i64),
            &input_shape,
            2,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &attention_mask_tensor,
        );
        if (status != null) {
            self.api.*.ReleaseStatus.?(status);
            return error.TensorCreationError;
        }
        defer self.api.*.ReleaseValue.?(attention_mask_tensor);

        var token_type_ids_tensor: ?*c.OrtValue = null;
        status = self.api.*.CreateTensorWithDataAsOrtValue.?(
            self.memory_info,
            @ptrCast(flat_token_type_ids.ptr),
            total_elements * @sizeOf(i64),
            &input_shape,
            2,
            c.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            &token_type_ids_tensor,
        );
        if (status != null) {
            self.api.*.ReleaseStatus.?(status);
            return error.TensorCreationError;
        }
        defer self.api.*.ReleaseValue.?(token_type_ids_tensor);

        // Run inference
        const input_names = [_][*c]const u8{ "input_ids", "attention_mask", "token_type_ids" };
        const output_names = [_][*c]const u8{"last_hidden_state"};
        const inputs = [_]?*const c.OrtValue{ input_ids_tensor, attention_mask_tensor, token_type_ids_tensor };

        var output_tensor: ?*c.OrtValue = null;
        status = self.api.*.Run.?(
            self.session,
            null,
            &input_names,
            &inputs,
            3,
            &output_names,
            1,
            @ptrCast(&output_tensor),
        );
        if (status != null) {
            const msg = self.api.*.GetErrorMessage.?(status);
            std.debug.print("ONNX Batch Run Error: {s}\n", .{msg});
            self.api.*.ReleaseStatus.?(status);
            return error.InferenceError;
        }
        defer self.api.*.ReleaseValue.?(output_tensor);

        // Get output data - shape is [batch, seq_len, hidden_dim]
        var output_data: ?*f32 = null;
        status = self.api.*.GetTensorMutableData.?(output_tensor, @ptrCast(&output_data));
        if (status != null) {
            self.api.*.ReleaseStatus.?(status);
            return error.OutputError;
        }

        // Get output shape
        var type_info: ?*c.OrtTensorTypeAndShapeInfo = null;
        status = self.api.*.GetTensorTypeAndShape.?(output_tensor, &type_info);
        if (status != null) {
            self.api.*.ReleaseStatus.?(status);
            return error.ShapeError;
        }
        defer self.api.*.ReleaseTensorTypeAndShapeInfo.?(type_info);

        var dims: [4]i64 = undefined;
        var dims_count: usize = 0;
        _ = self.api.*.GetDimensionsCount.?(type_info, &dims_count);
        _ = self.api.*.GetDimensions.?(type_info, &dims, dims_count);

        const output_seq_len: usize = @intCast(dims[1]);
        const hidden_dim: usize = @intCast(dims[2]);

        // Allocate results
        const results = try self.allocator.alloc([]f32, batch_size);
        errdefer {
            for (results) |r| self.allocator.free(r);
            self.allocator.free(results);
        }

        const src_ptr: [*]f32 = @ptrCast(output_data.?);

        // Mean pooling for each item in batch
        for (0..batch_size) |b| {
            results[b] = try self.allocator.alloc(f32, hidden_dim);
            @memset(results[b], 0);

            var valid_tokens: f32 = 0;
            const batch_offset = b * output_seq_len * hidden_dim;
            const mask = batch_attention_mask[b];

            for (0..output_seq_len) |s| {
                if (mask[s] == 1) {
                    valid_tokens += 1;
                    for (0..hidden_dim) |h| {
                        results[b][h] += src_ptr[batch_offset + s * hidden_dim + h];
                    }
                }
            }

            // Normalize
            if (valid_tokens > 0) {
                for (results[b]) |*v| v.* /= valid_tokens;
            }

            // L2 normalize
            var norm: f32 = 0;
            for (results[b]) |v| norm += v * v;
            norm = @sqrt(norm);
            if (norm > 0) {
                for (results[b]) |*v| v.* /= norm;
            }
        }

        return results;
    }
};

test "onnx session" {
    // Would need model file
}
