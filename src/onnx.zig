const std = @import("std");
pub const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

/// ONNX Runtime wrapper for embedding inference
pub const OnnxSession = struct {
    allocator: std.mem.Allocator,
    api: *const c.OrtApi,
    env: *c.OrtEnv,
    session: *c.OrtSession,
    memory_info: *c.OrtMemoryInfo,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8) !Self {
        // Get the API
        const api_base = c.OrtGetApiBase();
        const api = api_base.*.GetApi.?(c.ORT_API_VERSION);
        if (api == null) return error.OnnxApiError;

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

        // Use 8 threads for maximum throughput
        _ = api.*.SetIntraOpNumThreads.?(session_options, 8);
        _ = api.*.SetSessionGraphOptimizationLevel.?(session_options, c.ORT_ENABLE_ALL);

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
};

test "onnx session" {
    // Would need model file
}
