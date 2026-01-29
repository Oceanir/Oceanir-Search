const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "oceanir-search",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Link C libraries
    exe.linkLibC();

    // Add stb_image for image loading
    exe.addCSourceFile(.{
        .file = b.path("src/stb_impl.c"),
        .flags = &.{"-O2"},
    });
    exe.addIncludePath(b.path("src"));

    // ONNX Runtime - link from Homebrew installation
    exe.addIncludePath(.{ .cwd_relative = "/usr/local/Cellar/onnxruntime/1.22.2_7/include/onnxruntime" });
    exe.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
    exe.linkSystemLibrary("onnxruntime");

    // Add rpath for runtime library loading
    exe.addRPath(.{ .cwd_relative = "/usr/local/lib" });

    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run Oceanir Search");
    run_step.dependOn(&run_cmd.step);

    // Tests
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    unit_tests.linkLibC();
    unit_tests.addIncludePath(.{ .cwd_relative = "/usr/local/Cellar/onnxruntime/1.22.2_7/include/onnxruntime" });
    unit_tests.addLibraryPath(.{ .cwd_relative = "/usr/local/lib" });
    unit_tests.linkSystemLibrary("onnxruntime");

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
