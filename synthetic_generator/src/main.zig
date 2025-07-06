const rl = @import("raylib");
const std = @import("std");

// Configuration
const SCREEN_WIDTH = 800;
const SCREEN_HEIGHT = 600;
const OUTPUT_DIR = "dataset";
const IMAGES_DIR = "images";
const LABELS_DIR = "labels";
const MAX_SHAPES_PER_SCENE = 10;
const MIN_SHAPE_SIZE = 30;
const MAX_SHAPE_SIZE = 100;

// Shape types
const ShapeType = enum {
    circle,
    rectangle,
    triangle,
};

// Shape structure
const Shape = struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    shape_type: ShapeType,
    color: rl.Color,
    bounding_box: rl.Rectangle,
};

// Scene structure
const Scene = struct {
    shapes: std.ArrayList(Shape),
    background_color: rl.Color,

    pub fn init(allocator: std.mem.Allocator) Scene {
        return Scene{
            .shapes = std.ArrayList(Shape).init(allocator),
            .background_color = rl.Color{ .r = 240, .g = 240, .b = 240, .a = 255 },
        };
    }

    pub fn deinit(self: *Scene) void {
        self.shapes.deinit();
    }

    pub fn addShape(self: *Scene, shape: Shape) !void {
        try self.shapes.append(shape);
    }

    pub fn draw(self: *Scene) void {
        // Draw background
        rl.clearBackground(self.background_color);

        // Draw shapes
        for (self.shapes.items) |shape| {
            switch (shape.shape_type) {
                .circle => {
                    const center_x = @as(i32, @intFromFloat(shape.x + shape.width / 2));
                    const center_y = @as(i32, @intFromFloat(shape.y + shape.height / 2));
                    const radius = @as(i32, @intFromFloat(@min(shape.width, shape.height) / 2));
                    rl.drawCircle(center_x, center_y, @as(f32, @floatFromInt(radius)), shape.color);
                },
                .rectangle => {
                    rl.drawRectangle(@as(i32, @intFromFloat(shape.x)), @as(i32, @intFromFloat(shape.y)), @as(i32, @intFromFloat(shape.width)), @as(i32, @intFromFloat(shape.height)), shape.color);
                },
                .triangle => {
                    const x1 = @as(i32, @intFromFloat(shape.x + shape.width / 2));
                    const y1 = @as(i32, @intFromFloat(shape.y));
                    const x2 = @as(i32, @intFromFloat(shape.x));
                    const y2 = @as(i32, @intFromFloat(shape.y + shape.height));
                    const x3 = @as(i32, @intFromFloat(shape.x + shape.width));
                    const y3 = @as(i32, @intFromFloat(shape.y + shape.height));
                    rl.drawTriangle(rl.Vector2{ .x = @as(f32, @floatFromInt(x1)), .y = @as(f32, @floatFromInt(y1)) }, rl.Vector2{ .x = @as(f32, @floatFromInt(x2)), .y = @as(f32, @floatFromInt(y2)) }, rl.Vector2{ .x = @as(f32, @floatFromInt(x3)), .y = @as(f32, @floatFromInt(y3)) }, shape.color);
                },
            }
        }

        // Draw bounding boxes
        for (self.shapes.items) |shape| {
            rl.drawRectangleLinesEx(shape.bounding_box, 2.0, rl.Color{ .r = 255, .g = 0, .b = 0, .a = 255 });
        }
    }

    fn drawShapes(self: *Scene) void {
        rl.clearBackground(self.background_color);
        for (self.shapes.items) |shape| {
            switch (shape.shape_type) {
                .circle => {
                    const center_x = @as(i32, @intFromFloat(shape.x + shape.width / 2));
                    const center_y = @as(i32, @intFromFloat(shape.y + shape.height / 2));
                    const radius = @as(i32, @intFromFloat(@min(shape.width, shape.height) / 2));
                    rl.drawCircle(center_x, center_y, @as(f32, @floatFromInt(radius)), shape.color);
                },
                .rectangle => {
                    rl.drawRectangle(@as(i32, @intFromFloat(shape.x)), @as(i32, @intFromFloat(shape.y)), @as(i32, @intFromFloat(shape.width)), @as(i32, @intFromFloat(shape.height)), shape.color);
                },
                .triangle => {
                    const x1 = @as(i32, @intFromFloat(shape.x + shape.width / 2));
                    const y1 = @as(i32, @intFromFloat(shape.y));
                    const x2 = @as(i32, @intFromFloat(shape.x));
                    const y2 = @as(i32, @intFromFloat(shape.y + shape.height));
                    const x3 = @as(i32, @intFromFloat(shape.x + shape.width));
                    const y3 = @as(i32, @intFromFloat(shape.y + shape.height));
                    rl.drawTriangle(rl.Vector2{ .x = @as(f32, @floatFromInt(x1)), .y = @as(f32, @floatFromInt(y1)) }, rl.Vector2{ .x = @as(f32, @floatFromInt(x2)), .y = @as(f32, @floatFromInt(y2)) }, rl.Vector2{ .x = @as(f32, @floatFromInt(x3)), .y = @as(f32, @floatFromInt(y3)) }, shape.color);
                },
            }
        }
    }
};

// Random number generator
var prng: std.rand.DefaultPrng = undefined;
var rand: ?*std.rand.Random = null;

fn randomFloat(min: f32, max: f32) f32 {
    const random_value = std.crypto.random.float(f32);
    return min + (max - min) * random_value;
}

fn randomInt(min: i32, max: i32) i32 {
    return std.crypto.random.intRangeAtMost(i32, min, max);
}

fn randomColor() rl.Color {
    return rl.Color{
        .r = @as(u8, @intCast(randomInt(50, 200))),
        .g = @as(u8, @intCast(randomInt(50, 200))),
        .b = @as(u8, @intCast(randomInt(50, 200))),
        .a = 255,
    };
}

fn randomShapeType() ShapeType {
    const types = [_]ShapeType{ .circle, .rectangle, .triangle };
    return types[@as(usize, @intCast(randomInt(0, types.len - 1)))];
}

// Generate a random scene
fn generateScene(allocator: std.mem.Allocator) !Scene {
    var scene = Scene.init(allocator);

    const num_shapes = randomInt(1, MAX_SHAPES_PER_SCENE);

    for (0..@as(usize, @intCast(num_shapes))) |_| {
        const shape_type = randomShapeType();
        const width = randomFloat(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE);
        const height = randomFloat(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE);
        const x = randomFloat(50, SCREEN_WIDTH - width - 50);
        const y = randomFloat(50, SCREEN_HEIGHT - height - 50);

        const shape = Shape{
            .x = x,
            .y = y,
            .width = width,
            .height = height,
            .shape_type = shape_type,
            .color = randomColor(),
            .bounding_box = rl.Rectangle{ .x = x, .y = y, .width = width, .height = height },
        };

        try scene.addShape(shape);
    }

    return scene;
}

// Convert bounding box to YOLO format (normalized coordinates)
fn boundingBoxToYOLO(shape: Shape, image_width: f32, image_height: f32) struct { x_center: f32, y_center: f32, width: f32, height: f32, class_id: u8 } {
    const x_center = (shape.x + shape.width / 2) / image_width;
    const y_center = (shape.y + shape.height / 2) / image_height;
    const width = shape.width / image_width;
    const height = shape.height / image_height;

    // Class ID: 0=circle, 1=rectangle, 2=triangle
    const class_id: u8 = switch (shape.shape_type) {
        .circle => 0,
        .rectangle => 1,
        .triangle => 2,
    };

    return .{
        .x_center = x_center,
        .y_center = y_center,
        .width = width,
        .height = height,
        .class_id = class_id,
    };
}

// Create output directories
fn createOutputDirectories() !void {
    const allocator = std.heap.page_allocator;
    // Create main dataset directory
    try std.fs.cwd().makePath(OUTPUT_DIR);
    // Create images and labels subdirectories
    const images_path = try std.fs.path.join(allocator, &.{ OUTPUT_DIR, IMAGES_DIR });
    defer allocator.free(images_path);
    try std.fs.cwd().makePath(images_path);
    const labels_path = try std.fs.path.join(allocator, &.{ OUTPUT_DIR, LABELS_DIR });
    defer allocator.free(labels_path);
    try std.fs.cwd().makePath(labels_path);
}

// Save screenshot
fn saveScreenshot(filename: []const u8) !void {
    const image = try rl.loadImageFromScreen();
    defer rl.unloadImage(image);

    const path = try std.fmt.allocPrint(std.heap.page_allocator, "{s}/{s}/{s}", .{ OUTPUT_DIR, IMAGES_DIR, filename });
    defer std.heap.page_allocator.free(path);
    var buf: [1024]u8 = undefined;
    const path_len = path.len;
    std.mem.copyForwards(u8, buf[0..path_len], path);
    buf[path_len] = 0;
    const zfull_path: [:0]const u8 = buf[0..path_len :0];
    if (!rl.exportImage(image, zfull_path)) {
        std.debug.print("Failed to export image: {s}\n", .{filename});
    }
}

// Save YOLO format labels
fn saveLabels(scene: *Scene, filename: []const u8) !void {
    const allocator = std.heap.page_allocator;

    const label_filename = try std.fmt.allocPrint(allocator, "{s}.txt", .{filename[0 .. filename.len - 4]} // Remove .png extension
    );
    defer allocator.free(label_filename);

    const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}/{s}", .{ OUTPUT_DIR, LABELS_DIR, label_filename });
    defer allocator.free(full_path);

    const file = try std.fs.cwd().createFile(full_path, .{});
    defer file.close();

    var writer = file.writer();

    for (scene.shapes.items) |shape| {
        const yolo_bbox = boundingBoxToYOLO(shape, @as(f32, @floatFromInt(SCREEN_WIDTH)), @as(f32, @floatFromInt(SCREEN_HEIGHT)));

        try writer.print("{d} {d:.6} {d:.6} {d:.6} {d:.6}\n", .{
            yolo_bbox.class_id,
            yolo_bbox.x_center,
            yolo_bbox.y_center,
            yolo_bbox.width,
            yolo_bbox.height,
        });
    }
}

// Generate a single dataset sample
fn generateSample(allocator: std.mem.Allocator, sample_index: u32) !void {
    // Generate random scene
    var scene = try generateScene(allocator);
    defer scene.deinit();

    // Draw only shapes (no bounding boxes)
    rl.beginDrawing();
    scene.drawShapes();
    rl.endDrawing();
    std.time.sleep(50 * std.time.ns_per_ms);

    // Save screenshot
    const image_filename = try std.fmt.allocPrint(allocator, "sample_{d:0>4}.png", .{sample_index});
    defer allocator.free(image_filename);
    try saveScreenshot(image_filename);

    // Save labels
    try saveLabels(&scene, image_filename);

    // Optionally, draw shapes with bounding boxes for visualization
    rl.beginDrawing();
    scene.draw();
    rl.endDrawing();

    std.debug.print("Generated sample {d}: {s}\n", .{ sample_index, image_filename });
}

pub fn main() anyerror!void {
    const allocator = std.heap.page_allocator;

    // Create output directories
    try createOutputDirectories();

    // Initialize raylib
    rl.initWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Synthetic Data Generator");
    defer rl.closeWindow();

    rl.setTargetFPS(0);

    // Generate dataset samples
    const num_samples = 100;

    std.debug.print("Generating {d} synthetic dataset samples...\n", .{num_samples});

    for (0..num_samples) |i| {
        try generateSample(allocator, @as(u32, @intCast(i)));
    }

    std.debug.print("Dataset generation complete! Check the '{s}' directory.\n", .{OUTPUT_DIR});
    std.debug.print("Images: {s}/{s}/\n", .{ OUTPUT_DIR, IMAGES_DIR });
    std.debug.print("Labels: {s}/{s}/\n", .{ OUTPUT_DIR, LABELS_DIR });

    // Keep window open for a moment to show the last generated scene
    var frame_count: u32 = 0;
    while (!rl.windowShouldClose() and frame_count < 180) {
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(.white);
        rl.drawText("Dataset generation complete!", 200, 250, 30, .black);
        rl.drawText("Check the 'dataset' directory for results", 150, 300, 20, .gray);

        frame_count += 1;
    }
}
