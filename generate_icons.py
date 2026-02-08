"""
Generate PWA app icons for CropAI.
Creates 192x192 and 512x512 PNG icons with a green gradient rounded rectangle
and a white leaf/plant symbol.
"""

from PIL import Image, ImageDraw, ImageFilter
import math


def create_gradient(size, color_top, color_bottom):
    """Create a vertical gradient image."""
    img = Image.new("RGB", (size, size))
    pixels = img.load()
    for y in range(size):
        ratio = y / (size - 1)
        r = int(color_top[0] + (color_bottom[0] - color_top[0]) * ratio)
        g = int(color_top[1] + (color_bottom[1] - color_top[1]) * ratio)
        b = int(color_top[2] + (color_bottom[2] - color_top[2]) * ratio)
        for x in range(size):
            pixels[x, y] = (r, g, b)
    return img


def create_rounded_mask(size, radius):
    """Create a rounded rectangle mask."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    return mask


def draw_leaf(draw, cx, cy, scale, fill="white"):
    """
    Draw a stylized leaf/plant shape centered at (cx, cy).
    The shape consists of a main leaf, a smaller side leaf, and a stem.
    """

    # --- Main large leaf (tilted slightly left) ---
    leaf_points = []
    # Parametric leaf: goes up from stem, curves out, comes to a tip
    # Using a custom Bezier-like set of points
    num_pts = 60

    # Leaf dimensions
    leaf_h = 0.38 * scale   # height of the main leaf
    leaf_w = 0.18 * scale   # max half-width

    # Leaf tip at top
    tip_x = cx - 0.02 * scale
    tip_y = cy - 0.32 * scale

    # Leaf base (where it meets the stem)
    base_x = cx
    base_y = cy + 0.06 * scale

    # Build right side of leaf (from base up to tip)
    right_side = []
    for i in range(num_pts + 1):
        t = i / num_pts  # 0 = base, 1 = tip
        # y goes from base_y to tip_y
        y = base_y + (tip_y - base_y) * t
        # width bulge: sin curve, max at ~0.35 of the way up
        bulge = math.sin(t * math.pi) * (1 - t * 0.3)
        # slight x drift toward tip
        x_center = base_x + (tip_x - base_x) * t
        x = x_center + leaf_w * bulge
        right_side.append((x, y))

    # Left side is mirror (reflected around the center line from base to tip)
    left_side = []
    for i in range(num_pts + 1):
        t = i / num_pts
        y = base_y + (tip_y - base_y) * t
        bulge = math.sin(t * math.pi) * (1 - t * 0.3)
        x_center = base_x + (tip_x - base_x) * t
        x = x_center - leaf_w * bulge
        left_side.append((x, y))

    # Combine: right side base->tip, then left side tip->base
    leaf_points = right_side + left_side[::-1]
    draw.polygon(leaf_points, fill=fill)

    # --- Leaf vein (center line) ---
    vein_pts = []
    for i in range(num_pts + 1):
        t = i / num_pts
        y = base_y + (tip_y - base_y) * t
        x = base_x + (tip_x - base_x) * t
        vein_pts.append((x, y))
    vein_width = max(1, int(0.008 * scale))
    for i in range(len(vein_pts) - 1):
        draw.line([vein_pts[i], vein_pts[i + 1]], fill="#4a7c23", width=vein_width)

    # --- Small secondary leaf on the right ---
    small_leaf_pts = []
    sl_base_x = cx + 0.01 * scale
    sl_base_y = cy + 0.02 * scale
    sl_tip_x = cx + 0.18 * scale
    sl_tip_y = cy - 0.08 * scale
    sl_h = 0.18 * scale
    sl_w = 0.07 * scale
    num_sl = 40

    right_sl = []
    for i in range(num_sl + 1):
        t = i / num_sl
        x = sl_base_x + (sl_tip_x - sl_base_x) * t
        y = sl_base_y + (sl_tip_y - sl_base_y) * t
        bulge = math.sin(t * math.pi) * (1 - t * 0.4)
        # Perpendicular offset (rotate 90 deg from direction)
        dx = sl_tip_x - sl_base_x
        dy = sl_tip_y - sl_base_y
        length = math.sqrt(dx * dx + dy * dy)
        nx = -dy / length  # normal x
        ny = dx / length   # normal y
        right_sl.append((x + nx * sl_w * bulge, y + ny * sl_w * bulge))

    left_sl = []
    for i in range(num_sl + 1):
        t = i / num_sl
        x = sl_base_x + (sl_tip_x - sl_base_x) * t
        y = sl_base_y + (sl_tip_y - sl_base_y) * t
        bulge = math.sin(t * math.pi) * (1 - t * 0.4)
        dx = sl_tip_x - sl_base_x
        dy = sl_tip_y - sl_base_y
        length = math.sqrt(dx * dx + dy * dy)
        nx = -dy / length
        ny = dx / length
        left_sl.append((x - nx * sl_w * bulge, y - ny * sl_w * bulge))

    small_leaf_points = right_sl + left_sl[::-1]
    draw.polygon(small_leaf_points, fill=fill)

    # Small leaf vein
    for i in range(num_sl):
        t1 = i / num_sl
        t2 = (i + 1) / num_sl
        x1 = sl_base_x + (sl_tip_x - sl_base_x) * t1
        y1 = sl_base_y + (sl_tip_y - sl_base_y) * t1
        x2 = sl_base_x + (sl_tip_x - sl_base_x) * t2
        y2 = sl_base_y + (sl_tip_y - sl_base_y) * t2
        draw.line([(x1, y1), (x2, y2)], fill="#4a7c23", width=max(1, vein_width // 2))

    # --- Stem ---
    stem_top_y = base_y
    stem_bot_y = cy + 0.30 * scale
    stem_width = max(2, int(0.02 * scale))

    # Slightly curved stem using multiple line segments
    stem_pts = []
    for i in range(30):
        t = i / 29
        y = stem_top_y + (stem_bot_y - stem_top_y) * t
        # Gentle curve to the right at the bottom
        x = cx + 0.03 * scale * math.sin(t * math.pi * 0.5)
        stem_pts.append((x, y))

    for i in range(len(stem_pts) - 1):
        draw.line([stem_pts[i], stem_pts[i + 1]], fill=fill, width=stem_width)

    # --- Small root lines at the bottom of stem ---
    root_base_x, root_base_y = stem_pts[-1]
    root_width = max(1, stem_width // 2)
    # Three small roots
    for angle_offset, length_factor in [(-0.6, 0.08), (0.0, 0.06), (0.5, 0.07)]:
        rx = root_base_x + math.sin(angle_offset) * length_factor * scale
        ry = root_base_y + math.cos(angle_offset) * length_factor * scale * 0.5
        draw.line([(root_base_x, root_base_y), (rx, ry)], fill=fill, width=root_width)


def generate_icon(size, output_path):
    """Generate a single icon at the given size."""

    # Colors
    color_dark = (45, 80, 22)     # #2d5016
    color_primary = (74, 124, 35) # #4a7c23
    color_light = (107, 163, 46)  # #6ba32e

    # Create gradient background (dark at top-left feel, light at bottom-right)
    gradient = create_gradient(size, color_dark, color_light)

    # Apply diagonal gradient overlay for richer look
    overlay = Image.new("RGB", (size, size))
    overlay_pixels = overlay.load()
    for y in range(size):
        for x in range(size):
            # Diagonal ratio: top-left = 0, bottom-right = 1
            ratio = (x + y) / (2 * (size - 1))
            r = int(color_dark[0] + (color_light[0] - color_dark[0]) * ratio)
            g = int(color_dark[1] + (color_light[1] - color_dark[1]) * ratio)
            b = int(color_dark[2] + (color_light[2] - color_dark[2]) * ratio)
            overlay_pixels[x, y] = (r, g, b)

    # Blend gradient and overlay
    img = Image.blend(gradient, overlay, 0.5)

    # Add a subtle inner glow / lighter center
    center_overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    center_draw = ImageDraw.Draw(center_overlay)
    center_x, center_y = size // 2, size // 2
    max_radius = int(size * 0.4)
    for r in range(max_radius, 0, -1):
        alpha = int(30 * (1 - r / max_radius))
        center_draw.ellipse(
            [center_x - r, center_y - r, center_x + r, center_y + r],
            fill=(255, 255, 255, alpha),
        )

    img = img.convert("RGBA")
    img = Image.alpha_composite(img, center_overlay)

    # Create rounded rectangle mask
    corner_radius = int(size * 0.18)
    mask = create_rounded_mask(size, corner_radius)

    # Apply mask - create final RGBA image
    final = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    final.paste(img, (0, 0), mask)

    # Draw the leaf/plant symbol
    draw = ImageDraw.Draw(final)
    draw_leaf(draw, size / 2, size / 2, size)

    # Add a subtle border highlight inside the rounded rect
    border_overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    border_draw = ImageDraw.Draw(border_overlay)
    border_width = max(1, int(size * 0.005))
    border_draw.rounded_rectangle(
        [border_width, border_width, size - 1 - border_width, size - 1 - border_width],
        radius=corner_radius - border_width,
        outline=(255, 255, 255, 40),
        width=border_width,
    )
    final = Image.alpha_composite(final, border_overlay)

    # Save
    final.save(output_path, "PNG")
    print(f"Saved: {output_path} ({size}x{size})")


if __name__ == "__main__":
    import os

    output_dir = r"d:\AI-Powered Crop Recommendation System with Explainable AI and Economic Analysis\webapp\static\images"
    os.makedirs(output_dir, exist_ok=True)

    generate_icon(192, os.path.join(output_dir, "icon-192.png"))
    generate_icon(512, os.path.join(output_dir, "icon-512.png"))
    print("Done! Both icons generated successfully.")
