import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from math import ceil, atan2, pi

# === Utility functions ===
def crop_to_aspect(img: Image.Image, target_aspect=3/4, max_size=1024):
    # Resize large images first to prevent memory issues
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        w, h = new_w, new_h
    
    aspect = w / h
    if aspect > target_aspect:
        new_w = int(h * target_aspect)
        left = (w - new_w) // 2
        return img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_aspect)
        top = (h - new_h) // 2
        return img.crop((0, top, w, top + new_h))

def centered_box(rows, cols, box_rows, box_cols, offset_r=0, offset_c=0):
    r0 = (rows - box_rows) // 2 + offset_r
    c0 = (cols - box_cols) // 2 + offset_c
    r1 = r0 + box_rows - 1
    c1 = c0 + box_cols - 1
    # Ensure the box stays within grid bounds
    if r0 < 0:
        r0, r1 = 0, box_rows - 1
    elif r1 >= rows:
        r1, r0 = rows - 1, rows - box_rows
    if c0 < 0:
        c0, c1 = 0, box_cols - 1
    elif c1 >= cols:
        c1, c0 = cols - 1, cols - box_cols
    return r0, r1, c0, c1

def chebyshev_distance_to_box(r, c, r0, r1, c0, c1):
    dx = 0 if c0 <= c <= c1 else (c0 - c if c < c0 else c - c1)
    dy = 0 if r0 <= r <= r1 else (r0 - r if r < r0 else r - r1)
    return max(abs(dx), abs(dy))

def angle_to_box_center(r, c, r0, r1, c0, c1):
    cy = (r0 + r1) / 2.0
    cx = (c0 + c1) / 2.0
    ang = atan2(-(r - cy), (c - cx))
    ang = (pi / 2 - ang)
    while ang < 0: ang += 2 * pi
    while ang >= 2 * pi: ang -= 2 * pi
    return ang

def interleave_opposites(sorted_cells):
    n = len(sorted_cells)
    if n <= 2:
        return sorted_cells
    half = n // 2
    out = []
    for i in range(half):
        out.append(sorted_cells[i])
        if i + half < n:
            out.append(sorted_cells[i + half])
    if n % 2 == 1:
        out.append(sorted_cells[-1])
    return out
def surround_order_balanced(rows:int, cols:int, box_rows:int, box_cols:int, n:int, offset_r=0, offset_c=0):
    """Return first n (r,c) positions growing in rings with edge-centered ordering."""
    r0, r1, c0, c1 = centered_box(rows, cols, box_rows, box_cols, offset_r, offset_c)
    out = []
    # Enough rings to cover n cells even on small grids
    max_L = max(c0, cols-1-c1, r0, rows-1-r1) + 4
    for L in range(max_L+1):
        if len(out) >= n:
            break
        ring = ring_cells(rows, cols, r0, r1, c0, c1, L)
        if not ring:
            continue
        ring_seq = edge_centered_sequence(ring, r0, r1, c0, c1)
        need = n - len(out)
        out.extend(ring_seq[:need])
    return out

def surround_order(rows, cols, box_rows, box_cols, offset_r=0, offset_c=0):
    r0, r1, c0, c1 = centered_box(rows, cols, box_rows, box_cols, offset_r, offset_c)
    layers = {}
    for r in range(rows):
        for c in range(cols):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            L = chebyshev_distance_to_box(r, c, r0, r1, c0, c1)
            ang = angle_to_box_center(r, c, r0, r1, c0, c1)
            layers.setdefault(L, []).append((ang, (r, c)))
    order = []
    layer_keys = sorted(layers.keys())
    total_cells = sum(len(layers[L]) for L in layer_keys)
    # For dynamic placement, we don't know n_photos here, so handle in create_composite
    for idx, L in enumerate(layer_keys):
        ring = sorted(layers[L], key=lambda x: x[0])  # sort by angle for symmetry
        ring_cells = [rc for _, rc in ring]
        # For the outermost ring, distribute images evenly if not all cells are filled
        if idx == len(layer_keys) - 1:
            # Only fill as many positions as needed, spaced evenly
            # We'll select positions later in create_composite
            order.append((L, ring_cells))
        else:
            order.extend(ring_cells)
    return order

def pick_grid_for_reserved(n_photos, reserved_rows, reserved_cols, 
                           canvas_w, canvas_h, outer_margin=60, gutter=20,
                           target_aspect=3/4):
    # choose rows/cols to fit n_photos + reserved area
    reserved_cells = reserved_rows * reserved_cols
    max_dim = int(ceil((n_photos + reserved_cells)**0.5)) + reserved_rows
    max_rows = max_dim * 2
    max_cols = max_dim * 2
    aspect_canvas = canvas_w / canvas_h
    best = None
    for r in range(reserved_rows, max_rows + 1):
        for c in range(reserved_cols, max_cols + 1):
            capacity = r * c - reserved_cells
            if capacity < n_photos:
                continue
            cell_w = (canvas_w - 2 * outer_margin - (c - 1) * gutter) / c
            cell_h = (canvas_h - 2 * outer_margin - (r - 1) * gutter) / r
            if cell_w <= 0 or cell_h <= 0:
                continue
            eff_w = min(cell_w, cell_h * target_aspect)
            grid_aspect = c / r
            aspect_diff = abs(grid_aspect - aspect_canvas)
            score = eff_w - aspect_diff * 0.1
            if (best is None) or (score > best[0]):
                best = (score, r, c, cell_w, cell_h)
    if best is None:
        r = c = int(ceil((n_photos + reserved_cells)**0.5))
        cell_w = (canvas_w - 2 * outer_margin - (c - 1) * gutter) / c
        cell_h = (canvas_h - 2 * outer_margin - (r - 1) * gutter) / r
        best = (0, r, c, cell_w, cell_h)
    return best[1], best[2]

def grid_cells(canvas_w, canvas_h, rows, cols, outer_margin=60, gutter_x=20, gutter_y=None):
    if gutter_y is None: gutter_y = gutter_x
    x0 = outer_margin
    y0 = outer_margin
    avail_w = canvas_w - 2 * outer_margin
    avail_h = canvas_h - 2 * outer_margin
    cell_w = (avail_w - (cols - 1) * gutter_x) / cols
    cell_h = (avail_h - (rows - 1) * gutter_y) / rows
    rects = []
    for r in range(rows):
        for c in range(cols):
            x = int(x0 + c * (cell_w + gutter_x))
            y = int(y0 + r * (cell_h + gutter_y))
            rects.append((x, y, int(cell_w), int(cell_h)))
    return rects

def create_composite(student_imgs, canvas_w, canvas_h, reserved_rows, reserved_cols, 
                     school_name, emblem_img, outer_margin=60, gutter=20, 
                     target_aspect=3/4, bg_color=(255,255,255), fg_color=(0,0,0),
                     center_offset_r=0, center_offset_c=0):
    n_photos = len(student_imgs)
    rows, cols = pick_grid_for_reserved(n_photos, reserved_rows, reserved_cols,
                                        canvas_w, canvas_h, outer_margin, gutter, target_aspect)
    
    # Get grid positions and create canvas first
    rects = grid_cells(canvas_w, canvas_h, rows, cols, outer_margin, gutter, gutter)
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    # Reserved box
    r0, r1, c0, c1 = centered_box(rows, cols, reserved_rows, reserved_cols, center_offset_r, center_offset_c)
    x_min, y_min, x_max, y_max = canvas_w, canvas_h, 0, 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            idx = r * cols + c
            x, y, w, h = rects[idx]
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
    draw.rectangle([x_min, y_min, x_max, y_max], fill=(245, 245, 245))

    # Draw emblem and school name
    cur_x = x_min + 10
    reserved_width = x_max - x_min
    reserved_height = y_max - y_min
    if emblem_img is not None:
        em_max_h = reserved_height * 0.7
        em_max_w = reserved_width * 0.3
        em_ratio = emblem_img.width / emblem_img.height
        if em_max_w / em_ratio < em_max_h:
            target_w = em_max_w
            target_h = em_max_w / em_ratio
        else:
            target_h = em_max_h
            target_w = em_max_h * em_ratio
        em = emblem_img.resize((int(target_w), int(target_h)), Image.LANCZOS)
        em_y = int(y_min + (reserved_height - em.height) / 2)
        canvas.paste(em, (int(cur_x), em_y), em if em.mode == "RGBA" else None)
        cur_x += em.width + 20
    # Adjust font size
    max_w = reserved_width - (cur_x - x_min) - 10
    max_h = reserved_height * 0.8
    size = 10
    for fs in range(10, 200, 2):
        try:
            f = ImageFont.truetype("DejaVuSans-Bold.ttf", fs)
        except Exception:
            f = ImageFont.load_default()
        w, h = f.getbbox(school_name)[2:]
        if w <= max_w and h <= max_h:
            size = fs
        else:
            break
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        font = ImageFont.load_default()
    w, h = font.getbbox(school_name)[2:]
    text_x = cur_x + (max_w - w) / 2
    text_y = y_min + (reserved_height - h) / 2
    draw.text((text_x, text_y), school_name, fill=fg_color, font=font)

    # Crop and paste faces using balanced ordering
    positions = surround_order_balanced(rows, cols, reserved_rows, reserved_cols, n_photos, center_offset_r, center_offset_c)
    processed = [crop_to_aspect(img, target_aspect) for img in student_imgs]
    
    for i, (r, c) in enumerate(positions):
        idx = r * cols + c
        x, y, w, h = rects[idx]
        target_w = min(w, h * target_aspect)
        target_h = target_w / target_aspect
        im = processed[i].resize((int(target_w), int(target_h)), Image.LANCZOS)
        dx = int(x + (w - target_w) / 2)
        dy = int(y + (h - target_h) / 2)
        canvas.paste(im, (dx, dy))
    
    return canvas

def ring_cells(rows:int, cols:int, r0:int, r1:int, c0:int, c1:int, L:int):
    """Return all (r,c) at Chebyshev distance L from the reserved box."""
    top = r0 - L - 1
    bottom = r1 + L + 1
    left = c0 - L - 1
    right = c1 + L + 1
    cells = []

    # top and bottom edges (include corners here)
    if 0 <= top < rows:
        for c in range(max(0,left), min(cols-1,right)+1):
            cells.append((top, c))
    if 0 <= bottom < rows:
        for c in range(max(0,left), min(cols-1,right)+1):
            cells.append((bottom, c))

    # left and right edges (exclude already added corners)
    if 0 <= left < cols:
        for r in range(max(0,top+1), min(rows-1,bottom-1)+1):
            cells.append((r, left))
    if 0 <= right < cols:
        for r in range(max(0,top+1), min(rows-1,bottom-1)+1):
            cells.append((r, right))
    return cells

def edge_centered_sequence(ring, r0, r1, c0, c1):
    """Order a ring: T0, B0, R0, L0, then expand center-out per edge, interleaving T/B/R/L."""
    cy = (r0 + r1) / 2.0
    cx = (c0 + c1) / 2.0
    top, bottom, left, right = [], [], [], []

    for r, c in ring:
        if r < r0:   top.append((r, c))
        elif r > r1: bottom.append((r, c))
        elif c < c0: left.append((r, c))
        elif c > c1: right.append((r, c))

    # sort each edge by distance from its center
    top.sort(key=lambda rc: abs(rc[1] - cx))
    bottom.sort(key=lambda rc: abs(rc[1] - cx))
    left.sort(key=lambda rc: abs(rc[0] - cy))
    right.sort(key=lambda rc: abs(rc[0] - cy))

    # interleave one-by-one: T, B, R, L
    seq, buckets = [], [top, bottom, right, left]
    while any(buckets):
        for b in buckets:
            if b:
                seq.append(b.pop(0))
    return seq

# === Streamlit UI ===
st.title("Class Composite Generator")
st.markdown(
    "Upload your face crops, select a canvas size, optionally reserve a central area "
    "for your school name and emblem, and generate a class photo with concentric rings."
)

uploaded_files = st.file_uploader(
    "Choose face images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    st.info(f"Uploaded {len(uploaded_files)} images. Large images will be automatically resized to prevent memory issues.")
    if len(uploaded_files) > 30:
        st.warning("⚠️ Many images detected. Processing may take longer or fail on limited memory systems.")

# Canvas selection
canvas_option = st.selectbox(
    "Choose canvas size",
    [
        "A4 Portrait (2480x3508)",
        "A4 Landscape (3508x2480)",
        "Square (3000x3000)",
        "Custom",
    ],
)
if canvas_option == "A4 Portrait (2480x3508)":
    canvas_w, canvas_h = 2480, 3508
elif canvas_option == "A4 Landscape (3508x2480)":
    canvas_w, canvas_h = 3508, 2480
elif canvas_option == "Square (3000x3000)":
    canvas_w, canvas_h = 3000, 3000
else:
    canvas_w = st.number_input("Canvas width (px)", value=3508, step=100)
    canvas_h = st.number_input("Canvas height (px)", value=2480, step=100)

# Reserved central area in grid units
reserved_rows = st.number_input("Reserved rows (center area)", min_value=1, max_value=10, value=2)
reserved_cols = st.number_input("Reserved columns (center area)", min_value=1, max_value=10, value=3)

# Center position controls
st.subheader("Center Area Position")
center_offset_r = st.slider("Vertical offset (negative = up, positive = down)", 
                           min_value=-3, max_value=3, value=0, step=1)
center_offset_c = st.slider("Horizontal offset (negative = left, positive = right)", 
                           min_value=-3, max_value=3, value=0, step=1)

school_name = st.text_input("School name", value="My School")
emblem_file = st.file_uploader("Upload emblem (optional)", type=["jpg", "jpeg", "png"])

if st.button("Generate Composite"):
    if not uploaded_files:
        st.error("Please upload some face images first.")
    elif len(uploaded_files) > 50:
        st.error("Please upload no more than 50 images to prevent memory issues.")
    else:
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Loading images...")
            progress_bar.progress(0.1)
            
            # Load images as PIL objects with error handling
            faces = []
            for i, f in enumerate(uploaded_files):
                try:
                    img = Image.open(f).convert("RGB")
                    faces.append(img)
                except Exception as e:
                    st.warning(f"Could not load image {f.name}: {e}")
                    continue
            
            if not faces:
                st.error("No valid images could be loaded.")
                st.stop()
            
            progress_bar.progress(0.3)
            status_text.text("Processing emblem...")
            
            emblem_img = None
            if emblem_file:
                try:
                    emblem_img = Image.open(emblem_file).convert("RGBA")
                except Exception as e:
                    st.warning(f"Could not load emblem: {e}")
            
            progress_bar.progress(0.5)
            status_text.text("Generating composite...")
            
            composite = create_composite(
                faces,
                canvas_w=int(canvas_w),
                canvas_h=int(canvas_h),
                reserved_rows=int(reserved_rows),
                reserved_cols=int(reserved_cols),
                school_name=school_name.strip(),
                emblem_img=emblem_img,
                center_offset_r=center_offset_r,
                center_offset_c=center_offset_c,
            )
            
            progress_bar.progress(0.9)
            status_text.text("Displaying result...")
            
            st.image(composite, caption="Composite")
            
            progress_bar.progress(1.0)
            status_text.text("Complete!")
            
            # Offer download
            from io import BytesIO
            buf = BytesIO()
            composite.save(buf, format="PNG")
            st.download_button(
                label="Download composite",
                data=buf.getvalue(),
                file_name="class_composite.png",
                mime="image/png",
            )
            
        except Exception as e:
            st.error(f"An error occurred while generating the composite: {str(e)}")
            st.error("This might be due to memory limitations. Try using fewer or smaller images.")
