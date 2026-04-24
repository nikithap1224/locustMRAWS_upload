import cv2
import numpy as np

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

# Camera calibration
with np.load('calibration_data.npz') as data:
    MTX  = data['mtx']
    DIST = data['dist']

# Grid physical dimensions (mm)
GRID_OUTER_W  = 150.0   # total width  of printed grid sheet
GRID_OUTER_H  = 150.0   # total height of printed grid sheet

# White border around the actual grid lines (mm)
BORDER_LEFT   = 0.0
BORDER_RIGHT  = 0.0
BORDER_TOP    = 0.0
BORDER_BOTTOM = 0.0

# Inner grid (where the lines actually are)
INNER_W = GRID_OUTER_W - BORDER_LEFT - BORDER_RIGHT   # 150 mm
INNER_H = GRID_OUTER_H - BORDER_TOP  - BORDER_BOTTOM  # 150 mm

GRID_CELLS  = 10   # 10 × 10

# ArUco marker size (cm → but we work in mm here)
MARKER_SIZE_MM = 50.0   # 5 cm

# ──────────────────────────────────────────────
# 3-D world points for the OUTER grid corners
# Origin = TOP-LEFT corner of the outer sheet
# X → right, Y → down  (standard image convention)
# ──────────────────────────────────────────────
GRID_WORLD_CORNERS = np.array([
    [0,            0,           0],   # TL
    [GRID_OUTER_W, 0,           0],   # TR
    [GRID_OUTER_W, GRID_OUTER_H, 0],  # BR
    [0,            GRID_OUTER_H, 0],  # BL
], dtype=np.float32)

# ──────────────────────────────────────────────
# ArUco setup
# ──────────────────────────────────────────────
aruco_dict  = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters  = cv2.aruco.DetectorParameters()
detector    = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def detect_grid_corners(frame):
    """
    Detect the 4 outer corners of the printed grid using contour analysis.
    Returns (TL, TR, BR, BL) as float32 pixel coordinates, or None.
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Largest contour should be the grid border
    c = max(cnts, key=cv2.contourArea)
    peri  = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) != 4:
        return None

    pts = approx.reshape(4, 2).astype(np.float32)

    # Sort: TL, TR, BR, BL
    s   = pts.sum(axis=1)
    d   = np.diff(pts, axis=1).ravel()
    tl  = pts[np.argmin(s)]
    br  = pts[np.argmax(s)]
    tr  = pts[np.argmin(d)]
    bl  = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def grid_corners_from_pose(rvec, tvec):
    """
    Project the 4 known world corners back to image pixels using camera pose.
    Used as fallback / verification.
    """
    img_pts, _ = cv2.projectPoints(GRID_WORLD_CORNERS, rvec, tvec, MTX, DIST)
    return img_pts.reshape(4, 2).astype(np.float32)


def marker_center_image(corner):
    """Return pixel centre of an ArUco corner quad."""
    return corner.reshape(4, 2).mean(axis=0)


def pixel_to_grid_mm(px_point, H_inv):
    """
    Map a pixel coordinate to grid-mm using the inverse homography.
    Returns (x_mm, y_mm) where (0,0) is the TOP-LEFT of the outer grid sheet.
    """
    pt  = np.array([[[px_point[0], px_point[1]]]], dtype=np.float32)
    res = cv2.perspectiveTransform(pt, H_inv)
    x_mm, y_mm = res[0][0]
    return float(x_mm), float(y_mm)


def mm_to_cell(x_mm, y_mm):
    """
    Convert mm position (relative to top-left of outer sheet) to
    fractional cell indices on the inner 10×10 grid.
    Returns (col, row) — can be fractional, integer part = cell index (0-based).
    """
    # Shift to inner grid origin
    x_inner = x_mm - BORDER_LEFT
    y_inner = y_mm - BORDER_TOP

    col = x_inner / (INNER_W / GRID_CELLS)
    row = y_inner / (INNER_H / GRID_CELLS)
    return col, row


def draw_grid_overlay(frame, corners_px):
    """Draw the detected grid outline and inner lines onto the frame."""
    pts = corners_px.astype(np.int32)

    # Homography: mm world coords → image pixels
    H, _ = cv2.findHomography(GRID_WORLD_CORNERS[:, :2], corners_px)

    cell_w = INNER_W / GRID_CELLS   # mm per cell horizontally (135/10 = 13.5)
    cell_h = INNER_H / GRID_CELLS   # mm per cell vertically   (120/10 = 12.0)

    # ── Inner grid boundary (offset inward by the white border) ──
    inner_corners_mm = np.array([
        [[BORDER_LEFT,              BORDER_TOP]],
        [[BORDER_LEFT + INNER_W,    BORDER_TOP]],
        [[BORDER_LEFT + INNER_W,    BORDER_TOP + INNER_H]],
        [[BORDER_LEFT,              BORDER_TOP + INNER_H]],
    ], dtype=np.float32)
    inner_corners_px = cv2.perspectiveTransform(inner_corners_mm, H)
    inner_pts = inner_corners_px.reshape(4, 2).astype(np.int32)

    # Draw outer sheet boundary (thin, dim)
    cv2.polylines(frame, [pts], True, (0, 140, 180), 1)
    # Draw inner grid boundary (bright)
    cv2.polylines(frame, [inner_pts], True, (0, 220, 255), 2)

    # ── Inner grid lines aligned to physical markings ──
    for i in range(1, GRID_CELLS):
        x_mm = BORDER_LEFT + i * cell_w   # absolute x in sheet mm
        y_mm = BORDER_TOP  + i * cell_h   # absolute y in sheet mm

        # Vertical line: top of inner grid → bottom of inner grid
        p1 = np.array([[[x_mm, BORDER_TOP]]], dtype=np.float32)
        p2 = np.array([[[x_mm, BORDER_TOP + INNER_H]]], dtype=np.float32)
        p1p = cv2.perspectiveTransform(p1, H)[0][0].astype(int)
        p2p = cv2.perspectiveTransform(p2, H)[0][0].astype(int)
        cv2.line(frame, tuple(p1p), tuple(p2p), (0, 220, 255), 1)

        # Horizontal line: left of inner grid → right of inner grid
        p3 = np.array([[[BORDER_LEFT, y_mm]]], dtype=np.float32)
        p4 = np.array([[[BORDER_LEFT + INNER_W, y_mm]]], dtype=np.float32)
        p3p = cv2.perspectiveTransform(p3, H)[0][0].astype(int)
        p4p = cv2.perspectiveTransform(p4, H)[0][0].astype(int)
        cv2.line(frame, tuple(p3p), tuple(p4p), (0, 220, 255), 1)

    # ── Origin marker at top-left of INNER grid ──
    origin_mm = np.array([[[BORDER_LEFT, BORDER_TOP]]], dtype=np.float32)
    origin_px  = cv2.perspectiveTransform(origin_mm, H)[0][0].astype(int)
    cv2.circle(frame, tuple(origin_px), 8, (0, 0, 255), -1)
    cv2.putText(frame, "Origin (0,0)",
                (origin_px[0] + 10, origin_px[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

cap = cv2.VideoCapture(0)

# Cache the last good homography so we don't recompute every frame if grid
# detection glitches for a frame.
H_inv_cache   = None
grid_px_cache = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── 1. Detect grid corners ──────────────────
    grid_corners_px = detect_grid_corners(frame)

    if grid_corners_px is not None:
        grid_px_cache = grid_corners_px

        # Homography: image pixels  →  mm world coords (Z=0 plane)
        H, mask = cv2.findHomography(grid_corners_px,
                                     GRID_WORLD_CORNERS[:, :2])
        if H is not None:
            H_inv_cache = np.linalg.inv(H)   # world→pixel direction (unused)
            # We need pixel→world, which is H itself
            H_pix_to_world = H

    # ── 2. Draw grid overlay ────────────────────
    if grid_px_cache is not None:
        draw_grid_overlay(frame, grid_px_cache)

    # ── 3. Detect ArUco markers ─────────────────
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and grid_px_cache is not None and H_pix_to_world is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Also estimate 3-D pose for each marker (distance info)
        obj_points_marker = np.array([
            [-MARKER_SIZE_MM/2,  MARKER_SIZE_MM/2, 0],
            [ MARKER_SIZE_MM/2,  MARKER_SIZE_MM/2, 0],
            [ MARKER_SIZE_MM/2, -MARKER_SIZE_MM/2, 0],
            [-MARKER_SIZE_MM/2, -MARKER_SIZE_MM/2, 0],
        ], dtype=np.float32)

        for i, corner in enumerate(corners):
            img_pts = corner.reshape(4, 2).astype(np.float32)
            center_px = img_pts.mean(axis=0)

            # ── Grid-relative position (2-D homography) ──
            pt_h = np.array([[[center_px[0], center_px[1]]]], dtype=np.float32)
            world_pt = cv2.perspectiveTransform(pt_h, H_pix_to_world)
            # Subtract border so (0,0) = top-left of the actual inner grid lines
            x_mm = float(world_pt[0][0][0]) - BORDER_LEFT
            y_mm = float(world_pt[0][0][1]) - BORDER_TOP

            col, row = mm_to_cell(x_mm + BORDER_LEFT, y_mm + BORDER_TOP)

            # ── 3-D pose for Z-distance ──────────────────
            success, rvec, tvec = cv2.solvePnP(
                obj_points_marker, img_pts, MTX, DIST)

            dist_mm = tvec[2][0] if success else 0.0

            # ── Draw marker info ─────────────────────────
            cx, cy = int(center_px[0]), int(center_px[1])

            # Cross-hair at marker centre
            cv2.drawMarker(frame, (cx, cy), (0, 255, 0),
                           cv2.MARKER_CROSS, 20, 2)

            # Line from inner-grid origin pixel to marker centre
            H_world_to_px = np.linalg.inv(H_pix_to_world)
            origin_inner_mm = np.array([[[BORDER_LEFT, BORDER_TOP]]], dtype=np.float32)
            origin_px = tuple(cv2.perspectiveTransform(origin_inner_mm, H_world_to_px)[0][0].astype(int))
            cv2.line(frame, origin_px, (cx, cy), (255, 150, 0), 1)

            # Text block
            label_lines = [
                f"ID: {ids[i][0]}",
                f"X: {x_mm:+.1f} mm",
                f"Y: {y_mm:+.1f} mm",
                f"Cell: ({col:.1f}, {row:.1f})",
                f"Dist: {dist_mm:.1f} mm",
            ]
            tx, ty = cx + 12, cy - 60
            box_w, box_h = 180, len(label_lines) * 20 + 8
            cv2.rectangle(frame,
                          (tx - 4, ty - 4),
                          (tx + box_w, ty + box_h),
                          (0, 0, 0), -1)
            for j, line in enumerate(label_lines):
                color = (0, 255, 0) if j == 0 else (255, 255, 255)
                cv2.putText(frame, line,
                            (tx, ty + j * 20 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.52, color, 1, cv2.LINE_AA)

    # ── 4. Status bar ───────────────────────────
    grid_status = "Grid: LOCKED" if grid_px_cache is not None else "Grid: searching..."
    grid_color  = (0, 255, 100) if grid_px_cache is not None else (0, 100, 255)
    cv2.putText(frame, grid_status,
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, grid_color, 2)

    cv2.putText(frame, "Origin = grid top-left corner  |  Q to quit",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.imshow("ArUco Grid Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
