import os
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = r"U:\skipping count\Chinese teenager breaks own record in jump rope contest.mp4"
OUTPUT_PATH = "output/skipping_line_counter.mp4"
MODEL_PATH = "yolo26n-pose.pt"

RESIZE_ENABLED = True
RESIZE_WIDTH = 960
RESIZE_HEIGHT = None

CONF_THRESHOLD = 0.25


# ============================================================
# COUNTING LINE
# ============================================================

# Default position of the line.
# 0.78 = 78% of frame height.

COUNT_LINE_RATIO = 0.78

# Pixels around the line considered as touching.

LINE_TOLERANCE = 12

# Smoothing for ankle movement.

ANKLE_SMOOTHING_ALPHA = 0.55

# Minimum time between two counts.

MIN_TIME_BETWEEN_COUNTS_SEC = 0.06

# Foot must go this many pixels ABOVE the line
# before it can count again.

REARM_MARGIN = 18


# ============================================================
# INTERACTIVE LINE
# ============================================================

WINDOW_NAME = "AI Skipping Counter - Foot Line Crossing"

line_y = None
dragging_line = False

# Mouse line movement enabled.

EDIT_LINE_WITH_MOUSE = True


# ============================================================
# COCO KEYPOINTS
# ============================================================

L_ANKLE = 15
R_ANKLE = 16


SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6)
]


# ============================================================
# COLORS
# ============================================================

ACCENT = (60, 220, 255)
ACCENT_2 = (255, 130, 60)

PANEL_BG = (25, 20, 15)

WHITE = (255, 255, 255)

GREEN = (100, 255, 120)
RED = (80, 80, 255)

SKELETON_COLOR = (255, 220, 80)
JOINT_COLOR = (60, 220, 255)

LINE_DRAG_COLOR = (0, 255, 0)


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global line_y
    global dragging_line

    if not EDIT_LINE_WITH_MOUSE:
        return

    # --------------------------------------------------------
    # Mouse pressed
    # --------------------------------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:

        dragging_line = True
        line_y = y

    # --------------------------------------------------------
    # Mouse moving
    # --------------------------------------------------------

    elif event == cv2.EVENT_MOUSEMOVE:

        if dragging_line:
            line_y = y

    # --------------------------------------------------------
    # Mouse released
    # --------------------------------------------------------

    elif event == cv2.EVENT_LBUTTONUP:

        dragging_line = False
        line_y = y


# ============================================================
# ROUNDED PANEL
# ============================================================

def draw_rounded_panel(
    img,
    x1,
    y1,
    x2,
    y2,
    color,
    alpha=0.55,
    radius=18
):

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (x1 + radius, y1),
        (x2 - radius, y2),
        color,
        -1
    )

    cv2.rectangle(
        overlay,
        (x1, y1 + radius),
        (x2, y2 - radius),
        color,
        -1
    )

    for cx, cy in [
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ]:

        cv2.circle(
            overlay,
            (cx, cy),
            radius,
            color,
            -1
        )

    cv2.addWeighted(
        overlay,
        alpha,
        img,
        1 - alpha,
        0,
        img
    )


# ============================================================
# DRAW SKELETON
# ============================================================

def draw_skeleton(
    frame,
    kpts,
    conf,
    box=None
):

    # --------------------------------------------------------
    # Skeleton
    # --------------------------------------------------------

    for i, j in SKELETON:

        if (
            conf[i] > CONF_THRESHOLD
            and
            conf[j] > CONF_THRESHOLD
        ):

            p1 = tuple(
                kpts[i].astype(int)
            )

            p2 = tuple(
                kpts[j].astype(int)
            )

            cv2.line(
                frame,
                p1,
                p2,
                SKELETON_COLOR,
                4,
                cv2.LINE_AA
            )

            cv2.line(
                frame,
                p1,
                p2,
                WHITE,
                1,
                cv2.LINE_AA
            )

    # --------------------------------------------------------
    # Joints
    # --------------------------------------------------------

    for idx in range(len(kpts)):

        if conf[idx] > CONF_THRESHOLD:

            p = tuple(
                kpts[idx].astype(int)
            )

            cv2.circle(
                frame,
                p,
                6,
                JOINT_COLOR,
                -1,
                cv2.LINE_AA
            )

            cv2.circle(
                frame,
                p,
                6,
                (20, 20, 20),
                1,
                cv2.LINE_AA
            )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    if box is not None:

        x1, y1, x2, y2 = box.astype(int)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            ACCENT,
            2,
            cv2.LINE_AA
        )


# ============================================================
# DRAW COUNT LINE
# ============================================================

def draw_count_line(
    frame,
    current_line_y,
    w,
    last_crossed="",
    dragging=False
):

    current_line_y = int(current_line_y)

    # --------------------------------------------------------
    # Tolerance area
    # --------------------------------------------------------

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            0,
            max(
                0,
                current_line_y - LINE_TOLERANCE
            )
        ),
        (
            w,
            min(
                frame.shape[0],
                current_line_y + LINE_TOLERANCE
            )
        ),
        (30, 30, 30),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.25,
        frame,
        0.75,
        0,
        frame
    )

    # --------------------------------------------------------
    # Main horizontal line
    # --------------------------------------------------------

    line_color = (
        LINE_DRAG_COLOR
        if dragging
        else ACCENT
    )

    cv2.line(
        frame,
        (0, current_line_y),
        (w, current_line_y),
        line_color,
        4,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Center marker
    # --------------------------------------------------------

    cv2.circle(
        frame,
        (w // 2, current_line_y),
        8,
        line_color,
        -1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # No text is drawn on the line.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Last crossed foot
    # --------------------------------------------------------

    if last_crossed:

        cv2.putText(
            frame,
            f"{last_crossed} FOOT",
            (
                w - 220,
                max(
                    30,
                    current_line_y - 15
                )
            ),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            GREEN,
            2,
            cv2.LINE_AA
        )


# ============================================================
# HUD
# ============================================================

def draw_hud(
    frame,
    count,
    rate,
    status,
    fps,
    w,
    h
):

    # --------------------------------------------------------
    # Counter panel
    # --------------------------------------------------------

    draw_rounded_panel(
        frame,
        24,
        24,
        360,
        160,
        PANEL_BG,
        alpha=0.65
    )

    cv2.putText(
        frame,
        "SKIPPING COUNTER",
        (44, 65),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        ACCENT,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        str(count),
        (44, 140),
        cv2.FONT_HERSHEY_DUPLEX,
        2.2,
        WHITE,
        4,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "jumps",
        (180, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (200, 200, 200),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Jump rate panel
    # --------------------------------------------------------

    draw_rounded_panel(
        frame,
        w - 260,
        24,
        w - 24,
        105,
        PANEL_BG,
        alpha=0.65
    )

    cv2.putText(
        frame,
        "JUMPS / MIN",
        (w - 240, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        ACCENT_2,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        f"{rate:.0f}",
        (w - 240, 92),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        WHITE,
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Status panel
    # --------------------------------------------------------

    draw_rounded_panel(
        frame,
        24,
        h - 65,
        330,
        h - 24,
        PANEL_BG,
        alpha=0.55
    )

    status_color = (
        GREEN
        if status == "Person Detected"
        else RED
    )

    cv2.circle(
        frame,
        (45, h - 44),
        7,
        status_color,
        -1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        status,
        (62, h - 37),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        WHITE,
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"{fps:.1f} FPS",
        (w - 120, h - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Controls
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "R: Reset Line | Q: Quit",
        (
            w // 2 - 130,
            h - 35
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (190, 190, 190),
        1,
        cv2.LINE_AA
    )


# ============================================================
# FOOT LINE COUNTER
# ============================================================

class FootLineCounter:

    def __init__(
        self,
        line_y,
        fps
    ):

        self.line_y = float(line_y)

        self.count = 0

        self.prev_left_y = None
        self.prev_right_y = None

        self.smooth_left_y = None
        self.smooth_right_y = None

        # Foot must return above the line
        # before another count.

        self.left_armed = True
        self.right_armed = True

        self.last_count_frame = -999999

        self.min_gap_frames = max(
            1,
            int(
                fps *
                MIN_TIME_BETWEEN_COUNTS_SEC
            )
        )

        self.count_frames = []

        self.last_crossed = ""

    # --------------------------------------------------------
    # Smoothing
    # --------------------------------------------------------

    def smooth(
        self,
        previous,
        current
    ):

        if previous is None:
            return float(current)

        return (
            ANKLE_SMOOTHING_ALPHA *
            float(current)
            +
            (1 - ANKLE_SMOOTHING_ALPHA) *
            previous
        )

    # --------------------------------------------------------
    # Update one foot
    # --------------------------------------------------------

    def update_foot(
        self,
        foot_y,
        foot_name,
        frame_idx
    ):

        if foot_y is None:
            return False

        if foot_name == "LEFT":

            self.smooth_left_y = self.smooth(
                self.smooth_left_y,
                foot_y
            )

            current_y = self.smooth_left_y

            previous_y = self.prev_left_y

            armed = self.left_armed

        else:

            self.smooth_right_y = self.smooth(
                self.smooth_right_y,
                foot_y
            )

            current_y = self.smooth_right_y

            previous_y = self.prev_right_y

            armed = self.right_armed

        # ----------------------------------------------------
        # Re-arm
        # ----------------------------------------------------

        if (
            current_y <
            self.line_y -
            REARM_MARGIN
        ):

            armed = True

        # ----------------------------------------------------
        # Detect downward crossing
        # ----------------------------------------------------

        crossed_downward = False

        if previous_y is not None:

            was_above = (
                previous_y <
                self.line_y -
                LINE_TOLERANCE
            )

            now_touching_or_below = (
                current_y >=
                self.line_y -
                LINE_TOLERANCE
            )

            if (
                was_above
                and
                now_touching_or_below
            ):

                crossed_downward = True

        # ----------------------------------------------------
        # Minimum frame gap
        # ----------------------------------------------------

        enough_time = (
            frame_idx -
            self.last_count_frame
            >=
            self.min_gap_frames
        )

        counted = False

        # ----------------------------------------------------
        # COUNT
        # ----------------------------------------------------

        if (
            armed
            and
            crossed_downward
            and
            enough_time
        ):

            self.count += 1

            self.last_count_frame = frame_idx

            self.count_frames.append(
                frame_idx
            )

            self.last_crossed = foot_name

            armed = False

            counted = True

        # ----------------------------------------------------
        # Save state
        # ----------------------------------------------------

        if foot_name == "LEFT":

            self.prev_left_y = current_y

            self.left_armed = armed

        else:

            self.prev_right_y = current_y

            self.right_armed = armed

        return counted

    # --------------------------------------------------------
    # Update both feet
    # --------------------------------------------------------

    def update(
        self,
        left_y,
        right_y,
        frame_idx
    ):

        left_counted = self.update_foot(
            left_y,
            "LEFT",
            frame_idx
        )

        right_counted = self.update_foot(
            right_y,
            "RIGHT",
            frame_idx
        )

        return (
            left_counted
            or
            right_counted
        )

    # --------------------------------------------------------
    # Jump rate
    # --------------------------------------------------------

    def jump_rate(self, fps):

        if len(self.count_frames) < 2:
            return 0.0

        recent = self.count_frames[-30:]

        span_frames = (
            recent[-1] -
            recent[0]
        )

        if span_frames <= 0:
            return 0.0

        span_seconds = (
            span_frames /
            fps
        )

        return (
            (len(recent) - 1)
            /
            span_seconds
            *
            60
        )


# ============================================================
# RESIZE FRAME
# ============================================================

def resize_frame(
    frame,
    target_w,
    target_h
):

    h, w = frame.shape[:2]

    if target_h is None:

        target_h = int(
            h *
            (
                target_w /
                w
            )
        )

    return cv2.resize(
        frame,
        (
            target_w,
            target_h
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# MAIN
# ============================================================

def main():

    global line_y

    print(
        "Loading YOLO Pose model..."
    )

    model = YOLO(
        MODEL_PATH
    )

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        print(
            "ERROR: Could not open video:"
        )

        print(
            VIDEO_PATH
        )

        return

    # ========================================================
    # VIDEO INFORMATION
    # ========================================================

    fps_in = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 30
    )

    original_w = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    original_h = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    # ========================================================
    # RESIZE
    # ========================================================

    if RESIZE_ENABLED:

        if RESIZE_HEIGHT is None:

            w = RESIZE_WIDTH

            h = int(
                original_h *
                (
                    RESIZE_WIDTH /
                    original_w
                )
            )

        else:

            w = RESIZE_WIDTH

            h = RESIZE_HEIGHT

    else:

        w = original_w

        h = original_h

    print(
        f"Original video: "
        f"{original_w}x{original_h}"
    )

    print(
        f"Processing video: "
        f"{w}x{h}"
    )

    # ========================================================
    # DEFAULT LINE
    # ========================================================

    default_line_y = int(
        h *
        COUNT_LINE_RATIO
    )

    line_y = default_line_y

    print(
        f"Default line Y: "
        f"{line_y}"
    )

    # ========================================================
    # CREATE WINDOW
    # ========================================================

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        w,
        h
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    out_dir = os.path.dirname(
        OUTPUT_PATH
    )

    if out_dir:

        os.makedirs(
            out_dir,
            exist_ok=True
        )

    # ========================================================
    # VIDEO WRITER
    # ========================================================

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        fourcc,
        fps_in,
        (w, h)
    )

    # ========================================================
    # COUNTER
    # ========================================================

    counter = FootLineCounter(
        line_y,
        fps_in
    )

    frame_idx = 0

    print()
    print("=" * 60)
    print("PROCESSING STARTED")
    print("=" * 60)
    print()
    print("Mouse LEFT CLICK + DRAG = Move Line")
    print("R = Reset Line")
    print("Q = Quit")
    print()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        if RESIZE_ENABLED:

            frame = resize_frame(
                frame,
                w,
                h
            )

        # ----------------------------------------------------
        # Update counter line
        # ----------------------------------------------------

        counter.line_y = float(
            line_y
        )

        status_text = (
            "No Person Detected"
        )

        # ====================================================
        # YOLO POSE
        # ====================================================

        results = model.predict(
            frame,
            verbose=False,
            conf=CONF_THRESHOLD
        )[0]

        # ====================================================
        # KEYPOINTS AVAILABLE
        # ====================================================

        if (
            results.keypoints is not None
            and
            len(results.keypoints.xy) > 0
        ):

            kpts_all = (
                results.keypoints.xy
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Keypoint confidence
            # ------------------------------------------------

            if (
                results.keypoints.conf
                is not None
            ):

                confs_all = (
                    results.keypoints.conf
                    .cpu()
                    .numpy()
                )

            else:

                confs_all = np.ones(
                    kpts_all.shape[:2]
                )

            # ------------------------------------------------
            # Bounding boxes
            # ------------------------------------------------

            boxes = None

            if (
                results.boxes
                is not None
            ):

                boxes = (
                    results.boxes.xyxy
                    .cpu()
                    .numpy()
                )

            # ------------------------------------------------
            # Select largest person
            # ------------------------------------------------

            if (
                boxes is not None
                and
                len(boxes) > 0
            ):

                areas = (
                    (boxes[:, 2] - boxes[:, 0])
                    *
                    (boxes[:, 3] - boxes[:, 1])
                )

                best_idx = int(
                    np.argmax(areas)
                )

            else:

                best_idx = 0

            # ------------------------------------------------
            # Selected person
            # ------------------------------------------------

            kpts = kpts_all[
                best_idx
            ]

            conf = confs_all[
                best_idx
            ]

            box = (
                boxes[best_idx]
                if boxes is not None
                else None
            )

            # ------------------------------------------------
            # Draw skeleton
            # ------------------------------------------------

            draw_skeleton(
                frame,
                kpts,
                conf,
                box
            )

            status_text = (
                "Person Detected"
            )

            # =================================================
            # ANKLES
            # =================================================

            left_y = None

            right_y = None

            if (
                conf[L_ANKLE]
                >
                CONF_THRESHOLD
            ):

                left_y = float(
                    kpts[
                        L_ANKLE
                    ][1]
                )

            if (
                conf[R_ANKLE]
                >
                CONF_THRESHOLD
            ):

                right_y = float(
                    kpts[
                        R_ANKLE
                    ][1]
                )

            # =================================================
            # UPDATE COUNTER
            # =================================================

            counted = counter.update(
                left_y,
                right_y,
                frame_idx
            )

            if counted:

                print(
                    f"COUNT: "
                    f"{counter.count} | "
                    f"Foot: "
                    f"{counter.last_crossed} | "
                    f"Frame: "
                    f"{frame_idx}"
                )

        # ====================================================
        # DRAW LINE
        # ====================================================

        draw_count_line(
            frame,
            line_y,
            w,
            counter.last_crossed,
            dragging_line
        )

        # ====================================================
        # RATE
        # ====================================================

        rate = counter.jump_rate(
            fps_in
        )

        # ====================================================
        # HUD
        # ====================================================

        draw_hud(
            frame,
            counter.count,
            rate,
            status_text,
            fps_in,
            w,
            h
        )

        # ====================================================
        # WRITE OUTPUT
        # ====================================================

        out.write(
            frame
        )

        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        # ====================================================
        # KEYBOARD
        # ====================================================

        key = (
            cv2.waitKey(1)
            &
            0xFF
        )

        # ----------------------------------------------------
        # Q = QUIT
        # ----------------------------------------------------

        if key == ord("q"):

            break

        # ----------------------------------------------------
        # R = RESET LINE
        # ----------------------------------------------------

        elif key == ord("r"):

            line_y = (
                default_line_y
            )

            counter.line_y = float(
                line_y
            )

            print(
                f"Line reset to Y={line_y}"
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    out.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Total jumps: "
        f"{counter.count}"
    )

    print(
        f"Output saved to: "
        f"{OUTPUT_PATH}"
    )

    print("=" * 60)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
