import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import json
import tempfile
import os
import subprocess
import urllib.request
import hashlib
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFont
st.set_page_config(page_title="AI Coach", page_icon="🏐")
st.title("🏐 AI Coach - Hỗ trợ học sinh lớp 10 rèn luyện kỹ thuật chuyền bóng tại trường THPT Đoàn Văn Tố")

MODEL = Path(__file__).parent / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/"
    "pose_landmarker_lite.task"
)


def get_model():
    if not MODEL.exists() or MODEL.stat().st_size < 1_000_000:
        urllib.request.urlretrieve(MODEL_URL, str(MODEL))
# ==============================
# NHẬN DIỆN HƯỚNG CƠ THỂ
# ==============================

def detect_body_direction(points, width, height):
    left_shoulder = points[11]
    right_shoulder = points[12]
    left_hip = points[23]
    right_hip = points[24]

    if min(
        left_shoulder.visibility,
        right_shoulder.visibility,
        left_hip.visibility,
        right_hip.visibility
    ) < 0.5:
        return "Không xác định"

    shoulder_width = abs(
        left_shoulder.x - right_shoulder.x
    ) * width

    shoulder_y = (
        left_shoulder.y + right_shoulder.y
    ) / 2

    hip_y = (
        left_hip.y + right_hip.y
    ) / 2

    torso_length = abs(
        hip_y - shoulder_y
    ) * height

    if torso_length < 1:
        return "Không xác định"

    ratio = shoulder_width / torso_length

    if ratio >= 0.65:
        return "Gần chính diện"
    elif ratio >= 0.35:
        return "Góc chéo"
    elif ratio >= 0.20:
        return "Gần nghiêng ngang"
    else:
        return "Nghiêng ngang"
# ==============================
# PHÂN TÍCH TƯ THẾ CHUẨN BỊ
# TIÊU CHÍ 1 CỦA GIÁO VIÊN
# ==============================

def analyze_ready_posture(points, width, height):

    # Lấy các điểm khung xương cần thiết
    left_shoulder = points[11]
    right_shoulder = points[12]

    left_hip = points[23]
    right_hip = points[24]

    left_knee = points[25]
    right_knee = points[26]

    left_ankle = points[27]
    right_ankle = points[28]

    required_points = [
        left_shoulder, right_shoulder,
        left_hip, right_hip,
        left_knee, right_knee,
        left_ankle, right_ankle
    ]

    # Kiểm tra độ rõ của khung xương
    if min(p.visibility for p in required_points) < 0.5:
        return {
            "trang_thai": "Chưa đủ dữ liệu"
        }

    # Chuyển tọa độ chuẩn hóa sang pixel
    def pixel(point):
        return (
            point.x * width,
            point.y * height
        )

    # Tính góc tại một khớp
    def calculate_angle(a, b, c):
        a = np.array(pixel(a))
        b = np.array(pixel(b))
        c = np.array(pixel(c))

        ba = a - b
        bc = c - b

        denominator = (
            np.linalg.norm(ba) *
            np.linalg.norm(bc)
        )

        if denominator == 0:
            return None

        cosine = np.dot(ba, bc) / denominator
        cosine = np.clip(cosine, -1.0, 1.0)

        return round(
            float(np.degrees(np.arccos(cosine))), 1
        )

    # Độ rộng hai chân so với hai vai
    shoulder_width = abs(
        left_shoulder.x - right_shoulder.x
    ) * width

    ankle_width = abs(
        left_ankle.x - right_ankle.x
    ) * width

    if shoulder_width > 1:
        stance_ratio = round(
            ankle_width / shoulder_width, 2
        )
    else:
        stance_ratio = None

    # Góc gập hai đầu gối
    left_knee_angle = calculate_angle(
        left_hip, left_knee, left_ankle
    )

    right_knee_angle = calculate_angle(
        right_hip, right_knee, right_ankle
    )

    # Độ nghiêng thân người trong hình ảnh
    shoulder_center = np.array([
        (left_shoulder.x + right_shoulder.x) / 2 * width,
        (left_shoulder.y + right_shoulder.y) / 2 * height
    ])

    hip_center = np.array([
        (left_hip.x + right_hip.x) / 2 * width,
        (left_hip.y + right_hip.y) / 2 * height
    ])

    dx = abs(hip_center[0] - shoulder_center[0])
    dy = abs(hip_center[1] - shoulder_center[1])

    torso_lean = round(
        float(np.degrees(np.arctan2(dx, dy))), 1
    )

    return {
        "trang_thai": "Đủ dữ liệu đo",
        "do_rong_chan_so_voi_vai": stance_ratio,
        "goc_goi_trai": left_knee_angle,
        "goc_goi_phai": right_knee_angle,
        "do_nghieng_than_trong_hinh": torso_lean
    }
def analyze(path):
    get_model()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Không mở được video. Hãy thử video MP4 khác.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    font = ImageFont.truetype(
        str(Path(__file__).parent / "DejaVuSans.ttf"),
        22
    )
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=2,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )

    output = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4"
    ).name

    writer = cv2.VideoWriter(
        output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps / 2,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Không tạo được video đầu ra.")

    total = 0
    detected = 0
    preview = None
    # Bộ nhớ theo dõi người chơi
    tracks = {}
    next_id = 1
    seen_ids = set()
    pose_data = []
    frame_count = 0
    max_missing = 240
    try:
        with vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:
            frame_no = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_no % 2 == 0:
                    total += 1

                    rgb = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2RGB
                    )
                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb
                    )

                    timestamp = int(frame_no * 1000 / fps)
                    result = landmarker.detect_for_video(
                        image, timestamp
                    )

                    frame_count += 1

                    # Cập nhật thời gian vắng mặt của các ID
                    for track in tracks.values():
                        track["missing"] += 1

                    # Xóa ID đã biến mất quá lâu
                    tracks = {
                        track_id: track
                        for track_id, track in tracks.items()
                        if track["missing"] <= max_missing
                    }

                    people = []
                    duplicate_count = 0

                    if result.pose_landmarks:
                        detected += len(result.pose_landmarks)
                        for pose_idx, points in enumerate(result.pose_landmarks):
                            # Ưu tiên lấy vị trí hai hông
                            center_points = [
                                points[i] for i in (23, 24)
                                if points[i].visibility > 0.5
                            ]

                            # Nếu không thấy hông, lấy hai vai
                            if not center_points:
                                center_points = [
                                    points[i] for i in (11, 12)
                                    if points[i].visibility > 0.5
                                ]

                            # Nếu vẫn không có, lấy các điểm nhìn thấy
                            if not center_points:
                                center_points = [
                                    p for p in points
                                    if p.visibility > 0.5
                                ]

                            if center_points:
                                center_x = sum(
                                    p.x for p in center_points
                                ) / len(center_points)

                                center_y = sum(
                                    p.y for p in center_points
                                ) / len(center_points)

                                center = (center_x, center_y)
                            else:
                                center = None

                            if center is not None:
                                visible = [
                                    p for p in points
                                    if p.visibility > 0.5
                                ]

                                xs = [p.x for p in visible]
                                ys = [p.y for p in visible]

                                bbox = (
                                    min(xs), min(ys),
                                    max(xs), max(ys)
                                )

                                duplicate = False
                                for old_person in people:
                                    old = old_person["bbox"]

                                    x1 = max(bbox[0], old[0])
                                    y1 = max(bbox[1], old[1])
                                    x2 = min(bbox[2], old[2])
                                    y2 = min(bbox[3], old[3])

                                    intersection = (
                                        max(0, x2 - x1)
                                        * max(0, y2 - y1)
                                    )

                                    area1 = (
                                        (bbox[2] - bbox[0])
                                        * (bbox[3] - bbox[1])
                                    )

                                    area2 = (
                                        (old[2] - old[0])
                                        * (old[3] - old[1])
                                    )

                                    union = area1 + area2 - intersection

                                    min_area = min(area1, area2)

                                    if min_area > 0 and intersection / min_area > 0.7:
                                        duplicate = True
                                        break
                                if duplicate:
                                    duplicate_count += 1
                                else:
                                    people.append({
                                        "points": points,
                                        "world_points": result.pose_world_landmarks[pose_idx],
                                        "center": center,
                                        "bbox": bbox
                                    })
                    # Theo dõi tối đa 5 người bằng vị trí và kích thước cơ thể
                    pairs = []

                    # Tính khung bao cơ thể của từng người
                    for person_idx, person in enumerate(people):
                        visible = [
                            p for p in person["points"]
                            if p.visibility > 0.5
                        ]

                        if visible:
                            xs = [p.x for p in visible]
                            ys = [p.y for p in visible]

                            person["bbox"] = (
                                min(xs), min(ys),
                                max(xs), max(ys)
                            )
                        else:
                            person["bbox"] = None

                    # So sánh người mới với ID đã có
                    for person_idx, person in enumerate(people):
                        if person["center"] is None:
                            continue

                        for track_id, track in tracks.items():
                            dx = (
                                person["center"][0]
                                - track["center"][0]
                            )
                            dy = (
                                person["center"][1]
                                - track["center"][1]
                            )

                            distance = (dx ** 2 + dy ** 2) ** 0.5

                            size_difference = 0

                            if (
                                person["bbox"] is not None
                                and track.get("bbox") is not None
                            ):
                                old = track["bbox"]
                                new = person["bbox"]

                                old_w = old[2] - old[0]
                                old_h = old[3] - old[1]
                                new_w = new[2] - new[0]
                                new_h = new[3] - new[1]

                                size_difference = (
                                    abs(old_w - new_w) / max(old_w, new_w, 1)
                                    + abs(old_h - new_h) / max(old_h, new_h, 1)
                                )

                            score = distance + 0.10 * size_difference

                            if score < 0.40 or (
                                track["missing"] > 0 and score < 0.60
                            ):
                                pairs.append(
                                    (score, person_idx, track_id)
                                )
                    # Ghép những cặp phù hợp nhất
                    pairs.sort()

                    assigned = {}
                    used_tracks = set()

                    for score, person_idx, track_id in pairs:
                        if (
                            person_idx not in assigned
                            and track_id not in used_tracks
                        ):
                            assigned[person_idx] = track_id
                            used_tracks.add(track_id)

                            tracks[track_id]["center"] = (
                                people[person_idx]["center"]
                            )
                            tracks[track_id]["bbox"] = (
                                people[person_idx]["bbox"]
                            )
                            tracks[track_id]["missing"] = 0

                    # Cấp ID cho người chưa được ghép
                    for person_idx, person in enumerate(people):
                        if (
                            person["center"] is not None
                            and person_idx not in assigned
                        ):
                            track_id = next_id
                            next_id += 1

                            assigned[person_idx] = track_id

                            tracks[track_id] = {
                                "center": person["center"],
                                "bbox": person["bbox"],
                                "missing": 0
                            }
                    # Lưu lại các ID đã xuất hiện
                    seen_ids.update(assigned.values())
                    # Thu thập dữ liệu khung xương theo từng ID
                    for person_idx, person in enumerate(people):
                        if person_idx not in assigned:
                            continue

                        track_id = assigned[person_idx]

                        landmarks = []

                        for point in person["points"]:
                            landmarks.append({
                                "x": float(point.x),
                                "y": float(point.y),
                                "z": float(point.z),
                                "visibility": float(point.visibility)
                            })
                        world_landmarks = []

                        for point in person["world_points"]:
                            world_landmarks.append({
                                "x": float(point.x),
                                "y": float(point.y),
                                "z": float(point.z),
                                "visibility": float(point.visibility)
                            })
                        # Phân tích tư thế chuẩn bị theo từng ID
                        ready_posture = analyze_ready_posture(
                            person["points"], width, height
                        )
                        pose_data.append({
                            "frame": int(frame_no),
                            "timestamp": round(frame_no / fps, 3),
                            "id": int(track_id),
                            "direction": detect_body_direction(
                                person["points"], width, height
                            ),
                            "ready_posture": ready_posture,
                            "landmarks": landmarks,
                            "world_landmarks": world_landmarks
                        })                 
                        # Vẽ các điểm cơ thể và ID
                    labels = []

                    for person_idx, person in enumerate(people):
                        points = person["points"]

                        for p in points:
                            if p.visibility > 0.5:
                                x = int(p.x * width)
                                y = int(p.y * height)

                                cv2.circle(
                                    frame, (x, y), 4,
                                    (0, 255, 0), -1
                                )

                        if person_idx in assigned:
                            track_id = assigned[person_idx]
                            track = tracks[track_id]

                            if "display_center" not in track:
                                track["display_center"] = person["center"]

                            smooth = 0.35

                            old_x, old_y = track["display_center"]
                            new_x, new_y = person["center"]

                            track["display_center"] = (
                                old_x * (1 - smooth) + new_x * smooth,
                                old_y * (1 - smooth) + new_y * smooth
                            )

                            center_x = int(track["display_center"][0] * width)
                            center_y = int(track["display_center"][1] * height)

                            direction = detect_body_direction(
                                points, width, height
                            )                           
                            print("HUONG CO THE:", track_id, direction)

                            labels.append(
                                (center_x, center_y, track_id, direction)
                            )

                    # Chuyển ảnh sang Pillow để viết chữ tiếng Việt
                    frame_pil = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )
                    draw = ImageDraw.Draw(frame_pil)

                    for center_x, center_y, track_id, direction in labels:
                        draw.text(
                            (center_x, center_y - 30),
                            f"ID: {track_id} | {direction}",
                            font=font,
                            fill=(255, 255, 0)
                        )

                    # Chuyển ảnh trở lại OpenCV
                    frame = cv2.cvtColor(
                        np.array(frame_pil),
                        cv2.COLOR_RGB2BGR
                    )

                    cv2.putText(
                        frame, f"PEOPLE: {len(people)}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2
                    )

                    writer.write(frame)

                    preview = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2RGB
                    )
                frame_no += 1

    finally:
        cap.release()
        writer.release()

    if total == 0:
        raise ValueError("Video không có khung hình đọc được.")

    h264_output = tempfile.NamedTemporaryFile(
        delete=False, suffix=".mp4"
    ).name

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", output,
                "-i", path,
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-shortest",
                "-movflags", "+faststart",
                h264_output
            ],
            check=True,
            capture_output=True,
            text=True
        )

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "Không chuyển mã được video: " + e.stderr[-1000:]
        )

    finally:
        if os.path.exists(output):
            os.remove(output)

    return h264_output, preview, total, detected, sorted(seen_ids), pose_data

# ==============================
# GIAO DIỆN AI COACH
# ==============================

st.subheader("Hệ thống nhận diện và phân tích kỹ thuật bóng chuyền")


st.markdown(
    """
    Phân tích chuyển động, nhận diện người chơi và hỗ trợ
    đánh giá kỹ thuật chuyền bóng thấp tay và cao tay bằng trí tuệ nhân tạo.
    """
)

st.divider()

# Lựa chọn chế độ phân tích
st.header("⚙️ Thiết lập phân tích")

mode = st.radio(
    "Chọn kỹ thuật cần phân tích:",
    [
        "Chuyền bóng thấp tay",
        "Chuyền bóng cao tay",
        "Phân tích cả hai kỹ thuật"
    ],
    horizontal=True
)

st.divider()

# Khu vực tải video
st.header("🎥 Video đầu vào")
# Lưu kết quả phân tích trong phiên làm việc
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "last_video_hash" not in st.session_state:
    st.session_state.last_video_hash = None

video = st.file_uploader(
    "Tải video cần phân tích lên",
    type=["mp4", "mov", "avi", "m4v"]
)

# Kiểm tra video mới
if video is not None:
    current_video_hash = hashlib.sha256(
        video.getvalue()
    ).hexdigest()

    if (
        st.session_state.last_video_hash is not None
        and current_video_hash != st.session_state.last_video_hash
    ):
        st.session_state.analysis_result = None

    st.session_state.last_video_hash = current_video_hash

if video:
    st.video(video)

    if st.button("🚀 Bắt đầu phân tích", use_container_width=True):
        temp = None
        normalized = None
        result_file = None

        try:
            suffix = Path(video.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as f:
                f.write(video.getbuffer())
                temp = f.name

            # Chuẩn hóa chiều video bằng FFmpeg
            normalized = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ).name

            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", temp,
                    "-map", "0:v:0",
                    "-map", "0:a?",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    normalized
                ],
                check=True,
                capture_output=True,
                text=True
            )

            with st.spinner("Đang nhận diện và phân tích video..."):
                result_file, preview, total, detected, player_ids, pose_data = analyze(normalized)

            # Đọc video kết quả thành dữ liệu trong bộ nhớ
            with open(result_file, "rb") as f:
                result_video = f.read()

            # Lưu toàn bộ kết quả vào session_state
            st.session_state.analysis_result = {
                "preview": preview,
                "total": total,
                "detected": detected,
                "player_ids": player_ids,
                "pose_data": pose_data,
                "result_video": result_video
            }

            st.success("Đã xử lý video thành công!")

        except Exception as e:
            st.error("Phân tích thất bại:")
            st.code(str(e))

        finally:
            for file_path in (temp, normalized):
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)

    # Hiển thị kết quả đã lưu
    if st.session_state.analysis_result is not None:

        result = st.session_state.analysis_result

        json_data = json.dumps(
            result["pose_data"],
            ensure_ascii=False,
            indent=2
        )

        st.download_button(
            "📥 Tải dữ liệu khung xương (JSON)",
            data=json_data,
            file_name="du_lieu_khung_xuong.json",
            mime="application/json"
        )

        # Thông tin tổng quan
        st.header("📊 Kết quả tổng quan")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Tổng số khung hình", result["total"])

        with col2:
            st.metric(
                "Khung hình nhận diện được",
                result["detected"]
            )

        # Hình ảnh phân tích
        if result["preview"] is not None:
            st.subheader("🖼️ Hình ảnh nhận diện")
            st.image(
                result["preview"],
                caption="Khung hình phân tích cuối cùng"
            )

        # Video kết quả
        st.subheader("🎬 Video sau phân tích")
        st.video(result["result_video"])

        st.download_button(
            "⬇️ Tải video kết quả",
            data=result["result_video"],
            file_name="ket_qua.mp4",
            mime="video/mp4",
            use_container_width=True
        )

        # Khu vực kết quả từng người
        st.divider()
        st.header("👥 Kết quả theo từng người")

        player_ids = result["player_ids"]

        if player_ids:
            st.write(
                f"Tổng số người chơi đã nhận diện: {len(player_ids)}"
            )

            cols = st.columns(len(player_ids))

            for col, track_id in zip(cols, player_ids):
                with col:
                    st.metric(
                        f"Người chơi ID {track_id}",
                        "Đã nhận diện"
                    )
        else:
            st.warning("Chưa nhận diện được người chơi nào.")

        # Khu vực đánh giá kỹ thuật
        st.header("📝 Đánh giá kỹ thuật")

        st.info(
            "Các chỉ số và tiêu chuẩn đánh giá sẽ được "
            "bổ sung sau khi có dữ liệu từ giáo viên."
        )

        if result["detected"] == 0:
            st.warning(
                "Không nhận diện được cơ thể. "
                "Hãy kiểm tra ánh sáng và góc quay."
            )
