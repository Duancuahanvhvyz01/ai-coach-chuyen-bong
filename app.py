import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import os
import subprocess
import urllib.request
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="AI Coach", page_icon="🏐")
st.title("🏐 AI Coach chuyền bóng thấp tay")

MODEL = Path(__file__).parent / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/"
    "pose_landmarker_lite.task"
)


def get_model():
    if not MODEL.exists() or MODEL.stat().st_size < 1_000_000:
        urllib.request.urlretrieve(MODEL_URL, str(MODEL))


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

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
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

                    if result.pose_landmarks:
                        detected += 1
                        points = result.pose_landmarks[0]

                        for p in points:
                            if p.visibility > 0.5:
                                x = int(p.x * width)
                                y = int(p.y * height)
                                cv2.circle(
                                    frame, (x, y), 4,
                                    (0, 255, 0), -1
                                )

                        cv2.putText(
                            frame, "BODY DETECTED",
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

    return h264_output, preview, total, detected


video = st.file_uploader(
    "Chọn video",
    type=["mp4", "mov", "avi", "m4v"]
)

if video:
    st.video(video)

    if st.button("Phân tích video"):
        temp = None
        result_file = None

        try:
            suffix = Path(video.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=suffix
            ) as f:
                f.write(video.getbuffer())
                temp = f.name

            with st.spinner("Đang phân tích..."):
                result_file, preview, total, detected = analyze(temp)

            st.success("Đã xử lý xong!")
            st.write("Số khung hình:", total)
            st.write("Khung hình nhận diện được cơ thể:", detected)

            if preview is not None:
                st.image(
                    preview,
                    caption="Khung hình phân tích cuối cùng"
                )
            st.subheader("🎥 Video phân tích")
            st.video(result_file)
            with open(result_file, "rb") as f:
                st.download_button(
                    "Tải video kết quả",
                    f.read(),
                    file_name="ket_qua.mp4",
                    mime="video/mp4"
                )

            if detected == 0:
                st.warning(
                    "Không nhận diện được cơ thể. "
                    "Hãy kiểm tra ánh sáng và góc quay."
                )

            st.info(
                "Bản này mới nhận diện tư thế cơ thể, "
                "chưa chấm đúng/sai kỹ thuật chuyền bóng."
            )

        except Exception as e:
            st.error("Phân tích thất bại:")
            st.code(str(e))

        finally:
            if temp and os.path.exists(temp):
                os.remove(temp)
