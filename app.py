import os
import time
import tempfile
import threading
from collections import deque

import av
import cv2
import pandas as pd
import streamlit as st
import torch
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from losat import AdaptiveLOSAT
from model import load_model
from utils import add_alert_border, compute_motion_metric, ensure_log_file, log_event, preprocess_clip, read_event_log


st.set_page_config(page_title="LOSAT based Violence detector", layout="wide")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(APP_DIR, "logs", "events.csv")
MODEL_PATH = os.path.join(APP_DIR, "best_model.pth")
INFER_FPS = 6.0
UPLOAD_INFER_FPS = 2.0
UPLOAD_FRAME_WIDTH = 320
CLIP_LEN = 16
UPLOAD_VIOLENCE_RATIO = 0.5
UPLOAD_AVG_SCORE_THRESHOLD = 0.7
LIVE_MIN_SCORE = 0.72
LIVE_MIN_MOTION = 0.03
LIVE_CONFIRM_CLIPS = 1


@st.cache_resource
def get_cpu_model():
    path = MODEL_PATH if os.path.exists(MODEL_PATH) else None
    return load_model(path, device="cpu")


def infer_score(model, clip_rgb):
    x = preprocess_clip(clip_rgb, size=112)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        return float(probs[0, 1].item())


def format_seconds(seconds: float) -> str:
    total = int(seconds)
    mm = total // 60
    ss = total % 60
    return f"{mm:02d}:{ss:02d}"


def prepare_analysis_frame(frame_bgr):
    h, w = frame_bgr.shape[:2]
    if w <= UPLOAD_FRAME_WIDTH:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    scale = UPLOAD_FRAME_WIDTH / float(w)
    resized = cv2.resize(frame_bgr, (UPLOAD_FRAME_WIDTH, max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def extract_video_thumbnail(video_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        temp_path = tmp.name

    cap = cv2.VideoCapture(temp_path)
    success, frame_bgr = cap.read()
    cap.release()

    try:
        os.remove(temp_path)
    except OSError:
        pass

    if not success:
        return None

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    thumb_width = 220
    scale = thumb_width / float(w)
    thumb = cv2.resize(frame_rgb, (thumb_width, max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
    return thumb


def render_summary_card(label: str, value: str):
    st.markdown(
        f"""
        <div style="padding: 0.3rem 0.25rem 0.55rem 0.25rem;">
            <div style="font-size: 1.7rem; font-weight: 700; color: #374151; margin-bottom: 0.25rem;">{label}</div>
            <div style="font-size: 3.2rem; font-weight: 800; line-height: 0.95; color: #1f2937;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(text: str):
    st.markdown(
        f"""
        <div style="font-size: 1.8rem; font-weight: 700; color: #374151; margin-bottom: 0.2rem;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert_block(text: str, alert_type: str):
    if alert_type == "violence":
        background = "#fee2e2"
        foreground = "#b91c1c"
    else:
        background = "#dcfce7"
        foreground = "#166534"

    st.markdown(
        f"""
        <div style="
            background: {background};
            color: {foreground};
            padding: 1rem 1.15rem;
            border-radius: 0.6rem;
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.2;
            margin: 0.15rem 0 0.8rem 0;
        ">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def process_uploaded_video(video_path: str, model, progress_bar=None, status_text=None) -> pd.DataFrame:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return pd.DataFrame(columns=["timestamp", "score", "threshold", "motion", "decision"])

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = float(src_fps) if src_fps and src_fps > 0 else 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_stride = max(1, int(round(src_fps / UPLOAD_INFER_FPS)))

    buffer = deque(maxlen=CLIP_LEN)
    losat = AdaptiveLOSAT(alpha=0.8, beta=0.2, init_threshold=0.5)
    rows = []

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % sample_stride != 0:
            frame_idx += 1
            continue

        frame_rgb = prepare_analysis_frame(frame_bgr)
        buffer.append(frame_rgb)

        if len(buffer) == CLIP_LEN:
            clip = list(buffer)
            motion = compute_motion_metric(clip)
            score = infer_score(model, clip)
            threshold, alert = losat.update(score, motion)
            decision = "Violence" if alert else "Non-Violence"

            rows.append(
                {
                    "timestamp": format_seconds(frame_idx / src_fps),
                    "score": round(score, 4),
                    "threshold": round(threshold, 4),
                    "motion": round(motion, 4),
                    "decision": decision,
                }
            )

        if total_frames > 0 and progress_bar is not None:
            progress = min(frame_idx / total_frames, 1.0)
            progress_bar.progress(progress)
            if status_text is not None:
                status_text.text(f"Processing uploaded video... {int(progress * 100)}%")

        frame_idx += 1

    cap.release()
    if progress_bar is not None:
        progress_bar.progress(1.0)
    if status_text is not None:
        status_text.text("Processing complete.")
    return pd.DataFrame(rows, columns=["timestamp", "score", "threshold", "motion", "decision"])


class ViolenceVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = get_cpu_model()
        self.buffer = deque(maxlen=CLIP_LEN)
        self.losat = AdaptiveLOSAT(alpha=0.8, beta=0.2, init_threshold=0.5)
        self.lock = threading.Lock()

        self.last_proc_time = 0.0
        self.live_violence_streak = 0
        self.latest = {
            "score": 0.0,
            "threshold": 0.5,
            "motion": 0.0,
            "decision": "Non-Violence",
        }

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        clip = None
        do_infer = False
        with self.lock:
            self.buffer.append(img_rgb)
            now = time.time()
            do_infer = len(self.buffer) == CLIP_LEN and (now - self.last_proc_time) >= (1.0 / INFER_FPS)
            if do_infer:
                clip = list(self.buffer)
                self.last_proc_time = now

            alert_now = self.latest["decision"] == "Violence"

        if do_infer and clip is not None:
            motion = compute_motion_metric(clip)
            score = infer_score(self.model, clip)
            threshold, raw_alert = self.losat.update(score, motion)

            strong_alert = raw_alert and score >= LIVE_MIN_SCORE and motion >= LIVE_MIN_MOTION
            if strong_alert:
                self.live_violence_streak += 1
            else:
                self.live_violence_streak = 0

            alert = self.live_violence_streak >= LIVE_CONFIRM_CLIPS
            decision = "Violence" if alert else "Non-Violence"

            with self.lock:
                self.latest = {
                    "score": score,
                    "threshold": threshold,
                    "motion": motion,
                    "decision": decision,
                }
                alert_now = self.latest["decision"] == "Violence"
            log_event(LOG_PATH, score, threshold, motion, decision)

        out = add_alert_border(img_bgr, alert_now)
        return av.VideoFrame.from_ndarray(out, format="bgr24")

    def get_latest(self):
        with self.lock:
            return dict(self.latest)


st.title("LOSAT based Violence detector")
st.caption("3D CNN (R3D-18) + Adaptive LOSAT Thresholding")

ensure_log_file(LOG_PATH)
_ = get_cpu_model()

mode = st.radio("Choose Mode", ["Realtime Webcam", "Upload Video"], horizontal=True)

if mode == "Realtime Webcam":
    if "cam_on" not in st.session_state:
        st.session_state.cam_on = False

    if st.button("Start Webcam"):
        st.session_state.cam_on = True

    left, right = st.columns([2, 1])

    with left:
        ctx = None
        if st.session_state.cam_on:
            ctx = webrtc_streamer(
                key="violence-detector",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={
                    "video": {"width": {"ideal": 640}, "height": {"ideal": 360}, "frameRate": {"ideal": 24}},
                    "audio": False,
                },
                video_processor_factory=ViolenceVideoProcessor,
                async_processing=True,
            )
        else:
            st.info("Click 'Start Webcam' to begin live detection.")

    with right:
        score_ph = st.empty()
        thr_ph = st.empty()
        mot_ph = st.empty()
        dec_ph = st.empty()
        log_ph = st.empty()

    if st.session_state.cam_on and ctx is not None and ctx.state.playing:
        while ctx.state.playing:
            if ctx.video_processor:
                latest = ctx.video_processor.get_latest()
                score_ph.metric("Violence Score", f"{latest['score']:.4f}")
                thr_ph.metric("Adaptive Threshold", f"{latest['threshold']:.4f}")
                mot_ph.metric("Motion Metric", f"{latest['motion']:.4f}")

                decision = latest["decision"]
                if decision == "Violence":
                    dec_ph.error("Alert Status: Violence")
                else:
                    dec_ph.success("Alert Status: Non-Violence")

                df = read_event_log(LOG_PATH, tail=20)
                log_ph.dataframe(df, use_container_width=True, height=350)

            time.sleep(0.1)
    else:
        df = read_event_log(LOG_PATH, tail=20)
        score_ph.metric("Violence Score", "0.0000")
        thr_ph.metric("Adaptive Threshold", "0.5000")
        mot_ph.metric("Motion Metric", "0.0000")
        dec_ph.info("Alert Status: Non-Violence")
        log_ph.dataframe(df, use_container_width=True, height=350)

else:
    uploaded_file = st.file_uploader("Upload Video Clip", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is None:
        st.info("Upload a video to run violence prediction using LOSAT.")
    else:
        if "upload_results" not in st.session_state:
            st.session_state.upload_results = None
            st.session_state.uploaded_name = None

        if st.session_state.uploaded_name != uploaded_file.name:
            st.session_state.upload_results = None
            st.session_state.uploaded_name = uploaded_file.name

        preview_col, result_col = st.columns([0.45, 1.75], vertical_alignment="top")
        with preview_col:
            st.caption("Video Preview")
            uploaded_bytes = uploaded_file.getvalue()
            thumb = extract_video_thumbnail(uploaded_file.getvalue())
            if thumb is not None:
                st.image(thumb)
            else:
                st.info("Preview unavailable.")
            with st.expander("Open video player", expanded=False):
                st.video(uploaded_bytes)
            if st.button("Predict Uploaded Video", use_container_width=True):
                suffix = os.path.splitext(uploaded_file.name)[1] if "." in uploaded_file.name else ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    temp_path = tmp.name

                progress_bar = st.progress(0.0)
                status_text = st.empty()
                with st.spinner("Processing video..."):
                    df_pred = process_uploaded_video(temp_path, get_cpu_model(), progress_bar, status_text)

                try:
                    os.remove(temp_path)
                except OSError:
                    pass

                st.session_state.upload_results = df_pred

        with result_col:
            render_section_heading("Prediction Summary")
            df_pred = st.session_state.upload_results
            if df_pred is None:
                st.info("Run prediction to see the video summary here.")
            elif df_pred.empty:
                st.warning("Could not process video. Please try another file.")
            else:
                violence_count = int((df_pred["decision"] == "Violence").sum())
                non_violence_count = int(len(df_pred) - violence_count)
                violence_ratio = violence_count / float(len(df_pred))
                avg_score = float(df_pred["score"].mean())
                avg_threshold = float(df_pred["threshold"].mean())
                avg_motion = float(df_pred["motion"].mean())
                majority_decision = (
                    "Violence"
                    if violence_ratio >= UPLOAD_VIOLENCE_RATIO and avg_score >= UPLOAD_AVG_SCORE_THRESHOLD
                    else "Non-Violence"
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    render_summary_card("Score", f"{avg_score:.4f}")
                with c2:
                    render_summary_card("Threshold", f"{avg_threshold:.4f}")
                with c3:
                    render_summary_card("Motion", f"{avg_motion:.4f}")

                render_section_heading("Final Alert")
                if majority_decision == "Violence":
                    render_alert_block(f"Violence ({violence_count}/{len(df_pred)} clips, {violence_ratio:.0%})", "violence")
                else:
                    render_alert_block(f"Non-Violence ({non_violence_count}/{len(df_pred)} clips)", "non_violence")

                with st.expander("Per-Clip Predictions", expanded=False):
                    st.dataframe(df_pred, use_container_width=True, height=300)

                    csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Prediction CSV",
                        data=csv_bytes,
                        file_name="uploaded_video_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
