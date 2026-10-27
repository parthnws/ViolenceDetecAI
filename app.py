"""Streamlit UI for real-time violence detection.  Run: streamlit run app.py"""
import os
import tempfile
import time
from datetime import datetime

import cv2
import streamlit as st

from detector import LEVELS, ViolenceDetector

st.set_page_config(page_title="Violence Detection System", page_icon="🚨", layout="wide")

EMERGENCY = [
    ("🚨 Unified Emergency (all services)", "112"),
    ("👮 Police", "100"),
    ("🚑 Ambulance", "102 / 108"),
    ("🔥 Fire", "101"),
    ("👩 Women Helpline", "1091 / 181"),
    ("🧒 Child Helpline", "1098"),
    ("🚓 Cyber Crime", "1930"),
]

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
.lvl {padding: 14px; border-radius: 10px; color: white; font-weight: 700;
      font-size: 1.3rem; text-align: center;}
.contact {padding: 8px 12px; border-left: 4px solid #c62828; background: #fafafa;
          margin-bottom: 6px; border-radius: 4px; color: #222;}
</style>""", unsafe_allow_html=True)

# ---------------- sidebar ----------------
with st.sidebar:
    st.title("⚙️ Settings")
    source = st.radio("Video source", ["Upload video", "Webcam / CCTV (RTSP URL)"])
    st.subheader("Severity thresholds")
    t1 = st.slider("Level 1 – Low", 0.1, 0.9, 0.50, 0.01)
    t2 = st.slider("Level 2 – Medium", 0.1, 0.95, 0.70, 0.01)
    t3 = st.slider("Level 3 – High", 0.1, 1.0, 0.88, 0.01)
    skip = st.slider("Process every Nth frame", 1, 5, 2)
    st.divider()
    st.subheader("📞 Emergency Contacts (India)")
    for label, num in EMERGENCY:
        st.markdown(f"<div class='contact'><b>{label}</b><br>{num}</div>",
                    unsafe_allow_html=True)
    st.caption("If someone is in danger, call 112 immediately.")


@st.cache_resource
def load_detector(a, b, c):
    return ViolenceDetector(t1=a, t2=b, t3=c)


det = load_detector(t1, t2, t3)
det.t1, det.t2, det.t3 = t1, t2, t3

st.title("🚨 AI Violence Detection – Video Surveillance")
if det.mode == "pretrained":
    st.success("Using pretrained violence model (downloaded automatically, no training needed).")
elif det.mode == "demo":
    st.warning("Demo mode: no violence model could be loaded, so a motion-only heuristic is "
               "used. Check your internet connection or add `models/violence_model.keras`.")
if det.weapon_model is None:
    st.info("Weapon detection is off: model could not be loaded.")
for note in det.load_notes:
    st.caption(f"⚠️ {note}")

# ---------------- source ----------------
cap, path = None, None
if source == "Upload video":
    up = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if up:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1])
        tmp.write(up.read())
        tmp.close()
        path = tmp.name
else:
    path = st.text_input("Webcam index (0) or RTSP/HTTP URL", "0")

start = st.button("▶ Start analysis", type="primary", disabled=not path)

col_v, col_s = st.columns([3, 2])
frame_slot = col_v.empty()
with col_s:
    level_slot = st.empty()
    m1, m2 = st.columns(2)
    score_slot, weapon_slot = m1.empty(), m2.empty()
    chart_slot = st.empty()
    st.subheader("Incident log")
    log_slot = st.empty()
stop = st.button("⏹ Stop")

if start and path:
    src = int(path) if str(path).isdigit() else path
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        st.error("Could not open the video source.")
        st.stop()
    det.buffer.clear(); det.smooth.clear(); det.prev_gray = None
    scores, log, last_lvl, i = [], [], 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % skip:
            continue
        frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        r = det.process(frame)
        frame_slot.image(cv2.cvtColor(r.frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        name, color = LEVELS[r.level]
        level_slot.markdown(f"<div class='lvl' style='background:{color}'>{name}</div>",
                            unsafe_allow_html=True)
        score_slot.metric("Violence score", f"{r.violence_score:.2f}")
        wnames = ", ".join(sorted({w[0] for w in r.weapons})) or "None"
        weapon_slot.metric("Weapon", wnames)
        scores.append(r.violence_score)
        chart_slot.line_chart(scores[-150:], height=150)

        if r.level > last_lvl and r.level > 0:
            log.insert(0, {"Time": datetime.now().strftime("%H:%M:%S"),
                           "Level": name, "Score": round(r.violence_score, 2),
                           "Weapon": wnames})
            log_slot.dataframe(log, use_container_width=True)
            if r.level == 3:
                st.toast("HIGH severity – call 112 immediately!", icon="🚨")
        last_lvl = r.level
        if stop:
            break
        time.sleep(max(0, 1 / fps - 0.005))
    cap.release()
    st.success("Analysis finished.")
