# Posture Analyzer — Hackathon MVP Spec

**Time budget:** 1.5 hours, 4 people. Goal: working demo, not production code.

## What we're building

Upload front + side standing photos → detect joints → compute posture metrics → Claude generates a posture report. Streamlit UI.

## Tech stack

- **UI:** Streamlit
- **Pose detection:** MediaPipe Pose (33 landmarks, CPU, no model download wait)
- **Math:** numpy
- **LLM:** Claude API, model `claude-sonnet-4-6`, vision enabled
- **Env:** Python 3.10+, single `.env` for `ANTHROPIC_API_KEY`

Fallback if MediaPipe install breaks on someone's machine: `ultralytics` YOLOv8-pose.

## File layout & ownership

```
app.py              # Ze     — Streamlit UI
pose_detector.py    # Roy    — MediaPipe wrapper
measurements.py     # Harper — metric formulas
llm_report.py       # Eric   — Claude API + prompt
requirements.txt    # Anyone — pin deps
.env.example        # Anyone — template
```

## Contracts between files (LOCK THESE FIRST)

These signatures are the only thing blocking parallel work. Agree on them before splitting up. If you change one, tell the others.

```python
# pose_detector.py
def detect_pose(image: PIL.Image.Image) -> tuple[np.ndarray | None, PIL.Image.Image]:
    """
    Returns (landmarks, annotated_image).
    landmarks: shape (33, 4) — x, y, z, visibility. Normalized 0-1.
                None if no person detected.
    annotated_image: original image with skeleton drawn on top.
    """
```

```python
# measurements.py
def measure_front(landmarks: np.ndarray) -> dict:
    """
    Returns:
      {
        "shoulder_height_diff_pct": float,   # |L-R shoulder y| / shoulder_width, %
        "pelvic_tilt_deg": float,            # angle of L-R hip line from horizontal
        "knee_alignment": str,               # "normal" | "X-leg" | "O-leg"
        "knee_angle_deg": float,             # hip-knee-ankle angle, avg of L+R
      }

def measure_side(landmarks: np.ndarray) -> dict:
    """
    Returns:
      {
        "head_forward_pct": float,           # ear-x minus shoulder-x, / shoulder-hip dist
        "body_line_deviation_deg": float,    # angle of shoulder-hip-ankle from vertical
        "pelvic_rotation_deg": float,        # hip-knee line angle from vertical (proxy)
      }
```

```python
# llm_report.py
def generate_report(
    front_img: PIL.Image.Image,         # annotated
    side_img: PIL.Image.Image,          # annotated
    front_metrics: dict,
    side_metrics: dict,
) -> str:
    """Returns markdown report."""
```

## Metric formulas (Harper — concrete starting points)

MediaPipe landmark indices: https://google.github.io/mediapipe/solutions/pose

| Metric | Landmarks | Formula |
|---|---|---|
| shoulder_height_diff_pct | 11 (L shoulder), 12 (R shoulder) | `abs(y11 - y12) / abs(x11 - x12) * 100` |
| pelvic_tilt_deg | 23 (L hip), 24 (R hip) | `degrees(atan2(y23 - y24, x23 - x24))` |
| knee_angle_deg | 23-25-27 (L hip-knee-ankle), 24-26-28 (R) | inner angle at knee; <170° suggests X/O |
| knee_alignment | 25, 26 vs 27, 28 | knees-closer-than-ankles → X; opposite → O |
| head_forward_pct | 7 or 8 (ear), 11 or 12 (shoulder) | `(ear_x - shoulder_x) / dist(shoulder, hip)` |
| body_line_deviation_deg | 11, 23, 27 (shoulder-hip-ankle, one side) | angle of shoulder→ankle line from vertical |
| pelvic_rotation_deg | 23, 25 (or 24, 26) | angle of hip→knee from vertical |

For side view, pick the side that's facing the camera (larger visibility score on landmarks 7 or 8). All thresholds for "normal vs abnormal" go in the LLM prompt, not in code.

## LLM prompt structure (Eric — starting template)

```
System: You are a posture analysis assistant. You receive measurements and
annotated photos of a person standing front-on and side-on. Identify posture
issues, rate severity (mild/moderate/significant), and give 2-3 actionable
suggestions. Output markdown with sections: Overall Summary, Front-View
Findings, Side-View Findings, Recommendations.

User content blocks:
  - text: "Front view metrics: {json.dumps(front_metrics)}"
  - image: front_img (annotated)
  - text: "Side view metrics: {json.dumps(side_metrics)}"
  - image: side_img (annotated)
  - text: "Analyze and produce the report."
```

Reference thresholds to include in the system prompt:
- shoulder_height_diff_pct > 2% → notable; > 5% → significant
- pelvic_tilt_deg > 3° → notable
- head_forward_pct > 0.1 → head forward posture
- body_line_deviation_deg > 5° → posture line off

Use `anthropic` SDK, `client.messages.create(model="claude-sonnet-4-6", max_tokens=1500, ...)`.

## UI layout (Ze)

```
[Title: Posture Analyzer]

[ Upload front photo ]   [ Upload side photo ]
        |                         |
        v                         v
  [annotated front]        [annotated side]
        |_________________________|
                    |
            [Analyze button]
                    |
                    v
        ===== Metrics Table =====
        ===== LLM Report (markdown) =====
```

Use `st.columns(2)` for the two upload boxes. `st.image()` for annotated previews. `st.markdown()` for the report. Cache pose detection with `@st.cache_data` on the image bytes so re-runs are instant.

## requirements.txt

```
streamlit>=1.30
mediapipe>=0.10
opencv-python>=4.8
numpy
pillow
anthropic>=0.40
python-dotenv
```

## Risks & fallbacks

| Risk | Fallback |
|---|---|
| MediaPipe install fails | swap to `ultralytics` YOLOv8-pose (different landmark indices, but same wrapper API) |
| No API key ready at start | Eric writes mock `generate_report` returning hardcoded markdown; swap in last 5 min |
| Pose detection misses on bad photo | Show user-friendly "couldn't detect person, try a clearer photo" |
| Time runs out | UI + pose + measurements alone is already a demo-able product; skip LLM, show metrics table |

## Out of scope (do NOT build)

- Video / live camera
- Sitting / movement postures
- User accounts, history, saved reports
- Multiple people in one photo
- Mobile-responsive UI tweaks
- Tests beyond a manual smoke test

## Definition of done

1. Run `streamlit run app.py` locally
2. Upload two sample photos (find on Google Images during the hour)
3. See: annotated photos, metrics table, Claude-generated report
4. Demo it
