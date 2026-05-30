# med_ai_hack — Posture Analyzer

Upload front + side standing photos → detect joints → measure posture → Claude generates a report.

## Setup

**Requires Python 3.10–3.12.** MediaPipe ≥0.10.22 dropped the `solutions.pose` API, and Python 3.13+ has no wheels with it. Use `python3.12`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in ANTHROPIC_API_KEY
streamlit run app.py
```

App runs without an API key — `llm_report.py` falls back to a mock report.

## Ownership

| File | Owner |
|---|---|
| `app.py` | Ze |
| `pose_detector.py` | Roy |
| `measurements.py` | Roy |
| `llm_report.py` | Harper |

Full spec: [`docs/superpowers/specs/2026-05-30-posture-analyzer-design.md`](docs/superpowers/specs/2026-05-30-posture-analyzer-design.md)
