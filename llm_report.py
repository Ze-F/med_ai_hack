"""Claude API posture report generator. Owner: Harper."""

import base64
import io
import json
import os

from PIL import Image

# from anthropic import Anthropic
# client = Anthropic()  # reads ANTHROPIC_API_KEY from env


SYSTEM_PROMPT = """You are a posture analysis assistant. You receive measurements
and annotated photos of a person standing front-on and side-on. Identify posture
issues, rate severity (mild / moderate / significant), and give 2-3 actionable
suggestions per finding.

Reference thresholds:
- shoulder_height_diff_pct > 2% → notable; > 5% → significant
- pelvic_tilt_deg > 3° → notable
- knee_angle_deg < 170° → notable knee bend
- head_forward_pct > 0.10 → forward head posture
- body_line_deviation_deg > 5° → posture line off vertical
- pelvic_rotation_deg > 5° → pelvic rotation

Output markdown with sections:
## Overall Summary
## Front-View Findings
## Side-View Findings
## Recommendations
"""


def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def generate_report(
    front_img: Image.Image,
    side_img: Image.Image,
    front_metrics: dict,
    side_metrics: dict,
) -> str:
    """Send metrics + annotated images to Claude, return markdown report."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return _mock_report(front_metrics, side_metrics)

    # TODO Harper: real implementation
    #   from anthropic import Anthropic
    #   client = Anthropic()
    #   message = client.messages.create(
    #       model="claude-sonnet-4-6",
    #       max_tokens=1500,
    #       system=SYSTEM_PROMPT,
    #       messages=[{
    #           "role": "user",
    #           "content": [
    #               {"type": "text", "text": f"Front view metrics: {json.dumps(front_metrics)}"},
    #               {"type": "image", "source": {
    #                   "type": "base64", "media_type": "image/png",
    #                   "data": _img_to_b64(front_img)}},
    #               {"type": "text", "text": f"Side view metrics: {json.dumps(side_metrics)}"},
    #               {"type": "image", "source": {
    #                   "type": "base64", "media_type": "image/png",
    #                   "data": _img_to_b64(side_img)}},
    #               {"type": "text", "text": "Analyze and produce the report."},
    #           ],
    #       }],
    #   )
    #   return message.content[0].text
    return _mock_report(front_metrics, side_metrics)


def _mock_report(front_metrics: dict, side_metrics: dict) -> str:
    return f"""## Overall Summary
*(Mock report — set `ANTHROPIC_API_KEY` to get a real Claude analysis.)*

Posture appears within normal range based on placeholder measurements.

## Front-View Findings
- Shoulder height difference: {front_metrics.get('shoulder_height_diff_pct', 0):.1f}%
- Pelvic tilt: {front_metrics.get('pelvic_tilt_deg', 0):.1f}°
- Knee alignment: {front_metrics.get('knee_alignment', 'n/a')}

## Side-View Findings
- Head forward: {side_metrics.get('head_forward_pct', 0):.2f}
- Body line deviation: {side_metrics.get('body_line_deviation_deg', 0):.1f}°
- Pelvic rotation: {side_metrics.get('pelvic_rotation_deg', 0):.1f}°

## Recommendations
1. Mock recommendation — replace with Claude output.
2. Mock recommendation — replace with Claude output.
"""
