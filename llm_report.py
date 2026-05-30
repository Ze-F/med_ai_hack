import os
import json
import base64
import io
from PIL import Image
import anthropic
from dotenv import load_dotenv

# 加载环境变量 (获取 ANTHROPIC_API_KEY)
load_dotenv()

def _encode_image(img: Image.Image, max_dim: int = 2000) -> tuple[str, str]:
    """将 PIL Image 转换为 Anthropic API 要求的 base64 格式

    Resize so the longest side is at most max_dim. Claude rejects images
    over 8000px on any side, and larger images are slower + more expensive
    without helping posture analysis quality.
    """
    img = img.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((max_dim, max_dim))
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return "image/jpeg", base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_report(
    front_img: Image.Image,
    side_img: Image.Image,
    front_metrics: dict,
    side_metrics: dict,
) -> str:
    """Returns markdown report."""
    
    # 风险预案：如果环境变量中没有 API Key，直接返回 Mock 假数据，不阻塞 UI 开发
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _mock_report(front_metrics, side_metrics)

    client = anthropic.Anthropic(api_key=api_key)

    # 1. 转换两张图片为 Base64
    front_media_type, front_b64 = _encode_image(front_img)
    side_media_type, side_b64 = _encode_image(side_img)

    # 2. 严格遵循文档的 System Prompt (包含判断阈值)
    system_prompt = """You are a posture analysis assistant. You receive measurements and
annotated photos of a person standing front-on and side-on. Identify posture
issues, rate severity (mild/moderate/significant), and give 2-3 actionable
suggestions. Output markdown with sections: Overall Summary, Front-View
Findings, Side-View Findings, Recommendations.

Reference thresholds:
- shoulder_height_diff_pct > 2% → notable; > 5% → significant
- pelvic_tilt_deg > 3° → notable
- head_forward_pct > 0.1 → head forward posture
- body_line_deviation_deg > 5° → posture line off"""

    # 3. 组装带有 Vision 图片与 JSON 文本的 Message
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Front view metrics: {json.dumps(front_metrics)}"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": front_media_type,
                                "data": front_b64,
                            }
                        },
                        {
                            "type": "text",
                            "text": f"Side view metrics: {json.dumps(side_metrics)}"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": side_media_type,
                                "data": side_b64,
                            }
                        },
                        {
                            "type": "text",
                            "text": "Analyze and produce the report."
                        }
                    ]
                }
            ]
        )
        return response.content[0].text
    except Exception as e:
        # API 请求出错时也返回带有报错信息的降级内容
        return f"**Error generating report:** {str(e)}\n\n" + _mock_report(front_metrics, side_metrics)

def _mock_report(front_metrics: dict, side_metrics: dict) -> str:
    """Mock fallback: 在最后5分钟如果 API 没跑通用来应付 Demo 的保底方法"""
    return f"""## Overall Summary
*(Note: This is a generated mock report because the API key is missing or the request failed)*

The subject shows mild postural imbalances, particularly in head forward posture and slight shoulder asymmetry.

## Front-View Findings
- **Shoulder Height Difference:** {front_metrics.get('shoulder_height_diff_pct', 0):.1f}% 
- **Pelvic Tilt:** {front_metrics.get('pelvic_tilt_deg', 0):.1f}° 

## Side-View Findings
- **Head Forward Ratio:** {side_metrics.get('head_forward_pct', 0):.2f}
- **Body Line Deviation:** {side_metrics.get('body_line_deviation_deg', 0):.1f}°

## Recommendations
1. Ensure your computer monitor is at eye level to reduce forward head posture.
2. Incorporate chest-opening stretches to correct shoulder alignment.
3. Take frequent standing breaks to reset your pelvic tilt.
"""


