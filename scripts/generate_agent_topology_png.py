from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = REPO_ROOT / "docs" / "current_agent_topology.png"


@dataclass
class NodeSpec:
    key: str
    title: str
    llm_status: str
    llm_color: tuple[int, int, int]
    inputs: list[str]
    outputs: list[str]
    notes: list[str]


NODES = [
    NodeSpec(
        key="route",
        title="1. Route Agent",
        llm_status="LLM: optional",
        llm_color=(46, 204, 113),
        inputs=[
            "clean_record",
            "input_summary",
            "rule route baseline",
        ],
        outputs=[
            "route_result",
            "route_llm_used",
            "route llm audit",
        ],
        notes=[
            "Rule first: detect_route_by_rules()",
            "Heuristic refine: route_agent_refine()",
            "LLM refine: GovernanceLLMBridge.route_refine()",
        ],
    ),
    NodeSpec(
        key="rule_dedup",
        title="2. Rule Dedup Agent",
        llm_status="LLM: no",
        llm_color=(231, 76, 60),
        inputs=[
            "clean_record",
            "markdown_text",
            "dedup manifest",
        ],
        outputs=[
            "rule_dedup_result",
            "content_hash",
            "normalized_hash",
            "record_hash",
        ],
        notes=[
            "Exact duplicate check",
            "Normalized text duplicate check",
            "Manifest update inputs are prepared here",
        ],
    ),
    NodeSpec(
        key="semantic_dedup",
        title="3. Semantic Dedup Agent",
        llm_status="LLM: no",
        llm_color=(231, 76, 60),
        inputs=[
            "clean_record",
            "markdown_text",
            "rule_dedup_result",
            "semantic_store.find_candidates()",
        ],
        outputs=[
            "semantic_dedup_result",
            "semantic_candidate_count",
            "semantic_top_hit_score",
        ],
        notes=[
            "Embedding + vector search",
            "semantic_dedup_agent() applies thresholds",
            "same_series_threshold / similarity_threshold",
        ],
    ),
    NodeSpec(
        key="quality",
        title="4. Quality Agent",
        llm_status="LLM: optional",
        llm_color=(46, 204, 113),
        inputs=[
            "clean_record",
            "markdown_text",
            "rule quality baseline",
        ],
        outputs=[
            "quality_result",
            "quality_llm_used",
            "rag_readiness",
            "training_readiness",
        ],
        notes=[
            "Rule quality: run_rule_quality()",
            "Heuristic refine: quality_agent_refine()",
            "LLM refine: GovernanceLLMBridge.quality_refine()",
        ],
    ),
    NodeSpec(
        key="decision",
        title="5. Decision Agent",
        llm_status="LLM: no",
        llm_color=(231, 76, 60),
        inputs=[
            "route_result",
            "semantic_dedup_result",
            "quality_result",
        ],
        outputs=[
            "decision_result",
            "accept / repair / review / drop",
        ],
        notes=[
            "Deterministic rule gate",
            "Final release authority remains rule-based",
            "No direct LLM decision today",
        ],
    ),
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                r"C:\Windows\Fonts\msyhbd.ttc",
                r"C:\Windows\Fonts\simhei.ttf",
                r"C:\Windows\Fonts\arialbd.ttf",
            ]
        )
    candidates.extend(
        [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    radius: int = 24,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_multiline_text(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    lines: list[str],
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    line_gap: int = 8,
) -> int:
    current_y = y
    for line in lines:
        draw.text((x, current_y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, current_y), line, font=font)
        current_y += (bbox[3] - bbox[1]) + line_gap
    return current_y


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
    draw.line([start, end], fill=color, width=5)
    arrow_size = 12
    ex, ey = end
    draw.polygon(
        [
            (ex, ey),
            (ex - arrow_size, ey - arrow_size),
            (ex + arrow_size, ey - arrow_size),
        ],
        fill=color,
    )


def render_topology(output_path: Path) -> Path:
    width = 2200
    height = 2000
    image = Image.new("RGB", (width, height), (247, 249, 252))
    draw = ImageDraw.Draw(image)

    title_font = load_font(46, bold=True)
    section_font = load_font(28, bold=True)
    body_font = load_font(24)
    small_font = load_font(20)

    draw.text((70, 45), "Current Agent Topology", font=title_font, fill=(23, 43, 77))
    draw.text(
        (72, 105),
        "Rule-first governance flow with selective LLM enhancement on route and quality.",
        font=section_font,
        fill=(74, 85, 104),
    )

    legend_x = 1460
    legend_y = 52
    draw_rounded_box(draw, (legend_x, legend_y, 2100, 215), fill=(255, 255, 255), outline=(206, 214, 224), radius=20, width=2)
    draw.text((legend_x + 24, legend_y + 20), "Legend", font=section_font, fill=(23, 43, 77))
    draw.rounded_rectangle((legend_x + 26, legend_y + 78, legend_x + 56, legend_y + 108), radius=8, fill=(46, 204, 113))
    draw.text((legend_x + 72, legend_y + 74), "LLM optional / can be enabled", font=body_font, fill=(51, 65, 85))
    draw.rounded_rectangle((legend_x + 26, legend_y + 128, legend_x + 56, legend_y + 158), radius=8, fill=(231, 76, 60))
    draw.text((legend_x + 72, legend_y + 124), "No LLM, deterministic or embedding-based", font=body_font, fill=(51, 65, 85))
    draw.text((legend_x + 24, legend_y + 174), "Main graph: route -> rule_dedup -> semantic_dedup -> quality -> decision", font=small_font, fill=(100, 116, 139))

    input_box = (80, 170, 660, 300)
    draw_rounded_box(draw, input_box, fill=(235, 245, 255), outline=(72, 149, 239), radius=24)
    draw.text((110, 195), "Global Inputs", font=section_font, fill=(23, 43, 77))
    draw_multiline_text(
        draw,
        110,
        235,
        [
            "- clean_record",
            "- markdown_text",
            "- batch_id / trace_id / input_summary",
        ],
        font=body_font,
        fill=(51, 65, 85),
        line_gap=10,
    )

    output_box = (1500, 1710, 2100, 1910)
    draw_rounded_box(draw, output_box, fill=(237, 247, 237), outline=(82, 196, 26), radius=24)
    draw.text((1530, 1735), "Global Outputs", font=section_font, fill=(23, 43, 77))
    draw_multiline_text(
        draw,
        1530,
        1775,
        [
            "- GovernanceResult",
            "- review / repair / audit artifacts",
            "- metadata + audit_logs + manifests",
        ],
        font=body_font,
        fill=(51, 65, 85),
        line_gap=10,
    )

    node_left = 360
    node_width = 1480
    node_height = 250
    start_y = 350
    gap_y = 38
    centers: list[tuple[int, int]] = []

    for index, node in enumerate(NODES):
        top = start_y + index * (node_height + gap_y)
        bottom = top + node_height
        box = (node_left, top, node_left + node_width, bottom)
        draw_rounded_box(draw, box, fill=(255, 255, 255), outline=(203, 213, 225), radius=28, width=3)

        draw.rounded_rectangle((node_left + 24, top + 20, node_left + 260, top + 66), radius=12, fill=(31, 41, 55))
        draw.text((node_left + 42, top + 28), node.title, font=section_font, fill=(255, 255, 255))

        draw.rounded_rectangle((node_left + 1210, top + 20, node_left + 1435, top + 66), radius=14, fill=node.llm_color)
        draw.text((node_left + 1238, top + 28), node.llm_status, font=body_font, fill=(255, 255, 255))

        draw.text((node_left + 36, top + 92), "Inputs", font=section_font, fill=(30, 41, 59))
        draw_multiline_text(
            draw,
            node_left + 36,
            top + 132,
            [f"- {line}" for line in node.inputs],
            font=body_font,
            fill=(71, 85, 105),
            line_gap=8,
        )

        draw.text((node_left + 590, top + 92), "Outputs", font=section_font, fill=(30, 41, 59))
        draw_multiline_text(
            draw,
            node_left + 590,
            top + 132,
            [f"- {line}" for line in node.outputs],
            font=body_font,
            fill=(71, 85, 105),
            line_gap=8,
        )

        draw.text((node_left + 1045, top + 92), "Behavior", font=section_font, fill=(30, 41, 59))
        draw_multiline_text(
            draw,
            node_left + 1045,
            top + 132,
            [f"- {line}" for line in node.notes],
            font=small_font,
            fill=(71, 85, 105),
            line_gap=8,
        )

        centers.append((node_left + node_width // 2, bottom))

    input_arrow_start = (input_box[2], (input_box[1] + input_box[3]) // 2)
    input_arrow_end = (node_left, start_y + 42)
    draw.line([input_arrow_start, (260, input_arrow_start[1]), (260, input_arrow_end[1]), input_arrow_end], fill=(72, 149, 239), width=5)
    draw.polygon(
        [
            input_arrow_end,
            (input_arrow_end[0] - 16, input_arrow_end[1] - 10),
            (input_arrow_end[0] - 16, input_arrow_end[1] + 10),
        ],
        fill=(72, 149, 239),
    )

    for idx in range(len(centers) - 1):
        sx = node_left + node_width // 2
        sy = start_y + idx * (node_height + gap_y) + node_height
        ex = sx
        ey = start_y + (idx + 1) * (node_height + gap_y)
        draw_arrow(draw, (sx, sy + 6), (ex, ey - 10), (94, 108, 132))

    last_bottom = start_y + (len(NODES) - 1) * (node_height + gap_y) + node_height
    out_start = (node_left + node_width, last_bottom - 60)
    out_end = (output_box[0], output_box[1] + 70)
    draw.line([out_start, (1940, out_start[1]), (1940, out_end[1]), out_end], fill=(82, 196, 26), width=5)
    draw.polygon(
        [
            out_end,
            (out_end[0] - 16, out_end[1] - 10),
            (out_end[0] - 16, out_end[1] + 10),
        ],
        fill=(82, 196, 26),
    )

    footer_box = (80, 1710, 1360, 1910)
    draw_rounded_box(draw, footer_box, fill=(255, 255, 255), outline=(203, 213, 225), radius=24, width=2)
    draw.text((110, 1735), "Current Summary", font=section_font, fill=(23, 43, 77))
    draw_multiline_text(
        draw,
        110,
        1776,
        [
            "- LLM-intervened nodes: route, quality",
            "- Non-LLM nodes: rule_dedup, semantic_dedup, decision",
            "- Final decision authority stays in deterministic decision_agent()",
            "- Audit captures llm_used, parsed payload, and final adopted outputs",
        ],
        font=body_font,
        fill=(51, 65, 85),
        line_gap=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


def main() -> int:
    saved = render_topology(OUTPUT_PATH)
    print(f"[PASS] saved topology png: {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
