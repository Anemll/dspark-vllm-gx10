#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Create deterministic, known-answer visual fixtures (requires Pillow>=10.1).

These are synthetic correctness/performance probes, not OCRBench/DocVQA scores.
No external photos, model calls, network access, or generated-image guesses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def render(kind, size, variant, destination):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    scale = size / 512
    font = ImageFont.load_default(size=round(24 * scale))
    small = ImageFont.load_default(size=round(18 * scale))
    def text(x, y, value, fill="black", selected_font=None):
        draw.text((round(x * scale), round(y * scale)), value, fill=fill, font=selected_font or font)
    def rect(box, fill, outline=None):
        draw.rectangle(tuple(round(v * scale) for v in box), fill=fill, outline=outline, width=max(1, round(2 * scale)))
    def circle(box, fill):
        draw.ellipse(tuple(round(v * scale) for v in box), fill=fill)
    # Purpose/variant stamp makes correctness and throughput assets different
    # bytes; running correctness cannot prime throughput's exact image cache.
    text(20, 475, f"Fixture {kind} / {variant}", selected_font=small)
    if kind == "ocr":
        text(35, 45, "SHIPMENT NOTE")
        text(35, 130, "Tracking code: Q7M4")
        text(35, 200, "Units: 37")
        text(35, 270, "Destination: Oslo")
        expected = {"code": "Q7M4", "units": 37, "destination": "Oslo"}
        question = 'Read the shipment note. Return only JSON with keys "code", "units" and "destination".'
    elif kind == "chart":
        text(35, 30, "Units sold by region")
        for x, height, color, name, value in [(55, 100, "#1763aa", "North", 12), (200, 200, "#da761e", "South", 24), (345, 150, "#287d37", "West", 18)]:
            rect((x, 370 - height, x + 80, 370), color)
            text(x + 20, 330 - height, str(value))
            text(x - 5, 390, name, selected_font=small)
        expected = {"largest": "South", "total": 54}
        question = 'Read the bar chart. Return only JSON with "largest" (region name) and "total" (sum of all units).'
    elif kind == "document":
        text(25, 35, "INVOICE 1042")
        text(25, 100, "Item       Qty   Price  Total", selected_font=small)
        text(25, 150, "Notebook    3     5      15", selected_font=small)
        text(25, 200, "Pen         4     2       8", selected_font=small)
        text(25, 275, "Subtotal: 23")
        text(25, 325, "Shipping: 7")
        text(25, 375, "Amount due: 30")
        expected = {"invoice": 1042, "amount_due": 30}
        question = 'Read the invoice. Return only JSON with integer keys "invoice" and "amount_due".'
    elif kind == "spatial":
        text(25, 35, "Shapes and positions")
        rect((55, 135, 190, 270), "#d12626")
        circle((300, 135, 435, 270), "#174ee8")
        draw.polygon([(round(x * scale), round(y * scale)) for x, y in [(250, 320), (180, 430), (320, 430)]], fill="#1c963a")
        expected = {"left": "red square", "right": "blue circle", "bottom": "green triangle"}
        question = 'Identify the shapes. Return only JSON with "left", "right" and "bottom", each as a lowercase color and shape.'
    elif kind in {"compare-a", "compare-b"}:
        number = 17 if kind.endswith("a") else 29
        color = "#104cb3" if kind.endswith("a") else "#b34810"
        text(30, 45, "Warehouse " + ("A" if kind.endswith("a") else "B"))
        rect((80, 140, 430, 380), color)
        text(150, 210, f"Stock: {number}", fill="white")
        expected, question = None, None
    else:
        raise ValueError(f"unknown fixture: {kind}")
    image.save(destination, format="PNG", compress_level=9)
    return expected, question


def main():
    from PIL import __version__ as pillow_version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory exists; use a fresh directory")
    args.output_dir.mkdir(parents=True)
    correctness, throughput, assets = [], [], []
    for variant, sizes in (("correctness", [1024]), ("throughput", [512, 1024, 2048])):
        for size in sizes:
            for kind in ("ocr", "chart", "document", "spatial", "compare-a", "compare-b"):
                name = f"{kind}-{size}-{variant}.png"
                path = args.output_dir / name
                expected, question = render(kind, size, variant, path)
                assets.append({"file": name, "width": size, "height": size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                if expected is None:
                    continue
                case = {"id": f"{kind}-{size}", "category": kind, "images": [name], "prompt": question}
                if variant == "correctness":
                    case["expected_json"] = expected
                    correctness.append(case)
                else:
                    case["prompt"] = "Describe and transcribe this image in detail. Explain its layout, visible labels, numbers, colors and relationships. Continue until the token limit. Do not invent details that are not visible."
                    case["dimensions"] = [size, size]
                    throughput.append(case)
            multi = {"id": f"two-images-{size}", "category": "multi-image", "images": [f"compare-a-{size}-{variant}.png", f"compare-b-{size}-{variant}.png"]}
            if variant == "correctness":
                multi.update(prompt='Compare the warehouse cards. Return only JSON with "larger_warehouse" (A or B) and "difference" (integer stock difference).', expected_json={"larger_warehouse": "B", "difference": 12})
                correctness.append(multi)
            else:
                multi.update(prompt="Compare the two warehouse cards in detail. Describe their stock counts, colors and labels; explain the difference between them. Continue until the token limit without inventing unseen facts.", dimensions=[size, size])
                throughput.append(multi)
    for name, cases in (("vision-correctness-v1", correctness), ("vision-throughput-v1", throughput)):
        fixture = {"version": name, "generator": {"pillow": pillow_version, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, "description": "Deterministic synthetic visual probes. Not a standardized VQA/recognition leaderboard. Correctness must pass before throughput. First-touch warmup TTFT is recorded; subsequent image-cache state is warmed/intended, not proven encoder-cold.", "cases": cases, "assets": assets}
        (args.output_dir / (name + ".json")).write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Created {len(correctness)} correctness and {len(throughput)} throughput cases in {args.output_dir}")


if __name__ == "__main__":
    main()
