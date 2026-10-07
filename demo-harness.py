import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def run_llamaparse(pdf_path: Path) -> tuple[bool, str | None, str | None]:
    api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    if not api_key:
        return False, None, "SKIPPED: LLAMA_CLOUD_API_KEY not set."

    try:
        import markdown as md_lib
        from llama_cloud import AsyncLlamaCloud

        # images_dir = pdf_path.parent / "images"

        async def _parse() -> str:
            client = AsyncLlamaCloud(api_key=api_key)
            file_obj = await client.files.create(file=str(pdf_path), purpose="parse")

            result = await client.parsing.parse(
                file_id=file_obj.id,
                tier="agentic",
                version="latest",
                expand=["markdown"],
            )

            # markdown is paginated: result.markdown.pages[i].markdown
            md = "\n\n".join(p.markdown for p in result.markdown.pages)

            return md

        combined_md = asyncio.run(_parse())
        body = md_lib.markdown(combined_md, extensions=["tables", "fenced_code"])
        html = f"<html><body>{body}</body></html>"
        return True, html, None
    except Exception:
        return False, None, traceback.format_exc()


TOOLS = {
    "llamaparse": run_llamaparse,
}


def wrap_html_doc(title: str, body_html: str) -> str:
    if "<html" in body_html.lower():
        return body_html
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
</head>
<body>
{body_html}
</body>
</html>"""


def build_index_html(file_name: str, results: dict) -> str:
    rows = []
    for tool, r in results.items():
        status = (
            "✅ Success"
            if r["success"]
            else "❌ "
            + ("Skipped" if r.get("error", "").startswith("SKIPPED") else "Failed")
        )
        link = f'<a href="{tool}.html">{tool}.html</a>' if r["success"] else "—"
        duration = (
            f"{r['duration_seconds']:.1f}s"
            if r.get("duration_seconds") is not None
            else "—"
        )
        rows.append(
            f"<tr><td>{tool}</td><td>{status}</td><td>{duration}</td><td>{link}</td></tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Parsed — {file_name}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 700px; margin: 40px auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; }}
  th {{ background: #f2f2f2; }}
</style>
</head>
<body>
<h1>Parsed: {file_name}</h1>
<table>
<tr><th>Tool</th><th>Status</th><th>Time</th><th>Output</th></tr>
{"".join(rows)}
</table>
</body>
</html>"""


def run_harness(pdf_path: Path, outdir: Path):
    if not pdf_path.exists():
        log(f"ERROR: file not found: {pdf_path}")
        sys.exit(1)

    file_name = pdf_path.stem
    result_dir = outdir / file_name
    result_dir.mkdir(parents=True, exist_ok=True)

    log(f"Running harness on: {pdf_path}")
    log(f"Output directory:   {result_dir}")

    results = {}

    for tool_name, runner in TOOLS.items():
        log(f"--- Running {tool_name} ---")
        start = time.time()
        success, html, error = runner(pdf_path)
        duration = time.time() - start

        if success and html:
            out_file = result_dir / f"{tool_name}.html"
            out_file.write_text(
                wrap_html_doc(f"{tool_name} — {file_name}", html), encoding="utf-8"
            )
            log(f"{tool_name}: OK ({duration:.1f}s) -> {out_file}")
        else:
            if error and error.startswith("SKIPPED"):
                log(f"{tool_name}: SKIPPED ({error})")
            else:
                log(f"{tool_name}: FAILED ({duration:.1f}s)")
                if error:
                    log(error.strip().splitlines()[-1])

        results[tool_name] = {
            "success": success,
            "duration_seconds": duration,
            "error": error,
        }

    report = {
        "paper": file_name,
        "source_pdf": str(pdf_path),
        "run_at": datetime.now(timezone.utc).isoformat(),
        "results": {
            tool: {
                "success": r["success"],
                "duration_seconds": round(r["duration_seconds"], 2),
                "error": (r["error"] if not r["success"] else None),
            }
            for tool, r in results.items()
        },
    }
    (result_dir / "run_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    log("--- Done ---")


def main():
    ap = argparse.ArgumentParser(description="RUN LlamaParse on a PDF.")
    ap.add_argument("pdf_path", type=str)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    run_harness(Path(args.pdf_path), Path(args.outdir))


if __name__ == "__main__":
    main()
