"""
web/main.py — FastAPI web interface for the Financial Analysis system.

Adds the project root to sys.path so orchestrator.py and sibling modules
can be imported without Docker volume mounts.
"""
import asyncio
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from orchestrator import FinancialAnalysisOrchestrator
from dotenv import load_dotenv

load_dotenv()

# Resolve the file-share directory to an absolute path once, at import time,
# so uploads and the analysis pipeline write to the same place regardless of
# the process working directory.
_FILE_SHARE_PATH_ABS = Path(
    os.getenv("FILE_SHARE_PATH", "data/financials")
).resolve()
_FILE_SHARE_PATH_ABS.mkdir(parents=True, exist_ok=True)

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
)

# One analysis may run at a time — analysis takes 10-15 minutes and hammers
# the local Ollama server. Serialising prevents resource-exhaustion DoS.
_ANALYSIS_SEMAPHORE = asyncio.Semaphore(1)

# Hard cap on a single analysis run so a hung model or request cannot block
# the server indefinitely.
_ANALYSIS_TIMEOUT_SECONDS = 45 * 60  # 45 minutes

# Upload limits. Files outside these bounds are rejected before any work.
_MIN_PDF_BYTES = 512
_MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB

# Only safe, flat PDF filenames are accepted. Rejects path separators and any
# characters that could traverse or escape the file-share directory.
_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,254}$")

# PDF magic bytes: "%PDF"
_PDF_MAGIC = b"%PDF"


async def _validate_upload(file: UploadFile) -> Path:
    """Validate and write a single upload to the file-share directory.

    Rejects unsafe filenames, non-PDF content, and out-of-range sizes.
    Returns the destination path.
    """
    filename = Path(file.filename or "").name  # strip any directory components
    if not _SAFE_FILENAME_RE.match(filename) or not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Rejected filename '{file.filename}': only flat PDF filenames allowed.",
        )

    content = await file.read()
    size = len(content)
    if size < _MIN_PDF_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' too small to be a financial PDF ({size} bytes).",
        )
    if size > _MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File '{filename}' exceeds the {_MAX_PDF_BYTES}-byte upload limit.",
        )
    if not content.startswith(_PDF_MAGIC):
        raise HTTPException(
            status_code=400,
            detail=f"File '{filename}' is not a valid PDF (missing PDF header).",
        )

    dest_path = Path(_FILE_SHARE_PATH_ABS) / filename
    with open(dest_path, "wb") as fh:
        fh.write(content)
    return dest_path


async def _run_analysis_blocking() -> tuple[str, str]:
    """Run the full analysis pipeline in a worker thread with a timeout."""
    VALUATION_PDF_PATH = os.getenv("VALUATION_PDF_PATH", "data/valuation_parameters.pdf")

    orchestrator = FinancialAnalysisOrchestrator(
        file_share_path=str(_FILE_SHARE_PATH_ABS),
        valuation_pdf_path=VALUATION_PDF_PATH,
    )

    loop = asyncio.get_event_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, orchestrator.run_analysis),
        timeout=_ANALYSIS_TIMEOUT_SECONDS,
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(Path(__file__).parent / "static" / "index.html", "r") as f:
        return f.read()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(files: list[UploadFile] = File(...)):
    try:
        for file_obj in files:
            await _validate_upload(file_obj)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Upload failed: {exc}"})

    if _ANALYSIS_SEMAPHORE.locked():
        return JSONResponse(
            status_code=429,
            content={"error": "An analysis is already running. Please wait for it to finish."},
        )

    async with _ANALYSIS_SEMAPHORE:
        try:
            report, company_name = await _run_analysis_blocking()
            return {"report": report, "company_name": company_name}
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"error": "Analysis exceeded the time limit and was aborted."},
            )
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
