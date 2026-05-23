"""
web/main.py — FastAPI web interface for the Financial Analysis system.

Adds the project root to sys.path so orchestrator.py and sibling modules
can be imported without Docker volume mounts.
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from orchestrator import FinancialAnalysisOrchestrator
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent / "static"),
    name="static",
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
    FILE_SHARE_PATH    = os.getenv("FILE_SHARE_PATH",    "data/financials")
    VALUATION_PDF_PATH = os.getenv("VALUATION_PDF_PATH", "data/valuation_parameters.pdf")

    os.makedirs(FILE_SHARE_PATH, exist_ok=True)

    for file in files:
        dest_path = os.path.join(FILE_SHARE_PATH, file.filename)
        with open(dest_path, "wb") as f:
            content = await file.read()
            f.write(content)

    try:
        orchestrator = FinancialAnalysisOrchestrator(
            file_share_path=FILE_SHARE_PATH,
            valuation_pdf_path=VALUATION_PDF_PATH,
        )
        loop = asyncio.get_event_loop()
        report, company_name = await loop.run_in_executor(
            None, orchestrator.run_analysis
        )
        return {"report": report, "company_name": company_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
