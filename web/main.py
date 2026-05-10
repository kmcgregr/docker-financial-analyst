from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from pathlib import Path
import sys, os
sys.path.append(os.path.abspath(".."))
from main import FinancialAnalysisOrchestrator
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Serve static files (for JS, CSS if needed)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    # Load simple HTML template
    with open(Path(__file__).parent / "static" / "index.html", "r") as f:
        return f.read()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(files: list[UploadFile] = File(...)):
    FILE_SHARE_PATH = os.getenv("FILE_SHARE_PATH", "data/financials")
    VALUATION_PDF_PATH = os.getenv("VALUATION_PDF_PATH", "data/valuation_parameters.pdf")
    COMPANY_NAME = os.getenv("COMPANY_NAME", "Demo Company")

    # Ensure directories exist
    os.makedirs(FILE_SHARE_PATH, exist_ok=True)

    # Save uploaded files
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
        report = orchestrator.run_analysis(COMPANY_NAME)
        return {"report": report}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
