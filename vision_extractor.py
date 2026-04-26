"""
Vision Document Extractor
Extracts text and data from financial PDF documents using vision models.

Fixes applied vs original:
  U1 - __init__ catches OllamaUnavailableError and ModelNotFoundError from
       check_model_availability() and re-raises as RuntimeError with context.
       Previously check_model_availability() called sys.exit(1) which killed
       the process silently with no traceback or actionable error message.

  U2 - Benefit inherited from utils.py: model names with digest suffixes
       (e.g. "qwen2.5vl:7b-q4_K_M") are now correctly recognised.

  V1 - Model detection: qwen2.5vl / qwen2.5-vl and other modern variants
       added to vision keyword list. The original missed "qwen2.5vl:7b"
       (the model configured in .env) and silently fell back to text-only
       extraction on every page.

  V2 - Page pre-screening: PyMuPDF text is extracted first on every page.
       The vision API is only called when text is sparse (<= TEXT_THRESHOLD
       chars), i.e. charts, scanned images, or diagram-heavy pages.
       Text-heavy pages (narrative MD&A, footnotes) skip the API call.
       Typical MD&A: 80%+ pages are text-heavy → ~80% fewer API calls.

  V3 - Typed PageResult dataclass replaces bare strings.
       method field records how each page was extracted:
         "text"     — PyMuPDF text, sufficient quality, no API call
         "vision"   — vision model called and returned usable content
         "fallback" — vision called but returned <VISION_MIN_CHARS chars;
                      PyMuPDF text used instead
         "empty"    — no content from any method
       Callers (main.py Fix A) can now distinguish failure from blank page.

  V4 - Page-level extraction cache. Results serialized to a sidecar .json
       in <pdf_dir>/.extraction_cache/, keyed on MD5 of (path + mtime +
       model_name). Re-runs load from cache — zero API calls. Cache
       invalidates automatically when PDF or model changes.

  V5 - Shorter, focused vision prompt (~150 tokens vs original ~500).
       Reduces per-call token cost and produces more consistent outputs
       across smaller models like qwen2.5vl:7b.

  V6 - Extraction summary logged after every PDF: per-method page counts
       let callers tell "10 pages were blank" from "10 pages failed vision".
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests
from langchain_community.llms import Ollama

from utils import check_model_availability, OllamaUnavailableError, ModelNotFoundError

# PyMuPDF — optional import guard
try:
    import fitz          # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    fitz = None
    FITZ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# V2: pages with >= this many chars of PyMuPDF text skip vision entirely
TEXT_THRESHOLD = 100

# V3: minimum chars from vision API to be considered usable content
VISION_MIN_CHARS = 50

# V4: cache directory name, created alongside each PDF file
CACHE_DIR_NAME = ".extraction_cache"

# V5: focused vision prompt (~150 tokens — replaces original ~500-token block)
VISION_PROMPT = """Extract ALL financial data from this document page image.

Return structured text covering:
- All revenue, expense, profit, and cash flow figures with their labels and periods
- Financial ratios and KPIs (margins, ROE, ROA, EPS, growth rates)
- Table data with row and column headers preserved
- Chart trends described numerically where possible
- Section type: income statement / balance sheet / cash flow / narrative / other

Be precise with numbers. Include units and time periods for every figure.
If the page is blank or contains no financial data, say: EMPTY PAGE."""


# ---------------------------------------------------------------------------
# V3 — Typed page result
# ---------------------------------------------------------------------------

@dataclass
class PageResult:
    """
    Result of extracting a single PDF page.

    Attributes:
        page_num:   0-indexed page number.
        content:    Extracted text (empty string for empty/failed pages).
        method:     How content was obtained —
                      "text"     PyMuPDF text only, no API call
                      "vision"   vision model returned usable content
                      "fallback" vision returned < VISION_MIN_CHARS chars;
                                 PyMuPDF text used instead
                      "empty"    no content from any method
        char_count: len(content), set automatically in __post_init__.
        elapsed_s:  Wall-clock seconds spent on this page.
    """
    page_num:   int
    content:    str
    method:     str
    char_count: int   = field(init=False)
    elapsed_s:  float = 0.0

    def __post_init__(self) -> None:
        self.char_count = len(self.content)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class VisionDocumentExtractor:
    """
    Extracts information from PDF documents using vision-language models.

    Key behaviours (all new vs original):
    - Text-heavy pages bypass the vision API (V2 pre-screening).
    - Results are cached per (pdf, mtime, model) so re-runs are fast (V4).
    - Each page is typed with its extraction method (V3).
    - Vision model detection covers qwen2.5vl and other modern variants (V1).
    - Startup model check raises typed exceptions instead of sys.exit (U1).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialise the vision extractor.

        U1: Catches OllamaUnavailableError / ModelNotFoundError from
            check_model_availability() and re-raises as RuntimeError so
            main.py's top-level handler produces a full traceback.

        Raises:
            RuntimeError: if Ollama is unreachable or the vision model is missing.
            ImportError:  if PyMuPDF is not installed (deferred — raised on first use).
        """
        self.model_name = model_name or os.getenv("VISION_MODEL", "llama3.2-vision:11b")
        self.base_url   = base_url   or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        print("  Initializing Vision Extractor")
        print(f"    Model:          {self.model_name}")
        print(f"    Base URL:       {self.base_url}")
        print(f"    Text threshold: {TEXT_THRESHOLD} chars "
              f"(pages above this skip the vision API)")

        if not FITZ_AVAILABLE:
            print("    WARNING: PyMuPDF (fitz) not installed — "
                  "PDF extraction will fail at runtime.")

        # U1: typed exception handling — no more sys.exit in utils.py
        try:
            check_model_availability(self.model_name)
        except OllamaUnavailableError as exc:
            raise RuntimeError(
                f"Cannot initialise Vision Extractor — Ollama is not reachable.\n{exc}"
            ) from exc
        except ModelNotFoundError as exc:
            raise RuntimeError(
                f"Cannot initialise Vision Extractor — vision model missing.\n{exc}"
            ) from exc

        # Text-only LLM — used when vision is unavailable but text is sparse
        self.llm = Ollama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.0,   # deterministic for extraction
        )

        # V1: expanded keyword list — see _is_vision_capable() below
        self.is_vision_model = self._is_vision_capable()
        print(f"    Vision capable: {'Yes' if self.is_vision_model else 'No'}")

    # ------------------------------------------------------------------
    # V1 — Model capability detection
    # ------------------------------------------------------------------

    def _is_vision_capable(self) -> bool:
        """
        Return True if the configured model supports image input.

        V1 fix: original list contained only "qwen2-vl" and missed
        "qwen2.5vl" (no dash, the format used in .env). This caused
        every page to be processed as text-only silently.

        The check is a substring match against the lowercased model name
        so partial tags like "7b", "q4", etc. don't interfere.
        """
        vision_keywords = [
            "vision",            # llama3.2-vision, llava, etc.
            "llava",
            "qwen2-vl",          # original Qwen vision series
            "qwen2.5-vl",        # newer dash-separated variant
            "qwen2.5vl",         # no-dash variant — was missing in original
            "llama3.2-vision",
            "minicpm-v",
            "moondream",
            "bakllava",
        ]
        name_lower = self.model_name.lower()
        return any(kw in name_lower for kw in vision_keywords)

    # ------------------------------------------------------------------
    # V4 — Page-level extraction cache
    # ------------------------------------------------------------------

    def _cache_key(self, pdf_path: str) -> str:
        """MD5 of (absolute path + mtime + model name) — unique cache key."""
        abs_path = os.path.abspath(pdf_path)
        mtime    = str(os.path.getmtime(pdf_path))
        raw      = f"{abs_path}:{mtime}:{self.model_name}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, pdf_path: str) -> Path:
        """Return the Path object for this PDF's cache file."""
        cache_dir = Path(pdf_path).parent / CACHE_DIR_NAME
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{self._cache_key(pdf_path)}.json"

    def _load_cache(self, pdf_path: str) -> Optional[Dict[str, dict]]:
        """
        Load cached page results if available.

        Returns:
            Dict keyed by str(page_num) on cache hit, None on miss or error.
        """
        cp = self._cache_path(pdf_path)
        if not cp.exists():
            return None
        try:
            with open(cp, encoding="utf-8") as fh:
                data = json.load(fh)
            print(f"    ✓ Cache hit — {len(data)} pages loaded (no API calls needed)")
            return data
        except Exception as exc:
            print(f"    ⚠ Cache read failed ({exc}) — re-extracting from scratch")
            return None

    def _save_cache(self, pdf_path: str, pages: Dict[str, dict]) -> None:
        """Persist page results to the cache file for future runs."""
        cp = self._cache_path(pdf_path)
        try:
            with open(cp, "w", encoding="utf-8") as fh:
                json.dump(pages, fh, ensure_ascii=False, indent=2)
            print(f"    ✓ Extraction cached → {cp}")
        except Exception as exc:
            print(f"    ⚠ Cache write failed ({exc}) — results not cached")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all content from a PDF document.

        On first run: processes each page, calls vision API only for sparse
        pages (V2), saves results to cache (V4), logs summary (V6).

        On subsequent runs with the same PDF and model: loads from cache
        with zero API calls (V4).

        Returns:
            Single string with per-page content, prefixed by document header.
            Page headers include the extraction method for auditability.

        Raises:
            FileNotFoundError: pdf_path does not exist.
            ImportError:       PyMuPDF not installed.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not FITZ_AVAILABLE:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install with: pip install PyMuPDF"
            )

        # V4: try cache first
        cached = self._load_cache(pdf_path)

        doc         = fitz.open(pdf_path)
        total_pages = len(doc)
        filename    = os.path.basename(pdf_path)

        print(f"    Processing {filename} ({total_pages} pages)")

        page_results: List[PageResult] = []
        method_icon = {"text": "T", "vision": "V", "fallback": "F", "empty": "-"}

        for page_num in range(total_pages):
            if cached and str(page_num) in cached:
                # Restore from cache — no API call made
                cd = cached[str(page_num)]
                pr = PageResult(
                    page_num=cd["page_num"],
                    content=cd["content"],
                    method=cd["method"],
                    elapsed_s=0.0,
                )
            else:
                t0 = time.monotonic()
                pr = self._extract_page(doc, page_num)
                pr.elapsed_s = round(time.monotonic() - t0, 2)

                icon = method_icon.get(pr.method, "?")
                print(
                    f"      [{icon}] page {page_num + 1:>3}/{total_pages}"
                    f" — {pr.char_count:>6} chars  ({pr.elapsed_s}s)"
                )

            page_results.append(pr)

        doc.close()

        # V4: save to cache only if we extracted fresh (not from cache)
        if not cached:
            pages_dict = {
                str(pr.page_num): {
                    "page_num": pr.page_num,
                    "content":  pr.content,
                    "method":   pr.method,
                }
                for pr in page_results
            }
            self._save_cache(pdf_path, pages_dict)

        # V6: extraction summary
        self._log_summary(page_results, filename)

        # Assemble final string — page headers include method for auditability
        sections = [
            f"DOCUMENT: {filename}",
            f"Total Pages: {total_pages}",
            "=" * 70,
        ]
        for pr in page_results:
            if pr.content:
                sections.append(f"\n--- PAGE {pr.page_num + 1} ({pr.method}) ---")
                sections.append(pr.content)

        final_content = "\n".join(sections)
        print(f"    ✓ Total extracted: {len(final_content):,} chars")
        return final_content

    def extract_from_multiple_pdfs(
        self, pdf_paths: List[str]
    ) -> Dict[str, str]:
        """Extract content from multiple PDF files."""
        results: Dict[str, str] = {}
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            print(f"\n  Extracting {filename}...")
            try:
                results[filename] = self.extract_from_pdf(pdf_path)
            except Exception as exc:
                print(f"  ✗ Failed to extract {filename}: {exc}")
                results[filename] = (
                    f"ERROR: Could not extract content from {filename}: {exc}"
                )
        return results

    # ------------------------------------------------------------------
    # Internal — page extraction (V2 decision tree)
    # ------------------------------------------------------------------

    def _extract_page(self, doc, page_num: int) -> PageResult:
        """
        Extract content from a single page using the best available method.

        V2 decision tree:
          1. Extract PyMuPDF text.
          2. If len(text) >= TEXT_THRESHOLD → return "text" (no API call).
          3. If vision model available → call vision API.
               a. Response >= VISION_MIN_CHARS → return "vision".
               b. Response <  VISION_MIN_CHARS → return PyMuPDF text as
                  "fallback", or "empty" if PyMuPDF also returned nothing.
          4. No vision model + sparse text → LLM-enhance raw text ("text").
          5. Nothing produced any content → return "empty".
        """
        try:
            page         = doc[page_num]
            text_content = page.get_text().strip()

            # Step 2: text-heavy page — skip vision entirely
            if len(text_content) >= TEXT_THRESHOLD:
                return PageResult(
                    page_num=page_num,
                    content=text_content,
                    method="text",
                )

            # Step 3: sparse page — try vision API
            if self.is_vision_model:
                vision_content = self._vision_extract_page(
                    page, page_num, fallback_text=text_content
                )
                if vision_content and len(vision_content.strip()) >= VISION_MIN_CHARS:
                    return PageResult(
                        page_num=page_num,
                        content=vision_content,
                        method="vision",
                    )
                # Vision returned too little — fall back to PyMuPDF text
                return PageResult(
                    page_num=page_num,
                    content=text_content,
                    method="fallback" if text_content else "empty",
                )

            # Step 4: no vision model — LLM-enhance sparse text if present
            if text_content:
                enhanced = self._enhance_with_llm(text_content, page_num)
                return PageResult(
                    page_num=page_num,
                    content=enhanced or text_content,
                    method="text",
                )

            # Step 5: nothing worked
            return PageResult(page_num=page_num, content="", method="empty")

        except Exception as exc:
            print(f"      ⚠ Could not process page {page_num + 1}: {exc}")
            return PageResult(page_num=page_num, content="", method="empty")

    # ------------------------------------------------------------------
    # V5 — Focused vision prompt
    # ------------------------------------------------------------------

    def _vision_extract_page(
        self,
        page,
        page_num: int,
        fallback_text: str = "",
    ) -> str:
        """
        Render a PDF page to PNG and call the Ollama vision API.

        V5: uses the module-level VISION_PROMPT constant (~150 tokens)
        instead of the original inline ~500-token instruction block.

        Returns:
            Model response string. Empty string on any failure.
        """
        if not FITZ_AVAILABLE:
            return fallback_text

        try:
            mat     = fitz.Matrix(2.0, 2.0)   # 2× zoom for legibility
            pix     = page.get_pixmap(matrix=mat)
            img_b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
            return self._call_ollama_api_with_image(VISION_PROMPT, img_b64)

        except Exception as exc:
            print(f"      ⚠ Vision extraction failed for page {page_num + 1}: {exc}")
            return fallback_text

    def _call_ollama_api_with_image(self, prompt: str, image_base64: str) -> str:
        """POST to Ollama /api/generate with an image payload."""
        url     = f"{self.base_url}/api/generate"
        payload = {
            "model":   self.model_name,
            "prompt":  prompt,
            "images":  [image_base64],
            "stream":  False,
            "options": {"temperature": 0.0},
        }
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.exceptions.RequestException as exc:
            print(f"      ⚠ Ollama API error: {exc}")
            return ""

    def _enhance_with_llm(self, text_content: str, page_num: int) -> str:
        """
        LLM-enhance sparse PyMuPDF text when no vision model is configured.
        Falls back to raw text if the LLM call fails.
        """
        prompt = (
            "Extract and organize all financial data from this document text.\n\n"
            f"Raw text:\n{text_content[:3000]}\n\n"
            "Return structured text with all numbers, labels, and periods preserved."
        )
        try:
            return self.llm.invoke(prompt)
        except Exception as exc:
            print(f"      ⚠ LLM enhancement failed for page {page_num + 1}: {exc}")
            return text_content

    # ------------------------------------------------------------------
    # V6 — Extraction summary
    # ------------------------------------------------------------------

    @staticmethod
    def _log_summary(page_results: List[PageResult], filename: str) -> None:
        """
        V6: Log a per-method page count after every PDF is processed.

        Lets operators distinguish blank pages from vision failures at a
        glance, rather than having to count lines in the verbose log.

        Example output:
            Extraction summary for annual_report.pdf:
              text      :  42 pages   (fast path, no API call)
              vision    :   6 pages   (vision API used)
              fallback  :   1 page    (vision returned too little; PyMuPDF used)
              empty     :   1 page    (no content from any method)
              ──────────────────────────────────────────────────
              Total pages      : 50
              Vision API calls : 7   (skipped 42 text-heavy pages)
              Total chars      : 128,450
        """
        counts = Counter(pr.method for pr in page_results)
        total  = len(page_results)

        method_notes = {
            "text":     "fast path, no API call",
            "vision":   "vision API used",
            "fallback": "vision returned too little; PyMuPDF used",
            "empty":    "no content from any method",
        }

        print(f"\n    Extraction summary for {filename}:")
        for method in ("text", "vision", "fallback", "empty"):
            n = counts.get(method, 0)
            if n:
                label = f"{n} page{'s' if n != 1 else ''}"
                print(f"      {method:<10}: {label:>10}   ({method_notes[method]})")

        vision_calls = counts.get("vision", 0) + counts.get("fallback", 0)
        skipped      = counts.get("text",   0)
        total_chars  = sum(pr.char_count for pr in page_results)

        print(f"      {'─' * 50}")
        print(f"      Total pages      : {total}")
        print(f"      Vision API calls : {vision_calls}"
              f"   (skipped {skipped} text-heavy pages)")
        print(f"      Total chars      : {total_chars:,}")


# ---------------------------------------------------------------------------
# CLI test helper
# ---------------------------------------------------------------------------

def test_vision_extractor(pdf_path: str) -> None:
    """Run the extractor on a single PDF and print results."""
    print(f"\nTesting Vision Extractor on: {pdf_path}")
    print("=" * 80)
    extractor = VisionDocumentExtractor()
    content   = extractor.extract_from_pdf(pdf_path)
    print("\nExtracted Content:")
    print("=" * 80)
    print(content)
    print("=" * 80)
    print(f"\nTotal characters extracted: {len(content)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_vision_extractor(sys.argv[1])
    else:
        print("Usage: python vision_extractor.py <path_to_pdf>")
        print("       Tests the vision extractor on a single PDF file.")