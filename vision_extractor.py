"""
Vision Document Extractor
Extracts text and data from financial PDF documents using vision models.

Fixes applied vs original:
  V1 - Model detection: qwen2.5vl / qwen2.5-vl and other variants added to
       vision keyword list. The original missed the model in .env (qwen2.5vl:7b)
       and silently fell back to text-only extraction on every page.

  V2 - Page pre-screening: PyMuPDF text is extracted first on every page.
       The vision API is only called when text is sparse (<= TEXT_THRESHOLD
       chars), i.e. charts, scanned images, or diagram-heavy pages.
       Text-heavy pages (narrative MD&A, footnotes) skip the API call entirely.
       Typical MD&A: 80%+ pages are text-heavy → ~80% fewer vision API calls.

  V3 - Typed PageResult dataclass replaces bare strings.
       method field records how each page was extracted:
         "text"     — PyMuPDF text, sufficient quality, no API call
         "vision"   — vision model called and returned usable content
         "fallback" — vision called but returned <50 chars; PyMuPDF used
         "empty"    — page had no extractable content at all
       Callers (main.py Fix A) can now distinguish failure from blank page.

  V4 - Page-level extraction cache. Results are serialized to a sidecar
       .json file in <pdf_dir>/.extraction_cache/, keyed on an MD5 of
       (pdf_path + mtime + model_name). Re-runs load from cache instead of
       re-calling the vision API. Cache is invalidated automatically when
       the PDF changes or the model is swapped.

  V5 - Shorter, focused vision prompt. The original ~500-token instruction
       block is replaced with a tighter ~150-token version that asks for
       structured extraction without verbose preamble. Reduces per-call
       token cost and gives more consistent outputs across smaller models.

  V6 - Failure/blank distinction surfaced in extraction summary logged at
       the end of extract_from_pdf(). Callers see per-page method counts
       so they can tell "10 pages were blank" from "10 pages failed vision".
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests
from langchain_community.llms import Ollama
from utils import check_model_availability

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

# V2: pages with more than this many chars of PyMuPDF text skip vision entirely
TEXT_THRESHOLD = 100

# V3: minimum chars from vision API to be considered usable
VISION_MIN_CHARS = 50

# V4: cache directory name, created alongside each PDF
CACHE_DIR_NAME = ".extraction_cache"

# V5: focused vision prompt (replaces the original ~500-token block)
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
        method:     How content was obtained:
                      "text"     — PyMuPDF text only, no API call
                      "vision"   — vision model returned usable content
                      "fallback" — vision returned <VISION_MIN_CHARS chars;
                                   PyMuPDF text used instead
                      "empty"    — no content from any method
        char_count: len(content)
        elapsed_s:  Wall-clock seconds spent on this page.
    """
    page_num:  int
    content:   str
    method:    str
    char_count: int = field(init=False)
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        self.char_count = len(self.content)


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class VisionDocumentExtractor:
    """
    Extracts information from PDF documents using vision-language models.

    Key behaviours:
    - Text-heavy pages bypass the vision API (V2 pre-screening).
    - Results are cached per (pdf, mtime, model) so re-runs are fast (V4).
    - Each page is typed with its extraction method (V3).
    - Vision model detection covers qwen2.5vl and other modern variants (V1).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.model_name = model_name or os.getenv("VISION_MODEL", "llama3.2-vision:11b")
        self.base_url   = base_url   or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        print(f"  Initializing Vision Extractor")
        print(f"    Model:    {self.model_name}")
        print(f"    Base URL: {self.base_url}")

        if not FITZ_AVAILABLE:
            print("    WARNING: PyMuPDF (fitz) not available — PDF extraction will fail.")

        check_model_availability(self.model_name)

        # Text-only LLM for the enhancement path (non-vision fallback)
        self.llm = Ollama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.0,
        )

        # V1: expanded keyword list covers qwen2.5vl, qwen2.5-vl, etc.
        self.is_vision_model = self._is_vision_capable()
        print(f"    Vision capable: {'Yes' if self.is_vision_model else 'No'}")
        print(f"    Text threshold: {TEXT_THRESHOLD} chars (pages above skip vision API)")

    # ------------------------------------------------------------------
    # V1 — Model capability detection
    # ------------------------------------------------------------------

    def _is_vision_capable(self) -> bool:
        """
        Return True if the configured model supports image input.

        V1 fix: original list missed qwen2.5vl (the model in .env).
        Rule: any model whose name contains one of the keywords below
        is treated as vision-capable.
        """
        vision_keywords = [
            "vision",           # llama3.2-vision, llava, etc.
            "llava",
            "qwen2-vl",         # original qwen vision series
            "qwen2.5-vl",       # newer dash variant
            "qwen2.5vl",        # no-dash variant — was missing in original
            "llama3.2-vision",
            "minicpm-v",
            "moondream",
            "bakllava",
        ]
        name_lower = self.model_name.lower()
        return any(kw in name_lower for kw in vision_keywords)

    # ------------------------------------------------------------------
    # V4 — Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, pdf_path: str) -> str:
        """MD5 of (absolute path + mtime + model name) — cache key."""
        abs_path = os.path.abspath(pdf_path)
        mtime    = str(os.path.getmtime(pdf_path))
        raw      = f"{abs_path}:{mtime}:{self.model_name}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _cache_path(self, pdf_path: str) -> Path:
        cache_dir = Path(pdf_path).parent / CACHE_DIR_NAME
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{self._cache_key(pdf_path)}.json"

    def _load_cache(self, pdf_path: str) -> Optional[Dict[str, dict]]:
        """
        Load cached page results if they exist.
        Returns a dict keyed by str(page_num), or None on miss.
        """
        cp = self._cache_path(pdf_path)
        if not cp.exists():
            return None
        try:
            with open(cp, encoding="utf-8") as fh:
                data = json.load(fh)
            print(f"    ✓ Cache hit — loading {len(data)} pages from cache")
            return data
        except Exception as exc:
            print(f"    ⚠ Cache read failed ({exc}), re-extracting")
            return None

    def _save_cache(self, pdf_path: str, pages: Dict[str, dict]) -> None:
        """Persist page results to cache file."""
        cp = self._cache_path(pdf_path)
        try:
            with open(cp, "w", encoding="utf-8") as fh:
                json.dump(pages, fh, ensure_ascii=False, indent=2)
            print(f"    ✓ Cache saved → {cp}")
        except Exception as exc:
            print(f"    ⚠ Cache write failed ({exc})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all relevant information from a PDF document.

        Returns a single string with content from all pages, prefixed by
        a document header. Logs a per-page method summary on completion.

        Raises:
            FileNotFoundError: if pdf_path does not exist.
            ImportError:       if PyMuPDF is not installed.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not FITZ_AVAILABLE:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install with: pip install PyMuPDF"
            )

        # V4: attempt cache load
        cached = self._load_cache(pdf_path)

        doc         = fitz.open(pdf_path)
        total_pages = len(doc)
        filename    = os.path.basename(pdf_path)

        print(f"    {total_pages} pages in {filename}")

        page_results: List[PageResult] = []

        for page_num in range(total_pages):
            if cached and str(page_num) in cached:
                # Restore from cache — no API call
                cd = cached[str(page_num)]
                pr = PageResult(
                    page_num=cd["page_num"],
                    content=cd["content"],
                    method=cd["method"],
                    elapsed_s=0.0,
                )
                page_results.append(pr)
            else:
                t0 = time.monotonic()
                pr = self._extract_page(doc, page_num)
                pr.elapsed_s = round(time.monotonic() - t0, 2)
                page_results.append(pr)

                method_icon = {"text": "T", "vision": "V", "fallback": "F", "empty": "-"}
                icon = method_icon.get(pr.method, "?")
                print(
                    f"      [{icon}] page {page_num+1:>3}/{total_pages} "
                    f"— {pr.char_count:>6} chars  ({pr.elapsed_s}s)"
                )

        doc.close()

        # V4: save to cache if we extracted anything fresh
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

        # V6: log extraction summary
        self._log_summary(page_results, filename)

        # Assemble final content string
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

    def extract_from_multiple_pdfs(self, pdf_paths: List[str]) -> Dict[str, str]:
        """Extract content from multiple PDF files."""
        results: Dict[str, str] = {}
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            print(f"\n  Extracting {filename}...")
            try:
                results[filename] = self.extract_from_pdf(pdf_path)
            except Exception as exc:
                print(f"  ✗ Failed to extract {filename}: {exc}")
                results[filename] = f"ERROR: Could not extract content from {filename}: {exc}"
        return results

    # ------------------------------------------------------------------
    # Internal — page extraction
    # ------------------------------------------------------------------

    def _extract_page(self, doc, page_num: int) -> PageResult:
        """
        Extract content from a single page using the best available method.

        Decision tree (V2 pre-screening):
          1. Extract PyMuPDF text.
          2. If text >= TEXT_THRESHOLD chars → return as "text" (no API call).
          3. If vision model available → call vision API.
               a. Vision returns >= VISION_MIN_CHARS → return as "vision".
               b. Vision returns < VISION_MIN_CHARS  → return PyMuPDF text
                  as "fallback" (distinguishable from genuine blank).
          4. If no vision model → LLM-enhance the PyMuPDF text.
          5. If nothing yielded content → return "empty".
        """
        try:
            page         = doc[page_num]
            text_content = page.get_text().strip()

            # V2: text-heavy page — skip vision entirely
            if len(text_content) >= TEXT_THRESHOLD:
                return PageResult(
                    page_num=page_num,
                    content=text_content,
                    method="text",
                )

            # Sparse page — try vision
            if self.is_vision_model:
                vision_content = self._vision_extract_page(page, page_num, text_content)
                if vision_content and len(vision_content.strip()) >= VISION_MIN_CHARS:
                    # V3: genuine vision success
                    return PageResult(
                        page_num=page_num,
                        content=vision_content,
                        method="vision",
                    )
                else:
                    # V3: vision returned too little — use PyMuPDF text if any
                    return PageResult(
                        page_num=page_num,
                        content=text_content,
                        method="fallback" if text_content else "empty",
                    )

            # No vision model — try LLM enhancement of raw text
            if text_content:
                enhanced = self._enhance_with_llm(text_content, page_num)
                return PageResult(
                    page_num=page_num,
                    content=enhanced or text_content,
                    method="text",
                )

            # Nothing worked
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
        Call the Ollama vision API with a focused, shorter prompt (V5).
        Returns the model's response string (may be empty on failure).
        """
        if not FITZ_AVAILABLE:
            return fallback_text

        try:
            mat      = fitz.Matrix(2.0, 2.0)   # 2× zoom for readability
            pix      = page.get_pixmap(matrix=mat)
            img_b64  = base64.b64encode(pix.tobytes("png")).decode("utf-8")

            # V5: use the module-level focused prompt constant
            return self._call_ollama_api_with_image(VISION_PROMPT, img_b64)

        except Exception as exc:
            print(f"      ⚠ Vision extraction failed for page {page_num + 1}: {exc}")
            return fallback_text

    def _call_ollama_api_with_image(self, prompt: str, image_base64: str) -> str:
        """POST to Ollama /api/generate with image payload."""
        url     = f"{self.base_url}/api/generate"
        payload = {
            "model":  self.model_name,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
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
        LLM-enhance sparse PyMuPDF text when no vision model is available.
        Falls back to raw text if the LLM call fails.
        """
        prompt = (
            f"Extract and organize all financial data from this document text.\n\n"
            f"Raw text:\n{text_content[:3000]}\n\n"
            f"Return structured text with all numbers, labels, and periods preserved."
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
        V6: Print a per-method page count so callers can distinguish
        blank pages from vision failures.

        Example output:
            Extraction summary for annual_report.pdf:
              text     :  42 pages   (fast path, no API call)
              vision   :   6 pages   (vision API used)
              fallback :   1 page    (vision returned too little; PyMuPDF used)
              empty    :   1 page    (no content from any method)
        """
        from collections import Counter
        counts = Counter(pr.method for pr in page_results)
        total  = len(page_results)

        print(f"\n    Extraction summary for {filename}:")
        method_notes = {
            "text":     "fast path, no API call",
            "vision":   "vision API used",
            "fallback": "vision returned too little; PyMuPDF used",
            "empty":    "no content from any method",
        }
        for method in ("text", "vision", "fallback", "empty"):
            n = counts.get(method, 0)
            if n:
                label = f"{n} page{'s' if n != 1 else ''}"
                print(f"      {method:<10}: {label:>10}   ({method_notes[method]})")

        total_chars = sum(pr.char_count for pr in page_results)
        vision_calls = counts.get("vision", 0) + counts.get("fallback", 0)
        skipped      = counts.get("text",   0)
        print(f"      {'─'*50}")
        print(f"      Total pages   : {total}")
        print(f"      Vision API calls : {vision_calls}  (skipped {skipped} text-heavy pages)")
        print(f"      Total chars   : {total_chars:,}")


# ---------------------------------------------------------------------------
# CLI test helper (unchanged interface)
# ---------------------------------------------------------------------------

def test_vision_extractor(pdf_path: str) -> None:
    """Test the vision extractor on a single PDF."""
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
        print("\nThis will test the vision extractor on a single PDF file.")