"""
Utility Functions

Fixes applied vs original:
  U1 - OllamaUnavailableError and ModelNotFoundError replace sys.exit(1).
       Callers can now catch each failure mode independently and handle it
       with proper error messages and stack traces instead of a silent exit.

  U2 - Substring match replaces exact equality check for model names.
       Ollama frequently returns names with digest suffixes or quantisation
       tags (e.g. "qwen2.5vl:7b-q4_K_M"). The original exact match failed
       on these and silently killed the process even when the model was
       present and working.

  U3 - 5-second timeout added to the HTTP request. A hung or slow Ollama
       server previously blocked the process indefinitely at startup with
       no feedback to the user.
"""
from __future__ import annotations

import json
import os
import urllib.request
from urllib.error import URLError
from urllib.request import urlopen

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# U1 — Typed exceptions
# ---------------------------------------------------------------------------

class OllamaUnavailableError(RuntimeError):
    """
    Raised when the Ollama server cannot be reached.
    Covers: connection refused, DNS failure, timeout, non-200 HTTP status.
    """


class ModelNotFoundError(RuntimeError):
    """
    Raised when the requested model is not present in the Ollama registry.
    The exception message includes the list of available models and the
    exact pull command needed to fix the issue.
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_model_availability(model_name: str) -> None:
    """
    Verify that model_name is available in the local Ollama instance.

    U1: Raises typed exceptions instead of calling sys.exit(1).
        Callers in agents.py and vision_extractor.py catch these and
        re-raise as RuntimeError with context, so main.py's top-level
        handler produces a full traceback instead of a silent exit code.

    U2: Uses substring matching so digest-suffixed or quantisation-tagged
        model names are correctly recognised.
        e.g. "qwen2.5vl:7b"   matches "qwen2.5vl:7b-q4_K_M"
             "gemma3:12b"      matches "gemma3:12b-it-qat-Q4_K_M"
             "nomic-embed-text" matches "nomic-embed-text:latest"

    U3: 5-second connect/read timeout prevents an indefinite hang when
        Ollama is slow to respond at startup.

    Args:
        model_name: The model name as configured in .env
                    (e.g. "qwen2.5vl:7b", "gemma3:12b").

    Raises:
        OllamaUnavailableError: Server not reachable or returned non-200.
        ModelNotFoundError:     model_name not found in the registry.
    """
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    url = f"{ollama_base_url}/api/tags"

    # --- Fetch model list from Ollama ---
    try:
        req = urllib.request.Request(url)
        # U3: explicit timeout — prevents indefinite hang
        with urlopen(req, timeout=5) as response:
            if response.status != 200:
                raise OllamaUnavailableError(
                    f"Ollama returned HTTP {response.status} from {url}.\n"
                    f"  Ensure Ollama is running and accessible."
                )
            data = json.loads(response.read().decode())

    except URLError as exc:
        raise OllamaUnavailableError(
            f"Could not connect to Ollama at {ollama_base_url}.\n"
            f"  Ensure the Ollama server is running.\n"
            f"  Original error: {exc}"
        ) from exc

    except TimeoutError as exc:
        raise OllamaUnavailableError(
            f"Connection to Ollama at {ollama_base_url} timed out after 5 seconds.\n"
            f"  Check that Ollama is running and not overloaded."
        ) from exc

    except json.JSONDecodeError as exc:
        raise OllamaUnavailableError(
            f"Ollama returned non-JSON from {url}.\n"
            f"  Original error: {exc}"
        ) from exc

    # --- U2: substring match against all available model names ---
    available: list[str] = [m["name"] for m in data.get("models", [])]

    matched_name = next(
        (name for name in available
         if model_name in name or name.startswith(model_name)),
        None,
    )

    if matched_name is None:
        available_str = (
            "\n    ".join(available) if available else "(no models installed)"
        )
        raise ModelNotFoundError(
            f"Model '{model_name}' not found in Ollama registry.\n"
            f"  Fix: ollama pull {model_name}\n"
            f"  Available models:\n    {available_str}"
        )

    # Log when we matched a suffixed variant so it's visible in the output
    if matched_name != model_name:
        print(
            f"  ✓ '{model_name}' matched registry entry '{matched_name}'"
        )
    else:
        print(f"  ✓ Model '{model_name}' confirmed available in Ollama")


# ---------------------------------------------------------------------------
# Pipeline helper functions (pure Python, no external deps)
# ---------------------------------------------------------------------------

MAX_CONTEXT_CHARS = 12_000

FALLBACK_VALUATION_QUERY = (
    "What are all the valuation parameters, methodologies, discount rates, "
    "and comparable multiples?"
)


def extract_summary_block(result: str) -> str:
    """Pull the === SUMMARY === block; fall back to last paragraph."""
    start = result.find("=== SUMMARY ===")
    end   = result.find("=== END SUMMARY ===")
    if start != -1 and end != -1:
        return result[start : end + len("=== END SUMMARY ===")]
    paragraphs = [p.strip() for p in result.split("\n\n") if p.strip()]
    return paragraphs[-1] if paragraphs else result[-500:]


def cap_context(context: str) -> str:
    """Truncate context to MAX_CONTEXT_CHARS, keeping the most recent content."""
    if len(context) <= MAX_CONTEXT_CHARS:
        return context
    truncated = context[-MAX_CONTEXT_CHARS:]
    first_newline = truncated.find("\n")
    if first_newline != -1:
        truncated = truncated[first_newline:]
    return "[Earlier analysis truncated to stay within context limits]\n" + truncated


def build_valuation_rag_query(context: str) -> str:
    """
    Build a targeted RAG query from the growth analyst's summary block.
    Falls back to a generic query if parsing fails.
    """
    summary  = extract_summary_block(context)
    metrics  = ""
    findings = ""

    for line in summary.splitlines():
        line = line.strip()
        if line.startswith("KEY_METRICS:"):
            metrics  = line.replace("KEY_METRICS:",  "").strip()
        elif line.startswith("KEY_FINDINGS:"):
            findings = line.replace("KEY_FINDINGS:", "").strip()

    if not metrics and not findings:
        return FALLBACK_VALUATION_QUERY

    parts = [
        "What valuation methodologies, discount rates, and multiples apply "
        "to a company with:"
    ]
    if metrics:
        parts.append(f"these financial metrics: {metrics}")
    if findings:
        parts.append(f"and these characteristics: {findings}")
    parts.append(
        "Include DCF parameters, comparable company multiples, "
        "and any relevant benchmarks."
    )
    return " ".join(parts)