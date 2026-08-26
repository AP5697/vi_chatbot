"""
llm.py
======

Google Gemini client for the VI Copilot, with quota-aware resilience.

Install:  pip install google-genai
API key:  GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment or a .env file.
          Get one at https://aistudio.google.com/apikey

WHY THE FALLBACK CHAIN
----------------------
Gemini's free tier meters requests per project PER MODEL PER DAY
(quotaId "GenerateRequestsPerDayPerProjectPerModel-FreeTier"), and the newest
models have the smallest allowances -- gemini-3.6-flash gives only 20/day. Since
each model has its own separate counter, exhausting one does not touch the
others. So instead of dying on a 429, we transparently move down a chain of
models, which multiplies the usable free requests per day.

Short 429s (a per-minute burst limit, which report a small retryDelay) are
handled differently: we simply wait and retry the same model, because that quota
refills in seconds.
"""

from __future__ import annotations

import os
import re
import time

# Primary first, then fall back in order. Each entry has its own daily quota.
# Ordered newest/highest-quality first; the "-lite" models are last because
# they are the weakest but usually have the largest free allowance.
DEFAULT_MODEL_CHAIN = [
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "gemini-3.5-flash-lite",
]

# A 429 reporting a retryDelay at or below this many seconds is a short-window
# limit worth waiting out; anything longer means the daily quota is gone and we
# should switch models instead of blocking the user.
SHORT_RETRY_CEILING_SECONDS = 15

# Attempts per model before moving on, and the backoff between them.
ATTEMPTS_PER_MODEL = 3
BACKOFF_SECONDS = [1.0, 3.0]

# Hard ceiling on one reply(). Without it, a Google-side outage could walk the
# whole chain (models x attempts x backoff) and leave a rep on a live call
# staring at a spinner for minutes. Better to stop early and say "try again".
MAX_TOTAL_SECONDS = 45.0

# Server-side hiccups: the model is fine, it is just busy or briefly broken.
# These are worth retrying and then failing over to a different model.
TRANSIENT_MARKERS = (
    "503", "UNAVAILABLE",        # "model is currently experiencing high demand"
    "500", "INTERNAL",
    "502", "504", "DEADLINE_EXCEEDED",
)


def _get_setting(name: str) -> str | None:
    """
    Look up a config value from the environment, falling back to Streamlit's
    secrets store.

    Streamlit Cloud injects values from secrets.toml into `st.secrets`, but
    NOT reliably into `os.environ` across all Streamlit versions -- so relying
    on os.getenv() alone can silently fail after a deploy even though the
    secret is set correctly in the dashboard. Checking st.secrets explicitly
    makes this work regardless of that version behaviour.

    Safe to call outside Streamlit (e.g. from chatbot/eval.py on the CLI):
    importing streamlit or reading st.secrets with no secrets.toml present
    both raise, and are simply treated as "not found there".
    """
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name)
    except Exception:
        return None


def get_api_key() -> str | None:
    return _get_setting("GEMINI_API_KEY") or _get_setting("GOOGLE_API_KEY")


def resolve_model_chain() -> list[str]:
    """
    Build the model chain. GEMINI_MODEL, if set, becomes the primary model and
    the defaults follow it as fallbacks (deduplicated, order preserved).
    """
    chain = list(DEFAULT_MODEL_CHAIN)
    preferred = (_get_setting("GEMINI_MODEL") or "").strip()
    if preferred:
        chain = [preferred] + [m for m in chain if m != preferred]
    return chain


class CopilotUnavailableError(RuntimeError):
    """No model in the chain could answer. Base class for the two reasons."""


class QuotaExhaustedError(CopilotUnavailableError):
    """Every model in the chain is out of free-tier quota for today."""


class ServiceUnavailableError(CopilotUnavailableError):
    """Every model in the chain was busy/erroring server-side (503 etc.)."""


def _is_quota_error(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _is_transient_error(exc: Exception) -> bool:
    text = str(exc)
    return any(marker in text for marker in TRANSIENT_MARKERS)


def _retry_delay_seconds(exc: Exception) -> float | None:
    """Pull the server-suggested retry delay out of a 429, if it gave one."""
    match = re.search(r"retry in ([0-9.]+)s", str(exc))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    match = re.search(r"'retryDelay':\s*'(\d+)s'", str(exc))
    if match:
        return float(match.group(1))
    return None


class GeminiCopilot:
    """
    Wrapper holding the system instruction (the grounded plan data) and the
    model fallback chain. Conversation state lives in the caller, so the same
    instance can serve reruns without accumulating hidden history.
    """

    def __init__(self, system_instruction: str, model: str | None = None,
                 api_key: str | None = None, temperature: float = 0.2,
                 max_total_seconds: float = MAX_TOTAL_SECONDS,
                 valid_plan_ids: set[str] | None = None, guardrails: bool = True):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The google-genai package is required. Install it with:\n"
                "    pip install google-genai"
            ) from exc

        key = api_key or get_api_key()
        if not key:
            raise RuntimeError(
                "No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY) "
                "in your environment or a .env file. Get one at "
                "https://aistudio.google.com/apikey"
            )

        self._types = types
        # Disable the SDK's OWN retry loop (default: several tenacity retries on
        # 503, ~30s per call). We run our own multi-MODEL fallback chain, so a
        # single model must fail FAST -- otherwise the SDK's internal retries eat
        # the whole time budget on model #1 and we never reach the fallbacks.
        try:
            http_options = types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1)
            )
            self._client = genai.Client(api_key=key, http_options=http_options)
        except Exception:
            # Older SDK without retry_options: fall back to default client.
            self._client = genai.Client(api_key=key)
        self._chain = [model] + [m for m in resolve_model_chain() if m != model] \
            if model else resolve_model_chain()
        self._config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )
        self._max_total_seconds = max_total_seconds
        self._guardrails = guardrails
        self._valid_plan_ids = valid_plan_ids or set()
        # The model that answered most recently, so the UI can show it.
        self.active_model = self._chain[0]

    @property
    def model_chain(self) -> list[str]:
        return list(self._chain)

    def _to_contents(self, history: list[dict]):
        """Map [{'role': 'user'|'assistant', 'content': str}] to Gemini format."""
        types = self._types
        contents = []
        for msg in history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
            )
        return contents

    def reply(self, history: list[dict]) -> str:
        """
        Answer the conversation, with guardrails when enabled.

        Input guard runs first: a jailbreak / prompt-injection attempt is
        refused WITHOUT calling the API (saving a request). Otherwise the model
        answers, and the output is fact-checked (cited plan codes must exist)
        before being returned.
        """
        if not self._guardrails:
            return self._generate(history)

        from . import guardrails as gr

        question = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"), ""
        )
        refusal = gr.check_input(question)
        if refusal:
            return refusal
        answer = self._generate(history)
        return gr.verify_output(answer, self._valid_plan_ids)

    def _generate(self, history: list[dict]) -> str:
        """
        Send the conversation and return the answer text.

        Walks the model chain, distinguishing three kinds of failure:
          * short-window 429  -> wait the server-suggested delay, retry same model
          * daily-quota 429   -> that model is done for today, try the next one
          * 503 / 500 / 504   -> transient overload: back off, retry, then next model
          * anything else     -> a real error (bad key, bad request): raise at once

        Raises QuotaExhaustedError or ServiceUnavailableError only after every
        model in the chain has been tried.
        """
        contents = self._to_contents(history)
        out_of_quota: list[str] = []
        unavailable: list[str] = []
        last_error: Exception | None = None
        deadline = time.monotonic() + self._max_total_seconds

        for model in self._chain:
            if time.monotonic() >= deadline:
                break
            for attempt in range(ATTEMPTS_PER_MODEL):
                if time.monotonic() >= deadline:
                    break
                try:
                    resp = self._client.models.generate_content(
                        model=model, contents=contents, config=self._config,
                    )
                    self.active_model = model
                    return (resp.text or "").strip() or (
                        "I couldn't generate a response for that. Please rephrase."
                    )
                except Exception as exc:
                    last_error = exc
                    is_last_attempt = attempt == ATTEMPTS_PER_MODEL - 1

                    remaining = deadline - time.monotonic()

                    if _is_quota_error(exc):
                        delay = _retry_delay_seconds(exc)
                        # A small retryDelay means a per-minute burst limit,
                        # which refills in seconds -- worth waiting out once,
                        # but only if we still have the time budget for it.
                        if (attempt == 0 and delay is not None
                                and delay <= SHORT_RETRY_CEILING_SECONDS
                                and delay + 0.5 < remaining):
                            time.sleep(delay + 0.5)
                            continue
                        out_of_quota.append(model)
                        break

                    if _is_transient_error(exc):
                        backoff = BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]
                        if not is_last_attempt and backoff < remaining:
                            time.sleep(backoff)
                            continue
                        unavailable.append(model)
                        break

                    raise  # not retryable -- surface it immediately

        # Every model failed. Report the dominant reason so the message is useful.
        if out_of_quota and not unavailable:
            raise QuotaExhaustedError(
                "Every configured Gemini model is out of free-tier quota for today "
                f"({', '.join(out_of_quota)}). Free-tier quotas reset daily. To "
                "continue now, enable billing at https://aistudio.google.com/apikey."
            ) from last_error

        raise ServiceUnavailableError(
            "No Gemini model could answer right now. "
            + (f"Out of quota: {', '.join(out_of_quota)}. " if out_of_quota else "")
            + (f"Busy or erroring: {', '.join(unavailable)}. " if unavailable else "")
            + "This is usually a temporary spike in demand -- try again in a moment."
        ) from last_error
