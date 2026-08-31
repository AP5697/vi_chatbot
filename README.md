# VI Telecom Sales & Customer-Care Copilot

A workbench for VI (Vodafone Idea) care/sales reps — answers **any** plan
question, lets reps compare plans and find the right one for a customer, and
lets an admin replace the dataset without touching code.

Built on **Streamlit** with a choice of two answer engines: a fully offline
**rule-based** one (no API key, no quota) or **Google Gemini** for free-form
natural language.

## Two answer engines

The sidebar lets you switch between them at any time:

| | **Rule-based (no LLM)** | **Gemini-powered** |
| --- | --- | --- |
| API key | not needed | required |
| Network | fully offline | calls Google |
| Quota | none | free-tier limits apply |
| Speed | instant | ~2–10 s per answer |
| Hallucination risk | impossible by construction | very low (grounded + instructed) |
| Free-form phrasing | needs concrete criteria | handles it well |

If no API key is present the app starts in rule-based mode automatically — it
always works.

## Three tabs (Chat · Compare · Plan Finder)

**💬 Chat** — free-form Q&A. Handles plan lookups, field questions (price,
validity, data, 5G, OTT…), filters, superlatives, comparisons, counts,
alternatives to a plan, profile-based recommendations, and multi-turn follow-ups
("what about its validity?", "compare the first two").

**⚖️ Compare** — pick 2–4 plans from a searchable dropdown; get a
side-by-side table with rows that are identical across all plans hidden, leaving
only the differences. Exportable as Markdown.

**🎯 Plan Finder** — guided questionnaire (budget / data usage / validity /
must-haves like 5G, OTT, roaming). Returns the cheapest matching plans with
bullet reasons. If nothing meets every constraint it relaxes the softest one
first and says what it dropped — always gives a usable result.

## Admin panel

A second Streamlit page (**Admin**, auto-listed in the sidebar) lets a
non-technical operator update the dataset without touching code or redeploying:

1. Upload a new plan CSV
2. The panel validates it (schema, duplicate codes, blank cells, non-numeric
   price/validity) and refuses it with exact reasons if anything is wrong
3. Shows a diff — X added · Y removed · Z changed — before anything is touched
4. On confirmation, backs up the current CSV automatically and makes the new
   one live (cache cleared, no restart needed)
5. A **Rollback** tab restores any earlier backup with one click; the restore
   itself is backed up first, so it's always undoable

The admin page is passcode-protected when `ADMIN_PASSCODE` is set.

## Project layout

```
DATA_segregator/
├── input/                              # master catalogue CSV (read-only source)
├── build_wide_dataset.py               # generates the clean wide dataset
├── output/
│   ├── vi_plans_wide_dataset.csv       # the chatbot's data source
│   └── backups/                        # admin-panel backups (git-ignored)
├── app.py                              # Streamlit chat UI  ← run this
├── pages/
│   └── 1_Admin.py                      # dataset admin page (auto-listed by Streamlit)
├── cli.py                              # standalone CLI (no Streamlit)
├── chatbot/
│   ├── catalog.py                      # loads the wide CSV
│   ├── rules_engine.py                 # offline deterministic engine
│   ├── dataset_admin.py                # admin logic (validate / diff / backup)
│   ├── export.py                       # chat/comparison → Markdown export
│   ├── guardrails.py                   # jailbreak + fact-check guards (Gemini mode)
│   ├── prompt.py                       # system prompt builder
│   ├── llm.py                          # Gemini client with 5-model fallback chain
│   ├── eval.py                         # live accuracy check (calls Gemini API)
│   ├── test_rules.py                   # offline rule-engine tests
│   ├── test_guardrails.py              # offline guardrail tests
│   ├── test_llm.py                     # offline fallback-chain tests
│   ├── test_export.py                  # offline export tests
│   └── test_dataset_admin.py           # offline admin-logic tests
├── run_tests.py                        # runs all 5 offline suites at once
├── requirements.txt
└── README.md
```

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build the dataset** (first run only)
   ```bash
   python build_wide_dataset.py
   ```

3. **(Optional) Add a Gemini API key** — skip to run rule-based only.
   Get a free key at https://aistudio.google.com/apikey, then:
   ```bash
   copy .env.example .env
   # edit .env and set GEMINI_API_KEY=your_key_here
   ```
   On Streamlit Cloud use **Settings → Secrets** instead of `.env`.

4. **Run the app**
   ```bash
   streamlit run app.py
   ```
   Opens at http://localhost:8501. The Admin page appears in the sidebar.

## Configuration

| Env var            | Default            | Purpose                                          |
| ------------------ | ------------------ | ------------------------------------------------ |
| `GEMINI_API_KEY`   | *(optional)*       | Required for Gemini mode only                    |
| `GEMINI_MODEL`     | `gemini-3.6-flash` | Preferred model; becomes first in the chain      |
| `ADMIN_PASSCODE`   | *(optional)*       | Locks the Admin page; set before sharing publicly |

## Running tests

113 offline tests (no API key, no network):

```bash
python run_tests.py
```

Covers: rule engine (40), guardrails (24), Gemini fallback chain (11), export
(19), dataset admin (19). All deterministic — safe to run on every change.

Live accuracy test against the Gemini API (spends one free-tier request):

```bash
python -m chatbot.eval
```

## Quota & reliability

Google's free tier meters requests **per model per day**. Two things keep the
app usable within that:

**Fewer tokens per question.** The entire catalogue is injected as context on
every question. Sending it as CSV (header once, not 58 names × 62 plans) and
hoisting the 10 columns identical for every plan reduces **~36,000 → ~14,500
tokens per request (-60%)**.

**Automatic model failover.**

```
gemini-3.6-flash → gemini-flash-latest → gemini-3.5-flash
                 → gemini-flash-lite-latest → gemini-3.5-flash-lite
```

- burst 429 (small retry delay) → waits it out, retries same model
- daily 429 → that model is done today, moves to the next
- 503 "high demand" → backs off, retries, then moves to the next model
- anything else (bad key, bad request) → raised immediately

A hard 45-second ceiling per question prevents a Google-side outage from
hanging a rep on a live call. When everything is spent the UI says so in plain
language and suggests switching to rule-based mode.

## Security

- API key read from environment / `.env` — never hard-coded.
- `.env` is git-ignored.
- Admin panel locked with `ADMIN_PASSCODE` when deployed publicly.
- Dataset stays local; only the question + catalogue context go to Gemini per turn.
- Guardrails block jailbreak/prompt-injection attempts before they reach the model,
  and fact-check plan codes in every Gemini answer.
