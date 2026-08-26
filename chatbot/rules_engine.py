"""
rules_engine.py
===============

A deterministic, zero-dependency answer engine for the VI plan catalogue.

No LLM, no API key, no network, no quota. Questions are parsed with keyword and
regex rules, executed as pandas filters against the wide dataset, and rendered
with fixed templates. Because every answer is built from DataFrame values, the
engine cannot state a fact that isn't in the data -- hallucination is structurally
impossible rather than merely discouraged.

It exposes the same `.reply(history)` interface as GeminiCopilot, so app.py can
switch between the two without any other change.

WHAT IT HANDLES
    - single-field lookups     "price of the 449 plan", "does 26106058 have 5G"
    - full plan details        "complete details of plan 26106058"
    - filtered search          "plans under 500 with unlimited voice and 5G"
    - superlatives             "cheapest plan with 2GB/day", "longest validity"
    - comparisons              "compare 299 and 449"
    - counts                   "how many prepaid plans have OTT"
    - benefit search           "which plans include Netflix"
    - alternatives             "alternatives to plan 449" (similar live plans)
    - recommendations          "recommend a plan for a heavy data user"
    - follow-ups               "what about its validity" (reuses the last plan)
    - positional follow-ups    "compare the first two", "details of the 2nd one"

WHAT IT DOES NOT HANDLE
    Open-ended reasoning with no concrete criterion ("surprise me"). Those get
    an honest "give me a priority" reply rather than a guess. Recommendations
    map a customer profile (budget / heavy data / traveller / OTT / long
    validity) to a transparent sort -- they are not free-form persuasion.
"""

from __future__ import annotations

import re

import pandas as pd

# --------------------------------------------------------------------------
# Column vocabulary. Keyword -> (column, human label). Order matters: the
# first matching keyword in a question wins, so put specific terms before
# generic ones ("daily data" before "data").
# --------------------------------------------------------------------------
FIELD_SYNONYMS: list[tuple[tuple[str, ...], str, str]] = [
    (("plan code", "soc id", "soc_id", "plan id"), "Plan_SOC_ID", "plan code"),
    (("gst", "base tariff", "tariff breakdown"), "Plan_Rental_GST", "price with GST"),
    (("price", "cost", "rental", "rent", "how much", "charge"), "Plan_Rental", "price"),
    (("validity", "valid for", "how many days", "duration"), "Validity_Days", "validity"),
    (("daily data", "data per day", "per day data"), "Daily_Data_Limit", "daily data"),
    (("total data", "data pool"), "Total_Data_Limit", "total data"),
    (("rollover", "roll over"), "Data_Rollover", "data rollover"),
    (("night data",), "Night_Data", "night data"),
    (("weekend data",), "Weekend_Data", "weekend data"),
    (("fup", "fair usage"), "FUP_Limit", "FUP limit"),
    (("data",), "Data_Benefit", "data"),
    (("voice", "call", "calling", "minutes"), "Voice_Benefit", "voice"),
    (("sms", "text message"), "SMS_Benefit", "SMS"),
    (("5g",), "5G_Eligible", "5G"),
    (("4g",), "4G_Eligible", "4G"),
    (("ott", "netflix", "prime", "hotstar", "sonyliv", "zee5", "streaming", "subscription"),
     "OTT_Platform", "OTT"),
    (("video",), "Video_Benefit", "video"),
    (("roaming",), "Roaming_Benefit", "roaming"),
    (("isd", "international call"), "ISD_Benefit", "ISD"),
    (("circle", "state", "region", "available in"), "Go_Live_Circle", "circles"),
    (("recharge",), "Recharge_Channel", "recharge channels"),
    (("activation", "activate"), "Activation_Channel", "activation channels"),
    (("segment",), "Segment", "segment"),
    (("prepaid", "postpaid"), "Product_Type", "product type"),
    (("auto renewal", "auto-renewal"), "Auto_Renewal", "auto renewal"),
    (("status", "active"), "Plan_Status", "status"),
    (("brief", "summary", "about"), "Plan_Brief", "summary"),
    (("additional benefit", "extra benefit", "other benefit"), "Additional_Benefits",
     "additional benefits"),
]

# Fields shown, in order, for a "complete details" request.
DETAIL_FIELDS = [
    ("Plan_SOC_ID", "Plan code"),
    ("Plan_Rental", "Price"),
    ("Validity_Days", "Validity"),
    ("Product_Type", "Type"),
    ("Plan_Type", "Category"),
    ("Segment", "Segment"),
    ("Voice_Benefit", "Voice"),
    ("Data_Benefit", "Data"),
    ("Daily_Data_Limit", "Daily data"),
    ("Total_Data_Limit", "Total data"),
    ("SMS_Benefit", "SMS"),
    ("5G_Eligible", "5G"),
    ("OTT_Platform", "OTT"),
    ("Roaming_Benefit", "Roaming"),
    ("ISD_Benefit", "ISD"),
    ("Data_Rollover", "Rollover"),
    ("Additional_Benefits", "Additional benefits"),
    ("Circle_Count", "Circles live"),
    ("Plan_Status", "Status"),
]

# Feature filters: keyword -> (column, predicate on the cell's lowercase text).
FEATURE_FILTERS: list[tuple[tuple[str, ...], str, str]] = [
    (("unlimited voice", "unlimited call"), "Voice_Benefit", "unlimited"),
    (("5g",), "5G_Eligible", "yes"),
    (("rollover", "roll over"), "Data_Rollover", "yes"),
    (("prepaid",), "Product_Type", "prepaid"),
    (("postpaid",), "Product_Type", "postpaid"),
]

# OTT brands recognised in questions, matched against OTT_Platform text.
OTT_BRANDS = ("netflix", "hotstar", "disney", "prime", "sonyliv", "zee5", "vi movies")

NO_VALUE_MARKERS = ("no ", "not ", "nil", "none")


# --------------------------------------------------------------------------
# Value parsing
# --------------------------------------------------------------------------

def _num(value) -> float | None:
    """Parse a numeric-looking cell ('449', '1.5') into a float."""
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _daily_gb(value) -> float | None:
    """
    Extract GB/day from cells like '1.5 GB/Day'.

    'No Daily Cap (Total 2 GB Pool)' deliberately returns None: the 2 GB there
    is a total pool, not a daily rate, and treating it as daily would make
    "plans with 2GB/day" quietly wrong.
    """
    text = str(value)
    if "no daily cap" in text.lower():
        return None
    match = re.search(r"([\d.]+)\s*GB\s*/\s*Day", text, re.I)
    return float(match.group(1)) if match else None


def _is_negative(value) -> bool:
    """True when a cell means absence ('No OTT Bundled', 'Not Included')."""
    low = str(value).strip().lower()
    return low.startswith(NO_VALUE_MARKERS) or low in {"", "na", "n/a"}


def _rupees(value) -> str:
    n = _num(value)
    return f"₹{int(n)}" if n is not None and n == int(n) else f"₹{value}"


def _days_text(value) -> str:
    """'1 day' / '28 days' -- the catalogue has genuine 1-day booster packs."""
    return f"{value} day" if _num(value) == 1 else f"{value} days"


# --------------------------------------------------------------------------
# Question parsing
# --------------------------------------------------------------------------

def find_plans(question: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve plan references in the question, most specific first:
    exact SOC ID, then plan name, then price ("the 449 plan").
    """
    q = question.lower()

    ids = [i for i in re.findall(r"\b\d{6,10}\b", question)
           if i in set(df["Plan_SOC_ID"])]
    if ids:
        return df[df["Plan_SOC_ID"].isin(ids)]

    by_name = df[df["Plan_Name"].str.lower().apply(lambda n: n in q)]
    if not by_name.empty:
        return by_name

    # "compare 299 and 449", "the ₹449 plan"
    numbers = re.findall(r"₹?\s*(\d{2,4})\b", question)
    prices = {float(n) for n in numbers}
    if prices:
        by_price = df[df["Plan_Rental"].apply(_num).isin(prices)]
        if not by_price.empty:
            return by_price

    return df.iloc[0:0]


def parse_price_filters(question: str) -> list[tuple[str, float]]:
    """Extract price constraints as (operator, value) pairs."""
    q = question.lower()
    out: list[tuple[str, float]] = []

    between = re.search(r"between\s*₹?\s*(\d+)\s*(?:and|-|to)\s*₹?\s*(\d+)", q)
    if between:
        lo, hi = sorted((float(between.group(1)), float(between.group(2))))
        return [(">=", lo), ("<=", hi)]

    for pattern, op in (
        (r"(?:under|below|less than|cheaper than|upto|up to|within|max)\s*₹?\s*(\d+)", "<="),
        (r"(?:over|above|more than|greater than|at least|min)\s*₹?\s*(\d+)", ">="),
    ):
        for m in re.finditer(pattern, q):
            out.append((op, float(m.group(1))))
    return out


def parse_data_filter(question: str) -> tuple[str, float] | None:
    """Extract a daily-data constraint like '2GB/day' or 'more than 1.5 gb'."""
    q = question.lower()
    m = re.search(r"(?:more than|at least|over|above|minimum|min)\s*([\d.]+)\s*gb", q)
    if m:
        return (">=", float(m.group(1)))
    m = re.search(r"(?:under|less than|below|max)\s*([\d.]+)\s*gb", q)
    if m:
        return ("<=", float(m.group(1)))
    m = re.search(r"([\d.]+)\s*gb\s*(?:/|\s*per\s*)?\s*day", q)
    if m:
        return (">=", float(m.group(1)))
    return None


def parse_validity_filter(question: str) -> tuple[str, float] | None:
    q = question.lower()
    m = re.search(r"(?:more than|at least|over|above)\s*(\d+)\s*days?", q)
    if m:
        return (">=", float(m.group(1)))
    m = re.search(r"(?:under|less than|below|within)\s*(\d+)\s*days?", q)
    if m:
        return ("<=", float(m.group(1)))
    m = re.search(r"(\d+)\s*days?\s*(?:validity|plan|pack)?", q)
    if m:
        return ("==", float(m.group(1)))
    return None


def detect_field(question: str) -> tuple[str, str] | None:
    """Map the question to a single dataset column, if it asks about one."""
    q = question.lower()
    for keywords, column, label in FIELD_SYNONYMS:
        if any(k in q for k in keywords):
            return column, label
    return None


def apply_filters(question: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply every constraint found in the question. Returns (rows, descriptions)."""
    q = question.lower()
    out = df
    applied: list[str] = []

    for op, value in parse_price_filters(question):
        prices = out["Plan_Rental"].apply(_num)
        out = out[prices <= value] if op == "<=" else out[prices >= value]
        applied.append(f"price {op} ₹{int(value)}")

    data_filter = parse_data_filter(question)
    if data_filter:
        op, value = data_filter
        gb = out["Daily_Data_Limit"].apply(_daily_gb)
        keep = gb >= value if op == ">=" else gb <= value
        out = out[keep.fillna(False)]
        applied.append(f"daily data {op} {value}GB")

    if any(w in q for w in ("validity", "days", "day plan")):
        vf = parse_validity_filter(question)
        if vf:
            op, value = vf
            days = out["Validity_Days"].apply(_num)
            keep = {">=": days >= value, "<=": days <= value, "==": days == value}[op]
            out = out[keep.fillna(False)]
            applied.append(f"validity {op} {int(value)} days")

    # Words already consumed by a feature phrase must not be reused below as a
    # Plan_Type match -- otherwise "unlimited voice" would also filter
    # Plan_Type == "Voice", silently dropping most of the real results.
    consumed: set[str] = set()

    for keywords, column, needle in FEATURE_FILTERS:
        matched = [k for k in keywords if k in q]
        if matched:
            out = out[out[column].str.lower().str.contains(needle, na=False)]
            applied.append(f"{column.replace('_', ' ').lower()} = {needle}")
            for phrase in matched:
                consumed.update(phrase.split())

    brands = [b for b in OTT_BRANDS if b in q]
    if brands:
        pattern = "|".join(re.escape(b) for b in brands)
        out = out[out["OTT_Platform"].str.lower().str.contains(pattern, na=False)]
        applied.append(f"OTT includes {', '.join(brands)}")
    elif any(w in q for w in ("ott", "streaming", "subscription")) and "no ott" not in q:
        out = out[~out["OTT_Platform"].apply(_is_negative)]
        applied.append("has OTT")

    for plan_type in df["Plan_Type"].str.lower().unique():
        # "data" and "voice" are also generic benefit words, so only treat them
        # as a Plan_Type filter when they weren't already used as a feature.
        if plan_type in consumed or plan_type == "data":
            continue
        if re.search(rf"\b{re.escape(plan_type)}\b", q):
            out = out[out["Plan_Type"].str.lower() == plan_type]
            applied.append(f"plan type = {plan_type}")
            break

    return out, applied


# --------------------------------------------------------------------------
# Answer rendering
# --------------------------------------------------------------------------

def _plan_label(row: pd.Series) -> str:
    return f"**{row['Plan_Name']}** ({row['Plan_SOC_ID']})"


def _format_cell(column: str, value) -> str:
    """
    Display formatting for the columns the CSV stores as bare numbers.

    Total_Data_Limit is the one that actually misleads: shown raw it reads as
    "Total data: 84", which a rep could take for days or rupees rather than GB.
    Every other column already carries its own unit in the data.
    """
    if column == "Plan_Rental":
        return _rupees(value)
    if column == "Validity_Days":
        return _days_text(value)
    if column == "Total_Data_Limit" and _num(value) is not None:
        return f"{value} GB"
    return str(value)


def _table_cell(column: str, value) -> str:
    """
    Markdown-table-safe version of _format_cell.

    Several dataset columns legitimately contain a '|' as a separator --
    "Bundled OTT Subscriptions | Vi Movies & TV Access", "Retail Outlet | Cash",
    the GST breakdown, all the Tagging_* fields. Dropped into a Markdown table
    raw, that pipe opens a phantom column and silently shifts every value after
    it, so the rep reads the wrong plan's benefit. Escaping keeps the cell
    intact and still displays as a plain '|'.
    """
    return _format_cell(column, value).replace("|", "\\|")


def render_field(rows: pd.DataFrame, column: str, label: str) -> str:
    return "\n\n".join(
        f"{_plan_label(r)} — {label}: {_format_cell(column, r[column])}"
        for _, r in rows.iterrows()
    )


def render_details(row: pd.Series) -> str:
    lines = [f"**{row['Plan_Name']}**", ""]
    for column, label in DETAIL_FIELDS:
        if column not in row.index:
            continue
        lines.append(f"- **{label}:** {_format_cell(column, row[column])}")
    return "\n".join(lines)


def render_table(rows: pd.DataFrame, limit: int = 25) -> str:
    shown = rows.head(limit)
    columns = ["Plan_Name", "Plan_SOC_ID", "Plan_Rental", "Validity_Days",
               "Daily_Data_Limit", "Voice_Benefit", "5G_Eligible", "OTT_Platform"]
    header = "| Plan | Code | Price | Validity | Data | Voice | 5G | OTT |"
    sep = "| --- | --- | ---: | ---: | --- | --- | --- | --- |"
    body = [
        "| " + " | ".join(_table_cell(c, r[c]) for c in columns) + " |"
        for _, r in shown.iterrows()
    ]
    table = "\n".join([header, sep] + body)
    if len(rows) > limit:
        table += f"\n\n_Showing {limit} of {len(rows)} matches._"
    return table


def find_alternatives(target: pd.Series, df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    """
    Rank plans similar to `target` by a transparent distance score.

    Similarity = closeness in price + validity + daily data, restricted to the
    same product type (prepaid/postpaid) so we don't offer a postpaid plan as an
    "alternative" to a prepaid one. Deterministic and explainable -- no ML, no
    LLM. The target itself and withdrawn/legacy plans are excluded.
    """
    def voice_class(v) -> str:
        return "unlimited" if "unlimited" in str(v).lower() else "limited"

    pool = df[df["Plan_SOC_ID"] != target["Plan_SOC_ID"]]
    pool = pool[pool["Plan_Status"].str.lower() == "live"]
    same_type = pool[pool["Product_Type"] == target["Product_Type"]]
    if not same_type.empty:
        pool = same_type
    # Keep the same voice class where possible: a data-only booster is not a
    # sensible "alternative" to an unlimited-voice plan, even if its price and
    # validity happen to be close.
    same_voice = pool[pool["Voice_Benefit"].apply(voice_class)
                      == voice_class(target["Voice_Benefit"])]
    if len(same_voice) >= k:
        pool = same_voice

    t_price = _num(target["Plan_Rental"]) or 0.0
    t_days = _num(target["Validity_Days"]) or 0.0
    t_gb = _daily_gb(target["Daily_Data_Limit"])

    def score(row) -> float:
        # Normalised absolute differences so no single axis dominates.
        s = 0.0
        p = _num(row["Plan_Rental"])
        if p is not None and t_price:
            s += abs(p - t_price) / t_price
        d = _num(row["Validity_Days"])
        if d is not None and t_days:
            s += abs(d - t_days) / t_days
        g = _daily_gb(row["Daily_Data_Limit"])
        if t_gb is not None:
            # Penalise a missing daily rate instead of skipping it, so a
            # "No Daily Cap" plan can't score artificially close by dropping
            # this axis entirely.
            s += abs(g - t_gb) / t_gb if g is not None else 1.0
        return s

    scored = pool.assign(_score=pool.apply(score, axis=1)).sort_values("_score")
    return scored.drop(columns="_score").head(k)


def render_alternatives(target: pd.Series, alts: pd.DataFrame) -> str:
    if alts.empty:
        return (f"No comparable live alternatives to **{target['Plan_Name']}** "
                f"were found in the dataset.")
    head = (f"Alternatives to **{target['Plan_Name']}** "
            f"({_rupees(target['Plan_Rental'])}, {target['Validity_Days']} days, "
            f"{target['Daily_Data_Limit']}):\n")
    return head + "\n" + render_table(alts)


# Customer profiles -> the filter/sort intent behind a "recommend for X" request.
RECOMMEND_PROFILES: list[tuple[tuple[str, ...], str, str]] = [
    (("heavy data", "lot of data", "lots of data", "heavy user", "streaming",
      "binge"), "data", "most daily data"),
    (("budget", "cheap", "affordable", "low cost", "student", "save money",
      "tight"), "cheap", "lowest price"),
    (("light user", "basic", "minimal", "occasional", "senior", "elderly"),
     "cheap", "lowest price"),
    (("long validity", "long term", "annual", "yearly", "less recharge",
      "infrequent"), "validity", "longest validity"),
    (("traveller", "traveler", "travel", "roaming"), "roaming", "roaming benefit"),
    (("ott", "entertainment", "movies", "shows", "netflix"), "ott", "OTT bundle"),
]


def detect_recommend_profile(question: str) -> tuple[str, str] | None:
    q = question.lower()
    for keywords, key, label in RECOMMEND_PROFILES:
        if any(k in q for k in keywords):
            return key, label
    return None


def recommend_for_profile(key: str, df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Pick the top-k live plans for a customer profile. Transparent sort."""
    pool = df[df["Plan_Status"].str.lower() == "live"]

    if key == "cheap":
        return pool.assign(_k=pool["Plan_Rental"].apply(_num)).sort_values("_k") \
                   .drop(columns="_k").head(k)
    if key == "data":
        p = pool.assign(_k=pool["Daily_Data_Limit"].apply(_daily_gb)).dropna(subset="_k")
        return p.sort_values("_k", ascending=False).drop(columns="_k").head(k)
    if key == "validity":
        return pool.assign(_k=pool["Validity_Days"].apply(_num)) \
                   .sort_values("_k", ascending=False).drop(columns="_k").head(k)
    if key == "roaming":
        p = pool[~pool["Roaming_Benefit"].apply(_is_negative)]
        return p.assign(_k=p["Plan_Rental"].apply(_num)).sort_values("_k") \
                .drop(columns="_k").head(k)
    if key == "ott":
        p = pool[~pool["OTT_Platform"].apply(_is_negative)]
        return p.assign(_k=p["Plan_Rental"].apply(_num)).sort_values("_k") \
                .drop(columns="_k").head(k)
    return pool.head(k)


# --------------------------------------------------------------------------
# Guided plan finder
#
# Backs the "Plan Finder" questionnaire in the UI: instead of the rep having to
# phrase a query, they answer a few fixed questions and we resolve them against
# the dataset. Same guarantee as the rest of this module -- every result is a
# real row, so nothing can be invented.
# --------------------------------------------------------------------------

# Must-have key -> (human label, predicate on a plan row).
# Note on roaming: nearly every plan carries free NATIONAL roaming, so treating
# "roaming" as a requirement would filter almost nothing. The one that actually
# narrows the catalogue -- and the one a rep is asked for -- is international.
QUIZ_NEEDS: dict[str, tuple[str, object]] = {
    "5g": ("5G ready",
           lambda r: str(r["5G_Eligible"]).strip().lower() == "yes"),
    "ott": ("OTT bundled",
            lambda r: not _is_negative(r["OTT_Platform"])),
    "voice": ("Unlimited calling",
              lambda r: "unlimited" in str(r["Voice_Benefit"]).lower()),
    "roaming": ("International roaming",
                lambda r: "international" in str(r["Roaming_Benefit"]).lower()
                or "ir " in str(r["Roaming_Benefit"]).lower()),
    "rollover": ("Data rollover",
                 lambda r: str(r["Data_Rollover"]).strip().lower() == "yes"),
}


def find_best_plans(prefs: dict, df: pd.DataFrame, k: int = 3) -> tuple[pd.DataFrame, list[str]]:
    """
    Rank live plans against a filled-in Plan Finder questionnaire.

    Constraints are dropped one at a time when nothing survives, so the rep
    always gets a usable answer PLUS an explicit note about which requirement
    had to give -- better than an empty result with no explanation. They are
    relaxed least-important first, which puts budget last: a customer's price
    ceiling is usually the hardest of their constraints, so it is the one we
    break only as a final resort.

    Ranking is "the cheapest plan that still qualifies", tie-broken by more
    daily data then longer validity. That is a sales rule a human can audit,
    not an opaque score.

    prefs keys (all optional): max_price, min_daily_gb, min_days, max_days,
    product_type, needs (list of QUIZ_NEEDS keys).

    Returns (rows, relaxed) -- `relaxed` names the constraints that were
    dropped, in the order they were dropped.
    """
    pool = df[df["Plan_Status"].str.lower() == "live"]

    product_type = (prefs.get("product_type") or "").strip().lower()
    if product_type:
        typed = pool[pool["Product_Type"].str.lower() == product_type]
        if not typed.empty:
            pool = typed

    # Ordered LEAST important first -- this is the drop order.
    constraints: list[tuple[str, object]] = []

    if prefs.get("max_days"):
        v = float(prefs["max_days"])
        constraints.append((f"validity up to {int(v)} days",
                            lambda r, v=v: (_num(r["Validity_Days"]) or 0) <= v))
    if prefs.get("min_days"):
        v = float(prefs["min_days"])
        constraints.append((f"validity of {int(v)}+ days",
                            lambda r, v=v: (_num(r["Validity_Days"]) or 0) >= v))
    if prefs.get("min_daily_gb"):
        v = float(prefs["min_daily_gb"])
        constraints.append((f"at least {v:g} GB/day",
                            lambda r, v=v: (_daily_gb(r["Daily_Data_Limit"]) or 0) >= v))
    for key in prefs.get("needs") or []:
        if key in QUIZ_NEEDS:
            constraints.append(QUIZ_NEEDS[key])
    if prefs.get("max_price"):
        v = float(prefs["max_price"])
        constraints.append((f"budget under ₹{int(v)}",
                            lambda r, v=v: (_num(r["Plan_Rental"]) or 0) <= v))

    relaxed: list[str] = []
    active = list(constraints)
    while True:
        rows = pool
        for _, predicate in active:
            if rows.empty:
                break
            rows = rows[rows.apply(predicate, axis=1)]
        if not rows.empty or not active:
            break
        relaxed.append(active.pop(0)[0])

    if rows.empty:
        return rows, relaxed

    ranked = rows.assign(
        _price=rows["Plan_Rental"].apply(lambda v: _num(v) if _num(v) is not None else 0.0),
        _gb=rows["Daily_Data_Limit"].apply(lambda v: _daily_gb(v) or 0.0),
        _days=rows["Validity_Days"].apply(lambda v: _num(v) or 0.0),
    ).sort_values(["_price", "_gb", "_days"], ascending=[True, False, False])
    return ranked.drop(columns=["_price", "_gb", "_days"]).head(k), relaxed


def explain_pick(row: pd.Series, prefs: dict) -> list[str]:
    """
    Short bullet reasons this plan was picked, drawn straight from its row.
    Used under each Plan Finder result so the rep can justify the pitch.
    """
    reasons = [
        f"{_rupees(row['Plan_Rental'])} for {_days_text(row['Validity_Days'])}",
        f"Data: {row['Daily_Data_Limit']}",
        f"Voice: {row['Voice_Benefit']}",
    ]
    for key in prefs.get("needs") or []:
        if key in QUIZ_NEEDS:
            label, predicate = QUIZ_NEEDS[key]
            if predicate(row):
                detail = {
                    "ott": row["OTT_Platform"],
                    "roaming": row["Roaming_Benefit"],
                }.get(key)
                reasons.append(f"{label}" + (f" — {detail}" if detail else " ✓"))
    return reasons


def render_comparison(rows: pd.DataFrame) -> str:
    plans = [rows.iloc[i] for i in range(min(len(rows), 4))]
    header = ("| Feature | "
              + " | ".join(_table_cell("Plan_Name", p["Plan_Name"]) for p in plans)
              + " |")
    sep = "| --- |" + " --- |" * len(plans)
    lines = [header, sep]
    for column, label in DETAIL_FIELDS:
        if column == "Plan_SOC_ID":
            continue
        values = [_table_cell(column, p[column]) for p in plans]
        if len(set(values)) == 1 and column not in ("Plan_Rental", "Validity_Days"):
            continue  # skip rows where every plan is identical
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    prices = [_num(p["Plan_Rental"]) for p in plans]
    if len(prices) == 2 and all(p is not None for p in prices):
        diff = abs(prices[0] - prices[1])
        cheaper = plans[0] if prices[0] < prices[1] else plans[1]
        lines.append("")
        lines.append(f"{cheaper['Plan_Name']} is ₹{int(diff)} cheaper.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class RulesEngine:
    """Deterministic answer engine. Mirrors GeminiCopilot's `.reply()`."""

    active_model = "rule-based (no LLM)"

    def __init__(self, df: pd.DataFrame):
        self.df = df

    @property
    def model_chain(self) -> list[str]:
        return ["rule-based (no LLM)"]

    def reply(self, history: list[dict]) -> str:
        question = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"), ""
        )
        if not question.strip():
            return "Ask me something about the VI plans."

        plans = find_plans(question, self.df)

        # Follow-ups that reference the PREVIOUS result set rather than naming a
        # plan: "compare the first two", "what about them", "the second one".
        if plans.empty:
            prev = self._previous_result_set(history)
            positional = self._resolve_positional(question, prev)
            if positional is not None:
                plans = positional
            elif prev is not None and self._is_group_followup(question):
                plans = prev

        # Simple follow-up ("what about its validity") -> reuse the most recent
        # single plan named earlier.
        if plans.empty:
            for msg in reversed(history[:-1]):
                if msg["role"] == "user":
                    previous = find_plans(msg["content"], self.df)
                    if not previous.empty:
                        plans = previous
                        break

        return self._answer(question, plans)

    def _previous_result_set(self, history: list[dict]):
        """The plan rows referenced by the most recent earlier USER turn."""
        for msg in reversed(history[:-1]):
            if msg["role"] == "user":
                prev = find_plans(msg["content"], self.df)
                if not prev.empty:
                    return prev
                rows, applied = apply_filters(msg["content"], self.df)
                if applied and not rows.empty:
                    return rows
        return None

    @staticmethod
    def _is_group_followup(question: str) -> bool:
        q = question.lower()
        return any(w in q for w in ("them", "those", "these", "that list",
                                    "the results", "all of them"))

    ORDINALS = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2,
                "3rd": 2, "fourth": 3, "4th": 3, "last": -1}

    def _resolve_positional(self, question: str, prev):
        """
        Resolve 'the first two', 'the second one', 'first and third' against the
        previous result set. Returns the selected rows, or None if no positional
        reference is present.
        """
        if prev is None or prev.empty:
            return None
        q = question.lower()

        # "the first two" / "top 3" / "first three"
        count_words = {"two": 2, "three": 3, "four": 4}
        m = re.search(r"(?:first|top)\s+(two|three|four|\d+)", q)
        if m:
            token = m.group(1)
            n = count_words.get(token, int(token) if token.isdigit() else 0)
            if n:
                return prev.head(n)

        # Individual ordinals mentioned: "the second one", "first and third"
        picked = [idx for word, idx in self.ORDINALS.items()
                  if re.search(rf"\b{word}\b", q)]
        if picked:
            rows = [prev.iloc[i] for i in picked if -len(prev) <= i < len(prev)]
            if rows:
                return pd.DataFrame(rows)
        return None

    def _answer(self, question: str, plans: pd.DataFrame) -> str:
        q = question.lower()

        # A count question only when it's counting PLANS. "How many days
        # validity does plan X have" is a field lookup, not a count.
        counts_plans = re.search(r"(how many|number of|count of)\s+\w*\s*plans?\b", q) \
            or (any(w in q for w in ("how many", "number of")) and plans.empty)
        if counts_plans and not (plans.shape[0] == 1 and "plan" in q and "plans" not in q):
            rows, applied = apply_filters(question, self.df)
            criteria = f" matching {', '.join(applied)}" if applied else ""
            return f"**{len(rows)}** plans{criteria}."

        # "alternatives to plan X" / "similar to the 449 plan" / "other options"
        wants_alts = any(w in q for w in
                         ("alternative", "similar plan", "similar to", "instead of",
                          "other option", "like this plan", "comparable"))
        if wants_alts and not plans.empty:
            target = plans.iloc[0]
            return render_alternatives(target, find_alternatives(target, self.df))

        # "recommend a plan for a heavy data user" / "best plan for a traveller".
        # Only treat as a recommendation when it names a customer profile AND no
        # concrete plan was referenced (so "recommend plan 449" still shows 449).
        wants_reco = any(w in q for w in
                         ("recommend", "suggest", "best plan for", "which plan should",
                          "good for", "suitable for", "what should i offer"))
        if wants_reco and plans.empty:
            profile = detect_recommend_profile(question)
            if profile:
                key, label = profile
                picks = recommend_for_profile(key, self.df)
                if not picks.empty:
                    head = f"Top picks for **{label}**:\n\n"
                    return head + render_table(picks)
            return (
                "Tell me the customer's priority and I'll pick from the dataset — "
                "e.g. *lowest price*, *most data*, *long validity*, *roaming*, or "
                "*OTT/entertainment*. For example: "
                "*recommend a plan for a heavy data user*."
            )

        is_compare = any(w in q for w in ("compare", " vs ", "versus", "difference between"))
        if is_compare and len(plans) >= 2:
            # A bare price like "299" can hit several plans (current, postpaid,
            # legacy). Prefer the live ones so the rep compares what they can
            # actually sell, and say so when others were set aside.
            live = plans[plans["Plan_Status"].str.lower() == "live"]
            note = ""
            if 2 <= len(live) < len(plans):
                note = (f"\n\n_Comparing live plans only; "
                        f"{len(plans) - len(live)} withdrawn/legacy match(es) hidden._")
                plans = live
            return render_comparison(plans) + note

        wants_details = any(w in q for w in
                            ("detail", "everything", "full info", "all info", "tell me about"))
        if not plans.empty:
            if wants_details or len(q.split()) <= 3:
                if len(plans) == 1:
                    return render_details(plans.iloc[0])
                return render_table(plans)

            field = detect_field(question)
            if field:
                column, label = field
                return render_field(plans, column, label)

            if len(plans) == 1:
                return render_details(plans.iloc[0])
            return render_table(plans)

        # A plan-code-shaped number that matched nothing is a wrong code, not an
        # unparseable question -- say so plainly.
        unknown_codes = [n for n in re.findall(r"\b\d{6,10}\b", question)
                         if n not in set(self.df["Plan_SOC_ID"])]
        if unknown_codes:
            return (f"No plan with code {unknown_codes[0]} is in the dataset. "
                    f"Please re-check the plan code.")

        # No specific plan named -> treat it as a search over the catalogue.
        rows, applied = apply_filters(question, self.df)

        if not applied and not any(
            w in q for w in ("cheapest", "lowest", "most expensive", "highest",
                             "longest", "shortest", "most ", "max", "all plans",
                             "list", "show", "every plan")):
            return (
                "I couldn't turn that into a lookup on the dataset.\n\n"
                "This mode matches on concrete criteria — try naming a plan "
                "(code, name, or price), or filtering by price, data, validity, "
                "5G, OTT, voice, or prepaid/postpaid.\n\n"
                "For example: *plans under ₹500 with unlimited voice and 5G*."
            )

        matched_count = len(rows)   # before any superlative narrows it to one
        rows, sort_note = self._apply_superlative(q, rows)

        if rows.empty:
            criteria = f" for {', '.join(applied)}" if applied else ""
            return (f"No plans match{criteria}. Try relaxing one of the filters.")

        field = detect_field(question)
        single_field_query = field and len(rows) <= 5 and applied

        header = ""
        if applied:
            header = f"**{matched_count}** plans match: {', '.join(applied)}."
            if sort_note:
                header += f" {sort_note}"
            header += "\n\n"
        elif sort_note:
            header = f"{sort_note}\n\n"

        if len(rows) == 1:
            return header + render_details(rows.iloc[0])
        if single_field_query:
            column, label = field
            return header + render_field(rows, column, label)
        return header + render_table(rows)

    def _apply_superlative(self, q: str, rows: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Handle cheapest / most expensive / most data / longest validity."""
        if rows.empty:
            return rows, ""

        if any(w in q for w in ("cheapest", "lowest price", "least expensive", "most affordable")):
            ordered = rows.assign(_k=rows["Plan_Rental"].apply(_num)).sort_values("_k")
            return ordered.drop(columns="_k").head(1), "Sorted by lowest price."

        if any(w in q for w in ("most expensive", "highest price", "costliest", "priciest")):
            ordered = rows.assign(_k=rows["Plan_Rental"].apply(_num)).sort_values("_k")
            return ordered.drop(columns="_k").tail(1), "Sorted by highest price."

        if any(w in q for w in ("most data", "highest data", "maximum data", "max data")):
            ordered = rows.assign(_k=rows["Daily_Data_Limit"].apply(_daily_gb))
            ordered = ordered.dropna(subset="_k").sort_values("_k")
            if not ordered.empty:
                return ordered.drop(columns="_k").tail(1), "Sorted by highest daily data."

        if any(w in q for w in ("longest validity", "most validity", "maximum validity")):
            ordered = rows.assign(_k=rows["Validity_Days"].apply(_num)).sort_values("_k")
            return ordered.drop(columns="_k").tail(1), "Sorted by longest validity."

        return rows, ""
