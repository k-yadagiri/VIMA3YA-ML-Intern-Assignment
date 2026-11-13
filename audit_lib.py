"""Rule-based reference checker built ONLY from SCHEMA.md. Used for auditing/cleaning train labels
(detect + fix). It is NOT used at inference time."""
import json, re
from datetime import date, timedelta
from decimal import Decimal

KEYS = ["category","priority","amount_paise","txn_date","txn_id","channel","language","needs_human"]
CATS = ["payment_failed","refund","fraud","account_access","kyc","offers","other"]
MONTHS = {m:i+1 for i,m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}

def split_ticket(user):
    head, body = user.split("\n\n", 1)
    received = re.search(r"Received: (\d{4})-(\d\d)-(\d\d)", head)
    rec = date(*map(int, received.groups()))
    source = re.search(r"Source: (\S+)", head).group(1)
    return rec, source, body

def clean_body(body):
    """Drop quoted earlier emails ('>' lines), boilerplate footers, signatures and subject lines."""
    out = []
    for ln in body.split("\n"):
        s = ln.strip()
        if s.startswith(">"): continue
        if re.match(r"(Subject:|Regards,|Your request has been registered|Our team will get back|If you have already received)", s): continue
        out.append(s)
    return "\n".join(out)

def sentences(text):
    text = re.sub(r"\b(Rs|No)\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?])\s+|\n|\s+FYI\s+", text)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]

# ---------- legal / prior-contact ----------
LEGAL = re.compile(r"\bRBI\b|ombudsman|consumer court|lawyer|legal notice|legal action|\blegal\b|vakil|advocate", re.I)
PRIOR = re.compile(r"third time|second complaint|contacted support|again and again|teesri baar|pehle bhi complaint|baar baar|"
                   r"already written \d+ times|written \d+ times|raised this before|already (contacted|complained)", re.I)

# ---------- txn id ----------
def extract_txn_id(text):
    m = re.search(r"\bnp[\s\-]*((?:\d[\s\-]*){10})", text, re.I)
    if not m: return None
    return "NP" + re.sub(r"\D", "", m.group(1))[:10]

# ---------- amount ----------
SKIP_AMT = re.compile(r"balance|daily limit|\blimit\b|pade hain|right now|ticket|bank ref|ref no|app version|\bfees?\b|charges? of", re.I)
NUM = r"\d[\d,]*(?:\.\d+)?"
AMT_PATTERNS = [
    re.compile(r"(?:Rs\.?|INR|₹)\s*(" + NUM + r")\s*(k|lakhs?)?(?![a-z])", re.I),
    re.compile(r"(" + NUM + r")\s*(k|lakhs?)(?![a-z])", re.I),
    re.compile(r"(" + NUM + r")\s*()(?:rupees|rupaye|rupaiye)", re.I),
]
CUR = r"(?:(?:Rs\.?|INR|₹)\s*" + NUM + r"\s*(?:k|lakhs?)?(?![a-z])|" + NUM + r"\s*(?:k|lakhs?|rupees|rupaye|rupaiye)(?![a-z]))"
DISTRACTORS = [
    re.compile(r"(?:FYI\s+)?my daily limit is\s*" + CUR, re.I),
    re.compile(r"my account balance is\s*" + CUR + r"(?:\s*right now)?", re.I),
    re.compile(r"account mein abhi\s*" + CUR + r"\s*pade hain", re.I),
    re.compile(r"(?:earlier|pichla|previous)\s+ticket (?:number)?\s*#?\d+(?: tha)?", re.I),
    re.compile(r"ticket number\s*#?\d+", re.I),
    re.compile(r"bank ref(?: no)?\s*\d+(?: hai)?", re.I),
    re.compile(r"app version(?: is)?\s*[\d.]+(?: hai)?", re.I),
    re.compile(r"\bnp[\s\-]*(?:\d[\s\-]*){10}", re.I),
    re.compile(r"(?:my phone is an?|mera phone)\s+(?:redmi note|iphone|samsung|vivo|pixel|realme|oneplus|xiaomi|oppo|poco)(?:\s+(?:note|nord|[a-z]?\d+[a-z]?))*(?: hai)?", re.I),
]
def strip_distractors(text):
    for p in DISTRACTORS: text = p.sub(" ", text)
    return text
def parse_amount(num, suf):
    v = Decimal(num.replace(",", "").rstrip("."))
    suf = (suf or "").lower()
    if suf == "k": v *= 1000
    elif suf.startswith("lakh"): v *= 100000
    return int(v * 100)

MON = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
DATE_SPANS = [re.compile(r"\b\d{4}-\d\d-\d\d\b"), re.compile(r"(?<!\d)\d{1,2}[-/]\d{1,2}[-/]\d{2,4}(?!\d)"),
              re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+" + MON + r"\.?(?:,?\s+20\d\d)?", re.I),
              re.compile(r"\b" + MON + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+20\d\d)?\b", re.I),
              re.compile(r"\b\d+\s+(?:days ago|din pehle)", re.I)]
def strip_dates(text):
    for p in DATE_SPANS: text = p.sub(" ", text)
    return text

def extract_amount(text):
    text = strip_dates(strip_distractors(text))
    for s in sentences(text):
        if SKIP_AMT.search(s): continue
        for p in AMT_PATTERNS:
            m = p.search(s)
            if m: return parse_amount(m.group(1), m.group(2))
        # bare number amount sentences e.g. "4500 kat gaye." / "Amount 5000 tha."
        m = re.fullmatch(r"(?:Amount:?\s*|Rakam\s*)?(" + NUM + r")\s*(?:tha|hai|kat gaye|in total|is the amount|was the value|ka tha|ka transaction tha|ka transaction hai)?\.?", s, re.I)
        if m and re.search(r"\d", s): return parse_amount(m.group(1), None)
    return None

# ---------- date ----------
def resolve_dm(day, month, rec, year=None):
    if year is None:
        for y in (rec.year, rec.year - 1):
            try:
                d = date(y, month, day)
            except ValueError:
                continue
            if d <= rec: return d
        return None
    return date(year, month, day)

def extract_date(text, rec):
    t = strip_distractors(text)
    low = t.lower()
    m = re.search(r"\b(\d{4})-(\d\d)-(\d\d)\b", t)
    if m: return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = re.search(r"(?<![\d])(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})(?!\d)", t)
    if m:
        dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if yy < 100: yy += 2000
        return date(yy, mm, dd)
    mon = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+" + mon + r"\.?(?:,?\s+(20\d\d))?", t, re.I)
    if m: return resolve_dm(int(m.group(1)), MONTHS[m.group(2).lower()], rec, int(m.group(3)) if m.group(3) else None)
    m = re.search(r"\b" + mon + r"\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?\b", t, re.I)
    if m: return resolve_dm(int(m.group(2)), MONTHS[m.group(1).lower()], rec, int(m.group(3)) if m.group(3) else None)
    if re.search(r"day before yesterday|parso", low): return rec - timedelta(days=2)
    m = re.search(r"(\d+)\s+(?:days ago|din pehle)", low)
    if m: return rec - timedelta(days=int(m.group(1)))
    if re.search(r"\byesterday\b|\bkal\b", low): return rec - timedelta(days=1)
    if re.search(r"\btoday\b|\baaj\b", low): return rec
    return None

# ---------- channel ----------
def extract_channel(text):
    low = text.lower()
    if re.search(r"\bupi\b", low): return "upi"
    if re.search(r"net\s?banking|internet banking", low): return "netbanking"
    if re.search(r"\bwallet\b", low): return "wallet"
    low = re.sub(r"(scratch|srcatch|virtual)\s+cards?", " ", low)
    if re.search(r"\bcard\b|visa|mastercard", low): return "card"
    return None

# ---------- language ----------
HI = set("hai hain nahi nahin kya mera meri mere mujhe maine kar karo kiya gaya gaye hua tha ho ke ki ko mein se par lekin paise paisa "
         "aaj kal parso bhai namaste dhanyavaad jaldi wapas abhi kaise kab kyun ek baar rakam din pehle chuka raha rahi chahiye "
         "aur bhi yeh woh usko apna humara bol kat nikal mila aaya karna karun karne karen karta kiye sakte karwana".split())
def is_hinglish(text):
    toks = re.findall(r"[a-z]+", text.lower())
    return sum(t in HI for t in toks) >= 2

def derive_priority(cat, amt, legal):
    if cat == "fraud" or (isinstance(amt, int) and amt >= 5_000_000) or legal: return "P1"
    if (cat in ("payment_failed","refund") and isinstance(amt, int) and amt >= 500_000) or cat == "account_access": return "P2"
    return "P3"

def reference(user):
    """Everything except `category` (which needs semantic judgement) derived from text."""
    rec, src, body = split_ticket(user)
    text = clean_body(body)
    d = extract_date(text, rec)
    return {"amount_paise": extract_amount(text), "txn_date": d.isoformat() if d else None,
            "txn_id": extract_txn_id(text), "channel": extract_channel(text),
            "language": "hi-en" if is_hinglish(text) else "en",
            "legal": bool(LEGAL.search(text)), "prior": bool(PRIOR.search(text))}


# =====================================================================================
# Audit pipeline: detect defects per training row, decide fix/drop, verify the fixed record
# =====================================================================================
JUNK_BODY = re.compile(r"^(?:test|asdf|123|\s)*$", re.I)

def parse_label(text):
    try:
        o = json.loads(text)
    except Exception:
        return None
    return o if isinstance(o, dict) else None

def validate_record(o, user):
    """Return list of SCHEMA.md violations of a (supposedly final) record, given the ticket text."""
    errs = []
    if not isinstance(o, dict) or set(o) != set(KEYS): return ["keys"]
    if o["category"] not in CATS: errs.append("category_enum")
    if o["priority"] not in ("P1", "P2", "P3"): errs.append("priority_enum")
    a = o["amount_paise"]
    if a is not None and (type(a) is not int or a < 0): errs.append("amount_type")
    if o["txn_date"] is not None:
        try:
            d = date.fromisoformat(o["txn_date"]); assert re.fullmatch(r"\d{4}-\d\d-\d\d", o["txn_date"])
            if d > split_ticket(user)[0]: errs.append("date_after_received")
        except Exception: errs.append("date_format")
    if o["txn_id"] is not None and not (isinstance(o["txn_id"], str) and re.fullmatch(r"NP\d{10}", o["txn_id"])): errs.append("txn_id_format")
    if o["channel"] not in (None, "upi", "card", "netbanking", "wallet"): errs.append("channel_enum")
    if o["language"] not in ("en", "hi-en"): errs.append("language_enum")
    if type(o["needs_human"]) is not bool: errs.append("needs_human_type")
    if o["category"] in ("account_access", "kyc", "other") and any(o[k] is not None for k in ("amount_paise", "txn_date", "txn_id", "channel")):
        errs.append("txn_fields_not_null")
    return errs

def audit_row(row):
    """Returns (defects:list[str], fixed_label:dict|None, decision:'valid'|'fix'|'drop', detail:str)"""
    user = row["messages"][1]["content"]
    raw = row["messages"][2]["content"]
    rec, src, body = split_ticket(user)
    text = clean_body(body)
    o = parse_label(raw)
    if o is None:
        return ["truncated_or_invalid_json"], None, "drop", "label is not parseable JSON (truncated); not reconstructed from rules to avoid inventing a label"
    if JUNK_BODY.match(text.replace("\n", " ").strip()):
        return ["junk_ticket_unsupported_label"], None, "drop", f"ticket body is placeholder text ({text.strip()[:20]!r}); label values cannot be derived from the text"
    ref = reference(user)
    defects, notes = [], []
    f = dict(o)
    # 1. renamed key
    if "prio" in f and "priority" not in f:
        defects.append("wrong_key_prio_instead_of_priority"); notes.append(f"key 'prio'={f['prio']!r}")
        f["priority"] = None; del f["prio"]
    if set(f) - set(KEYS) or set(KEYS) - set(f):
        return defects + ["other_key_mismatch"], None, "drop", f"unexpected keys {sorted(set(f) ^ set(KEYS))}"
    # 2. date format
    if isinstance(f["txn_date"], str) and re.fullmatch(r"\d{2}/\d{2}/\d{4}", f["txn_date"]):
        dd, mm, yy = map(int, f["txn_date"].split("/")); iso = date(yy, mm, dd).isoformat()
        defects.append("date_not_iso_DDMMYYYY"); notes.append(f"{f['txn_date']} -> {iso}"); f["txn_date"] = iso
    # 3. txn id normalisation
    if isinstance(f["txn_id"], str) and not re.fullmatch(r"NP\d{10}", f["txn_id"]):
        norm = "NP" + re.sub(r"\D", "", f["txn_id"])
        if re.fullmatch(r"NP\d{10}", norm):
            defects.append("txn_id_not_normalised"); notes.append(f"{f['txn_id']!r} -> {norm}"); f["txn_id"] = norm
    # 4. amount unit / value, verified against the ticket text
    if f["category"] not in ("account_access", "kyc", "other") and ref["amount_paise"] is not None and f["amount_paise"] != ref["amount_paise"]:
        lab = f["amount_paise"]
        kind = "amount_in_rupees_not_paise" if (isinstance(lab, int) and lab == ref["amount_paise"] // 100) else "amount_wrong_other"
        defects.append(kind); notes.append(f"{lab} -> {ref['amount_paise']}"); f["amount_paise"] = ref["amount_paise"]
    # 5. priority rule
    exp = derive_priority(f["category"], f["amount_paise"], ref["legal"])
    if f["priority"] != exp:
        if "wrong_key_prio_instead_of_priority" not in defects: defects.append("priority_violates_rules")
        notes.append(f"priority recomputed by SCHEMA rule -> {exp}" if f["priority"] is None else f"priority {f['priority']!r} violates SCHEMA rule -> {exp}")
        f["priority"] = exp
    # 6. final verification of the repaired record: every field must agree with the independent text-derived reference
    problems = validate_record(f, user)
    if f["category"] not in ("account_access", "kyc", "other"):
        for k in ("amount_paise", "txn_date", "txn_id", "channel"):
            if f[k] != ref[k]: problems.append(f"{k}_disagrees_with_text")
    if f["language"] != ref["language"]: problems.append("language_disagrees_with_text")
    nh = f["priority"] == "P1" or ref["prior"]
    if f["needs_human"] != nh: problems.append("needs_human_disagrees_with_rule")
    if problems:
        return defects + ["unresolved"], None, "drop", "could not be repaired confidently: " + ",".join(problems)
    if not defects:
        return [], f, "valid", ""
    return defects, f, "fix", "; ".join(notes)


def audit_dataset(rows, dev_rows=None):
    """Full pipeline over the training rows -> {id: (defects, fixed_label, decision, detail)}.
    dev_rows is used ONLY for a text-level decontamination check (train rows whose ticket text is identical to a dev ticket
    are dropped; no dev label is read or used)."""
    results = {r["id"]: audit_row(r) for r in rows}
    dev_text = {d["messages"][1]["content"]: d["id"] for d in (dev_rows or [])}
    for r in rows:
        u = r["messages"][1]["content"]
        if u in dev_text:
            defects = results[r["id"]][0]
            results[r["id"]] = (defects + ["dev_leakage_verbatim_copy"], None, "drop",
                                f"ticket text identical to dev ticket {dev_text[u]}; training on it would violate 'never train on dev'")
    # exact duplicates (same Received/Source/message text): keep the first surviving row, drop later copies
    seen = {}
    for r in rows:
        defects, fixed, dec, detail = results[r["id"]]
        if dec == "drop": continue
        key = r["messages"][1]["content"]
        if key in seen:
            results[r["id"]] = (defects + ["exact_duplicate"], None, "drop", f"exact duplicate of {seen[key]} (identical ticket text and label)")
        else:
            seen[key] = r["id"]
    return results
