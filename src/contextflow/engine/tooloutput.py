"""Tool-output compressor: JSON / API responses / logs -> small structured summary. Rule-based."""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from typing import Any

LOG_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?Z?|\d{2}:\d{2}:\d{2}(?:[.,]\d+)?")
LEVEL_RE = re.compile(r"\b(TRACE|DEBUG|INFO|WARN(?:ING)?|ERROR|ERR|FATAL|CRITICAL)\b")
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
HEX_RE = re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE)


def _short(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _type(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "object"
    return type(v).__name__


def _schema_of_list(items: list[Any]) -> dict[str, Counter]:
    schema: dict[str, Counter] = {}
    for it in items:
        if isinstance(it, dict):
            for k, v in it.items():
                schema.setdefault(k, Counter())[_type(v)] += 1
    return schema


def _field_stats(items: list[dict], key: str) -> str:
    vals = [it.get(key) for it in items if isinstance(it, dict) and key in it]
    nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if nums and len(nums) >= max(2, len(vals) // 2):
        mean = statistics.fmean(nums)
        return f"min {min(nums):g}, max {max(nums):g}, mean {mean:.4g}"
    strs = [v for v in vals if isinstance(v, str)]
    if strs:
        distinct = Counter(strs)
        if len(distinct) <= 6:
            return "values: " + ", ".join(f"{k} x{c}" for k, c in distinct.most_common())
        return f"{len(distinct)} distinct, e.g. {_short(strs[0], 40)!r}"
    bools = [v for v in vals if isinstance(v, bool)]
    if bools:
        return f"true x{sum(bools)}, false x{len(bools) - sum(bools)}"
    return ""


def summarize_json_value(v: Any, depth: int = 0, max_depth: int = 4, max_items: int = 3, max_str: int = 100, indent: str = "") -> list[str]:
    lines: list[str] = []
    pad = indent
    if isinstance(v, dict):
        compact = json.dumps(v, ensure_ascii=False, separators=(", ", ": "))
        if depth >= max_depth or (depth > 0 and len(compact) <= 110):
            return [f"{pad}{_short(compact, 160)}"] if len(compact) <= 160 or depth >= max_depth else [f"{pad}{{{len(v)} keys: {', '.join(list(v)[:8])}}}"]
        for k, val in v.items():
            if isinstance(val, (dict, list)) and val:
                head = f"{pad}{k}:"
                if isinstance(val, list):
                    head += f"  ({len(val)} items)"
                lines.append(head)
                lines += summarize_json_value(val, depth + 1, max_depth, max_items, max_str, pad + "  ")
            else:
                lines.append(f"{pad}{k}: {_scalar(val, max_str)}")
    elif isinstance(v, list):
        if not v:
            return [f"{pad}[]"]
        if all(isinstance(x, dict) for x in v):
            schema = _schema_of_list(v)
            fields = ", ".join(f"{k}:{'/'.join(t for t, _ in c.most_common(2))}" for k, c in list(schema.items())[:14])
            lines.append(f"{pad}schema: {{{fields}}}")
            for k in list(schema)[:14]:
                st = _field_stats(v, k)
                if st:
                    lines.append(f"{pad}  {k} -> {st}")
            for i, item in enumerate(v[:max_items]):
                if i == 0 and depth < 2:
                    lines.append(f"{pad}[0] (full example)")
                    lines += summarize_json_value(item, depth + 1, min(max_depth, depth + 3), 2, max_str, pad + "  ")
                else:
                    lines.append(f"{pad}[{i}] {_short(json.dumps(item, ensure_ascii=False, separators=(',', ':')), 140)}")
            if len(v) > max_items:
                lines.append(f"{pad}... {len(v) - max_items} more items omitted")
        else:
            scal = [_scalar(x, 40) for x in v[:8]]
            nums = [x for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
            tail = f" (min {min(nums):g}, max {max(nums):g}, mean {statistics.fmean(nums):.4g})" if len(nums) == len(v) and nums else ""
            lines.append(f"{pad}[{', '.join(scal)}{', ...' if len(v) > 8 else ''}] {len(v)} items{tail}")
    else:
        lines.append(f"{pad}{_scalar(v, max_str)}")
    return lines


def _scalar(v: Any, max_str: int) -> str:
    if isinstance(v, str):
        if len(v) > max_str:
            return json.dumps(_short(v, max_str), ensure_ascii=False) + f" ({len(v)} chars)"
        return json.dumps(v, ensure_ascii=False)
    return json.dumps(v)


def summarize_json(text: str, mode: str = "balanced") -> str:
    obj = json.loads(text)
    max_items = {"light": 5, "balanced": 3, "aggressive": 1, "prompt_ready": 3, "custom": 3}.get(mode, 3)
    max_str = {"light": 200, "balanced": 100, "aggressive": 60}.get(mode, 100)
    max_depth = {"light": 6, "balanced": 4, "aggressive": 3}.get(mode, 4)
    kind = "array" if isinstance(obj, list) else "object"
    head = f"JSON {kind}" + (f" with {len(obj)} items" if isinstance(obj, list) else f" with {len(obj)} top-level keys")
    body = summarize_json_value(obj, 0, max_depth, max_items, max_str)
    return "\n".join([head] + body)


def _template(line: str) -> str:
    t = LOG_TS_RE.sub("<ts>", line)
    t = HEX_RE.sub("<hex>", t)
    t = NUM_RE.sub("<n>", t)
    return t.strip()


def summarize_logs(text: str, mode: str = "balanced") -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    ts = LOG_TS_RE.findall(text)
    levels = Counter(m.upper().replace("WARNING", "WARN").replace("ERR", "ERROR") if m.upper() != "ERROR" else "ERROR" for m in LEVEL_RE.findall(text))
    templates = Counter(_template(ln) for ln in lines)
    n_err = {"light": 12, "balanced": 8, "aggressive": 4}.get(mode, 8)
    n_top = {"light": 10, "balanced": 6, "aggressive": 3}.get(mode, 6)
    out = [f"Log: {len(lines)} lines" + (f", {ts[0]} -> {ts[-1]}" if len(ts) >= 2 else "")]
    if levels:
        out.append("Levels: " + ", ".join(f"{k} {v}" for k, v in levels.most_common()))
    errs = [ln for ln in lines if re.search(r"\b(ERROR|FATAL|CRITICAL|Exception|Traceback|panic|failed)\b", ln, re.IGNORECASE)]
    if errs:
        out.append(f"Errors/failures ({len(errs)}), first {min(n_err, len(errs))}:")
        seen: set[str] = set()
        shown = 0
        for ln in errs:
            key = _template(ln)
            if key in seen:
                continue
            seen.add(key)
            out.append("  " + _short(ln.strip(), 220))
            shown += 1
            if shown >= n_err:
                break
    warns = [ln for ln in lines if re.search(r"\bWARN(?:ING)?\b", ln)]
    if warns and mode != "aggressive":
        out.append(f"Warnings ({len(warns)}), e.g.: " + _short(warns[0].strip(), 160))
    out.append("Most repeated patterns:")
    for tpl, c in templates.most_common(n_top):
        out.append(f"  x{c}  {_short(tpl, 160)}")
    return "\n".join(out)
