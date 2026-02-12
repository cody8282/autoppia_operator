from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import json
import hashlib
import os
import re
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit
from html.parser import HTMLParser

from fastapi import Body, FastAPI, HTTPException

from llm_gateway import openai_chat_completions, is_sandbox_gateway_base_url

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None


FIXED_AUTBOOKS_URL = os.getenv(
    "FIXED_AUTBOOKS_URL",
    "http://84.247.180.192:8001/books/book-original-002?seed=36",
)

app = FastAPI(title="Autoppia Web Agent API")

# In-memory loop detection per task_id (best-effort; process-local).
_TASK_STATE: dict[str, dict[str, object]] = {}


@app.get("/health", summary="Health check")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


# -----------------------------
# IWA Selector helpers
# -----------------------------

def _sel_attr(attribute: str, value: str, case_sensitive: bool = False) -> Dict[str, Any]:
    return {
        "type": "attributeValueSelector",
        "attribute": attribute,
        "value": value,
        "case_sensitive": case_sensitive,
    }


def _sel_text(value: str, case_sensitive: bool = False) -> Dict[str, Any]:
    return {
        "type": "tagContainsSelector",
        "value": value,
        "case_sensitive": case_sensitive,
    }


def _sel_custom(value: str, case_sensitive: bool = False) -> Dict[str, Any]:
    return {
        "type": "attributeValueSelector",
        "attribute": "custom",
        "value": value,
        "case_sensitive": case_sensitive,
    }


def _selector_repr(selector: Dict[str, Any]) -> str:
    t = selector.get("type")
    a = selector.get("attribute")
    v = selector.get("value")
    if t == "attributeValueSelector":
        vv = str(v)
        if len(vv) > 80:
            vv = vv[:77] + "..."
        return f"attr[{a}]={vv}"
    if t == "tagContainsSelector":
        return f"text~={v}"
    return str(selector)


# -----------------------------
# Candidate extraction
# -----------------------------

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


class _Candidate:
    def __init__(
        self,
        selector: Dict[str, Any],
        text: str,
        tag: str,
        attrs: Dict[str, str],
        *,
        text_selector: Optional[Dict[str, Any]] = None,
        context: str = "",
    ):
        self.selector = selector
        self.text_selector = text_selector
        self.text = text
        self.tag = tag
        self.attrs = attrs
        self.context = context

    def click_selector(self) -> Dict[str, Any]:
        # Prefer stable attribute selectors when we have them. Text selectors are often ambiguous
        # (e.g. multiple "Sign in" buttons/links).
        if isinstance(self.selector, dict) and self.selector.get("type") == "attributeValueSelector":
            attr = str(self.selector.get("attribute") or "")
            if attr in {"id", "href", "data-testid", "name", "aria-label", "placeholder", "title"}:
                return self.selector

        if self.text_selector:
            ts_val = str(self.text_selector.get("value") or "").strip()
            generic = {
                "view details",
                "view info",
                "details",
                "learn more",
                "more",
                "ok",
                "yes",
                "no",
                "close",
                "cancel",
                "submit",
                "sign in",
                "log in",
                "login",
                "sign up",
                "register",
                "send",
                "save",
            }
            if ts_val and ts_val.lower() not in generic and len(ts_val) >= 6:
                return self.text_selector

        return self.selector


class _CandidateExtractor(HTMLParser):
    """Fallback extractor when BeautifulSoup isn't available."""

    def __init__(self) -> None:
        super().__init__()
        self._current_text: List[str] = []
        self._last_tag: Optional[str] = None
        self._last_attrs: Dict[str, str] = {}
        self.candidates: List[_Candidate] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attr_map = {k: (v or "") for k, v in attrs}
        self._last_tag = tag
        self._last_attrs = attr_map

        if tag in {"button", "a", "input", "textarea", "select"} or attr_map.get("role") in {"button", "link"}:
            label = attr_map.get("aria-label") or attr_map.get("placeholder") or attr_map.get("title") or ""
            selector = _build_selector(tag, attr_map, text=label)
            self.candidates.append(_Candidate(selector, label, tag, attr_map, context=""))

    def handle_data(self, data: str) -> None:
        if self._last_tag in {"button", "a"} and data.strip():
            self._current_text.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag == self._last_tag and self._current_text and self.candidates:
            text = " ".join(self._current_text)[:120]
            c = self.candidates[-1]
            c.text = text or c.text
            if c.tag in {"button", "a"} and c.text:
                c.text_selector = _sel_text(c.text, case_sensitive=False)
        self._current_text = []


def _attrs_to_str_map(attrs: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (attrs or {}).items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            out[k] = " ".join(str(x) for x in v if x is not None).strip()
        else:
            out[k] = str(v)
    return out


def _build_selector(tag: str, attrs: Dict[str, str], *, text: str) -> Dict[str, Any]:
    # Prefer attributes that map directly to IWA Selector.to_playwright_selector().
    if attrs.get("id"):
        return _sel_attr("id", attrs["id"])
    if attrs.get("data-testid"):
        return _sel_attr("data-testid", attrs["data-testid"])
    if tag == "a" and attrs.get("href") and not attrs["href"].lower().startswith("javascript:"):
        return _sel_attr("href", attrs["href"])
    if attrs.get("aria-label"):
        return _sel_attr("aria-label", attrs["aria-label"])
    if attrs.get("name"):
        return _sel_attr("name", attrs["name"])
    if attrs.get("placeholder"):
        return _sel_attr("placeholder", attrs["placeholder"])
    if attrs.get("title"):
        return _sel_attr("title", attrs["title"])
    if attrs.get("class"):
        return _sel_attr("class", attrs["class"])
    if text and tag in {"button", "a"}:
        return _sel_text(text, case_sensitive=False)
    return _sel_custom(tag)


def _extract_label_from_bs4(soup, el, attr_map: Dict[str, str]) -> str:
    tag = str(getattr(el, "name", "") or "")

    if tag in {"a", "button"}:
        t = _norm_ws(el.get_text(" ", strip=True))
        if t:
            return t[:120]

    for key in ("aria-label", "placeholder", "title"):
        if attr_map.get(key):
            return _norm_ws(attr_map[key])[:120]

    if attr_map.get("aria-labelledby"):
        lid = attr_map["aria-labelledby"].split()[0]
        if lid:
            lab = soup.find(id=lid)
            if lab is not None:
                t = _norm_ws(lab.get_text(" ", strip=True))
                if t:
                    return t[:120]

    if attr_map.get("id"):
        lab = soup.find("label", attrs={"for": attr_map["id"]})
        if lab is not None:
            t = _norm_ws(lab.get_text(" ", strip=True))
            if t:
                return t[:120]

    parent_label = el.find_parent("label")
    if parent_label is not None:
        t = _norm_ws(parent_label.get_text(" ", strip=True))
        if t:
            return t[:120]

    return ""


def _extract_candidates_bs4(html: str, *, max_candidates: int) -> List[_Candidate]:
    soup = BeautifulSoup(html, "lxml")

    selectors = [
        "button",
        "a[href]",
        "input",
        "textarea",
        "select",
        "[role='button']",
        "[role='link']",
    ]

    els = []
    for sel in selectors:
        els.extend(soup.select(sel))

    seen: set[tuple[str, str, str]] = set()
    out: List[_Candidate] = []

    for el in els:
        tag = str(getattr(el, "name", "") or "")
        attr_map = _attrs_to_str_map(getattr(el, "attrs", {}) or {})

        # Skip obvious non-interactives.
        if tag == "input" and attr_map.get("type", "").lower() == "hidden":
            continue
        if attr_map.get("disabled") is not None or attr_map.get("aria-disabled", "").lower() == "true":
            continue

        label = _extract_label_from_bs4(soup, el, attr_map)
        dom_label = label
        context = ""
        title = ""
        try:
            parent = el.find_parent(["li", "tr", "article", "section", "div"])
            if parent is not None:
                context = _norm_ws(parent.get_text(" ", strip=True))
                # Try to pull a nearby title so identical buttons become distinguishable.
                h = parent.find(["h1", "h2", "h3", "h4"])
                if h is not None:
                    title = _norm_ws(h.get_text(" ", strip=True))
                if not title:
                    t = parent.find(attrs={"class": re.compile(r"title", re.I)})
                    if t is not None:
                        title = _norm_ws(t.get_text(" ", strip=True))
        except Exception:
            context = ""
            title = ""

        if title and dom_label and dom_label.strip().lower() in {"view details", "view info", "details"}:
            label = f"{dom_label} - {title}"[:120]

        if context and len(context) > 180:
            context = context[:177] + "..."
        primary = _build_selector(tag, attr_map, text=label)

        text_sel = None
        if tag in {"a", "button"} and dom_label:
            # Click by DOM text, even if we augmented label for prompting.
            text_sel = _sel_text(dom_label, case_sensitive=False)

        sig = (
            str(primary.get("type") or ""),
            str(primary.get("attribute") or ""),
            str(primary.get("value") or ""),
        )
        if sig in seen:
            continue
        seen.add(sig)

        out.append(_Candidate(primary, label, tag, attr_map, text_selector=text_sel, context=context))
        if len(out) >= max_candidates:
            break

    return out


def _extract_candidates(html: str, max_candidates: int = 30) -> List[_Candidate]:
    if not html:
        return []

    if BeautifulSoup is not None:
        try:
            return _extract_candidates_bs4(html, max_candidates=max_candidates)
        except Exception:
            pass

    parser = _CandidateExtractor()
    try:
        parser.feed(html)
    except Exception:
        return []
    return parser.candidates[:max_candidates]


def _summarize_html(html: str, limit: int = 1200) -> str:
    if not html:
        return ""

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "lxml")
            text = _norm_ws(soup.get_text(" ", strip=True))
            return text[:limit]
        except Exception:
            pass

    try:
        text = re.sub(r"<[^>]+>", " ", html)
        return _norm_ws(text)[:limit]
    except Exception:
        return ""


def _dom_digest(html: str, limit: int = 1400) -> str:
    # Compact, structured page digest to help the LLM reason without sending full HTML.
    if not html:
        return ""

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "lxml")
            for t in soup(["script", "style", "noscript"]):
                try:
                    t.decompose()
                except Exception:
                    pass

            parts: list[str] = []

            title = ""
            try:
                if soup.title and soup.title.get_text(strip=True):
                    title = _norm_ws(soup.title.get_text(" ", strip=True))
            except Exception:
                title = ""
            if title:
                parts.append(f"TITLE: {title[:160]}")

            heads: list[str] = []
            for h in soup.find_all(["h1", "h2", "h3"], limit=12):
                t = _norm_ws(h.get_text(" ", strip=True))
                if t:
                    heads.append(t[:140])
            if heads:
                parts.append("HEADINGS: " + " | ".join(heads[:10]))

            forms_bits: list[str] = []
            for form in soup.find_all("form", limit=4):
                els = form.find_all(["input", "textarea", "select"], limit=12)
                items: list[str] = []
                for el in els:
                    try:
                        attrs = _attrs_to_str_map(getattr(el, "attrs", {}) or {})
                        itype = (attrs.get("type") or "").lower()
                        label = _extract_label_from_bs4(soup, el, attrs)
                        blob = " ".join([label, attrs.get("name",""), attrs.get("id",""), attrs.get("placeholder",""), attrs.get("aria-label",""), itype]).strip()
                        blob = _norm_ws(blob)
                        if not blob:
                            continue
                        items.append(blob[:140])
                    except Exception:
                        continue
                if items:
                    forms_bits.append("; ".join(items[:8]))
            if forms_bits:
                parts.append("FORMS: " + " || ".join(forms_bits[:3]))

            ctas: list[str] = []
            for el in soup.select("button,a[href],[role='button'],[role='link']"):
                try:
                    if len(ctas) >= 14:
                        break
                    t = _norm_ws(el.get_text(" ", strip=True))
                    if not t:
                        t = _norm_ws(str(el.get("aria-label") or "") or "")
                    if not t:
                        continue
                    t_l = t.lower()
                    if t_l in {"home", "logo"}:
                        continue
                    if t not in ctas:
                        ctas.append(t[:90])
                except Exception:
                    continue
            if ctas:
                parts.append("CTAS: " + " | ".join(ctas[:12]))

            out = "\n".join(parts).strip()
            return out[:limit]
        except Exception:
            pass

    return _summarize_html(html, limit=limit)


# -----------------------------
# Ranking and prompting
# -----------------------------

# -----------------------------
# Structured hints (entity extraction)
# -----------------------------

def _parse_movie_snippet(snippet: str) -> Dict[str, Any]:
    """Best-effort parse of a movie card text snippet.

    Expected-ish patterns seen in demo webs:
      "Action The Lord of the Rings: The Fellowship of the Ring 2001 · 178 min View detail"
      "Comedy 5 Amelie ..."

    We keep this intentionally heuristic.
    """
    out: Dict[str, Any] = {}
    sn = _norm_ws(snippet)

    # year + duration pattern
    m = re.search(r"(?P<title>.+?)\s+(?P<year>\d{4})\s*[·\-|]\s*(?P<dur>\d{2,3})\s*min", sn, flags=re.I)
    if m:
        out["title"] = _norm_ws(m.group('title'))[:120]
        out["year"] = int(m.group('year'))
        out["duration_min"] = int(m.group('dur'))

    # crude genre guess: first token if it's a word and not too long
    first = sn.split(' ', 1)[0] if sn else ''
    if first and first.isalpha() and 3 <= len(first) <= 20:
        out.setdefault('genre', first)

    # rating: single number early (e.g. "Comedy 5 Amelie")
    m2 = re.search(r"\b(?P<rating>[0-9](?:\.[0-9])?)\b", sn)
    if m2:
        try:
            out.setdefault('rating', float(m2.group('rating')))
        except Exception:
            pass

    out['snippet'] = sn[:220]
    return out


def _structured_hints(task: str, candidates: List[_Candidate]) -> Dict[str, Any]:
    """Build compact, structured hints to help the LLM disambiguate UI."""
    task_l = (task or '').lower()

    # Inputs
    inputs: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates):
        if c.tag not in {'input', 'textarea', 'select'}:
            continue
        attrs = {k: (c.attrs.get(k) or '') for k in ('type', 'name', 'id', 'placeholder', 'aria-label')}
        label = (c.text or '').strip()
        blob = ' '.join([label, c.context or '', attrs.get('name',''), attrs.get('id',''), attrs.get('placeholder',''), attrs.get('aria-label','')]).lower()

        kind = 'text'
        if 'password' in blob or attrs.get('type','').lower() == 'password':
            kind = 'password'
        elif 'email' in blob:
            kind = 'email'
        elif any(k in blob for k in ['search', 'buscar', 'query', 'find']):
            kind = 'search'
        elif any(k in blob for k in ['user', 'username', 'login']):
            kind = 'username'

        inputs.append({
            'candidate_id': i,
            'kind': kind,
            'label': label[:80],
            'attrs': {k: v for k, v in attrs.items() if v},
        })

    # Movie detail buttons
    movie_buttons: List[Dict[str, Any]] = []
    for i, c in enumerate(candidates):
        if c.tag not in {'a', 'button'}:
            continue
        label = (c.text or '').strip().lower()
        if 'view detail' in label or 'view details' in label or 'details' == label:
            parsed = _parse_movie_snippet(c.context or '')
            parsed['candidate_id'] = i
            movie_buttons.append(parsed)
        # Also pick common id patterns.
        cid = (c.attrs.get('id') or '').lower()
        if cid and any(k in cid for k in ['view-details', 'film-details', 'details-btn', 'view-info']):
            parsed = _parse_movie_snippet(c.context or '')
            parsed['candidate_id'] = i
            movie_buttons.append(parsed)

    

    # Primary buttons (submit/signin/signup/send/save)
    buttons: list[dict[str, Any]] = []
    for i, c in enumerate(candidates):
        if c.tag not in {'a', 'button'}:
            continue
        label = (c.text or '').strip()
        blob = ' '.join([
            label,
            c.context or '',
            c.attrs.get('id',''),
            c.attrs.get('name',''),
            c.attrs.get('type',''),
            c.attrs.get('aria-label',''),
            c.attrs.get('role',''),
            c.attrs.get('href',''),
        ]).lower()

        kind = None
        if 'type=submit' in blob or 'submit' in blob:
            kind = 'submit'
        if any(k in blob for k in ['sign in', 'log in', 'login', 'signin']):
            kind = 'signin'
        if any(k in blob for k in ['sign up', 'signup', 'register', 'create account']):
            kind = 'signup'
        if any(k in blob for k in ['send', 'message', 'contact']):
            kind = 'send'
        if any(k in blob for k in ['save', 'update', 'apply', 'confirm']):
            kind = kind or 'save'

        if kind:
            buttons.append({'candidate_id': i, 'kind': kind, 'label': label[:90]})

    # Limit size
    buttons = buttons[:12]
# Dedup movie buttons by candidate_id
    seen = set()
    mb2: List[Dict[str, Any]] = []
    for mb in movie_buttons:
        ci = mb.get('candidate_id')
        if not isinstance(ci, int) or ci in seen:
            continue
        seen.add(ci)
        mb2.append(mb)

    # Limit size
    inputs = inputs[:12]
    mb2 = mb2[:12]

    return {
        'task_intent': {
            'login': 'login' in task_l or 'sign in' in task_l,
            'register': 'register' in task_l or 'sign up' in task_l,
            'contact': 'contact' in task_l or 'get in touch' in task_l,
            'edit': 'edit' in task_l or 'update' in task_l,
            'delete': 'delete' in task_l or 'remove' in task_l,
            'search': 'search' in task_l or 'find' in task_l,
            'detail': 'show details' in task_l or 'details' in task_l,
        },
        'inputs': inputs,
        'movie_detail_buttons': mb2,
        'buttons': buttons,
        'login_form_present': any(i.get('kind') in {'username','email'} for i in inputs) and any(i.get('kind') == 'password' for i in inputs),

    }

def _tokenize(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{2,}", (s or "").lower())}


def _score_candidate(task: str, c: _Candidate) -> float:
    task_toks = _tokenize(task)
    label = (c.text or "")
    ctx = (c.context or "")
    label_toks = _tokenize(label + " " + ctx)

    score = 0.0

    # Token overlap.
    score += 2.0 * len(task_toks & label_toks)

    task_l = task.lower()
    label_l = label.lower()

    # Intent keywords.
    kw_map = {
        "login": {"login", "log in", "sign in", "signin"},
        "logout": {"logout", "sign out", "signout"},
        "register": {"register", "sign up", "signup", "create account"},
        "search": {"search", "find"},
        "delete": {"delete", "remove"},
        "save": {"save", "submit", "confirm", "apply", "update"},
        "checkout": {"checkout", "pay", "place order"},
    }

    for intent, kws in kw_map.items():
        if intent in task_l and any(k in label_l for k in kws):
            score += 5.0

    # Prefer inputs when task implies typing.
    if any(k in task_l for k in ["type", "enter", "fill", "write", "password", "email", "username"]):
        if c.tag in {"input", "textarea"}:
            score += 3.0

    # Boost obvious form fields/buttons for common multi-step tasks (login/register/contact).
    blob = (label_l + " " + (ctx or "").lower() + " " + " ".join(f"{k}={v}" for k, v in (c.attrs or {}).items())).lower()
    if any(k in task_l for k in ["login", "sign in", "log in"]):
        if c.tag in {"input", "textarea"} and any(k in blob for k in ["user", "username", "email", "password"]):
            score += 6.0
        if c.tag in {"button", "a"} and any(k in blob for k in ["sign in", "log in", "submit", "continue"]):
            score += 4.0
    if any(k in task_l for k in ["register", "sign up", "signup", "create account"]):
        if c.tag in {"input", "textarea"} and any(k in blob for k in ["user", "username", "email", "password"]):
            score += 5.0
        if c.tag in {"button", "a"} and any(k in blob for k in ["register", "sign up", "create", "submit"]):
            score += 4.0
    if "contact" in task_l or "message" in task_l:
        if c.tag in {"input", "textarea"} and any(k in blob for k in ["name", "email", "message", "subject"]):
            score += 5.0
        if c.tag in {"button", "a"} and any(k in blob for k in ["send", "submit", "contact"]):
            score += 4.0

    # Penalize generic.
    if c.selector.get("attribute") == "custom" and c.selector.get("value") in {"a", "button", "input"}:
        score -= 2.0

    # Small label penalty.
    if len(label.strip()) <= 2 and not any(x in task_l for x in ["ok", "yes", "no"]):
        score -= 2.0

    return score


def _rank_candidates(task: str, candidates: List[_Candidate], max_candidates: int) -> List[_Candidate]:
    ranked = sorted(candidates, key=lambda c: _score_candidate(task, c), reverse=True)
    return ranked[:max_candidates]


def _parse_llm_json(content: str) -> Dict[str, Any]:
    try:
        obj = json.loads(content)
    except Exception as e:
        raise ValueError(f"LLM returned non-JSON: {content[:200]}") from e
    if not isinstance(obj, dict):
        raise ValueError("LLM returned non-object JSON")
    return obj


def _history_hint(history: List[Dict[str, Any]] | None) -> str:
    if not history:
        return ""

    last = history[-6:]
    # Detect simple repetition: same action+cid repeated.
    repeats = 0
    prev = None
    for h in last:
        k = (str(h.get("action") or ""), h.get("candidate_id"))
        if prev is not None and k == prev and k != ("", None):
            repeats += 1
        prev = k

    if repeats >= 2:
        return "You appear to be repeating the same action. Choose a DIFFERENT candidate or try scroll."

    return ""


def _rewrite_task_for_llm(task: str, relevant_data: Dict[str, Any]) -> str:
    """Rewrite task text to prefer canonical placeholders from relevant_data.

    This avoids cases where the task string contains ambiguous identifiers (e.g. a numeric web_agent_id)
    while the actual credentials live in relevant_data.
    """
    try:
        if isinstance(relevant_data, dict) and isinstance(relevant_data.get("user_for_login"), dict):
            return "Log in using <username> and <password>."
        if isinstance(relevant_data, dict) and isinstance(relevant_data.get("user_for_register"), dict):
            return "Register a new account using <username>, <email>, and <password>."
    except Exception:
        pass
    return task


def _llm_decide(
    *,
    task_id: str,
    task: str,
    step_index: int,
    url: str,
    candidates: List[_Candidate],
    page_summary: str,
    dom_digest: str,
    history: List[Dict[str, Any]] | None,
    credentials_hint: str = "",
    extra_hint: str = "",
    state_delta: str = "",
) -> Dict[str, Any]:
    items: List[str] = []
    for i, c in enumerate(candidates):
        label = (c.text or "").strip() or c.attrs.get("placeholder", "") or c.attrs.get("aria-label", "")
        attrs_bits: List[str] = []
        for k in ("type", "name", "placeholder", "aria-label", "href", "role"):
            v = c.attrs.get(k)
            if v:
                attrs_bits.append(f"{k}={v}")
        attrs_str = (" | " + ", ".join(attrs_bits)) if attrs_bits else ""
        ctx = (c.context or "").strip()
        ctx_s = (" | ctx=" + ctx) if ctx else ""
        items.append(
            f"{i}: <{c.tag}> '{label}' sel={_selector_repr(c.selector)} click_sel={_selector_repr(c.click_selector())}{attrs_str}{ctx_s}"
        )
    system_msg = (
        "You are a web automation agent. Given the task, step number, state, history, and state diff, choose ONE next action. "
        "Return JSON only (no markdown). Think step-by-step privately before answering. "
        "Do NOT provide detailed chain-of-thought. "
        "Return a JSON object with keys: action, candidate_id, text, evaluation_previous_goal, memory, next_goal. "
        "If STRUCTURED STATE indicates login_form_present=true, you MUST type username and password before clicking a sign-in/submit button. "
        "If you already typed both, then click the sign-in/submit button (prefer attribute selectors). "
        "action must be one of: click,type,select,scroll_down,scroll_up,done. "
        "Constraints: for click/type/select, candidate_id must be an integer index into the candidate list. "
        "For type/select, text must be non-empty. "
                "Avoid done unless the task is clearly completed."
    )

    history_lines: List[str] = []
    for h in (history or [])[-6:]:
        step = h.get("step", "?")
        action = h.get("action", "")
        cid = h.get("candidate_id")
        text = h.get("text", "")
        ok = h.get("exec_ok", True)
        history_lines.append(f"{step}. {action} cid={cid} text={text} [{'OK' if ok else 'FAILED'}]")

    hint = _history_hint(history)

    structured = _structured_hints(task, candidates)
    user_msg = (
        f"You have a task and must decide the next single browser action.\n"
        f"TASK: {task}\n"
        f"STEP: {int(step_index)}\n"
        f"URL: {url}\n\n"
        f"CURRENT STATE (TEXT SUMMARY):\n{page_summary}\n\n"
        + (f"CREDENTIALS (placeholders):\n{credentials_hint}\n\n" if credentials_hint else "")
        + (f"DOM DIGEST (STRUCTURED):\n{dom_digest}\n\n" if dom_digest else "")
        + f"STRUCTURED STATE (JSON):\n{json.dumps(structured, ensure_ascii=True)}\n\n"
        + (f"HISTORY (last steps):\n{chr(10).join(history_lines)}\n\n" if history_lines else "")
        + (f"STATE HINT: {extra_hint}\n\n" if extra_hint else "")
        + (f"STATE DELTA (prev -> current): {state_delta}\n\n" if state_delta else "")
        + f"CANDIDATES (choose by candidate_id; 0 is best-ranked):\n{chr(10).join(items)}\n\n"
        + "Instructions:\n"
        + "- Output JSON only.\n"
        + "- Return ONE action for this step (no multi-step sequences).\n"
        + "- If you need to do a multi-step procedure (login/register/contact), pick the best next step only.\n"
        + "- Use candidate_id for click/type/select and ensure it is in-range.\n"
        + "- For type/select, include non-empty text.\n"
        + "- Provide evaluation_previous_goal, memory, next_goal (each 1 sentence). Do NOT provide detailed chain-of-thought.\n"
        + "- If credentials are needed, use placeholders: <username>, <password>, <email>.\n"
    )

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "350"))

    usages: List[Dict[str, Any]] = []

    def _call(extra_system: str = "") -> Dict[str, Any]:
        sys_msg = system_msg + (" " + extra_system if extra_system else "")
        resp = openai_chat_completions(
            task_id=task_id,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            model=str(model),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            u = resp.get("usage")
            if isinstance(u, dict):
                usages.append(u)
        except Exception:
            pass
        content = resp["choices"][0]["message"]["content"]
        obj = _parse_llm_json(content)
        try:
            obj["_meta"] = {"llm_calls": len(usages), "llm_usages": usages, "model": str(model)}
        except Exception:
            pass
        return obj

    def _valid(obj: Dict[str, Any]) -> bool:
        a = (obj.get("action") or "").lower()
        if a not in {"click", "type", "select", "scroll_down", "scroll_up", "done"}:
            return False
        if a in {"click", "type", "select"}:
            cid = obj.get("candidate_id")
            if isinstance(cid, str) and cid.isdigit():
                cid = int(cid)
            if not isinstance(cid, int) or not (0 <= cid < len(candidates)):
                return False
            if a in {"type", "select"}:
                t = obj.get("text")
                if not isinstance(t, str) or not t.strip():
                    return False
        return True

    try:
        obj = _call()
    except Exception:
        obj = _call("Return ONLY valid JSON. No markdown. No commentary.")

    if not _valid(obj):
        obj = _call(
            "Your previous JSON was invalid. Fix it. "
            f"candidate_id must be an integer in [0, {len(candidates) - 1}]. "
            "If action is type/select you must include non-empty text. "
            "If stuck, scroll_down."
        )
    return obj


def _update_task_state(task_id: str, url: str, sig: str) -> None:
    if not task_id:
        return
    try:
        st = _TASK_STATE.get(task_id)
        if not isinstance(st, dict):
            st = {}
            _TASK_STATE[task_id] = st
        last_sig = str(st.get("last_sig") or "")
        last_url = str(st.get("last_url") or "")
        if sig and sig == last_sig and str(url) == last_url:
            st["repeat"] = int(st.get("repeat") or 0) + 1
        else:
            st["repeat"] = 0
        st["last_sig"] = str(sig)
        st["last_url"] = str(url)
    except Exception:
        return




def _compute_state_delta(
    *,
    task_id: str,
    url: str,
    page_summary: str,
    dom_digest: str,
    candidates: List[_Candidate],
) -> str:
    """Compute a compact diff signal between current and previous observed state."""
    if not task_id:
        return ""

    try:
        st = _TASK_STATE.get(task_id)
        if not isinstance(st, dict):
            st = {}
            _TASK_STATE[task_id] = st

        prev_url = str(st.get("prev_url") or "")
        prev_summary = str(st.get("prev_summary") or "")
        prev_digest = str(st.get("prev_digest") or "")
        prev_sig_set = set(st.get("prev_sig_set") or [])

        cur_sig_set = set()
        for c in candidates[:30]:
            sig = f"{_selector_repr(c.selector)}|{(c.text or '')[:80]}"
            cur_sig_set.add(sig)

        added = len(cur_sig_set - prev_sig_set) if prev_sig_set else len(cur_sig_set)
        removed = len(prev_sig_set - cur_sig_set) if prev_sig_set else 0
        unchanged = len(cur_sig_set & prev_sig_set) if prev_sig_set else 0

        # Simple summary change heuristic.
        ps = _norm_ws(prev_summary)
        cs = _norm_ws(page_summary)
        pd = _norm_ws(prev_digest)
        cd = _norm_ws(dom_digest)

        same_summary = bool(ps and cs and ps[:240] == cs[:240])
        same_digest = bool(pd and cd and pd[:240] == cd[:240])

        # Persist current state for next step.
        st["prev_url"] = str(url)
        st["prev_summary"] = str(page_summary)
        st["prev_digest"] = str(dom_digest)
        st["prev_sig_set"] = list(cur_sig_set)

        parts = [
            f"url_changed={str(prev_url != str(url)).lower()}" if prev_url else "url_changed=unknown",
            f"summary_changed={str(not same_summary).lower()}" if (ps and cs) else "summary_changed=unknown",
            f"digest_changed={str(not same_digest).lower()}" if (pd and cd) else "digest_changed=unknown",
            f"candidate_added={added}",
            f"candidate_removed={removed}",
            f"candidate_unchanged={unchanged}",
        ]
        return ", ".join(parts)
    except Exception:
        return ""

# -----------------------------
# HTTP entrypoint
# -----------------------------

@app.post("/act", summary="Decide next agent actions")
async def act(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    def _resp(actions: list[dict[str, Any]], metrics: dict[str, Any] | None = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"actions": actions}
        if return_metrics and metrics is not None:
            out["metrics"] = metrics
        return out

    task_id = str(payload.get("task_id") or "")
    task = payload.get("prompt") or payload.get("task_prompt") or ""
    url = payload.get("url") or ""
    step_index = int(payload.get("step_index") or 0)
    return_metrics = os.getenv("AGENT_RETURN_METRICS", "0").lower() in {"1", "true", "yes"}
    html = payload.get("snapshot_html") or ""
    history = payload.get("history") if isinstance(payload.get("history"), list) else None
    relevant_data = payload.get("relevant_data") if isinstance(payload.get("relevant_data"), dict) else {}
    page_summary = _summarize_html(html)
    dom_digest = _dom_digest(html)
    task = str(task or "")
    task_for_llm = task
    credentials_hint = ""


    # Extract + rank candidates before LLM.
    candidates = _extract_candidates(html, max_candidates=80)
    candidates_all = list(candidates)
    candidates = _rank_candidates(task, candidates, max_candidates=30)

    if task_id == "check":
        # Permit repo self-checks without requiring OPENAI credentials.
        if candidates:
            return _resp([{"type": "ClickAction", "selector": candidates[0].click_selector()}], {"decision": "check_click", "candidate_id": 0})
        return _resp([{"type": "WaitAction", "time_seconds": 0.1}], {"decision": "check_wait"})


    st = _TASK_STATE.get(task_id) if task_id else None
    extra_hint = ""
    state_delta = _compute_state_delta(task_id=task_id, url=str(url), page_summary=page_summary, dom_digest=dom_digest, candidates=candidates)
    try:
        if isinstance(st, dict):
            last_url = str(st.get("last_url") or "")
            repeat = int(st.get("repeat") or 0)
            if last_url and last_url == str(url) and repeat >= 2:
                extra_hint = "You appear stuck on the same URL after repeating an action. Choose a different element or scroll."
    except Exception:
        extra_hint = ""

    try:
        base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        if not os.getenv("OPENAI_API_KEY") and not is_sandbox_gateway_base_url(base_url):
            raise RuntimeError("OPENAI_API_KEY not set")
        decision = _llm_decide(
            task_id=task_id,
            task=task_for_llm,
            step_index=step_index,
            url=url,
            candidates=candidates,
            page_summary=page_summary,
            dom_digest=dom_digest,
            history=history,
            credentials_hint=credentials_hint,
            extra_hint=extra_hint,
            state_delta=state_delta,
        )
        if os.getenv("AGENT_LOG_DECISIONS", "0").lower() in {"1", "true", "yes"}:
            try:
                top = []
                for i, c in enumerate(candidates[:5]):
                    top.append({
                        "i": i,
                        "tag": c.tag,
                        "text": (c.text or "")[:80],
                        "context": (c.context or "")[:80],
                        "sel": _selector_repr(c.selector),
                        "click_sel": _selector_repr(c.click_selector()),
                    })
                print(json.dumps({
                    "task_id": task_id,
                    "url": url,
                    "task": task_for_llm[:200],
                    "decision": decision,
                    "top_candidates": top,
                }, ensure_ascii=True))
            except Exception:
                pass
    except Exception as e:
        # Fail closed: don't navigate away on internal errors (it destroys the episode).
        # If you want detailed errors during local dev, set AGENT_DEBUG_ERRORS=1.
        if os.getenv("AGENT_DEBUG_ERRORS", "0").lower() in {"1", "true", "yes"}:
            raise HTTPException(status_code=500, detail=str(e)[:400])
        if task_id != "check" and os.getenv("AGENT_LOG_ERRORS", "0").lower() in {"1", "true", "yes"}:
            try:
                key = os.getenv("OPENAI_API_KEY", "")
                key_fpr = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12] if key else "missing"
                base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
                print(json.dumps({"event": "agent_error", "task_id": task_id, "error": str(e)[:400], "key_fpr": key_fpr, "base_url": base_url}, ensure_ascii=True))
            except Exception:
                pass
        return _resp([{"type": "WaitAction", "time_seconds": 1.0}], {"decision": "error_wait"})

    action = (decision.get("action") or "").lower()
    cid = decision.get("candidate_id")
    text = decision.get("text")

    # Be forgiving if model emits numbers as strings.
    if isinstance(cid, str) and cid.isdigit():
        cid = int(cid)


    if action in {"scroll_down", "scroll_up"}:
        _update_task_state(task_id, str(url), f"{action}")
        return _resp([{"type": "ScrollAction", "down": action == "scroll_down", "up": action == "scroll_up"}], {"decision": decision.get("action"), "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
    if action == "wait":
        # Avoid wait; it is rarely the right move in this benchmark.
        if candidates:
            selector = candidates[0].click_selector()
            _update_task_state(task_id, str(url), f"click_override:{_selector_repr(selector)}")
            return _resp([{"type": "ClickAction", "selector": selector}], {"decision": "click", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
        _update_task_state(task_id, str(url), "scroll_override")
        return _resp([{"type": "ScrollAction", "down": True, "up": False}], {"decision": "scroll_override", "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
    if action == "done":
        _update_task_state(task_id, str(url), "done")
        return _resp([], {"decision": "done", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})

    if action in {"click", "type", "select"} and isinstance(cid, int) and 0 <= cid < len(candidates):
        c = candidates[cid]

        if action == "click":
            selector = c.click_selector()
            _update_task_state(task_id, str(url), f"click:{_selector_repr(selector)}")
            return _resp([{"type": "ClickAction", "selector": selector}], {"decision": "click", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})

        selector = c.selector
        if action == "type":
            if not text:
                raise HTTPException(status_code=400, detail="type action missing text")
            _update_task_state(task_id, str(url), f"type:{_selector_repr(selector)}")
            return _resp([{"type": "TypeAction", "selector": selector, "text": str(text)}], {"decision": "type", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
        if action == "select":
            if not text:
                raise HTTPException(status_code=400, detail="select action missing text")
            _update_task_state(task_id, str(url), f"select:{_selector_repr(selector)}")
            return _resp([{"type": "SelectDropDownOptionAction", "selector": selector, "text": str(text)}], {"decision": "select", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})

    if candidates and step_index < 5:
        selector = candidates[0].click_selector()
        _update_task_state(task_id, str(url), f"fallback_click:{_selector_repr(selector)}")
        return _resp([{"type": "ClickAction", "selector": selector}], {"decision": "click_override", "candidate_id": 0 if candidates else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
    _update_task_state(task_id, str(url), "fallback_wait")
    return _resp([{"type": "WaitAction", "time_seconds": 2.0}], {"decision": "fallback_wait", "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})


@app.post("/step", summary="Alias for /act")
async def step(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await act(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
