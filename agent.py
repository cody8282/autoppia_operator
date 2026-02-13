from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import json
import hashlib
import os
import re
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit
from html.parser import HTMLParser

from fastapi import Body, FastAPI, HTTPException

# Default this branch to the Chutes provider (OpenAI-compatible).
os.environ.setdefault("LLM_PROVIDER", "chutes")

from llm_gateway import openai_chat_completions, is_sandbox_gateway_base_url

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None


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




def _sel_xpath(value: str) -> Dict[str, Any]:
    return {
        "type": "xpathSelector",
        "attribute": None,
        "value": value,
        "case_sensitive": False,
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
        group: str = "",
        container_chain: list[str] | None = None,
    ):
        self.selector = selector
        self.text_selector = text_selector
        self.text = text
        self.tag = tag
        self.attrs = attrs
        self.context = context
        self.group = group
        self.container_chain = container_chain or []

    def click_selector(self) -> Dict[str, Any]:
        """Selector for click-like actions.

        Prefer stable attribute selectors. Avoid class-based selectors because IWA converts them to `.class` CSS
        and tailwind-style tokens often include `/` or `:` which breaks CSS parsing in Playwright.

        Note: keep this logic generic (no site-specific button text shortcuts).
        """
        if isinstance(self.selector, dict) and self.selector.get("type") == "attributeValueSelector":
            attr = str(self.selector.get("attribute") or "")
            if attr in {"id", "href", "data-testid", "name", "aria-label", "placeholder", "title"}:
                return self.selector

        # If the primary selector isn't a safe attribute selector, try to derive one from attrs.
        for a in ("id", "data-testid", "href", "aria-label", "name", "placeholder", "title"):
            v = (self.attrs or {}).get(a)
            if v:
                return _sel_attr(a, v)

        # Fall back to the element text selector (can be ambiguous, but generic).
        # Generic refinement: if we only have element text, prefer a Playwright :has-text() selector
        # scoped to the tag (button/a). This reduces ambiguity without hardcoding any website logic.
        try:
            t = (self.text or '').strip()
            if t and self.tag in {'button', 'a'}:
                return _sel_custom(f"{self.tag}:has-text({json.dumps(t)})")
        except Exception:
            pass

        if self.text_selector:
            return self.text_selector

        return self.selector

    def type_selector(self) -> Dict[str, Any]:
        """Selector for type/select actions.

        Avoid class selectors for the same reason as click_selector().
        """
        if isinstance(self.selector, dict) and self.selector.get("type") == "attributeValueSelector":
            attr = str(self.selector.get("attribute") or "")
            if attr and attr != "class":
                return self.selector

        for a in ("id", "data-testid", "name", "aria-label", "placeholder", "title"):
            v = (self.attrs or {}).get(a)
            if v:
                return _sel_attr(a, v)

        return _sel_custom(self.tag)


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
            group = 'FORM' if tag in {'input','textarea','select'} else ('LINKS' if tag=='a' else 'BUTTONS')
            self.candidates.append(_Candidate(selector, label, tag, attr_map, context="", group=group, container_chain=[group]))

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




def _container_chain_from_el(soup, el) -> list[str]:
    """Return a short container path for an element to render a simplified DOM tree."""
    chain: list[str] = []
    try:
        # Limit depth to keep prompts small.
        ancestors = list(el.parents) if hasattr(el, 'parents') else []
        # BeautifulSoup yields [element, ..., document]; reverse to go top-down.
        for a in reversed(ancestors):
            try:
                tag = str(getattr(a, 'name', '') or '')
                if not tag or tag in {'[document]', 'html', 'body'}:
                    continue
                if tag not in {'header', 'nav', 'main', 'form', 'section', 'article', 'aside', 'footer', 'ul', 'ol', 'table', 'div'}:
                    continue

                aid = ''
                try:
                    aid = str(a.get('id') or a.get('name') or '').strip()
                except Exception:
                    aid = ''

                role = ''
                try:
                    role = str(a.get('role') or '').strip()
                except Exception:
                    role = ''

                # Try to pull a nearby heading (h1-h3) for more semantic labeling.
                heading = ''
                try:
                    h = a.find(['h1', 'h2', 'h3'])
                    if h is not None:
                        heading = _norm_ws(h.get_text(' ', strip=True))
                except Exception:
                    heading = ''

                label_bits = [tag]
                if aid:
                    label_bits.append(f"#{aid}")
                if role and role not in {'presentation'}:
                    label_bits.append(f"role={role}")
                if heading:
                    label_bits.append(heading[:50])

                label = ' '.join([b for b in label_bits if b])
                label = _norm_ws(label)
                if label and (not chain or chain[-1] != label):
                    chain.append(label)
                if len(chain) >= 4:
                    break
            except Exception:
                continue
    except Exception:
        return chain

    # Keep last 3 containers for focus.
    return chain[-3:]



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

        group = 'PAGE'
        try:
            # Group by semantic containers for a more browser-use-like state view.
            if el.find_parent('nav') is not None:
                group = 'NAV'
            elif el.find_parent('header') is not None:
                group = 'HEADER'
            elif el.find_parent('footer') is not None:
                group = 'FOOTER'
            elif el.find_parent('form') is not None:
                form = el.find_parent('form')
                fid = ''
                try:
                    fid = str(form.get('id') or form.get('name') or '').strip()
                except Exception:
                    fid = ''
                group = f"FORM:{fid}" if fid else 'FORM'
        except Exception:
            group = group

        # Skip obvious non-interactives.
        if tag == "input" and attr_map.get("type", "").lower() == "hidden":
            continue
        if attr_map.get("disabled") is not None or attr_map.get("aria-disabled", "").lower() == "true":
            continue

        label = _extract_label_from_bs4(soup, el, attr_map)

        dom_label = label
        context = ""
        context_raw = ""
        title = ""
        try:
            parent = el.find_parent(["li", "tr", "article", "section", "div"])
            if parent is not None:
                # Preserve line breaks for card-like metadata extraction.
                context_raw = parent.get_text("\n", strip=True)
                context = _norm_ws(context_raw)
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
            context_raw = ""
            title = ""


        if context and len(context) > 180:
            context = context[:177] + "..."

        # Build a selector. Use dom_label for text-based fallbacks to avoid including long meta.
        primary = _build_selector(tag, attr_map, text=(dom_label or label))

        # Improve selectorability + promptability for <select> elements that lack stable attributes.
        if tag == "select":
            # Capture options for prompting and build a selector that uniquely identifies the select.
            opts: list[tuple[str, str]] = []
            try:
                tmp: list[tuple[str, str]] = []
                for o in el.find_all("option")[:12]:
                    t = ""
                    v = ""
                    try:
                        t = o.get_text(" ", strip=True)
                        v = str(o.get("value") or "").strip()
                    except Exception:
                        t = ""
                        v = ""
                    if t:
                        tmp.append((t, v))
                opts = tmp
            except Exception:
                opts = []

            if isinstance(primary, dict) and primary.get("type") == "attributeValueSelector" and str(primary.get("attribute") or "") == "custom" and str(primary.get("value") or "") == "select":
                first_opt = ""
                try:
                    if opts:
                        first_opt = str(opts[0][0] or "").strip()
                except Exception:
                    first_opt = ""
                if first_opt:
                    safe = first_opt.replace("\"", "'")
                    # This is typically unique and avoids strict-mode ambiguity.
                    primary = _sel_custom(f'select:has(option:has-text("{safe}"))')

            if opts:
                show: list[str] = []
                for t, v in opts[:8]:
                    if v and v != t:
                        show.append(f"{t} (value={v})")
                    else:
                        show.append(t)
                opt_preview = ", ".join(show)
                label = (dom_label or "select") + f" options=[{opt_preview}]"
                label = label[:200]

        container_chain = []
        try:
            container_chain = _container_chain_from_el(soup, el)
        except Exception:
            container_chain = []

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

        out.append(_Candidate(primary, label, tag, attr_map, text_selector=text_sel, context=context, group=group, container_chain=container_chain))
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
            'required': bool((c.attrs or {}).get('required') is not None),
            'value_len': len(str((c.attrs or {}).get('value') or '')),
            'attrs': {k: v for k, v in attrs.items() if v},
        })
    return {
        'inputs': inputs[:20],
        'clickables': [
            {
                'candidate_id': i,
                'tag': c.tag,
                'label': (c.text or '')[:90],
                'href': (c.attrs or {}).get('href','') or (c.attrs or {}).get('data-href',''),
                'context': (c.context or '')[:220],
                'attrs': {k: str((c.attrs or {}).get(k) or '') for k in ('id','name','type','placeholder','aria-label','role') if (c.attrs or {}).get(k)},
            }
            for i, c in sorted(
                [(i, c) for i, c in enumerate(candidates) if c.tag in {'a','button'}],
                key=lambda t: len((t[1].context or '').strip()),
                reverse=True,
            )
        ][:25],
    }

def _tokenize(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]{2,}", (s or "").lower())}


def _score_candidate(task: str, c: _Candidate) -> float:
    """Structural scoring only.

    Avoids task-specific string heuristics; prefers stable selectors and form-relevant elements.
    """
    score = 0.0

    if c.tag in {'input', 'textarea', 'select'}:
        score += 6.0
    elif c.tag == 'button':
        score += 4.0
    elif c.tag == 'a':
        score += 2.0

    attrs = c.attrs or {}
    if attrs.get('id'):
        score += 4.0
    if attrs.get('name'):
        score += 2.0
    if attrs.get('aria-label'):
        score += 2.0
    if attrs.get('placeholder'):
        score += 1.0
    if attrs.get('href'):
        score += 1.0
    if attrs.get('role') in {'button','link'}:
        score += 0.5

    if attrs.get('required') is not None and c.tag in {'input','textarea','select'}:
        score += 2.0

    if c.selector.get('attribute') == 'custom' and c.selector.get('value') in {'a','button','input','select','textarea'}:
        score -= 2.0

    if (c.text or '').strip():
        score += 1.0
    if (c.context or '').strip():
        score += 0.5

    return score

def _rank_candidates(task: str, candidates: List[_Candidate], max_candidates: int) -> List[_Candidate]:
    scored = [(i, _score_candidate(task, c), c) for i, c in enumerate(candidates)]
    scored.sort(key=lambda t: (t[1], -t[0]), reverse=True)
    return [c for _, _, c in scored[:max_candidates]]


def _select_candidates_for_llm(task: str, candidates_all: List[_Candidate], current_url: str, max_total: int = 60) -> List[_Candidate]:
    """Pick a diverse, usable candidate set for the LLM.

    Structural selection only (no task keyword heuristics):
    - keep all form controls (input/textarea/select)
    - keep primary buttons
    - keep a slice of anchors/buttons that have non-trivial surrounding context (cards)
    """
    if not candidates_all:
        return []

    controls = []
    primaries = []
    contextual = []
    others = []
    for c in candidates_all:
        # Skip self-links (common in nav) to avoid loops when already on the target page.
        try:
            from urllib.parse import urlparse
            if c.tag == "a":
                href = str((c.attrs or {}).get("href") or "")
                if href:
                    ph = urlparse(href)
                    pc = urlparse(current_url or "")
                    if ph.path and pc.path and ph.path == pc.path:
                        # Same path; let other non-nav elements be considered.
                        continue
        except Exception:
            pass
        if c.tag in {"input", "textarea", "select"}:
            controls.append(c)
            continue
        if c.tag == "button":
            primaries.append(c)
            continue
        if c.tag in {"a", "button"} and (c.context or "").strip():
            if len((c.context or "").strip()) >= 40:
                contextual.append(c)
            else:
                others.append(c)
            continue
        others.append(c)

    picked = []
    seen = set()
    def add_many(arr, limit):
        nonlocal picked
        for c in arr:
            sig = f"{_selector_repr(c.selector)}|{(c.text or "")[:80]}"
            if sig in seen:
                continue
            seen.add(sig)
            picked.append(c)
            if len(picked) >= max_total or len(picked) >= limit:
                return

    # Order: controls first, then contextual card links, then buttons, then the rest.
    add_many(controls, max_total)
    if len(picked) < max_total:
        add_many(contextual, max_total)
    if len(picked) < max_total:
        add_many(primaries, max_total)
    if len(picked) < max_total:
        add_many(others, max_total)

    return picked[:max_total]



def _parse_llm_json(content: str) -> Dict[str, Any]:
    if not isinstance(content, str):
        raise ValueError(f"LLM returned non-text content type={type(content)}")

    raw = content.strip()
    # Common case: pure JSON.
    try:
        obj = json.loads(raw)
    except Exception:
        # Best-effort recovery: strip code-fences and/or extract the first JSON object.
        s = raw
        if s.startswith("```"):
            # Remove leading/trailing fenced blocks like ```json ... ```
            s2 = s
            if s2.startswith("```json"):
                s2 = s2[len("```json") :]
            elif s2.startswith("```"):
                s2 = s2[len("```") :]
            if s2.endswith("```"):
                s2 = s2[: -len("```")]
            s = s2.strip()
        start = s.find("{")
        end = s.rfind("}")
        if 0 <= start < end:
            try:
                obj = json.loads(s[start : end + 1])
            except Exception as e:
                raise ValueError(f"LLM returned non-JSON: {raw[:200]}") from e
        else:
            raise ValueError(f"LLM returned non-JSON: {raw[:200]}")
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




def _format_browser_state(*, candidates: List[_Candidate], prev_sig_set: set[str] | None) -> str:
    """Browser-use-like state view: numbered interactives, with a simplified DOM tree."""

    # Build a tree based on container_chain (preferred) or group fallback.
    class _TNode:
        __slots__ = ("name", "children", "items")

        def __init__(self, name: str) -> None:
            self.name = name
            self.children: dict[str, _TNode] = {}
            self.items: list[tuple[int, _Candidate]] = []

    root = _TNode('ROOT')

    def _chain_for(c: _Candidate) -> list[str]:
        ch = []
        try:
            ch = list(getattr(c, 'container_chain', []) or [])
        except Exception:
            ch = []
        if not ch:
            g = (getattr(c, 'group', '') or 'PAGE').strip() or 'PAGE'
            ch = [g]
        # keep it small
        return [str(x)[:80] for x in ch if str(x).strip()][:3]

    # Insert candidates
    for i, c in enumerate(candidates):
        node = root
        for part in _chain_for(c):
            if part not in node.children:
                node.children[part] = _TNode(part)
            node = node.children[part]
        node.items.append((i, c))

    def _render(node: _TNode, indent: str = '') -> list[str]:
        lines: list[str] = []
        # render items first within this container
        for i, c in node.items:
            label = (c.text or '').strip() or (c.attrs or {}).get('placeholder', '') or (c.attrs or {}).get('aria-label', '')
            label = str(label).strip()

            sig = f"{_selector_repr(c.selector)}|{(c.text or '')[:80]}"
            is_new = bool(prev_sig_set) and (sig not in (prev_sig_set or set()))
            star = '* ' if is_new else ''

            attrs_bits: list[str] = []
            for k in ('id','name','type','placeholder','aria-label','href','role'):
                v = (c.attrs or {}).get(k)
                if v:
                    vv = str(v)
                    if len(vv) > 60:
                        vv = vv[:57] + '...'
                    attrs_bits.append(f"{k}={vv}")
            attrs_str = (' | ' + ', '.join(attrs_bits)) if attrs_bits else ''

            ctx = ''
            try:
                if c.tag in {'a','button'} and (c.context or '').strip():
                    # Help disambiguate repeated CTAs like 'View Details' without site-specific parsing.
                    ctx = ' :: ' + _norm_ws(c.context)[:120]
            except Exception:
                ctx = ''

            lines.append(f"{indent}{star}[{i}]<{c.tag}>{label}</{c.tag}>{attrs_str}{ctx}")

        # then render child containers
        for name, child in node.children.items():
            lines.append(f"{indent}{name}:")
            lines.extend(_render(child, indent + "	"))

        return lines

    rendered = _render(root, '')
    return "\n".join(rendered)




def _format_credentials_hint(relevant_data: Dict[str, Any]) -> str:
    """Format credentials from relevant_data for the LLM.

    This is generic and keeps secrets within the evaluator-provided payload.
    """
    if not isinstance(relevant_data, dict):
        return ""

    # Common shapes used by evaluators.
    for key in ("user_for_login", "user_for_register", "credentials", "user"):
        u = relevant_data.get(key)
        if isinstance(u, dict):
            username = str(u.get("username") or u.get("user") or "").strip()
            password = str(u.get("password") or "").strip()
            email = str(u.get("email") or "").strip()
            bits = []
            if username:
                bits.append(f"username={username}")
            if email:
                bits.append(f"email={email}")
            if password:
                bits.append(f"password={password}")
            if bits:
                return "\n".join(bits)

    return ""

def _rewrite_task_for_llm(task: str, relevant_data: Dict[str, Any]) -> str:
    """Rewrite task text to remove placeholder-like confusion.

    The agent should use actual credential values from CREDENTIALS when provided.
    """
    try:
        if isinstance(relevant_data, dict) and isinstance(relevant_data.get("user_for_login"), dict):
            return "Log in using the provided credentials."
        if isinstance(relevant_data, dict) and isinstance(relevant_data.get("user_for_register"), dict):
            return "Register a new account using the provided credentials."
    except Exception:
        pass
    return task

def _resolve_url(url: str, base_url: str) -> str:
    """Resolve possibly-relative URL against a base URL."""
    try:
        from urllib.parse import urljoin
        u = str(url or "").strip()
        b = str(base_url or "").strip()
        if not u:
            return ""
        # urljoin handles absolute u (returns u unchanged).
        return urljoin(b, u) if b else u
    except Exception:
        return str(url or "").strip()


def _path_query(url: str, base_url: str = "") -> tuple[str, str]:
    try:
        from urllib.parse import urlparse
        resolved = _resolve_url(url, base_url)
        pu = urlparse(resolved or "")
        return (pu.path or ""), (pu.query or "")
    except Exception:
        s = (url or "").strip()
        return s, ""


def _same_path_query(a: str, b: str, *, base_a: str = "", base_b: str = "") -> bool:
    """Compare (path,query) for URLs, resolving relatives against provided bases."""
    try:
        return _path_query(a, base_a) == _path_query(b, base_b)
    except Exception:
        return (a or "").strip() == (b or "").strip()

def _preserve_seed_url(target_url: str, current_url: str) -> str:
    """If current_url has a seed param, ensure target_url keeps it.

    Demo webs are seeded; the validator expects the seed to stay consistent across navigations.
    """
    try:
        from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
        cur = urlparse(current_url or "")
        tgt = urlparse(target_url or "")
        cur_seed = (parse_qs(cur.query).get("seed") or [None])[0]
        if not cur_seed:
            return target_url
        q = parse_qs(tgt.query)
        if (q.get("seed") or [None])[0] == str(cur_seed):
            return target_url
        q["seed"] = [str(cur_seed)]
        new_q = urlencode(q, doseq=True)
        fixed = tgt._replace(query=new_q)
        if not fixed.scheme and not fixed.netloc:
            return urlunparse(("", "", fixed.path, fixed.params, fixed.query, fixed.fragment))
        return urlunparse(fixed)
    except Exception:
        return target_url



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
    prev_sig_set: set[str] | None = None,
) -> Dict[str, Any]:
    browser_state = _format_browser_state(candidates=candidates, prev_sig_set=prev_sig_set)
    system_msg = (
        "You are a web automation agent. Given the task, step number, state, history, and state diff, choose ONE next action. "
        "Return JSON only (no markdown). Think step-by-step privately before answering. "
        "Elements prefixed with '*' in BROWSER_STATE are new since the previous step (URL unchanged). "
        "Do NOT provide detailed chain-of-thought. "
        "Return a JSON object with keys: action, candidate_id, text, url, evaluation_previous_goal, memory, next_goal. Preserve the current URL query parameters (e.g., seed) unless the task requires changing them. "
        "If a login/register/contact form is present, usually fill required fields before submitting. "
        "After you fill the required fields, submit the form (prefer stable attribute selectors). If the task asks you to navigate to a specific item that matches multiple attributes (e.g., rating/duration/year), prefer selecting a card/link whose surrounding context shows those attributes; do not type numeric constraints into a generic search box unless the task explicitly asks to search by text. "
        "action must be one of: click,type,select,navigate,scroll_down,scroll_up,done. "
        "Constraints: for click/type/select, candidate_id must be an integer index into the BROWSER_STATE list (the number inside [..]). "
        "For type/select, text must be non-empty. "
        "Avoid done unless the task is clearly completed."
    )

    history_lines: List[str] = []
    for h in (history or [])[-6:]:
        step = h.get("step", "?")
        action = h.get("action", "")
        cid = h.get("candidate_id")
        text = h.get("text", "")
        ok = h.get('exec_ok', True)
        err = h.get('error')
        suffix = 'OK' if ok else f"FAILED err={str(err)[:80]}"
        history_lines.append(f"{step}. {action} cid={cid} text={text} [{suffix}]")

    hint = _history_hint(history)

    structured = _structured_hints(task, candidates)
    agent_mem = ""
    try:
        st2 = _TASK_STATE.get(task_id) if task_id else None
        if isinstance(st2, dict):
            pm = str(st2.get("memory") or "").strip()
            pg = str(st2.get("next_goal") or "").strip()
            if pm or pg:
                agent_mem = f"PREVIOUS MEMORY: {pm}\nPREVIOUS NEXT_GOAL: {pg}\n"
    except Exception:
        agent_mem = ""

    user_msg = (
        f"You have a task and must decide the next single browser action.\n"
        f"TASK: {task}\n"
        f"STEP: {int(step_index)}\n"
        f"URL: {url}\n\n"
        f"CURRENT STATE (TEXT SUMMARY):\n{page_summary}\n\n"
        + (f"CREDENTIALS:\n{credentials_hint}\n\n" if credentials_hint else "")
        + (f"DOM DIGEST (STRUCTURED):\n{dom_digest}\n\n" if dom_digest else "")
        + f"STRUCTURED STATE (JSON):\n{json.dumps(structured, ensure_ascii=True)}\n\n"
        + (f"HISTORY (last steps):\n{chr(10).join(history_lines)}\n\n" if history_lines else "")
        + (f"STATE HINT: {extra_hint}\n\n" if extra_hint else "")
        + (f"STATE DELTA (prev -> current): {state_delta}\n\n" if state_delta else "")
        + "BROWSER_STATE (interactive elements):\n" + browser_state + "\n\n"
        + "Instructions:\n"
        + "- Output JSON only.\n"
        + "- Return ONE action for this step (no multi-step sequences).\n"
        + "- If you need to do a multi-step procedure (login/register/contact), pick the best next step only.\n"
        + "- Use candidate_id for click/type/select and ensure it is in-range.\n"
        + "- Use navigate with a full URL when you need to change pages (prefer preserving existing query params like seed).\n"
        + "- For type/select, include non-empty text.\n"
        + "- Provide evaluation_previous_goal, memory, next_goal (each 1 sentence). Do NOT provide detailed chain-of-thought.\n"
        + "- If CREDENTIALS are provided, use those exact values when typing.\n"
    )

    model = os.getenv("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3-0324")
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
        if a not in {"click", "type", "select", "navigate", "scroll_down", "scroll_up", "done"}:
            return False
        if a == "navigate":
            u = obj.get("url")
            if not isinstance(u, str) or not u.strip():
                return False
            try:
                if _same_path_query(str(u).strip(), str(url).strip(), base_a=str(url).strip(), base_b=""):
                    return False
            except Exception:
                if str(u).strip() == str(url).strip():
                    return False
            return True
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
    credentials_hint = _format_credentials_hint(relevant_data)
    page_summary = _summarize_html(html)
    dom_digest = _dom_digest(html)
    task = str(task or "")
    task_for_llm = _rewrite_task_for_llm(task, relevant_data)


    # Extract + rank candidates before LLM.
    candidates = _extract_candidates(html, max_candidates=80)
    candidates_all = list(candidates)
    candidates = _select_candidates_for_llm(task, candidates_all, current_url=str(url), max_total=60)

    if task_id == "check":
        # Permit repo self-checks without requiring OPENAI credentials.
        if candidates:
            return _resp([{"type": "ClickAction", "selector": candidates[0].click_selector()}], {"decision": "check_click", "candidate_id": 0})
        return _resp([{"type": "WaitAction", "time_seconds": 0.1}], {"decision": "check_wait"})


    st = _TASK_STATE.get(task_id) if task_id else None
    effective_url = str(url)
    try:
        if isinstance(st, dict):
            eu = str(st.get("effective_url") or "").strip()
            if eu:
                effective_url = eu
    except Exception:
        effective_url = str(url)
    extra_hint = ""
    prev_sig_set = None
    try:
        if isinstance(st, dict):
            prev = st.get('prev_sig_set')
            if isinstance(prev, list):
                prev_sig_set = set(str(x) for x in prev)
    except Exception:
        prev_sig_set = None

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
            url=effective_url,
            candidates=candidates,
            page_summary=page_summary,
            dom_digest=dom_digest,
            history=history,
            credentials_hint=credentials_hint,
            extra_hint=extra_hint,
            state_delta=state_delta,
            prev_sig_set=prev_sig_set,
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

    try:
        if task_id:
            st3 = _TASK_STATE.get(task_id)
            if isinstance(st3, dict):
                if isinstance(decision.get("memory"), str):
                    st3["memory"] = decision.get("memory")
                if isinstance(decision.get("next_goal"), str):
                    st3["next_goal"] = decision.get("next_goal")
    except Exception:
        pass

    action = (decision.get("action") or "").lower()
    cid = decision.get("candidate_id")
    text = decision.get("text")

    # Be forgiving if model emits numbers as strings.
    if isinstance(cid, str) and cid.isdigit():
        cid = int(cid)


    if action == "navigate":
        nav_url_raw = str(decision.get("url") or "").strip()
        if not nav_url_raw:
            return _resp([{"type": "WaitAction", "time_seconds": 1.0}], {"decision": "navigate_missing_url"})

        nav_url = _resolve_url(nav_url_raw, effective_url or str(url))

        # If the evaluator reports a stale URL, track an effective URL ourselves to prevent navigate loops.
        if _same_path_query(nav_url, effective_url, base_a=effective_url, base_b=""):
            _update_task_state(task_id, str(url), "navigate_same_url_scroll")
            return _resp([{ "type": "ScrollAction", "down": True, "up": False}], {"decision": "scroll_override"})

        _update_task_state(task_id, str(url), f"navigate:{nav_url}")
        try:
            if task_id and isinstance(_TASK_STATE.get(task_id), dict):
                _TASK_STATE[task_id]["effective_url"] = str(nav_url)
        except Exception:
            pass
        return _resp([{"type": "NavigateAction", "url": nav_url, "go_back": False, "go_forward": False}], {"decision": "navigate", "url": nav_url, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})

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
            # Preserve seed across internal navigations when clicking href-based links.
            try:
                if isinstance(selector, dict) and selector.get("type") == "attributeValueSelector" and selector.get("attribute") == "href":
                    href = str(selector.get("value") or "")
                    fixed = _preserve_seed_url(href, effective_url or str(url))
                    if fixed and fixed != href:
                        fixed_abs = _resolve_url(fixed, effective_url or str(url))
                        # Avoid endless navigate loops if this would keep us on the same page.
                        if _same_path_query(fixed_abs, effective_url, base_a=effective_url, base_b=""):
                            _update_task_state(task_id, str(url), "navigate_seed_fix_same_url_scroll")
                            return _resp([{ "type": "ScrollAction", "down": True, "up": False}], {"decision": "scroll_override"})
                        _update_task_state(task_id, str(url), f"navigate_seed_fix:{fixed_abs}")
                        try:
                            if task_id and isinstance(_TASK_STATE.get(task_id), dict):
                                _TASK_STATE[task_id]["effective_url"] = str(fixed_abs)
                        except Exception:
                            pass
                        return _resp([{ "type": "NavigateAction", "url": fixed_abs, "go_back": False, "go_forward": False}], {"decision": "navigate", "url": fixed_abs, "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
            except Exception:
                pass
            _update_task_state(task_id, str(url), f"click:{_selector_repr(selector)}")
            return _resp([{"type": "ClickAction", "selector": selector}], {"decision": "click", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})

        if action == "type":
            if not text:
                raise HTTPException(status_code=400, detail="type action missing text")
            selector = c.type_selector()
            _update_task_state(task_id, str(url), f"type:{_selector_repr(selector)}")
            return _resp([{"type": "TypeAction", "selector": selector, "text": str(text)}], {"decision": "type", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
        if action == "select":
            if not text:
                raise HTTPException(status_code=400, detail="select action missing text")
            selector = c.type_selector()
            _update_task_state(task_id, str(url), f"select:{_selector_repr(selector)}")
            return _resp([{"type": "SelectDropDownOptionAction", "selector": selector, "text": str(text), "timeout_ms": int(os.getenv("AGENT_SELECT_TIMEOUT_MS", "4000"))}], {"decision": "select", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})

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
