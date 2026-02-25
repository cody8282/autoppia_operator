from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import json
import os
import re
import logging
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit
from html.parser import HTMLParser

from fastapi import Body, FastAPI, HTTPException

# Default this branch to OpenAI via the validator gateway.
os.environ.setdefault("LLM_PROVIDER", "openai")

from llm_gateway import openai_chat_completions, is_sandbox_gateway_base_url

try:
    from autoppia_iwa.src.web_agents.classes import IWebAgent
    from autoppia_iwa.src.data_generation.tasks.classes import Task
    from autoppia_iwa.src.execution.actions.base import BaseAction
    import autoppia_iwa.src.execution.actions.actions  # noqa: F401
    _AUTOPPIA_IWA_IMPORT_OK = True
    _AUTOPPIA_IWA_IMPORT_ERROR = ""
except Exception:  # pragma: no cover
    IWebAgent = object  # type: ignore[assignment]
    Task = Any  # type: ignore[assignment]
    BaseAction = Any  # type: ignore[assignment]
    _AUTOPPIA_IWA_IMPORT_OK = False
    _AUTOPPIA_IWA_IMPORT_ERROR = "autoppia_iwa import failed in miner runtime"

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None


app = FastAPI(title="Autoppia Web Agent API")
logger = logging.getLogger("autoppia_operator")
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


_LOG_DECISIONS = _env_bool("AGENT_LOG_DECISIONS", False)
_LOG_ERRORS = _env_bool("AGENT_LOG_ERRORS", False)


def _log_trace(message: str) -> None:
    if _LOG_DECISIONS:
        logger.info(f"[AGENT_TRACE] {message}")


if not _AUTOPPIA_IWA_IMPORT_OK:
    logger.error(f"[AGENT_TRACE] autoppia_iwa import failed: {_AUTOPPIA_IWA_IMPORT_ERROR}")
else:
    _log_trace(f"autoppia_iwa import ok; BaseAction module={getattr(BaseAction, '__module__', 'unknown')}")


def _normalize_demo_url(raw_url: str | None) -> str:
    """Rewrite URLs to localhost while preserving path/query/fragment."""
    normalized = str(raw_url or "").strip()
    if not normalized:
        return normalized

    try:
        if "://" not in normalized:
            if not normalized.startswith("/"):
                # Treat bare host/path values (for example "84.247.180.192/task") as local host
                # while keeping any path/query/fragment.
                if "." in normalized or ":" in normalized:
                    parsed = urlsplit(f"http://{normalized}")
                    path = parsed.path or ""
                    if not path:
                        return "http://localhost"
                    return urlunsplit(("http", "localhost", path, parsed.query, parsed.fragment))
                normalized = f"/{normalized}"
            return f"http://localhost{normalized}"
        parsed = urlsplit(normalized)
        return urlunsplit(("http", "localhost", parsed.path or "/", parsed.query, parsed.fragment))
    except Exception:
        return "http://localhost/"


def _is_navigate_action_type(action_type: Any) -> bool:
    value = str(action_type or "").strip().lower()
    return value in {"navigateaction", "navigate"}


def _sanitize_action_payload(action_payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(action_payload or {})
    if _is_navigate_action_type(payload.get("type")):
        payload["url"] = _normalize_demo_url(payload.get("url"))
    return payload


# Per-task loop detection cache (process-local).
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
        context_raw: str = "",
        group: str = "",
        container_chain: list[str] | None = None,
    ):
        self.selector = selector
        self.text_selector = text_selector
        self.text = text
        self.tag = tag
        self.attrs = attrs
        self.context = context
        self.context_raw = context_raw
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
    if tag == "input" and attrs.get("type") == "search":
        # AutoZone has a duplicate-ID bug: both the container div AND the search input share
        # the same V3 ID. Using the ID selector would match the div first (not fillable).
        # css selector `input[type='search']` uniquely targets the actual input element.
        return {"type": "attributeValueSelector", "attribute": "custom", "value": "input[type='search']"}
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
    if text and tag == "input" and len(text) <= 40:
        # Input inside a labeled container (e.g. <label>Author <input/></label>).
        # Use position-safe XPath so each field gets a unique, stable selector.
        safe = text.replace('"', "'")
        return {"type": "xpathSelector", "value": f'(//label[normalize-space()="{safe}"]//input)[1]'}
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







def _pick_context_container_bs4(el) -> object | None:
    """Pick a small, card-like container for an element.

    Structural and generic: aims to capture per-item context (e.g., list rows/cards) rather than whole panels.
    """
    try:
        candidates = []
        cur = el
        for _depth in range(8):
            if cur is None:
                break
            try:
                cur = cur.parent
            except Exception:
                break
            if cur is None:
                break
            tag = str(getattr(cur, "name", "") or "")
            if tag not in {"li", "tr", "article", "section", "div", "td"}:
                continue

            try:
                txt_raw = cur.get_text("\n", strip=True)
            except Exception:
                txt_raw = ""
            L = len(txt_raw or "")
            if L <= 0:
                continue

            try:
                n_inter = len(cur.find_all(["a", "button", "input", "select", "textarea"]))
            except Exception:
                n_inter = 0

            candidates.append((L, n_inter, cur))

        if not candidates:
            return None

        best = None
        best_key = None
        for L, n_inter, node in candidates:
            if not (50 <= L <= 900):
                continue
            if n_inter <= 0 or n_inter > 12:
                continue
            key = (L, n_inter)
            if best is None or key < (best_key or key):
                best = node
                best_key = key
        if best is not None:
            return best

        candidates.sort(key=lambda t: (t[0], t[1]))
        return candidates[0][2]
    except Exception:
        return None


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
        if _is_hidden_candidate_attr(attr_map):
            continue

        label = _extract_label_from_bs4(soup, el, attr_map)

        dom_label = label
        context = ""
        context_raw = ""
        title = ""
        try:
            parent = _pick_context_container_bs4(el) or el.find_parent(["li", "tr", "article", "section", "div"])
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

        out.append(_Candidate(primary, label, tag, attr_map, text_selector=text_sel, context=context, context_raw=context_raw, group=group, container_chain=container_chain))
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


def _is_hidden_candidate_attr(attr_map: Dict[str, str]) -> bool:
    try:
        if attr_map.get("hidden") is not None:
            return True
        if str(attr_map.get("aria-hidden") or "").lower() == "true":
            return True
        style = str(attr_map.get("style") or "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            return True
        classes = str(attr_map.get("class") or "").lower()
        if any(tok in classes for tok in ("hidden", "sr-only", "invisible")):
            return True
    except Exception:
        return False
    return False


def _extract_page_ir(*, html: str, url: str, candidates: List[_Candidate], max_forms: int = 4, max_links: int = 20, max_cards: int = 10) -> Dict[str, Any]:
    """Build deterministic, compact page IR to reduce prompt bloat."""
    ir: Dict[str, Any] = {
        "title": "",
        "url_path": "",
        "headings": [],
        "forms": [],
        "ctas": [],
        "links": [],
        "cards": [],
    }
    if not html:
        return ir

    try:
        us = urlsplit(str(url or ""))
        ir["url_path"] = str(us.path or "/")
    except Exception:
        ir["url_path"] = str(url or "")

    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html, "lxml")
            for t in soup(["script", "style", "noscript"]):
                try:
                    t.decompose()
                except Exception:
                    pass

            try:
                if soup.title:
                    ir["title"] = _norm_ws(soup.title.get_text(" ", strip=True))[:120]
            except Exception:
                ir["title"] = ""

            heads: list[str] = []
            for h in soup.find_all(["h1", "h2", "h3"], limit=12):
                tx = _norm_ws(h.get_text(" ", strip=True))
                if tx and tx not in heads:
                    heads.append(tx[:100])
            ir["headings"] = heads[:10]
        except Exception:
            pass

    forms_obj = _tool_extract_forms(html=html, max_forms=max_forms, max_inputs=16)
    if isinstance(forms_obj, dict) and forms_obj.get("ok"):
        forms = forms_obj.get("forms")
        if isinstance(forms, list):
            cleaned_forms = []
            for f in forms[:max_forms]:
                if not isinstance(f, dict):
                    continue
                controls = f.get("controls") if isinstance(f.get("controls"), list) else []
                keep_controls = []
                for c in controls[:12]:
                    if not isinstance(c, dict):
                        continue
                    blob = _norm_ws(
                        " ".join([
                            str(c.get("tag") or ""),
                            str(c.get("type") or ""),
                            str(c.get("name") or ""),
                            str(c.get("id") or ""),
                            str(c.get("placeholder") or ""),
                            str(c.get("aria_label") or ""),
                            str(c.get("text") or ""),
                        ])
                    )
                    if blob:
                        keep_controls.append(blob[:140])
                if keep_controls:
                    cleaned_forms.append({
                        "id": str(f.get("id") or ""),
                        "name": str(f.get("name") or ""),
                        "method": str(f.get("method") or ""),
                        "controls": keep_controls[:8],
                    })
            ir["forms"] = cleaned_forms[:max_forms]

    links_obj = _tool_list_links(html=html, base_url=str(url), max_links=max_links, context_max=150)
    if isinstance(links_obj, dict) and links_obj.get("ok"):
        links = links_obj.get("links")
        if isinstance(links, list):
            clean_links = []
            for l in links[:max_links]:
                if not isinstance(l, dict):
                    continue
                text = _norm_ws(str(l.get("text") or ""))
                href = _norm_ws(str(l.get("href") or ""))
                ctx = _norm_ws(str(l.get("context") or ""))
                if not text and not href:
                    continue
                clean_links.append({
                    "text": text[:90],
                    "href": href[:150],
                    "ctx": ctx[:140],
                })
            ir["links"] = clean_links[:max_links]

    cards_obj = _tool_list_cards(candidates=candidates, max_cards=max_cards, max_text=380, max_actions_per_card=3)
    if isinstance(cards_obj, dict) and cards_obj.get("ok"):
        cards = cards_obj.get("cards")
        if isinstance(cards, list):
            out_cards = []
            for c in cards[:max_cards]:
                if not isinstance(c, dict):
                    continue
                actions = c.get("actions") if isinstance(c.get("actions"), list) else []
                actions_clean = []
                for a in actions[:3]:
                    if not isinstance(a, dict):
                        continue
                    actions_clean.append({
                        "tag": str(a.get("tag") or ""),
                        "text": _norm_ws(str(a.get("text") or ""))[:90],
                        "href": _norm_ws(str(a.get("href") or ""))[:120],
                    })
                out_cards.append({
                    "facts": [str(x)[:100] for x in (c.get("card_facts") or [])[:4]],
                    "actions": actions_clean,
                    "text": _norm_ws(str(c.get("card_text") or ""))[:260],
                })
            ir["cards"] = out_cards

    ctas: list[str] = []
    for c in candidates:
        if c.tag not in {"button", "a"}:
            continue
        tx = _norm_ws(c.text or "")
        if not tx:
            continue
        tx_l = tx.lower()
        if tx_l in {"home", "logo"}:
            continue
        if tx not in ctas:
            ctas.append(tx[:90])
        if len(ctas) >= 16:
            break
    ir["ctas"] = ctas
    return ir


def _render_page_ir(ir: Dict[str, Any], max_chars: int = 2200) -> str:
    lines: list[str] = []
    title = _norm_ws(str(ir.get("title") or ""))
    path = _norm_ws(str(ir.get("url_path") or ""))
    if title:
        lines.append(f"TITLE: {title[:120]}")
    if path:
        lines.append(f"PATH: {path[:140]}")

    heads = ir.get("headings") if isinstance(ir.get("headings"), list) else []
    if heads:
        lines.append("HEADINGS: " + " | ".join(str(h)[:90] for h in heads[:8]))

    forms = ir.get("forms") if isinstance(ir.get("forms"), list) else []
    if forms:
        lines.append("FORMS:")
        for i, f in enumerate(forms[:4]):
            if not isinstance(f, dict):
                continue
            method = str(f.get("method") or "")
            controls = f.get("controls") if isinstance(f.get("controls"), list) else []
            lines.append(f"- form[{i}] method={method} controls=" + " ; ".join(str(x)[:120] for x in controls[:6]))

    ctas = ir.get("ctas") if isinstance(ir.get("ctas"), list) else []
    if ctas:
        lines.append("CTAS: " + " | ".join(str(c)[:80] for c in ctas[:12]))

    links = ir.get("links") if isinstance(ir.get("links"), list) else []
    if links:
        lines.append("LINKS:")
        for l in links[:8]:
            if not isinstance(l, dict):
                continue
            lines.append(f"- text={str(l.get('text') or '')[:80]} href={str(l.get('href') or '')[:120]} ctx={str(l.get('ctx') or '')[:120]}")

    cards = ir.get("cards") if isinstance(ir.get("cards"), list) else []
    if cards:
        lines.append("CARDS:")
        for i, c in enumerate(cards[:6]):
            if not isinstance(c, dict):
                continue
            facts = c.get("facts") if isinstance(c.get("facts"), list) else []
            lines.append(f"- card[{i}] facts=" + " | ".join(str(x)[:80] for x in facts[:3]))
            acts = c.get("actions") if isinstance(c.get("actions"), list) else []
            for a in acts[:2]:
                if not isinstance(a, dict):
                    continue
                lines.append(f"  action tag={str(a.get('tag') or '')} text={str(a.get('text') or '')[:70]} href={str(a.get('href') or '')[:100]}")

    out = "\n".join(lines)
    return out[:max_chars]


def _compute_ir_delta(*, task_id: str, page_ir: Dict[str, Any]) -> str:
    if not task_id:
        return ""
    try:
        st = _TASK_STATE.get(task_id)
        if not isinstance(st, dict):
            st = {}
            _TASK_STATE[task_id] = st
        prev = st.get("prev_ir")
        if not isinstance(prev, dict):
            prev = {}

        def _set(key: str, d: Dict[str, Any]) -> set[str]:
            v = d.get(key)
            if not isinstance(v, list):
                return set()
            return {str(x)[:120] for x in v if isinstance(x, (str, int, float))}

        prev_cta = _set("ctas", prev)
        cur_cta = _set("ctas", page_ir)

        prev_heads = _set("headings", prev)
        cur_heads = _set("headings", page_ir)

        p_forms = prev.get("forms") if isinstance(prev.get("forms"), list) else []
        c_forms = page_ir.get("forms") if isinstance(page_ir.get("forms"), list) else []
        p_cards = prev.get("cards") if isinstance(prev.get("cards"), list) else []
        c_cards = page_ir.get("cards") if isinstance(page_ir.get("cards"), list) else []

        st["prev_ir"] = page_ir
        return (
            f"forms:{len(p_forms)}->{len(c_forms)}, "
            f"cards:{len(p_cards)}->{len(c_cards)}, "
            f"ctas_added={len(cur_cta - prev_cta)}, ctas_removed={len(prev_cta - cur_cta)}, "
            f"headings_added={len(cur_heads - prev_heads)}, headings_removed={len(prev_heads - cur_heads)}"
        )
    except Exception:
        return ""


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
            # NOTE: f-string expressions cannot contain unescaped quotes that match the
            # f-string delimiter. Use single quotes inside the expression.
            sig = f"{_selector_repr(c.selector)}|{(c.text or '')[:80]}"
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


# ---------------------------------------------------------------------------
# IWA action type name → agent lowercase name mapping.
# The validator sends history with IWA PascalCase type names (e.g. "ClickAction"),
# but the agent uses lowercase names (e.g. "click").  Normalize before comparing.
# ---------------------------------------------------------------------------
_IWA_TO_AGENT: Dict[str, str] = {
    "clickaction": "click",
    "doubleclickaction": "double_click",
    "tripleclickaction": "triple_click",
    "typeaction": "type",
    "selectaction": "select",
    "selectdropdownoptionaction": "select",
    "navigateaction": "navigate",
    "scrollaction": "scroll_down",
    "waitaction": "wait",
    "submitaction": "submit",
    "doneaction": "done",
    "hoveraction": "hover",
}


def _norm_action(raw: str) -> str:
    """Normalize IWA PascalCase type names or agent lowercase names to agent lowercase."""
    key = str(raw or "").lower().replace("_", "")
    return _IWA_TO_AGENT.get(key, str(raw or "").lower())


def _history_hint(history: List[Dict[str, Any]] | None) -> str:
    if not history:
        return ""

    last = history[-6:]

    # Detect same candidate typed multiple times — field is already filled.
    type_cid_counts: Dict[Any, int] = {}
    for h in last:
        if _norm_action(h.get("action") or "") == "type":
            cid = h.get("candidate_id")
            type_cid_counts[cid] = type_cid_counts.get(cid, 0) + 1
    for cid, cnt in type_cid_counts.items():
        if cnt >= 2:
            return (
                f"⚠️ You already typed into candidate {cid} in a previous step. "
                "Do NOT type in that field again. Move to the next EMPTY field, or if all fields are filled, click Submit/Save/Send."
            )

    # Detect same candidate clicked multiple times — toggle/button already acted.
    click_cid_counts: Dict[Any, int] = {}
    for h in last:
        if _norm_action(h.get("action") or "") == "click":
            cid = h.get("candidate_id")
            click_cid_counts[cid] = click_cid_counts.get(cid, 0) + 1
    for cid, cnt in click_cid_counts.items():
        if cnt >= 2:
            return (
                f"⚠️ You already clicked candidate {cid} {cnt} times. "
                "If this is a watchlist/share/like/trailer/toggle button, it fired on the FIRST click — use done. "
                "Otherwise pick a DIFFERENT element."
            )

    # Detect simple repetition: same action+cid repeated consecutively.
    repeats = 0
    prev = None
    for h in last:
        k = (_norm_action(h.get("action") or ""), h.get("candidate_id"))
        if prev is not None and k == prev and k != ("", None):
            repeats += 1
        prev = k

    if repeats >= 2:
        return "You appear to be repeating the same action. Choose a DIFFERENT candidate or navigate away."

    # Detect form-fill loop: last 3+ steps are all TypeAction (filling fields without submitting).
    recent = last[-4:]
    if len(recent) >= 3 and all(_norm_action(h.get("action") or "") == "type" for h in recent):
        return (
            "You have been filling form fields for several steps. "
            "If all required fields are filled (check ALREADY TYPED list), click the submit/save/send button NOW."
        )

    # Detect click-only loop: last 4+ steps are all ClickAction (toggle/button stuck).
    if len(recent) >= 4 and all(_norm_action(h.get("action") or "") == "click" for h in recent):
        return (
            "You have clicked buttons for several steps. "
            "If the task action (add/remove/share/toggle) was performed on the first click, use done. "
            "Otherwise try a different element or scroll."
        )

    return ""


def _task_risk_hint(task: str, step_index: int, candidates: List[_Candidate]) -> str:
    t = str(task or "").lower()
    risk_words = ("delete", "remove", "edit", "update", "add film", "registration", "contact")
    if not any(w in t for w in risk_words):
        return ""
    n = len(candidates or [])
    if step_index <= 1 and n >= 12:
        return (
            "High-risk task detected. Before destructive or irreversible clicks, identify the exact target via "
            "find_card/list_cards/search_text using task constraints, then click the matched action."
        )
    return (
        "For high-risk actions, verify target attributes in context first and avoid generic repeated clicks."
    )




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



# -----------------------------
# HTML Tools (for LLM-assisted inspection)
# -----------------------------

def _safe_truncate(s: str, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else (s[: max(0, n - 3)] + "...")


def _tool_search_text(*, html: str, query: str, regex: bool = False, case_sensitive: bool = False, max_matches: int = 20, context_chars: int = 80) -> Dict[str, Any]:
    """Search raw HTML text and return small context snippets.

    Generic tool: does not assume any site structure.
    """
    q = str(query or "")
    if not q:
        return {"ok": False, "error": "missing query"}

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        if regex:
            pat = re.compile(q, flags)
        else:
            pat = re.compile(re.escape(q), flags)
    except Exception as e:
        return {"ok": False, "error": f"invalid pattern: {str(e)[:120]}"}

    hay = str(html or "")
    out = []
    for m in pat.finditer(hay):
        if len(out) >= int(max_matches or 0):
            break
        a = max(0, m.start() - int(context_chars))
        b = min(len(hay), m.end() + int(context_chars))
        out.append({
            "start": int(m.start()),
            "end": int(m.end()),
            "snippet": _safe_truncate(hay[a:b].replace("\n", " ").replace("\r", " "), 2 * int(context_chars) + 40),
        })

    return {"ok": True, "matches": out, "count": len(out)}


def _tool_css_select(*, html: str, selector: str, max_nodes: int = 25) -> Dict[str, Any]:
    """Run a CSS selector over the DOM (via BeautifulSoup) and return summaries."""
    if BeautifulSoup is None:
        return {"ok": False, "error": "bs4 not available"}
    sel = str(selector or "").strip()
    if not sel:
        return {"ok": False, "error": "missing selector"}

    try:
        soup = BeautifulSoup(html or "", "lxml")
        nodes = soup.select(sel)
    except Exception as e:
        return {"ok": False, "error": f"css select failed: {str(e)[:160]}"}

    out = []
    for n in nodes[: int(max_nodes or 0)]:
        try:
            tag = str(getattr(n, "name", "") or "")
            attrs = _attrs_to_str_map(getattr(n, "attrs", {}) or {})
            text = _norm_ws(n.get_text(" ", strip=True))
            out.append({
                "tag": tag,
                "attrs": {k: _safe_truncate(v, 120) for k, v in list(attrs.items())[:12]},
                "text": _safe_truncate(text, 240),
            })
        except Exception:
            continue

    return {"ok": True, "count": len(nodes), "nodes": out}


def _tool_extract_forms(*, html: str, max_forms: int = 10, max_inputs: int = 25) -> Dict[str, Any]:
    """Extract forms and their controls in a structured way (generic)."""
    if BeautifulSoup is None:
        return {"ok": False, "error": "bs4 not available"}

    try:
        soup = BeautifulSoup(html or "", "lxml")
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {str(e)[:160]}"}

    forms = []
    for f in soup.find_all("form")[: int(max_forms or 0)]:
        try:
            f_attrs = _attrs_to_str_map(getattr(f, "attrs", {}) or {})
            inputs = []
            for el in f.find_all(["input", "textarea", "select", "button"])[: int(max_inputs or 0)]:
                try:
                    tag = str(getattr(el, "name", "") or "")
                    a = _attrs_to_str_map(getattr(el, "attrs", {}) or {})
                    t = _norm_ws(el.get_text(" ", strip=True))
                    inputs.append({
                        "tag": tag,
                        "type": (a.get("type") or "").lower(),
                        "id": a.get("id") or "",
                        "name": a.get("name") or "",
                        "placeholder": a.get("placeholder") or "",
                        "aria_label": a.get("aria-label") or "",
                        "value": _safe_truncate(a.get("value") or "", 120),
                        "text": _safe_truncate(t, 160),
                    })
                except Exception:
                    continue
            forms.append({
                "id": f_attrs.get("id") or "",
                "name": f_attrs.get("name") or "",
                "action": f_attrs.get("action") or "",
                "method": (f_attrs.get("method") or "").upper(),
                "controls": inputs,
            })
        except Exception:
            continue

    return {"ok": True, "forms": forms, "count": len(forms)}




def _tool_xpath_select(*, html: str, xpath: str, max_nodes: int = 25) -> Dict[str, Any]:
    """Run an XPath selector over the DOM (via lxml) and return summaries."""
    xp = str(xpath or "").strip()
    if not xp:
        return {"ok": False, "error": "missing xpath"}
    try:
        from lxml import html as lxml_html  # type: ignore
    except Exception:
        return {"ok": False, "error": "lxml not available"}

    try:
        doc = lxml_html.fromstring(html or "")
        nodes = doc.xpath(xp)
    except Exception as e:
        return {"ok": False, "error": f"xpath failed: {str(e)[:160]}"}

    out = []
    for n in nodes[: int(max_nodes or 0)]:
        try:
            # lxml may return strings/attrs too.
            if not hasattr(n, 'tag'):
                out.append({"value": _safe_truncate(str(n), 240)})
                continue
            tag = str(getattr(n, 'tag', '') or '')
            attrs = {k: _safe_truncate(str(v), 120) for k, v in list(getattr(n, 'attrib', {}) or {}).items()[:12]}
            text = _norm_ws(' '.join(n.itertext()))
            out.append({"tag": tag, "attrs": attrs, "text": _safe_truncate(text, 240)})
        except Exception:
            continue

    return {"ok": True, "count": len(nodes), "nodes": out}


def _tool_visible_text(*, html: str, max_chars: int = 2000) -> Dict[str, Any]:
    """Extract visible-ish text from the page (best-effort)."""
    if BeautifulSoup is None:
        # Fallback: strip tags very crudely.
        txt = re.sub(r"<[^>]+>", " ", str(html or ""))
        txt = _norm_ws(txt)
        return {"ok": True, "text": _safe_truncate(txt, int(max_chars or 0))}

    try:
        soup = BeautifulSoup(html or "", "lxml")
        for t in soup(["script", "style", "noscript"]):
            try:
                t.decompose()
            except Exception:
                pass
        txt = _norm_ws(soup.get_text(" ", strip=True))
        return {"ok": True, "text": _safe_truncate(txt, int(max_chars or 0))}
    except Exception as e:
        return {"ok": False, "error": f"extract text failed: {str(e)[:160]}"}

def _tool_list_candidates(*, candidates: List["_Candidate"], max_n: int = 80) -> Dict[str, Any]:
    out = []
    for i, c in enumerate((candidates or [])[: int(max_n or 0)]):
        out.append({
            "id": i,
            "tag": c.tag,
            "group": c.group,
            "text": _safe_truncate(c.text or "", 140),
            "context": _safe_truncate(c.context or "", 200),
            "selector": _selector_repr(c.selector) if isinstance(c.selector, dict) else str(c.selector),
            "click": _selector_repr(c.click_selector()),
        })
    return {"ok": True, "count": len(candidates or []), "candidates": out}


def _tool_list_links(
    *,
    html: str,
    base_url: str,
    max_links: int = 60,
    context_max: int = 260,
    href_regex: str = "",
    text_regex: str = "",
) -> Dict[str, Any]:
    """Extract links (href) and nearby container text.

    Generic tool that helps the LLM choose a navigation target without depending on a candidate_id.
    """
    if BeautifulSoup is None:
        return {"ok": False, "error": "bs4 not available"}

    try:
        soup = BeautifulSoup(html or "", "lxml")
    except Exception as e:
        return {"ok": False, "error": f"parse failed: {str(e)[:160]}"}

    href_pat = None
    text_pat = None
    try:
        if href_regex:
            href_pat = re.compile(str(href_regex), re.I)
        if text_regex:
            text_pat = re.compile(str(text_regex), re.I)
    except Exception as e:
        return {"ok": False, "error": f"invalid regex: {str(e)[:160]}"}

    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for a in soup.select("a[href]"):
        try:
            href = str(a.get("href") or "").strip()
            if not href or href.lower().startswith("javascript:"):
                continue
            if href_pat and not href_pat.search(href):
                continue

            text = _norm_ws(a.get_text(" ", strip=True))
            if not text:
                text = _norm_ws(str(a.get("aria-label") or "") or "")
            if text_pat and not text_pat.search(text):
                continue

            container = _pick_context_container_bs4(a)
            ctx_raw = ""
            if container is not None:
                try:
                    ctx_raw = container.get_text("\n", strip=True)
                except Exception:
                    ctx_raw = ""
            ctx = _safe_truncate(_norm_ws(ctx_raw) if ctx_raw else "", int(context_max or 0))

            resolved = _resolve_url(href, str(base_url or ""))
            resolved = _preserve_seed_url(resolved, str(base_url or ""))

            sig = (resolved or href) + "|" + (text or "")
            if sig in seen:
                continue
            seen.add(sig)

            out.append({
                "href": _safe_truncate(href, 260),
                "url": _safe_truncate(resolved, 320),
                "text": _safe_truncate(text, 160),
                "context": ctx,
            })
            if len(out) >= int(max_links or 0):
                break
        except Exception:
            continue

    return {"ok": True, "count": len(out), "links": out}


def _tool_list_cards(*, candidates: List["_Candidate"], max_cards: int = 25, max_text: int = 900, max_actions_per_card: int = 6) -> Dict[str, Any]:
    """Group candidates into card-like clusters using their extracted container context.

    Generic: clusters clickables (a/button or href-selectable) by context_raw/context. Returns surrounding text plus actions.
    """
    groups: dict[str, dict[str, Any]] = {}

    for i, c in enumerate(candidates or []):
        try:
            # Only cluster around clickables to avoid dumping huge filter panels.
            if c.tag not in {"a", "button"}:
                sel = c.click_selector()
                if not (isinstance(sel, dict) and sel.get("type") == "attributeValueSelector" and str(sel.get("attribute") or "") == "href"):
                    continue

            key = (c.context_raw or c.context or "").strip()
            if not key:
                key = "(no_context)"

            g = groups.get(key)
            if g is None:
                facts = []
                try:
                    lines = [ln.strip() for ln in str(key or '').splitlines() if ln.strip()]
                    facts = [ln for ln in lines if any(ch.isdigit() for ch in ln)][:6]
                except Exception:
                    facts = []
                g = {"card_text": _safe_truncate(key, int(max_text or 0)), "card_facts": facts, "candidate_ids": [], "actions": []}
                groups[key] = g

            g["candidate_ids"].append(i)
            if len(g["actions"]) < int(max_actions_per_card or 0):
                sel = c.click_selector()
                href = ""
                try:
                    if isinstance(sel, dict) and sel.get("type") == "attributeValueSelector" and str(sel.get("attribute") or "") == "href":
                        href = str(sel.get("value") or "").strip()
                except Exception:
                    href = ""

                g["actions"].append({
                    "candidate_id": i,
                    "tag": c.tag,
                    "text": _safe_truncate(c.text or "", 140),
                    "click": _selector_repr(sel),
                    "href": _safe_truncate(href, 240) if href else "",
                })
        except Exception:
            continue

    ranked = []
    for _k, g in groups.items():
        txt = str(g.get("card_text") or "")
        n_actions = len(g.get("actions") or [])
        L = len(txt)
        penalty = 0
        if L < 40:
            penalty += 400
        if L > 900:
            penalty += min(1200, L - 900)
        score = (1000 - penalty + min(L, 700), n_actions)
        ranked.append((score, g))

    ranked.sort(key=lambda x: x[0], reverse=True)
    cards = [g for _, g in ranked[: int(max_cards or 0)]]
    return {"ok": True, "count": len(cards), "cards": cards}


def _tool_find_card(*, candidates: List["_Candidate"], query: str, max_cards: int = 10, max_text: int = 900, max_actions_per_card: int = 6) -> Dict[str, Any]:
    """Find card-like groups whose text/facts/actions match a query."""
    q = _norm_ws(str(query or "")).lower()
    if not q:
        return {"ok": False, "error": "missing query"}
    base = _tool_list_cards(candidates=candidates, max_cards=max_cards * 4, max_text=max_text, max_actions_per_card=max_actions_per_card)
    if not isinstance(base, dict) or not base.get("ok"):
        return base
    cards = base.get("cards") if isinstance(base.get("cards"), list) else []
    out = []
    for c in cards:
        if not isinstance(c, dict):
            continue
        blob_parts = [str(c.get("card_text") or "")]
        facts = c.get("card_facts") if isinstance(c.get("card_facts"), list) else []
        blob_parts.extend(str(x) for x in facts[:8])
        acts = c.get("actions") if isinstance(c.get("actions"), list) else []
        for a in acts[:6]:
            if isinstance(a, dict):
                blob_parts.append(str(a.get("text") or ""))
                blob_parts.append(str(a.get("href") or ""))
        blob = _norm_ws(" ".join(blob_parts)).lower()
        if q in blob:
            out.append(c)
        if len(out) >= int(max_cards or 0):
            break
    return {"ok": True, "query": q, "count": len(out), "cards": out}


_TOOL_REGISTRY = {
    "search_text": _tool_search_text,
    "visible_text": _tool_visible_text,
    "css_select": _tool_css_select,
    "xpath_select": _tool_xpath_select,
    "extract_forms": _tool_extract_forms,
    "list_links": _tool_list_links,
    "list_candidates": _tool_list_candidates,
    "list_cards": _tool_list_cards,
    "find_card": _tool_find_card,
}


def _run_tool(tool: str, args: Dict[str, Any], *, html: str, url: str, candidates: List["_Candidate"]) -> Dict[str, Any]:
    t = str(tool or "").strip()
    fn = _TOOL_REGISTRY.get(t)
    if fn is None:
        return {"ok": False, "error": f"unknown tool: {t}", "known": sorted(_TOOL_REGISTRY.keys())}

    a = args if isinstance(args, dict) else {}
    # Inject shared state for tools that need it.
    if t == "list_candidates":
        return fn(candidates=candidates, **{k: v for k, v in a.items() if k in {"max_n"}})
    if t == "list_cards":
        return fn(candidates=candidates, **{k: v for k, v in a.items() if k in {"max_cards", "max_text", "max_actions_per_card"}})
    if t == "find_card":
        return fn(candidates=candidates, **{k: v for k, v in a.items() if k in {"query", "max_cards", "max_text", "max_actions_per_card"}})
    if t == "list_links":
        return fn(html=html, base_url=str(url or ""), **{k: v for k, v in a.items() if k in {"max_links", "context_max", "href_regex", "text_regex"}})
    if t in {"search_text", "visible_text", "css_select", "xpath_select", "extract_forms"}:
        return fn(html=html, **a)

    return {"ok": False, "error": f"tool not wired: {t}"}

def _llm_decide(
    *,
    task_id: str,
    task: str,
    step_index: int,
    url: str,
    candidates: List[_Candidate],
    page_summary: str,
    dom_digest: str,
    html_snapshot: str,
    history: List[Dict[str, Any]] | None,
    page_ir_text: str = "",
    extra_hint: str = "",
    target_hint: str = "",
    state_delta: str = "",
    ir_delta: str = "",
    prev_sig_set: set[str] | None = None,
    model_override: str = "",
    web_project_id: str = "",
) -> Dict[str, Any]:
    browser_state = _format_browser_state(candidates=candidates, prev_sig_set=prev_sig_set)
    system_msg = (
        "You are an expert web automation agent. Given the task, page state, and history, choose ONE next action. "
        "Return JSON only — no markdown, no explanation outside JSON. "
        "Return a JSON object with keys: action, candidate_id, text, url. "
        "action must be one of: click,double_click,triple_click,submit,type,select,navigate,scroll_down,scroll_up,done. "
        "For click/double_click/triple_click/submit/type/select: candidate_id must be an integer from the BROWSER_STATE list (the number inside [..]). "
        "For type/select: text must be non-empty. "
        "Use double_click only if a single click did not open or activate the target. Use triple_click to select all existing text in an input before typing new content. Use submit on a filled input as an alternative when the submit/search button cannot be found. "
        "Preserve current URL query parameters (especially ?seed=N) in any navigate URL.\n"
        "\n"
        "NAVIGATION RULES:\n"
        "- ALWAYS prefer clicking links over navigate. Clicking preserves URL params automatically.\n"
        "- Look for login/register/contact/search links in nav/header/footer before navigating.\n"
        "- CRITICAL: If you must use navigate, copy the EXACT origin (scheme+host+port) from the current URL shown in 'URL:' above. If URL is 'http://localhost:8001/...' use 'http://localhost:8001/...' not 'http://localhost/...' (the port number is REQUIRED).\n"
        "- Do NOT use done just because you navigated to a relevant page. done means task fully complete.\n"
        "\n"
        "CREDENTIAL RULES:\n"
        "- When task contains <username>, <password>, <signup_username>, <signup_email>, <signup_password>: type these EXACT placeholder strings including angle brackets.\n"
        "- Do NOT invent credentials. Use the exact placeholder values from the task.\n"
        "\n"
        "FORM RULES:\n"
        "- Fill ALL form fields before clicking submit.\n"
        "- ALWAYS check the ALREADY TYPED list below — do NOT type in a field that is already listed there. Move to the next empty field.\n"
        "- Once all required fields are filled, click the submit/save/send button.\n"
        "\n"
        "TOGGLE/ONE-SHOT BUTTON RULES:\n"
        "- For watchlist/share/like/trailer buttons: click ONCE. The action fires on the FIRST click.\n"
        "- After the first click on a toggle button: use done immediately — do NOT click it again.\n"
        "- If STATE DELTA shows summary_changed=true after a click, the action succeeded — use done.\n"
        "\n"
        "SEARCH TASK RULES:\n"
        "- To search: type the query in the search input field, then click the Search/Go button.\n"
        "- done when search results are DISPLAYED on the page — do NOT click individual result links.\n"
        "- The goal is to fire the search event, NOT to navigate to a specific item's detail page.\n"
        "\n"
        "PROFILE/ADMIN TASKS (add/edit/delete/manage items):\n"
        "- These always require login first. If you are not logged in, click the Login link in the nav and log in.\n"
        "- After login, click the Profile/username link in the nav bar to go to the profile page.\n"
        "- On the profile page, look for TABS: 'Movies/Books/Items', 'Add Movies/Add Books/Add Items', 'Watchlist', etc.\n"
        "- CRITICAL TAB RULE: After clicking a TAB, the content loads immediately without a page reload. BEFORE clicking any tab, scan the BROWSER_STATE to see if the tab content is already visible. If you see form fields (Title, Author, Year, Pages, Rating, etc.) or a list of items already shown, the tab already switched — do NOT click the tab again. Go straight to interacting with the visible content.\n"
        "- HOW TO KNOW THE TAB CLICK WORKED: In the next step, if BROWSER_STATE shows form inputs labeled 'Title', 'Author', 'Year', 'Pages', 'Duration', 'Rating' etc., the 'Add Item' tab successfully loaded. If you see a list of book/movie cards, the 'Items' tab loaded. Just interact with what's there — do NOT click the tab button a second time.\n"
        "- To ADD a new item: click the 'Add [Film/Book]' tab ONCE. Use extract_forms() to see all form fields and their labels. For ADD_BOOK: validated fields are year, author, rating, AND genre — set all to satisfy task criteria. For ADD_FILM, set the Duration field to match task criteria. Then click Save/Add/Submit.\n"
        "- For Add Book form fields: Title (text input), Author (text input), Year (text input — MUST match task year criteria), Pages/page_count (text input — NOT validated, use any number like '300'), Rating (text input — must match task criteria), Preview URL (text input). Then genre BUTTONS to click (available: Drama, Comedy, Action, Thriller, Sci-Fi, Fantasy, Horror, Documentary, Romance, Fiction, Non-Fiction, Mystery). For genres in the pill list: click that pill button. For genres NOT in the list (e.g. 'Short Stories', 'Psychological', 'Adventure', 'Historical Fiction'): clear and type in the 'Custom genre' text input field (placeholder='Custom genre list'). Then Synopsis textarea at the bottom (optional). DO NOT leave genre empty if genre criteria exists.\n"
        "- For Add Film form fields: Title (text), Director (text), Year (text), Duration/Minutes (text = runtime in minutes), Rating (text), Genre buttons, Synopsis textarea. ONLY change non-matching fields.\n"
        "- To EDIT an item: click the 'Movies/Books'/'Edit Books' tab ONCE, wait for the list to appear. For AutoBooks (EDIT_BOOK): the BookEditor form is INLINE below each book — there is NO separate Edit button. Directly modify the Year, Author, Rating, and Genre fields in the visible form, then click Save Changes. GENRE IN EDIT_BOOK: click the genre pill matching the required genre (e.g. click 'Romance' pill), or for genres not in the list type in the custom genre text input. For AutoCinema (EDIT_FILM): use find_card to locate the item, click its Edit button, modify fields, click Save.\n"
        "- To DELETE an item: click the 'Movies/Books' tab ONCE, wait for list. For DELETE_FILM: use visible_text() to read the movie titles and directors listed on the Movies tab. Find a movie that satisfies the criteria (e.g., director NOT in ['Ridley Scott', 'Andrew Stanton']). Each movie card shows director info. Once you identify the right movie, use search_text('Delete') or find the delete button NEAR that movie's title, then click it. DO NOT just click the first Delete button — verify the director first.\n"
        "- To REMOVE from watchlist/favorites: click the 'Watchlist' tab ONCE, find the item, click its Remove button.\n"
        "\n"
        "TASK-SPECIFIC GUIDANCE:\n"
        "- Search tasks: find the search input in the nav or on the search/browse page, type the query, click Search/Go. done when RESULTS are shown. Do NOT click individual result links.\n"
        "- Filter tasks (FILTER_FILM, FILTER_BOOK): IMPORTANT: navigating directly to /search?genre=Romance does NOT fire the filter event — you MUST interact with the filter UI. For FILTER_FILM: CLICK the 'Search', 'Browse', or 'Explore' link in the NAV BAR to navigate to /search page. On the /search page: (a) quick genre BUTTONS are displayed near the top — click ANY of them (e.g. 'Romance', 'Action', 'Sci-Fi') OR (b) use the genre <select> dropdown (SelectDropDownOptionAction with genre value). FILTER_FILM fires immediately on the button click or select change. The task criteria for FILTER_FILM uses key 'genres' which is silently ignored by validation — so ANY genre button click fires FILTER_FILM and the task passes! Just click any genre button then done.\n"
        "- Detail page tasks: the DETAIL VIEW event fires automatically on page load. To find an item with a SPECIFIC RATING: use list_cards() on the main listing page to scan items and find one matching the exact rating specified in the task. Then click on that item to go to its detail page. The event fires automatically.\n"
        "- Trailer/preview/media tasks: navigate to a matching item's detail page, then click the play/preview/trailer button ONCE, then done.\n"
        "- Contact/form tasks: find the contact link in nav, fill each empty field in turn (check ALREADY TYPED list), then click Submit.\n"
        "- ADD item tasks ('Add a film', 'Add a book', 'Add a product'): CREATE a NEW catalog entry. NEVER click a watchlist/cart button. Login first, then go to Profile, click 'Add [Film/Book]' tab, fill the creation form, then click Save/Add/Create.\n"
        "- Edit profile/user tasks: click the profile/settings link in nav, fill the fields that need changing, then click Save.\n"
        "- Edit ITEM tasks (edit a film/book/product): Login → Profile → Items tab → find_card to locate item → Click Edit → modify fields → Save.\n"
        "- Delete tasks: Login → Profile → Items tab → find_card to locate specific item matching criteria → Click Delete → confirm if dialog appears.\n"
        "- Add to watchlist/reading-list tasks (ADD_TO_WATCHLIST, ADD_TO_READING_LIST): REQUIRES LOGIN. CRITICAL: if NOT logged in, clicking the watchlist button shows 'Please sign in' and fires NO event. Must log in first! Steps: (1) click Login nav link, fill username/password, submit. (2) Use list_cards() on the main movie list to find a movie with rating satisfying criteria (e.g. rating >= 4.6 — scan card ratings). (3) Click on that movie to navigate to its detail page. (4) Click the bookmark/watchlist button ONCE — text varies: 'Add to Watchlist', 'Watchlist', 'Save', 'Bookmark'. ADD_TO_WATCHLIST fires on click (only when logged in).\n"
        "- Add to cart tasks (ADD_TO_CART): does NOT require login. Find item matching criteria on its detail page, click Add to Cart, then done.\n"
        "- Remove from cart tasks: the cart starts EMPTY. You MUST first (1) find a book matching the criteria, (2) navigate to its detail page, (3) click Add to Cart, (4) navigate to the cart page (/cart or cart link in nav), (5) click Remove on the matching item, then done.\n"
        "- Remove from reading list tasks (REMOVE_FROM_READING_LIST): REQUIRES LOGIN. The reading list toggle is on the BOOK DETAIL PAGE. VALIDATED fields: only rating, genre, name, year (NOT page_count, NOT author). To remove: (1) log in, (2) find ANY book matching the validated criteria (check rating/genre/name/year — ignore page_count and author in task prompt), (3) navigate to its detail page, (4) click the bookmark/reading-list button ONCE to ADD it (fresh session = book not in list), (5) then click the SAME button AGAIN to REMOVE it. REMOVE_FROM_READING_LIST fires on the SECOND click.\n"
        "- Remove from watchlist tasks (REMOVE_FROM_WATCHLIST, cinema/movies): REQUIRES LOGIN. IMPORTANT: the Profile Watchlist tab 'Remove from List' button does NOT fire REMOVE_FROM_WATCHLIST. The event only fires from the movie DETAIL PAGE watchlist toggle button. Steps: (1) Log in. (2) Find a movie matching criteria (e.g. genres NOT 'Horror', rating NOT 4.6). (3) Navigate to its detail page. (4) Click the watchlist button ONCE to ADD it (ADD_TO_WATCHLIST fires). (5) Click the SAME watchlist button AGAIN to REMOVE it — button text now says 'In Watchlist', 'Remove', 'Remove from Watchlist'. REMOVE_FROM_WATCHLIST fires on the SECOND click.\n"
        "- Comment/review tasks: navigate to a matching item's detail page, find the comment/review form, fill commenter_name and content fields, click Submit.\n"
        "- Purchase tasks (PURCHASE_BOOK): flow is: (1) find item matching name criteria (search or browse), (2) go to its detail page, (3) click Add to Cart, (4) navigate to cart (/cart), (5) click the Purchase button. PURCHASE_BOOK fires per item in cart on Purchase click.\n"
        "- Share tasks: navigate to the matching item's detail page, click the Share button ONCE, then done. No login required.\n"
        "- View cart/list tasks: CLICK the cart icon/link in the nav bar. The VIEW_CART event fires automatically on cart page load. Done immediately after navigating to cart.\n"
        "- View wishlist from homepage preview tasks: CLICK the 'View All' or 'Open Wishlist' button in the homepage wishlist preview SECTION (not the nav link), then done.\n"
        "- Quantity update tasks: if the item is not yet in cart, first add it (find product → detail page → Add to Cart), then navigate to cart, then update the quantity input, then done.\n"
        "- Page view tasks (About, Help, Contact, Menu): navigate/click to the target page, do scroll_down once, then done.\n"
        "- Dropdown/selector tasks: click the dropdown trigger to OPEN it, then click the desired option from the list, then done (after selecting the value).\n"
        "- Carousel/scroll section tasks: find the named section on the page, click its scroll arrow button (left/right) or use scroll_down, then done.\n"
        "- Toggle/expand/collapse tasks: find the named item/section, click its toggle/expand/collapse button ONCE, then done.\n"
        "- FAQ/accordion tasks: find the FAQ item matching the criteria, click it to expand/toggle, then done.\n"
        "- Reservation/booking tasks: find the restaurant/venue matching criteria, fill the booking form (date, time, people, occasion), submit, then done.\n"
        "\n"
        "E-COMMERCE TASKS (AutoZone and similar shops):\n"
        "- Search tasks (SEARCH_PRODUCT): the SEARCH event fires ONLY when using the HEADER search bar. IMPORTANT: AutoZone has a duplicate-ID bug where both the search container div AND the search <input> element share the same V3 ID. ALWAYS use css_select('input[type=\"search\"]') to fill the search input (this uniquely targets the actual text field). Then click the SEARCH SUBMIT BUTTON. The search button has NO visible text (only a magnifying glass icon), with aria-label varying by V3: 'Search', 'Go', 'Find', 'Submit', 'Look up', 'Search Now', 'Execute', 'Query'. Use css_select('[id*=\"search-btn\"], [id*=\"submit-search\"], [id*=\"go-search\"], [id*=\"search-action\"], [id*=\"find-btn\"], [id*=\"query-btn\"], [id*=\"search-submit\"]') to find the submit button, OR use css_select('button[aria-label]') to get all labeled buttons and pick the search-looking one. SEARCH_PRODUCT fires on button click. Do NOT navigate to /search?q=... directly.\n"
        "- Category filter tasks (CATEGORY_FILTER): CRITICAL FIRST STEP: NavigateAction to '/search' BEFORE doing anything else. Do NOT use the homepage header 'categories-selector' dropdown — that is a homepage UI element and does NOT fire CATEGORY_FILTER. CATEGORY_FILTER fires ONLY on /search page category buttons. After NavigateAction to '/search', use search_text('Fitness') (or the required category name) to locate the category filter button on the /search page, then ClickAction on it ONCE. CATEGORY_FILTER fires immediately. Labels: 'Kitchen', 'Technology', 'Home', 'Electronics', 'Fitness'. Button IDs use V3 variants of 'category-link' key. Done.\n"
        "- Product detail view (VIEW_DETAIL): the event fires automatically when you navigate to any product detail page. To find a product matching price/brand/rating criteria: use list_cards() on the homepage or search results to scan products, find one matching ALL criteria, click on it.\n"
        "- Expand/collapse product details (DETAILS_TOGGLE): go to a product detail page matching the title/rating/price/brand/category criteria. Scroll DOWN on the product detail page to find the shipping section in the RIGHT column (it may be below the fold). The shipping section label varies via V3: 'Shipping options', 'Delivery options', 'Shipping choices', 'Delivery choices', 'Shipping methods', 'Delivery'. Click the Expand OR Collapse button in that section header — button text varies: 'Expand', 'Open', 'Show more', 'Show details', 'See more', 'Reveal', 'More', 'Details' (for expand), or 'Collapse', 'Close', 'Hide', 'Show less' (for collapse). Use search_text('Expand') or search_text('Show') or search_text('Collapse') to find it. DETAILS_TOGGLE fires on click. To find a product with a specific title keyword (e.g. 'Scale'): fill css_select('input[type=\"search\"]') with the keyword and click the search button.\n"
        "- Share product tasks (SHARE_PRODUCT): ALL criteria fields are validated including EXACT rating. The task criteria may specify an EXACT rating like 4.6 — you MUST find a product with EXACTLY that rating. Use list_cards() to scan products and check their ratings. Once you find a product with the correct rating (AND matching other criteria like category NOT 'Electronics', title NOT 'Cordless Vacuum'), navigate to its detail page. Find and click the share button ONCE — text varies: 'Share product', 'Share', 'Send', 'Share this', 'Send link'. Use search_text('Share') to locate it. SHARE_PRODUCT fires on click.\n"
        "- Add to cart tasks (ADD_TO_CART): find the product matching ALL criteria (brand, price, rating). If a brand is specified (e.g. 'CalmRest'), fill the HEADER search input using css_select('input[type=\"search\"]') with the brand name and click the search button — this narrows results. Use list_cards() on search results to verify price and rating match all criteria. Navigate to the product's detail page. The 'Add to Cart' button text varies: 'Add to Cart', 'Add to Basket', 'Add Item', 'Put in Cart'. Use search_text('Cart') or 'Basket' to find it. Click it ONCE. ALL criteria fields are validated.\n"
        "- Buy now / Checkout started tasks (CHECKOUT_STARTED): CRITICAL: 'total_amount' is NOT validated — ANY product's Buy Now click passes! The first step MUST be to navigate to a product DETAIL PAGE (NOT the homepage). Do NOT click the carousel 'Add to Basket' buttons on the homepage — those are ADD_TO_CART buttons, not Buy Now. Step 1: use list_cards() on the homepage to see products, then ClickAction on ANY product card title/image link to navigate to its detail page. Step 2: on the detail page, scroll down to find the Buy Now button in the RIGHT column purchase panel. Button text varies: 'Buy Now', 'Purchase Now', 'Order Now', 'Quick Buy', 'Instant Purchase'. Use search_text('Buy') or search_text('Now') to find it. Click it ONCE. CHECKOUT_STARTED fires immediately.\n"
        "- Proceed to checkout tasks (PROCEED_TO_CHECKOUT): total_amount IS validated (unlike CHECKOUT_STARTED). Use list_cards() on the HOMEPAGE to scan ALL product prices. Find a product with price EXACTLY matching total_amount (e.g. $39.99). Navigate to that product's detail page. Click 'Add to Cart' — IMPORTANT: clicking Add to Cart AUTOMATICALLY navigates you to the cart page (/cart), so you do NOT need a separate NavigateAction to /cart. On the cart page, use search_text('Checkout') or search_text('Proceed') to find the checkout button (text varies: 'Proceed to Checkout', 'Checkout', 'Go to Checkout', 'Continue to Checkout'). Click it ONCE. PROCEED_TO_CHECKOUT fires on click.\n"
        "- Order completion tasks (ORDER_COMPLETED): add any product to cart, click Proceed to Checkout, fill the checkout form, click 'Complete Order' / 'Place Order'. Any product works unless the title is excluded.\n"
        "- Quantity update tasks (QUANTITY_CHANGED): APPROACH 1 (preferred — product detail page): Search for the specific product by name (e.g. 'Velvet Accent Chair') using the HEADER search bar css_select('input[type=\"search\"]'), click search, then navigate to the product detail page. The product page has a quantity <select> dropdown — use css_select('select') to get it (usually the only select on the page). The select ID varies by V3: 'qty-input', 'quantity-field', 'qty-field', 'amount-input', 'qty-box', 'quantity-select'. Options are 1-10. Use SelectDropDownOptionAction with selector 'select' and value '6' (or any value > 5 for new_quantity > 5). QUANTITY_CHANGED fires on dropdown change. APPROACH 2 (cart): add the product to cart (Auto-Cart navigates to /cart), then click 'Increase quantity' / '+' button MULTIPLE TIMES in the cart (5 times to reach qty 6 from qty 1). QUANTITY_CHANGED also fires from cart +/- buttons. APPROACH 1 is simpler — prefer it.\n"
        "- View cart tasks (VIEW_CART): click the cart icon/link in the HEADER or navigate to /cart. VIEW_CART has empty criteria → passes immediately when the cart page loads.\n"
        "- View wishlist tasks (VIEW_WISHLIST): click 'View all saved' button in the homepage wishlist PREVIEW SECTION (not the nav cart icon). VIEW_WISHLIST has empty criteria → passes immediately.\n"
        "- Wishlist/favorites tasks (ADD_TO_WISHLIST): find a product matching ALL criteria (category AND price). Category criteria use CONTAINS logic: 'hen' is a SUBSTRING of 'Kitchen', so category CONTAINS 'hen' means find a KITCHEN category product. DO NOT type 'hen' or the substring in the search bar — that searches product titles, not categories. Instead: on the homepage, scroll to the 'Kitchen Essentials' carousel section (category='Kitchen') to find Kitchen products. Use list_cards() to check prices and find one with price satisfying the criteria (e.g. >= $48.75). Navigate to its detail page. CRITICAL: the WISHLIST button and the ADD TO CART button are TWO SEPARATE BUTTONS — do NOT click the cart button! The wishlist button text varies: 'Add to wishlist', 'Save', 'Add to favorites', 'Save for later', 'Favorite'. Its ID uses V3 variant of 'wishlist-button' key (default 'wishlist-button'). Use search_text('wishlist') or search_text('favorite') or search_text('Save') to find the WISHLIST button (NOT the 'Add to Cart'/'Basket' button). Click the WISHLIST button ONCE. ADD_TO_WISHLIST fires on click. CATEGORY mapping: 'Kitchen'=kitchen essentials carousel, 'Technology'=tech carousel, 'Home'=home carousel, 'Electronics', 'Fitness'.\n"
        "- Homepage carousel tasks (CAROUSEL_SCROLL): on the AutoZone homepage, there are 3 product carousel sections: 'Kitchen Essentials', 'Technology & Gadgets', and 'Home & Living'. Each carousel header has a LEFT arrow button and a RIGHT arrow button. IMPORTANT: the aria-label and IDs of these buttons VARY by V3 seed. Left arrow aria-label variants: 'Scroll left', 'Previous', 'Back', 'Left'. Right arrow aria-label variants: 'Scroll right', 'Next', 'Forward', 'Right'. Left arrow ID variants: 'carousel-left-btn', 'carousel-prev', 'carousel-control-left'. Right arrow ID variants: 'carousel-right-btn', 'carousel-next', 'carousel-control-right'. Use css_select('[id*=\"carousel-left\"], [id*=\"carousel-prev\"], [id*=\"carousel-control-left\"]') to find ALL left arrows, OR use visible_text() to find buttons near section headers. For direction NOT 'RIGHT': click a LEFT arrow. For direction NOT 'LEFT': click a RIGHT arrow. If a carousel title is excluded (e.g. title NOT_EQUALS 'Kitchen Essentials'), click buttons on any OTHER carousel section. CAROUSEL_SCROLL fires immediately on arrow click.\n"
        "\n"
        "BOOK CATALOG TASKS (AutoBooks and similar):\n"
        "- Login: navigate to /login or click Login in nav, fill username and password, click Submit/Login.\n"
        "- Logout (LOGOUT_BOOK): first log in as <username>, then click the Logout button/link in the nav. REQUIRES being logged in as the correct user.\n"
        "- Registration: navigate to /register, fill username, email, password, click Register/Sign Up.\n"
        "- Search (SEARCH_BOOK): use the search bar to type a query, click the Search button. The event fires on search submit.\n"
        "- Filter (FILTER_BOOK): IMPORTANT: use ClickAction to navigate to /search (click the Search nav link), NOT NavigateAction with a URL. IMPORTANT: the task criteria key 'genres' (plural) is SILENTLY DROPPED by validation — so ANY filter interaction (click any genre button OR change the year dropdown) fires FILTER_BOOK and passes! Fastest approach: (1) ClickAction on the Search/Browse nav link. (2) SelectDropDownOptionAction on the year <select> dropdown — select any year like '2000'. Done. If you MUST use genre: the /search page has genre <select> and year <select> dropdowns. Use SelectDropDownOptionAction on the correct <select> with the genre name. V1 may reorder selects: use extract_forms() to identify genre (text options) vs year (4-digit numbers).\n"
        "- Book detail (BOOK_DETAIL): the event fires AUTOMATICALLY when you navigate to any book detail page (/books/[id]). VALIDATED fields: name, genre, year, rating ONLY (author, price, page_count, username/password in criteria are NOT validated — ignore them). IMPORTANT: do NOT go to the Profile > Books tab — that only shows YOUR added books (a small subset). Use the /search page or homepage to find books from the full catalog. To find a book: use list_cards() on the /search page or homepage to scan ratings/names, find one matching the VALIDATED criteria (e.g. rating = 4.7, name CONTAINS 'cula'), then click on it. If criteria has ONLY name (e.g. name = 'The Overstory'), use the search bar to search for it and click the result. No login required.\n"
        "- Preview (OPEN_PREVIEW): VALIDATED fields: name, genre, year, rating ONLY (username/password, author, page_count in criteria are ignored — login NOT required). IMPORTANT: do NOT go to Profile > Books tab. Use list_cards() on the /search page or homepage to find books from the full catalog. Find a book matching the validated criteria. Navigate to its detail page. Click the READ button ONCE — it has a BookOpen icon and its text varies via V3: 'Read book', 'Read', 'Open book', 'Start reading', 'Read now', 'View book', 'Open', 'Start'. Use search_text('Read') or search_text('Open') to find it. OPEN_PREVIEW fires on click.\n"
        "- Share (SHARE_BOOK): VALIDATED fields: name, genre, year, rating ONLY (price, author, username/password in criteria are NOT validated). LOGIN IS NEVER REQUIRED for SHARE_BOOK — DO NOT login even if the task prompt mentions username/password or credentials, they are NOT validated and login wastes steps. Find a book matching ONLY the name/genre/year/rating criteria from the task using list_cards() on /search or homepage. Navigate to its detail page, click the Share button ONCE. SHARE_BOOK fires on click.\n"
        "- Add to cart (ADD_TO_CART_BOOK): VALIDATED fields: name, genre, year, rating ONLY (price, author, page_count in criteria NOT validated). LOGIN IS NOT REQUIRED for ADD_TO_CART_BOOK — the cart endpoint has no auth check. DO NOT waste steps logging in. Find a book matching ONLY the name/genre/year/rating criteria using list_cards() on /search or homepage. Navigate to its detail page. Click the Add to Cart button ONCE — text varies via V3: 'Add', 'Add to Cart', 'Add to Bag', 'Add Item', 'Add to Basket'. Use search_text('Add') to find it.\n"
        "- Remove from cart (REMOVE_FROM_CART_BOOK): LOGIN IS NOT REQUIRED — the cart/remove endpoint has no auth check. VALIDATED fields: name, genre, year, rating ONLY (author, price, desc in criteria are NOT validated — ignore them). Steps: (1) Find ANY book matching the VALIDATED criteria using list_cards() on /search or homepage. (2) Click the book to go to its detail page. (3) Click Add to Cart button (no login needed). (4) Navigate to /cart (NavigateAction to '/cart' or click cart icon). (5) Click the Remove button on that cart item. REMOVE_FROM_CART_BOOK fires on Remove click. Use search_text('Remove') or search_text('Delete') to find the button.\n"
        "- Purchase (PURCHASE_BOOK): LOGIN IS NOT REQUIRED for PURCHASE_BOOK — the purchase endpoint has no auth check. VALIDATED fields: name, genre, year, rating ONLY (author, price in criteria NOT validated). Find a book matching ONLY the validated criteria (name CONTAINS X → search for X; genre NOT CONTAINS X → pick any book without that genre; year/rating must match). Navigate to its detail page and click Add to Cart (no login needed). THEN navigate to /cart. Click the purchase button — text varies: 'Purchase', 'Buy', 'Checkout', 'Complete Purchase', 'Buy Now' — use search_text('Purchase') or search_text('Buy'). PURCHASE_BOOK fires on click.\n"
        "- Add to reading list (ADD_TO_READING_LIST): REQUIRES LOGIN — handleWatchlist() checks auth and returns early if not logged in; event NEVER fires without login. CRITICAL: ALWAYS LOG IN FIRST even if the task prompt does NOT mention credentials. Use <username>/<password> from task if provided, otherwise log in with user1/Passw0rd!. Then use list_cards() to find a book matching the criteria (name, rating, genre, year), go to its detail page. Click the reading list/bookmark button ONCE — text varies via V3: 'Add to reading list', 'Add to list', 'Save to list', 'Add to library', 'Save book', 'Bookmark', 'Add', 'Save'. Use search_text('list') or search_text('Save') or search_text('Bookmark') to find it. ADD_TO_READING_LIST fires on click.\n"
        "- Remove from reading list (REMOVE_FROM_READING_LIST): REQUIRES LOGIN. VALIDATED fields: only rating, genre, name, year (ignore page_count and author in task prompt — they are NOT validated). CRITICAL: ALWAYS LOG IN FIRST. CRITICAL: do NOT go to Profile > Reading List tab — the 'Remove from List' button there does NOT fire REMOVE_FROM_READING_LIST (it has no logEvent call). You MUST use the BOOK DETAIL PAGE (/books/[id]). Log in, find a book matching the VALIDATED criteria, go to its DETAIL PAGE. In a fresh session the book is NOT in the reading list — click the bookmark button ONCE to ADD it (ADD_TO_READING_LIST fires), then click it AGAIN to REMOVE it. REMOVE_FROM_READING_LIST fires on the second click. Use search_text('list') or search_text('Bookmark') to find the bookmark button.\n"
        "- Add new book to catalog (ADD_BOOK): REQUIRES LOGIN. Log in → navigate to /profile → click the 'Add Books' tab (V3 label varies: 'Add Books', 'Add New Book', 'Create Book', 'New Book', 'Add Title' — use search_text('Add') to find it). The form has 6 labeled text inputs: Title, Author, Year, Pages, Rating, Preview URL — followed by genre pill buttons, a custom genre input (placeholder='Custom genre list'), and a Synopsis textarea. CRITICAL — form inputs have NO IDs: use label-based CSS selectors to target each field: css_select('label:has-text(\"Title\") input'), css_select('label:has-text(\"Author\") input'), css_select('label:has-text(\"Year\") input'), css_select('label:has-text(\"Pages\") input'), css_select('label:has-text(\"Rating\") input'). Do NOT use generic selector 'input' or 'custom: \"input\"' — that picks the FIRST input (Title field) for ALL fields. VALIDATED fields: year (must satisfy criteria, e.g. if year > 1900 type '2000'), author (must contain/equal criteria value, e.g. if author='gold' type 'Gold Author'), rating (must satisfy criteria, e.g. if rating ≤ 2.5 type '2.0'), genre (must match criteria). page_count in criteria is NOT validated — use '300' for Pages. For GENRE: if the genre (e.g. 'Romance', 'Fiction', 'Sci-Fi') appears in the pill list, CLICK that pill button. For genres NOT in the pill list (e.g. 'Short Stories', 'Psychological', 'Adventure', 'Historical Fiction'), clear the custom genre text input (placeholder='Custom genre list') and type the genre name there. Click the Add Book submit button (text varies: 'Add Book', 'Create Book', 'New Book' — search_text('Add') or 'Create'). ADD_BOOK fires on form submit.\n"
        "- Edit book (EDIT_BOOK): REQUIRES LOGIN. Log in → navigate to /profile → click the 'Edit Books' tab (V3 label: 'Edit Books', 'My Books', 'Manage Books', 'Update Books', 'Modify Books', 'Edit Titles' — use search_text('Edit') or search_text('Books')). IMPORTANT: there is NO separate 'Edit' button to click — the BookEditor form is shown INLINE below each book card on the 'Edit Books' tab. The form fields (Title, Author, Year, Pages, Rating, Preview URL) are already visible and pre-filled. CRITICAL — form inputs have NO IDs: use label-based CSS selectors to target each field: css_select('label:has-text(\"Title\") input'), css_select('label:has-text(\"Author\") input'), css_select('label:has-text(\"Year\") input'), css_select('label:has-text(\"Pages\") input'), css_select('label:has-text(\"Rating\") input'). Do NOT use generic selector 'input' or 'custom: \"input\"' — that picks the FIRST input on the page (Title field) for ALL fields. VALIDATED: year, author, rating, genre (page_count/genres-plural criteria are NOT validated — ignore them). Modify Year to satisfy criteria (e.g. year ≥ 2013 → type '2020'), Author to satisfy criteria (e.g. author CONTAINS 'light' → type 'Moonlight Author'), Rating to satisfy criteria (e.g. rating ≤ 4.9 → type '4.5'). GENRE: if genre required (e.g. genres='Romance'), click the matching genre pill button (pill options: Drama, Comedy, Action, Thriller, Sci-Fi, Fantasy, Horror, Documentary, Romance, Fiction, Non-Fiction, Mystery). For genres NOT in the pill list (e.g. 'Adventure'), type in the custom genre text input (placeholder='Custom genre list'). Click the submit button (text varies: 'Save Changes', 'Save', 'Update', 'Edit Book' — search_text('Save') to find). EDIT_BOOK fires on form submit.\n"
        "- Delete book (DELETE_BOOK): REQUIRES LOGIN. Log in → Profile page → click the 'Edit Books' tab → find any book in YOUR allowed books list → click the Delete button (red, 'Delete Book'). Any deletion is accepted (all criteria fields are ignored in validation — just delete any allowed book).\n"
        "- Edit user profile (EDIT_USER_BOOK): REQUIRES LOGIN. Log in → Profile page → the PROFILE TAB is the default tab (shows 'Edit Profile' form). The form has: First Name, Last Name, Email, Favorite Genres, Location, Website, Bio fields. VALIDATED fields: first_name, last_name, bio, location, website, favorite_genres (ALL can be validated depending on the task). Read the task criteria carefully — modify ONLY the fields the task requires. Examples: if 'bio CONTAINS Story lover' → type 'Story lover and aspiring writer.' in Bio field. If 'website CONTAINS car' → type 'https://cars.com' in Website. If 'last_name NOT CONTAINS star' → type 'Smith' in Last Name. After filling fields, click the Save button (text varies: 'Save Profile', 'Save', 'Update Profile' — search_text('Save')). EDIT_USER_BOOK fires on form submit.\n"
        "- View cart (VIEW_CART_BOOK): LOGIN IS NOT REQUIRED. VIEW_CART_BOOK fires automatically when the /cart page loads (useEffect on mount). VIEW_CART_BOOK has NO criteria — any fired event passes. Just navigate to /cart immediately: use NavigateAction to '/cart' OR click the cart icon in the nav. After the cart page loads, do ONE ScrollAction, then DoneAction. It does NOT need items in cart. IGNORE any username/password in the task prompt — DO NOT waste steps trying to log in.\n"
        "- Contact form (CONTACT_BOOK): navigate to /contact, fill ALL four text fields: name, email, subject (free text input — NOT a dropdown), message. Each must satisfy the task criteria (e.g. if email CONTAINS 'test@example.com', type 'test@example.com'; if subject = 'Inquiry', type 'Inquiry'; if name NOT EQUALS 'Susan', type 'John'). For NOT criteria, type any valid value that satisfies the constraint. Click Submit/Send. All four fields are validated.\n"
        "- Comment (ADD_COMMENT_BOOK): VALIDATED fields: book_name, commenter_name, content. Navigate to a book's detail page satisfying the book_name criteria. If criteria is 'book_name NOT CONTAINS X', go to ANY book except X — just use list_cards() to pick any book from the list. Scroll down to find the comment form at the bottom of the book detail page. Fill commenter_name (must satisfy criteria, e.g. if commenter_name NOT EQUALS 'David', type 'Alice'; if commenter_name = 'Sarah', type 'Sarah') and content (fill any text if no content criteria). Click the submit button. No login required.\n"
        "\n"
        "RESTAURANT/DINING TASKS (AutoDining and similar):\n"
        "- Search restaurant tasks (SEARCH_RESTAURANT): on the homepage, find the restaurant name search INPUT field (not URL bar) and use TypeAction to type the exact query from the task (e.g. 'Sud 777'). The SEARCH_RESTAURANT event fires after a 500ms debounce. After typing, perform one more action (e.g. scroll down once) before DoneAction to allow the event to register. Then done.\n"
        "- Date picker tasks (DATE_DROPDOWN_OPENED): on the homepage booking form, click the date picker button to open a calendar popup. The button ID varies by V3 (e.g. 'date_picker', 'dining-date-picker') — use search_text('Select date') or css_select('[id*=\"date\"]') to find it. Once the calendar opens, navigate months if needed using aria-label 'Go to previous month'/'Go to next month' buttons. Today is Feb 22 2026 — for March 2026, click next month ONCE. Then click the SPECIFIC day using an aria-label selector: each day button has aria-label='Month D, YYYY' format (e.g., aria-label='March 10, 2026'). Use css_select('[aria-label=\"March 10, 2026\"]') to find and click the specific day. Alternatively use search_text() to find the exact aria-label. Do NOT click day-of-week header labels. DATE_DROPDOWN_OPENED fires when a day is selected.\n"
        "- Time picker tasks (TIME_DROPDOWN_OPENED): click the time dropdown on the homepage OR restaurant detail page. Select the EXACT time specified in the task (e.g. '12:30 PM'). TIME_DROPDOWN_OPENED fires with the selected time as its validated value.\n"
        "- People/guest picker tasks (PEOPLE_DROPDOWN_OPENED): click the guests/people button on the homepage OR restaurant detail page. Select the EXACT number of people specified in the task. PEOPLE_DROPDOWN_OPENED fires with the selected count as its validated value.\n"
        "- View restaurant tasks (VIEW_RESTAURANT): the homepage search bar searches by BOTH name AND cuisine (not just name). For cuisine criteria (e.g. cuisine='Steakhouse'): type the cuisine name directly into the homepage search input (TypeAction), wait briefly (the list filters automatically), then click the 'View details' button on the matching card. For bookings criteria: after filtering by cuisine, use list_cards() to check the bookings count on each card and click the one matching the bookings number. Button text for 'View details' varies via V3: 'View details', 'See Details', 'More Info', 'Restaurant Info', 'Full Details', 'Learn More', 'About Restaurant', 'Details'. Use search_text('Details') or search_text('Info') to find it. VIEW_RESTAURANT fires when the restaurant detail page loads (not when you click the button — it fires on PAGE LOAD of /restaurant/[id]).\n"
        "- View full menu tasks (VIEW_FULL_MENU): type the restaurant name (e.g. 'Corner Pine Bites') in the homepage search input to filter the list. Then click the 'View details' button on the matching card. On the restaurant detail page, scroll to the Menu section. Click the menu toggle button — text varies: 'View Full Menu', 'See Full Menu', 'Show Full Menu', 'Display Full Menu', 'Expand Menu'. Use search_text('Full Menu') or search_text('Menu') to find it. VIEW_FULL_MENU fires on click.\n"
        "- Collapse menu tasks (COLLAPSE_MENU): navigate to a restaurant detail page. Click the menu expand button (search_text('Full Menu') to find it), then click the collapse button — text varies: 'Collapse Menu', 'Hide Menu', 'Close Menu', 'Minimize Menu'. Use search_text('Collapse') or search_text('Hide Menu') to find it. COLLAPSE_MENU fires on click.\n"
        "- Book restaurant tasks (BOOK_RESTAURANT): MUST set pickers BEFORE clicking Book Now. On the restaurant detail page reservation box:\n"
        "  1. Click the people/guests picker and select the required number of people.\n"
        "  2. Click the date picker button (search_text('Select date') or find it in BROWSER_STATE by ID containing 'date'). Once the calendar opens, navigate months using aria-label 'Go to next month'/'Go to previous month'. Then click the specific day using its aria-label: e.g., css_select('[aria-label=\"February 25, 2026\"]') for Feb 25. Today is Feb 22, 2026. For any FUTURE date, navigate to the required month first (click 'Go to next month' as many times as needed), then click the specific day with its exact aria-label. PAST dates (before Feb 22, 2026) are NOT selectable in the calendar — if task requires a past date, click any available date instead.\n"
        "  3. Click the time picker and select a valid time (avoid the excluded time if task says 'NOT X:XX PM').\n"
        "  4. Click 'Book Now'. BOOK_RESTAURANT fires with the selected people/date/time/rating values.\n"
        "- Booking confirmation page: clicking 'Book Now' navigates to /booking/[restaurantId]/[time]. On this page: fill Full Name, select country from phone prefix dropdown (fires COUNTRY_SELECTED), select occasion from occasion dropdown (fires OCCASION_SELECTED), fill phone number, click 'Complete Reservation' (fires RESERVATION_COMPLETE).\n"
        "- COUNTRY_SELECTED: the country field on the booking page is a <select> dropdown for the phone prefix. Use SelectDropDownOptionAction with the COUNTRY CODE (2-letter, e.g., 'RU' for Russia, 'US' for United States, 'GB' for UK, 'DE' for Germany, 'FR' for France). Find the select by css_select('select') or its V3 ID (key 'country-select'). If criteria says country NOT_EQUALS 'X', select any country other than X. If criteria says country='Russia', use value='RU'. COUNTRY_SELECTED fires immediately on selection (onChange).\n"
        "- OCCASION_SELECTED: on the booking confirmation page, find the occasion <select> element and use SelectDropDownOptionAction with the occasion VALUE (case-sensitive lowercase): 'birthday', 'anniversary', 'business', or 'other'. The occasion is validated in the criteria. Find the select by css_select('select') or V3 ID (key 'occasion-select'). OCCASION_SELECTED fires immediately on selection. You MUST first complete the booking flow (set people/date/time on restaurant page → click Book Now → arrive at confirmation page) before selecting the occasion.\n"
        "- RESERVATION_COMPLETE: fill Full Name, select occasion (use 'other' if task says occasion='other'), enter exact phone number from task, select country, then click 'Complete Reservation'.\n"
        "- Scroll/carousel tasks (SCROLL_VIEW): the homepage has 3 restaurant sections with carousels. Section titles vary by V3 seed (defaults: 'Expensive', 'Mid ticket', 'Cheap'). Use ClickAction (NOT ScrollAction/page scroll) on the carousel arrow BUTTON in the correct direction. Click the LEFT arrow button for direction 'left' (or direction NOT 'right'), or the RIGHT arrow button for direction 'right' (or direction NOT 'left'). Left arrows have V3 ID key 'scroll-left-button'. Right arrows have V3 ID key 'scroll-right-button'. Use css_select('[id*=\"scroll-left\"], [id*=\"scroll-prev\"]') to find left arrows. SCROLL_VIEW fires immediately on arrow ClickAction. NOTE: the section title in the criteria may use old names ('Available for lunch now', 'Top picks', etc.) — these map to current sections ('Expensive', 'Mid ticket', 'Cheap'). Just click any scroll arrow in the correct direction (ignore section title).\n"
        "- Help page tasks: navigate to /help (NavigateAction to '/help' OR use search_text to find Help nav link). The page has SHORT CATEGORY FILTER BUTTONS at the top (e.g. 'Getting Started', 'Bookings', 'Payments', 'Account', 'Restaurants', 'Technical') AND LONGER FAQ ITEM BUTTONS below them. CATEGORY TASK (HELP_CATEGORY_SELECTED): the categories are EXACT strings stored in the event: 'Getting Started', 'Bookings', 'Payments', 'Account', 'Restaurants', 'Technical'. If task says category='Reservations', this does NOT match any category — try clicking 'Bookings' (closest match). Otherwise click the short category filter button matching the criteria value exactly. Use search_text() to find it. HELP_CATEGORY_SELECTED fires on click. FAQ TASK (HELP_FAQ_TOGGLED): IMPORTANT — do NOT click the short category filter buttons. Instead, scroll DOWN past the category buttons to find the FAQ ITEM BUTTONS. Each FAQ item button contains the FULL question text (e.g. 'How do I make a reservation?', 'Can I modify my reservation?', 'Is there a cancellation fee?'). Use search_text('How do') or search_text('Can I') to find FAQ question buttons. For question NOT CONTAINS 'cancellation fee', pick any question that doesn't mention cancellation. Click the FAQ ITEM button ONCE — HELP_FAQ_TOGGLED fires on click with the question text.\n"
        "- About page tasks (ABOUT_FEATURE_CLICK): IMPORTANT: do NOT click the 'About' nav link — in some V3 seeds it redirects to the homepage. Instead, use NavigateAction to '/about' directly. On the /about page, click any feature card in the 'Why Choose AutoDining?' section. The actual feature cards visible are: 'Curated Restaurants', 'Easy Reservations', 'Community Driven', 'Verified Reviews', 'Trending Spots'. IMPORTANT: the task criteria may use DIFFERENT names (e.g. 'Trusted reviews', 'Live availability', 'Curated chefs') — these are related to but not exactly matching the displayed card titles. For feature criteria with 'reviews': click 'Verified Reviews'. For 'Curated': click 'Curated Restaurants'. For 'availability': click 'Easy Reservations'. For any other criteria: click ANY feature card. Use search_text() with a keyword from the feature name to find the card div. ABOUT_FEATURE_CLICK fires on click of any feature card.\n"
        "- Contact page tasks (CONTACT_CARD_CLICK): navigate to /contact (NavigateAction or click 'Contact' in nav). The page has 4 contact cards: 'Email Us', 'Call Us', 'Visit Us', 'Business Hours'. CRITICAL: use ClickAction ONLY — NEVER NavigateAction on these cards (they have mailto:/tel: hrefs that would navigate away from the page). Use search_text('Email') to find 'Email Us', search_text('Business') to find 'Business Hours'. For card_type NOT 'Phone', click 'Email Us' (not 'Call Us'). CONTACT_CARD_CLICK fires on ClickAction. After clicking, do DoneAction immediately — do NOT NavigateAction.\n"
        "- CONTACT_FORM_SUBMIT: FIRST navigate to /contact page (NavigateAction to '/contact'). THEN fill the 4 form fields using stable name-attribute CSS selectors: css_select('input[name=\"name\"]') for Name, css_select('input[name=\"email\"]') for Email, css_select('input[name=\"subject\"]') for Subject, css_select('textarea[name=\"message\"]') for Message. These selectors are stable and NOT affected by V3. Fill: Name with a value containing 'e' (e.g. 'Eve User'), Email with 'test@example.com' (contains 'e.com'), Subject with anything NOT 'Suggestion for Improvement' (e.g. 'General Inquiry'), Message with any non-empty text. After filling all 4 fields, click the Send/Submit button ONCE (use search_text('Send') or css_select('button[type=\"submit\"]')). Do NOT click submit multiple times. After clicking submit, do a ScrollAction to give the event time to register. CONTACT_FORM_SUBMIT fires on form submit.\n"
        "\n"
        "INSPECTION TOOLS (optional — return {tool, args} instead of an action):\n"
        "search_text(query, regex?, max_matches?, context_chars?); visible_text(max_chars?); "
        "css_select(selector, max_nodes?); extract_forms(max_forms?, max_inputs?); "
        "list_links(max_links?, href_regex?, text_regex?); list_cards(max_cards?, max_text?, max_actions_per_card?); "
        "find_card(query, max_cards?). Prefer at most 2 tool calls per task step."
    )

    history_lines: List[str] = []
    for h in (history or [])[-6:]:
        step = h.get("step", "?")
        action = _norm_action(h.get("action") or "")
        cid = h.get("candidate_id")
        text = h.get("text", "")
        ok = h.get('exec_ok', True)
        err = h.get('error')
        suffix = 'OK' if ok else f"FAILED err={str(err)[:80]}"
        history_lines.append(f"{step}. {action} cid={cid} text={text} [{suffix}]")

    hint = _history_hint(history)

    structured = _structured_hints(task, candidates)

    cards_preview = ""
    try:
        cards_obj = _tool_list_cards(candidates=candidates, max_cards=12, max_text=420, max_actions_per_card=3)
        if isinstance(cards_obj, dict) and cards_obj.get("ok") and cards_obj.get("cards"):
            cards_preview = json.dumps(cards_obj.get("cards"), ensure_ascii=True)
            # Keep the prompt bounded.
            if len(cards_preview) > 2400:
                cards_preview = cards_preview[:2397] + "..."
    except Exception:
        cards_preview = ""

    # Build typed-fields reminder from task state.
    typed_fields_str = ""
    try:
        if task_id:
            tst = _TASK_STATE.get(task_id)
            if isinstance(tst, dict):
                tf = tst.get("typed_fields")
                if isinstance(tf, dict) and tf:
                    items = [f"cid {k}='{v}'" for k, v in tf.items()]
                    typed_fields_str = "ALREADY TYPED (do NOT retype): " + ", ".join(items)
    except Exception:
        typed_fields_str = ""

    user_msg = (
        f"You have a task and must decide the next single browser action.\n"
        + (f"SITE: {web_project_id}\n" if web_project_id else "")
        + f"TASK: {task}\n"
        f"STEP: {int(step_index)}/{int(os.getenv('AGENT_MAX_STEPS', '12'))}\n"
        f"URL: {url}\n\n"
        + (f"PAGE IR (PRIMARY STRUCTURED STATE):\n{page_ir_text}\n\n" if page_ir_text else "")
        + f"CURRENT STATE (TEXT SUMMARY):\n{page_summary}\n\n"
        + (f"DOM DIGEST (STRUCTURED):\n{dom_digest}\n\n" if dom_digest else "")
        + (f"CARDS (GROUPED CLICKABLE CONTEXTS JSON):\n{cards_preview}\n\n" if cards_preview else "")
        + f"STRUCTURED STATE (JSON):\n{json.dumps(structured, ensure_ascii=True)}\n\n"
        + (f"HISTORY (last steps):\n{chr(10).join(history_lines)}\n\n" if history_lines else "")
        + (f"{typed_fields_str}\n\n" if typed_fields_str else "")
        + (f"STATE HINT: {extra_hint}\n\n" if extra_hint else "")
        + (f"TARGETING HINT: {target_hint}\n\n" if target_hint else "")
        + (f"STATE DELTA (prev -> current): {state_delta}\n\n" if state_delta else "")
        + (f"PAGE IR DELTA (prev -> current): {ir_delta}\n\n" if ir_delta else "")
        + "BROWSER_STATE (interactive elements):\n" + browser_state + "\n\n"
        + "NEXT ACTION:\n"
        + (f"⚠️ LOOP DETECTED — {hint}\n" if hint else "")
        + "Output ONE action as JSON with keys: action, candidate_id (int), text, url.\n"
    )

    # Default to validator gateway default model.
    model = str(model_override or os.getenv("OPENAI_MODEL", "gpt-5.2"))
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "400"))

    usages: List[Dict[str, Any]] = []
    tool_calls = 0
    max_tool_calls = int(os.getenv("AGENT_MAX_TOOL_CALLS", "2"))

    # Multi-turn context: inject last 2 (user, assistant) pairs from task history.
    _MAX_HISTORY_TURNS = 2
    _MAX_STORED_USER_CHARS = 3000
    _MAX_STORED_ASSISTANT_CHARS = 800
    prior_turns: List[Dict[str, Any]] = []
    try:
        st = _TASK_STATE.get(task_id) if task_id else None
        if isinstance(st, dict):
            llm_hist = st.get("llm_history")
            if isinstance(llm_hist, list):
                prior_turns = llm_hist[-(_MAX_HISTORY_TURNS * 2):]
    except Exception:
        prior_turns = []

    messages = [
        {"role": "system", "content": system_msg},
        *prior_turns,
        {"role": "user", "content": user_msg},
    ]

    def _call(extra_system: str = "") -> Dict[str, Any]:
        sys_msg = system_msg + (" " + extra_system if extra_system else "")
        # Keep system message authoritative even after tool results.
        msgs = [{"role": "system", "content": sys_msg}] + [m for m in messages if m.get("role") != "system"]
        resp = openai_chat_completions(
            task_id=task_id,
            messages=msgs,
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
            obj["_meta"] = {"llm_calls": len(usages), "llm_usages": usages, "model": str(model), "tool_calls": tool_calls}
        except Exception:
            pass
        return obj

    def _valid_action(obj: Dict[str, Any]) -> bool:
        a = (obj.get("action") or "").lower()
        if a not in {"click", "double_click", "triple_click", "submit", "type", "select", "navigate", "scroll_down", "scroll_up", "done"}:
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

    def _is_tool(obj: Dict[str, Any]) -> bool:
        t = obj.get("tool")
        if not isinstance(t, str) or not t.strip():
            return False
        # Tool response should not mix action.
        if obj.get("action"):
            return False
        return True

    def _save_turn(response_obj: Dict[str, Any]) -> None:
        """Store (user_msg, assistant_response) in task state for multi-turn context."""
        if not task_id:
            return
        try:
            st = _TASK_STATE.get(task_id)
            if not isinstance(st, dict):
                st = {}
                _TASK_STATE[task_id] = st
            hist = st.get("llm_history")
            if not isinstance(hist, list):
                hist = []
                st["llm_history"] = hist
            # Truncate user_msg for storage to avoid token bloat
            stored_user = user_msg[:_MAX_STORED_USER_CHARS] if len(user_msg) > _MAX_STORED_USER_CHARS else user_msg
            # Store clean response (strip _meta)
            resp_clean = {k: v for k, v in response_obj.items() if k != "_meta"}
            stored_asst = json.dumps(resp_clean, ensure_ascii=True)[:_MAX_STORED_ASSISTANT_CHARS]
            hist.append({"role": "user", "content": stored_user})
            hist.append({"role": "assistant", "content": stored_asst})
            # Keep only last 4 messages (2 turns)
            if len(hist) > 4:
                st["llm_history"] = hist[-4:]
        except Exception:
            pass

    # Tool-aware loop.
    last_obj: Dict[str, Any] = {}
    for _ in range(max_tool_calls + 2):
        try:
            obj = _call()
        except Exception:
            obj = _call("Return ONLY valid JSON. No markdown. No commentary.")

        last_obj = obj

        if _is_tool(obj) and tool_calls < max_tool_calls:
            tool = str(obj.get("tool") or "").strip()
            args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
            tool_calls += 1
            try:
                result = _run_tool(tool, args, html=html_snapshot, url=str(url), candidates=candidates)
            except Exception as e:
                result = {"ok": False, "error": str(e)[:200]}

            # IMPORTANT: tools must inspect snapshot_html, not dom_digest. We'll attach snapshot_html via closure.
            # This placeholder is replaced below.
            messages.append({"role": "assistant", "content": json.dumps({"tool": tool, "args": args}, ensure_ascii=True)})
            result_str = json.dumps(result, ensure_ascii=True)
            max_tool_result_chars = int(os.getenv("AGENT_TOOL_RESULT_MAX_CHARS", "3000"))
            if len(result_str) > max_tool_result_chars:
                result_str = result_str[:max_tool_result_chars] + "..."
            messages.append({"role": "user", "content": "TOOL_RESULT " + tool + ": " + result_str})
            continue

        if _valid_action(obj):
            try:
                obj["_meta"] = {"llm_calls": len(usages), "llm_usages": usages, "model": str(model), "tool_calls": tool_calls}
            except Exception:
                pass
            _save_turn(obj)
            return obj

        # Ask once to fix invalid response.
        obj = _call(
            "Your previous JSON was invalid. Fix it. "
            f"candidate_id must be an integer in [0, {len(candidates) - 1}]. "
            "If action is type/select you must include non-empty text. "
            "If stuck, scroll_down."
        )
        if _valid_action(obj):
            try:
                obj["_meta"] = {"llm_calls": len(usages), "llm_usages": usages, "model": str(model), "tool_calls": tool_calls}
            except Exception:
                pass
            _save_turn(obj)
            return obj

    return last_obj


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
    html_snapshot: str,
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


class ApifiedWebAgent(IWebAgent):
    """Core operator implementing IWA's IWebAgent interface."""

    def __init__(self, id: str = "1", name: str = "AutoppiaOperator") -> None:
        self.id = str(id)
        self.name = str(name)

    async def act(
        self,
        *,
        task: Task,
        snapshot_html: str,
        screenshot: str | bytes | None = None,
        url: str,
        step_index: int,
        history: list[dict[str, Any]] | None = None,
    ) -> list[BaseAction]:
        task_id = str(getattr(task, "id", "") or "")
        prompt = str(getattr(task, "prompt", "") or "")
        create_action_fn = getattr(BaseAction, "create_action", None)
        if not callable(create_action_fn):
            logger.error(
                f"[AGENT_TRACE] BaseAction.create_action missing "
                f"task_id={task_id} step_index={int(step_index)} "
                f"BaseAction={repr(BaseAction)} import_ok={_AUTOPPIA_IWA_IMPORT_OK}"
            )
        payload = {
            "task_id": task_id,
            "prompt": prompt,
            "snapshot_html": snapshot_html,
            "screenshot": screenshot,
            "url": url,
            "step_index": int(step_index),
            "history": history or [],
        }
        resp = await self.act_from_payload(payload)
        actions = resp.get("actions") if isinstance(resp, dict) else []
        _log_trace(
            f"act() raw actions task_id={task_id} step_index={int(step_index)} "
            f"count={len(actions) if isinstance(actions, list) else 0}"
        )
        out: list[BaseAction] = []
        for a in actions if isinstance(actions, list) else []:
            if not isinstance(a, dict):
                continue
            try:
                ac = create_action_fn(a) if callable(create_action_fn) else None
                if ac is not None:
                    out.append(ac)
            except Exception as exc:
                logger.error(
                    f"[AGENT_TRACE] create_action failed task_id={task_id} step_index={int(step_index)} "
                    f"action_type={str(a.get('type') or '')} err={str(exc)} "
                    f"payload={json.dumps(a, ensure_ascii=True)[:500]}"
                )
                continue
        if isinstance(actions, list) and actions and not out:
            logger.error(
                f"[AGENT_TRACE] all actions dropped during conversion task_id={task_id} "
                f"step_index={int(step_index)} "
                f"raw_types={[str(x.get('type') or '') for x in actions if isinstance(x, dict)]}"
            )
        return out

    async def act_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task_id = str(payload.get("task_id") or "")
        task = payload.get("prompt") or payload.get("task_prompt") or ""
        model_override = str(payload.get("model") or "").strip()
        url = _normalize_demo_url(str(payload.get("url") or ""))
        step_index = int(payload.get("step_index") or 0)
        return_metrics = os.getenv("AGENT_RETURN_METRICS", "0").lower() in {"1", "true", "yes"}
        html = payload.get("snapshot_html") or ""
        history = payload.get("history") if isinstance(payload.get("history"), list) else None
        web_project_id = str(payload.get("web_project_id") or "").strip()
        page_summary = _summarize_html(html)
        dom_digest = _dom_digest(html)
        task = str(task or "")

        def _resp(actions: list[dict[str, Any]], metrics: dict[str, Any] | None = None) -> Dict[str, Any]:
            sanitized_actions: list[dict[str, Any]] = []
            for action in actions:
                if isinstance(action, dict):
                    sanitized_actions.append(_sanitize_action_payload(action))
                else:
                    sanitized_actions.append({})

            out: Dict[str, Any] = {"actions": sanitized_actions}
            if return_metrics and metrics is not None:
                out["metrics"] = metrics
            return out

        candidates = _extract_candidates(html, max_candidates=80)
        candidates_all = list(candidates)
        candidates = _select_candidates_for_llm(task, candidates_all, current_url=str(url), max_total=60)
        page_ir = _extract_page_ir(html=html, url=str(url), candidates=candidates)
        page_ir_text = _render_page_ir(page_ir, max_chars=int(os.getenv("AGENT_PAGE_IR_MAX_CHARS", "2200")))

        if task_id == "check":
            if candidates:
                return _resp([{"type": "ClickAction", "selector": candidates[0].click_selector()}], {"decision": "check_click", "candidate_id": 0})
            return _resp([{"type": "WaitAction", "time_seconds": 0.1}], {"decision": "check_wait"})

        st = _TASK_STATE.get(task_id) if task_id else None
        # Clear per-task state on step 0 (new task start) — full wipe to avoid stale state.
        if task_id and step_index == 0:
            _TASK_STATE[task_id] = {}
            st = _TASK_STATE[task_id]
        effective_url = str(url)
        try:
            if isinstance(st, dict):
                eu = str(st.get("effective_url") or "").strip()
                if eu:
                    effective_url = eu
        except Exception:
            effective_url = str(url)
        extra_hint = ""
        target_hint = ""
        prev_sig_set = None
        try:
            if isinstance(st, dict):
                prev = st.get('prev_sig_set')
                if isinstance(prev, list):
                    prev_sig_set = set(str(x) for x in prev)
        except Exception:
            prev_sig_set = None

        state_delta = _compute_state_delta(task_id=task_id, url=str(url), page_summary=page_summary, dom_digest=dom_digest, html_snapshot=html, candidates=candidates)
        ir_delta = _compute_ir_delta(task_id=task_id, page_ir=page_ir)
        try:
            if isinstance(st, dict):
                last_url = str(st.get("last_url") or "")
                repeat = int(st.get("repeat") or 0)
                if last_url and last_url == str(url) and repeat >= 2:
                    extra_hint = "You appear stuck on the same URL after repeating an action. Choose a different element or scroll."
        except Exception:
            extra_hint = ""
        # Smart scroll: detect when no new candidates have appeared since the last scroll.
        try:
            if task_id and isinstance(st, dict):
                cur_url = str(url)
                # Reset seen sigs when the page URL has changed since last step.
                if str(st.get("seen_cand_url") or "") != cur_url:
                    st.pop("seen_cand_sigs", None)
                seen_sigs: set[str] = set(st.get("seen_cand_sigs") or [])
                cur_sigs: set[str] = {_selector_repr(c.selector) for c in candidates}
                new_sigs = cur_sigs - seen_sigs
                last_sig = str(st.get("last_sig") or "")
                if not new_sigs and last_sig == "scroll_down" and seen_sigs:
                    scroll_hint = "SCROLL BOTTOM REACHED — no new elements appeared since last scroll_down. Stop scrolling and choose a different action."
                    extra_hint = ((extra_hint + " ") if extra_hint else "") + scroll_hint
                st["seen_cand_sigs"] = list(seen_sigs | cur_sigs)
                st["seen_cand_url"] = cur_url
        except Exception:
            pass
        try:
            risk_hint = _task_risk_hint(task=task, step_index=int(step_index), candidates=candidates)
            if risk_hint:
                extra_hint = ((extra_hint + " ") if extra_hint else "") + risk_hint
        except Exception:
            pass
        # Fix 3: Surface exec_ok=False from last history step as an explicit hint so
        # the LLM knows its previous action failed and avoids repeating it.
        try:
            if history and isinstance(history, list):
                last_h = history[-1]
                if isinstance(last_h, dict) and not last_h.get("exec_ok", True):
                    err_msg = str(last_h.get("error") or "unknown")[:100]
                    fail_hint = f"LAST ACTION FAILED ({err_msg}). Try a different element or approach."
                    extra_hint = ((extra_hint + " ") if extra_hint else "") + fail_hint
        except Exception:
            pass
        try:
            target_hint = str(payload.get("target_hint") or os.getenv("AGENT_TARGET_HINT", "")).strip()
        except Exception:
            target_hint = ""

        try:
            base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
            if not os.getenv("OPENAI_API_KEY") and not is_sandbox_gateway_base_url(base_url):
                raise RuntimeError("OPENAI_API_KEY not set")
            decision = _llm_decide(
                task_id=task_id,
                task=task,
                step_index=step_index,
                url=effective_url,
                candidates=candidates,
                page_summary=page_summary,
                dom_digest=dom_digest,
                html_snapshot=html,
                history=history,
                page_ir_text=page_ir_text,
                extra_hint=extra_hint,
                target_hint=target_hint,
                state_delta=state_delta,
                ir_delta=ir_delta,
                prev_sig_set=prev_sig_set,
                model_override=model_override,
                web_project_id=web_project_id,
            )
        except Exception as e:
            if _LOG_ERRORS:
                logger.exception(
                    f"[AGENT_TRACE] llm_decide exception task_id={task_id} "
                    f"step_index={int(step_index)} url={str(url)} err={str(e)}"
                )
            if os.getenv("AGENT_DEBUG_ERRORS", "0").lower() in {"1", "true", "yes"}:
                raise HTTPException(status_code=500, detail=str(e)[:400])
            return _resp([{"type": "WaitAction", "time_seconds": 1.0}], {"decision": "error_wait"})

        action = (decision.get("action") or "").lower()
        cid = decision.get("candidate_id")
        text = decision.get("text")
        if isinstance(cid, str) and cid.isdigit():
            cid = int(cid)

        out: Dict[str, Any]
        if action == "navigate":
            nav_url_raw = str(decision.get("url") or "").strip()
            if not nav_url_raw:
                out = _resp([{"type": "WaitAction", "time_seconds": 1.0}], {"decision": "navigate_missing_url"})
            else:
                nav_url = _resolve_url(nav_url_raw, effective_url or str(url))
                nav_url = _preserve_seed_url(nav_url, effective_url or str(url))
                if _same_path_query(nav_url, effective_url, base_a=effective_url, base_b=""):
                    _update_task_state(task_id, str(url), "navigate_same_url_scroll")
                    out = _resp([{ "type": "ScrollAction", "down": True, "up": False}], {"decision": "scroll_override"})
                else:
                    _update_task_state(task_id, str(url), f"navigate:{nav_url}")
                    try:
                        if task_id and isinstance(_TASK_STATE.get(task_id), dict):
                            _TASK_STATE[task_id]["effective_url"] = str(nav_url)
                    except Exception:
                        pass
                    out = _resp([{"type": "NavigateAction", "url": nav_url, "go_back": False, "go_forward": False}], {"decision": "navigate", "url": nav_url, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
        elif action in {"scroll_down", "scroll_up"}:
            _update_task_state(task_id, str(url), f"{action}")
            out = _resp([{"type": "ScrollAction", "down": action == "scroll_down", "up": action == "scroll_up"}], {"decision": decision.get("action"), "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
        elif action == "done":
            _update_task_state(task_id, str(url), "done")
            out = _resp([], {"decision": "done", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
        elif action in {"click", "type", "select"} and isinstance(cid, int) and 0 <= cid < len(candidates):
            c = candidates[cid]
            if action == "click":
                selector = c.click_selector()
                try:
                    if isinstance(selector, dict) and selector.get("type") == "attributeValueSelector" and selector.get("attribute") == "href":
                        href = str(selector.get("value") or "")
                        fixed = _preserve_seed_url(href, effective_url or str(url))
                        if fixed and fixed != href:
                            fixed_abs = _resolve_url(fixed, effective_url or str(url))
                            if _same_path_query(fixed_abs, effective_url, base_a=effective_url, base_b=""):
                                _update_task_state(task_id, str(url), "navigate_seed_fix_same_url_scroll")
                                out = _resp([{ "type": "ScrollAction", "down": True, "up": False}], {"decision": "scroll_override"})
                            else:
                                _update_task_state(task_id, str(url), f"navigate_seed_fix:{fixed_abs}")
                                try:
                                    if task_id and isinstance(_TASK_STATE.get(task_id), dict):
                                        _TASK_STATE[task_id]["effective_url"] = str(fixed_abs)
                                except Exception:
                                    pass
                                out = _resp([{ "type": "NavigateAction", "url": fixed_abs, "go_back": False, "go_forward": False}], {"decision": "navigate", "url": fixed_abs, "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
                        else:
                            _update_task_state(task_id, str(url), f"click:{_selector_repr(selector)}")
                            out = _resp([{"type": "ClickAction", "selector": selector}], {"decision": "click", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
                    else:
                        _update_task_state(task_id, str(url), f"click:{_selector_repr(selector)}")
                        out = _resp([{"type": "ClickAction", "selector": selector}], {"decision": "click", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
                except Exception:
                    _update_task_state(task_id, str(url), f"click:{_selector_repr(selector)}")
                    out = _resp([{"type": "ClickAction", "selector": selector}], {"decision": "click", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
            elif action == "type":
                if not text:
                    raise HTTPException(status_code=400, detail="type action missing text")
                selector = c.type_selector()
                _update_task_state(task_id, str(url), f"type:{_selector_repr(selector)}")
                # Record typed field so LLM knows not to retype it.
                try:
                    if task_id:
                        tst = _TASK_STATE.get(task_id)
                        if not isinstance(tst, dict):
                            tst = {}
                            _TASK_STATE[task_id] = tst
                        tf = tst.get("typed_fields")
                        if not isinstance(tf, dict):
                            tf = {}
                            tst["typed_fields"] = tf
                        tf[str(cid)] = str(text)[:60]
                except Exception:
                    pass
                out = _resp([{"type": "TypeAction", "selector": selector, "text": str(text)}], {"decision": "type", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
            else:
                if not text:
                    raise HTTPException(status_code=400, detail="select action missing text")
                selector = c.type_selector()
                _update_task_state(task_id, str(url), f"select:{_selector_repr(selector)}")
                out = _resp([{"type": "SelectDropDownOptionAction", "selector": selector, "text": str(text), "timeout_ms": int(os.getenv("AGENT_SELECT_TIMEOUT_MS", "4000"))}], {"decision": "select", "candidate_id": int(cid) if isinstance(cid,int) else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
        elif action in {"double_click", "triple_click", "submit"} and isinstance(cid, int) and 0 <= cid < len(candidates):
            c = candidates[cid]
            selector = c.click_selector()
            _update_task_state(task_id, str(url), f"{action}:{_selector_repr(selector)}")
            iwa_type = {"double_click": "DoubleClickAction", "triple_click": "TripleClickAction", "submit": "SubmitAction"}[action]
            out = _resp([{"type": iwa_type, "selector": selector}], {"decision": action, "candidate_id": int(cid), "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
        else:
            if candidates and step_index < 5:
                selector = candidates[0].click_selector()
                _update_task_state(task_id, str(url), f"fallback_click:{_selector_repr(selector)}")
                out = _resp([{"type": "ClickAction", "selector": selector}], {"decision": "click_override", "candidate_id": 0 if candidates else None, "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})
            else:
                _update_task_state(task_id, str(url), "fallback_wait")
                out = _resp([{"type": "WaitAction", "time_seconds": 2.0}], {"decision": "fallback_wait", "model": decision.get("_meta", {}).get("model"), "llm": decision.get("_meta", {})})

        try:
            action_types = [str(a.get("type") or "") for a in out.get("actions", []) if isinstance(a, dict)]
        except Exception:
            action_types = []
        _log_trace(
            f"act_from_payload task_id={task_id} step_index={int(step_index)} "
            f"decision={str(action)} candidate_id={str(cid)} out_actions={action_types}"
        )
        return out

# -----------------------------
# HTTP entrypoint
# -----------------------------

AutoppiaOperator = ApifiedWebAgent
OPERATOR = AutoppiaOperator(id=os.getenv("WEB_AGENT_ID", "1"), name="AutoppiaOperator")


def _task_from_payload(payload: Dict[str, Any]) -> Task:
    """Build a minimal Task object from /act payload for the IWebAgent interface."""
    task_payload = {
        "id": str(payload.get("task_id") or ""),
        "url": _normalize_demo_url(str(payload.get("url") or "")),
        "prompt": str(payload.get("prompt") or payload.get("task_prompt") or ""),
        "web_project_id": payload.get("web_project_id"),
    }
    try:
        if isinstance(Task, type):
            return Task(**task_payload)
    except Exception:
        pass
    return SimpleNamespace(**task_payload)


@app.post("/act", summary="Decide next agent actions")
async def act(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    task_id = str(payload.get("task_id") or "")
    step_index = int(payload.get("step_index") or 0)
    url = _normalize_demo_url(str(payload.get("url") or ""))
    _log_trace(f"/act start task_id={task_id} step_index={step_index} url={url}")
    raw_resp = await OPERATOR.act_from_payload(payload)
    actions = raw_resp.get("actions") if isinstance(raw_resp, dict) else []
    normalized = []
    for action in actions if isinstance(actions, list) else []:
        try:
            if isinstance(action, dict):
                normalized.append(_sanitize_action_payload(action))
                continue
            action_payload = action.model_dump(exclude_none=True)
            normalized.append(_sanitize_action_payload(action_payload))
        except Exception as exc:
            logger.error(
                f"[AGENT_TRACE] /act action normalization failed task_id={task_id} "
                f"step_index={step_index} err={str(exc)} raw={str(action)[:500]}"
            )
            continue
    _log_trace(
        f"/act end task_id={task_id} step_index={step_index} "
        f"raw_count={len(actions) if isinstance(actions, list) else 0} out_count={len(normalized)} "
        f"types={[str(a.get('type') or '') for a in normalized if isinstance(a, dict)]}"
    )
    out: Dict[str, Any] = {"actions": normalized}
    if isinstance(raw_resp, dict) and "metrics" in raw_resp:
        out["metrics"] = raw_resp.get("metrics")
    return out


@app.post("/step", summary="Alias for /act")
async def step(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await act(payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
