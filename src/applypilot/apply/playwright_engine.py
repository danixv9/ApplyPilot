"""Playwright-based auto-apply engine (no Claude Code CLI required).

This engine is intentionally conservative: it only supports a small set of
common ATS application flows (Lever, Greenhouse, Ashby). Unsupported ATS URLs
are marked as manual so they don't get retried endlessly.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Pattern

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from applypilot import config
from applypilot.apply.chrome import BASE_CDP_PORT, cleanup_worker, launch_chrome, setup_worker_profile
from applypilot.apply.launcher import acquire_job, mark_result, release_lock
from applypilot.database import get_connection

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 60_000


@dataclass(frozen=True)
class ApplyOutcome:
    status: str
    error: str | None = None


def _first_last(full_name: str) -> tuple[str, str]:
    parts = (full_name or "").strip().split()
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[-1]


def _location(personal: dict) -> str:
    parts = [personal.get("city"), personal.get("province_state"), personal.get("country")]
    return ", ".join([p for p in parts if p])


def _read_cover_text(path: str | None, limit: int = 5000) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""

    # Prefer the .txt version if a PDF path is stored.
    if p.suffix.lower() != ".txt":
        txt = p.with_suffix(".txt")
        if txt.exists():
            p = txt
        else:
            return ""

    try:
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""
    return text[:limit]


def _resolve_resume_upload_path(job: dict) -> str | None:
    raw = (job.get("tailored_resume_path") or "").strip()
    if raw:
        p = Path(raw)
        if p.suffix.lower() == ".txt":
            pdf = p.with_suffix(".pdf")
            if pdf.exists():
                return str(pdf)
            # Avoid passing .txt files to ATS uploads; prefer base PDF fallback.
            if config.RESUME_PDF_PATH.exists():
                return str(config.RESUME_PDF_PATH)
        if p.exists() and p.suffix.lower() in (".pdf", ".doc", ".docx", ".rtf"):
            return str(p)

    if config.RESUME_PDF_PATH.exists():
        return str(config.RESUME_PDF_PATH)
    return None


def _resolve_cover_upload_path(job: dict) -> str | None:
    raw = (job.get("cover_letter_path") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.suffix.lower() == ".txt":
        pdf = p.with_suffix(".pdf")
        if pdf.exists():
            return str(pdf)
        return None
    if p.suffix.lower() == ".pdf" and p.exists():
        return str(p)
    # Try sibling PDF for any other stored extension.
    pdf = p.with_suffix(".pdf")
    if pdf.exists():
        return str(pdf)
    return None


def _looks_captcha(page) -> bool:
    try:
        def _token_filled(sel: str) -> bool:
            loc = page.locator(sel)
            if loc.count() == 0:
                return False
            try:
                v = loc.first.input_value(timeout=1000)
            except Exception:
                try:
                    v = loc.first.evaluate("el => el.value")
                except Exception:
                    return False
            return bool((v or "").strip())

        token_present = any(
            _token_filled(s)
            for s in (
                "textarea[name='g-recaptcha-response']",
                "textarea[name='h-captcha-response']",
                "input[name='cf-turnstile-response']",
            )
        )

        # hCaptcha / Turnstile are typically blocking when present without a filled token.
        if (
            page.locator("iframe[src*='hcaptcha'], .h-captcha, [data-hcaptcha-sitekey]").count() > 0
            or page.locator("[data-cf-turnstile], iframe[src*='challenges.cloudflare.com']").count() > 0
        ):
            return not token_present

        # reCAPTCHA is often "invisible" (v3 / invisible-v2) and will only
        # populate a response token during submission. Avoid treating it as a
        # blocking captcha unless a visible widget/challenge is shown.
        recaptcha_challenge = page.locator("iframe[src*='recaptcha/api2/bframe'], iframe[src*='recaptcha/api2/fallback']").count() > 0
        recaptcha_visible = page.locator(
            "iframe[src*='recaptcha/api2/anchor']:not([src*='size=invisible'])"
        ).count() > 0
        if recaptcha_challenge or recaptcha_visible:
            return not token_present

        # Explicit error banners about reCAPTCHA connectivity/challenges.
        if page.get_by_text(re.compile(r"could not connect to the re\\s*captcha service", re.I)).count() > 0:
            return True

        # Fallback: generic challenge pages/banners (avoid matching "reCAPTCHA").
        if page.get_by_text(re.compile(r"verify you are human|\\bcaptcha\\b", re.I)).count() > 0:
            return True
    except PlaywrightError:
        return False
    return False


def _playwright_max_attempts() -> int:
    """Retry budget for Playwright-supported ATS jobs.

    Keep this higher than the generic default so transient browser/form issues
    can be retried after engine improvements.
    """
    base = int(config.DEFAULTS.get("max_apply_attempts", 3) or 3)
    env_raw = os.environ.get("APPLYPILOT_PLAYWRIGHT_MAX_ATTEMPTS", "").strip()
    if env_raw:
        try:
            return max(1, int(env_raw))
        except ValueError:
            pass
    return max(base, 6)


def _release_stale_in_progress(stale_minutes: int = 15) -> int:
    """Release jobs stuck in in_progress from dead worker sessions."""
    stale_minutes = max(3, int(stale_minutes or 15))
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=stale_minutes)).isoformat()
    conn = get_connection()
    cur = conn.execute(
        """
        UPDATE jobs
        SET apply_status = 'failed',
            apply_error = CASE
                WHEN COALESCE(apply_error, '') = '' THEN 'stale_in_progress'
                ELSE apply_error
            END,
            agent_id = NULL
        WHERE apply_status = 'in_progress'
          AND applied_at IS NULL
          AND last_attempted_at IS NOT NULL
          AND datetime(last_attempted_at) < datetime(?)
        """,
        (cutoff,),
    )
    conn.commit()
    return int(cur.rowcount or 0)


def _artifact_dir() -> Path:
    # <repo>/output/playwright/engine
    return Path(__file__).resolve().parents[3] / "output" / "playwright" / "engine"


def _slug(s: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (s or "").strip()).strip("_")
    return (s[:max_len] or "job").lower()


def _dump_debug(page, job: dict, tag: str) -> None:
    # Never fail the job because of debug artifact creation.
    out_dir = _artifact_dir()
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = f"{ts}_{_slug(job.get('site',''))}_{_slug(job.get('title',''))}_{_slug(tag, 20)}"

    try:
        (out_dir / f"{base}.txt").write_text(
            f"job_url={job.get('url')}\n"
            f"application_url={job.get('application_url')}\n"
            f"page_url={getattr(page, 'url', '')}\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    # Screenshot can fail on some Chromium/system setups; fall back to a normal
    # viewport screenshot if full_page isn't supported.
    try:
        page.screenshot(path=str(out_dir / f"{base}.png"), full_page=True)
    except Exception:
        try:
            page.screenshot(path=str(out_dir / f"{base}.png"), full_page=False)
        except Exception:
            pass

    try:
        (out_dir / f"{base}.html").write_text(page.content(), encoding="utf-8", errors="ignore")
    except Exception:
        pass


def _wait_for_captcha_clear(page, seconds: int) -> bool:
    if seconds <= 0:
        return False
    end = time.time() + seconds
    while time.time() < end:
        if not _looks_captcha(page):
            return True
        try:
            page.wait_for_timeout(2000)
        except PlaywrightError:
            time.sleep(2)
    return not _looks_captcha(page)


def _ashby_wait_recaptcha_ready(page, timeout_ms: int = 15_000) -> None:
    """Best-effort wait for Ashby's invisible reCAPTCHA runtime.

    Ashby submit can get stuck in loading state if submit is clicked before
    `window.grecaptcha.execute` is ready.
    """
    try:
        if page.locator("script#recaptchaScript, .grecaptcha-badge").count() == 0:
            return
    except PlaywrightError:
        return

    try:
        page.wait_for_function(
            "() => typeof window.grecaptcha !== 'undefined' "
            "&& typeof window.grecaptcha.execute === 'function'",
            timeout=timeout_ms,
        )
        page.wait_for_timeout(250)
    except PlaywrightError:
        # Keep moving; not every Ashby form initializes recaptcha the same way.
        pass


def _acquire_target_job(target_url: str, min_score: int, worker_id: int) -> dict | None:
    """Acquire a specific job with safer matching than launcher.acquire_job.

    The launcher implementation strips query params and can accidentally
    match the wrong row for URL types like Indeed (where the job id lives
    in the query string). This function prefers exact matches first.
    """
    target = (target_url or "").strip()
    if not target:
        return None

    target = target.split("#", 1)[0].rstrip("/")
    target_no_query = target.split("?", 1)[0].rstrip("/")
    target_no_apply = target_no_query[:-6] if target_no_query.endswith("/apply") else target_no_query

    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")

        row = None
        for cand in [target, target_no_query, target_no_apply]:
            row = conn.execute(
                """
                SELECT url, title, site, application_url, tailored_resume_path,
                       fit_score, location, full_description, cover_letter_path
                FROM jobs
                WHERE (url = ? OR application_url = ?)
                  AND tailored_resume_path IS NOT NULL
                  AND (apply_status IS NULL OR apply_status != 'in_progress')
                LIMIT 1
                """,
                (cand, cand),
            ).fetchone()
            if row:
                break

        if not row:
            # Fallback to LIKE matches for minor URL differences.
            like_patterns: list[str] = []
            for cand in [target, target_no_query, target_no_apply]:
                if cand and len(cand) >= 20:
                    like_patterns.append(f"%{cand}%")
            # If this is an Indeed link, match on jk=... token.
            m = re.search(r"[?&]jk=([a-zA-Z0-9]+)", target)
            if m:
                like_patterns.insert(0, f"%jk={m.group(1)}%")

            for pat in dict.fromkeys(like_patterns):
                row = conn.execute(
                    """
                    SELECT url, title, site, application_url, tailored_resume_path,
                           fit_score, location, full_description, cover_letter_path
                    FROM jobs
                    WHERE (url LIKE ? OR application_url LIKE ?)
                      AND tailored_resume_path IS NOT NULL
                      AND (apply_status IS NULL OR apply_status != 'in_progress')
                    ORDER BY fit_score DESC, url
                    LIMIT 1
                    """,
                    (pat, pat),
                ).fetchone()
                if row:
                    break

        if not row:
            conn.rollback()
            return None

        from applypilot.config import is_manual_ats
        apply_url = row["application_url"] or row["url"]
        if is_manual_ats(apply_url):
            conn.execute(
                "UPDATE jobs SET apply_status = 'manual', apply_error = 'manual ATS' WHERE url = ?",
                (row["url"],),
            )
            conn.commit()
            return None

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            UPDATE jobs SET apply_status = 'in_progress',
                           agent_id = ?,
                           last_attempted_at = ?
            WHERE url = ?
            """,
            (f"worker-{worker_id}", now, row["url"]),
        )
        conn.commit()
        return dict(row)
    except Exception:
        conn.rollback()
        raise


def _acquire_playwright_job(min_score: int, worker_id: int) -> dict | None:
    """Acquire the next job supported by the Playwright engine.

    This avoids spending worker cycles on unsupported ATS links.
    """
    conn = get_connection()
    try:
        max_attempts = _playwright_max_attempts()
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT url, title, site, application_url, tailored_resume_path,
                   fit_score, location, full_description, cover_letter_path
            FROM jobs
            WHERE tailored_resume_path IS NOT NULL
              AND applied_at IS NULL
              AND COALESCE(fit_score, 0) >= ?
              AND (apply_status IS NULL OR apply_status = 'failed')
              AND COALESCE(apply_attempts, 0) < ?
              AND (
                    lower(coalesce(application_url, url)) LIKE '%jobs.ashbyhq.com%'
                 OR lower(coalesce(application_url, url)) LIKE '%jobs.lever.co%'
                 OR lower(coalesce(application_url, url)) LIKE '%lever.co%'
                 OR lower(coalesce(application_url, url)) LIKE '%greenhouse.io%'
              )
            ORDER BY COALESCE(apply_attempts, 0) ASC,
                     COALESCE(fit_score, 0) DESC,
                     COALESCE(last_attempted_at, '')
            LIMIT 1
            """,
            (int(min_score), int(max_attempts)),
        ).fetchone()

        if not row:
            conn.rollback()
            return None

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            UPDATE jobs SET apply_status = 'in_progress',
                           agent_id = ?,
                           last_attempted_at = ?
            WHERE url = ?
            """,
            (f"worker-{worker_id}", now, row["url"]),
        )
        conn.commit()
        return dict(row)
    except Exception:
        conn.rollback()
        raise


def _looks_expired(page) -> bool:
    # Keep this broad; false positives are cheaper than wasting time on dead postings.
    patterns = [
        r"no longer accepting applications",
        r"job is no longer available",
        r"this job has expired",
        r"position has been filled",
        r"job not found",
        r"the job you requested was not found",
        r"page you were looking for",
        r"\b404\b",
    ]
    for pat in patterns:
        try:
            if page.get_by_text(re.compile(pat, re.I)).count() > 0:
                return True
        except PlaywrightError:
            continue
    return False


def _fill_textbox(page, label_pat: Pattern[str], value: str) -> bool:
    val = (value or "").strip()
    if not val:
        return False

    loc = page.get_by_role("textbox", name=label_pat)
    if loc.count() == 0:
        loc = page.get_by_label(label_pat)
    if loc.count() == 0:
        return False

    try:
        loc.first.fill(val)
        return True
    except PlaywrightError:
        return False


def _set_file_input(page, selectors: list[str], file_path: str) -> bool:
    p = Path(file_path)
    if not p.exists():
        return False

    for sel in selectors:
        loc = page.locator(sel)
        if loc.count() == 0:
            continue
        try:
            loc.first.set_input_files(str(p))
            return True
        except PlaywrightError:
            continue

    # Fallback: any file input on the page.
    loc = page.locator("input[type=file]")
    if loc.count() == 0:
        return False
    try:
        loc.first.set_input_files(str(p))
        return True
    except PlaywrightError:
        return False


def _upload_via_file_chooser(page, trigger, file_path: str) -> bool:
    p = Path(file_path)
    if not p.exists():
        return False
    try:
        with page.expect_file_chooser(timeout=10_000) as fc_info:
            trigger.click()
        fc = fc_info.value
        fc.set_files(str(p))
        return True
    except Exception:
        return False


def _click_radio(page, name_pat: Pattern[str]) -> bool:
    loc = page.get_by_role("radio", name=name_pat)
    if loc.count() == 0:
        return False
    try:
        loc.first.click()
        return True
    except PlaywrightError:
        return False


def _ashby_field_entry(page, title_pat: Pattern[str]):
    entry = page.locator("div.ashby-application-form-field-entry", has_text=title_pat)
    if entry.count() > 0:
        return entry.first
    # Fallback: locate the label then walk up to the field container.
    label = page.locator("label", has_text=title_pat)
    if label.count() == 0:
        return None
    return label.first.locator(
        "xpath=ancestor::*[contains(@class,'ashby-application-form-field-entry')][1]"
    )


def _ashby_click_yesno(page, title_pat: Pattern[str], answer_yes: bool) -> bool:
    entry = _ashby_field_entry(page, title_pat)
    if entry is None:
        return False
    name = "Yes" if answer_yes else "No"
    btn = entry.get_by_role("button", name=re.compile(rf"^{name}$", re.I))
    if btn.count() == 0:
        return False
    try:
        btn.first.click()
        return True
    except PlaywrightError:
        return False


def _ashby_fill_location(page, title_pat: Pattern[str], value: str) -> bool:
    val = (value or "").strip()
    if not val:
        return False
    entry = _ashby_field_entry(page, title_pat)
    if entry is None:
        return False

    box = entry.locator("input[role='combobox']")
    if box.count() == 0:
        box = entry.get_by_role("combobox")
    if box.count() == 0:
        return False

    try:
        box.first.click()
        box.first.fill(val)
        page.wait_for_timeout(300)

        # Prefer selecting from the listbox controlled by this input.
        try:
            controls = box.first.get_attribute("aria-controls") or ""
        except PlaywrightError:
            controls = ""
        if controls:
            # Ashby often uses React-style ids like ":r0:" for the listbox.
            # Those can't be referenced via `#id` without CSS escaping, so
            # target by attribute instead.
            listbox_sel = f'[id="{controls}"]'
            try:
                page.wait_for_selector(f"{listbox_sel} [role='option']", timeout=5_000)
            except PlaywrightError:
                pass
            opt = page.locator(f"{listbox_sel} [role='option']")
            if opt.count() > 0:
                chosen = None
                want = val.lower()
                city = val.split(",", 1)[0].strip().lower()
                # Pick the best match rather than the first (some forms list
                # country-only options first, but require City, Country).
                try:
                    texts = [t.strip() for t in opt.all_inner_texts()]
                except PlaywrightError:
                    texts = []
                if texts:
                    for i, t in enumerate(texts):
                        tl = t.lower()
                        if tl == want:
                            chosen = i
                            break
                    if chosen is None:
                        for i, t in enumerate(texts):
                            if want and want in t.lower():
                                chosen = i
                                break
                    if chosen is None and city:
                        for i, t in enumerate(texts):
                            if city and city in t.lower() and "," in t:
                                chosen = i
                                break
                    if chosen is None:
                        for i, t in enumerate(texts):
                            if "," in t:
                                chosen = i
                                break
                    if chosen is None:
                        chosen = 0
                try:
                    (opt.nth(int(chosen)) if chosen is not None else opt.first).click()
                    return True
                except PlaywrightError:
                    pass

        # Fallback: click the first visible option in any listbox.
        opt = page.locator("[role='listbox'] [role='option']")
        if opt.count() > 0:
            chosen = None
            want = val.lower()
            city = val.split(",", 1)[0].strip().lower()
            try:
                texts = [t.strip() for t in opt.all_inner_texts()]
            except PlaywrightError:
                texts = []
            if texts:
                for i, t in enumerate(texts):
                    if t.lower() == want:
                        chosen = i
                        break
                if chosen is None:
                    for i, t in enumerate(texts):
                        if want and want in t.lower():
                            chosen = i
                            break
                if chosen is None and city:
                    for i, t in enumerate(texts):
                        if city in t.lower() and "," in t:
                            chosen = i
                            break
                if chosen is None:
                    for i, t in enumerate(texts):
                        if "," in t:
                            chosen = i
                            break
                if chosen is None:
                    chosen = 0
            try:
                (opt.nth(int(chosen)) if chosen is not None else opt.first).click()
                return True
            except PlaywrightError:
                pass

        # Last resort: keyboard select.
        box.first.press("ArrowDown")
        box.first.press("Enter")
        return True
    except PlaywrightError:
        return False


def _ashby_fill_required_fields(page, profile: dict) -> int:
    """Best-effort fill for required Ashby fields that frequently block submit."""
    filled = 0
    personal = profile.get("personal", {}) or {}
    city = (personal.get("city") or "").strip()
    country = (personal.get("country") or "").strip()
    loc_val = ", ".join([p for p in [city, country] if p]) or _location(personal)

    if _ashby_fill_location(page, re.compile(r"From which country will you work\\?", re.I), loc_val):
        filled += 1
    if _ashby_fill_location(page, re.compile(r"Current Location", re.I), loc_val):
        filled += 1

    try:
        invalid_combos = page.locator(
            "input[role='combobox'][aria-invalid='true'], [aria-invalid='true'] input[role='combobox']"
        )
        n = min(invalid_combos.count(), 3)
        for i in range(n):
            box = invalid_combos.nth(i)
            try:
                box.click()
                if loc_val:
                    box.fill(loc_val)
                page.wait_for_timeout(200)
                box.press("ArrowDown")
                box.press("Enter")
                filled += 1
            except PlaywrightError:
                continue
    except PlaywrightError:
        pass

    try:
        required_cb = page.locator("input[type='checkbox'][required]")
        for i in range(required_cb.count()):
            cb = required_cb.nth(i)
            try:
                if not cb.is_checked():
                    cb.check(force=True)
                    filled += 1
            except PlaywrightError:
                try:
                    cb.click(force=True)
                    filled += 1
                except PlaywrightError:
                    continue
    except PlaywrightError:
        pass

    return filled


def _submit(page) -> bool:
    # Try accessible button first, then HTML submit controls.
    btn = page.get_by_role("button", name=re.compile(r"submit application|submit\b", re.I))
    if btn.count() > 0:
        try:
            btn.first.click()
            return True
        except PlaywrightError:
            pass

    # Fallback: some sites (notably Ashby) render the submit button with
    # aria-hidden, which keeps it out of the accessibility tree.
    loc = page.locator(
        "button.ashby-application-form-submit-button, button:has-text('Submit Application')"
    )
    if loc.count() > 0:
        try:
            loc.first.click()
            return True
        except PlaywrightError:
            pass

    loc = page.locator("button[type=submit], input[type=submit]")
    if loc.count() == 0:
        return False
    try:
        loc.first.click()
        return True
    except PlaywrightError:
        return False


def _greenhouse_label_text(page, box) -> str:
    try:
        labelledby = (box.get_attribute("aria-labelledby") or "").strip()
    except PlaywrightError:
        labelledby = ""
    if not labelledby:
        return ""

    chunks: list[str] = []
    for raw_id in labelledby.split():
        rid = raw_id.strip()
        if not rid:
            continue
        try:
            txt = page.locator(f'[id="{rid}"]').first.inner_text(timeout=1000).strip()
        except Exception:
            txt = ""
        if txt:
            chunks.append(txt)
    return " ".join(chunks).strip()


def _greenhouse_pick_option_index(texts: list[str], label_text: str, location_seed: str) -> int | None:
    if not texts:
        return None

    ll = label_text.lower()
    choices = [t.strip() for t in texts]
    lower = [t.lower() for t in choices]

    def _contains(pats: tuple[str, ...]) -> int | None:
        for i, t in enumerate(lower):
            for p in pats:
                if p and p in t:
                    return i
        return None

    # Prefer safe opt-outs when available.
    idx = _contains(("prefer not", "not specified", "decline to answer"))
    if idx is not None:
        return idx

    if "location" in ll and location_seed:
        loc = location_seed.lower()
        idx = _contains((loc,))
        if idx is not None:
            return idx
        city = location_seed.split(",", 1)[0].strip().lower()
        if city:
            idx = _contains((city,))
            if idx is not None:
                return idx
        idx = _contains(("canada", "united states", "remote"))
        if idx is not None:
            return idx

    if "year" in ll or "experience" in ll:
        idx = _contains(("3-5", "2-4", "2-5", "3+", "2+", "1-3"))
        if idx is not None:
            return idx

    if "sponsorship" in ll or "authorized" in ll or "visa" in ll:
        idx = _contains(("no", "not require", "do not", "authorized"))
        if idx is not None:
            return idx

    if "ai" in ll:
        idx = _contains(("daily", "often", "regularly", "yes"))
        if idx is not None:
            return idx

    idx = _contains(("no", "none", "n/a"))
    if idx is not None:
        return idx
    return 0


def _greenhouse_fill_required_fields(page, profile: dict) -> int:
    """Best-effort fallback for required custom Greenhouse fields."""
    filled = 0
    personal = profile.get("personal", {})
    location_seed = _location(personal)

    # Required combobox-style fields (React Select).
    boxes = page.locator('input[role="combobox"][aria-required="true"]')
    try:
        box_count = min(boxes.count(), 30)
    except PlaywrightError:
        box_count = 0

    for i in range(box_count):
        box = boxes.nth(i)
        try:
            current = (box.input_value(timeout=500) or "").strip()
        except Exception:
            current = ""
        if current:
            continue

        label_text = _greenhouse_label_text(page, box)
        seed = ""
        ll = label_text.lower()
        if "location" in ll and location_seed:
            seed = location_seed
        elif "year" in ll or "experience" in ll:
            seed = "3"
        elif "sponsorship" in ll or "authorized" in ll or "visa" in ll:
            seed = "No"
        elif "ai" in ll:
            seed = "Yes"
        else:
            seed = "No"

        try:
            box.click(timeout=1000)
            box.fill(seed, timeout=1500)
            page.wait_for_timeout(200)
        except PlaywrightError:
            continue

        try:
            opts = page.locator("[role='listbox'] [role='option']")
            opt_count = min(opts.count(), 40)
        except PlaywrightError:
            opt_count = 0

        texts: list[str] = []
        visible_indexes: list[int] = []
        for j in range(opt_count):
            opt = opts.nth(j)
            try:
                if not opt.is_visible():
                    continue
                t = (opt.inner_text(timeout=500) or "").strip()
            except Exception:
                t = ""
            if not t:
                continue
            visible_indexes.append(j)
            texts.append(t)

        if not texts:
            continue

        idx = _greenhouse_pick_option_index(texts=texts, label_text=label_text, location_seed=location_seed)
        if idx is None:
            idx = 0
        try:
            opts.nth(visible_indexes[int(idx)]).click(timeout=1500)
            page.wait_for_timeout(150)
            filled += 1
        except PlaywrightError:
            continue

    # Required consent checkboxes.
    checks = page.locator('input[type="checkbox"][required], input[type="checkbox"][aria-required="true"]')
    try:
        check_count = min(checks.count(), 20)
    except PlaywrightError:
        check_count = 0
    for i in range(check_count):
        c = checks.nth(i)
        try:
            if c.is_checked():
                continue
            c.check(force=True)
            filled += 1
        except PlaywrightError:
            continue

    return filled


def _ashby_submit_loading(page) -> bool:
    try:
        btn = page.locator("button.ashby-application-form-submit-button")
        if btn.count() == 0:
            return False
        cls = (btn.first.get_attribute("class") or "").lower()
        return "loading" in cls
    except PlaywrightError:
        return False


def _confirm_applied(page) -> bool:
    try:
        if "thank" in (page.url or "").lower():
            return True
    except PlaywrightError:
        pass

    # Ashby displays an in-page "Success" panel instead of navigating away.
    try:
        if page.locator(".ashby-application-form-success-container").count() > 0:
            return True
    except PlaywrightError:
        pass

    pats = [
        r"thank(s)? you",
        r"application submitted",
        r"successfully submitted",
        r"we received your application",
        r"thanks for applying",
    ]
    for pat in pats:
        try:
            if page.get_by_text(re.compile(pat, re.I)).count() > 0:
                return True
        except PlaywrightError:
            continue
    return False


def _apply_lever(page, job: dict, profile: dict, dry_run: bool, captcha_wait_s: int = 0) -> ApplyOutcome:
    url = (job.get("application_url") or job.get("url") or "").strip()
    if not url:
        return ApplyOutcome("failed", "missing_url")

    page.goto(url, wait_until="domcontentloaded", timeout=DEFAULT_TIMEOUT_MS)
    page.wait_for_timeout(750)

    # Lever "application_url" is sometimes the job detail page (not /apply).
    # If we aren't on the form yet, click through.
    try:
        on_form = page.get_by_role("heading", name=re.compile(r"submit your application", re.I)).count() > 0
    except PlaywrightError:
        on_form = False
    if not on_form and "/apply" not in (page.url or ""):
        apply_cta = page.get_by_role("link", name=re.compile(r"apply for this job|apply\b", re.I))
        if apply_cta.count() == 0:
            apply_cta = page.get_by_role("button", name=re.compile(r"apply for this job|apply\b", re.I))
        if apply_cta.count() > 0:
            try:
                apply_cta.first.click()
                page.wait_for_load_state("domcontentloaded", timeout=DEFAULT_TIMEOUT_MS)
                page.wait_for_timeout(750)
            except PlaywrightError:
                pass

    if _looks_expired(page):
        return ApplyOutcome("expired", "expired_or_closed")
    if _looks_captcha(page):
        if captcha_wait_s:
            logger.info("CAPTCHA detected (Lever). Waiting up to %ss for manual solve...", captcha_wait_s)
            _wait_for_captcha_clear(page, captcha_wait_s)
        if _looks_captcha(page):
            _dump_debug(page, job, "captcha")
            return ApplyOutcome("captcha", "captcha")

    # Avoid re-submitting if Lever indicates we've already applied.
    try:
        if page.get_by_text(re.compile(r"already applied|application submitted", re.I)).count() > 0:
            return ApplyOutcome("applied", "already_applied")
    except PlaywrightError:
        pass

    personal = profile.get("personal", {})
    work_auth = profile.get("work_authorization", {})

    resume_path = _resolve_resume_upload_path(job)
    if not resume_path:
        return ApplyOutcome("failed", "missing_resume")

    if not _set_file_input(
        page,
        [
            'input[type="file"][name="resume"]',
            'input[type="file"][data-qa="resume-upload-input"]',
        ],
        resume_path,
    ):
        # Some Lever forms render the input lazily.
        attach = page.get_by_role("button", name=re.compile(r"attach resume", re.I))
        if attach.count() > 0:
            try:
                attach.first.click()
                page.wait_for_timeout(300)
            except PlaywrightError:
                pass
        if not _set_file_input(page, [], resume_path):
            return ApplyOutcome("failed", "resume_upload_failed")

    _fill_textbox(page, re.compile(r"^Full name", re.I), personal.get("full_name", ""))
    _fill_textbox(page, re.compile(r"^Email", re.I), personal.get("email", ""))

    phone = (personal.get("phone") or "").strip()
    if not phone:
        return ApplyOutcome("failed", "missing_phone")
    _fill_textbox(page, re.compile(r"^Phone", re.I), phone)

    _fill_textbox(page, re.compile(r"^Current location", re.I), _location(personal))

    exp = profile.get("experience", {}) or {}
    _fill_textbox(page, re.compile(r"^Current company", re.I), exp.get("current_company", ""))

    _fill_textbox(page, re.compile(r"LinkedIn URL", re.I), personal.get("linkedin_url", ""))
    _fill_textbox(page, re.compile(r"GitHub URL", re.I), personal.get("github_url", ""))
    _fill_textbox(page, re.compile(r"Portfolio URL", re.I), personal.get("portfolio_url", ""))
    _fill_textbox(page, re.compile(r"Other website", re.I), personal.get("website_url", ""))

    sponsorship = (work_auth.get("require_sponsorship") or "").strip().lower()
    if sponsorship in ("yes", "no"):
        _click_radio(page, re.compile(rf"^{sponsorship}$", re.I))

    cover_text = _read_cover_text(job.get("cover_letter_path"))
    if cover_text:
        _fill_textbox(page, re.compile(r"^Additional information", re.I), cover_text)

    if dry_run:
        return ApplyOutcome("applied", "dry_run")

    if not _submit(page):
        _dump_debug(page, job, "submit_missing")
        return ApplyOutcome("failed", "submit_button_not_found")

    try:
        page.wait_for_load_state("networkidle", timeout=DEFAULT_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        pass
    page.wait_for_timeout(1000)

    # Give Lever a chance to render either confirmation or inline validation errors.
    for _ in range(40):  # ~20s
        if _confirm_applied(page):
            return ApplyOutcome("applied")
        if _looks_captcha(page):
            if captcha_wait_s:
                logger.info("CAPTCHA detected after submit (Lever). Waiting up to %ss...", captcha_wait_s)
                _wait_for_captcha_clear(page, captcha_wait_s)
            if _looks_captcha(page):
                _dump_debug(page, job, "captcha_after_submit")
                return ApplyOutcome("captcha", "captcha")
        try:
            if page.locator('[aria-invalid="true"]').count() > 0:
                _dump_debug(page, job, "invalid_fields")
                return ApplyOutcome("failed", "form_error")
        except PlaywrightError:
            pass
        page.wait_for_timeout(500)

    if _confirm_applied(page):
        return ApplyOutcome("applied")

    try:
        if page.get_by_text(re.compile(r"\berror\b|required", re.I)).count() > 0:
            _dump_debug(page, job, "form_error")
            return ApplyOutcome("failed", "form_error")
    except PlaywrightError:
        pass

    _dump_debug(page, job, "no_confirmation")
    return ApplyOutcome("failed", "no_confirmation")


def _apply_greenhouse(page, job: dict, profile: dict, dry_run: bool, captcha_wait_s: int = 0) -> ApplyOutcome:
    url = (job.get("application_url") or job.get("url") or "").strip()
    if not url:
        return ApplyOutcome("failed", "missing_url")

    page.goto(url, wait_until="domcontentloaded", timeout=DEFAULT_TIMEOUT_MS)
    page.wait_for_timeout(750)

    if _looks_expired(page):
        return ApplyOutcome("expired", "expired_or_closed")
    if _looks_captcha(page):
        if captcha_wait_s:
            logger.info("CAPTCHA detected (Greenhouse). Waiting up to %ss for manual solve...", captcha_wait_s)
            _wait_for_captcha_clear(page, captcha_wait_s)
        if _looks_captcha(page):
            _dump_debug(page, job, "captcha")
            return ApplyOutcome("captcha", "captcha")

    try:
        if page.get_by_text(re.compile(r"already applied|application submitted", re.I)).count() > 0:
            return ApplyOutcome("applied", "already_applied")
    except PlaywrightError:
        pass

    personal = profile.get("personal", {})

    # Greenhouse links can be job descriptions (job-boards.greenhouse.io) where the
    # application form is below the fold or behind an "Apply" CTA.
    try:
        on_form = page.locator("input[type=file]#resume, input[type=file][name='resume'], input[type=file][id*='resume']").count() > 0
    except PlaywrightError:
        on_form = False
    if not on_form:
        apply_cta = page.get_by_role("link", name=re.compile(r"apply now|apply\b", re.I))
        if apply_cta.count() == 0:
            apply_cta = page.get_by_role("button", name=re.compile(r"apply now|apply\b", re.I))
        if apply_cta.count() > 0:
            try:
                apply_cta.first.click()
                page.wait_for_timeout(750)
            except PlaywrightError:
                pass
        try:
            if _looks_captcha(page):
                if captcha_wait_s:
                    logger.info("CAPTCHA detected after Apply CTA (Greenhouse). Waiting up to %ss...", captcha_wait_s)
                    _wait_for_captcha_clear(page, captcha_wait_s)
                if _looks_captcha(page):
                    _dump_debug(page, job, "captcha_after_apply_cta")
                    return ApplyOutcome("captcha", "captcha")
        except PlaywrightError:
            pass

    resume_path = _resolve_resume_upload_path(job)
    if not resume_path:
        return ApplyOutcome("failed", "missing_resume")

    if not _set_file_input(
        page,
        [
            'input[type="file"]#resume',
            'input[type="file"][name="resume"]',
            'input[type="file"][id*="resume"]',
        ],
        resume_path,
    ):
        return ApplyOutcome("failed", "resume_upload_failed")

    # Optional cover letter upload.
    cover_pdf = _resolve_cover_upload_path(job)
    if cover_pdf:
        _set_file_input(
            page,
            [
                'input[type="file"]#cover_letter',
                'input[type="file"][name="cover_letter"]',
                'input[type="file"][id*="cover_letter"]',
            ],
            cover_pdf,
        )

    first, last = _first_last(personal.get("full_name", ""))
    _fill_textbox(page, re.compile(r"First Name", re.I), first)
    _fill_textbox(page, re.compile(r"Last Name", re.I), last)
    _fill_textbox(page, re.compile(r"Email", re.I), personal.get("email", ""))

    phone = (personal.get("phone") or "").strip()
    if phone:
        _fill_textbox(page, re.compile(r"Phone", re.I), phone)

    _fill_textbox(page, re.compile(r"Location", re.I), _location(personal))
    _fill_textbox(page, re.compile(r"LinkedIn", re.I), personal.get("linkedin_url", ""))
    _greenhouse_fill_required_fields(page, profile)

    if dry_run:
        return ApplyOutcome("applied", "dry_run")

    if not _submit(page):
        _dump_debug(page, job, "submit_missing")
        return ApplyOutcome("failed", "submit_button_not_found")

    try:
        page.wait_for_load_state("networkidle", timeout=DEFAULT_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        pass
    page.wait_for_timeout(1000)

    if _confirm_applied(page):
        return ApplyOutcome("applied")

    try:
        if page.locator('[aria-invalid="true"]').count() > 0:
            if _greenhouse_fill_required_fields(page, profile) > 0 and _submit(page):
                try:
                    page.wait_for_load_state("networkidle", timeout=DEFAULT_TIMEOUT_MS)
                except PlaywrightTimeoutError:
                    pass
                page.wait_for_timeout(800)
                if _confirm_applied(page):
                    return ApplyOutcome("applied")
        if page.get_by_text(re.compile(r"\berror\b|required", re.I)).count() > 0:
            _dump_debug(page, job, "form_error")
            return ApplyOutcome("failed", "form_error")
    except PlaywrightError:
        pass

    _dump_debug(page, job, "no_confirmation")
    return ApplyOutcome("failed", "no_confirmation")


def _apply_ashby(page, job: dict, profile: dict, dry_run: bool, captcha_wait_s: int = 0) -> ApplyOutcome:
    url = (job.get("application_url") or job.get("url") or "").strip()
    if not url:
        return ApplyOutcome("failed", "missing_url")

    # Prefer the /application route when given a job detail URL.
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(url)
        if "jobs.ashbyhq.com" in parsed.netloc.lower() and "/application" not in parsed.path:
            path = parsed.path.rstrip("/") + "/application"
            parsed = parsed._replace(path=path, query="")  # tracking params not needed
            url = urlunparse(parsed)
    except Exception:
        pass

    page.goto(url, wait_until="domcontentloaded", timeout=DEFAULT_TIMEOUT_MS)
    page.wait_for_timeout(750)

    if _looks_expired(page):
        return ApplyOutcome("expired", "expired_or_closed")
    if _looks_captcha(page):
        if captcha_wait_s:
            logger.info("CAPTCHA detected (Ashby). Waiting up to %ss for manual solve...", captcha_wait_s)
            _wait_for_captcha_clear(page, captcha_wait_s)
        if _looks_captcha(page):
            _dump_debug(page, job, "captcha")
            return ApplyOutcome("captcha", "captcha")

    try:
        if _confirm_applied(page):
            return ApplyOutcome("applied", "already_applied")
        if page.get_by_text(re.compile(r"already applied|application submitted", re.I)).count() > 0:
            return ApplyOutcome("applied", "already_applied")
    except PlaywrightError:
        pass

    personal = profile.get("personal", {})

    resume_path = _resolve_resume_upload_path(job)
    if not resume_path:
        return ApplyOutcome("failed", "missing_resume")

    # Ashby pages can include multiple file inputs (autofill + required resume).
    # Upload specifically to the required resume input.
    try:
        page.wait_for_selector("input#_systemfield_resume[type=file]", timeout=DEFAULT_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        pass

    uploaded = _set_file_input(page, ["input#_systemfield_resume[type='file']"], resume_path)
    if not uploaded:
        # Fallback: some forms use only a required input without stable ids.
        uploaded = _set_file_input(page, ["input[type='file'][required]"], resume_path)
    if not uploaded:
        if _looks_expired(page):
            return ApplyOutcome("expired", "expired_or_closed")
        _dump_debug(page, job, "resume_upload_failed")
        return ApplyOutcome("failed", "resume_upload_failed")
    page.wait_for_timeout(500)

    first, last = _first_last(personal.get("full_name", ""))
    filled = False
    filled |= _fill_textbox(page, re.compile(r"First name", re.I), first)
    filled |= _fill_textbox(page, re.compile(r"Last name", re.I), last)
    if not filled:
        _fill_textbox(page, re.compile(r"Full name|Name", re.I), personal.get("full_name", ""))

    _fill_textbox(page, re.compile(r"Email", re.I), personal.get("email", ""))
    _fill_textbox(page, re.compile(r"Phone", re.I), personal.get("phone", ""))
    _fill_textbox(page, re.compile(r"LinkedIn", re.I), personal.get("linkedin_url", ""))
    _fill_textbox(page, re.compile(r"GitHub", re.I), personal.get("github_url", ""))

    # Ashby frequently includes required custom questions. Keep answers truthful
    # and derived from the user's tailored cover letter / resume when possible.
    cover_text = _read_cover_text(job.get("cover_letter_path"))
    why = ""
    if cover_text:
        # Strip greeting/signature for question-style inputs.
        lines = [ln.strip() for ln in cover_text.splitlines() if ln.strip()]
        if lines and lines[0].lower().startswith("dear "):
            lines = lines[1:]
        if lines and len(lines[-1].split()) <= 2:
            lines = lines[:-1]
        why = "\n\n".join(lines).strip()
    if not why:
        why = (
            "I'm interested in this role because it combines front-end craftsmanship with payments systems "
            "reliability. I've shipped React/Next.js applications with real-world payment flows (including "
            "Stripe integrations), and I care about building secure, observable, and accessible user experiences "
            "at scale."
        )

    _fill_textbox(page, re.compile(r"Why are you interested in working at Kraken\\?", re.I), why)

    # Optional: favorite aspect of the platform.
    _fill_textbox(
        page,
        re.compile(r"What is your favorite aspect of our platform\\?", re.I),
        "A strong security-first posture paired with a fast, reliable user experience for high-volume interactions.",
    )

    # Yes/No questions are rendered as buttons in Ashby.
    _ashby_click_yesno(page, re.compile(r"Have you used a Kraken product.*six months\\?", re.I), answer_yes=False)

    sponsorship = (profile.get("work_authorization", {}) or {}).get("require_sponsorship", "")
    sponsorship_yes = str(sponsorship).strip().lower() in ("yes", "y", "true", "1")
    _ashby_click_yesno(
        page,
        re.compile(r"need sponsorship to work in your location\\?", re.I),
        answer_yes=sponsorship_yes,
    )

    _ashby_click_yesno(
        page,
        re.compile(r"experience working with blockchains|blockchain networks", re.I),
        answer_yes=False,
    )
    _ashby_click_yesno(
        page,
        re.compile(r"payment processing systems.*react", re.I),
        answer_yes=True,
    )

    # Location combobox: requires selecting a suggestion.
    city = (personal.get("city") or "").strip()
    country = (personal.get("country") or "").strip()
    loc_val = ", ".join([p for p in [city, country] if p]) or _location(personal)
    _ashby_fill_location(page, re.compile(r"From which country will you work\\?", re.I), loc_val)
    _ashby_fill_location(page, re.compile(r"Current Location", re.I), loc_val)
    _ashby_fill_required_fields(page, profile)

    _fill_textbox(
        page,
        re.compile(r"Please briefly outline.*payments processing system.*delivered\\?", re.I),
        (
            "I delivered a full-stack checkout flow for an e-commerce marketplace built with React/Next.js and Stripe. "
            "The frontend collected payment details (Stripe Elements), handled client-side validation/error states, and "
            "created PaymentIntents via a backend API. The backend processed webhooks for async events (success/failure/"
            "refunds), ensured idempotency, and updated order state in the database. We added logging/metrics around key "
            "steps and automated tests to prevent regressions."
        ),
    )

    if dry_run:
        return ApplyOutcome("applied", "dry_run")

    _ashby_wait_recaptcha_ready(page)

    if not _submit(page):
        _dump_debug(page, job, "submit_missing")
        return ApplyOutcome("failed", "submit_button_not_found")

    # Ashby submit is in-page and can take time (invisible reCAPTCHA + API).
    # Poll for terminal states instead of relying on networkidle.
    end = time.time() + 75
    saw_loading = False
    retried_required = False
    while time.time() < end:
        if _confirm_applied(page):
            return ApplyOutcome("applied")

        try:
            if page.locator('[aria-invalid="true"]').count() > 0:
                if not retried_required and _ashby_fill_required_fields(page, profile) > 0 and _submit(page):
                    retried_required = True
                    page.wait_for_timeout(800)
                    continue
                _dump_debug(page, job, "invalid_fields")
                return ApplyOutcome("failed", "form_error")
        except PlaywrightError:
            pass

        try:
            if page.get_by_text(re.compile(r"your form needs corrections|missing entry|required field", re.I)).count() > 0:
                if not retried_required and _ashby_fill_required_fields(page, profile) > 0 and _submit(page):
                    retried_required = True
                    page.wait_for_timeout(800)
                    continue
                _dump_debug(page, job, "form_error")
                return ApplyOutcome("failed", "form_error")
        except PlaywrightError:
            pass

        if _looks_captcha(page):
            if captcha_wait_s:
                _wait_for_captcha_clear(page, captcha_wait_s)
            if _looks_captcha(page):
                _dump_debug(page, job, "captcha")
                return ApplyOutcome("captcha", "captcha")

        loading = _ashby_submit_loading(page)
        saw_loading = saw_loading or loading

        # If loading started and then stopped without success/error, it's a
        # submit failure that should be retried.
        if saw_loading and not loading:
            break

        page.wait_for_timeout(1000)

    if _ashby_submit_loading(page):
        _dump_debug(page, job, "submit_stuck")
        return ApplyOutcome("failed", "submit_stuck")

    _dump_debug(page, job, "no_confirmation")
    return ApplyOutcome("failed", "no_confirmation")


def _route_apply(url: str) -> str:
    u = (url or "").lower()
    if "jobs.lever.co" in u or "lever.co" in u:
        return "lever"
    if "greenhouse.io" in u:
        return "greenhouse"
    if "jobs.ashbyhq.com" in u:
        return "ashby"
    return "unknown"


def _is_browser_crash_error(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}:{exc}".lower()
    crash_markers = (
        "page crashed",
        "navigation failed because page crashed",
        "targetclosederror",
        "target page, context or browser has been closed",
        "browser has been closed",
        "connection closed",
    )
    return any(marker in msg for marker in crash_markers)


def _start_worker_session(playwright, worker_id: int, headless: bool):
    """Launch Chrome + attach Playwright CDP with retries."""
    port = BASE_CDP_PORT + int(worker_id)
    last_err: Exception | None = None

    for _launch_attempt in range(4):
        try:
            proc = launch_chrome(worker_id=worker_id, port=port, headless=headless)
        except Exception as exc:
            last_err = exc
            time.sleep(2)
            continue
        browser = None

        for _ in range(35):
            try:
                browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
                break
            except Exception as exc:
                last_err = exc
                time.sleep(1)

        if browser is not None:
            context = browser.contexts[0] if browser.contexts else browser.new_context(ignore_https_errors=True)
            page = context.pages[0] if context.pages else context.new_page()
            page.set_default_timeout(DEFAULT_TIMEOUT_MS)
            try:
                page.set_viewport_size({"width": 1280, "height": 900})
            except PlaywrightError:
                pass
            return proc, browser, context, page

        rc = None
        try:
            rc = proc.poll()
        except Exception:
            rc = None

        cleanup_worker(worker_id, proc)
        if rc not in (None, 0):
            logger.warning(
                "Chrome exited during startup (worker=%s, rc=%s).",
                worker_id,
                rc,
            )

        # Self-heal: worker profile can become corrupt/locked and cause
        # deterministic startup failures on a single worker id.
        if _launch_attempt == 1:
            try:
                worker_profile = config.CHROME_WORKER_DIR / f"worker-{worker_id}"
                shutil.rmtree(worker_profile, ignore_errors=True)
                setup_worker_profile(worker_id)
                logger.warning("Rebuilt worker profile for worker-%s after repeated startup failures.", worker_id)
            except Exception:
                logger.exception("Failed rebuilding worker-%s profile", worker_id)
        time.sleep(2)

    raise RuntimeError(f"cdp_connect_failed:{type(last_err).__name__}")


def main(
    limit: int = 1,
    target_url: str | None = None,
    min_score: int = 7,
    headless: bool = False,
    dry_run: bool = False,
    continuous: bool = False,
    poll_interval: int = 60,
    worker_id: int = 0,
) -> None:
    """Apply to jobs using Playwright directly (no Claude Code).

    Args:
        limit: Max jobs to apply to (0 means unlimited).
        target_url: Apply to a specific URL.
        min_score: Minimum fit_score threshold.
        headless: Run the browser in headless mode.
        dry_run: Fill forms but do not click final submit.
        continuous: Keep running forever, polling for new jobs.
        poll_interval: Seconds between DB polls when the queue is empty.
        worker_id: Worker profile id to reuse.
    """
    config.load_env()
    config.ensure_dirs()

    profile = config.load_profile()
    # profile.json can be locked down by ACLs on some Windows setups; allow
    # one-off overrides via env vars so the user can still run auto-apply.
    phone_override = os.environ.get("APPLYPILOT_PHONE", "").strip()
    if phone_override:
        profile.setdefault("personal", {})["phone"] = phone_override
    phone = (profile.get("personal", {}).get("phone") or "").strip()
    if not phone:
        raise SystemExit(
            f"Profile phone is empty ({config.PROFILE_PATH}). "
            "Add your phone number to profile.json before running auto-apply."
        )

    captcha_wait_s = 0
    try:
        captcha_wait_s = int(os.environ.get("APPLYPILOT_CAPTCHA_WAIT", "0") or "0")
    except ValueError:
        captcha_wait_s = 0

    # Playwright mode does not require logged-in browser cookies for the ATS
    # forms we support; clean profiles avoid persistent corruption/lock issues.
    os.environ.setdefault("APPLYPILOT_CLEAN_PROFILE", "1")

    # Use an isolated profile so we can keep cookies (LinkedIn/ATS) without
    # risking corruption of the user's real Chrome profile.
    clean_profile = os.environ.get("APPLYPILOT_CLEAN_PROFILE", "").strip().lower() in ("1", "true", "yes")
    if not clean_profile:
        setup_worker_profile(worker_id)

    applied = 0
    failed = 0
    processed = 0

    proc = None
    try:
        with sync_playwright() as p:
            proc, browser, context, page = _start_worker_session(p, worker_id=worker_id, headless=headless)
            stale_env = os.environ.get("APPLYPILOT_STALE_IN_PROGRESS_MIN", "8").strip()
            try:
                stale_window_min = int(stale_env or "8")
            except ValueError:
                stale_window_min = 8
            stale_window_min = max(3, min(stale_window_min, 60))
            last_stale_sweep = 0.0

            while True:
                now_ts = time.time()
                if now_ts - last_stale_sweep >= 60:
                    released = _release_stale_in_progress(stale_minutes=stale_window_min)
                    if released:
                        logger.info("Released %d stale in_progress jobs.", released)
                    last_stale_sweep = now_ts

                if limit and processed >= limit:
                    break

                if target_url:
                    job = _acquire_target_job(target_url=target_url, min_score=min_score, worker_id=worker_id)
                else:
                    job = _acquire_playwright_job(min_score=min_score, worker_id=worker_id)
                if not job:
                    if continuous:
                        time.sleep(max(5, int(poll_interval)))
                        continue
                    break

                start = time.time()
                status = "failed"
                error: str | None = None
                job_page = page
                created_page = None
                try:
                    created_page = context.new_page()
                    created_page.set_default_timeout(DEFAULT_TIMEOUT_MS)
                    job_page = created_page
                except Exception:
                    job_page = page
                try:
                    apply_url = job.get("application_url") or job.get("url") or ""
                    engine = _route_apply(apply_url)
                    if engine == "lever":
                        out = _apply_lever(
                            job_page, job, profile, dry_run=dry_run, captcha_wait_s=(0 if headless else captcha_wait_s)
                        )
                    elif engine == "greenhouse":
                        out = _apply_greenhouse(
                            job_page, job, profile, dry_run=dry_run, captcha_wait_s=(0 if headless else captcha_wait_s)
                        )
                    elif engine == "ashby":
                        out = _apply_ashby(
                            job_page, job, profile, dry_run=dry_run, captcha_wait_s=(0 if headless else captcha_wait_s)
                        )
                    else:
                        out = ApplyOutcome("manual", "unsupported_ats")

                    status, error = out.status, out.error
                    duration_ms = int((time.time() - start) * 1000)

                    # In dry-run mode we never persist results (especially "applied")
                    # because no final submit click happened.
                    if dry_run:
                        release_lock(job["url"])
                        processed += 1
                        if target_url:
                            break
                        continue

                    if status == "applied":
                        mark_result(job["url"], "applied", duration_ms=duration_ms)
                        applied += 1
                    elif status in ("expired", "manual", "captcha"):
                        mark_result(job["url"], status, error=error or status, permanent=True, duration_ms=duration_ms)
                        failed += 1
                    else:
                        # Normal retryable failure.
                        mark_result(
                            job["url"], "failed", error=error or "unknown", permanent=False, duration_ms=duration_ms
                        )
                        failed += 1
                except Exception as exc:
                    duration_ms = int((time.time() - start) * 1000)
                    logger.exception("Playwright engine failed for job: %s", job.get("url"))
                    try:
                        mark_result(job["url"], "failed", error=f"engine_error:{type(exc).__name__}", duration_ms=duration_ms)
                    except Exception:
                        release_lock(job["url"])
                    failed += 1
                    if _is_browser_crash_error(exc):
                        logger.warning(
                            "Playwright browser/page crashed (worker=%s). Restarting browser session.",
                            worker_id,
                        )
                        try:
                            browser.close()
                        except Exception:
                            pass
                        cleanup_worker(worker_id, proc)
                        proc, browser, context, page = _start_worker_session(
                            p, worker_id=worker_id, headless=headless
                        )
                finally:
                    if created_page is not None:
                        try:
                            created_page.close()
                        except Exception:
                            pass

                processed += 1
                if target_url:
                    break

            try:
                browser.close()
            except Exception:
                pass
    finally:
        cleanup_worker(worker_id, proc)

    logger.info("Playwright apply done: %d applied, %d failed", applied, failed)
