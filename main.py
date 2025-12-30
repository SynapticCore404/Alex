import os
import asyncio
import logging
import re
import hashlib
import json
import time
import sqlite3
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import NotFound
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

load_dotenv()
REL_THRESHOLD_ENV = os.getenv("RELEVANCE_THRESHOLD")
try:
    REL_THRESHOLD = float(REL_THRESHOLD_ENV) if REL_THRESHOLD_ENV is not None else 0.25
    if REL_THRESHOLD < 0.0:
        REL_THRESHOLD = 0.0
    if REL_THRESHOLD > 1.0:
        REL_THRESHOLD = 1.0
except Exception:
    REL_THRESHOLD = 0.25
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TARGET_CHANNEL_ID_RAW = os.getenv("TARGET_CHANNEL_ID")
TARGET_CHANNEL_ID = int(TARGET_CHANNEL_ID_RAW) if TARGET_CHANNEL_ID_RAW else None
RUN_MODE = (os.getenv("RUN_MODE") or "POLLING").strip().upper()
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
WEBHOOK_PATH = (os.getenv("WEBHOOK_PATH") or "").strip()
LISTEN = (os.getenv("LISTEN") or "0.0.0.0").strip()
PORT = int(os.getenv("PORT") or 8080)
WEBHOOK_SECRET_TOKEN = os.getenv("WEBHOOK_SECRET_TOKEN")
GEMINI_FAST_SWITCH = True if (os.getenv("GEMINI_FAST_SWITCH") or "1").strip() not in ("0", "false", "False") else False

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

genai.configure(api_key=GEMINI_API_KEY)

# Derive our bot ID from the token prefix (Telegram bot tokens start with the bot's numeric ID)
try:
    MY_BOT_ID = int((TELEGRAM_BOT_TOKEN.split(":", 1)[0]).strip()) if ":" in TELEGRAM_BOT_TOKEN else None
except Exception:
    MY_BOT_ID = None
BOT_USERNAME = None
BOT_USERNAME_STATIC = (os.getenv("BOT_USERNAME") or "CommentatorAlex_bot").strip().lstrip("@").lower()
def _normalize_model_name(name: str) -> str:
    return name.split("/")[-1] if "/" in name else name


async def _ensure_bot_username(context) -> str:
    global BOT_USERNAME
    if BOT_USERNAME:
        return BOT_USERNAME
    try:
        uname = getattr(context.bot, "username", None)
        if not uname:
            me = await context.bot.get_me()
            uname = getattr(me, "username", None)
        if uname:
            BOT_USERNAME = str(uname).lower()
    except Exception:
        BOT_USERNAME = BOT_USERNAME or ""
    return BOT_USERNAME or ""


def _message_mentions_bot(msg, bot_username: str) -> bool:
    try:
        text = (getattr(msg, "text", None) or getattr(msg, "caption", None) or "")
        at = f"@{bot_username}" if bot_username else None
        ents = (getattr(msg, "entities", []) or []) + (getattr(msg, "caption_entities", []) or [])
        def _etype(v) -> str:
            try:
                if hasattr(v, "value") and getattr(v, "value"):
                    return str(getattr(v, "value")).lower()
                s = str(v).lower() if v is not None else ""
                if s.startswith("messageentitytype."):
                    s = s.split(".", 1)[1]
                return s
            except Exception:
                return str(v).lower() if v is not None else ""
        for e in ents:
            t_str = _etype(getattr(e, "type", None))
            if t_str == "mention" and at:
                try:
                    start = int(getattr(e, "offset", 0) or 0)
                    length = int(getattr(e, "length", 0) or 0)
                    piece = text[start : start + length]
                except Exception:
                    piece = None
                if piece:
                    piece_l = piece.lower()
                    at_l = at.lower()
                    if piece_l == at_l:
                        return True
                    # Tolerate trailing punctuation included by some clients
                    piece_trim = piece_l.rstrip(".,:;!?…—–-)]}>\"'“”’»」』】）》")
                    if piece_trim == at_l:
                        return True
            elif t_str == "text_mention":
                u = getattr(e, "user", None)
                if u and getattr(u, "id", None) == MY_BOT_ID:
                    return True
            elif t_str == "text_link":
                url = getattr(e, "url", None) or getattr(e, "text_link", None)
                if url and _url_points_to_bot(url, bot_username):
                    return True
        if at:
            # Fallback: exact @username match with word boundaries (avoid partials like @usernameX)
            try:
                pat = re.compile(rf"(?i)(?<!\w){re.escape(at)}(?!\w)")
                if pat.search(text or ""):
                    return True
                # Accept t.me/username, telegram.me/username, tg://resolve?domain=username in plain text
                tme = re.compile(rf"(?i)(?:https?://)?(?:t(?:elegram)?\.me|telegram\.me)/{re.escape(bot_username)}(?:\b|/|$)")
                if tme.search(text or ""):
                    return True
                tgres = re.compile(rf"(?i)tg://resolve\?domain={re.escape(bot_username)}(?:\b|&|$)")
                if tgres.search(text or ""):
                    return True
            except Exception:
                pass
        # Last-resort fallback: simple substring match (case-insensitive)
        # This helps in edge-cases where Telegram entities are missing and
        # boundary regex fails due to unusual surrounding characters.
        if at and (text or ""):
            try:
                if at.lower() in (text or "").lower():
                    return True
            except Exception:
                pass
        # Extreme fallback: plain username without '@' with word boundaries
        if bot_username and (text or ""):
            try:
                nm = bot_username.lower()
                pat2 = re.compile(rf"(?i)(?<!\w){re.escape(nm)}(?!\w)")
                if pat2.search((text or "").lower()):
                    return True
                if nm in (text or "").lower():
                    return True
            except Exception:
                pass
        # Also treat plain display name (e.g., 'Alex') as a mention
        try:
            name_raw = (os.getenv("BOT_NAME") or "Alex").strip()
            if name_raw:
                pat_name = re.compile(rf"(?i)(?<!\w){re.escape(name_raw)}(?!\w)")
                if pat_name.search(text or ""):
                    return True
        except Exception:
            pass
    except Exception:
        return False
    return False


def _strip_bot_mention(text: str, bot_username: str) -> str:
    try:
        if not text or not bot_username:
            return text
        at = f"@{bot_username}"
        out = text
        # Remove all occurrences of @bot_username (case-insensitive) with word boundaries
        pat_at = re.compile(rf"(?i)(?<!\w){re.escape(at)}(?!\w)")
        out = pat_at.sub(" ", out)
        # Also remove plain username without '@' with word boundaries
        pat_nm = re.compile(rf"(?i)(?<!\w){re.escape(bot_username)}(?!\w)")
        out = pat_nm.sub(" ", out)
        # Also remove display name (e.g., 'Alex') if configured
        try:
            name_raw = (os.getenv("BOT_NAME") or "Alex").strip()
            if name_raw:
                pat_name = re.compile(rf"(?i)(?<!\w){re.escape(name_raw)}(?!\w)")
                out = pat_name.sub(" ", out)
        except Exception:
            pass
        # Remove t.me/username & tg://resolve?domain=username
        out = re.sub(rf"(?i)(?:https?://)?(?:t(?:elegram)?\.me|telegram\.me)/{re.escape(bot_username)}(?:\b|/|$)", " ", out)
        out = re.sub(rf"(?i)tg://resolve\?domain={re.escape(bot_username)}(?:\b|&|$)", " ", out)
        return _clean_text(out)
    except Exception:
        return text

def _url_points_to_bot(url: str, bot_username: str) -> bool:
    try:
        if not url or not bot_username:
            return False
        u = str(url).strip()
        nm = bot_username.lower()
        if re.search(rf"(?i)^(?:https?://)?(?:t(?:elegram)?\.me|telegram\.me)/{re.escape(nm)}(?:\b|/|$)", u):
            return True
        if re.search(rf"(?i)^tg://resolve\?domain={re.escape(nm)}(?:\b|&|$)", u):
            return True
        return False
    except Exception:
        return False

def _is_anonymous_author(msg) -> bool:
    try:
        fu = getattr(msg, "from_user", None)
        if fu and getattr(fu, "is_bot", False) and getattr(fu, "id", None) == 1087968824:
            return True
        sc = getattr(msg, "sender_chat", None)
        chat = getattr(msg, "chat", None)
        if sc and chat and getattr(sc, "type", None) == "supergroup" and getattr(sc, "id", None) == getattr(chat, "id", None):
            return True
        return False
    except Exception:
        return False

def _parse_retry_delay_secs(err) -> Optional[float]:
    try:
        s = str(err) or ""
        m = re.search(r"(?i)please\s+retry\s+in\s*([0-9]+(?:\.[0-9]+)?)s", s)
        if m:
            return float(m.group(1))
        m2 = re.search(r"(?i)retry_delay\s*\{\s*seconds\s*:\s*([0-9]+)", s)
        if m2:
            return float(m2.group(1))
        return None
    except Exception:
        return None

def _build_model_chain():
    env_models = (os.getenv("GEMINI_MODELS") or "").strip()
    env_model = (os.getenv("GEMINI_MODEL") or "").strip()
    # Discover available models first (best-effort)
    available = []
    try:
        for m in genai.list_models():
            methods = set(getattr(m, "supported_generation_methods", []) or [])
            if "generateContent" in methods:
                nm = getattr(m, "name", "") or ""
                if nm:
                    available.append(nm)
    except Exception:
        available = []
    try:
        logging.info("gemini discover: %s generative models available", len(available))
    except Exception:
        pass
    # Prefer text-out models only when auto-selecting (skip image/tts/robotics/computer-use)
    def _is_text_out(nm: str) -> bool:
        n = (nm or "").lower()
        banned = (
            "-tts", "tts", "image", "vision", "computer-use", "robotics",
            "exp-image", "image-generation", "audio", "speech",
            "preview", "latest", "deep-research", "nano", "exp"
        )
        return not any(b in n for b in banned)
    if env_models:
        prefs = [p.strip() for p in env_models.split(",") if p.strip()]
    elif env_model:
        prefs = [env_model]
    else:
        # If no explicit models provided, prefer models actually available to this API key
        def _score(nm: str) -> tuple:
            n = nm.lower()
            # version priority
            v = 0
            if "gemini-3" in n:
                v = 5
            if "2.5" in n:
                v = max(v, 4)
            elif "2.0" in n:
                v = 3
            elif "1.5" in n:
                v = 2
            elif "1.0" in n or "-pro" in n or n.endswith("/gemini-pro"):
                v = 1
            # type priority within version
            t = 9
            if "flash" in n and "8b" not in n:
                t = 0
            elif "flash-8b" in n:
                t = 1
            elif "pro" in n:
                t = 2
            elif "flash-lite" in n or "lite" in n:
                t = 3
            return (-v, t, n)
        # Only include text-out by default
        filtered = [nm for nm in available if _is_text_out(nm)]
        if filtered:
            prefs = sorted(filtered, key=_score)
        else:
            # Fallback to a reasonable default list
            prefs = [
                "models/gemini-2.5-flash",
                "models/gemini-2.5-flash-lite",
                "models/gemini-3-flash",
                "models/gemini-1.5-flash",
                "models/gemini-1.5-flash-8b",
                "models/gemini-1.5-pro",
                "models/gemma-3-12b-it",
                "models/gemma-3-4b-it",
                "models/gemma-3-2b-it",
            ]
    try:
        logging.info("gemini MODEL_PREFS: %s", prefs[:12] + (["..."] if len(prefs) > 12 else []))
    except Exception:
        pass
    # Use JSON mode where supported (Gemini models). Gemma IT models don't support JSON mode.
    cfg_json = {"temperature": 0.6, "top_p": 0.9, "top_k": 32, "max_output_tokens": 256, "response_mime_type": "application/json"}
    cfg_text = {"temperature": 0.6, "top_p": 0.9, "top_k": 32, "max_output_tokens": 256}
    def _supports_json_mode(nm: str) -> bool:
        n = (nm or "").lower()
        # Gemini family supports JSON mode; Gemma does not
        return n.startswith("models/gemini-") and ("tts" not in n) and ("image" not in n)
    chain = []
    seen = set()
    for n in prefs:
        candidates = [n]
        if not n.startswith("models/"):
            candidates.append(f"models/{n}")
        # Add -001 alias for Gemini models when not explicitly versioned
        more = []
        for base in list(candidates):
            bname = base.split("/", 1)[-1]
            if bname.startswith("gemini-") and not bname.endswith("-001"):
                more.append(f"models/{bname}-001")
        candidates.extend(more)
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            try:
                use_cfg = cfg_json if _supports_json_mode(cand) else cfg_text
                model = genai.GenerativeModel(cand, generation_config=use_cfg)
                chain.append((cand, model))
            except Exception as e:
                logging.info("gemini model init failed: %s [%s]", cand, e)
                continue
    return chain


MODEL_CHAIN = _build_model_chain()
try:
    logging.info("gemini MODEL_CHAIN: %s", [name for name, _ in MODEL_CHAIN])
except Exception:
    pass


def _detect_lang(text: str) -> str:
    if re.search(r"[\u0400-\u04FF]", text):
        return "ru"
    lowered = text.lower().strip()
    if re.search(r"\b(hi|hello|hey|everyone|thanks|please|good (morning|evening|afternoon))\b", lowered):
        return "en"
    if re.search(r"\b(salom|assalomu alaykum|rahmat|iltimos)\b", lowered):
        return "uz"
    if re.search(r"\b(privet|spasibo|pozhaluysta|zdravstvu(y|j)?(te)?)\b", lowered):
        return "ru"
    en_hits = sum(1 for w in (" the ", " and ", " is ", " to ", " for ", " with ") if w in f" {lowered} ")
    if en_hits >= 1:
        return "en"
    return "uz"


def _heuristic_comment(text: str) -> str:
    lang = _detect_lang(text)
    t = text.strip()
    has_q = "?" in t
    h = int(hashlib.md5(t.encode("utf-8", "ignore")).hexdigest(), 16)
    def pick(arr):
        return arr[h % len(arr)] if arr else ""
    def extract_keywords(tx: str, l: str) -> list[str]:
        sw_en = {"the","and","with","this","that","from","into","about","your","our","their","have","has","had","for","are","was","were","will","would","could","should","not","you","they","than","then","when","what","which","where","how","why","there","here","such","more","less","very","just","like","also","been","being","over","under","after","before","on","in","at","to","of","as","by","or","an","a","is"}
        sw_ru = {"и","в","во","на","с","со","как","что","это","к","по","от","за","для","из","же","ли","но","а","не","ни","мы","вы","они","оно","она","его","ее","их","у","над","под","при","через","после","перед","без","про","или","так","же","бы","если","то","там","тут","все","есть","был","были","будет","будут","может"}
        sw_uz = {"va","ham","bilan","uchun","bu","shu","siz","biz","ular","undan","emas","yoʻq","yoq","bor","yoʻq","edi","deb","qilib","qildi","qilgan","qilinadi","bunda","shunda","shuningdek","yana","koʻproq","koproq","kamroq","qachon","nima","qanday","qaerda","nega","bo‘ladi","bo'ladi","ekan","bo‘lsa","bo'lsa","lekin","ammo","yoki"}
        sw = sw_en if l == "en" else (sw_ru if l == "ru" else sw_uz)
        tokens = re.findall(r"[\w'’]+", tx.lower())
        freq = {}
        for tok in tokens:
            if len(tok) < 4:
                continue
            if tok in sw:
                continue
            freq[tok] = freq.get(tok, 0) + 1
        top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
        return [k for k,_ in top]
    kws = extract_keywords(t, lang)
    if lang == "ru":
        qs = [
            "Хороший вопрос. Что здесь ключевое, как считаете?",
            "Есть смысл. Какие риски видите?",
            "Логично звучит. Что бы вы добавили?",
            "Дельная мысль. Какой следующий шаг?",
        ]
        st = [
            "Мысль здравая, спасибо.",
            "Хорошая подача — коротко и по делу.",
            "Содержательно; основная идея ясна.",
            "По сути. Акценты расставлены верно.",
        ]
        return pick(qs if has_q else st)
    if lang == "en":
        qs = [
            "Good question. What’s the key factor?",
            "Makes sense. Any real examples?",
            "On point. What’s the next step?",
            "Interesting—what’s the main risk?",
        ]
        st = [
            "Clear and relevant point.",
            "Concise and on point—thanks.",
            "Sounds logical and practical.",
            "Good take; core idea is clear.",
        ]
        return pick(qs if has_q else st)
    qs = [
        "Savol o‘rinli. Qaysi jihati eng muhim deb o‘ylaysiz?",
        "Mantiqli savol. Amaliy misollar bormi?",
        "Mazmunli savol. Keyingi qadam qanday bo‘lishi mumkin?",
        "Qiziqarli. Bu yerda asosiy risk nima?",
    ]
    st = [
        "Fikr aniq va dolzarb.",
        "Qisqa va mazmunli — rahmat.",
        "Mantiqiy yondashuv, amaliyga yaqin.",
        "Asosiy g‘oya ravshan ko‘rinmoqda.",
    ]
    return pick(qs if has_q else st)


def _heuristic_chat_reply(history: list[str], new_text: str) -> str:
    lang = _detect_lang("\n".join(history + [new_text]))
    t = new_text.strip()
    has_q = "?" in t
    if _is_greeting(new_text):
        return _greeting_reply(lang)
    if lang == "ru":
        return ("Уточните, пожалуйста: о чём именно речь?" if has_q else "Понял, спасибо за пояснение.")
    if lang == "en":
        return ("Could you clarify which part you mean?" if has_q else "Got it, thanks for the note.")
    return ("Aniqroq aytsangiz: qaysi jihati nazarda tutilgan?" if has_q else "Tushundim, rahmat.")


def _clean_text(s: str) -> str:
    s = re.sub(r"\*\*|`|^[-*#]+\s+", "", s)
    s = s.strip().strip('"').strip("'")
    return s.strip()


def _enforce_bot_name(s: str) -> str:
    try:
        name = (os.getenv("BOT_NAME") or "Alex").strip()
        if not name:
            return s
        out = re.sub(r"(?i)\bgoogle\s+assistant\b", name, s)
        return _clean_text(out)
    except Exception:
        return s


def _sanitize_output(raw: str, lang: str) -> str:
    lines = [l.strip() for l in (raw or "").splitlines() if l.strip()]
    if not lines:
        return ""
    head_pat = re.compile(r"(?i)^(o['’]?zbekcha|uzbek|ruscha|russian|inglizcha|english)\s*:\s*$")
    sections = {}
    cur = None
    for l in lines:
        if head_pat.match(l):
            key = head_pat.match(l).group(1).lower()
            cur = key
            sections.setdefault(cur, [])
            continue
        if cur is not None:
            sections[cur].append(l)
    target_keys = {
        "uz": ["o'zbekcha", "ozbekcha", "uzbek"],
        "ru": ["ruscha", "russian"],
        "en": ["inglizcha", "english"],
    }.get(lang, ["o'zbekcha", "ozbekcha", "uzbek"])
    if sections:
        for k in target_keys:
            if k in sections and sections[k]:
                cand = _clean_text(" ".join(sections[k]))
                return _enforce_bot_name(cand)
        all_text = []
        for arr in sections.values():
            all_text.extend(arr)
        return _enforce_bot_name(_clean_text(" ".join(all_text)))
    joined = " ".join(lines)
    meta_pat = re.compile(
        r"(?i)^(?:"
        r"here\s+is\s+(?:the\s+)?(?:requested\s+)?json|"
        r"here'?s\s+(?:the\s+)?(?:requested\s+)?json|"
        r"here\s+is\s+(?:the\s+)?json\s+requested|"
        r"here'?s\s+(?:the\s+)?json\s+requested|"
        r"this\s+is\s+(?:a\s+)?json(?:\s+object)?(?:\s*\(javascript\s+object\s+notation\))?\b|"
        r"json\s*\(javascript\s+object\s+notation\)\b|"
        r"bu\s+json\s+obyekt\w*\b|"
        r"requeste\b|"
        r"requested(?:\s+\w+){0,3}\b|"
        r"request(?:\s+\w+){0,3}\b|"
        r"response(?:\s+\w+){0,3}\b|"
        r"reply(?:\s+\w+){0,3}\b|"
        r"\bre\b|"
        r"answer(?:\s+\w+){0,3}\b|"
        r"output(?:\s+\w+){0,3}\b|"
        r"result(?:\s+\w+){0,3}\b|"
        r"json\b"
        r")\s*[:;\.\-—–：]*\s*"
    )
    while True:
        new_joined = meta_pat.sub("", joined).strip()
        if new_joined == joined:
            break
        joined = new_joined
    joined = _clean_text(joined)
    return _enforce_bot_name(joined)


def _extract_json(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start : end + 1])
    except Exception:
        return None
    return None


def _overlap_score(a: str, b: str) -> float:
    ta = set(t for t in re.findall(r"[\w'’]+", a.lower()) if len(t) >= 3)
    tb = set(t for t in re.findall(r"[\w'’]+", b.lower()) if len(t) >= 3)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    base = max(1, min(len(ta), len(tb)))
    return inter / base


def _is_greeting(text: str) -> bool:
    t = text.lower()
    return bool(
        re.search(r"\b(hi|hello|hey|everyone|good (morning|evening|afternoon))\b", t)
        or re.search(r"\b(салют|привет|здравствуйте)\b", t)
        or re.search(r"\b(salom|assalomu alaykum)\b", t)
    )


def _greeting_reply(lang: str) -> str:
    if lang == "ru":
        return "Привет! Рады видеть обновления."
    if lang == "en":
        return "Hello! Thanks for the update."
    return "Salom! Yangilik uchun rahmat."


async def generate_comment(text: str) -> Optional[str]:
    lang = _detect_lang(text)
    bn = (os.getenv("BOT_NAME") or "Alex").strip()
    prompt = (
        "Sizdan faqat JSON obyektini qaytarish talab qilinadi. Hech qanday matn, izoh yoki markdown yo‘q. \n"
        "Postga qisqa va lo‘nda, ortiqcha gaplarsiz, mantiqli izoh yozing. Uzunligi qat’iy emas: kerak bo‘lsa bitta so‘z, kerak bo‘lsa bir nechta gap. Javob post tilida bo‘lsin (lang_hint). \n"
        "Agar mazmuni juda qisqa yoki mavzuga bog‘lash qiyin bo‘lsa, relevance ni 0.0–0.4 atrofida baholang. \n"
        f"Sizning ismingiz '{bn}'. Ismingiz yoki kimligingiz so‘ralsa, faqat shu nomni ayting; 'Google Assistant' demang.\n"
        "Agar JSON formatini bera olmasangiz: faqat javob matnini qaytaring. Sarlavha, 'json', 'reply', 'response' kabi so‘zlarsiz, bitta til va qisqa.\n"
        "JSON schema: {\n"
        "  \"reply\": string,        // faqat post tilida, qisqa va lo‘nda\n"
        "  \"relevance\": number,    // 0.0..1.0, javobning postga mosligi\n"
        "  \"lang\": string,         // uz | ru | en\n"
        "  \"reason\": string        // qisqa izoh\n"
        "}\n\n"
        "Faqat bitta JSON obyektini qaytaring. Tashqarida hech qanday matn/izoh bo‘lmasin. 'Here is...', 'Requested:', 'Response:' kabi sarlavhalar bo‘lmasin. Javob '{' bilan boshlansin va '}' bilan tugasin.\n"
        f"lang_hint: {lang}\n"
        f"post:\n{text}"
    )

    def _call():
        try:
            chain = MODEL_CHAIN
            for idx, (name, m) in enumerate(chain):
                attempt = 0
                while attempt < 3:
                    try:
                        resp = m.generate_content(prompt)
                        out = getattr(resp, "text", None)
                        if out:
                            out = out.strip()
                            logging.info("gemini model used: %s", name)
                            return out
                        break
                    except NotFound:
                        logging.info("gemini model not found: %s", name)
                        break
                    except Exception as e:
                        logging.info("gemini model failed: %s [%s]", e, name)
                        s = str(e) or ""
                        if ("429" in s) or ("retry_delay" in s.lower()) or ("please retry in" in s.lower()) or ("quota exceeded" in s.lower()):
                            # Detect daily quota errors: stop retrying this model, but try next model in chain
                            if ("GenerateRequestsPerDay" in s) or ("PerDay" in s) or ("per day" in s.lower()):
                                logging.info("gemini daily quota exceeded: skipping model [%s]", name)
                                break
                            if GEMINI_FAST_SWITCH and idx < (len(chain) - 1):
                                logging.info("gemini fast-switch: skip waiting and try next model after 429 [%s]", name)
                                break
                            d = _parse_retry_delay_secs(e) or 12.0
                            try:
                                d = max(1.0, min(90.0, float(d)))
                            except Exception:
                                d = 12.0
                            logging.info("gemini rate limit: backoff %.1fs (attempt %s/3) [%s]", d, attempt + 1, name)
                            time.sleep(d)
                            attempt += 1
                            continue
                        break
            return None
        except Exception as e:
            logging.exception("Gemini call failed: %s", e)
            return None

    out = await asyncio.to_thread(_call)
    if out:
        obj = _extract_json(out)
        if obj and isinstance(obj, dict):
            reply = _clean_text(str(obj.get("reply", "")))
            if not reply:
                for k in ("text", "message", "content", "answer", "output"):
                    v = obj.get(k)
                    if v:
                        reply = _clean_text(str(v))
                        logging.info("gemini comment used alt key: %s", k)
                        break
            rel = float(obj.get("relevance", 0) or 0)
            lang_out = str(obj.get("lang", lang) or lang).lower()
            if not reply:
                logging.info("skip: gemini returned empty reply")
                # fall through to try raw text below
            else:
                final = _sanitize_output(reply, lang_out)
                # Do not block on relevance in Gemini-only mode; log for observability only
                logging.info("gemini comment relevance=%.2f overlap=%.2f", rel, _overlap_score(text, final))
                meta_only_pat = re.compile(r"(?i)^(?:requeste|requested|request|response|reply|re|output|result|json)\s*[:;\.-—–：]*$")
                if (not final) or meta_only_pat.match(final):
                    logging.info("skip: comment sanitized empty or meta-only content")
                    return None
                if len(final) > 400:
                    final = final[:400].rstrip() + "…"
                return final
        # If JSON was missing or empty reply, try using raw Gemini text as-is
        m = re.search(r'(?is)"reply"\s*:\s*"(.*?)"', out)
        if not m:
            m = re.search(r"(?is)'reply'\s*:\s*'(.*?)'", out)
        if m:
            cand = _clean_text(m.group(1).replace('\\"', '"'))
            if cand:
                final = _sanitize_output(cand, lang)
                logging.info("gemini comment reply extracted from raw JSON-like text")
                meta_only_pat = re.compile(r"(?i)^(?:requeste|requested|request|response|reply|re|output|result|json)\s*[:;\.-—–：]*$")
                if (not final) or meta_only_pat.match(final):
                    logging.info("skip: comment meta-only after JSON-like extract")
                    return None
                if len(final) > 400:
                    final = final[:400].rstrip() + "…"
                return final
        # If output is JSON-like but no usable reply, try to salvage a clean text
        if re.search(r'(?is)"reply"\s*:', out) or re.search(r'(?s)^\s*\{', out):
            raw0 = _clean_text(out)
            tmp = re.sub(r'(?is)"?(reply|relevance|lang|reason)"?\s*:\s*', '', raw0)
            tmp = re.sub(r'[{}\[\],]', ' ', tmp)
            final = _sanitize_output(tmp, lang).strip()
            if final:
                meta_only_pat = re.compile(r"(?i)^(?:requeste|requested|request|response|reply|re|output|result|json)\s*[:;\.-—–：]*$")
                if meta_only_pat.match(final):
                    logging.info("skip: comment meta-only after salvage")
                    return None
                if len(final) > 400:
                    final = final[:400].rstrip() + "…"
                logging.info("gemini comment salvaged from JSON-like text without reply")
                return final
            # Second-pass salvage: pick the first substantive non-meta line from raw JSON-like text
            for _line in (raw0 or "").splitlines():
                _line = _line.strip()
                if not _line:
                    continue
                meta_line = re.compile(r"(?i)^(?:here\s+is\s+(?:the\s+)?(?:requested\s+)?json|here'?s\s+(?:the\s+)?(?:requested\s+)?json|here\s+is\s+(?:the\s+)?json\s+requested|here'?s\s+(?:the\s+)?json\s+requested|this\s+is\s+(?:a\s+)?json(?:\s+object)?(?:\s*\(javascript\s+object\s+notation\))?|bu\s+json\s+obyekt\w*|json\s*\(javascript\s+object\s+notation\)|requeste\b|requested\b|request\b|response\b|reply\b|\bre\b|answer\b|output\b|result\b|json\b)\s*[:;\.-—–：]*$")
                if meta_line.match(_line):
                    continue
                if re.match(r"(?i)^(o['’]?zbekcha|uzbek|ruscha|russian|inglizcha|english)\s*:\s*$", _line):
                    continue
                final = _clean_text(_line)
                break
            if final:
                meta_only_pat = re.compile(r"(?i)^(?:requeste|requested|request|response|reply|re|output|result|json)\s*[:;\.-—–：]*$")
                if meta_only_pat.match(final):
                    logging.info("skip: comment meta-only after second-pass salvage")
                    return None
                if len(final) > 400:
                    final = final[:400].rstrip() + "…"
                logging.info("gemini comment salvaged by first substantive line from JSON-like text")
                return final
            logging.info("skip: gemini JSON-like text had no salvageable content (comment)")
            return None
        raw = _clean_text(out)
        if raw:
            final = _sanitize_output(raw, lang)
            logging.info("gemini comment using raw text (no JSON)")
            meta_only_pat = re.compile(r"(?i)^(?:requeste|requested|request|response|reply|re|output|result|json)\s*[:;\.-—–：]*$")
            if (not final) or meta_only_pat.match(final):
                logging.info("skip: comment meta-only after raw sanitize")
                return None
            if len(final) > 400:
                final = final[:400].rstrip() + "…"
            return final
    return None


CONV_MEM = {}
ROOT_INDEX = {}
TS_INDEX = {}
HIST_MAX_ENV = os.getenv("MEMORY_HIST_MAX")
try:
    HIST_MAX = max(4, int(HIST_MAX_ENV)) if HIST_MAX_ENV else 16
except Exception:
    HIST_MAX = 16
MEMORY_MAX_THREADS_ENV = os.getenv("MEMORY_MAX_THREADS")
try:
    MEMORY_MAX_THREADS = max(20, int(MEMORY_MAX_THREADS_ENV)) if MEMORY_MAX_THREADS_ENV else 200
except Exception:
    MEMORY_MAX_THREADS = 200
MEMORY_FILE_ENV = os.getenv("MEMORY_FILE")
DEFAULT_MEMORY_PATH = os.path.join(os.path.dirname(__file__), "memory.json")
MEMORY_PATH = (
    MEMORY_FILE_ENV if (MEMORY_FILE_ENV and os.path.isabs(MEMORY_FILE_ENV)) else (
        os.path.join(os.path.dirname(__file__), MEMORY_FILE_ENV) if MEMORY_FILE_ENV else DEFAULT_MEMORY_PATH
    )
)
MEMORY_BACKEND = (os.getenv("MEMORY_BACKEND") or "json").strip().lower()
MEMORY_DB_ENV = os.getenv("MEMORY_DB_FILE")
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "memory.db")
DB_PATH = (
    MEMORY_DB_ENV if (MEMORY_DB_ENV and os.path.isabs(MEMORY_DB_ENV)) else (
        os.path.join(os.path.dirname(__file__), MEMORY_DB_ENV) if MEMORY_DB_ENV else DEFAULT_DB_PATH
    )
)


def _db_get_conn() -> sqlite3.Connection:
    conn = getattr(_db_get_conn, "_conn", None)
    if conn is None:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conv_mem (chat_id INTEGER, root_id INTEGER, seq INTEGER, text TEXT, PRIMARY KEY(chat_id, root_id, seq))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS root_index (chat_id INTEGER, msg_id INTEGER, root_id INTEGER, PRIMARY KEY(chat_id, msg_id))"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ts_index (chat_id INTEGER, root_id INTEGER, ts REAL, PRIMARY KEY(chat_id, root_id))"
        )
        _db_get_conn._conn = conn
    return conn


def _db_full_save() -> None:
    try:
        conn = _db_get_conn()
        cur = conn.cursor()
        cur.execute("BEGIN")
        cur.execute("DELETE FROM conv_mem")
        cur.execute("DELETE FROM root_index")
        cur.execute("DELETE FROM ts_index")
        for (chat_id, root_id), hist in CONV_MEM.items():
            seq0 = max(0, len(hist) - HIST_MAX)
            for seq, line in enumerate(hist[seq0:], start=0):
                cur.execute(
                    "INSERT OR REPLACE INTO conv_mem(chat_id, root_id, seq, text) VALUES (?,?,?,?)",
                    (chat_id, root_id, seq, line),
                )
        for (chat_id, msg_id), root_id in ROOT_INDEX.items():
            cur.execute(
                "INSERT OR REPLACE INTO root_index(chat_id, msg_id, root_id) VALUES (?,?,?)",
                (chat_id, msg_id, root_id),
            )
        for (chat_id, root_id), ts in TS_INDEX.items():
            cur.execute(
                "INSERT OR REPLACE INTO ts_index(chat_id, root_id, ts) VALUES (?,?,?)",
                (chat_id, root_id, float(ts)),
            )
        conn.commit()
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        logging.exception("DB full save failed: %s", e)


def _db_full_load() -> None:
    try:
        conn = _db_get_conn()
        cur = conn.cursor()
        conv = {}
        for chat_id, root_id, seq, text in cur.execute(
            "SELECT chat_id, root_id, seq, text FROM conv_mem ORDER BY chat_id, root_id, seq"
        ):
            key = (int(chat_id), int(root_id))
            lst = conv.get(key, [])
            lst.append(text)
            conv[key] = lst
        idx = {}
        for chat_id, msg_id, root_id in cur.execute(
            "SELECT chat_id, msg_id, root_id FROM root_index"
        ):
            idx[(int(chat_id), int(msg_id))] = int(root_id)
        ts = {}
        for chat_id, root_id, tsval in cur.execute(
            "SELECT chat_id, root_id, ts FROM ts_index"
        ):
            ts[(int(chat_id), int(root_id))] = float(tsval)
        for k, v in conv.items():
            if len(v) > HIST_MAX:
                conv[k] = v[-HIST_MAX:]
        CONV_MEM.clear(); CONV_MEM.update(conv)
        ROOT_INDEX.clear(); ROOT_INDEX.update(idx)
        TS_INDEX.clear(); TS_INDEX.update(ts)
    except Exception as e:
        logging.exception("DB full load failed: %s", e)


def _save_persistent() -> None:
    try:
        if MEMORY_BACKEND == "sqlite":
            _db_full_save()
        else:
            data = {
                "conv_mem": {f"{k[0]}:{k[1]}": v for k, v in CONV_MEM.items()},
                "root_index": {f"{k[0]}:{k[1]}": v for k, v in ROOT_INDEX.items()},
                "ts_index": {f"{k[0]}:{k[1]}": float(v) for k, v in TS_INDEX.items()},
            }
            with open(MEMORY_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logging.exception("Persist save failed: %s", e)


def _load_persistent() -> None:
    try:
        if MEMORY_BACKEND == "sqlite":
            _db_full_load()
        else:
            if not os.path.exists(MEMORY_PATH):
                return
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            conv = {}
            for sk, v in (data.get("conv_mem") or {}).items():
                try:
                    a, b = sk.split(":", 1)
                    lst = list(v)
                    if len(lst) > HIST_MAX:
                        lst = lst[-HIST_MAX:]
                    conv[(int(a), int(b))] = lst
                except Exception:
                    continue
            idx = {}
            for sk, v in (data.get("root_index") or {}).items():
                try:
                    a, b = sk.split(":", 1)
                    idx[(int(a), int(b))] = int(v)
                except Exception:
                    continue
            ts = {}
            for sk, v in (data.get("ts_index") or {}).items():
                try:
                    a, b = sk.split(":", 1)
                    ts[(int(a), int(b))] = float(v)
                except Exception:
                    continue
            CONV_MEM.clear(); CONV_MEM.update(conv)
            ROOT_INDEX.clear(); ROOT_INDEX.update(idx)
            TS_INDEX.clear(); TS_INDEX.update(ts)
    except Exception as e:
        logging.exception("Persist load failed: %s", e)


_load_persistent()


def _touch_root(root_key: tuple[int, int]) -> None:
    try:
        TS_INDEX[root_key] = time.time()
    except Exception:
        TS_INDEX[root_key] = float(int(time.time()))


def _prune_threads_if_needed() -> None:
    try:
        total = len(CONV_MEM)
        if total <= MEMORY_MAX_THREADS:
            return
        remove_n = total - MEMORY_MAX_THREADS
        sorted_keys = sorted(CONV_MEM.keys(), key=lambda k: TS_INDEX.get(k, 0.0))
        to_remove = sorted_keys[:remove_n]
        removed = 0
        for rk in to_remove:
            chat_id, root_id = rk
            CONV_MEM.pop(rk, None)
            TS_INDEX.pop(rk, None)
            for k in list(ROOT_INDEX.keys()):
                if k[0] == chat_id and ROOT_INDEX.get(k) == root_id:
                    ROOT_INDEX.pop(k, None)
            removed += 1
        if removed:
            logging.info("memory pruned: %s threads removed; keep=%s", removed, len(CONV_MEM))
    except Exception as e:
        logging.exception("Prune failed: %s", e)


async def generate_chat_reply(history: list[str], new_text: str) -> Optional[str]:
    lang = _detect_lang("\n".join(history + [new_text]))
    bn = (os.getenv("BOT_NAME") or "Alex").strip()
    prompt = (
        "Sizdan faqat JSON obyektini qaytarish talab qilinadi. Hech qanday matn yoki markdown yo‘q. \n"
        "Suhbat kontekstiga tayanib, qisqa va lo‘nda javob yozing. Uzunligi qat’iy emas: kerak bo‘lsa bitta so‘z, kerak bo‘lsa bir nechta gap. Javob oxirgi xabar tilida bo‘lsin (lang_hint). \n"
        f"Sizning ismingiz '{bn}'. Ismingiz yoki kimligingiz so‘ralsa, faqat shu nomni ayting; 'Google Assistant' demang.\n"
        "Agar JSON formatini bera olmasangiz: faqat javob matnini qaytaring. Sarlavha, 'json', 'reply', 'response' kabi so‘zlarsiz, bitta til va qisqa.\n"
        "JSON schema: {\n"
        "  \"reply\": string,\n  \"relevance\": number,\n  \"lang\": string,\n  \"reason\": string\n} \n\n"
        "Faqat bitta JSON obyektini qaytaring. Tashqarida hech qanday matn/izoh bo‘lmasin. 'Here is...', 'Requested:', 'Response:' kabi sarlavhalar bo‘lmasin. Javob '{' bilan boshlansin va '}' bilan tugasin.\n"
        f"lang_hint: {lang}\n"
        "history:\n"
        + "\n".join(f"- {line.strip()}" for line in history[-HIST_MAX:])
        + "\n\nnew_message:\n"
        + new_text.strip()
    )

    def _call():
        try:
            chain = MODEL_CHAIN
            for idx, (name, m) in enumerate(chain):
                attempt = 0
                while attempt < 3:
                    try:
                        resp = m.generate_content(prompt)
                        out = getattr(resp, "text", None)
                        if out:
                            out = out.strip()
                            logging.info("gemini model used: %s", name)
                            return out
                        break
                    except NotFound:
                        logging.info("gemini model not found: %s", name)
                        break
                    except Exception as e:
                        logging.info("gemini model failed: %s [%s]", e, name)
                        s = str(e) or ""
                        if ("429" in s) or ("retry_delay" in s.lower()) or ("please retry in" in s.lower()) or ("quota exceeded" in s.lower()):
                            # Detect daily quota errors: stop retrying this model, but try next model in chain
                            if ("GenerateRequestsPerDay" in s) or ("PerDay" in s) or ("per day" in s.lower()):
                                logging.info("gemini daily quota exceeded: skipping model [%s]", name)
                                break
                            if GEMINI_FAST_SWITCH and idx < (len(chain) - 1):
                                logging.info("gemini fast-switch: skip waiting and try next model after 429 [%s]", name)
                                break
                            d = _parse_retry_delay_secs(e) or 12.0
                            try:
                                d = max(1.0, min(90.0, float(d)))
                            except Exception:
                                d = 12.0
                            logging.info("gemini rate limit: backoff %.1fs (attempt %s/3) [%s]", d, attempt + 1, name)
                            time.sleep(d)
                            attempt += 1
                            continue
                        break
            return None
        except Exception as e:
            logging.exception("Gemini call failed: %s", e)
            return None

    out = await asyncio.to_thread(_call)
    if out:
        obj = _extract_json(out)
        if obj and isinstance(obj, dict):
            reply = _clean_text(str(obj.get("reply", "")))
            if not reply:
                for k in ("text", "message", "content", "answer", "output"):
                    v = obj.get(k)
                    if v:
                        reply = _clean_text(str(v))
                        logging.info("gemini chat used alt key: %s", k)
                        break
            rel = float(obj.get("relevance", 0) or 0)
            lang_out = str(obj.get("lang", lang) or lang).lower()
            if not reply:
                logging.info("skip: gemini chat returned empty reply")
                # fall through to try raw text below
            else:
                final = _sanitize_output(reply, lang_out)
                # Log relevance vs context, but do not block sending in Gemini-only mode
                hist_text = "\n".join(history[-HIST_MAX:])
                score_hist = _overlap_score(hist_text, final)
                score_new = _overlap_score(new_text, final)
                logging.info("gemini chat relevance=%.2f overlap_hist=%.2f overlap_new=%.2f", rel, score_hist, score_new)
                meta_only_pat = re.compile(r"(?i)^(?:requeste|requested|request|response|reply|re|output|result|json)\s*[:;\.-—–：]*$")
                if (not final) or meta_only_pat.match(final):
                    logging.info("skip: chat meta-only content after sanitize")
                    return None
                if len(final) > 400:
                    final = final[:400].rstrip() + "…"
                return final
        # If JSON missing or empty, use raw Gemini text
        m = re.search(r'(?is)"reply"\s*:\s*"(.*?)"', out)
        if not m:
            m = re.search(r"(?is)'reply'\s*:\s*'(.*?)'", out)
        if m:
            cand = _clean_text(m.group(1).replace('\\"', '"'))
            if cand:
                final = _sanitize_output(cand, lang)
                logging.info("gemini chat reply extracted from raw JSON-like text")
                meta_only_pat = re.compile(r"(?i)^(?:requeste|requested|request|response|reply|re|output|result|json)\s*[:;\.-—–：]*$")
                if (not final) or meta_only_pat.match(final):
                    logging.info("skip: chat meta-only after JSON-like extract")
                    return None
                if len(final) > 400:
                    final = final[:400].rstrip() + "…"
                return final
        # If output is JSON-like but has no usable reply, try to salvage a clean text
        if re.search(r'(?is)"reply"\s*:', out) or re.search(r'(?s)^\s*\{', out):
            raw0 = _clean_text(out)
            tmp = re.sub(r'(?is)"?(reply|relevance|lang|reason)"?\s*:\s*', '', raw0)
            tmp = re.sub(r'[{}\[\],]', ' ', tmp)
            final = _sanitize_output(tmp, lang).strip()
            if final:
                meta_only_pat = re.compile(r"(?i)^(?:requeste|requested|request|response|reply|re|output|result|json)\s*[:;\.-—–：]*$")
                if meta_only_pat.match(final):
                    logging.info("skip: chat meta-only after salvage")
                    return None
                if len(final) > 400:
                    final = final[:400].rstrip() + "…"
                logging.info("gemini chat salvaged from JSON-like text without reply")
                return final
            # Second-pass salvage: pick the first substantive non-meta line from raw JSON-like text
            for _line in (raw0 or "").splitlines():
                _line = _line.strip()
                if not _line:
                    continue
                meta_line = re.compile(r"(?i)^(?:here\s+is\s+(?:the\s+)?(?:requested\s+)?json|here'?s\s+(?:the\s+)?(?:requested\s+)?json|here\s+is\s+(?:the\s+)?json\s+requested|here'?s\s+(?:the\s+)?json\s+requested|this\s+is\s+(?:a\s+)?json(?:\s+object)?(?:\s*\(javascript\s+object\s+notation\))?|bu\s+json\s+obyekt\w*|json\s*\(javascript\s+object\s+notation\)|requeste\b|requested\b|request\b|response\b|reply\b|\bre\b|answer\b|output\b|result\b|json\b)\s*[:;\.-—–：]*$")
                if meta_line.match(_line):
                    continue
                if re.match(r"(?i)^(o['’]?zbekcha|uzbek|ruscha|russian|inglizcha|english)\s*:\s*$", _line):
                    continue
                final = _clean_text(_line)
                break
            if final:
                if len(final) > 400:
                    final = final[:400].rstrip() + "…"
                logging.info("gemini chat salvaged by first substantive line from JSON-like text")
                return final
            logging.info("skip: gemini JSON-like text had no salvageable content (chat)")
            return None
        raw = _clean_text(out)
        if raw:
            final = _sanitize_output(raw, lang)
            if not final:
                for _line in (raw or "").splitlines():
                    _line = _line.strip()
                    if not _line:
                        continue
                    meta_line = re.compile(r"(?i)^(?:here\s+is\s+(?:the\s+)?(?:requested\s+)?json|here'?s\s+(?:the\s+)?(?:requested\s+)?json|here\s+is\s+(?:the\s+)?json\s+requested|here'?s\s+(?:the\s+)?json\s+requested|this\s+is\s+(?:a\s+)?json(?:\s+object)?(?:\s*\(javascript\s+object\s+notation\))?|bu\s+json\s+obyekt\w*|json\s*\(javascript\s+object\s+notation\)|requeste\b|requested\b|request\b|response\b|reply\b|\bre\b|answer\b|output\b|result\b|json\b)\s*[:;\.-—–：]*$")
                    if meta_line.match(_line):
                        continue
                    if re.match(r"(?i)^(o['’]?zbekcha|uzbek|ruscha|russian|inglizcha|english)\s*:\s*$", _line):
                        continue
                    final = _line
                    break
                final = _clean_text(final or "")
            logging.info("gemini chat using raw text (no JSON)")
            meta_only_pat = re.compile(r"(?i)^(?:requeste|requested|request|response|reply|re|output|result|json)\s*[:;\.-—–：]*$")
            if (not final) or meta_only_pat.match(final):
                logging.info("skip: chat meta-only after raw sanitize")
                return None
            if len(final) > 400:
                final = final[:400].rstrip() + "…"
            return final
    return None


async def on_group_auto_forward(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    if not msg:
        return
    is_auto = bool(getattr(msg, "is_automatic_forward", False))
    fwd_from = getattr(msg, "forward_from_chat", None)
    sender_chat = getattr(msg, "sender_chat", None)
    chat = getattr(msg, "chat", None)
    thread_id = getattr(msg, "message_thread_id", None)
    bot_un = await _ensure_bot_username(context)
    if not bot_un:
        bot_un = BOT_USERNAME_STATIC
    is_mention = _message_mentions_bot(msg, bot_un) if bot_un else False
    logging.info("bot username resolved: @%s mention=%s", bot_un, is_mention)
    if not is_mention and (getattr(msg, "text", None) or getattr(msg, "caption", None)):
        try:
            ents_dbg = []
            for e in (getattr(msg, "entities", []) or []) + (getattr(msg, "caption_entities", []) or []):
                ents_dbg.append(str(getattr(e, "type", None)))
            logging.info("mention miss debug: text=%r entities=%s", (getattr(msg, "text", None) or getattr(msg, "caption", None) or "")[:200], ents_dbg)
        except Exception:
            pass
    logging.info(
        "group msg: auto=%s sender_chat=%s sender_type=%s forward_from_chat=%s fwd_type=%s chat_id=%s chat_type=%s thread=%s has_text=%s has_caption=%s",
        is_auto,
        getattr(sender_chat, "id", None),
        getattr(sender_chat, "type", None),
        getattr(fwd_from, "id", None),
        getattr(fwd_from, "type", None),
        getattr(chat, "id", None),
        getattr(chat, "type", None),
        thread_id,
        bool(msg.text),
        bool(msg.caption),
    )

    is_channel_origin = (
        bool(is_auto)
        or bool(fwd_from and fwd_from.type == "channel")
        or bool(sender_chat and sender_chat.type == "channel" and getattr(msg, "reply_to_message", None) is None)
        # In some discussion groups, channel posts appear without from_user and without reply_to
        or (getattr(msg, "from_user", None) is None and getattr(msg, "reply_to_message", None) is None)
    )

    logging.info(
        "branch: is_channel_origin=%s reply_to_exists=%s from_user_is_bot=%s",
        is_channel_origin,
        bool(getattr(msg, "reply_to_message", None)),
        bool(getattr(getattr(msg, "from_user", None), "is_bot", False)),
    )
    try:
        if _is_anonymous_author(msg) and not is_channel_origin and not is_mention and (getattr(msg, "text", None) or getattr(msg, "caption", None)):
            is_mention = True
            logging.info("force mention: anonymous author trigger")
    except Exception:
        pass
    if is_mention:
        logging.info("bot mentioned: @%s", bot_un)

    if is_channel_origin and not is_mention:
        source_channel_id = sender_chat.id if (sender_chat and sender_chat.type == "channel") else (
            fwd_from.id if (fwd_from and fwd_from.type == "channel") else None
        )
        # Only enforce target filtering if we can reliably detect the source channel id
        if TARGET_CHANNEL_ID is not None and (source_channel_id is not None) and source_channel_id != TARGET_CHANNEL_ID:
            logging.info("skip: channel id %s != target %s", source_channel_id, TARGET_CHANNEL_ID)
            return

        text = msg.text or msg.caption
        if not text:
            logging.info("skip: no text or caption to summarize")
            return

        comment = await generate_comment(text)
        if not comment:
            logging.info("skip: no comment generated")
            return

        root_key = (msg.chat_id, thread_id if thread_id is not None else msg.message_id)
        hist = CONV_MEM.get(root_key, [])
        hist.append(text)
        if len(hist) > HIST_MAX:
            hist = hist[-HIST_MAX:]
        CONV_MEM[root_key] = hist
        _touch_root(root_key)
        _prune_threads_if_needed()
        _save_persistent()

        try:
            sent = await msg.reply_text(comment)
            ROOT_INDEX[(msg.chat_id, sent.message_id)] = root_key[1]
            hist.append(comment)
            if len(hist) > HIST_MAX:
                hist = hist[-HIST_MAX:]
            CONV_MEM[root_key] = hist
            _touch_root(root_key)
            _save_persistent()
            # If Telegram assigned a thread/topic id to our sent message, mirror memory under that thread id too
            alias_tid = getattr(sent, "message_thread_id", None)
            if alias_tid is not None:
                alias_key = (msg.chat_id, alias_tid)
                if alias_key not in CONV_MEM:
                    CONV_MEM[alias_key] = list(hist)
                    _touch_root(alias_key)
                    _prune_threads_if_needed()
                    _save_persistent()
                    logging.info("aliased memory to thread: %s", alias_key)
            logging.info("sent comment via reply_text")
        except Exception as e:
            logging.exception("Failed to send via reply_text: %s", e)
            try:
                if thread_id is not None:
                    sent2 = await context.bot.send_message(chat_id=msg.chat_id, text=comment, message_thread_id=thread_id)
                else:
                    sent2 = await context.bot.send_message(chat_id=msg.chat_id, text=comment)
                ROOT_INDEX[(msg.chat_id, sent2.message_id)] = root_key[1]
                hist.append(comment)
                if len(hist) > HIST_MAX:
                    hist = hist[-HIST_MAX:]
                CONV_MEM[root_key] = hist
                _touch_root(root_key)
                _save_persistent()
                alias_tid2 = getattr(sent2, "message_thread_id", None)
                if alias_tid2 is not None:
                    alias_key2 = (msg.chat_id, alias_tid2)
                    if alias_key2 not in CONV_MEM:
                        CONV_MEM[alias_key2] = list(hist)
                        _touch_root(alias_key2)
                        _prune_threads_if_needed()
                        _save_persistent()
                        logging.info("aliased memory to thread: %s", alias_key2)
                logging.info("sent comment via send_message")
            except Exception as e2:
                logging.exception("Failed to send via send_message: %s", e2)
        return

    if getattr(msg, "from_user", None) and getattr(msg.from_user, "is_bot", False):
        fu_id = getattr(msg.from_user, "id", None)
        if MY_BOT_ID is not None and fu_id == MY_BOT_ID:
            logging.info("skip: own bot message; id=%s", fu_id)
            return
        # Another bot's message; do not auto-return so that human replies to bot content in thread still work
        logging.info("note: message from another bot id=%s; continue processing", fu_id)

    logging.info(
        "enter reply path: chat_id=%s thread=%s reply_to_exists=%s",
        getattr(chat, "id", None),
        thread_id,
        bool(getattr(msg, "reply_to_message", None)),
    )

    text = msg.text or msg.caption
    if not text:
        return
    if is_mention:
        try:
            stripped = _strip_bot_mention(text, bot_un)
            if stripped:
                text = stripped
                logging.info("mention text stripped for reply: %s", text)
        except Exception:
            pass

    root_key = None
    logging.info(
        "thread msg: chat_id=%s thread=%s reply_to=%s from_user=%s is_bot=%s",
        getattr(chat, "id", None),
        thread_id,
        getattr(getattr(msg, "reply_to_message", None), "message_id", None),
        getattr(getattr(msg, "from_user", None), "id", None),
        getattr(getattr(msg, "from_user", None), "is_bot", None),
    )
    if thread_id is not None and (msg.chat_id, thread_id) in CONV_MEM:
        root_key = (msg.chat_id, thread_id)
    elif getattr(msg, "reply_to_message", None) is not None:
        rt = msg.reply_to_message
        cand = (msg.chat_id, getattr(rt, "message_thread_id", None) if getattr(rt, "message_thread_id", None) is not None else rt.message_id)
        if cand in CONV_MEM:
            root_key = cand
        else:
            root_id = ROOT_INDEX.get((msg.chat_id, rt.message_id))
            if root_id is not None:
                root_key = (msg.chat_id, root_id)
            elif getattr(rt, "sender_chat", None) and getattr(rt.sender_chat, "type", None) == "channel":
                root_key = (msg.chat_id, rt.message_id)

    if not root_key:
        if thread_id is not None and (is_channel_origin or is_mention or getattr(msg, "reply_to_message", None) is not None):
            root_key = (msg.chat_id, thread_id)
            CONV_MEM.setdefault(root_key, [])
            _touch_root(root_key)
            _prune_threads_if_needed()
            _save_persistent()
            try:
                cause = "channel" if is_channel_origin else ("mention" if is_mention else "reply")
                logging.info("init new root by thread id: %s cause=%s", root_key, cause)
            except Exception:
                pass
        else:
            if getattr(msg, "reply_to_message", None) is not None:
                rt = msg.reply_to_message
                root_key = (msg.chat_id, rt.message_id)
                CONV_MEM.setdefault(root_key, [])
                _touch_root(root_key)
                _prune_threads_if_needed()
                _save_persistent()
                logging.info("init new root by reply_to message id: %s", root_key)
            elif is_mention:
                root_key = (msg.chat_id, -int(getattr(msg, "message_id", 0) or 0))
                CONV_MEM.setdefault(root_key, [])
                _touch_root(root_key)
                _prune_threads_if_needed()
                _save_persistent()
                logging.info("init new root by mention message id (user mem): %s", root_key)
            else:
                return

    # If we resolved to a root different from this message's thread, alias the memory under the thread id too
    if thread_id is not None and (msg.chat_id, thread_id) not in CONV_MEM and root_key[1] >= 0:
        CONV_MEM[(msg.chat_id, thread_id)] = list(CONV_MEM.get(root_key, []))
        _touch_root((msg.chat_id, thread_id))
        _prune_threads_if_needed()
        _save_persistent()
        logging.info("aliased memory to thread (reply path): %s", (msg.chat_id, thread_id))

    hist = CONV_MEM.get(root_key, [])
    logging.info("generating reply: hist_len=%s thread=%s", len(hist), thread_id)
    reply = await generate_chat_reply(hist, text)
    if not reply:
        logging.info("skip: no chat reply generated (Gemini-only mode)")
        # Record user text to keep context for future turns
        hist.append(text)
        if len(hist) > HIST_MAX:
            hist = hist[-HIST_MAX:]
        CONV_MEM[root_key] = hist
        _touch_root(root_key)
        _prune_threads_if_needed()
        _save_persistent()
        return
    hist.append(text)
    hist.append(reply)
    if len(hist) > HIST_MAX:
        hist = hist[-HIST_MAX:]
    CONV_MEM[root_key] = hist
    _touch_root(root_key)
    _prune_threads_if_needed()
    _save_persistent()

    if thread_id is not None:
        # Prefer sending to the thread directly to avoid reply binding quirks
        try:
            sent4 = await context.bot.send_message(
                chat_id=msg.chat_id,
                text=reply,
                message_thread_id=thread_id,
                reply_to_message_id=getattr(msg, "message_id", None),
            )
            ROOT_INDEX[(msg.chat_id, sent4.message_id)] = root_key[1]
            _touch_root(root_key)
            _save_persistent()
            alias_tid4 = getattr(sent4, "message_thread_id", None)
            if alias_tid4 is not None and root_key[1] >= 0:
                alias_key4 = (msg.chat_id, alias_tid4)
                if alias_key4 not in CONV_MEM:
                    CONV_MEM[alias_key4] = list(CONV_MEM.get(root_key, []))
                    _touch_root(alias_key4)
                    _prune_threads_if_needed()
                    _save_persistent()
                    logging.info("aliased memory to thread: %s", alias_key4)
            logging.info("sent thread reply via send_message (primary)")
        except Exception as e2:
            logging.exception("Failed primary send_message to thread: %s", e2)
            try:
                sent3 = await msg.reply_text(reply)
                ROOT_INDEX[(msg.chat_id, sent3.message_id)] = root_key[1]
                _touch_root(root_key)
                _save_persistent()
                alias_tid3 = getattr(sent3, "message_thread_id", None)
                if alias_tid3 is not None and root_key[1] >= 0:
                    alias_key3 = (msg.chat_id, alias_tid3)
                    if alias_key3 not in CONV_MEM:
                        CONV_MEM[alias_key3] = list(CONV_MEM.get(root_key, []))
                        _touch_root(alias_key3)
                        _prune_threads_if_needed()
                        _save_persistent()
                        logging.info("aliased memory to thread: %s", alias_key3)
                logging.info("sent thread reply via reply_text (fallback)")
            except Exception as e3:
                logging.exception("Failed to send thread reply via reply_text fallback: %s", e3)
    else:
        try:
            sent3 = await msg.reply_text(reply)
            ROOT_INDEX[(msg.chat_id, sent3.message_id)] = root_key[1]
            _touch_root(root_key)
            _save_persistent()
            alias_tid3 = getattr(sent3, "message_thread_id", None)
            if alias_tid3 is not None and root_key[1] >= 0:
                alias_key3 = (msg.chat_id, alias_tid3)
                if alias_key3 not in CONV_MEM:
                    CONV_MEM[alias_key3] = list(CONV_MEM.get(root_key, []))
                    _touch_root(alias_key3)
                    _prune_threads_if_needed()
                    _save_persistent()
                    logging.info("aliased memory to thread: %s", alias_key3)
            logging.info("sent thread reply via reply_text")
        except Exception as e:
            logging.exception("Failed to send thread reply (no thread id): %s", e)


def main() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.ChatType.GROUPS, on_group_auto_forward))
    if RUN_MODE == "WEBHOOK":
        path = WEBHOOK_PATH if WEBHOOK_PATH else TELEGRAM_BOT_TOKEN
        url = WEBHOOK_URL.rstrip("/") + "/" + path if WEBHOOK_URL else None
        app.run_webhook(
            listen=LISTEN,
            port=PORT,
            url_path=path,
            webhook_url=url,
            secret_token=WEBHOOK_SECRET_TOKEN,
            drop_pending_updates=True,
        )
    else:
        app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
