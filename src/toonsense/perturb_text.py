"""Controlled linguistic perturbations for Bangla / English / code-mixed meme text.

Families (each with severities 1..3):
  translit      Bangla script -> colloquial Romanised Bangla ("Banglish") for a fraction of words
  roman_noise   full Romanisation + Banglish spelling variation (phonetic swaps, vowel drop, typos)
  bangla_noise  Bangla-script orthographic confusions (i/ii, u/uu, sibilants, n/N, ...)
  codemix       lexicon-based Bangla<->English word substitution (code-mixing variation)
  emoji_add     insertion of sentiment-laden emojis
  emoji_remove  removal of emojis -> emoticons -> all decorative symbols / punctuation

All functions are pure and deterministic given (sample_id, family, severity).
"""
from __future__ import annotations

import random
import re
import unicodedata
import zlib

# --------------------------------------------------------------------------------------
# Bangla -> Banglish transliteration (colloquial social-media convention, not ISO/ITRANS)
# --------------------------------------------------------------------------------------
VOWELS = {
    "অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u", "ঋ": "ri",
    "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
}
SIGNS = {
    "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ৃ": "ri",
    "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou",
}
CONS = {
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "ch", "জ": "j", "ঝ": "jh", "ঞ": "n",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "f", "ব": "b", "ভ": "bh", "ম": "m",
    "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh", "স": "s", "হ": "h",
    "ড়": "r", "ঢ়": "rh", "য়": "y",
}
NUKTA_FORMS = {"ড": "r", "ঢ": "rh", "য": "y"}
OTHERS = {"ং": "ng", "ঃ": "h", "ঁ": "", "ৎ": "t", "।": ".", "॥": "."}
DIGITS = {chr(0x09E6 + i): str(i) for i in range(10)}
HASANTA = "্"
NUKTA = "়"

_BN_LETTER = re.compile(r"[ঀ-৿]")
_TOKEN = re.compile(r"\S+")


def is_bangla(s: str) -> bool:
    return bool(_BN_LETTER.search(s))


def bn_to_roman(text: str) -> str:
    """Transliterate Bangla script to colloquial Romanised Bangla; other characters pass through."""
    text = unicodedata.normalize("NFC", text)
    chars = list(text)
    n = len(chars)
    out: list[str] = []
    i = 0
    after_hasanta = False
    while i < n:
        c = chars[i]
        if c in CONS:
            r = CONS[c]
            j = i + 1
            if j < n and chars[j] == NUKTA:
                r = NUKTA_FORMS.get(c, r)
                j += 1
            if after_hasanta and c == "য":  # ya-phala
                r = "y"
            elif after_hasanta and c == "ব":  # ba-phala is silent colloquially (স্বাধীন -> shadhin)
                r = ""
            nxt = chars[j] if j < n else ""
            if nxt in SIGNS:
                r += SIGNS[nxt]
                j += 1
                after_hasanta = False
            elif nxt == HASANTA:
                j += 1
                after_hasanta = True
            else:
                after_hasanta = False
                word_end = not (j < n and _BN_LETTER.match(chars[j]) and chars[j] not in "।॥")
                word_start = i == 0 or not _BN_LETTER.match(chars[i - 1])
                # Inherent vowel with schwa deletion: dropped word-finally, and word-medially in a
                # VC_CV context, i.e. when the next consonant carries an explicit vowel sign
                # (লালবাজার -> lalbajar, কলকাতা -> kolkata, সরকার -> sorkar).
                next_has_sign = (
                    j + 1 < n and chars[j] in CONS and (chars[j + 1] in SIGNS or chars[j + 1] == NUKTA and j + 2 < n and chars[j + 2] in SIGNS)
                )
                if word_start or (not word_end and not next_has_sign):
                    r += "o"
            out.append(r)
            i = j
            continue
        after_hasanta = False
        if c in VOWELS:
            out.append(VOWELS[c])
        elif c in OTHERS:
            out.append(OTHERS[c])
        elif c in DIGITS:
            out.append(DIGITS[c])
        elif c in SIGNS:  # stray sign
            out.append(SIGNS[c])
        elif c in (HASANTA, NUKTA, "‌", "‍"):
            pass
        else:
            out.append(c)
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------------------
# Spelling-noise models
# --------------------------------------------------------------------------------------
# Banglish spelling variants observed in social-media Romanised Bangla.
ROMAN_SUBS = [
    ("sh", "s"), ("s", "sh"), ("chh", "ch"), ("ch", "c"), ("bh", "v"), ("ph", "f"), ("f", "ph"),
    ("th", "t"), ("dh", "d"), ("kh", "k"), ("gh", "g"), ("i", "ee"), ("u", "oo"), ("o", "u"),
    ("oi", "oy"), ("y", "i"), ("w", "o"), ("ks", "x"), ("j", "z"), ("e", "a"), ("ou", "ow"),
]
VOWEL_RE = re.compile(r"(?<=[^aeiou\W])[aeiou](?=[^aeiou\W])")


def _roman_word_noise(w: str, rng: random.Random) -> str:
    ops = ["sub", "sub", "drop_vowel", "repeat", "swap", "delete"]
    for _ in range(4):  # try a few ops until one applies
        op = rng.choice(ops)
        if op == "sub":
            cands = [(a, b) for a, b in ROMAN_SUBS if a in w.lower()]
            if cands:
                a, b = rng.choice(cands)
                idx = [m.start() for m in re.finditer(re.escape(a), w.lower())]
                k = rng.choice(idx)
                return w[:k] + b + w[k + len(a):]
        elif op == "drop_vowel":
            idx = [m.start() for m in VOWEL_RE.finditer(w.lower())]
            if idx:
                k = rng.choice(idx)
                return w[:k] + w[k + 1:]
        elif op == "repeat" and len(w) >= 2:
            k = rng.randrange(len(w))
            if w[k].isalpha():
                return w[: k + 1] + w[k] * rng.randint(1, 2) + w[k + 1:]
        elif op == "swap" and len(w) >= 4:
            k = rng.randrange(1, len(w) - 2)
            return w[:k] + w[k + 1] + w[k] + w[k + 2:]
        elif op == "delete" and len(w) >= 4:
            k = rng.randrange(1, len(w) - 1)
            return w[:k] + w[k + 1:]
    return w


# Bangla orthographic confusion sets (common misspellings in user-generated Bangla).
BN_CONFUSIONS = [
    ("ি", "ী"), ("ী", "ি"), ("ু", "ূ"), ("ূ", "ু"), ("শ", "স"), ("স", "শ"), ("ষ", "শ"),
    ("ন", "ণ"), ("ণ", "ন"), ("জ", "য"), ("য", "জ"), ("ড়", "র"), ("র", "ড়"), ("ং", "ঙ"),
    ("য়", "য"), ("ো", "ৌ"), ("ে", "ৈ"), ("ত", "ৎ"), ("ছ", "চ"), ("ধ", "দ"), ("ভ", "ব"), ("ঢ", "ড"),
]


def _bangla_word_noise(w: str, rng: random.Random) -> str:
    w = unicodedata.normalize("NFC", w)
    cands = [(a, b) for a, b in BN_CONFUSIONS if a in w]
    if cands and rng.random() < 0.8:
        a, b = rng.choice(cands)
        idx = [m.start() for m in re.finditer(re.escape(a), w)]
        k = rng.choice(idx)
        return w[:k] + b + w[k + len(a):]
    # otherwise drop a vowel sign or hasanta (typing omission)
    idx = [k for k, ch in enumerate(w) if ch in SIGNS or ch == HASANTA]
    if idx:
        k = rng.choice(idx)
        return w[:k] + w[k + 1:]
    return w


# --------------------------------------------------------------------------------------
# Code-mixing lexicon (Bangla <-> English). Common meme/social-media vocabulary.
# --------------------------------------------------------------------------------------
BN_EN = {
    "আমি": "I", "আমার": "my", "আমাকে": "me", "আমরা": "we", "আমাদের": "our", "তুমি": "you", "তোমার": "your",
    "তোমাকে": "you", "তুই": "you", "তোর": "your", "তোকে": "you", "আপনি": "you", "আপনার": "your",
    "আপনাদের": "your", "সে": "he", "তার": "his", "ওর": "his", "তারা": "they", "এটা": "this", "ওটা": "that",
    "এই": "this", "ওই": "that", "কি": "what", "কী": "what", "কেন": "why", "কোথায়": "where", "কখন": "when",
    "কে": "who", "কেমন": "how", "কিভাবে": "how", "না": "no", "নাহ": "no", "হ্যাঁ": "yes", "হ্যা": "yes",
    "ভাই": "bro", "ভাইয়া": "bro", "বোন": "sister", "বন্ধু": "friend", "দোস্ত": "friend", "মা": "mom",
    "বাবা": "dad", "আব্বু": "dad", "আম্মু": "mom", "ছেলে": "boy", "মেয়ে": "girl", "বউ": "wife", "স্বামী": "husband",
    "স্যার": "sir", "শিক্ষক": "teacher", "ছাত্র": "student", "মানুষ": "people", "লোক": "man", "মহিলা": "lady",
    "ভালো": "good", "ভাল": "good", "খারাপ": "bad", "সুন্দর": "beautiful", "বড়": "big", "ছোট": "small",
    "নতুন": "new", "পুরানো": "old", "সত্যি": "really", "মিথ্যা": "lie", "সব": "all", "অনেক": "many",
    "একটু": "little", "খুব": "very", "বেশি": "more", "কম": "less", "আবার": "again", "শুধু": "only",
    "এখন": "now", "আজ": "today", "আজকে": "today", "কাল": "tomorrow", "সকাল": "morning", "রাত": "night",
    "দিন": "day", "সময়": "time", "বছর": "year", "মাস": "month", "সপ্তাহ": "week",
    "টাকা": "money", "বাড়ি": "home", "বাসা": "home", "খাবার": "food", "ভাত": "rice", "চা": "tea", "পানি": "water",
    "জল": "water", "পরীক্ষা": "exam", "স্কুল": "school", "কলেজ": "college", "অফিস": "office", "কাজ": "work",
    "চাকরি": "job", "রাস্তা": "road", "গাড়ি": "car", "ফোন": "phone", "বই": "book", "খেলা": "game",
    "প্রেম": "love", "ভালোবাসা": "love", "বিয়ে": "marriage", "মন": "mind", "হাসি": "smile", "কান্না": "crying",
    "রাগ": "anger", "ভয়": "fear", "দুঃখ": "sad", "খুশি": "happy", "মজা": "fun", "সমস্যা": "problem",
    "প্রশ্ন": "question", "উত্তর": "answer", "নাম": "name", "কথা": "talk", "গান": "song", "ছবি": "picture",
    "দেশ": "country", "সরকার": "government", "নেতা": "leader", "পুলিশ": "police", "ডাক্তার": "doctor",
    "করো": "do", "কর": "do", "করে": "doing", "করছে": "doing", "করেছে": "did", "যাও": "go", "যা": "go",
    "আসো": "come", "আয়": "come", "দেখো": "look", "দেখ": "look", "বলো": "say", "বল": "say", "বলে": "says",
    "খাও": "eat", "খাবি": "eat", "শোন": "listen", "শোনো": "listen", "দাও": "give", "দে": "give",
    "জানি": "know", "জানো": "know", "চাই": "want", "পারি": "can", "পারবো": "can", "ঘুম": "sleep",
    "মারো": "hit", "থামো": "stop", "চলো": "let's go", "অপেক্ষা": "wait", "ধন্যবাদ": "thanks",
    "দুঃখিত": "sorry", "প্লিজ": "please", "অবশ্যই": "obviously", "আসলে": "actually", "মানে": "means",
    "হয়েছে": "happened", "হবে": "will be", "ছিল": "was", "আছে": "have", "নেই": "nothing", "পাগল": "crazy",
    "বোকা": "fool", "চালাক": "smart", "গরিব": "poor", "ধনী": "rich", "শেষ": "end", "শুরু": "start",
}
EN_BN = {}
for _bn, _en in BN_EN.items():
    EN_BN.setdefault(_en.lower(), _bn)
EN_BN.update({"hello": "হ্যালো", "ok": "ঠিক আছে", "okay": "ঠিক আছে", "camera": "ক্যামেরা", "room": "রুম"})
BN_SUFFIXES = sorted(["টা", "টি", "রা", "দের", "কে", "ের", "র", "ে", "য়", "ও", "ই", "গুলো", "তে"], key=len, reverse=True)
_PUNCT = re.compile(r"^([^\wঀ-৿]*)(.*?)([^\wঀ-৿]*)$")


def _lookup_bn(core: str) -> str | None:
    if core in BN_EN:
        return BN_EN[core]
    for suf in BN_SUFFIXES:
        if core.endswith(suf) and core[: -len(suf)] in BN_EN:
            return BN_EN[core[: -len(suf)]]
    return None


# --------------------------------------------------------------------------------------
# Emoji
# --------------------------------------------------------------------------------------
EMOJI_POOL = ["😂", "🤣", "😭", "😡", "😍", "🙏", "😒", "😎", "🔥", "💔", "👍", "🤔", "😱", "🥺", "😆", "🙄"]
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0001F900-\U0001F9FF️‍]+"
)
EMOTICON_RE = re.compile(r"(?<!\w)(?::|;|=|x|X)-?(?:\)|\(|D|P|p|O|o|/|\|)+(?!\w)|<3|\^_\^|-_-")
SYMBOL_RE = re.compile(r"[^\w\sঀ-৿]")


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
TEXT_PERTURBATIONS = ["translit", "roman_noise", "bangla_noise", "codemix", "emoji_add", "emoji_remove"]

# severity -> parameter
PARAMS = {
    "translit": {1: 0.33, 2: 0.66, 3: 1.0},       # fraction of Bangla words romanised
    "roman_noise": {1: 0.15, 2: 0.30, 3: 0.50},   # fraction of words receiving a spelling variant
    "bangla_noise": {1: 0.10, 2: 0.20, 3: 0.35},  # fraction of Bangla words misspelt
    "codemix": {1: 0.33, 2: 0.66, 3: 1.0},        # prob. of switching a lexicon-covered word
    "emoji_add": {1: 1, 2: 3, 3: 6},              # number of inserted emojis
    "emoji_remove": {1: "emoji", 2: "emoticon", 3: "symbol"},
}


def make_rng(sample_id: str, family: str, severity: int) -> random.Random:
    return random.Random(zlib.crc32(f"{sample_id}|{family}|{severity}".encode()))


def perturb_text(text: str, family: str, severity: int, sample_id: str = "") -> str:
    rng = make_rng(sample_id, family, severity)
    p = PARAMS[family][severity]
    toks = _TOKEN.findall(text)

    if family == "translit":
        return " ".join(bn_to_roman(t) if is_bangla(t) and rng.random() < p else t for t in toks)

    if family == "roman_noise":
        rom = [bn_to_roman(t) for t in toks]
        return " ".join(_roman_word_noise(t, rng) if any(ch.isalpha() for ch in t) and rng.random() < p else t for t in rom)

    if family == "bangla_noise":
        return " ".join(_bangla_word_noise(t, rng) if is_bangla(t) and rng.random() < p else t for t in toks)

    if family == "codemix":
        out = []
        for t in toks:
            pre, core, post = _PUNCT.match(t).groups()
            rep = _lookup_bn(unicodedata.normalize("NFC", core)) if is_bangla(core) else EN_BN.get(core.lower())
            out.append(pre + rep + post if rep is not None and rng.random() < p else t)
        return " ".join(out)

    if family == "emoji_add":
        out = list(toks)
        for _ in range(p):
            out.insert(rng.randint(0, len(out)), rng.choice(EMOJI_POOL))
        return " ".join(out)

    if family == "emoji_remove":
        s = EMOJI_RE.sub("", text)
        if p in ("emoticon", "symbol"):
            s = EMOTICON_RE.sub("", s)
        if p == "symbol":
            s = SYMBOL_RE.sub(" ", s)
        return " ".join(s.split())

    raise ValueError(f"unknown text perturbation {family}")


def change_rate(a: str, b: str) -> float:
    """Token-level edit share (1 - difflib ratio over whitespace tokens); reported as perturbation coverage."""
    from difflib import SequenceMatcher

    ta, tb = a.split(), b.split()
    if not ta and not tb:
        return 0.0
    return 1 - SequenceMatcher(a=ta, b=tb, autojunk=False).ratio()
