"""Measured word properties: Zipf frequency and inflected family forms.

These are deliberately NOT asked of the language model. A model asked for a
Zipf value will return a confident, well-formatted, invented number, and a
number is the one thing in this app that reads as measured whether or not it
is. Frequency here comes from `wordfreq` (real corpus counts) and the family
forms come from `lemminflect`, a real inflection engine.

Both return None for "unknown". None renders as an em dash and stays NULL in
the database; it is never 0, never a guess, and never backfilled. A row with
no frequency is a word that has not been enriched yet, which is a fact worth
showing rather than papering over.
"""

import re
from functools import lru_cache

from lemminflect import getAllInflections
from wordfreq import zipf_frequency

# These two fields are MEASURED and are never written by a model. A model asked
# for a Zipf value returns a confident invented number, and one asked for
# inflections returns "runing"; a number or a word form reads as a fact whether
# or not it is one. Synonyms are the deliberate exception -- they are the model's
# own judgement, asked for once when the word is added and stored, not derived.
# Both helpers return None for "unknown": None is rendered as an em dash and
# stored as NULL, never 0, and never backfilled. 0 would read as "utterly
# rare", which is itself a claim.
MAX_FAMILY_FORMS = 6

def _is_word(form):
    """True when `form` actually exists in the corpus vocabulary.

    This is what makes the part-of-speech choice safe. lemminflect is willing to
    inflect "barn" as a verb and hand back "barning"; asking the same corpus
    that supplies the frequency whether that string is a word at all is what
    keeps the guess off the screen. Zipf 0 means the token never appeared in
    wordfreq's word list.

    The filter is loose in one direction -- wordfreq's list contains obscure
    real tokens, so an attested form is evidence, not proof -- which is why the
    wrong reading is discarded by scoring rather than by keeping whatever
    passes.
    """
    try:
        return zipf_frequency(form, "en") > 0
    except Exception:
        return False

# Cap on family forms. `run` gives 5 and reads fine; a long word like
# "antidisestablishmentarianism" would otherwise generate a dozen forms and
# turn the cell into a wall of text. 6 keeps the common cases complete and
# truncates the pathological ones rather than hiding them.
MAX_FAMILY_FORMS = 6

_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*$")

# Penn tags in the order their forms are displayed, so the list reads as the
# forms a learner meets: third person, then -ing, then the past.
_TAG_ORDER = ("VBZ", "VBG", "VBD", "VBN", "NNS", "JJR", "JJS")

# Which tags belong to which part of speech. Used only to choose BETWEEN the
# readings lemminflect offers an ambiguous word: "barn" is both a noun and a
# verb, and showing the verb's "barning" would be a guess dressed as grammar.
_POS_TAGS = {
    "N": ("NNS",),
    "V": ("VBZ", "VBG", "VBD", "VBN"),
    "J": ("JJR", "JJS"),
}

# Adjectives are checked before nouns but after verbs: a comparative is a
# positive signal of an adjective ("happier"), whereas an attested plural proves
# nothing, because a verb's third person is spelled the same way.
_POS_PREFERENCE = ("V", "J", "N")



def zipf_for(word):
    """Corpus Zipf frequency for a word or phrase, or None.

    Zipf is log10(rank-per-billion): 7 is among the commonest words in English,
    4 is roughly the top 40k, 1 is extremely rare. Phrases are supported --
    wordfreq tokenises them -- and a multi-word entry legitimately has its own
    measured frequency, so "kick the bucket" is a real answer here rather than
    a fallback.
    """
    text = (word or "").strip()
    if not text:
        return None
    try:
        value = zipf_frequency(text, "en")
    except Exception:
        # A malformed entry must never take a word insert down with it.
        return None
    return round(float(value), 2)


def _best_form(forms):
    """The most frequent form in a candidate set, or None.

    Weighing by frequency rather than by count is what separates the two
    readings of an ambiguous word. "leaf" is offered as both a noun (leaves) and
    a verb (leafs, leafing, leafed): the verb offers MORE forms, but leaves is
    the commoner word by a wide margin, and a learner looking up "leaf" wants
    leaves. "die" is the mirror image -- dies is the commonest noun form, but
    died and dying beat it, so the verb wins. Counting forms gets both wrong;
    this gets both right.
    """
    best, best_zipf = None, 0.0
    for form in forms:
        try:
            value = zipf_frequency(form, "en")
        except Exception:
            continue
        if value > best_zipf:
            best, best_zipf = form, value
    return best


def _reading_for(inflections, pos, text):
    """(forms, best_form) for one part of speech, both filtered by attestation.

    The noun reading keeps only its single most frequent plural. lemminflect lists
    EVERY valid plural under NNS -- for "leaf" that is ("leaves", "leafs") -- and
    keeping them all puts a second, wrong-looking plural in the cell even after
    "leaves" has been chosen as the best one.
    """
    forms = []
    for tag in _POS_TAGS[pos]:
        for form in inflections.get(tag, ()):
            if form.lower() != text.lower() and _is_word(form):
                forms.append(form)
    best = _best_form(forms)
    if pos == "N" and best:
        forms = [best]
    return [f.lower() for f in forms], best


def _zipf(form):
    try:
        return zipf_frequency(form, "en") if form else 0.0
    except Exception:
        return 0.0


# How far the verb reading has to beat the noun one when the two spellings are
# identical, before the verb is believed. "house" is a noun plural and a verb
# third person in exactly the same word, so the plural on its own says nothing;
# but for "die", died beats dies by a wide margin and the verb is obviously
# right. This threshold separates the two cases.
_SPELLING_TIE_MARGIN = 0.3


def _match_case(source, target):
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


# Irregular comparatives. lemminflect has no adjective entry for "better" at all
# -- it offers "betters" and "bettered", both wrong -- so these are answered
# before the engine is consulted. Short, because the list really is short.
_IRREGULAR_COMPARATIVE = {
    "better": "best", "worse": "worst", "more": "most", "less": "least",
    "further": "furthest", "farther": "farthest", "good": "better",
}


def family_for(word):
    """Inflected forms of a single word, or None.

    None for anything that is not one plain English word: a phrase has no
    inflection -- there is no other form of "aforementioned" -- so inventing one
    would be worse than showing nothing. Also None when no derived form is
    attested, since a lone lemma beside the word itself says nothing.

    Two stages. lemminflect does the grammar, and knows that the past of "run"
    is "ran" and the plural of "child" is "children", which no suffix rule
    reaches. wordfreq then decides which of the competing readings to believe,
    by counting how many of the generated forms the corpus has actually seen.
    Neither stage trusts the model.
    """
    text = (word or "").strip()
    if not _WORD_RE.match(text):
        return None

    if text.lower() in _IRREGULAR_COMPARATIVE:
        return " · ".join([text, _match_case(text, _IRREGULAR_COMPARATIVE[text.lower()])])

    try:
        inflections = getAllInflections(text)
    except Exception:
        # No entry for an out-of-vocabulary word. A normal outcome for a proper
        # noun, not an error worth surfacing.
        return None
    if not inflections:
        return None

    readings = {pos: _reading_for(inflections, pos, text)
                for pos in _POS_PREFERENCE}

    # The noun plural and the verb's third person are often the same string
    # ("house"/"houses", "leaf"/"leafs"). When that happens the plural carries no
    # evidence about which reading is right, so the noun wins unless the verb
    # clearly outranks it.
    n_forms, n_best = readings["N"]
    v_forms, v_best = readings["V"]
    third_person = (inflections.get("VBZ") or ("",))[0].lower()
    spelling_ambiguous = bool(n_best) and third_person == n_best.lower()

    candidates = []
    for pos in _POS_PREFERENCE:
        forms, best = readings[pos]
        if not best:
            continue
        if pos == "V" and spelling_ambiguous:
            if _zipf(v_best) <= _zipf(n_best) + _SPELLING_TIE_MARGIN:
                continue
        candidates.append((pos, forms, best))

    if not candidates:
        return None

    pos, derived, _ = max(
        candidates, key=lambda c: (_zipf(c[2]), -_POS_PREFERENCE.index(c[0]))
    )

    ordered, seen = [], set()
    for tag in _TAG_ORDER:
        if tag not in _POS_TAGS[pos]:
            continue
        for form in inflections.get(tag, ()):
            lowered = form.lower()
            if lowered in derived and lowered not in seen:
                seen.add(lowered)
                ordered.append(_match_case(text, form))
    return " · ".join([text] + ordered[:MAX_FAMILY_FORMS - 1])
