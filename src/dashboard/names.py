"""Common-name normalization for bird labels.

Label formats vary by source: "123: Blue Jay", "Cyanocitta cristata / Blue Jay",
"Blue Jay - Cyanocitta cristata", "Blue Jay (Cyanocitta cristata)", or a bare
scientific name. Display + storage use the common name only.
"unknown" is preserved exactly so DB filters keep working.
"""
import re

_LEADING_ID = re.compile(r"^\d+[\s:._-]+(.*)$")
_BINOMIAL = re.compile(r"^[A-Z][a-z]+(?: [a-z]+){1,2}$")

# Fallback for scientific-only labels (e.g. a labels file with no common names).
# Covers common Atlanta backyard visitors; extend as needed.
SCIENTIFIC_TO_COMMON = {
    "cardinalis cardinalis": "Northern Cardinal",
    "cyanocitta cristata": "Blue Jay",
    "poecile carolinensis": "Carolina Chickadee",
    "baeolophus bicolor": "Tufted Titmouse",
    "thryothorus ludovicianus": "Carolina Wren",
    "zenaida macroura": "Mourning Dove",
    "turdus migratorius": "American Robin",
    "haemorhous mexicanus": "House Finch",
    "haemorhous purpureus": "Purple Finch",
    "spinus tristis": "American Goldfinch",
    "dryobates pubescens": "Downy Woodpecker",
    "melanerpes carolinus": "Red-Bellied Woodpecker",
    "sitta carolinensis": "White-Breasted Nuthatch",
    "sitta pusilla": "Brown-Headed Nuthatch",
    "sturnus vulgaris": "European Starling",
    "quiscalus quiscula": "Common Grackle",
    "sialia sialis": "Eastern Bluebird",
    "dumetella carolinensis": "Gray Catbird",
    "mimus polyglottos": "Northern Mockingbird",
    "pipilo erythrophthalmus": "Eastern Towhee",
    "spizella passerina": "Chipping Sparrow",
    "melospiza melodia": "Song Sparrow",
    "passer domesticus": "House Sparrow",
    "junco hyemalis": "Dark-Eyed Junco",
    "setophaga pinus": "Pine Warbler",
    "setophaga coronata": "Yellow-Rumped Warbler",
    "archilochus colubris": "Ruby-Throated Hummingbird",
    "agelaius phoeniceus": "Red-Winged Blackbird",
    "toxostoma rufum": "Brown Thrasher",
    "corvus brachyrhynchos": "American Crow",
    "corvus ossifragus": "Fish Crow",
    "dryocopus pileatus": "Pileated Woodpecker",
    "sayornis phoebe": "Eastern Phoebe",
    "melanerpes pileatus": "Pileated Woodpecker",
    "colinus virginianus": "Northern Bobwhite",
    "chaetura pelagica": "Chimney Swift",
}


def _looks_scientific(s: str) -> bool:
    """True for binomial/trinomial Latin names like 'Cyanocitta cristata'."""
    return bool(_BINOMIAL.match(s.strip()))


def _tidy(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip(" -_.:")
    m = _LEADING_ID.match(s)
    if m and m.group(1).strip():
        s = m.group(1).strip()
    return re.sub(r"\s+", " ", s).strip(" -_.:")


def common_name(label) -> str:
    if label is None:
        return "unknown"
    s = str(label).strip()
    if not s:
        return "unknown"
    if s.lower() in ("unknown", "background", "none", "null"):
        return "unknown"
    # "123: Blue Jay" -> "Blue Jay"
    if ":" in s:
        head, _, tail = s.partition(":")
        if head.strip().isdigit() and tail.strip():
            s = tail.strip()

    # Gather candidates from every delimiter, then prefer the non-scientific one
    # regardless of order ("Sci / Common" or "Common - Sci" both work).
    candidates = [s]
    if "/" in s:
        candidates = [p.strip() for p in s.split("/") if p.strip()] or candidates
    else:
        for sep in (" — ", " – ", " - "):
            if sep in s:
                candidates = [p.strip() for p in s.split(sep) if p.strip()]
                break
    if "(" in s and ")" in s:
        before = s.split("(", 1)[0].strip()
        m = re.search(r"\(([^)]+)\)", s)
        inside = m.group(1).strip() if m else ""
        cands = [c for c in (before, inside) if c]
        if cands:
            candidates = cands

    candidates = [_tidy(c) for c in candidates if _tidy(c)]
    if not candidates:
        return "unknown"
    picked = next((c for c in candidates if not _looks_scientific(c)), candidates[-1])

    mapped = SCIENTIFIC_TO_COMMON.get(picked.lower())
    if mapped:
        return mapped
    if _looks_scientific(picked):
        # Scientific-only label with no mapping: return as-is (title-cased) rather
        # than guessing. Add it to SCIENTIFIC_TO_COMMON above to map it.
        return picked
    return picked.title()
