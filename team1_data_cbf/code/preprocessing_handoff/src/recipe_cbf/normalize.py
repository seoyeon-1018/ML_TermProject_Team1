"""Conservative, rule-based normalization shared by BOTH platforms.

No corpus statistics, Food.com mapping pickle, or evaluation labels are used.
We intentionally retain dried/fresh distinctions, meat cuts, fat/salt levels,
flour types and ambiguous compound ingredients. This is not an allergy parser.
"""
import ast
import html
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ALIASES = ROOT / "config" / "ingredient_aliases.json"
_NUMBER = r"(?:\d+\s+\d+/\d+|\d+\s*/\s*\d+|\d+(?:[.,]\d+)?)"
_QUANTITY = _NUMBER + r"(?:\s*-\s*" + _NUMBER + r")?"
_UNIT = r"(?:cups?|tablespoons?|teaspoons?|tbsp|tsp|ounces?|oz|pounds?|lbs?|grams?|g|kg|ml|liters?|litres?|pinch(?:es)?|dashes?|cans?|packages?|packets?)"


def parse_list(value):
    """Parse CSV Python-list cells with literal_eval, never eval/unpickle."""
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str):
        parsed = ast.literal_eval(value)
    else:
        raise ValueError("ingredients must be a list or a serialized list")
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError("ingredients must be a list of strings")
    return parsed


class IngredientNormalizer:
    def __init__(self, alias_path=DEFAULT_ALIASES):
        self.aliases = json.loads(Path(alias_path).read_text(encoding="utf-8"))

    @lru_cache(maxsize=100000)
    def normalize_one(self, text):
        # Removing one preparation layer may expose another ('beaten divided').
        # Reach a stable form so parsing and reusing canonical features agree.
        current = str(text)
        for _ in range(12):
            value = self._normalize_pass(current)
            if value == current:
                return value
            current = value
        raise ValueError(f"Normalization did not converge: {text!r}")

    def _normalize_pass(self, text):
        text = html.unescape(str(text)).strip()
        # Preserve space before Unicode fraction normalization (1½ -> 1 1/2).
        text = re.sub(r"([0-9])([¼½¾⅓⅔⅛⅜⅝⅞])", r"\1 \2", text)
        text = unicodedata.normalize("NFKC", text).lower().replace("⁄", "/")
        text = text.replace("’", "'").replace("–", "-").replace("—", "-")
        text = re.sub(r"<[^>]*>", " ", text)
        text = re.sub(r"^(?:for\s+(?:the\s+)?)?(?:sauce|rub|marinade|dressing|topping|filling|garnish|dough|batter)\s*:\s*", "", text)
        # A bare section heading is not an ingredient; 'sauce' alone is retained.
        if text.endswith(":") and len(text.split()) <= 4:
            return ""
        text = re.sub(r"\([^)]*\)", " ", text)
        quantity = re.match(r"^\s*" + _QUANTITY + r"\s+(.+)$", text)
        # '1% milk', '10-inch tortilla', '15 bean soup mix', '100 proof vodka'
        # contain product attributes, not measured ingredient quantities.
        if quantity and not re.match(r"(?:%|percent\b|bean\b|inch\b|proof\b|minute\b|calorie\b|up\b)", quantity[1]):
            text = quantity[1]
        text = re.sub(r"^\s*" + _UNIT + r"\.?\s+(?:of\s+)?", "", text)
        text = re.sub(r"^\s*(?:small|medium|large)\s+", "", text)
        text = re.sub(r",.*$", "", text)  # preparation after a comma
        text = re.sub(r"\b(?:to taste|as needed|for serving|optional)\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" .;,:")
        # Common source-corpus preparation suffixes, with conjunction removed
        # only inside a preparation suffix ('beans rinsed and drained').
        text = re.sub(r"(?:\s+(?:and\s+)?(?:rinsed|drained|undrained|thawed|beaten|melted|softened))+$", "", text)
        text = re.sub(r"^(?:packed|lightly packed|tightly packed)\s+", "", text)
        # Resolve exact meat aliases BEFORE removing prep words.
        text = self.aliases.get(text, text)
        text = re.sub(r"\b(?:finely|coarsely|roughly|freshly)\s+(?=(?:chopped|diced|minced|sliced|grated)\b)", "", text)
        text = re.sub(r"\b(?:chopped|diced|minced|sliced|grated|shredded|peeled|seeded|deseeded|divided)\b", " ", text)
        text = re.sub(r"\s+", " ", text).strip(" .;,:")
        for _ in range(4):
            mapped = self.aliases.get(text, text)
            if mapped == text:
                break
            text = mapped
        return text

    def normalize_list(self, ingredients):
        return sorted({x for s in parse_list(ingredients) if (x := self.normalize_one(s))})
