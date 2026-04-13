from Config import open_tuples, create_data_frame_as_file
from typing import List, Optional, Union, Pattern
import pandas as pd
import re

regex_patterns = [
    # Deutsche häufige Wörter (Stopwörter, Füllwörter)
    r"(?:ohne|das|in|extra|neu|gebraucht|wie|bestseller|Einsatzgebiet|band|contrast|nach|support|starter|super|testsieger|adapter|test|finger|roll|ab)",
    # Englische Artikel, Präpositionen, Konjunktionen
    r"(?:and|or|with|without|the|a|an|in|on|for|of|as|to|but|very|from|like|no|use|your|other|at|most|what|own|wont|comes)",
    # Produktmerkmale: Zustand & Basisqualität
    r"(?:extra|new|used|refurbished|reconditioned|open|box|second|hand|original|sealed|cheap|premium|limited|top|best|quality|latest|branded|special|capacitive|downgraded|downgrade|unbranded|hard|cheapest|internal|response|excludes|working|tested|listed|good|tough|accidental|prestige|deep|few|barebone|barebore|preowned|repair|Small)",
    # Produktmerkmale: Erweiterte Eigenschaften & technische Details
    r"(?:rotating|hot|large|natural|low|customized|custom|hreat|carrying|extremely|happen|newest|mid|advanced|reliable|installed|wide|advertisement|replacement|proprietary|great|detachable|motion|available|swap|maximum|flawless|optical|multifunction|swappable|embedded|hybrid|digital)",
    # Produktmengen, Sets & Varianten (Plural möglich)
    r"(?:packaging|bundle|set|pack|piece|details|model|edition|version|mini|keys)s?",
    # Versand, Verkauf & Sonderaktionen
    r"(?:free|shipping|delivery|sale|discount|gift|express|fast|topseller|wholesale|dropship|deals|price|pricing|prices|shopping|warrnty|wrnty|description|media|arrival|reviews|review|tinyDeal|market)",
    # Produktkategorien & Technik (Hardware, Software, Bürogeräte) – Geräte & Features
    r"(?:Performance|power|mobile|accessorie|online|gaming|kid|gamer|Game|student|labour|programming|product|technology|ordinateur|beats|business|table|series|phone|digitizer|switching|surfing|work|office|call|factory|stereo|printer|video)s?",
    # Produktkategorien & Technik – Zeitbegriffe & Wartung
    r"(?:read|reader|warranty|time|hour|day|year|weekly|manufacturer|print|made)s?",
    # Support, Extras & allgemeine technische Begriffe – Geräte & Funktionen
    r"(?:vology|supported|home|extras|winner|In-plane|all-in-one|brand|playback|Off-Lease|function|nit|build-in|vs|Corp\.|INC\.|form)",
    # Sonstiges
    r"(?:transformer|factor|diagnostic|pics|supplies|paper|skype|outdoor|jelly|non-parity|condition|order|meilleur|station|angle|folio|helicopter|ratio|definition|juegos|mavericks|combo|point)",
    # Hardware-Typen und Geräte (Plural möglich)
    r"(?:laptop|notebook|pc|personal|computer|desktop|workstation|Book|device|case|electronic|card|drive|flash|Output|client|network|cable|multi-capacitance|mode|docking|replicator|chassis|generation|upgrade|built-in|parts)s?",
    # Markennamen und Domains (inkl. Domainendungen)
    r"(?:alibaba|miniprice|amazon|eBay|thenerds|walmart|myGofer|overstock|catalog|tigerDirect|staples|hoh|softwareCity|buy|topendelectronics|techbuy)(?:\.com|\.ca|\.de|\.net)?",
    # Orte, Länder & Sprachangaben
    r"(?:japan|china|cn|australia|au|german|de|germany|usa|us|johannesburg|en|espanol|canada|ca|hongkong|mali|chinese|nicaragua|uk|spanish|state)"
]
regex_clean = r"[^\w\s\-,\.\*]"


def data_to_normalized_dataframe() -> pd.DataFrame:
    pd_dataframe = open_tuples()

    remove_words_by_regex(pd_dataframe, "title", regex_clean)

    for regex_str in regex_patterns:
        pattern = re.compile(rf"(?<![a-zA-Z0-9-])({regex_str})(?![a-zA-Z0-9-])", re.IGNORECASE)
        remove_words_by_regex(pd_dataframe, "title", pattern)

    remove_words_by_regex(pd_dataframe, "title", r"\s+")

    return pd_dataframe


def remove_words_by_regex(pd_dataframe: pd.DataFrame, column: str, regex) -> None:
    """
    Entfernt alle Vorkommen von Mustern, die dem gegebenen regulären Ausdruck entsprechen,
    in der angegebenen Spalte des DataFrames **direkt im Original-DataFrame**.

    Parameter:
    - df: Der DataFrame mit den Daten, der direkt verändert wird.
    - column: Die Spalte, in der die Muster entfernt werden sollen.
    - regex: Ein regulärer Ausdruck (entweder als String oder kompiliertes Pattern),
             der die zu entfernenden Muster beschreibt.

    Rückgabe:
    - None (das Original-DataFrame wird in-place verändert)
    """
    if isinstance(regex, str):
        pattern = re.compile(regex, flags=re.IGNORECASE)
    else:
        pattern = regex

    def clean_text(text):
        return pattern.sub(" ", str(text))

    pd_dataframe[column] = pd_dataframe[column].apply(clean_text)


if __name__ == "__main__":
    data = data_to_normalized_dataframe()
    create_data_frame_as_file(data)
