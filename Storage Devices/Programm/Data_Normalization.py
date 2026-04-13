from Config import open_tuples, create_data_frame_as_file
from typing import Dict
from Translator import Translator_mapping
import pandas as pd
import re


regex_clean = r"[^\w\s\-,\.\*]"

regex_patterns = [
    r"\b(?:and|with|the|a|an|in|on|for|of|as|very|from|like|no|your|other|at|is|has|by|only|new|one|just|up|better|real|fast|easy|ideal|perfect|limited|free|alone|minimal|special|empty|only|current|limited|worth|recommended|action|remove)\b",

    # Marketing/Verkaufsfloskeln
    r"\b(?:offer|price|prices|discount|package|included|attached|warranty|edition|unpack|product|market|app|special|life|generation|experience|capture|browsing|history|store|uniquely)\b",

    # Technische Begriffe
    r"\b(?:connectivity|connected|external|media|network|components|electronic|electronics|computer|computers|devices|solid|mobile|mobil|phones|cellular|technical|angle|surface|book|line|segment|spacing|section|plate|gate|secure|swivel|state)\b",

    # Plattform-/Markennamen
    r"\b(?:mediamarkt|amazon|app)\b",

    # Sonstige / spezifische Begriffe
    r"\b(?:clan|shooting|drones|laptops|years|difference|double|light|store|facing|rods|members|spick|collect|photo|rescue|savage|incorporated|know|use|squar|joined|combination|attached|hidden|unpack|capture|surveillance|isolation|standard|technologies)\b",
]

mapping_speed = {
    r"M(b(?!it)|B(?:yte)?)\s*(?:[\/\\]|per)?\s*(s|sec|second|sekunde|sek|sekunden|seconds)": "MB/s"
}


def data_to_normalized_dataframe() -> pd.DataFrame:
    from Blocking import apply_regex_mapping
    pd_dataframe = open_tuples()

    pd_dataframe["name"] = pd_dataframe["name"].fillna('')
    pd_dataframe["description"] = pd_dataframe["description"].fillna('')
    apply_mapping_efficient(pd_dataframe, "name", Translator_mapping)
    apply_mapping_efficient(pd_dataframe, "description", Translator_mapping)

    apply_regex_mapping(pd_dataframe, "description", mapping_speed)
    apply_regex_mapping(pd_dataframe, "description", mapping_speed)


    for regex_str in regex_patterns:
        pattern = re.compile(rf"(?<![a-zA-Z0-9-])({regex_str})(?![a-zA-Z0-9-])", re.IGNORECASE)
        remove_words_by_regex(pd_dataframe, "name", pattern)
        remove_words_by_regex(pd_dataframe, "description", pattern)

    remove_words_by_regex(pd_dataframe, "name", r'\s+')
    remove_words_by_regex(pd_dataframe, "description", r'\s+')


    pd_dataframe["title"] = pd_dataframe["name"] + ' ' + pd_dataframe["description"]

    pd_dataframe = pd_dataframe.rename(columns={"brand": "companies"})

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


def apply_mapping_efficient(df: pd.DataFrame, column: str, mapping: Dict[str, str]) -> None:
    """
    Ersetzt ganze Wörter in der angegebenen Spalte des DataFrames anhand eines Regex-Mappings.

    Jeder Schlüssel im Mapping wird als reguläres Ausdrucksmuster interpretiert und bei Übereinstimmung
    durch den zugehörigen Zielwert ersetzt – jedoch nur, wenn der Treffer einem ganzen Wort entspricht.

    Die Ersetzungen erfolgen case-insensitive. Anschließend werden überflüssige Leerzeichen entfernt,
    und alle Wörter in der Spalte werden dedupliziert und alphabetisch sortiert.

    Parameter:
    - df: Das DataFrame, das in-place modifiziert wird.
    - column: Name der Spalte, in der die Ersetzungen durchgeführt werden.
    - mapping: Dictionary mit Strings als Schlüsseln und den jeweiligen Ersetzungswerten als Werten.
    """

    keys_sorted = sorted(mapping.keys(), key=len, reverse=True)
    combined_pattern = r'\b(' + '|'.join(map(re.escape, keys_sorted)) + r')\b'
    regex = re.compile(combined_pattern, re.IGNORECASE)

    mapping_lower = {k.lower(): v for k, v in mapping.items()}

    def replacer(match):
        found = match.group(0).lower()
        return mapping_lower.get(found, match.group(0))

    def process_value(x):
        if x is None:
            return None
        x_str = str(x)
        x_str = regex.sub(replacer, x_str)
        x_str = re.sub(r'\s+', ' ', x_str).strip()
        return " ".join(sorted(set(x_str.split())))

    df[column] = df[column].apply(process_value)



if __name__ == "__main__":
    data = data_to_normalized_dataframe()
    create_data_frame_as_file(data)