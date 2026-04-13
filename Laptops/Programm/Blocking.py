from Config import create_data_frame_as_file
from Data_Normalization import data_to_normalized_dataframe
import pandas as pd
import polars as pl
from typing import List, Optional, Union, Pattern, Iterator, Tuple, Dict
import re

REGEX_LAPTOP_COMPANIES = r"Lenovo|IBM|HP|Dell|Asus|Acer|MSI|Apple|Samsung|Toshiba|Sony|Panasonic|Fujitsu|LG|Microsoft|Gigabyte|Razer|Packard Bell|Schenker|Teclast|Nokia"
REGEX_LAPTOP_BRANDS = r"ThinkPad|IdeaPad|Yoga|EliteBook|ProBook|Spectre|Pavilion|Envy|Omen|ZBook|Latitude|Inspiron|XPS|Vostro|Alienware|Precision|G3|G7|ZenBook|VivoBook|ROG|Aspire|Predator|Nitro|TravelMate|Modern|Prestige|Stealth|Titan|MacBook|Galaxy Book|Tecra|Satellite|Qosmio|VAIO|Toughbook|Toughpad|Lifebook|Stylistic|Celsius|Surface Pro|Surface Laptop|AORUS|Blade|XMG|Netbook|kindle|Presario|Folio|Compaq"
REGEX_CPU_GPU_COMPANIES = r"Nvidia|Amd|Radeon|Athlon|Intel|Celeron|Atom|Pentium|Xeon"

CPU_GPU_REGEX_MAPPING = {
    r"amd": "amd",
    r"intel": "intel",
    r"nvidia": "nvidia",
    r"radeon|athlon": "amd",
    r"celeron|atom|pentium|xeon": "intel",
}

REGEX_NVIDIA_GPU = r"(?<![a-zA-Z0-9])(?:geforce\s+)?(?:gtx|gts|gt|gx|gs)?\s*(9[0-9]{2}|8[0-9]{2}|7[0-9]{2}|6[0-9]{2}|5[0-9]{2}|4[0-9]{2}|3[0-9]{2}|2[0-9]{2}|1[0-9]{2}|[1-9][0-9]?)(?:\s*ti)?(?![a-zA-Z0-9])"
REGEX_INTEL_CPU = r"(?<![a-zA-Z0-9])(?:intel\s+core\s+)?i[\s\-]*[3579](?:[\s\-]*\d{4,5}[A-Z]*)?(?![a-zA-Z0-9])"


def generate_blocking() -> Iterator[Tuple[tuple, pl.DataFrame]]:
    pd_dataframe = data_to_normalized_dataframe()

    pattern_companies = re.compile(rf"(?<![a-zA-Z0-9-])({REGEX_LAPTOP_COMPANIES})(?![a-zA-Z0-9-])", re.IGNORECASE)
    insert_blocking_key_column(pd_dataframe, "title", pattern_companies, 1, "companies")

    pattern_brands = re.compile(rf"(?<![a-zA-Z0-9-])({REGEX_LAPTOP_BRANDS})(?![a-zA-Z0-9-])", re.IGNORECASE)
    insert_blocking_key_column(pd_dataframe, "title", pattern_brands, 2, "product series")

    pattern_cpu_gpu_companies = re.compile(rf"(?<![a-zA-Z0-9-])({REGEX_CPU_GPU_COMPANIES})(?![a-zA-Z0-9-])", re.IGNORECASE)
    insert_blocking_key_column(pd_dataframe, "title", pattern_cpu_gpu_companies, 3, "CPU/GPU companies")
    apply_regex_mapping(pd_dataframe, "CPU/GPU companies", CPU_GPU_REGEX_MAPPING, REGEX_CPU_GPU_COMPANIES)

    pattern_intel_cpu = re.compile(REGEX_INTEL_CPU, re.IGNORECASE)
    insert_blocking_key_column(pd_dataframe, "title", pattern_intel_cpu, 4, "intel CPU")

    pattern_nvidia_gpu = re.compile(REGEX_NVIDIA_GPU, re.IGNORECASE)
    insert_blocking_key_column(pd_dataframe, "title", pattern_nvidia_gpu, 5, "nvidia GPU")

    pl_dataframe = pl.from_pandas(pd_dataframe)

    pl_dataframe = pl_dataframe.group_by(["companies", "CPU/GPU companies", "intel CPU", "nvidia GPU"]).__iter__()

    return pl_dataframe


def join_or_na(lst: List[str]) -> Optional[str]:
    """
    Erstellt aus einer Liste von Strings einen sortierten, einzigartigen String,
    getrennt durch Leerzeichen. Falls die Liste leer ist, wird None zurückgegeben.

    Parameter:
    - lst: Liste von Strings.

    Rückgabe:
    - Sortierter, einzigartiger String mit Leerzeichen getrennt oder None, wenn die Liste leer ist.
       """
    if not lst:
        return None
    unique_sorted = sorted(set(lst))
    return " ".join(unique_sorted)


def insert_blocking_key_column(pd_dataframe: pd.DataFrame, column: str, pattern: Union[str, Pattern], new_col_pos: int, new_col_name: str) -> None:
    """
    Fügt dem DataFrame eine neue Spalte hinzu, deren Werte aus regulären
    Ausdrücken basierend auf einer vorhandenen Spalte extrahiert werden.

    Parameter:
    - pd_dataframe: Eingabe-DataFrame.
    - input_column: Name der Spalte, aus der die Werte extrahiert werden.
    - pattern: Regulärer Ausdruck (als String oder compiled Pattern).
    - new_col_pos: Position, an der die neue Spalte eingefügt wird.
    - new_col_name: Name der neuen Spalte.

    Rückgabe:
    - Modifizierter DataFrame mit der neuen Spalte, die aus
      einzigartigen, sortierten, kleingeschriebenen Treffern besteht.
      Wenn keine Treffer gefunden wurden, ist der Wert None.
    """
    if isinstance(pattern, str):
        pattern = re.compile(pattern)

    matches_series = pd_dataframe[column].astype(str).str.findall(pattern).apply(
        lambda lst: [m.lower() for m in lst])

    new_col = matches_series.apply(join_or_na)

    pd_dataframe.insert(loc=new_col_pos, column=new_col_name, value=new_col)


def apply_regex_mapping(pd_dataframe: pd.DataFrame, column: str, mapping: Dict[str, str], duplicate_regex: Optional[Union[str, Pattern]] = None) -> None:
    """
    Ersetzt Wörter in der angegebenen Spalte des DataFrames anhand eines Regex-Mappings.
    Jeder Treffer eines der Regex-Schlüssel wird durch den zugehörigen Wert ersetzt.
    Anschließend werden über ein optionales `duplicate_regex` alle Vorkommen, die diesem Muster
    entsprechen, extrahiert, in Kleinbuchstaben umgewandelt, duplizierte Einträge entfernt
    und sortiert zu einem einzigen String zusammengefügt.

    Parameter:
    - pd_dataframe: Das DataFrame, das in-place modifiziert wird.
    - column: Name der Spalte, in der die Ersetzungen stattfinden.
    - mapping: Dictionary mit Regex-Pattern-Strings als Schlüssel und den jeweiligen Ersetzungswerten als Werte.
    - duplicate_regex: Optionaler regulärer Ausdruck (als String oder Pattern).
      Wenn angegeben, wird die Spalte nach Ersetzungen nochmals auf Muster-Treffer durchsucht,
      um Duplikate zu entfernen und die Werte zu vereinheitlichen.
    """

    for pattern, repl in mapping.items():
        compiled_pattern = re.compile(pattern, re.IGNORECASE)
        pd_dataframe[column] = pd_dataframe[column].astype(str).str.replace(compiled_pattern,f" {repl} ", regex=True)

    pd_dataframe[column] = pd_dataframe[column].str.replace(r"\s+", " ", regex=True).str.strip()

    if duplicate_regex is not None:
        if isinstance(duplicate_regex, str):
            duplicate_regex = re.compile(duplicate_regex, re.IGNORECASE)
        pd_dataframe[column] = pd_dataframe[column].astype(str).str.findall(duplicate_regex).apply(
            lambda lst: [m.lower() for m in lst])
        pd_dataframe[column] = pd_dataframe[column].apply(join_or_na)


if __name__ == "__main__":
    data = generate_blocking()
    create_data_frame_as_file(data)

