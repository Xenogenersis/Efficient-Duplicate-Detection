from Config import create_data_frame_as_file
from Data_Normalization import data_to_normalized_dataframe
from typing import List, Optional, Union, Pattern, Iterator, Tuple, Dict
import pandas as pd
import polars as pl
import re

REGEX_DRIVES_CAPACITY = r"(?:1(?:\.0)?|2(?:\.0)?|4(?:\.0)?|8(?:\.0)?|16(?:\.0)?|32(?:\.0)?|64(?:\.0)?|128(?:\.0)?|256(?:\.0)?|512(?:\.0)?)\s*(?:GB|Gigabyte(?:s)?|TB|Terabyte(?:s)?)"

CAPACITY_UNIT_MAPPING = {
    r"(1(?:\.0)?)\s*(?:GB|Gigabyte(?:s)?)": "1 GB",
    r"(2(?:\.0)?)\s*(?:GB|Gigabyte(?:s)?)": "2 GB",
    r"(4(?:\.0)?)\s*(?:GB|Gigabyte(?:s)?)": "4 GB",
    r"(8(?:\.0)?)\s*(?:GB|Gigabyte(?:s)?)": "8 GB",
    r"(16(?:\.0)?)\s*(?:GB|Gigabyte(?:s)?)": "16 GB",
    r"(32(?:\.0)?)\s*(?:GB|Gigabyte(?:s)?)": "32 GB",
    r"(64(?:\.0)?)\s*(?:GB|Gigabyte(?:s)?)": "64 GB",
    r"(128(?:\.0)?)\s*(?:GB|Gigabyte(?:s)?)": "128 GB",
    r"(256(?:\.0)?)\s*(?:GB|Gigabyte(?:s)?)": "256 GB",
    r"(512(?:\.0)?)\s*(?:GB|Gigabyte(?:s)?)": "512 GB",

    r"(1(?:\.0)?)\s*(?:TB|Terabyte(?:s)?)": "1 TB",
    r"(2(?:\.0)?)\s*(?:TB|Terabyte(?:s)?)": "2 TB",
    r"(4(?:\.0)?)\s*(?:TB|Terabyte(?:s)?)": "4 TB",
}

REGEX_FLASH_FORMAT = r"(?:USB|Stick|Flash\sDriv|Thumb\sDrive|USB-Drive|jumpdrive|USB[123]|Type-A|Type-C|SD|Secure|Digital|SDHC|HC|SDXC|SDUC|UHS(?:-I|-II|-III)?|U3|U1|V\s*(?:6|10|30|60|90)|A1|A2|Class\s*[24610]|C\s*[24610]|ultimate|microSD(?:C|HC|XC)?|micro|CompactFlash|CF(?:-Karte)?|CFexpress|CF\sExpress|XQD)"

FLASH_FORMAT_MAPPING = {
    r"(?:USB|Stick|Flash\sDriv|Thumb\sDrive|USB-Drive|jumpdrive|USB[123]|Type-A|Type-C)": "USB",
    r"(?:microSD(?:C|HC|XC)?|micro)": "microSD",
    r"(?:SD|Secure|Digital|SDHC|HC|SDXC|SDUC|UHS(?:-I|-II|-III)?|U3|U1|V\s*(?:6|10|30|60|90)|A1|A2|Class\s*[24610]|C\s*[24610])": "SD",
    r"(?:CompactFlash|CF(?:-Karte)?|CFexpress|CF\sExpress)": "CF",
    r"(?:XQD)": "XQD",
}

REGEX_DRIVE_COMPANIES = r"(?:Intenso|Kingston|Lexar|PNY|Samsung|SanDisk|Sony|Toshiba|Transcend|Verbatim|Western(?:\s)?Digital|Apacer|Corsair|pendrive|datatraveler|microvault|transmemory|cruzer|exceria|evo|extreme|ultra|edge|canvas|ultimate)"

COMPANIES_MAPPING = {
    r"Intenso": "Intenso",
    r"Kingston|datatraveler": "Kingston",
    r"Lexar": "Lexar",
    r"PNY|pendrive": "PNY",
    r"Samsung|ultimate": "Samsung",
    r"SanDisk|cruzer|exceria|evo|extreme|ultra|edge|canvas": "SanDisk",
    r"Sony|microvault": "Sony",
    r"Toshiba|transmemory": "Toshiba",
    r"Transcend": "Transcend",
    r"Verbatim": "Verbatim",
    r"Western Digital": "Western Digital",
    r"WesternDigital": "Western Digital",
    r"Apacer": "Apacer",
    r"Corsair": "Corsair"
}

REGEX_product_series = r"(?:pendrive|datatraveler|microvault|transmemory|cruzer|exceria|evo|extreme|ultra|edge|canvas|ultimate)"

REGEX_DRIVES_SPEED = r"\d+(?:[.,]\d+)?\s*(?:mega|m)?(?:b(?:it|yte)?|aga)?(?:\/|\\?\\?u002[fF]|\s*per\s*|\s*pro\s*|\s*p\s*|\s*)s(?:ec(?:s|ond)?)*|\d+\s*x"

REGEX_SD_CLASSES   = r"(?:UHS(?:-I|-II|-III)?|U3|U1|V\s*(?:6|10|30|60|90)|A1|A2|Class\s*?[24610]|C[24610])"


def generate_blocking() -> Iterator[Tuple[tuple, pl.DataFrame]]:
    pd_dataframe = data_to_normalized_dataframe()

    if "companies" in pd_dataframe.columns:
        cols = list(pd_dataframe.columns)
        cols.remove("companies")
        cols.insert(1, "companies")
        pd_dataframe = pd_dataframe[cols]

    pattern_flash_format = re.compile(rf"\b({REGEX_DRIVE_COMPANIES})\b", re.IGNORECASE)
    insert_blocking_key_column(pd_dataframe, "title", pattern_flash_format, 1, "companies_from_title")

    pd_dataframe["companies"] = pd_dataframe["companies"].fillna("") + " " + pd_dataframe["companies_from_title"].fillna("")
    pd_dataframe = pd_dataframe.drop(columns=["companies_from_title"])

    apply_regex_mapping(pd_dataframe, "companies", COMPANIES_MAPPING,  REGEX_DRIVE_COMPANIES)

    pd_dataframe["companies"] = pd_dataframe["companies"].replace("", None)


    pattern_flash_format = re.compile(rf"\b({REGEX_FLASH_FORMAT})\b", re.IGNORECASE)
    insert_blocking_key_column(pd_dataframe, "title", pattern_flash_format, 2, "format")

    apply_regex_mapping(pd_dataframe, "format", FLASH_FORMAT_MAPPING,  REGEX_FLASH_FORMAT)

    pattern_space = re.compile(rf"\b({REGEX_DRIVES_CAPACITY})\b", re.IGNORECASE)
    insert_blocking_key_column(pd_dataframe, "title", pattern_space, 3, "capacity")

    apply_regex_mapping(pd_dataframe, "capacity", CAPACITY_UNIT_MAPPING,  REGEX_DRIVES_CAPACITY)

    pd_dataframe.loc[pd_dataframe["format"].str.contains("microSD", case=False, na=False), "format"] = \
        pd_dataframe.loc[pd_dataframe["format"].str.contains("microSD", case=False, na=False), "format"].str.replace(
            "sd", '', case=False, regex=False)

    apply_regex_mapping(pd_dataframe, "format", FLASH_FORMAT_MAPPING,  REGEX_FLASH_FORMAT)

    pattern_product_series = re.compile(rf"\b({REGEX_product_series})\b", re.IGNORECASE)
    insert_blocking_key_column(pd_dataframe, "title", pattern_product_series, 4, "product series")

    pl_dataframe = pl.from_pandas(pd_dataframe)

    groups = pl_dataframe.group_by(["companies","format","capacity","product series"])

    # groups = staged_adaptive_blocking_as_groupby(pl_dataframe, ["capacity", "format","companies","product series"], 2500) haben wir versucht

    return groups.__iter__()


def staged_adaptive_blocking_as_groupby(pl_dataframe: pl.DataFrame, blocking_cols: list[str], max_block_size: int) -> Iterator[Tuple[tuple, pl.DataFrame]]:
    combined_keys = []
    groups = pl_dataframe.group_by(blocking_cols[0]).__iter__()

    for primary_key, group_df in groups:
        if len(group_df) > max_block_size and len(blocking_cols) > 1:
            sub_keys = staged_adaptive_blocking_as_groupby(group_df, blocking_cols[1:], max_block_size)
            for secondary_key, group_df in sub_keys:
                combined_key = primary_key + secondary_key
                group=(combined_key, group_df)
                combined_keys.append(group)
        else:
            group = (primary_key, group_df)
            combined_keys.append(group)

    return iter(combined_keys)


def join_or_na(lst: List[str]) -> Optional[str]:
    """
    Erstellt aus einer Liste von Strings einen sortierten, einzigartigen String,
    getrennt durch Leerzeichen. Falls die Liste leer ist, wird None zurückgegeben.

    Parameter:
    - lst: Liste von Strings.

    Rückgabe:
    - Ein sortierter, einzigartiger String mit Leerzeichen getrennt, oder None, wenn die Liste leer ist.
    """
    if not lst:
        return None
    unique_sorted = sorted(set(lst))
    return " ".join(unique_sorted)


def insert_blocking_key_column(pd_dataframe: pd.DataFrame, input_column: str, pattern: Union[str, Pattern], new_col_pos: int, new_col_name: str) -> None:
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

    matches_series = pd_dataframe[input_column].astype(str).str.findall(pattern).apply(
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
