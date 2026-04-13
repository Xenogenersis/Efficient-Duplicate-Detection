import os
import pandas as pd
import polars as pl
import re
from Union_Find import UnionFind
from io import StringIO
from typing import Union, Iterator, Tuple
from collections import Counter
from difflib import SequenceMatcher


DATA = 2  # 1 oder 2 – wähle die Datei old oder updated
READ_LIMIT = None # Max. Anzahl der Zeilen, die aus den Tupeln eingelesen werden sollen (None = unbegrenzt)

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

TUPLES_FILES = {
    1: os.path.join(DATA_PATH, "..", "data", "Z1.csv"),
    2: os.path.join(DATA_PATH, "..", "data", "Z1_update.csv"),
}

MATCHING_PAIRS_FILES = {
    1: os.path.join(DATA_PATH, "..", "data", "ZY1.csv"),
    2: os.path.join(DATA_PATH, "..", "data", "ZY1_update.csv"),
}


def open_tuples() -> pd.DataFrame:
    """
    Lädt eine begrenzte Anzahl an Zeilen aus einer CSV-Datei,
    basierend auf dem aktuellen Schlüssel `DATA` und der Dateizuordnung `TUPLES_FILES`.

    Rückgabe:
    - pd.DataFrame mit den eingelesenen Datenzeilen.
    """
    filepath = TUPLES_FILES.get(DATA)

    df = pd.read_csv(filepath, nrows=READ_LIMIT)

    return df


def create_data_frame_as_file(df_or_groups: Union[pd.DataFrame, pl.DataFrame, Iterator[Tuple[tuple, pl.DataFrame]]], filename: str = "result/intermediate_result.csv") -> None:
    """
    Speichert ein DataFrame (pandas oder polars) oder einen Iterator von (key, polars DataFrame) Tupeln als CSV-Datei.

    Der Pfad ist relativ zur Datei config.py.

    Parameter:
    - df_or_groups: DataFrame (pandas oder polars) oder Iterator von (key, pl.DataFrame) Tupeln.
    - filename: Zielpfad der CSV-Datei (optional, Standard: 'result/intermediate_result.csv').

    Rückgabe:
    - None
    """

    config_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(config_dir, filename)

    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    def is_iterator_of_key_df(obj):
        try:
            iterator = iter(obj)
            first = next(iterator)
            if (
                isinstance(first, tuple) and
                len(first) == 2 and
                isinstance(first[1], pl.DataFrame)
            ):
                return True
        except Exception:
            return False
        return False

    if is_iterator_of_key_df(df_or_groups):
        group_dfs = [group_df for key, group_df in df_or_groups]
        combined_df = pl.concat(group_dfs)
        combined_df.write_csv(target_path)

    elif isinstance(df_or_groups, pd.DataFrame):
        df_or_groups.to_csv(target_path, index=False)

    elif isinstance(df_or_groups, pl.DataFrame):
        df_or_groups.write_csv(target_path)

    else:
        raise TypeError("Der Input muss ein pandas DataFrame, Polars DataFrame oder Iterator von (tuple, pl.DataFrame) sein.")
    try:
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(f"# Zeilenanzahl: {READ_LIMIT} Data: {DATA}\n")
    except NameError:
        pass

    print(f"\nErgebnis wurde in '{target_path}' gespeichert.")


def open_ground_truth() -> pd.DataFrame:
    """
    Lädt die Ground-Truth-Matching-Paare aus einer CSV-Datei.
    Es werden nur Paare eingelesen, bei denen sowohl 'lid' als auch 'rid' kleiner oder gleich READ_LIMIT - 2 sind.
    Die Paare werden so sortiert, dass immer lid <= rid und nach lid/rid sortiert.
    """
    filepath = MATCHING_PAIRS_FILES.get(DATA)
    if filepath is None:
        raise ValueError(f"Kein Pfad für DATA '{DATA}' gefunden.")
    df = pd.read_csv(filepath)

    if isinstance(READ_LIMIT, int):
        limit = READ_LIMIT - 2
        df = df[(df['rid'].isna() | (df['rid'] <= limit)) & (df['lid'].isna() | (df['lid'] <= limit))]

    df[['lid', 'rid']] = df[['lid', 'rid']].apply(
        lambda row: pd.Series(sorted([row['lid'], row['rid']])), axis=1
    )

    df = df.sort_values(by=['lid', 'rid']).reset_index(drop=True)
    return df


def open_results() -> tuple[pd.DataFrame, str]:
    """
    Lädt die gesamte result-Datei mit Matching-Paaren,
    die letzte Zeile wird separat als String zurückgegeben.

    Rückgabe:
    - pd.DataFrame mit den geladenen und sortierten Matching-Paaren (ohne letzte Zeile).
    - str: Inhalt der letzten Zeile als String.

    Raises:
    - FileNotFoundError: Falls die Datei nicht gefunden wird.
    """

    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", "result_data.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Die Datei '{filepath}' existiert nicht.")

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    footer_line = lines[-1].rstrip('\n')
    csv_content = "".join(lines[:-1])

    df = pd.read_csv(StringIO(csv_content))
    df_sorted = sortiere_matches(df)

    return df_sorted, footer_line


def sortiere_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Für jede Zeile im DataFrame werden die beiden IDs so angeordnet,
    dass die kleinere ID in der Spalte 'lid' steht.
    Anschließend wird der DataFrame zuerst nach 'lid' und dann nach 'rid' sortiert.

    :param df: DataFrame mit zwei Spalten 'lid' und 'rid'
    :return: sortierter DataFrame mit kleineren IDs links und sortierten Zeilen
    """
    # Tausch der IDs in den Zeilen, falls lid > rid
    mask = df['lid'] > df['rid']
    df.loc[mask, ['lid', 'rid']] = df.loc[mask, ['rid', 'lid']].values

    # Sortiere nach 'lid' und dann 'rid'
    df_sorted = df.sort_values(by=['lid', 'rid']).reset_index(drop=True)

    return df_sorted

def analyze_dataframe():
    df = open_tuples()

    # Gesamtanzahl der Produkte
    num_rows = len(df)

    # Alle Tokens aus dem Titel extrahieren
    all_tokens = df['title'].dropna().apply(lambda x: re.findall(r'\b\w+\b', x.lower()))
    token_counter = Counter(token for tokens in all_tokens for token in tokens)
    token_stats_df = pd.DataFrame(token_counter.items(), columns=['token', 'count']).sort_values(by='count', ascending=False)

    # Ergebnisse speichern
    os.makedirs("analysis", exist_ok=True)
    create_data_frame_as_file(token_stats_df, filename="analysis/all_tokens.csv")

    # Konsolenausgabe
    print("Anzahl Produkte:", num_rows)
    print("\nHäufigste Tokens:\n", token_stats_df.head(10))


def similar(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()

def analyze_ground_truth():

    ground_truth = open_ground_truth()
    uf_ground_truth = UnionFind()
    uf_ground_truth.add_pairs_from_df(ground_truth)
    ground_truth = uf_ground_truth.get_all_pairs_df()
    mapping_df = open_tuples()

    def mergmerge_mapping(result: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
        result = result.merge(mapping_df.add_prefix('l'), left_on='lid', right_on='lid', how='left')
        result = result.merge(mapping_df.add_prefix('r'), left_on='rid', right_on='rid', how='left')
        return result

    ground_truth = mergmerge_mapping(ground_truth, mapping_df)

    def token_overlap(a, b):
        tokens_a = set(re.findall(r'\w+', str(a).lower()))
        tokens_b = set(re.findall(r'\w+', str(b).lower()))
        if not tokens_a or not tokens_b:
            return 0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    # Ähnlichkeitsmetriken berechnen
    ground_truth["title_similarity"] = ground_truth.apply(lambda row: similar(row["ltitle"], row["rtitle"]), axis=1)
    ground_truth["title_token_overlap"] = ground_truth.apply(lambda row: token_overlap(row["ltitle"], row["rtitle"]),
                                                             axis=1)

    # Statistik-Ergebnisse
    stats = {
        "Anzahl Vergleiche": len(ground_truth),
        "Ø Titel-Ähnlichkeit (SequenceMatcher)": ground_truth["title_similarity"].mean(),
        "Ø Token-Overlap in Titeln": ground_truth["title_token_overlap"].mean(),
    }

    # Ergebnisse speichern
    os.makedirs("analysis", exist_ok=True)
    ground_truth.to_csv("analysis/ground_truth_analysis_detailed.csv", index=False)
    pd.Series(stats).to_csv("analysis/ground_truth_summary.csv")

    # Ausgabe
    print("=== Zusammenfassung ===")
    for k, v in stats.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    analyze_dataframe()
    analyze_ground_truth()


