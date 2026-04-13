import os
import pandas as pd
import polars as pl
import re
from Union_Find import UnionFind
from io import StringIO
from typing import Union, Iterator, Tuple
from collections import Counter
from difflib import SequenceMatcher

DATA = 1 # 1 – wähle die Datei old oder updated
READ_LIMIT = None # Max. Anzahl der Zeilen, die aus den Tupeln eingelesen werden sollen (None = unbegrenzt)

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# Dateipfade für deinen Daten tuples
TUPLES_FILES = {
    1: os.path.join(DATA_PATH, "..", "data", "Z2.csv")
}

# Dateipfade für deine matching pairs
MATCHING_PAIRS_FILES = {
    1: os.path.join(DATA_PATH, "..", "data", "ZY2.csv")
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

def create_data_frame_as_file(df_or_groups: Union[pd.DataFrame, pl.DataFrame, Iterator[Tuple[tuple, pl.DataFrame]]],filename: str = "result/intermediate_result.csv") -> None:
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
    # Daten laden
    df = open_tuples()

    # Gesamtanzahl der Produkte
    num_rows = len(df)

    # Anzahl unterschiedlicher Marken
    num_unique_brands = df['brand'].nunique()

    # Top 10 Marken
    top_brands = df['brand'].value_counts().head(10)
    brand_stats_df = pd.DataFrame(top_brands).reset_index()
    brand_stats_df.columns = ['brand', 'count']

    # Alle Tokens aus den Produktnamen extrahieren
    all_texts = pd.concat([df['name'].dropna(), df['description'].dropna()])
    all_tokens = all_texts.apply(lambda x: re.findall(r'\b\w+\b', x.lower()))
    token_counter = Counter(token for tokens in all_tokens for token in tokens)
    all_token_counts = list(token_counter.items())

    # Token DataFrame: alle Tokens, sortiert nach Häufigkeit
    token_stats_df = pd.DataFrame(all_token_counts, columns=['token', 'count']).sort_values(by='count', ascending=False)

    # Preisstatistiken
    price_stats = df['price'].describe()
    price_stats_df = pd.DataFrame(price_stats).reset_index()
    price_stats_df.columns = ['statistic', 'value']

    # Ergebnisse speichern
    create_data_frame_as_file(brand_stats_df, filename="analysis/top_brands.csv")
    create_data_frame_as_file(token_stats_df, filename="analysis/all_tokens.csv")
    create_data_frame_as_file(price_stats_df, filename="analysis/price_stats.csv")

    # Optional: Ausgabe auf der Konsole
    print("Anzahl Produkte:", num_rows)
    print("Anzahl verschiedener Marken:", num_unique_brands)
    print("\nTop 10 Marken:\n", brand_stats_df)
    print("\nAlle Tokens im Namen (sortiert):\n", token_stats_df.head(10))
    print("\nPreisstatistiken:\n", price_stats_df)


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

    # Preisvergleich
    ground_truth["price_diff"] = abs(ground_truth["lprice"] - ground_truth["rprice"])
    ground_truth["price_rel_diff"] = ground_truth["price_diff"] / ground_truth[["lprice", "rprice"]].max(axis=1)

    # Markenvergleich
    ground_truth["brand_match"] = (ground_truth["lbrand"].str.lower().fillna("") == ground_truth["rbrand"].str.lower().fillna(""))

    # Name-Ähnlichkeit (SequenceMatcher)
    ground_truth["name_similarity"] = ground_truth.apply(lambda row: similar(row["lname"], row["rname"]), axis=1)

    ground_truth["description_similarity"] = ground_truth.apply(lambda row: similar(row["ldescription"], row["rdescription"]), axis=1)

    # Token-Overlap bei Namen
    def token_overlap(a, b):
        tokens_a = set(re.findall(r'\w+', str(a).lower()))
        tokens_b = set(re.findall(r'\w+', str(b).lower()))
        if not tokens_a or not tokens_b:
            return 0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    ground_truth["name_token_overlap"] = ground_truth.apply(lambda row: token_overlap(row["lname"], row["rname"]), axis=1)
    ground_truth["description_token_overlap"] = ground_truth.apply(lambda row: token_overlap(row["ldescription"], row["rdescription"]), axis=1)

    # Statistik-Ergebnisse
    stats = {
        "Anzahl Vergleiche": len(ground_truth),
        "Ø Preisunterschied (absolut)": ground_truth["price_diff"].mean(),
        "Ø Preisunterschied (relativ)": ground_truth["price_rel_diff"].mean(),
        "Anteil exakt gleicher Marken": ground_truth["brand_match"].mean(),
        "Ø Name-Ähnlichkeit (SequenceMatcher)": ground_truth["name_similarity"].mean(),
        "Ø Description-Ähnlichkeit (SequenceMatcher)": ground_truth["description_similarity"].mean(),
        "Ø Token-Overlap in Namen": ground_truth["name_token_overlap"].mean(),
        "Ø Token-Overlap in description": ground_truth["description_token_overlap"].mean(),
    }

    # Optional: abspeichern
    os.makedirs("result", exist_ok=True)
    ground_truth.to_csv("analysis/ground_truth_analysis_detailed.csv", index=False)
    pd.Series(stats).to_csv("analysis/ground_truth_summary.csv")

    # Ausgabe
    print("=== Zusammenfassung ===")
    for k, v in stats.items():
        print(f"{k}: {v:.3f}")


if __name__ == "__main__":
    analyze_dataframe()
    analyze_ground_truth()

