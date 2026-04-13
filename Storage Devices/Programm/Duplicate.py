from Blocking import generate_blocking

import polars as pl
from typing import Iterator, Tuple, Optional
import textdistance
from Union_Find import UnionFind
from multiprocessing import Pool, cpu_count


def get_duplicate_pairs() -> None:
    """
    Führt Blocking und paarweisen Ähnlichkeitsvergleich durch,
    schreibt ähnliche Paare (basierend auf Textjaccard) in eine Datei.
    """
    block = generate_blocking()
    union = extend_union_jaccard(block, 'name')
    union.write_pairs_to_file()


def extend_union_jaccard(groups: Iterator[Tuple[tuple, pl.DataFrame]],
                         column: str,
                         uf: Optional[UnionFind] = None) -> UnionFind:
    """
    Verbindet IDs aus derselben Gruppe, deren Textwerte in 'column' sich ähnlich sind,
    unabhängig von Groß-/Kleinschreibung, basierend auf Jaccard-Ähnlichkeit.
    Die Verarbeitung erfolgt parallel über mehrere Prozesse.

    Parameter:
    - groups: Iterator von Tupeln (key, DataFrame), typischerweise aus einem GroupBy.
    - column: Die Spalte mit Texten, die verglichen werden sollen.
    - uf: Optionale bestehende UnionFind-Instanz.

    Rückgabe:
    - Die erweiterte oder neue UnionFind-Instanz.
    """
    if uf is None:
        uf = UnionFind()

    group_list = list(groups)
    args = [(df, column) for _, df in group_list]

    with Pool(processes=16) as pool:
        all_pairs = pool.map(_process_group, args)

    for group_pairs in all_pairs:
        for id1, id2 in group_pairs:
            uf.union(id1, id2)

    return uf


def _process_group(args: Tuple[pl.DataFrame, str]) -> list[Tuple[str, str]]:
    """
    Verarbeitet eine einzelne Gruppe von Datensätzen: Wandelt Texte in Kleinbuchstaben um,
    berechnet Jaccard-Ähnlichkeit und gibt ähnliche ID-Paare zurück.

    Parameter:
    - args: Tuple (DataFrame, column_name)

    Rückgabe:
    - Liste von ähnlichen ID-Paaren (Tuple[str, str])
    """
    df, column = args
    if df.height < 2:
        return []

    current_score = 87.5 if df.height > 7500 else 85

    df = df.with_columns(pl.col(column).str.to_lowercase().alias(column))

    texts = df[column].to_list()
    ids = df["id"].to_list()

    similar_pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = textdistance.jaccard(texts[i], texts[j])
            if score >= current_score / 100:
                similar_pairs.append((ids[i], ids[j]))

    return similar_pairs



if __name__ == "__main__":
    get_duplicate_pairs()