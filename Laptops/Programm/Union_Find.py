import itertools
from typing import List, Any
import pandas as pd
import os
from collections import defaultdict


class UnionFind:
    """
    Implementiert eine Union-Find (Disjoint Set)-Datenstruktur zur Verwaltung von Äquivalenzklassen.

    Methoden:
    - find(x): Findet den Wurzel-Knoten der Menge von x (mit Pfadkompression).
    - union(x, y): Vereint die Mengen, die x und y enthalten (mit Union by Rank).
    - add(x): Fügt ein neues Element als eigene Menge hinzu.
    - get_groups(): Gibt alle Äquivalenzklassen als Liste von Listen zurück.
    - get_all_pairs_df(): Gibt alle 2er-Tupel als pd.DataFrame innerhalb jeder Äquivalenzklasse zurück.
    - add_pairs_from_df(pairs_df): Fügt alle Paare aus einem DataFrame in die bestehende UnionFind-Instanz ein.
    - write_pairs_to_file(filename): Schreibt alle Paare in eine CSV-Datei.
    """
    def __init__(self):
        """
        Initialisiert die Union-Find-Struktur.

        Attribute:
        - parent (dict): Speichert für jedes Element den Eltern-Knoten.
                         Ein Element ist Wurzel, wenn es sein eigener Eltern-Knoten ist.
        - rank (dict):   Speichert die Ranghöhe (Tiefe) eines Wurzel-Knotens zur Optimierung.
        """
        self.parent = {}
        self.rank = {}

    def find(self, x) -> Any:
        """
        Bestimmt den Wurzel-Knoten der Menge, zu der x gehört.
        Führt dabei Pfadkompression durch, um die Struktur zu flachen.

        Falls x nicht existiert, wird es automatisch als neue Menge eingefügt.

        Parameter:
        - x (Any): Element, dessen Menge bestimmt werden soll.

        Rückgabe:
        - Wurzel-Knoten von x.
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y) -> None:
        """
        Vereinigt die Mengen von x und y mithilfe von Union by Rank.

        Der Baum mit geringerer Ranghöhe wird unter den mit höherem Rang gehängt.

        Parameter:
        - x (Any): Erstes Element.
        - y (Any): Zweites Element.
        """
        x_root = self.find(x)
        y_root = self.find(y)

        if x_root == y_root:
            return

        if self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        else:
            self.parent[y_root] = x_root
            if self.rank[x_root] == self.rank[y_root]:
                self.rank[x_root] += 1

    def add(self, x) -> None:
        """
        Fügt ein neues Element x als eigene Menge hinzu, falls es noch nicht existiert.

        Parameter:
        - x (Any): Hinzuzufügendes Element.
        """
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def get_groups(self) -> List[List[Any]]:
        """
        Gruppiert alle Elemente anhand ihres Wurzel-Knotens (Äquivalenzklassen).

        Rückgabe:
        - Liste von sortierten Äquivalenzklassen (Listen).
        - Die Gruppen selbst sind aufsteigend nach ihrem kleinsten Element sortiert.
        """
        groups = defaultdict(list)
        for element in self.parent:
            root = self.find(element)
            groups[root].append(element)

        grouped_lists = list(groups.values())
        for group in grouped_lists:
            group.sort()
        grouped_lists.sort(key=lambda g: g[0])

        return grouped_lists

    def get_all_pairs_df(self) -> pd.DataFrame:
        """
        Generiert ein DataFrame mit allen möglichen 2er-Kombinationen (lid, rid)
        innerhalb jeder Äquivalenzklasse.

        Rückgabe:
        - pd.DataFrame mit den Spalten 'lid' und 'rid'.
        """
        groups = self.get_groups()

        pairs = []
        for group in groups:
            if len(group) >= 2:
                pairs.extend(itertools.combinations(group, 2))

        df = pd.DataFrame(pairs, columns=['lid', 'rid'])
        return df

    def add_pairs_from_df(self, pairs_df: pd.DataFrame) -> None:
        """
        Fügt alle Matching-Paare aus einem DataFrame in die Union-Find-Struktur ein.

        Parameter:
        - pairs_df (pd.DataFrame): DataFrame mit den Spalten 'lid' und 'rid'.
                                   Jedes Paar wird mit union() verbunden.
        """
        for lid, rid in pairs_df[['lid', 'rid']].itertuples(index=False, name=None):
            self.union(lid, rid)

    def write_pairs_to_file(self, filename: str = "result/result_data.csv") -> None:
        """
        Speichert alle generierten Matching-Paare (lid, rid) als CSV-Datei ab
        und ergänzt eine Kommentarzeile mit Metainformationen.

        Parameter:
        - filename (str): Ziel-Dateiname (relativ zu union_find.py).

        Ablauf:
        - Bestimme absoluten Pfad.
        - Erstelle Verzeichnis bei Bedarf.
        - Schreibe alle Paare aus get_all_pairs_df().
        - Füge abschließend Kommentarzeile mit Metadaten an.
        """
        from Config import READ_LIMIT, DATA

        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, filename)

        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        df = self.get_all_pairs_df()
        df.to_csv(full_path, index=False)

        with open(full_path, "a", encoding="utf-8") as f:
            f.write(f"# Zeilenanzahl: {READ_LIMIT} Data: {DATA}\n")

