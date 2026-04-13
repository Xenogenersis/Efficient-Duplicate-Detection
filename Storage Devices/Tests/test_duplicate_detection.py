import unittest
import os
import sys
from collections import defaultdict
from itertools import combinations

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Programm')))
from Calc_F1_Score import evaluate_f1_and_write_csvs
import Duplicate
import Config


FORCE_RECREATE = True

class TestDuplicateDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if FORCE_RECREATE:
            print("FORCE_RECREATE ist aktiv – Datei wird neu erstellt.")
            Duplicate.get_duplicate_pairs()
        else:
            try:
                df, metadata = Config.open_results()
                print("Metadata geladen:", metadata)
            except FileNotFoundError:
                print("Datei nicht gefunden – erstelle neu.")
                Duplicate.get_duplicate_pairs()
                df, metadata = Config.open_results()

            erwartete_meta = f"# Zeilenanzahl: {Config.READ_LIMIT} Data: {Config.DATA}"
            if metadata != erwartete_meta:
                print("Andere Meta-Daten – erstelle neu.")
                Duplicate.get_duplicate_pairs()

    def test_Result(self):
        f1 = evaluate_f1_and_write_csvs()
        self.assertGreaterEqual(
            f1, 0.5,
            msg=f"F1-Score zu niedrig: {f1:.4f} (muss >= 0.5 sein)"
        )
    def test_transitive_closure_of_result_file(self):
        """Stellt sicher, dass der Graph transitiv abgeschlossen ist."""
        df, _ = Config.open_results()

        graph = defaultdict(set)
        for _, row in df.iterrows():
            a, b = int(row['lid']), int(row['rid'])
            graph[a].add(b)
            graph[b].add(a)

        edge_set = set()
        for a in graph:
            for b in graph[a]:
                edge_set.add(tuple(sorted((a, b))))

        missing_edges = set()
        for node in graph:
            neighbors = list(graph[node])
            for u, v in combinations(neighbors, 2):
                edge = tuple(sorted((u, v)))
                if edge not in edge_set:
                    missing_edges.add(edge)

        self.assertEqual(
            missing_edges,
            set(),
            msg=f"Fehlende transitive Kanten in Ergebnis-Datei: {missing_edges}"
        )

if __name__ == '__main__':
    unittest.main()