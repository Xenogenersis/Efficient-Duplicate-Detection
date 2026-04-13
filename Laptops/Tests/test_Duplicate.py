import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Programm')))
import Duplicate
import Union_Find

class TestExtendUnionJaccard(unittest.TestCase):

    def setUp(self):
        # Beispiel-Daten für Gruppen
        self.df1 = pl.DataFrame({
            "id": ["1", "2", "3"],
            "text": ["Haus", "haus", "Baum"]
        })
        self.df2 = pl.DataFrame({
            "id": ["4", "5"],
            "text": ["Ich gehe spazieren", "Ich gehe spaziere"]
        })

        # Iterator simulierend (wie von groupby)
        self.groups = iter([
            (("group1",), self.df1),
            (("group2",), self.df2)
        ])

    def test_process_group_similarity(self):
        # _process_group sollte Paare zurückgeben, die ähnlich sind (z.B. "Haus" und "haus")
        pairs = Duplicate._process_group((self.df1, "text"))
        # "Haus" und "haus" sollten ähnlich sein -> (1, 2)
        self.assertIn(("1", "2"), pairs)
        # "Haus" und "Baum" sind nicht ähnlich
        self.assertNotIn(("1", "3"), pairs)

    def test_extend_union_jaccard_creates_unionfind(self):
        uf = Duplicate.extend_union_jaccard(self.groups, "text")
        # Die IDs 1 und 2 sollten verbunden sein
        self.assertEqual(uf.find("1"), uf.find("2"))
        # Die IDs 4 und 5 sollten verbunden sein (Katze vs Katzen)
        self.assertEqual(uf.find("4"), uf.find("5"))
        # IDs 1 und 4 sollten nicht verbunden sein (verschiedene Gruppen)
        self.assertNotEqual(uf.find("1"), uf.find("4"))

    def test_extend_union_jaccard_with_existing_unionfind(self):
        uf = Union_Find.UnionFind()
        # Vorher zwei IDs verbinden
        uf.union("1", "3")
        # Erstelle neuen Iterator, da Iterator verbraucht wurde
        groups = iter([
            (("group1",), self.df1),
            (("group2",), self.df2)
        ])
        uf = Duplicate.extend_union_jaccard(groups, "text", uf)
        # Verbindung 1-3 soll bestehen bleiben
        self.assertEqual(uf.find("1"), uf.find("3"))
        # Verbindung 1-2 wird hinzugefügt
        self.assertEqual(uf.find("1"), uf.find("2"))

    @patch('Duplicate.cpu_count', return_value=1)
    def test_parallel_pool_called(self, _):
        # Testet, dass multiprocessing Pool benutzt wird (durch Patch cpu_count=1)
        groups = iter([
            (("group1",), self.df1)
        ])
        uf = Duplicate.extend_union_jaccard(groups, "text")
        self.assertIsInstance(uf, Union_Find.UnionFind)


if __name__ == "__main__":
    unittest.main()