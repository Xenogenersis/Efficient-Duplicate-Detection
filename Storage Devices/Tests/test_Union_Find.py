import sys
import unittest
from unittest.mock import patch
import tempfile
import os

# Pfad zur Programmstruktur hinzufügen
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Programm')))
import Union_Find



class TestUnionFind(unittest.TestCase):
    """
    Testet die Funktionalität der UnionFind-Klasse aus dem Modul Union_Find,
    insbesondere das Hinzufügen von Elementen, Vereinigung (Union),
    Finden der Repräsentanten (Find), Gruppenermittlung und Ausgabe von Paaren.
    """

    def test_add_and_find(self):
        """
        Prüft, ob hinzugefügte Elemente existieren und ob die Find-Operation
        auf das gleiche Element den gleichen Repräsentanten liefert,
        sowie dass unterschiedliche Elemente unterschiedliche Repräsentanten haben,
        wenn sie nicht vereinigt wurden.
        """
        uf = Union_Find.UnionFind()
        uf.add('A')
        uf.add('B')
        self.assertEqual(uf.find('A'), uf.find('A'))
        self.assertNotEqual(uf.find('A'), uf.find('B'))

    def test_union_and_find(self):
        """
        Prüft, ob nach einer Union-Operation zwischen zwei Elementen
        deren Repräsentanten identisch sind.
        """
        uf = Union_Find.UnionFind()
        uf.union('A', 'B')
        self.assertEqual(uf.find('A'), uf.find('B'))

    def test_get_groups(self):
        """
        Testet, ob die Methode get_groups() die korrekten Mengen
        an verbundenen Elementen zurückgibt, einschließlich einzelner Elemente.
        """
        uf = Union_Find.UnionFind()
        uf.union('A', 'B')
        uf.union('B', 'C')
        uf.add('D')
        groups = uf.get_groups()
        self.assertTrue(any(set(group) == {'A', 'B', 'C'} for group in groups))
        self.assertTrue(any(set(group) == {'D'} for group in groups))

    def test_get_all_pairs_df(self):
        """
        Prüft, ob alle Paarungen (Kanten) innerhalb verbundener Gruppen
        korrekt als DataFrame zurückgegeben werden.
        Einzelne Elemente ohne Verbindung sollten keine Paare liefern.
        """
        uf = Union_Find.UnionFind()
        uf.union(1, 2)
        uf.union(2, 3)
        uf.add(4)
        pairs_df = uf.get_all_pairs_df()
        expected_pairs = [(1, 2), (1, 3), (2, 3)]

        pairs_list = list(pairs_df.itertuples(index=False, name=None))
        self.assertCountEqual(pairs_list, expected_pairs)

    @patch('Config.READ_LIMIT', new=10)
    @patch('Config.DATA', new=1)
    def test_write_pairs_to_file(self):
        """
        Testet das Schreiben der Paare in eine CSV-Datei.
        Prüft, ob die Datei die korrekten Paarzeilen sowie eine Metadatenzeile enthält.
        """
        uf = Union_Find.UnionFind()
        uf.union('X', 'Y')
        uf.union('Y', 'Z')
        uf.add('W')

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_output.csv")
            uf.write_pairs_to_file(filename=filepath)

            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')

            expected_lines = ['lid,rid','X,Y', 'X,Z', 'Y,Z', '# Zeilenanzahl: 10 Data: 1']
            self.assertCountEqual(lines, expected_lines)

    @patch('Config.READ_LIMIT', new=10)
    @patch('Config.DATA', new=1)
    def test_write_pairs_to_file_empty_unionfind(self):
        """
        Prüft, ob beim Schreiben einer leeren UnionFind-Instanz
        die Ausgabedatei nur den Header und die Metadatenzeile enthält.
        """
        uf = Union_Find.UnionFind()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "empty.csv")
            with patch('Union_Find.__file__', os.path.join(tmpdir, 'union_find.py')):
                uf.write_pairs_to_file(filename="empty.csv")

            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.read().strip().splitlines()

            expected_lines = [
                "lid,rid",
                "# Zeilenanzahl: 10 Data: 1"
            ]
            self.assertEqual(lines, expected_lines)


if __name__ == "__main__":
    unittest.main()