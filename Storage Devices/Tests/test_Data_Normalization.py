import os
import sys
import unittest
import pandas as pd
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Programm')))
import Data_Normalization

class TestRemoveWordsByRegex(unittest.TestCase):
    """
    Testet die Funktion `remove_words_by_regex` aus `Data_Normalization`,
    die Wörter aus einer DataFrame-Spalte anhand eines regulären Ausdrucks entfernt.
    """

    def test_remove_words_by_regex_with_string_pattern(self):
        """
        Prüft, ob Wörter korrekt entfernt werden, wenn das Muster als String übergeben wird.
        Erwartet wird, dass das Wort "Neu" (unabhängig von Groß-/Kleinschreibung) entfernt wird.
        """
        df = pd.DataFrame({
            "title": ["Neu und gebraucht Laptop mit Garantie", "Schneller Versand und guter Service"]
        })
        Data_Normalization.remove_words_by_regex(df, "title", r"\bneu\b")

        # Erwartet, dass "Neu" entfernt wurde (Groß-/Kleinschreibung ignoriert)
        self.assertTrue("neu" not in df.loc[0, "title"].lower())

    def test_remove_words_by_regex_with_compiled_pattern(self):
        """
        Prüft das Entfernen von Wörtern mit einem vorkompilierten Regex-Muster.
        Die Wörter "Fast" und "Discount" sollen entfernt werden.
        """
        df = pd.DataFrame({
            "title": ["Fast Shipping and Best Price", "No Discount Available"]
        })
        pattern = re.compile(r"\b(fast|discount)\b", flags=re.IGNORECASE)

        Data_Normalization.remove_words_by_regex(df, "title", pattern)

        # 'Fast' und 'Discount' sollten entfernt sein
        self.assertTrue("fast" not in df.loc[0, "title"].lower())
        self.assertTrue("discount" not in df.loc[1, "title"].lower())

    def test_remove_words_by_regex_removes_multiple_occurrences(self):
        """
        Prüft, ob mehrfach vorkommende Wörter vollständig entfernt werden.
        Hier sollte kein "neu" mehr im Text vorhanden sein.
        """
        df = pd.DataFrame({
            "title": ["neu neu neu gebraucht neu"]
        })
        Data_Normalization.remove_words_by_regex(df, "title", r"\bneu\b")

        # Alle "neu" sollten entfernt sein, also keine 'neu' mehr im Text
        self.assertNotIn("neu", df.loc[0, "title"].lower())

    def test_remove_words_by_regex_no_change_for_non_matching(self):
        """
        Prüft, dass der Text unverändert bleibt, wenn das Muster nicht im Text vorkommt.
        """
        df = pd.DataFrame({
            "title": ["guter Laptop ohne Probleme"]
        })
        Data_Normalization.remove_words_by_regex(df, "title", r"\bnichtvorhanden\b")

        # Text sollte unverändert bleiben, weil Muster nicht vorkommt
        self.assertEqual(df.loc[0, "title"], "guter Laptop ohne Probleme")


class TestApplyMappingEfficient(unittest.TestCase):

    def test_basic_replacement(self):
        df = pd.DataFrame({'description': ['Fast SSD and HDD Storage', 'hdd and ssd are great']})
        mapping = {'SSD': 'SolidState', 'HDD': 'HardDisk'}

        Data_Normalization.apply_mapping_efficient(df, 'description', mapping)

        expected = [
            'Fast HardDisk SolidState Storage and',
            'HardDisk SolidState and are great'
        ]
        self.assertListEqual(df['description'].tolist(), expected)

    def test_case_insensitive(self):
        df = pd.DataFrame({'description': ['sSd and hDd are fine']})
        mapping = {'ssd': 'Solid', 'HDD': 'Disk'}

        Data_Normalization.apply_mapping_efficient(df, 'description', mapping)
        self.assertEqual(df.at[0, 'description'], 'Disk Solid and are fine')

    def test_word_boundaries(self):
        df = pd.DataFrame({'description': ['ssd-based storage is common']})
        mapping = {'ssd': 'Solid'}
        # "ssd-based" should not match → keine Ersetzung
        Data_Normalization.apply_mapping_efficient(df, 'description', mapping)
        self.assertEqual(df.at[0, 'description'], 'Solid-based common is storage')

    def test_whitespace_and_deduplication(self):
        df = pd.DataFrame({'description': ['  SSD  SSD  HDD  ']})
        mapping = {'SSD': 'Flash', 'HDD': 'Flash'}

        Data_Normalization.apply_mapping_efficient(df, 'description', mapping)
        self.assertEqual(df.at[0, 'description'], 'Flash')

    def test_non_matching_text(self):
        df = pd.DataFrame({'description': ['No devices mentioned here']})
        mapping = {'SSD': 'Solid'}

        Data_Normalization.apply_mapping_efficient(df, 'description', mapping)
        self.assertEqual(df.at[0, 'description'], 'No devices here mentioned')

    def test_empty_and_non_str_values(self):
        df = pd.DataFrame({'description': ['SSD', '', None, 123]})
        mapping = {'SSD': 'Solid'}

        Data_Normalization.apply_mapping_efficient(df, 'description', mapping)

        expected = ['Solid', '', None, '123']
        self.assertEqual(df['description'].tolist(), expected)

if __name__ == "__main__":
    unittest.main()