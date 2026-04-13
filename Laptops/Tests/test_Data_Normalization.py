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

if __name__ == "__main__":
    unittest.main()