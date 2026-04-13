import os
import sys
import unittest
import pandas as pd
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Programm')))
import Blocking  # Umbenennung von Data_Normalization zu Blocking


# --------------------------
# Tests für join_or_na
# --------------------------

class TestJoinOrNa(unittest.TestCase):
    """
    Testet die Hilfsfunktion `join_or_na` aus `Blocking`,
    die eine Liste zu einem sortierten, einzigartigen, durch Leerzeichen getrennten String zusammenfügt
    oder `None` zurückgibt, wenn die Liste leer ist.
    """

    def test_empty_list_returns_none(self):
        """
        Testet, ob eine leere Liste korrekt zu `None` konvertiert wird.
        """
        self.assertIsNone(Blocking.join_or_na([]))

    def test_unique_sorted_join(self):
        """
        Testet, ob Duplikate entfernt, Einträge sortiert und korrekt zusammengefügt werden.
        """
        input_list = ['banana', 'apple', 'banana', 'pear']
        expected = 'apple banana pear'
        self.assertEqual(Blocking.join_or_na(input_list), expected)

    def test_already_unique_sorted(self):
        """
        Testet, ob bereits sortierte und einzigartige Listen korrekt behandelt werden.
        """
        input_list = ['apple', 'banana', 'pear']
        expected = 'apple banana pear'
        self.assertEqual(Blocking.join_or_na(input_list), expected)

    def test_case_sensitivity(self):
        """
        Testet, ob die Funktion case-sensitive ist und Buchstaben in ihrer Originalform belässt.
        """
        input_list = ['Banana', 'banana', 'Apple']
        expected = 'Apple Banana banana'
        self.assertEqual(Blocking.join_or_na(input_list), expected)


# --------------------------
# Tests für insert_blocking_key_column
# --------------------------

class TestInsertBlockingKeyColumn(unittest.TestCase):
    """
    Testet die Funktion `insert_blocking_key_column` aus `Blocking`,
    die eine neue Spalte mit extrahierten Mustern aus einer Textspalte in einen DataFrame einfügt.
    """

    def setUp(self):
        """
        Initialisiert einen Beispiel-DataFrame für die Tests.
        """
        self.df = pd.DataFrame({
            'text': [
                'This is a Test with KEY123 and key456.',
                'Another line with key789 and KEY000.',
                'No keys here!',
                '',
                None
            ]
        })

    def test_insert_with_regex_string(self):
        """
        Testet das Einfügen der Spalte mit einem regulären Ausdruck als String.
        """
        pattern = r'key\d+'
        Blocking.insert_blocking_key_column(self.df, 'text', pattern, 1, 'blocking_key')
        expected = ['key456', 'key789', None, None, None]
        self.assertListEqual(self.df['blocking_key'].tolist(), expected)

    def test_insert_with_compiled_pattern(self):
        """
        Testet die Verwendung eines vorkompilierten regulären Ausdrucks mit Groß-/Kleinschreibungs-Ignorierung.
        """
        pattern = re.compile(r'key\d+', re.IGNORECASE)
        Blocking.insert_blocking_key_column(self.df, 'text', pattern, 1, 'blocking_key')
        expected = ['key123 key456', 'key000 key789', None, None, None]
        self.assertListEqual(self.df['blocking_key'].tolist(), expected)

    def test_insertion_position_and_column_name(self):
        """
        Testet, ob die Spalte korrekt an der angegebenen Position mit gewünschtem Namen eingefügt wird.
        """
        pattern = r'key\d+'
        Blocking.insert_blocking_key_column(self.df, 'text', pattern, 0, 'new_col')
        self.assertEqual(self.df.columns[0], 'new_col')
        expected = ['key456', 'key789', None, None, None]
        self.assertListEqual(self.df['new_col'].tolist(), expected)

    def test_non_string_values(self):
        """
        Testet, ob nicht-string-Werte (z. B. Zahlen oder None) robust behandelt werden.
        """
        df = pd.DataFrame({'text': [123, None, 'key123 key456']})
        pattern = r'key\d+'
        Blocking.insert_blocking_key_column(df, 'text', pattern, 1, 'keys')
        expected = [None, None, 'key123 key456']
        self.assertListEqual(df['keys'].tolist(), expected)

    def test_duplicates_and_sorting(self):
        """
        Testet, ob doppelte Treffer entfernt und Ergebnisse sortiert werden.
        """
        df = pd.DataFrame({'text': ['key456 KEY123 key123']})
        pattern = re.compile(r'key\d+', re.IGNORECASE)
        Blocking.insert_blocking_key_column(df, 'text', pattern, 1, 'blocking_key')
        expected = ['key123 key456']
        self.assertListEqual(df['blocking_key'].tolist(), expected)

    def test_empty_match_list_gives_none(self):
        """
        Testet, ob bei fehlenden Treffern `None` in der neuen Spalte steht.
        """
        df = pd.DataFrame({'text': ['no match']})
        pattern = r'key\d+'
        Blocking.insert_blocking_key_column(df, 'text', pattern, 1, 'blocking_key')
        self.assertIsNone(df['blocking_key'].iloc[0])

    def test_invalid_regex_raises_error(self):
        """
        Testet, ob ein fehlerhafter regulärer Ausdruck korrekt einen Fehler auslöst.
        """
        df = pd.DataFrame({'text': ['some text']})
        pattern = r'key['  # Ungültiger Regex
        with self.assertRaises(re.error):
            Blocking.insert_blocking_key_column(df, 'text', pattern, 1, 'blocking_key')


# --------------------------
# Tests für apply_regex_mapping
# --------------------------

class TestApplyRegexMapping(unittest.TestCase):
    """
    Umfassende Tests für die Funktion `apply_regex_mapping` aus `Blocking`.
    Diese ersetzt Begriffe in einer DataFrame-Spalte mithilfe von Regex-Mapping
    und bereinigt optional Duplikate mithilfe eines zweiten regulären Ausdrucks.
    """

    def test_mapping_with_duplicate_cleanup(self):
        """
        Standardfall: Mapping + duplicate_regex → Duplikate entfernen, sortieren, lowercased.
        """
        df = pd.DataFrame({'text': [
            'Intel Core i7 10th Gen and core i7 processor',
            'AMD Ryzen and RYZEN performance',
            'No match here',
            None
        ]})
        mapping = {
            r'\bcore\s*i7\b': 'Intel-i7',
            r'\bryzen\b': 'AMD-Ryzen',
        }
        duplicate_regex = r'Intel-i7|AMD-Ryzen'

        Blocking.apply_regex_mapping(df, 'text', mapping, duplicate_regex)

        expected = ['intel-i7', 'amd-ryzen', None, None]
        self.assertListEqual(df['text'].tolist(), expected)

    def test_mapping_only_without_duplicates(self):
        """
        Testet, ob das reine Mapping funktioniert, wenn kein duplicate_regex angegeben ist.
        """
        df = pd.DataFrame({'text': ['intel core i7 and ryzen 5']})
        mapping = {
            r'\bcore\s*i7\b': 'Intel-i7',
            r'\bryzen\b': 'AMD-Ryzen'
        }

        Blocking.apply_regex_mapping(df, 'text', mapping)
        expected = ['intel Intel-i7 and AMD-Ryzen 5']
        self.assertEqual(df['text'].iloc[0], expected[0])

    def test_duplicate_regex_as_pattern_object(self):
        """
        duplicate_regex wird als vorkompiliertes Pattern übergeben.
        """
        df = pd.DataFrame({'text': ['Intel-i7 and Intel-i7 and AMD-Ryzen']})
        pattern = re.compile(r'Intel-i7|AMD-Ryzen', re.IGNORECASE)
        Blocking.apply_regex_mapping(df, 'text', {}, pattern)
        expected = ['amd-ryzen intel-i7']
        self.assertEqual(df['text'].iloc[0], expected[0])

    def test_non_string_values_in_column(self):
        """
        Stellt sicher, dass nicht-String-Werte (int, None) korrekt verarbeitet werden.
        """
        df = pd.DataFrame({'text': [123, None, 'core i7']})
        mapping = {r'\bcore\s*i7\b': 'Intel-i7'}
        duplicate_regex = r'Intel-i7'
        Blocking.apply_regex_mapping(df, 'text', mapping, duplicate_regex)
        expected = [None, None, 'intel-i7']
        self.assertListEqual(df['text'].tolist(), expected)

    def test_empty_string_input(self):
        """
        Leere Strings sollen zu None werden, wenn kein Treffer vorliegt.
        """
        df = pd.DataFrame({'text': ['']})
        mapping = {r'\bcore\s*i7\b': 'Intel-i7'}
        duplicate_regex = r'Intel-i7'
        Blocking.apply_regex_mapping(df, 'text', mapping, duplicate_regex)
        self.assertIsNone(df['text'].iloc[0])

    def test_no_matching_mapping_rules(self):
        """
        Testet Verhalten, wenn Mapping keine Treffer hat (d.h. keine Änderung).
        """
        df = pd.DataFrame({'text': ['nothing to replace here']})
        mapping = {r'\bdoesnotmatch\b': 'irrelevant'}
        Blocking.apply_regex_mapping(df, 'text', mapping)
        self.assertEqual(df['text'].iloc[0], 'nothing to replace here')

    def test_whitespace_cleanup(self):
        """
        Überprüft, ob überschüssige Leerzeichen nach der Ersetzung entfernt werden.
        """
        df = pd.DataFrame({'text': ['core i7   and   core i7']})
        mapping = {r'\bcore\s*i7\b': 'Intel-i7'}
        duplicate_regex = r'Intel-i7'
        Blocking.apply_regex_mapping(df, 'text', mapping, duplicate_regex)
        expected = ['intel-i7']
        self.assertEqual(df['text'].iloc[0], expected[0])


if __name__ == '__main__':
    unittest.main()