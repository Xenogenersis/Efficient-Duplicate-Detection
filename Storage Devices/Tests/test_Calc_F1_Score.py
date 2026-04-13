import os
import sys
import unittest
import pandas as pd
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Programm')))
import Config
from Calc_F1_Score import evaluate_f1_and_write_csvs


def manual_evaluate(match_pairs: list, ground_truth: pd.DataFrame) -> float:
    """
    Berechnet manuell den F1-Score anhand der Übereinstimmungen (match_pairs)
    und der Ground-Truth-Daten (ground_truth).

    Args:
        match_pairs (list): Liste von Tupeln (lid, rid), die die gefundenen Matches repräsentieren.
        ground_truth (pd.DataFrame): DataFrame mit den korrekten Zuordnungen in den Spalten 'lid' und 'rid'.

    Returns:
        float: Der berechnete F1-Score.
    """
    gt = list(zip(ground_truth['lid'], ground_truth['rid']))
    tp = len(set(match_pairs).intersection(set(gt)))  # True Positives
    fp = len(set(match_pairs).difference(set(gt)))    # False Positives
    fn = len(set(gt).difference(set(match_pairs)))    # False Negatives
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return f1


class TestF1Evaluation(unittest.TestCase):
    """
    Testklasse für die Funktion `evaluate_f1_and_write_csvs` aus `Calc_F1_Score`.
    Testet die F1-Score Berechnung gegen eine manuelle Implementierung
    sowie verschiedene Randfälle (perfekter Treffer, kein Treffer, leere Daten).
    """

    def setUp(self):
        """
        Vorbereitung der Testdaten und Mocking der Dateizugriffe,
        damit keine echten Dateien gelesen oder geschrieben werden.
        """
        # Patch für DataFrame.to_csv, damit kein Schreibzugriff erfolgt
        patcher = patch('pandas.DataFrame.to_csv', autospec=True)
        self.mock_to_csv = patcher.start()
        self.addCleanup(patcher.stop)

        # Ground Truth DataFrame mit korrekten Zuordnungen
        self.ground_truth_df = pd.DataFrame({
            'lid': [1, 2, 3],
            'rid': [101, 102, 103]
        })

        # Gefundene Match-Paare, inkl. eines falschen Matches (4, 999)
        self.match_pairs = pd.DataFrame({
            'lid': [1, 2, 4],
            'rid': [101, 102, 999]
        })
        self.match_pairs_tuples = list(zip(self.match_pairs['lid'], self.match_pairs['rid']))

        # Zusätzliche Daten, falls benötigt (hier z.B. IDs und Namen)
        self.tuples = pd.DataFrame({
            'id': [1, 2, 3, 4, 999, 101, 102, 103],
            'name': [1, 2, 3, 4, 999, 101, 102, 103]
        })

        # Mock-Funktionen in Config, damit keine echten Dateien geladen werden
        Config.open_ground_truth = lambda: self.ground_truth_df
        Config.open_results = lambda: (pd.DataFrame(self.match_pairs, columns=['lid', 'rid']), "footer_line")
        Config.open_tuples = lambda: self.tuples

    def test_f1_score_matches_manual(self):
        """
        Testet, ob der von `evaluate_f1_and_write_csvs` berechnete F1-Score
        mit der manuellen Berechnung übereinstimmt.
        """
        expected_f1 = manual_evaluate(self.match_pairs_tuples, self.ground_truth_df)
        actual_f1 = evaluate_f1_and_write_csvs()
        self.assertAlmostEqual(actual_f1, expected_f1, places=5,
                               msg=f"Expected F1: {expected_f1}, but got {actual_f1}")

    def test_perfect_match(self):
        """
        Testet den Fall, bei dem die gefundenen Paare exakt mit der Ground Truth übereinstimmen,
        der F1-Score also 1.0 sein muss.
        """
        perfect_pairs = list(zip(self.ground_truth_df['lid'], self.ground_truth_df['rid']))
        Config.open_results = lambda: (pd.DataFrame(self.ground_truth_df, columns=['lid', 'rid']), "footer_line")
        expected_f1 = manual_evaluate(perfect_pairs, self.ground_truth_df)
        actual_f1 = evaluate_f1_and_write_csvs()
        self.assertEqual(expected_f1, 1.0)
        self.assertAlmostEqual(actual_f1, 1.0, places=5)

    def test_no_match(self):
        """
        Testet den Fall, bei dem keine Übereinstimmungen vorliegen,
        der F1-Score also 0 sein muss.
        """
        no_match_pairs = [(999, 999), (888, 888)]
        Config.open_results = lambda: (pd.DataFrame({'lid': [999, 888], 'rid': [999, 888]}), "footer_line")
        expected_f1 = manual_evaluate(no_match_pairs, self.ground_truth_df)
        actual_f1 = evaluate_f1_and_write_csvs()
        self.assertEqual(expected_f1, 0)
        self.assertEqual(actual_f1, 0)

    def test_empty_match_pairs(self):
        """
        Testet das Verhalten, wenn keine Match-Paare übergeben werden,
        der F1-Score sollte 0 sein.
        """
        empty_pairs = []
        Config.open_results = lambda: (pd.DataFrame(columns=['lid', 'rid']), "footer_line")
        expected_f1 = manual_evaluate(empty_pairs, self.ground_truth_df)
        actual_f1 = evaluate_f1_and_write_csvs()
        self.assertEqual(expected_f1, 0)
        self.assertEqual(actual_f1, 0)

    def test_empty_ground_truth(self):
        """
        Testet das Verhalten bei leerer Ground Truth,
        der F1-Score sollte 0 sein, unabhängig von Matches.
        """
        empty_gt_df = pd.DataFrame(columns=['lid', 'rid'])
        Config.open_ground_truth = lambda: empty_gt_df
        Config.open_results = lambda: (pd.DataFrame(self.match_pairs, columns=['lid', 'rid']), "footer_line")
        expected_f1 = manual_evaluate(self.match_pairs_tuples, empty_gt_df)
        actual_f1 = evaluate_f1_and_write_csvs()
        self.assertEqual(expected_f1, 0)
        self.assertEqual(actual_f1, 0)


if __name__ == "__main__":
    unittest.main()