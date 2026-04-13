import os
import sys
import unittest
from unittest.mock import patch, mock_open
import pandas as pd
import polars as pl

# Füge den Pfad zum Programmverzeichnis hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Programm')))
import Config


# --------------------------
# Tests für open_tuples
# --------------------------

class TestOpenTuples(unittest.TestCase):
    """
    Testet das Laden von Tupeln mittels der Funktion `open_tuples` aus `Programm.Config`.
    """

    @patch('Config.READ_LIMIT', new=10)
    @patch('Config.TUPLES_FILES', new={'valid_key': 'dummy_path.csv'})
    @patch('Config.DATA', new='valid_key')
    @patch('Config.pd.read_csv')
    def test_valid_open_tuples(self, mock_read_csv):
        """
        Prüft, ob open_tuples den korrekten DataFrame zurückgibt
        und die Lesegrenze (READ_LIMIT) beachtet.
        """
        mock_read_csv.return_value = pd.DataFrame({'title': ['Test']})

        with patch('builtins.print'):
            df = Config.open_tuples()

        mock_read_csv.assert_called_once_with('dummy_path.csv', nrows=10)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('title', df.columns)


# --------------------------
# Tests für open_ground_truth
# --------------------------

class TestOpenGroundTruth(unittest.TestCase):
    """
    Testet das Laden von Ground-Truth-Matching-Paaren mit `open_matching_pairs`.
    """

    @patch('Config.os.path.exists', return_value=True)
    @patch('Config.pd.read_csv')
    def test_valid_open_matching_pairs(self, mock_read_csv, mock_exists):
        """
        Prüft, ob unsortierte Paare korrekt sortiert und normalisiert werden.
        """
        mock_df = pd.DataFrame({'lid': [5, 2], 'rid': [1, 4]})
        mock_read_csv.return_value = mock_df

        df = Config.open_ground_truth()

        expected_df = pd.DataFrame({'lid': [1, 2], 'rid': [5, 4]}).sort_values(by=['lid', 'rid']).reset_index(drop=True)
        self.assertTrue(df.equals(expected_df))


# --------------------------
# Tests für open_results
# --------------------------

class TestOpenResults(unittest.TestCase):
    """
    Testet das Laden und Sortieren der Result-Datei mittels der Funktion `open_results`.
    """

    @patch('Config.os.path.exists', return_value=True)
    @patch('Config.open', new_callable=mock_open)
    def test_open_results(self, mock_file, mock_exists):
        # Simulierter Dateiinhalt
        mock_file().readlines.return_value = [
            'lid,rid\n',
            '5,1\n',
            '2,4\n',
            '# Zeilenanzahl: 10 Data: 1\n'
        ]

        df, metadata = Config.open_results()

        expected_df = pd.DataFrame({'lid': [1, 2], 'rid': [5, 4]})
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected_df)
        self.assertEqual(metadata, '# Zeilenanzahl: 10 Data: 1')

    @patch('Config.os.path.exists', return_value=False)
    def test_open_results_file_not_found(self, mock_exists):
        with self.assertRaises(FileNotFoundError) as context:
            Config.open_results()
        self.assertIn("existiert nicht", str(context.exception))


# --------------------------
# Tests für create_data_frame_as_file
# --------------------------

class TestCreateDataFrameAsFile(unittest.TestCase):
    """
    Testet das Speichern von DataFrames (pandas und polars) sowie GroupBy-Objekten
    als CSV-Datei mit `create_data_frame_as_file`.
    Überprüft auch das Verhalten bei ungültigen Eingaben.
    """

    def get_target_path(self, relative_path):
        """
        Hilfsmethode, um den absoluten Pfad relativ zum Config-Modul zu bestimmen.
        """
        config_dir = os.path.dirname(os.path.abspath(Config.__file__))
        return os.path.join(config_dir, relative_path)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pandas.DataFrame.to_csv")
    @patch("os.makedirs")
    @patch("Config.READ_LIMIT", new=1234)
    @patch("Config.DATA", new=1)
    def test_pandas_dataframe(self, mock_makedirs, mock_to_csv, mock_open_file):
        """
        Testet das Speichern eines pandas DataFrames
        und das Anhängen der Metadatenzeile.
        """
        df = pd.DataFrame({"a": [1, 2, 3]})
        rel_path = "result/test_pandas.csv"
        abs_path = self.get_target_path(rel_path)

        Config.create_data_frame_as_file(df, filename=rel_path)

        mock_makedirs.assert_called_once_with(os.path.dirname(abs_path), exist_ok=True)
        mock_to_csv.assert_called_once_with(abs_path, index=False)
        mock_open_file.assert_called_once_with(abs_path, "a", encoding="utf-8")
        handle = mock_open_file()
        handle.write.assert_called_once_with("# Zeilenanzahl: 1234 Data: 1\n")

    @patch("builtins.open", new_callable=mock_open)
    @patch("polars.DataFrame.write_csv")
    @patch("os.makedirs")
    @patch("Config.READ_LIMIT", new=1234)
    @patch("Config.DATA", new=1)
    def test_polars_dataframe(self, mock_makedirs, mock_write_csv, mock_open_file):
        """
        Testet das Speichern eines polars DataFrames
        und das Anhängen der Metadatenzeile.
        """
        df = pl.DataFrame({"x": [10, 20, 30], "y": [100, 200, 300]})
        rel_path = "result/test_polars.csv"
        abs_path = self.get_target_path(rel_path)

        Config.create_data_frame_as_file(df, filename=rel_path)

        mock_makedirs.assert_called_once_with(os.path.dirname(abs_path), exist_ok=True)
        mock_write_csv.assert_called_once_with(abs_path)
        mock_open_file.assert_called_once_with(abs_path, "a", encoding="utf-8")
        handle = mock_open_file()
        handle.write.assert_called_once_with("# Zeilenanzahl: 1234 Data: 1\n")

    @patch("builtins.open", new_callable=mock_open)
    @patch("polars.concat")
    @patch("polars.DataFrame.write_csv")
    @patch("os.makedirs")
    @patch("Config.READ_LIMIT", new=1234)
    @patch("Config.DATA", new=1)
    def test_polars_groupby(self, mock_makedirs, mock_write_csv, mock_concat, mock_open_file):
        """
        Testet das Speichern eines polars GroupBy-Objekts,
        bei dem vor dem Schreiben ein DataFrame zusammengesetzt wird.
        """
        df = pl.DataFrame({"a": [1, 2, 2], "b": [3, 4, 5]})
        groupby = df.group_by("a")

        mock_concat.return_value = df
        rel_path = "result/test_groupby.csv"
        abs_path = self.get_target_path(rel_path)

        Config.create_data_frame_as_file(groupby, filename=rel_path)

        mock_makedirs.assert_called_once_with(os.path.dirname(abs_path), exist_ok=True)
        mock_concat.assert_called_once()
        mock_write_csv.assert_called_once_with(abs_path)
        mock_open_file.assert_called_once_with(abs_path, "a", encoding="utf-8")
        handle = mock_open_file()
        handle.write.assert_called_once_with("# Zeilenanzahl: 1234 Data: 1\n")

    @patch("Config.READ_LIMIT", new=1234)
    def test_invalid_input_type(self):
        """
        Prüft, ob bei ungültigem Eingabetyp eine TypeError ausgelöst wird.
        """
        with self.assertRaises(TypeError):
            Config.create_data_frame_as_file("ungültiger Typ", filename="result/invalid.csv")


if __name__ == "__main__":
    unittest.main()