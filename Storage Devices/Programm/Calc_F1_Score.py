import Config
import os
import pandas as pd
from Union_Find import UnionFind
from Duplicate import get_duplicate_pairs
import time


def mergmerge_mapping(result: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    result = result.merge(mapping_df.add_prefix('l'), left_on='lid', right_on='lid', how='left')
    result = result.merge(mapping_df.add_prefix('r'), left_on='rid', right_on='rid', how='left')
    return result

def evaluate_f1_and_write_csvs() -> float:
    start_total = time.time()

    ground_truth = Config.open_ground_truth()
    matched_pairs, _ = Config.open_results()

    uf_ground_truth = UnionFind()
    uf_matched_pairs = UnionFind()
    uf_ground_truth.add_pairs_from_df(ground_truth)
    uf_matched_pairs.add_pairs_from_df(matched_pairs)

    ground_truth = uf_ground_truth.get_all_pairs_df()
    matched_pairs = uf_matched_pairs.get_all_pairs_df()

    intersection = pd.merge(ground_truth, matched_pairs, on=['lid', 'rid'], how='inner')
    diff_df = pd.merge(ground_truth, matched_pairs, on=['lid', 'rid'], how='left', indicator=True)
    only_in_ground_truth = diff_df[diff_df['_merge'] == 'left_only'].drop(columns=['_merge'])
    diff_df = pd.merge(matched_pairs, ground_truth, on=['lid', 'rid'], how='left', indicator=True)
    only_in_matched = diff_df[diff_df['_merge'] == 'left_only'].drop(columns=['_merge'])

    mapping_df = Config.open_tuples()
    intersection = mergmerge_mapping(intersection, mapping_df)
    only_in_ground_truth = mergmerge_mapping(only_in_ground_truth, mapping_df)
    only_in_matched = mergmerge_mapping(only_in_matched, mapping_df)
    ground_truth = mergmerge_mapping(ground_truth, mapping_df)
    matched_pairs = mergmerge_mapping(matched_pairs, mapping_df)

    t0 = time.time()
    target_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "F1_Score")
    os.makedirs(target_path, exist_ok=True)

    ground_truth.to_csv(os.path.join(target_path, "ground_truth.csv"), index=False, encoding='utf-8')
    matched_pairs.to_csv(os.path.join(target_path, "matched_pairs.csv"), index=False, encoding='utf-8')
    intersection.to_csv(os.path.join(target_path, "True_Positive.csv"), index=False, encoding='utf-8')
    only_in_ground_truth.to_csv(os.path.join(target_path, "False_Negative.csv"), index=False, encoding='utf-8')
    only_in_matched.to_csv(os.path.join(target_path, "False_Positive.csv"), index=False, encoding='utf-8')

    tp = len(intersection)
    fn = len(only_in_ground_truth)
    fp = len(only_in_matched)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"[{time.time() - t0:.2f}s] F1, Precision und Recall berechnet")
    print(f"F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
    print(f"[{time.time() - start_total:.2f}s] Gesamtzeit")

    return f1


if __name__ == "__main__":
    get_duplicate_pairs()
    evaluate_f1_and_write_csvs()