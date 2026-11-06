import csv

from magpie.data.review import select_for_review, write_review_csv
from magpie.types import PromptRecord


def _core():
    recs = []
    for i in range(20):
        pid = f"p{i}"
        cat = ["cyber", "violence"][i % 2]
        recs.append(PromptRecord(prompt_id=pid, text=f"en source {i}", form="en",
                                 category=cat, parallel_id=pid, is_harmful=True))
        recs.append(PromptRecord(prompt_id=f"{pid}#hing", text=f"hinglish {i}", form="hing",
                                 category=cat, parallel_id=pid, is_harmful=True,
                                 covariates={"cmi": 10.0, "agreement": i / 20}))
    return recs


def test_select_is_stratified_and_prioritises_low_agreement():
    picked = select_for_review(_core(), frac=0.25, seed=0, forms=("hing",))
    # 20 hing across 2 categories (10 each) -> ceil(0.25*10)=3 per category = 6.
    assert len(picked) == 6
    assert all(p.form == "hing" for p in picked)
    # The lowest-agreement rows are chosen first within a category.
    cyber = [p for p in picked if p.category == "cyber"]
    assert max(p.covariates["agreement"] for p in cyber) < 0.5


def test_write_review_csv_pairs_source_and_translation(tmp_path):
    p = tmp_path / "review.csv"
    n = write_review_csv(_core(), p, frac=0.2)
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == n > 0
    r0 = rows[0]
    assert r0["form"] == "hing" and r0["english_source"].startswith("en source")
    assert r0["translation"].startswith("hinglish") and r0["ok (y/n)"] == ""
