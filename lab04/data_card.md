# Data card: UCI Adult (Census Income), processed for Lab 4

## Source

* **Dataset:** Adult / Census Income. B. Becker and R. Kohavi, UCI Machine Learning Repository, 1996. DOI [10.24432/C5XW20](https://doi.org/10.24432/C5XW20).
* **Origin:** extracted by Barry Becker from the **1994 US Census Bureau** database (Current Population Survey). Records were filtered to `age > 16`, `AGI > 100`, `AFNLWGT > 1`, `HRSWK > 0`.
* **Access used here:** `sklearn.datasets.fetch_openml(name="adult", version=2, as_frame=True)`, OpenML data id **1590**. This is the UCI `adult.data` and `adult.test` files merged into one table.
* **Raw snapshot:** `data/adult/raw.csv` (git-ignored, re-created by the pipeline), SHA256 `8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc`.

## License

UCI lists the dataset under **Creative Commons Attribution 4.0 (CC BY 4.0)**: it may be shared and adapted, including commercially, with attribution to the creators above. OpenML reports the licence as "Public".

## Size and structure

| | Value |
|---|---|
| Rows | 48,842 |
| Columns | 15 (14 features + label) |
| Numeric features | 6: age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week |
| Categorical features | 8: workclass, education, marital-status, occupation, relationship, race, sex, native-country |
| Label | `income` (`class` in OpenML): `<=50K` 76.1%, `>50K` 23.9% |
| Missing values (`?`) | workclass 2,799, occupation 2,809, native-country 857. 3,620 rows (7.4%) have at least one |
| Exact duplicate rows | 52 |
| Raw size on disk (CSV) | about 5.2 MB |

**Processed dataset** (output of `lab04/src/lab4.py`, with derived features):

| Split | Shape | SHA256 |
|---|---|---|
| train (70%, 24 duplicates dropped) | 34,165 × 108 | `8ca18e7745f7c765b7799acbacbd32ae629a6e3dcfc8cfc29e3ae95a16090c6b` |
| test (30%, untouched) | 14,653 × 108 | `2f96451ec7c49d28a481fc4fed0ea936fec0363d40a13bba0026a4550513c098` |

107 model inputs: 8 numeric (6 original + `capital_net` + `hours_per_age`, median-imputed, clipped to the train 1st/99th percentile, standardised) and 99 one-hot columns. The last column is the binary label (1 = `>50K`). The hashes and the full config are in [`results/run_config.yaml`](results/run_config.yaml). The fitted transformer is [`results/preprocessor.joblib`](results/preprocessor.joblib).

## Intended use

Teaching and benchmarking of tabular binary classification and data-pipeline practice. It is **not** suitable for real decisions about individuals (credit, hiring, insurance). The data is 30 years old, the label is a coarse income threshold, and the biases below make such use harmful.

## Known biases and limitations

* **Sex imbalance:** 66.8% male, 33.2% female. The positive rate is 30.4% for men and 10.9% for women. A model trained on this data learns that gap and reproduces it, which is why the dataset is a standard fairness benchmark.
* **Race imbalance:** 85.5% White, 9.6% Black, 3.1% Asian-Pac-Islander, 1.0% Amer-Indian-Eskimo, 0.8% Other. The small groups give unreliable per-group metrics. Positive rate: White 25.4%, Black 12.1%, Amer-Indian-Eskimo 11.7%.
* **Geography:** 89.7% of rows have native-country United-States. Many countries have fewer than 100 rows.
* **Time:** 1994 dollars and a 1994 labour market. The `>50K` threshold is not inflation-adjusted, so its meaning has drifted.
* **Missingness is not random:** all 2,799 rows with a missing workclass also have a missing occupation, and the other 10 missing occupations are all `Never-worked`. Only 9.5% of rows with a missing workclass are `>50K`, against 23.9% overall. Mode imputation ("Private", "Prof-specialty") hides this signal.
* **Top-coding:** capital-gain 99999 (244 rows) and hours-per-week 99 (137 rows) are censored values, not real amounts.
* **Label noise:** 5 groups (10 rows) have identical 14 features and different labels. 4 rows contradict the data dictionary (relationship `Wife` with sex `Male` or `Husband` with `Female`). One row is an 18-year-old with a Masters degree. See [`results/label_audit.json`](results/label_audit.json).
* **Proxy features:** `relationship` (Husband/Wife) encodes sex, and `marital-status` and `native-country` correlate with protected attributes. Dropping `sex` and `race` does not remove the bias.

## Privacy concerns

* The data is **de-identified census microdata**: there are no names, addresses or IDs. It has been public since 1996.
* It still contains **quasi-identifiers** (age, sex, race, native-country, education, occupation, marital status). In combination these can single out rare individuals, for example a 90-year-old from a small country in an uncommon occupation. Rows should not be joined with other sources.
* `fnlwgt` is a Census sampling weight (the estimated number of people a row represents). It is not a personal attribute, but it is near-unique per row and works as a record fingerprint.
* Sensitive attributes (race, sex, marital status, income) are present. Outputs should be reported in aggregate only.

## Processing summary

Ingestion (no modification) → validation on all rows (reported, nothing dropped) → stratified 70/30 split (seed 42) → train-only deduplication → imputation, percentile clipping, scaling and one-hot encoding fitted on train only → derived features `capital_net` and `hours_per_age`. Reproduce with `python lab04/src/lab4.py`.
