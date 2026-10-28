# Validation report

Data: UCI Adult (OpenML id 1590), 48842 rows x 15 columns, raw SHA256 `8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc`.

Checks run on the full raw data before the split. Rows are **reported, never dropped**. Severity: `error` = schema violation, `warning` = plausible data issue, `info` = expected, counted for documentation.

| Check | Column | Rule | Severity | Violations | Example rows |
|---|---|---|---|---:|---|
| columns | * | exactly the 15 schema columns | error | 0 |  |
| type | age | integer dtype | error | 0 |  |
| allowed_values | workclass | value in data dictionary (incl. '?') | error | 0 |  |
| type | fnlwgt | integer dtype | error | 0 |  |
| allowed_values | education | value in data dictionary (incl. '?') | error | 0 |  |
| type | education-num | integer dtype | error | 0 |  |
| allowed_values | marital-status | value in data dictionary (incl. '?') | error | 0 |  |
| allowed_values | occupation | value in data dictionary (incl. '?') | error | 0 |  |
| allowed_values | relationship | value in data dictionary (incl. '?') | error | 0 |  |
| allowed_values | race | value in data dictionary (incl. '?') | error | 0 |  |
| allowed_values | sex | value in data dictionary (incl. '?') | error | 0 |  |
| type | capital-gain | integer dtype | error | 0 |  |
| type | capital-loss | integer dtype | error | 0 |  |
| type | hours-per-week | integer dtype | error | 0 |  |
| allowed_values | native-country | value in data dictionary (incl. '?') | error | 0 |  |
| allowed_values | income | value in data dictionary | error | 0 |  |
| range | age | 17 <= x <= 90 | error | 0 |  |
| range | hours-per-week | 1 <= x <= 99 | error | 0 |  |
| range | capital-gain | 0 <= x <= 99999 | error | 0 |  |
| range | capital-loss | 0 <= x <= 4356 | error | 0 |  |
| range | education-num | 1 <= x <= 16 | error | 0 |  |
| range | fnlwgt | x > 0 | error | 0 |  |
| null | age | no NaN (missing must be '?') | error | 0 |  |
| null | workclass | no NaN (missing must be '?') | error | 0 |  |
| null | fnlwgt | no NaN (missing must be '?') | error | 0 |  |
| null | education | no NaN (missing must be '?') | error | 0 |  |
| null | education-num | no NaN (missing must be '?') | error | 0 |  |
| null | marital-status | no NaN (missing must be '?') | error | 0 |  |
| null | occupation | no NaN (missing must be '?') | error | 0 |  |
| null | relationship | no NaN (missing must be '?') | error | 0 |  |
| null | race | no NaN (missing must be '?') | error | 0 |  |
| null | sex | no NaN (missing must be '?') | error | 0 |  |
| null | capital-gain | no NaN (missing must be '?') | error | 0 |  |
| null | capital-loss | no NaN (missing must be '?') | error | 0 |  |
| null | hours-per-week | no NaN (missing must be '?') | error | 0 |  |
| null | native-country | no NaN (missing must be '?') | error | 0 |  |
| null | income | no NaN (missing must be '?') | error | 0 |  |
| missing_token | workclass | count of '?' | info | 2799 | 4, 6, 13, 22, 35 |
| missing_token | education | count of '?' | info | 0 |  |
| missing_token | marital-status | count of '?' | info | 0 |  |
| missing_token | occupation | count of '?' | info | 2809 | 4, 6, 13, 22, 35 |
| missing_token | relationship | count of '?' | info | 0 |  |
| missing_token | race | count of '?' | info | 0 |  |
| missing_token | sex | count of '?' | info | 0 |  |
| missing_token | native-country | count of '?' | info | 857 | 19, 65, 83, 188, 253 |
| consistency | education/education-num | education-num matches education level | error | 0 |  |
| consistency | relationship/sex | Husband => Male, Wife => Female | warning | 4 | 5661, 16856, 23390, 43422 |
| consistency | relationship/marital-status | Husband/Wife => Married-* | warning | 0 |  |
| suspicious_value | capital-gain | 99999 is a top-code / sentinel | warning | 244 | 83, 346, 357, 418, 692 |
| suspicious_value | hours-per-week | 99 hours is a top-code | info | 137 | 384, 901, 1121, 1156, 1649 |
| duplicates | * | exact duplicate rows (all 15 columns) | warning | 52 | 864, 11189, 11212, 13848, 15960 |

**Schema errors: 0.** Warnings: 3 checks with hits (see table).
