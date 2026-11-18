# amex-default notes

Facts as of 2026-11-09, from memory of the competition; check them against the downloaded files.

- Competition ended 2022-08-24; 4,874 teams. Late submissions open, 5 per day.
- Metric: mean of a normalized Gini with negatives weighted 20x and the share of defaults caught in the
  top 4% of predictions (registered as `amex`; matches the published reference implementation).
- Size: train_data about 5.5M statements (16 GB CSV, about 190 columns) for about 459k customers;
  test_data about 11M statements (33 GB CSV) for about 925k customers. Needs the WSL memory raised to
  24 GB, and the statement roll-up runs in 8 key slices.
- Feature prefixes: D_ delinquency, S_ spend, P_ payment, B_ balance, R_ risk. Some columns are categorical
  codes (for example B_30, B_38, D_63, D_64, D_114, D_116, D_117, D_120, D_126, D_66, D_68).
- Out of scope: the popular community "denoised" copies of this data (they reverse noise Amex added to
  the files); we use the raw files.
