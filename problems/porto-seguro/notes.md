# porto-seguro notes

Facts as of 2026-11-09, from memory of the competition; check them against the downloaded files.

- Competition ended 2017-11-29; 5,156 teams. Late submissions open (Kaggle API: submissions not disabled),
  5 per day. Scored by normalized Gini = 2 x AUC - 1, so CV uses the same ranking as AUC.
- Real insurance data with anonymized feature names: about 595k train and 893k test rows, 57 features in
  groups ps_ind_, ps_reg_, ps_car_, ps_calc_. Suffix _bin = binary, _cat = categorical; -1 = missing.
- Purpose here: the first real-data check for genML (one table, no code changes needed): does a late
  submission score, and how does an honest pipeline rank on the private leaderboard?
- Out of scope: anything that exploits how the dataset was built rather than what it describes.
