# ieee-fraud notes

Facts as of 2026-11-09, from memory of the competition; check them against the downloaded files.

- Competition ended 2019-10-03; 6,351 teams. Late submissions open, 5 per day. Metric ROC AUC.
- Files: train_transaction (590,540 rows, about 394 columns, isFraud), train_identity (about 144k rows,
  41 columns, joined one-to-one on TransactionID), and the same for test (about 507k transactions).
  Test identity column names use hyphens (id-01) where train uses underscores (id_01); the table loader
  aligns them.
- TransactionDT is seconds from a reference time; test follows train by roughly a month or more.
- Rule (2026-11-09): no ideas from this competition's published solutions, discussions or notebooks.
  Run 20261109_132836_471 broke this and was deleted on 2026-11-12. A clean baseline is still to be run.
- Honest limits we keep: group statistics are past-only (live systems cannot see future transactions).
