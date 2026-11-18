# home-credit notes

Facts as of 2026-11-09, from memory of the competition; check them against the downloaded files.

- Competition ended 2018-08-29; 7,176 teams. Late submissions open, 5 per day. Metric ROC AUC.
- Tables (approximate rows): application_train 307,511 (122 columns, TARGET), application_test 48,744,
  bureau 1.7M, bureau_balance 27M, previous_application 1.7M, POS_CASH_balance 10M,
  installments_payments 13.6M, credit_card_balance 3.8M.
- Data quirk visible in the column itself: DAYS_EMPLOYED = 365243 (about 1,000 years) is a placeholder,
  most likely for applicants without employment (pensioners).
- Rule (2026-11-09): no ideas from this competition's published solutions, discussions or notebooks.
- Honest baseline: run 20261109_122058_850, generic roll-ups and credit ratios only: CV 0.79234,
  public 0.79423, private 0.79274 (about the top 41% of 7,180 on the public board).
