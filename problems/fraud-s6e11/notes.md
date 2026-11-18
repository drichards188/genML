# fraud-s6e11 notes

Facts about the competition and genML's own measurements only. Nothing here comes from the
competition's discussions, public notebooks or solutions (CLAUDE.md, purpose).

- Deadline 2026-11-30, 10 submissions per day. Metric ROC AUC.
- Public leaderboard on 2026-11-06: #1 0.95281, #50 0.95239, median 0.94819, 927 teams.
- Public scores tracked CV closely on three submissions (public about +0.0005 above CV).
- 2026-11-12: every run after 2026-11-03 was deleted, with its predictions, logs and ledger entries,
  because it was built on copied material. The only run left, 20261103_122601_146 (CV 0.94588,
  public 0.94630), predates the nested target-encoding leak fix, so it cannot be an anchor. A new
  baseline is needed before the autopilot runs again.

Scope (2026-11-07): this problem is a test bench for genML's process, not a ranking target. Out of
scope: other people's predictions, and modelling how the synthetic data was generated. See CLAUDE.md.
