# Notes merge emails

`Email merged notes` runs when a push to `main` changes posts, including a merged PR. It sends one summary to `arunabh1904@gmail.com` for added or substantively revised Arxiv Notes. Each entry uses the note's opening summary and links to the note and original paper. Other categories, metadata-only moves, and Paper Radar scaffolds are excluded.

The email confirms the merge; Pages may still be deploying. There is no scheduled polling or dependency on a local computer. The sender uses the repository secrets `NOTES_GMAIL_ADDRESS` and `NOTES_GMAIL_APP_PASSWORD` for Gmail SMTP and a read-only Sent Mail check. Credentials never enter the generated payload or repository.

A deterministic Message-ID keyed by the merge commit prevents repeat delivery on workflow retries. Failure to check Sent blocks sending. An ambiguous SMTP failure is not retried automatically; inspect Sent Mail before rerunning. Do not remove these notification emails from Sent if you intend to rerun an old merge.

For a missed merge, dispatch the workflow on `main` with the full `base_sha` immediately before the merge and `head_sha` of the merged commit. The script requires both commits to belong to the current main history. This recovery path has the same duplicate check as the automatic trigger.

Run `npm run test -- tests/notes-email.test.ts` for selection and delivery tests. To preview without sending, run `NOTES_BASE_SHA=<sha> NOTES_HEAD_SHA=<sha> node scripts/notifications/notes-email.mjs`, then inspect the generated `notes-email.json`.
