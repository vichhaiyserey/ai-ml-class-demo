# Week 1 — Kickoff notes
Date: 2026-01-06  •  Attendees: Priya (PM), Marco (ML), Dana (Data), Sam (Backend), Lee (Design)

## Goal
Ship an "AI Insights" panel that summarises customer feedback for the support team.

## Decisions
- Start with a retrieval-augmented summariser over existing support tickets.
- Use the local Qwen model for the demo; revisit hosted models if latency is a problem.
- Keep a human in the loop before any summary reaches a customer.

## Risks
- Ticket text is noisy (typos, mixed languages). Cleaning effort is unknown.
- We do not yet have a labelled set to measure summary quality.
- Limited CPU-only hardware; long prompts will be slow.

## Action items
- Marco: prototype chunking and TF-IDF retrieval by Jan 10.
- Dana: export two weeks of anonymised tickets by Jan 9.
- Sam: stand up the /summary endpoint skeleton by Jan 13.
- Priya: draft 10 example questions for evaluation by Jan 12.

## Open questions
- How fresh must the summaries be — hourly or daily?
- Do we need per-team views or one global summary?
