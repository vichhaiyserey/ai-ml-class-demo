# Week 2 — Sprint sync
Date: 2026-01-13  •  Attendees: Priya, Marco, Dana, Sam, Lee

## Progress
- Retrieval prototype works on 200 tickets; median response 4s on CPU.
- First summaries are too generic; retrieval often returns the wrong tickets.

## Decisions
- Switch chunking to paragraph-level with ~15% overlap (was 400-word blocks).
- Add a minimum similarity threshold of 0.1 to drop weak matches.
- Show the retrieved tickets in the UI so reviewers can audit the source.

## Findings
- Increasing Top-K from 3 to 6 noticeably improved coverage but added noise.
- Typos hurt retrieval; a light normalisation step improved the match rate.
- Temperature 0.2 gave more consistent summaries than the default.

## Risks / blockers
- No ground-truth labels yet; quality is judged by gut feel, which does not scale.
- Support wants severity in the summary, but severity lives in a different system.

## Action items
- Dana: build a 15-item evaluation set with expected key points by Jan 16.
- Marco: try a 0.1 vs 0.2 similarity threshold and log which chunks are missed.
- Lee: design the "show sources" panel; it needs chunk scores displayed.
- Sam: pull severity from the ticket API and attach it as metadata.
