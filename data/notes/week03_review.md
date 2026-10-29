# Week 3 — Review & pilot
Date: 2026-01-20  •  Attendees: Priya, Marco, Dana, Sam, Lee, plus two support agents

## Pilot results
- Agents rated 8/10 summaries as "useful" or better.
- Two summaries missed a critical escalation because it sat in a low-scoring chunk.

## Decisions
- Default to Top-K=6, similarity 0.1, chunk 200 words, overlap 40.
- Always render the source chunks under every summary for trust and debugging.
- Escalations must be surfaced by a rule, not left to the summariser.

## Lessons
- Retrieval was the bottleneck, not the language model.
- Showing sources turned the tool from "magic" into something agents trust.
- A tiny evaluation set stopped the team from tuning by vibes.

## Remaining risks
- Multilingual tickets still summarise poorly; English-only for now.
- Cost and latency will grow if we summarise on every page load.

## Action items
- Marco: cache summaries per ticket set; refresh nightly.
- Dana: expand the eval set to 40 items and track faithfulness.
- Priya: decide pilot scope (one team vs all support) by Jan 24.
