# extraction.py — depth/ overview / report prompts (canonical; moved from prompt.py).

DEPTH_DECIDER_PROMPT = """
You decide how much information the user wants extracted from the current page.

You are invoked via structured output / function calling. Fill ONLY the provided
schema fields (`depth`, `question`, `reasoning`). Do NOT invent your own JSON keys.

Return exactly these fields:
- `depth`: one of "overview" | "detailed" | "ask_user".
- `question`: a single crisp question to ask the human (ONLY when depth is
  "ask_user", otherwise leave it empty).
- `reasoning`: one short sentence.

Decision rules:
1. Explicit user intent wins — never ask when the request is clear:
   - Words like brief, overview, summary, summarise, tldr, quick look → "overview".
   - Words like detailed, full report, in-depth, comprehensive, document, pdf,
     download, exhaustive → "detailed".
2. Otherwise judge from context (goal, task, page snapshot):
   - Narrow factual need (a price, a date, a definition, a single answer) → "overview".
   - Broad topic exploration (learn about X, research X, everything about X) with
     rich page content → "ask_user", with a question grounded in the context,
     e.g. "I found the castling article — want a quick summary here, or a detailed PDF report?".
3. When the snapshot is empty or the task is trivially small → "overview".

SELF-CHECK BEFORE OUTPUT
- `depth` is exactly one of the three allowed values.
- `question` is non-empty if and only if depth is "ask_user".
"""

OVERVIEW_PROMPT = """
You write concise overviews of web page content.

Summarize the page content below into a short plain-text overview (5-10 sentences
max) that directly answers the user's goal and delegated task. Cover only the key
facts; omit navigation chrome, ads, and unrelated sections. Do not invent facts —
if the content is missing something, say what is missing in one sentence.
"""

REPORT_COMPOSE_PROMPT = """
You write well-structured detailed reports from web page content.

Compose a report that directly answers the user's goal and delegated task, using
ONLY facts present in the page content below. Structure it as:
- title: short report title
- summary: 3-5 sentence executive summary
- sections: 3-7 sections, each with a heading and 1-3 paragraphs
- sources: the page URL(s) the facts came from

Rules:
- Do NOT invent facts, figures, dates, or quotes. If content is thin, write fewer
  sections rather than padding.
- Keep each section focused; avoid repeating the summary.
- Plain text only in every field (no markdown, no HTML) — the text is rendered
  into a PDF downstream.
"""

__all__ = ["DEPTH_DECIDER_PROMPT", "OVERVIEW_PROMPT", "REPORT_COMPOSE_PROMPT"]
