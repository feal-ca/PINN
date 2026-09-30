# Writing guidelines and voice persona

Two parts: the **persona** (who is writing, and how they sound) and the
**tell-checklist** (what to audit every draft against). Research sources
that fed this are noted inline where the reasoning isn't obvious.

---

## Part 1: The persona

**Who's writing:** someone who actually built the thing, writing up what
happened, for other practitioners. Not a tutorial-content producer, not a
lecturer, not a lab report. The closest models are the real TDS PINN
articles studied for calibration, authors who write "I face this
challenge firsthand," "if you've ever tried to read existing literature
on PINNs, it's a tough read!", "So problem solved, right?"

**Five rules that define the voice:**

1. **First person, always.** "I built," "I got this wrong," "I assumed X
   and had to go measure it." Never "this project demonstrates" or "the
   build surfaced three issues." If a sentence could appear in a paper's
   abstract, it's wrong for this piece.
2. **Address the reader directly.** "If you've built a PINN before, you
   already know where this goes." Rhetorical questions are allowed and
   encouraged when they're real questions the reader is having.
3. **Every claim carries a specific.** A number, a file name, a failure
   mode, a decision and the reason for it. The research on why AI prose
   reads flat converges on this: it isn't only specificity, it's
   *moments*, a thing that happened at a point in time, with a
   consequence.
4. **Admit the bad parts in the same register as the good parts.** The
   two failed collocation experiments get the same matter-of-fact tone as
   the fixes that worked. No throat-clearing about honesty (see the tic
   list below), just report them.
5. **Vary the rhythm on purpose.** Short sentence. Then a longer one that
   develops the idea and earns its length. Paragraphs of wildly different
   lengths. The test: does the draft look like *a wall or a landscape*?

**Anchoring rule:** every first-person claim must trace to something that
actually happened in this build (dev-log.md, a script's output, a config).
No invented biography, no invented emotions-as-facts, no fabricated
timelines. The real events are specific enough to carry the voice on
their own.

---

## Part 2: The tell-checklist

Vocabulary matters least. Audit in this order.

### Tier 1, structural (the ones that actually give it away)

- **Third-person distance.** Locked-in "this project / the build / the
  network was trained" register with no human in any sentence. Diagnostic:
  *does a human belong in this sentence?*
- **Wall-shaped paragraphs.** Every block 3-5 sentences of 15-25 words.
  Fix: count words per sentence; if nothing is under 8, break something.
- **Parallel kickers.** Every section ending on a tidy punch. Fix: let
  most sections end flat; keep one punch where it's earned.
- **Too-clean wrap-up.** Test: delete the final paragraph. If the piece
  still works, it was padding.
- **Excessive uniform structure.** Repeated **Bold lead-in.** paragraph
  templates across consecutive sections; jumping straight to bullets
  instead of building an argument.

### Tier 2, rhetorical patterns (fine in moderation, damning on autopilot)

The useful frame here: these are legitimate devices used *without
intent*. Don't ban them, earn them.

- **Binary contrast ("not X, but Y" / "rather than").** Diagnostic: does
  the negated claim reflect what the reader actually believed? If nobody
  thought X, delete the setup and state Y.
- **Rule of three.** Diagnostic: does the third item surprise, or just
  complete the pattern? If deleting it loses nothing, delete it.
- **Colon reveals.** Diagnostic: does the pre-colon phrase do real work,
  or is it signposting ("Here's the thing:")?
- **Em-dashes: banned outright.** Not "in moderation", not "when earned".
  Zero. This is a hard rule, not a Tier 2 judgment call. It covers the
  em-dash character (U+2014), the en-dash (U+2013), and the double hyphen
  used as prose punctuation. An earlier version of this file allowed them
  "in moderation", a review counted 27 in one draft, and they still
  survived a revision pass. Hence the absolute rule.

  Rewrite the sentence instead of substituting punctuation. An em-dash
  aside is doing one of four jobs, each with a better fix:
  - appositive or definition: use commas, or split into its own sentence
  - a reveal or elaboration: use a colon
  - a true parenthetical: use parentheses
  - a dramatic pause before a punch: delete it, the pause was the tell

  Check before every handoff with `scripts/check_dashes.sh`, which must
  report zero across every article file.
- **Hedging.** "typically," "might be," "can be seen as." Take the
  position or cut the sentence.

### Tier 3, vocabulary

Flagged words: delve, leverage, utilize, foster, harness, streamline,
underscore, crucial, seamless, robust, cutting-edge, tapestry, landscape
(non-technical sense), realm, testament, underpinnings, furthermore,
moreover, additionally, "it's important to note," "in today's world."

**Project-specific tic list** (caught in earlier drafts of this article,
watch for regression): "honest/honestly," "genuine/genuinely," "real,"
"actually" used as *authenticity markers* rather than as meaningful
modifiers. Saying a result is honest is worse than just reporting it at
its real size. Also: "Two things..." as a section opener, and "worth
[gerund]-ing" as a transition.

### Titles

Documented AI title signature: evocative metaphor + colon + explainer
("Seeing the Invisible: Reconstructing...").

What actually runs on TDS, from the three PINN articles studied:
- *Physics-Informed Neural Networks for Inverse PDE Problems* / sub:
  *Solving the Heat Equation using DeepXDE*
- *Physics Informed Neural Networks (PINNs): An Intuitive Guide* / sub:
  *The (readable) what, how, and why of physics informed neural networks*

Pattern to copy: **flatly descriptive, keyword-forward, no metaphor.**
The subtitle names the concrete thing being done. Boring titles rank.

---

## Part 3: SEO and TDS editorial

- Keyword phrase in title, subtitle, first paragraph, and >=1 section
  header. Subtitle <=140 characters.
- Real code blocks, never screenshots. Own figures only.
- Cite real sources, format `[N] Author(s), "Title" (Year), Venue`.
- Lead with what's novel, why it matters, and how the piece proves it.
- No CTA buttons, no all-caps titles.
- Every number traces to dev-log.md, script output, or a verified source.
- Never oversell a weak result (the Phase 2 WSS number is 70.6%; the
  PINN's peak reaches 27% of the true peak, report both at full size).

## Part 4: Competitive positioning (checked Sept 2026)

Existing TDS PINN coverage and what it means for us:

| Article | Covers | Our differentiation |
|---|---|---|
| Essential Review Papers on PINNs | Literature curation | No overlap |
| PINNs for Inverse PDE Problems | Inverse problems, 1D heat equation, DeepXDE | Real 2D Navier-Stokes, from scratch, biomedical |
| PINNs: An Intuitive Guide | Beginner intro, projectile motion, TensorFlow | PyTorch, real PDE, real inverse parameter, runnable end-to-end |

Do **not** write another conceptual PINN intro, TDS has one. Part 1
teaches the mechanism through a real problem instead.

---

## Part 5: What the author's review of the first Part 1 draft added

Marked up line by line on `part1_draft_v1.md`. Treat these as rules.

- **The result has to be good.** A finding that the method half works is not an
  article. Improve the experiment first (here: a longer channel, more sensors, a
  cleaner reference), then write.
- **No takeaway box.** And no stack of numbers in the opening. Open by saying
  what the piece does and why the problem matters, in plain words.
- **Few numbers.** Use "close", "a little low", "within the noise". Keep exact
  values for the one or two the reader needs. A small table is fine, a paragraph
  of percentages is not.
- **Explain, don't narrate.** A section about a hard thing (the viscosity) is one
  explanation of why it is hard and what fixes it, not the run-by-run story of
  what happened when.
- **Plain headings.** "The setup", "Results". No clever headings.
- **Drop citations that do not earn their place.** A bracketed reference must
  point at something in the text. Do not add literature to look thorough.
- **Phrases that read as AI to this reader:** "nobody measured", "took me a while
  to see why that was wrong", "the flag that matters", "matters here", "where a
  step means something", "the X never sees any of it", "I could trust". Say the
  plain thing instead.
- **Give the reason for a choice in ordinary terms.** If the start value of a
  parameter looks odd, say why (here: to check the answer comes from the data,
  and you would normally start from the literature value).
- **Model the reference articles' voice:** conversational, reader-directed
  questions, intuition before code, descriptive imperative headings, an ending
  that stops when the content does.
