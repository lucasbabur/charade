---
title: "Visual lessons validation"
created-at: 2026-10-07
updated-at: 2026-10-07
---

# Local validation

- Eight SVGs parsed as XML; every viewBox is 0 0 1440 960.
- Every SVG has a title and description; chart marks have direct labels and a gallery text counterpart.
- Text uses neutral ink, minimum 22 px body / 18 px footer, with 40 px titles and 28 px headings.
- Conservative system-sans text estimates stay within the canvas and their cards, with no text/text overlaps.
- Gallery has eight lesson links, one initially visible diagram, and seven hidden lessons.
- All gallery resources and links are local. SVG and fragment links resolve.
- Eight PNG links correspond to the eight SVGs. PNG rendering is separate from this generator.
- Generator uses the Python standard library, without network requests or browser launches. Gallery uses no browser storage.
- Representative DR and Gamma arithmetic checks pass.
- Generator bounds checks are approximate. Actual browser checks, when recorded below, are a separate validation pass.

## Browser validation, 2026-10-07

- Rendered all eight final SVGs to matching 1440 × 960 PNGs with headless Chrome 153.
- Actual browser text rectangles show no text collisions or text outside the canvas in any diagram.
- Inspected the final first diagram and a contact sheet of all eight images.
- Desktop gallery checked at 1440 × 1000; mobile checked at 390 × 844, including a dark-theme render. Neither layout has horizontal page overflow.
- All eight gallery images loaded. No external resources or JavaScript exceptions were observed.
- ArrowRight changed the visible lesson from 1 to 2.
- Opened the local gallery in Zen and verified its desktop window title.
- Authoring and browser-check scripts are task scratch files, not changes to the Charade implementation.

## Repository checks, 2026-10-07

- `poe check` passed: Ruff lint and formatting, strict Pyright, import contracts, dead-code checks, both test suites and ML static gates.
- Charade tests: 223 passed, 1 skipped, 12 deselected, 4 warnings; coverage 90.94%, above the 85% gate.
- mlcheck tests: 111 passed. Static gates: 4 passed, no failures or warnings.
- Repository changes contain only the gallery, SVG/PNG images and these documentation files.

## Palette: every pair

These are WCAG luminance contrast ratios, not a claim that every pair passes a text threshold.
Body text uses ink or muted against the light background. Accent marks are directly labeled;
in particular aqua/background is below 3:1, so meaning never depends on that color alone.
The light grid is decorative. No accent color is used for text.

| Pair | Contrast |
|---|---|
| background / ink | 19.17:1 |
| background / muted | 7.73:1 |
| background / grid | 1.29:1 |
| background / blue | 4.30:1 |
| background / orange | 3.12:1 |
| background / aqua | 2.74:1 |
| ink / muted | 2.48:1 |
| ink / grid | 14.87:1 |
| ink / blue | 4.46:1 |
| ink / orange | 6.15:1 |
| ink / aqua | 6.99:1 |
| muted / grid | 6.00:1 |
| muted / blue | 1.80:1 |
| muted / orange | 2.48:1 |
| muted / aqua | 2.82:1 |
| grid / blue | 3.34:1 |
| grid / orange | 2.42:1 |
| grid / aqua | 2.13:1 |
| blue / orange | 1.38:1 |
| blue / aqua | 1.57:1 |
| orange / aqua | 1.14:1 |
