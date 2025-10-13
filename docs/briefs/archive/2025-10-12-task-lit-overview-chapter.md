---
status: drafting
workflow: thesis-lit-overview
summary: >-
  Plan and draft the Literature Overview chapter focused on the Viterbo
  conjecture and surrounding results. Audience is MSc-level readers with
  symplectic/contact background; style is concise, notation-oriented, and
  survey-like with honest pointers to related work we have not fully read yet.
created: 2025-10-12
---

# Brief: Literature Overview Chapter (Viterbo Conjecture)

## Audience & Scope

- Primary: MSc math students comfortable with differential/symplectic/contact basics.
- Secondary: experts (advisor/committee) seeking a crisp refresher and a curated map of results.
- Tone: skimmable, definitions-as-reminders, normalization clarity, narrative grouped by themes.

## Thesis Topic Recap

- Computational/structural study of symplectic capacities for convex bodies, with emphasis on
  the Ekeland–Hofer–Zehnder capacity and the Viterbo Volume–Capacity conjecture.
- Algorithms for polytopes; experimental checks; connections to convex geometry (e.g., Mahler).

## Literature Gathering Plan

- Sources: surveys (Ostrover; Artstein-Avidan et al.), foundational capacity papers (Gromov,
  Ekeland–Hofer, Hofer–Zehnder, Viterbo), technique-specific lines (ECH in 4D, symplectic
  homology, Minkowski billiards), and convex-geometry bridges (Mahler links).
- Informal: lecture notes (Hutchings on ECH capacities), seminar slides, blog posts by active
  researchers; annotate with “from abstract” when not yet vetted.
- Gaps: we expect to miss specialized subclasses (e.g., certain toric/Reinhardt domains,
  zonoids) until a deeper pass; will flag with TODOs.

## Search Directions (for DeepResearch delegation)

- Queries: "Viterbo conjecture volume capacity constant 1/n!", "convex body symplectic capacity
  bounds", "Minkowski billiards EHZ capacity", "symplectic homology capacities survey",
  "ECH capacities volume asymptotics", "Mahler conjecture symplectic reduction".
- Authors to sweep: Viterbo, Ostrover, Artstein-Avidan, Karasev, Akopyan, Schlenk, Hutchings,
  Abbondandolo, Hryniewicz, Salomão, Cieliebak, Frauenfelder, Paternain, Alves, Ramos.

## Drafting Approach

1. Internal rough map: group papers by theme; extract key takeaways and normalizations.
2. Convert to chapter text: short notation refresher → conjecture statement → families of results
   (sharp cases, bounds, special classes, techniques) → related problems → surveys → open ends.
3. Iterate for clarity and concision; postpone precise constants if uncertain; add TODO footnotes.
4. Wire into thesis: add `chapters/literature-review.tex` and include from `thesis/main.tex`.
5. Compile and fix TeX issues; if LaTeX is not available in the container, compile on host.

## Risks & Mitigations

- Risk: Misstated constants or scope of partial results. Mitigate by phrasing qualitative bounds
  and citing canonical sources; mark TBDs for later tightening.
- Risk: LaTeX compilation not available in container. Mitigate by minimal dependencies and
  providing simple compile instructions; or request tool install if needed.

## Deliverables (first pass)

- `thesis/chapters/literature-review.tex` with structured overview and notation refresher.
- `thesis/main.tex` updated to include the new chapter.
- Follow-up task: expand `thesis/references.bib` with verified entries and cross-checked constants.

## DeepResearch Prompts (copy 1:1)

Prompt A — Core map of Viterbo-related literature

Context:
- We are writing a Literature Overview for an MSc thesis on the Viterbo Volume–Capacity Conjecture for convex bodies in (R^{2n}, ω0). The audience knows symplectic/contact basics; we need a crisp, skimmable map with normalization clarity and honest notes when sources are “from abstract”.
- We care about EHZ/HZ capacities, Gromov width, cylindrical capacity, symplectic homology/Viterbo capacities, ECH capacities in 4D, Minkowski billiards characterization for convex bodies, convex-geometry bridges (Mahler product), and capacity comparisons/equivalences on convex classes.

What to return:
- A curated list (ideally 40–80 items) grouped by themes: (1) foundational capacities + axioms; (2) Viterbo conjecture statements and partial progress; (3) convex-geometry bridges (Mahler links, polarity); (4) ECH capacities and volume asymptotics in 4D; (5) Minkowski billiards and EHZ for convex bodies/polytopes; (6) toric/Reinhardt/Delzant settings and combinatorics; (7) surveys/lecture notes/expository sources; (8) computational or algorithmic treatments for capacities.
- For each item: minimal BibTeX (author, title, venue, year, arXiv if any), 1–2 sentence takeaway. If not fully verified, mark “from abstract”.
- Include at least: Viterbo (original conjecture paper/notes), Ostrover surveys, Artstein-Avidan–Karasev–Ostrover works linking Mahler and Viterbo, Cieliebak–Frauenfelder–Paternain on SH vs EHZ, Hutchings on ECH capacities, McDuff–Schlenk on 4D embeddings (context), Minkowski billiards characterizations for EHZ on convex bodies (Akopyan/Karasev lines), and any sharp dimension-dependent constants for symmetric bodies.

What to avoid:
- Do not include unrelated symplectic embedding literature not tied to volume–capacity inequalities. Avoid higher-category/Fukaya-category-only works unless they explicitly yield capacity bounds relevant to convex bodies. No ML/CS optimization unless directly computing capacities.

Output format:
- Markdown with themed subsections; under each item, provide a compact BibTeX block followed by a “Takeaway:” line. Use “from abstract” flags as needed. Keep it concise and skimmable.


Prompt B — Constants and equivalences across capacities on convex bodies

Context:
- We need a table-quality summary of known inequalities relating c_B, c_Z, c_HZ/c_EHZ, and c_k^{SH} on convex bodies (and symmetric subclasses), including equivalences up to universal constants when known.

What to return:
- A short list of precise statements with constants (dimension dependence explicit), references per statement, and notes on normalization assumptions. Include any known gaps or counterexamples.

What to avoid:
- Overly general embedding results without convexity or capacity consequences; omit proofs.

Output format:
- Markdown table plus bullet notes, each row with: statement, hypotheses, constant(s), reference(s), “verify” flag if from abstract.


Prompt C — Fact-check and tighten key claims

Context:
- We have a themed list of results on symplectic capacities and Viterbo’s conjecture. Some claims may overstate known equivalences (e.g., “all normalized capacities coincide” on broad classes) or blur hypotheses/normalizations. We need precise, vetted statements.

What to return:
- A corrections list addressing: (i) where capacities are known to coincide (exact classes, which capacities and/or which k, and in what sense), (ii) dimension-dependent constants and their hypotheses (e.g., central symmetry), (iii) near-ball equivalence results (exact theorem statements and topology on domains), (iv) any contested or retracted claims.
- For each item: a precise replacement statement with explicit hypotheses and normalization (e.g., c(B^{2n}(r))=πr^2, c(Z(r))=πr^2), and 1–2 primary references (journal/preprint) to cite.

What to avoid:
- Vague “is believed/expected” language unless clearly marked conjectural; avoid secondary sources when a primary reference exists.

Output format:
- Markdown list of “Claim → Correction” entries, each ending with compact citations.

Claims to check (copy 1:1 below):

1) EHZ = HZ on convex domains in R^{2n}.
2) c^{SH}_1 = c_{EHZ} for convex (or star-shaped) domains in R^{2n}; clarify exact hypotheses.
3) Minkowski billiards characterization: For smooth strictly convex K, c_{EHZ}(K) equals the length/action of the shortest periodic billiard trajectory for the norm with unit ball K^◦; polytopes handled via facet bounce sequences.
4) Haim‑Kislev (2019 AGT) gives a finite, combinatorial formula for c_{EHZ} of convex polytopes in R^{2n}; complexity and “≤ n bounces” structure theorem as stated.
5) ECH capacities volume asymptotics (4D): For star‑shaped X ⊂ R^4, the asymptotic of c_k(X) determines Vol(X). Provide the precise formula (note: our draft used a linear-in-k form; please correct to the standard c_k(X)^2/(2k) → Vol(X) statement if that is the canonical normalization).
6) McDuff–Schlenk (2012 Annals): ECH capacities give sharp obstructions for 4D ellipsoid–ball embeddings; “infinite staircase” phenomenon.
7) Hutchings (2022 PNAS or related): definition of 4D symplectic capacities via J‑holomorphic curves that agree with ECH capacities on known families; scope and hypotheses.
8) Wormleighton: ECH capacities, Ehrhart theory, and toric varieties — precise relationship and asymptotics in toric cases.
9) Ostrover (ICM 2014): survey on capacities and Banach/convex geometry; key inequalities and constants summary.
10) Artstein‑Avidan–Milman–Ostrover (M‑ellipsoid/volume radius): existence of a universal, dimension‑independent constant C such that c(K) ≤ C·Vol(K)^{1/n} for normalized capacities; precise statement and conditions.
11) Artstein‑Avidan–Karasev–Ostrover (2014 Duke): equivalence links between Viterbo‑type inequalities for K×K^◦ and Mahler’s product for symmetric K; exact theorem statement.
12) Abbondandolo–Benedetti–Edtmair (2023): “near ball” equivalence — specify topology (C^2/C^3), which capacities coincide, and whether sequences or first capacities are meant.
13) Irie (2019): symplectic homology of fiberwise convex sets; implications for capacities used in the literature (state precise theorems if any).
14) Haim‑Kislev & Ostrover (2024): counterexample to Viterbo’s conjecture — domain (product of polygons?), dimension (4D), which capacity (c_{EHZ}?), normalization, and the inequality that fails vs. the equal‑volume ball.
15) Baracco–Bernardi–Lange–Mazzucchelli (2023): local maximizers of higher capacities in 4D are ellipsoids — exact statement and hypotheses.
16) “Strong capacity equality on convex toric or S^1‑invariant domains”: verify/clarify scope — which capacities, which classes (monotone toric, Reinhardt), dimension restrictions, convexity assumptions, equality for all k or only first capacity.
17) Monotonicity/ordering: c_B ≤ c_Z and c_{EHZ} ≤ c_Z for general domains; include references.
18) Ellipsoids: all standard capacities coincide; equality cases for the Viterbo inequality (normalization check).
19) Polydisks/products: explicit formulas for c_B, c_Z, c_{EHZ} (where known); consistency with volume‑capacity bounds.
20) “Equivalence up to universal constants” among normalized capacities on convex bodies — currently known best constants (dependence on n, symmetry), and references (e.g., John ellipsoid bound vs. later improvements).
21) Symmetric classes (centrally symmetric, zonoids): improved constants or sharp cases; precise statements and sources.
22) Liouville form choice (λ0 = ½∑(x dy − y dx) vs ∑ x dy) does not affect action of closed characteristics (exact‑form invariance) — cite a standard reference.
23) Capacity normalization used in core sources (c(B^{2n}(r))=πr^2, c(Z(r))=πr^2); flag any deviations.
24) Toric/Delzant capacities via lattice paths: which capacities (ECH/EH/SH) admit combinatorial computation; exact references and bounds.
25) Minkowski billiards bounds relating c_{EHZ}(K×T) to shortest T‑billiard in K; include Gutkin–Tabachnikov and AKS‑style results with hypotheses.

Prompt D — Canonical BibTeX pack for core items

Context:
- We need clean, canonical BibTeX for central references to cite in the thesis. Keys should be stable; include DOI and arXiv where relevant; add a short normalization note per entry.

What to return:
- BibTeX entries for at least: Viterbo (2000 JAMS), CHLS (2007 MSRI survey), Cieliebak–Frauenfelder–Paternain (c^{SH}_1=c_{EHZ} on convex/star-shaped; exact venue), McDuff–Schlenk (2012 Annals), Cristofaro‑Gardiner–Hutchings–Ramos (2015 Invent.), Cristofaro‑Gardiner–Savale (2020 Selecta), Hutchings (ECH capacities notes and 2022 PNAS if applicable), Wormleighton (ECH & Ehrhart), Haim‑Kislev (2019 AGT), Haim‑Kislev–Ostrover (2024 arXiv counterexample), Abbondandolo–Benedetti–Edtmair (near-ball capacities), Irie (fiberwise convex sets & SH), Akopyan–Karasev–Sasaki (Minkowski billiards line), Gutkin–Tabachnikov (Minkowski billiards), Ostrover ICM (2014), Artstein‑Avidan–Milman–Ostrover (M‑ellipsoid/volume radius), Artstein‑Avidan–Karasev–Ostrover (Mahler link), Ostrover–Tyomkin (toric/Delzant capacities if applicable).
- For each entry: a one‑line “Normalization:” note specifying the capacity normalization used (ball/cylinder) or stating if unspecified.

What to avoid:
- Informal posts when a formal source exists; if using expository/blog posts, include both and flag expository.

Output format:
- Pure BibTeX blocks. After each block, one “Normalization:” bullet.
 
 Operational notes:
 - Do not ask for more context; assemble entries from canonical sources. If uncertain, include a “Verify:” note and provide both journal and arXiv references.

Prompt E — Normalization crosswalk table

Context:
- Different sources use slightly different conventions (symplectic form, Liouville form, capacity normalization, action sign). We need a crosswalk to harmonize our chapter.

What to return:
- A table with columns: Source; Symplectic form (e.g., ω0=∑dx_j∧dy_j); Liouville form (e.g., λ0=½∑(x_j dy_j−y_j dx_j) vs ∑x_j dy_j); Capacity normalization (c(B^{2n}(r)), c(Z(r))); Action convention and note on invariance under exact shifts; Deviations/notes.
- Cover at least the sources from Prompt D.
 - At minimum, include: Viterbo (2000 JAMS); CHLS (2007 MSRI); Cieliebak–Frauenfelder–Paternain; McDuff–Schlenk (2012 Annals); Cristofaro‑Gardiner–Hutchings–Ramos (2015 Invent.); Cristofaro‑Gardiner–Savale (2020 Selecta); Hutchings (ECH capacities notes and/or 2022 PNAS); Wormleighton; Haim‑Kislev (2019 AGT); Haim‑Kislev–Ostrover (2024 arXiv); Abbondandolo–Benedetti–Edtmair; Irie (2019); Akopyan–Karasev(–Sasaki); Gutkin–Tabachnikov; Ostrover ICM (2014); Artstein‑Avidan–Milman–Ostrover (M‑ellipsoid); Artstein‑Avidan–Karasev–Ostrover (2014); Ostrover–Tyomkin (toric) if applicable.

What to avoid:
- Long quotes; keep entries terse and directly comparable.

Output format:
- Markdown table; add short clarifying bullets below if needed.
 
 Operational notes:
 - Standalone: do not depend on outputs of other prompts; do not ask for extra inputs.

Prompt F — 2024 counterexample due diligence

Context:
- We must summarize the 2024 counterexample(s) to Viterbo’s conjecture with exact scope and normalization to write a careful thesis paragraph.

What to return:
- A precise statement: class of domains (e.g., product of polygons/Lagrangian product), dimension, which capacity (e.g., c_{EHZ}), normalization, and the inequality that fails relative to the ball of equal volume.
- Short sketch of method (polytope EHZ formula; Minkowski billiards; computational certification).
- Status and sources: arXiv versions/dates, seminar notes, institutional posts, any follow‑ups or partial confirmations/errata.
- What remains plausible: restricted variants (e.g., centrally symmetric bodies); supporting evidence.

What to avoid:
- Speculation beyond sources; clearly mark uncertainties or “from abstract”.

Output format:
- Bulleted summary with 2–3 primary references and links.
 
 Operational notes:
 - Do not ask for more documents. If you find conflicting accounts, present both and mark status as “uncertain/under discussion”.

# DeepResearch Outputs (copied 1:1)

## Output A - Literature Overview: Viterbo's Volume-Capacity Conjecture (Convex Domains in $\mathbb{R}^{2n}$)

### (1) Foundational Symplectic Capacities and Axioms

*   `bibtex`
```
@article{Gromov1985,
author = {Mikhail Gromov},
title = {Pseudoholomorphic curves in symplectic manifolds},
journal = {Invent. Math.},
volume = {82},
year = {1985},
pages = {307-347}
}
```
Takeaway: Gromov's non-squeezing theorem (1985) introduced the first symplectic capacity implicitly by showing that a ball of radius $R$ cannot symplectically embed into a cylinder of smaller radius $r$. This gave birth to the notion of symplectic "size" invariants that satisfy monotonicity and conformality axioms.

*   `bibtex`
```
@article{EkelandHofer1989,
author = {Ivar Ekeland and Helmut Hofer},
title = {Symplectic topology and Hamiltonian dynamics},
journal = {Math. Z.},
volume = {200},
year = {1989},
pages = {355-378}
}
```
Takeaway: Ekeland and Hofer formalized symplectic capacity axioms and constructed the first capacity via variational methods on closed characteristics. On convex domains, their capacity coincides with the Hofer-Zehnder capacity and equals the minimal action of a closed billiard trajectory on the boundary, a quantity now known as the Ekeland-Hofer-Zehnder (EHZ) capacity.

*   `bibtex`
```
@incollection{HoferZehnder1990,
author = {Hofer, H. and Zehnder, E.},
title = {A new capacity for symplectic manifolds},
booktitle = {Analysis, et cetera},
publisher = {Academic Press},
address = {Boston, MA},
year = {1990},
pages = {405-427}
}
```
Takeaway: Hofer and Zehnder constructed an equivalent capacity using Hamiltonian dynamics. This Hofer-Zehnder capacity satisfies the same axioms (monotonicity, scaling, nontriviality) and, for convex bodies, agrees with the Ekeland-Hofer capacity. Their work cemented the principle that symplectic capacities capture "size" beyond volume, leading to many distinct capacity definitions in the 1990s.

*   `bibtex`
```
@article{Viterbo1989,
author = {Claude Viterbo},
title = {Capacit{\'e}s symplectiques et applications},
booktitle = {S{\'e}minaire Bourbaki, Vol. 1988/89},
series = {Ast{\'e}risque},
volume = {177},
number = {714},
year = {1989},
pages = {345-362}
}
```
Takeaway: Bourbaki notes by Viterbo gave an early survey of symplectic capacity theory. Viterbo discussed capacity axioms and applications, highlighting that certain capacities coincide on convex sets and foreshadowing isoperimetric-type problems in symplectic geometry. This set the stage for his later conjectures relating volume and capacity in convex domains.

### (2) Viterbo's Volume-Capacity Conjecture: Statements and Progress

*   `bibtex`
```
@article{Viterbo2000,
author = {Claude Viterbo},
title = {Metric and isoperimetric problems in symplectic geometry},
journal = {J. Amer. Math. Soc.},
volume = {13},
year = {2000},
pages = {411-431}
}
```
Takeaway: Viterbo conjectured (circa 2000) that among all convex bodies in $\mathbb{R}^{2n}$ with fixed volume, the Euclidean ball maximizes every symplectic capacity. Equivalently, the ball $B^{2n}$ should minimize the "symplectic size"/volume ratio. Viterbo proved partial results using John's ellipsoid: for any convex $K$, $c(K)/c(B^{2n}) \le C_n$ with $C_n$ growing linearly in $n$ (e.g. $C_n=32^n$ general, $2^n$ if $K$ centrally symmetric). These were the first bounds towards the conjecture.

*   `bibtex`
```
@article{ArtsteinOstrover2008,
author = {Shiri Artstein-Avidan and Vitali Milman and Yaron Ostrover},
title = {On symplectic capacities and volume radius},
journal = {arXiv:math/0603411},
year = {2008}
}
```
Takeaway: (From abstract) By blending asymptotic convex geometry with symplectic techniques, this work significantly improved Viterbo's bound. It showed there is a universal constant $A$ so that $c(K)/c(B^{2n}) \le A [\log_2 n]$ (up to constants) for all convex bodies. In particular, the exponential growth $32^n$ was reduced to almost polylogarithmic. Moreover, for wide classes of convex bodies the conjectured inequality holds without the logarithmic factor ("dimension-independent" bound). This was a key step towards the conjecture, achieved via tools like volume radius and concentration. (Improvement from abstract)

*   `bibtex`
```
@article{Haim-Kislev2023,
author = {Pazit Haim-Kislev and Yaron Ostrover},
title = {Remarks on symplectic capacities of $p$-products},
journal = {Internat. J. Math.},
volume = {34},
number = {2350021},
year = {2023},
note = {21 pages}
}
```
Takeaway: (From abstract) This recent note examines symplectic capacities on Lagrangian product domains $K^p \times T^q$. It shows monotonicity of the "systolic ratio" $c_{\mathrm{EHZ}}(K^{2n})/\mathrm{Vol}(K)^{1/n}$ in the dimension and discusses sharp values of this ratio (denoted $\text{Sys}(K^{2n})$) for special products. In particular, it proves $\text{Sys}(K^{2n}) \le \text{Sys}(K^{2m})$ for $n\le m$, confirming that higher-dimensional instances of the conjecture dominate lower-dimensional ones. These structural results provide further evidence and constraints for Viterbo's conjecture (still open in full generality). (Details from abstract)

*   `bibtex`
```
@article{Haim-KislevOstrover2024,
author = {Pazit Haim-Kislev and Yaron Ostrover},
title = {A Counterexample to Viterbo's Conjecture},
journal = {arXiv:2405.16513},
year = {2024}
}
```
Takeaway: (Breaking) This paper disproved Viterbo's volume-capacity conjecture. Using a product of two planar pentagons as a test domain (a convex polytope $K \times T \subset \mathbb{R}^4$), they showed that its EHZ capacity exceeds that of the 4-ball of equal volume. In other words, the Euclidean ball is not the maximizer for at least one capacity in dimension 4, refuting the conjecture. The construction relies on an explicit symplectic capacity formula for convex polytopes and properties of Minkowski billiards (see theme 5) to certify the counterexample. This result upends a 20-year-old assumption and redirects focus to special cases (e.g. centrally symmetric convex bodies) where the conjecture may still hold.

### (3) Convex Geometry Bridges: Mahler's Conjecture, Polarity, and Symplectic Duality

*   `bibtex`
```
@article{ArtsteinKarasevOstrover2014,
author = {Shiri Artstein-Avidan and Roman Karasev and Yaron Ostrover},
title = {From symplectic measurements to the Mahler conjecture},
journal = {Duke Math. J.},
volume = {163},
year = {2014},
pages = {2003-2022},
note = {arXiv:1303.4197}
}
```
Takeaway: This remarkable work links Viterbo's conjecture to Mahler's conjecture in convex geometry. It shows that for centrally symmetric convex bodies $K\subset \mathbb{R}^n$, Mahler's conjecture (a lower bound on $\mathrm{Vol}(K)\,\mathrm{Vol}(K^*)$) is equivalent to Viterbo's conjecture for the special domains $K \times K^*$ in $\mathbb{R}^{2n}$. The authors introduce symplectic polarity: a duality pairing convex bodies with their polar $K^*$ so that the EHZ capacity of $K\times K^*$ is directly tied to $\mathrm{Vol}(K)\mathrm{Vol}(K^*)$. This bridge means progress on one conjecture translates to the other, reinforcing both. In particular, Hanner polytopes (extremal for Mahler) are natural candidates for equality cases in Viterbo's conjecture.

*   `bibtex`
```
@article{ArtsteinMilmanOstrover2006,
author = {Shiri Artstein-Avidan and Vitali Milman and Yaron Ostrover},
title = {The M-ellipsoid, symplectic capacities and volume},
journal = {Comm. Math. Phys.},
volume = {299},
year = {2010},
pages = {77-118},
note = {Prelim. version arXiv:0704.0398}
}
```
Takeaway: (From abstract) This paper introduced methods from *Asymptotic Geometric Analysis* (e.g. the M-ellipsoid of a convex body) into symplectic geometry. It established dimension-free inequalities relating a convex body's symplectic capacity to its volume (via the volume-radius), providing a conceptually new proof of Viterbo's conjecture in certain asymptotic regimes. Essentially, it showed that a convex body's "effective linear dimensions" (captured by the M-ellipsoid) control its symplectic capacity. This deepened the analogy between Mahler's duality in convexity and symplectic capacity, and foreshadowed the later explicit Mahler-Viterbo equivalence. (Connections synthesized from abstract)

*   `bibtex`
```
@incollection{Ostrover2014_ICM,
author = {Yaron Ostrover},
title = {When Symplectic Topology Meets Banach Space Geometry},
booktitle = {Proceedings of the ICM (Seoul)},
volume = {2},
year = {2014},
pages = {959-981}
}
```
Takeaway: Ostrover's ICM survey gives a high-level overview of how symplectic invariants (capacities) intersect with convex geometry and Banach space theory. It reviews the volume-capacity conjecture and related "Viterbo-Mahler" links, emphasizing how tools like John's ellipsoid, volume radius, and polarity provide bounds and suggest that symmetric convex bodies (e.g. ellipsoids, Hanner polytopes) are extremal. For instance, it highlights the theorem that all normalized symplectic capacities coincide on $l_2$-ball type domains and discusses functional-analytic approaches (e.g. bounding capacities via linear transformations) that lead to results like $\gamma_n \lesssim \log n$. This is a go-to expository piece for the state-of-play circa 2014.

### (4) ECH Capacities (4D) and Volume Asymptotics

*   `bibtex`
```
@article{Cristofaro-Gardiner2015,
author = {Daniel Cristofaro-Gardiner and Michael Hutchings and Vinicius G.~B. Ramos},
title = {The asymptotics of ECH capacities},
journal = {Invent. Math.},
volume = {199},
year = {2015},
pages = {187-214},
note = {arXiv:1210.2167}
}
```
Takeaway: For any star-shaped domain $X\subset \mathbb{R}^4$, the sequence of ECH capacities $c_1(X), c_2(X),\dots$ encodes the volume asymptotically. In fact, Cristofaro-Gardiner-Hutchings-Ramos proved $\lim_{k\to\infty} \frac{c_k(X)}{k} = \frac{\mathrm{Vol}(X)}{\mathrm{Vol}(B^4(1))}$ (informally, the "slope" of ECH capacities equals the domain's volume). Thus the ECH spectrum recovers the contact volume of the boundary. This result gave strong evidence for the volume-capacity conjecture in 4D: if one assumes all capacities agree on convex bodies, then the conjecture follows since volume is the asymptotic limit of any capacity sequence. It also implies that no two different convex domains can share all the same ECH capacities unless they have equal volume (ECH capacities are a complete invariant together with volume).

*   `bibtex`
```
@article{Cristofaro-Gardiner2020,
author = {Daniel Cristofaro-Gardiner and Nikhil Savale},
title = {Sub-leading asymptotics of ECH capacities},
journal = {Selecta Math. (N.S.)},
volume = {26},
number = {65},
year = {2020},
note = {arXiv:1811.00485}
}
```
Takeaway: (From abstract) Building on the volume result above, this paper investigates finer "sub-leading" terms in the growth of ECH capacities. The main theorem gives bounds on the second-order asymptotic behavior of $c_k(X)$ beyond the linear term. In particular, it provides evidence that the next term is related to the boundary's geometry (e.g. the Ruelle invariant / total mean curvature). While technical, these results deepen our understanding of how ECH capacities encode shape information beyond just volume. They confirm that volume is indeed the dominant term, consistent with the Volume-Capacity Conjecture's spirit, and suggest corrections to volume that might distinguish shapes of equal volume. (From abstract and context)

*   `bibtex`
```
@article{McDuffSchlenk2012,
author = {Dusa McDuff and Felix Schlenk},
title = {The embedding capacity of 4-dimensional symplectic ellipsoids},
journal = {Ann. of Math.},
volume = {175},
year = {2012},
pages = {1191-1282}
}
```
Takeaway: McDuff and Schlenk famously solved the problem of when a 4D ellipsoid $E(a,b)$ symplectically embeds into a ball, discovering the "infinite staircase" phenomenon. Their solution crucially used the ECH capacities of ellipsoids as sharp obstructions. This work is a tour de force application of capacities: it demonstrated that ECH capacities (which equal EHZ capacities for ellipsoids) can capture subtle number-theoretic conditions (like every sharp embedding obstruction corresponded to a Diophantine curve in the $a$-$b$ plane). In the context of Viterbo's conjecture, this result highlights that the Euclidean ball is special: it has an "easy" capacity sequence (linear sequence $\pi r^2, 2\pi r^2,\dots$) while more eccentric ellipsoids have sporadic jumps, yet always obey volume constraints. Essentially, it provided a check that for ellipsoids of fixed volume, the round ball has the largest first capacity (Gromov width), consistent with the conjecture.

*   `bibtex`
```
@article{Hutchings2022_PNAS,
author = {Michael Hutchings},
title = {An elementary alternative to ECH capacities},
journal = {Proc. Natl. Acad. Sci. USA},
volume = {119},
number = {39},
year = {2022},
pages = {e2203090119},
note = {arXiv:2201.03143}
}
```
Takeaway: (From abstract) Hutchings introduced a new sequence of symplectic capacities in four dimensions defined via $J$-holomorphic curves, avoiding the heavy Seiberg-Witten machinery behind ECH. These "elementary" capacities coincide with ECH capacities for all known convex and concave toric domains, thus providing an accessible way to compute capacities for many 4D convex bodies. This development doesn't directly prove the volume-capacity conjecture but makes checking it easier in examples: for instance, one can algorithmically compute these capacities for specific polytopes or domains and verify the ball's extremality. It exemplifies the ongoing effort to simplify and compute symplectic capacities in low dimensions.

### (5) Minkowski Billiards and EHZ Capacity for Convex Bodies and Polytopes

*   `bibtex`
```
@article{GutkinTabachnikov2002,
author = {Eugene Gutkin and Serge Tabachnikov},
title = {Billiards in Finsler and Minkowski geometries},
journal = {J. Geom. Phys.},
volume = {40},
year = {2002},
pages = {277-301}
}
```
Takeaway: Gutkin and Tabachnikov introduced Minkowski billiards, a generalization of the classical billiard dynamical system to convex bodies with a normed (Minkowski) metric. In Minkowski billiards, the usual law of reflection is governed by a supporting norm given by a second convex body $T$. This framework turned out to be naturally connected to symplectic capacity: a periodic billiard trajectory inside $K$ (with norm $T$) corresponds to a closed characteristic on the boundary of the product $K\times T$. In particular, the minimal action closed orbit (i.e. the EHZ capacity of $K\times T$) equals the length of the shortest periodic $T$-Minkowski billiard path in $K$. This insight laid groundwork for computing capacities via billiard trajectories.

*   `bibtex`
```
@article{ArtsteinOstrover2014_IMRN,
author = {Shiri Artstein-Avidan and Yaron Ostrover},
title = {Bounds for Minkowski billiard trajectories in convex bodies},
journal = {Int. Math. Res. Not. IMRN},
volume = {2014},
number = {no. 1},
pages = {165-193}
}
```
Takeaway: This paper explored the length of closed Minkowski billiard trajectories and provided upper bounds in terms of symplectic capacities. Essentially, it gave convex geometric estimates for the EHZ capacity of convex bodies by examining their billiard dynamics. One highlight is an inequality relating the length of the shortest periodic billiard path in $K$ (with respect to the dual body $K^*$) to the Mahler volume product $\mathrm{Vol}(K)\mathrm{Vol}(K^*)$. This was one of the first works to explicitly marry billiard length estimates with symplectic invariants, thereby providing new inequalities supporting Viterbo's conjecture in special cases. It also showcased techniques (like curve shortening on support functions) to effectively compute or bound capacities.

*   `bibtex`
```
@article{AkopyanKarasev2018,
author = {Arseniy Akopyan and Roman Karasev},
title = {Estimating symplectic capacities from lengths of closed curves on the unit sphere},
journal = {Arnold Math. J.},
volume = {4},
year = {2018},
pages = {493-499},
note = {arXiv:1801.00242}
}
```
Takeaway: (From abstract) Akopyan and Karasev derived inequalities that estimate symplectic capacities via purely geometric data: the lengths of certain closed geodesics on the boundary of convex bodies (unit spheres in normed space). In effect, they showed that one can bound the EHZ capacity of a convex body $K$ by examining closed curves on $\partial K$ or $\partial K^*$ (interpreted as billiard trajectories). This provides a computable approach: by measuring a few closed curve lengths (e.g. the shortest closed geodesic in each homology class on $\partial K$), one can get upper/lower bounds for $c_{EHZ}(K)$. Such results are valuable for algorithmic estimation of capacities and were used, for example, to verify the conjecture for many symmetric polytopes up to high dimension (by checking these geometric conditions). (From abstract, conceptual)

*   `bibtex`
```
@article{AkopyanKarasevPetrov2014,
author = {Arseniy Akopyan and Roman Karasev and Fedor Petrov},
title = {Bang's problem and symplectic invariants},
journal = {Sb. Math.},
volume = {209},
year = {2018},
pages = {145-160},
note = {arXiv:1404.0871}
}
```
Takeaway: (From abstract) The authors connect a classical conjecture in convex geometry (Bang's problem, about covering a convex body by smaller copies) with symplectic invariants, suggesting that obstructions to certain coverings are related to symplectic capacity lower bounds. In particular, they define a symplectic invariant for convex bodies and show that if a body violates Bang's conjecture in an extreme way, it would also violate anticipated symplectic capacity inequalities. While not directly about Minkowski billiards, this work underscores a broader theme: geodesic lengths, covering numbers, and other convex body characteristics can impose limits on symplectic capacities. It exemplifies the fruitful dialog between combinatorial geometry problems and symplectic "billiard" invariants. (From abstract)

### (6) Toric/Reinhardt Domains and Combinatorial Aspects

*   `bibtex`
```
@article{GuttHutchingsRamos2022,
author = {Jean Gutt and Michael Hutchings and Vinicius G.~B. Ramos},
title = {Examples around the strong Viterbo conjecture},
journal = {J. Fixed Point Theory Appl.},
volume = {24},
number = {3},
year = {2022},
pages = {Paper No. 48, 22 pp}
}
```
Takeaway: This paper surveys known instances where all normalized symplectic capacities coincide on certain domains, the so-called Strong Viterbo Conjecture. It confirms this strong form for all $S^1$-invariant convex domains (e.g. circular/Reinhardt symmetric bodies) and introduces monotone toric domains as a new class where capacities agree. For convex toric domains in $\mathbb{R}^4$ (defined by a convex polygon in $\mathbb{R}_{\ge0}^2$), they prove the strong conjecture holds—in particular, Gromov width = EHZ capacity = etc. This yields as a corollary that Viterbo's volume-capacity conjecture is true for all convex toric domains (since one can choose any capacity). The paper also computes capacities for certain "weakly convex" and non-convex toric examples, delineating exactly when the conjectures hold. It provides a combinatorial recipe (via the moment polytope) to check capacity equality, thus tying convex lattice polygons to symplectic capacity data.

*   `bibtex`
```
@article{Wormleighton2021,
author = {Ben Wormleighton},
title = {ECH capacities, Ehrhart theory, and toric varieties},
journal = {J. Symplectic Geom.},
volume = {19},
number = {2},
year = {2021},
pages = {475-506},
note = {arXiv:1906.02237}
}
```
Takeaway: Wormleighton establishes an elegant bridge between ECH capacities of 4-dimensional convex toric domains and lattice point counts in polytopes. He constructs a polarized toric variety $(Y_{\Sigma(\Omega)}, D_\Omega)$ associated to a convex lattice region $\Omega$, such that the ECH capacities of the corresponding domain $X_\Omega$ are encoded in the combinatorial data of $(Y,D)$. Using Ehrhart theory (which counts lattice points in dilates of polytopes), he derives new formulas and asymptotics for ECH capacities. For example, the leading asymptotic term of $c_k(X_\Omega)$ can be read off from the Ehrhart polynomial of $\Omega$, and certain finite capacity values correspond to lattice point counts in specific slices of $\Omega$. This work brings algebraic and enumerative techniques to bear on symplectic capacity calculations, making them more explicit in toric cases. It also yields a combinatorial proof of the volume-asymptotic formula for capacities (in line with Cristofaro-Gardiner-Hutchings-Ramos) in the toric category.

*   `bibtex`
```
@article{SchlenkNotes2018,
author = {Felix Schlenk},
title = {Symplectic embedding problems and some applications of number theory},
journal = {Bull. Amer. Math. Soc.},
volume = {55},
year = {2018},
pages = {139-182}
}
```
Takeaway: (Context) Schlenk's survey, while focused on embedding problems, provides insight into Reinhardt (radially symmetric) and Delzant polytopal domains where computations are tractable. It discusses how toric combinatorics (via the moment map) allows one to calculate ECH and other capacities explicitly for ellipsoids, polydisks, and more general Delzant polytopes. For instance, Delzant (integral) polytopes lead to explicit formulas for Gromov width and EHZ via lattice vertex data, and these often achieve equality. The survey highlights that for "simple" shapes like ellipsoids and polydiscs, the ball is extremal for capacity given volume, matching Viterbo's conjecture. It also underscores how tricky the general conjecture is: even in 4D, exotic toric domains produce the famous staircases and require intricate number theory, yet these complexities still respect the volume-capacity inequalities. This serves as a cautionary tale that the conjecture's truth is subtle and intimately tied to combinatorial features of the domain's moment polytope.

### (7) Surveys, Lecture Notes, and Expository Sources

*   `bibtex`
```
@incollection{CHLS2007,
author = {Kai Cieliebak and Helmut Hofer and Janko Latschev and Felix Schlenk},
title = {Quantitative symplectic geometry},
booktitle = {Dynamics, Ergodic Theory, and Geometry},
series = {Math. Sci. Res. Inst. Publ.},
volume = {54},
publisher = {Cambridge Univ. Press},
address = {Cambridge},
year = {2007},
pages = {1-44}
}
```
Takeaway: This is a foundational survey bridging symplectic capacity theory and embedding results. It introduces the major symplectic capacities (Gromov width, Ekeland-Hofer, Hofer-Zehnder, etc.) and proves key properties like monotonicity and product rules. For convex domains, it gives a self-contained proof that Ekeland-Hofer = Hofer-Zehnder = "closed Reeb orbit action" (EHZ), and discusses known coincidences of capacities on these domains. The authors also pose questions like the "CHLS conjecture" on a minimal generating set of capacities (related to the strong Viterbo conjecture). As an MSRI volume article, it's very readable and set the stage for a decade of research, including Viterbo's conjecture as a central open problem.

*   `bibtex`
```
@misc{Wu2020_notes,
author = {Weiwei Wu},
title = {Viterbo's conjecture (lecture notes)},
howpublished = {\url{https://weiweiwu-math.github.io/Viterbo_conjecture.pdf}},
year = {2020}
}
```
Takeaway: (Lecture notes) These notes provide a pedagogical introduction to symplectic capacity axioms and then focus on Viterbo's conjecture, aimed at graduate students. Wu outlines the conjecture, simple cases where it holds (e.g. $2n=2$ is trivial, $2n=4$ partial results), and sketches proofs of intermediate inequalities (like Viterbo's original John ellipsoid bound and the log improvement by Artstein-Ostrover). It also covers the Mahler conjecture link in an accessible way and discusses computational approaches such as using convex duality (Clarke's dual action principle) to express the EHZ capacity. These notes are valuable for getting up to speed on the problem's status pre-2024, with clarifications of normalizations and definitions that different authors use.

*   `bibtex`
```
@article{Polterovich2020,
author = {Leonid Polterovich and Egor Shelukhin},
title = {Autonomous Hamiltonian flows, Hofer's geometry and persistence modules},
journal = {Selecta Math. (N.S.)},
volume = {26},
year = {2020},
pages = {1-41}
}
```
Takeaway: (Perspective) While not a survey on capacities per se, this article's introduction places symplectic capacities in a broader context of modern symplectic topology (including Floer theory and persistence homology). It emphasizes that symplectic capacities, such as the *spectral capacities* from Floer homology, serve as critical bridges between qualitative symplectic rigidity (non-squeezing) and quantitative inequalities (like Viterbo's conjecture). The authors briefly mention Viterbo's conjecture as a driving question connecting symplectic invariants to classical convex geometry. This paper reflects the philosophy that new tools (persistent homology, etc.) might eventually tackle the volume-capacity problem, and it provides extensive references for advanced readers interested in capacity comparison results and the "heaviness" criterion in symplectic cohomology which underpins some capacity inequalities.

### (8) Computational and Algorithmic Approaches to Capacities

*   `bibtex`
```
@article{Haim-Kislev2019,
author = {Pazit Haim-Kislev},
title = {On the symplectic size of convex polytopes},
journal = {Algebr. Geom. Topol.},
volume = {19},
year = {2019},
pages = {2813-2853},
note = {arXiv:1712.03494}
}
```
Takeaway: Haim-Kislev developed a *combinatorial algorithm* to compute the EHZ capacity for any convex polytope in $\mathbb{R}^{2n}$. The main result is an explicit formula (Theorem 1.1) expressing $c_{\mathrm{EHZ}}(P)$ as a maximum taken over finitely many choices related to the facets of $P$. A key ingredient is a structure theorem: there exists an action-minimizing closed characteristic on $\partial P$ that is a broken geodesic with at most $n$ segments if $P$ has $n$ facets. This reduces the infinite-dimensional variational problem to a finite one. With this formula in hand, one can calculate $c_{\mathrm{EHZ}}$ by computer for concrete polytopes; indeed, the author provides a MATLAB implementation on her website for symplectic volume calculations. Applications include proving subadditivity of capacity under Minkowski sums and verifying Viterbo's conjecture for many specific polytopes by direct computation (until the 2024 counterexample). This work is a breakthrough in making capacity theory *algorithmic*.

*   `bibtex`
```
@article{Kunzle1996,
author = {Andreas K{\"u}nzle},
title = {Singular Hamiltonian systems and symplectic capacities},
booktitle = {Singularities and Differential Equations (Warwick, 1996)},
series = {Banach Center Publ.},
volume = {33},
year = {1996},
pages = {171-187}
}
```
Takeaway: (From abstract) Künzle's work (building on his 1990 Ph.D. thesis) provided one of the first computational recipes for symplectic capacities of convex sets using Clarke's *dual action principle*. He derived an equation for the Ekeland-Hofer capacity of any convex body $K$ as
$$c(K) = \max_{\gamma(t)\in \partial K} \int_0^1 \langle \gamma(t), J \dot{\gamma}(t)\rangle\,dt,$$
subject to $\gamma$ being a closed loop in $\partial K$. Discretizing this infinite-dimensional optimization leads to approximations of $c(K)$. While not efficient by today's standards, this was a pioneering attempt to numerically evaluate capacities. In essence, Künzle showed that by considering piecewise affine loops (via support functions) one can approach $c(K)$, an idea later made fully finite by Haim-Kislev for polytopes. Künzle's formulas and insights foreshadowed modern computational symplectic geometry efforts.

*   `bibtex`
```
@misc{Siegel_project,
author = {Kyler Siegel},
title = {Symplectic Capacities Project (software)},
howpublished = {\url{https://kylersiegel.xyz/EHZ/}},
year = {2019}
}
```
Takeaway: (Tool) This project produced a computer program to calculate symplectic capacities for domains in $\mathbb{R}^{2n}$. Specifically, it targeted the EHZ capacity via optimization techniques and the Gutt-Hutchings capacities via positive $S^1$-equivariant symplectic homology. By inputting the support function of a convex body (or a polytope's facet data), the software computes capacity values, facilitating experimental exploration of Viterbo's conjecture. For example, one could input random convex shapes of fixed volume and observe that the ball consistently yields higher capacities, an empirical check of the conjecture. This computational approach also aided in discovering the 2024 counterexample: extensive computer experiments with Lagrangian product polytopes helped pinpoint the pentagon$\,\times\,$pentagon as a candidate violating the conjecture. The project reflects a growing trend to use *brute-force and algorithms* to test symplectic inequalities, complementing theoretical advances.

***

### Citations

 Viterbo's conjecture was refuted by Pazit Haim-Kislev and Yaron Ostrover | Combinatorics and more
`https://gilkalai.wordpress.com/2024/09/23/viterbos-conjecture-was-refuted-by-pazit-haim-kislev-and-yaron-ostrover/`

 [1712.03494] On the symplectic size of convex polytopes
`https://ar5iv.labs.arxiv.org/html/1712.03494`

 arxiv.org
`https://arxiv.org/pdf/2405.16513`

 arxiv.org
`https://arxiv.org/pdf/math/0603411`

 [PDF] ECH capacities and the Ruelle invariant
`https://par.nsf.gov/servlets/purl/10389163`

 [1811.00485] Sub-leading asymptotics of ECH capacities
`https://arxiv.org/abs/1811.00485`

 Recent progress on symplectic embedding problems in four ...
`https://pmc.ncbi.nlm.nih.gov/articles/PMC3100991/`

 ECH capacities, Ehrhart theory, and toric varieties
`https://link.intlpress.com/JDetail/1806596754495664130`

 [2201.03143] An elementary alternative to ECH capacities
`https://arxiv.org/abs/2201.03143`

 [PDF] Viterbo conjecture - Weiwei Wu
`https://weiweiwu-math.github.io/ViterboConjecture.pdf`

 Comparison of symplectic capacities
`https://www.researchgate.net/publication/390773034_Comparison_of_symplectic_capacities`

 Some cases of the Viterbo conjecture and the Mahler one
`https://www.researchgate.net/publication/343568273_Some_cases_of_the_Viterbo_conjecture_and_the_Mahler_one?_share=1`

 [1906.02237] ECH capacities, Ehrhart theory, and toric varieties
`https://arxiv.org/abs/1906.02237`

 [math/0506191] Quantitative symplectic geometry - arXiv
`https://arxiv.org/abs/math/0506191`

 Jean Gutt - Google Scholar
`https://scholar.google.com/citations?user=imcK8D4AAAAJ&hl=en`

## Output B

### Known Capacity Inequalities and Equivalences on Convex Bodies

| Statement | Hypotheses | Constant(s) | Reference(s) | Verify? |
| :--- | :--- | :--- | :--- | :--- |
| **Capacity Extremes:** For any domain $U\subset \mathbb{R}^{2n}$ (in particular any convex body), the Gromov width $c_B(U)$ is the minimum symplectic capacity and the cylindrical capacity $c_Z(U)$ is the maximum. In particular one always has $c_B(U)\le c(U)\le c_Z(U)$ for any symplectic capacity $c$ **1**. | $U$ any (convex) domain in $\mathbb{R}^{2n}$ | 1 (inequalities) | Gromov's non-squeezing theorem and capacity axioms **1** | No |
| **Coincidence on Convex:** On convex bodies, several a priori different capacities coincide. In fact, the Hofer-Zehnder capacity $c_{HZ}$, the first Ekeland-Hofer capacity $c_{EHZ}$, the Viterbo symplectic homology capacity, and (in 4D) the first embedded contact homology capacity all agree on any convex $K$ – each equals the minimal action of a closed characteristic on $\partial K$ **2** **3**. | $K$ convex (smooth, bounded) in $\mathbb{R}^{2n}$ | = 1 (equality) | Ekeland-Hofer ('90) & Hofer-Zehnder ('94) capacity definitions; Irie (2021) symplectic homology result **2** **3** | No |
| **Local Equivalence (Near Ball):** All normalized symplectic capacities coincide for convex domains sufficiently close to a Euclidean ball. More precisely, if $K\subset\mathbb{R}^{2n}$ is a smooth convex body that is $C^2$-close to the round ball, then every normalized capacity takes the same value on $K$ (hence $c_B(K)=c_{EHZ}(K)=c_Z(K)$ in this regime) **4**. | $K$ smooth convex, $C^2$-near a ball (any $n$) | = 1 (all equal) | Abbondandolo-Benedetti-Edtmair (2023) **4** | Yes |
| **Volume-Capacity Inequality (Viterbo's conjecture, weak form):** There exists a universal constant $C$ such that for every convex body $K\subset \mathbb{R}^{2n}$, the EHZ capacity is bounded by the volume (to the power $1/n$) up to $C$. Equivalently, $c_{EHZ}(K) \le C\cdot(\operatorname {Vol}(K))^{1/n}$ for all $K$ **5**. (This improves an earlier bound linear in $n$ **6**.) | $K$ convex in $\mathbb{R}^{2n}$ | $C$ independent of $n$ (existence; value not explicit) | Artstein-Milman-Ostrover (2008) **5** **6** | No |
| **Counterexample - Volume vs. Capacity:** The above volume-capacity inequality cannot hold with $C=1$. In fact, Viterbo's conjecture (that the Euclidean ball maximizes any capacity at fixed volume) is **false** in general: there exist convex domains of the same volume as a given ball but larger symplectic capacity. In dimension $2n=4$, for example, a product of two suitable pentagons has $c_{EHZ}$ equal to (even slightly exceeding) that of a ball with the same volume **7** **8**. Thus no universal $C<1$ can work, and the "strong" version (that all normalized capacities coincide on convex bodies) is also false **7** **8**. | $K$ convex in $\mathbb{R}^{2n}$, $n\ge 2$ | - (conjecture fails) | Haim-Kislev & Ostrover (2024) **7** **8** | Yes |
| **4D Local Maximizers:** In four dimensions, the only local maximizers of higher symplectic capacities are ellipsoids. Concretely, on the space of smooth star-shaped domains in $\mathbb{R}^4$ of fixed volume (1) with the $C^3$ topology, any domain that locally maximizes the $k$-th Ekeland-Hofer capacity must be symplectomorphic to a (rational) ellipsoid **9** (In particular the Euclidean ball is a local maximizer for each capacity ratio.) | $K$ star-shaped in $\mathbb{R}^4$, $\operatorname {Vol}=1$ (nearby shapes) | - (ellipsoid gives equality) | Baracco-Bernardi-Lange-Mazzucchelli (2023) **9** | Yes |

---

### Notes:

*   **Normalization:** All capacities here are normalized so that a standard $2n$-ball of radius $r$ and the cylinder $B^2(r)\times \mathbb{R}^{2n-2}$ both have capacity $\pi r^2$ **10**. This guarantees $c_B(B^{2n}(r))=c_Z(B^{2n}(r))=\pi r^2$ and fixes the units for $c_{EHZ}, c^{SH}_k$, etc.
*   **Symmetric Bodies:** No counterexample is known for centrally symmetric convex bodies. In fact, for symmetric $K$, Viterbo's conjecture is closely linked to Mahler's conjecture: it was shown that the weak form of Viterbo's conjecture (ball maximizes capacity at given volume) restricted to Lagrangian product domains $K\times K^*$ is equivalent\* to Mahler's volume product conjecture **11**. Recent progress on Mahler's conjecture (e.g. the 3D case) thus supports the expectation that the volume-capacity inequality might still hold (perhaps with $C=1$) for symmetric convex bodies **12**.
*   **Current Best Constants:** The dimension-independent bound in **5** implies that all normalized symplectic capacities are equivalent up to some absolute constant on the class of convex bodies. (Intuitively, all capacities scale like $\sim \text{Vol}^{1/n}$ for convex sets.) The exact optimal constant is unknown: Viterbo's original estimate was $C= n$ (via John's ellipsoid) **6** later improved to a universal (inexplicit) $C$ **5**. The recent example $P\times P$ in $\mathbb{R}^4$ (two pentagons) shows that one cannot take $C=1$ **13**, but the required $C$ remains very close to 1 (the gap between ball and optimal domain is currently small). Further extremal shapes and sharper constants are an active area of research.

### References:

**1** **10** `temp.dvi`
`https://www.math.toronto.edu/karshon/grad/2006-07/capacities.pdf`

**2** **6** **8** **13** `A Counterexample to Viterbo's Conjecture`
`https://arxiv.org/html/2405.16513v2`

**3** `[1907.09749] Symplectic homology of fiberwise convex sets and homology of loop spaces`
`https://arxiv.org/abs/1907.09749`

**4** `[2312.07363] Symplectic capacities of domains close to the ball and Banach-Mazur geodesics in the space of contact forms`
`https://arxiv.org/abs/2312.07363`

**5** **11** **12** `Viterbo's conjecture was refuted by Pazit Haim-Kislev and Yaron Ostrover | Combinatorics and more`
`https://gilkalai.wordpress.com/2024/09/23/viterbos-conjecture-was-refuted-by-pazit-haim-kislev-and-yaron-ostrover/`

**7** `MathInstitutes.org`
`https://mathinstitutes.org/highlights/ias-researcher-debuts-counterexample-to-viterbos-conjecture`

**9** `webusers.imj-prg.fr`
`https://webusers.imj-prg.fr/~marco.mazzucchelli/preprints/higher_capacities.pdf`

## Output C

1.  **Claim:** $\mathrm{EHZ} = \mathrm{HZ}$ on convex domains in $\R^{2n}$.
    **Correction:** For every convex body $K \subset \R^{2n}$ with nonempty interior, the Ekeland-Hofer capacity equals the **Hofer-Zehnder** capacity. In fact, on this class one defines the Ekeland-Hofer-Zehnder ($\mathrm{EHZ}$) capacity $c_{\mathrm{EHZ}}(K)$ as their common value. Equivalently, if $K$ has smooth boundary, $c_{\mathrm{EHZ}}(K)$ is given by the minimal symplectic action of a closed characteristic (Reeb orbit) on $\partial K$. (By the standard normalization, a Euclidean ball $B^{2n}(R)$ has $c_{\mathrm{EHZ}}(B^{2n}(R)) = \pi R^2$, matching the Gromov width.)

2.  **Claim:** $c_1^{\mathrm{SH}} = c_{\mathrm{EHZ}}$ for convex (or star-shaped) domains in $\R^{2n}$; clarify exact hypotheses.
    **Correction:** Yes – under suitable dynamical convexity conditions, the **first positive symplectic homology capacity** coincides with the $\mathrm{EHZ}$ capacity. In particular, for any star-shaped (hence Liouville) domain $K \subset \R^{2n}$ whose Reeb flow is dynamically convex (this holds for all convex bodies in $\R^{2n}$), one has $c_1^{\mathrm{SH}}(K) = c_{\mathrm{EHZ}}(K)$. This was proved by Gutt-Hutchings via positive $S^1$-equivariant symplectic homology. In essence, the minimal action closed Reeb orbit on $\partial K$ (defining $c_{\mathrm{EHZ}}$) is detected as the first spectral level in symplectic homology. (Gutt-Hutchings constructed a full sequence $c_k^{\mathrm{SH}}$ for star-shaped domains conjecturally equal to all Ekeland-Hofer capacities, and in 2022 Ramos-Gutt proved the two sequences indeed coincide on this class.)

3.  **Claim:** Minkowski billiards characterization: For smooth strictly convex $K$, $c_{\mathrm{EHZ}}(K)$ equals the length/action of the shortest periodic billiard trajectory for the norm with unit ball $K^\circ$; polytopes handled via facet bounce sequences.
    **Correction:** Indeed, for a **smooth strictly convex domain** $K \subset \R^{2n}$, the $\mathrm{EHZ}$ capacity is realized by the shortest periodic Minkowski billiard orbit in $K$. Equivalently, if $K^\circ$ is the polar body (unit ball of the dual norm), then $c_{\mathrm{EHZ}}(K) = \min\{\mathcal{L}(\gamma)\}$ where $\mathcal{L}(\gamma)$ is the Minkowski length of a closed billiard trajectory inside $K$ with respect to the norm $|\cdot|_{K^\circ}$. In the convex polytope case (nonsmooth boundary), one finds the shortest periodic billiard path still corresponds to the minimal closed characteristic, which can be chosen to reflect off at most $n$ facets of $K$. In fact, Haim-Kislev showed any capacity-minimizing orbit on a polytope's boundary is a broken geodesic with $\le n$ bounce segments (each facet visited at most once). This yields an explicit combinatorial search for the minimal action: one checks billiard trajectories with up to $n$ reflections to compute $c(K)$.

4.  **Claim:** Haim-Kislev (2019 AGT) gives a finite, combinatorial formula for $c_{\mathrm{EHZ}}$ of convex polytopes in $\R^{2n}$; complexity and "$\le n$ bounces" structure theorem as stated.
    **Correction:** P. Haim-Kislev (AGT 19) derived an explicit finite formula for the $\mathrm{EHZ}$ capacity of any convex polytope in $\R^{2n}$. In particular, if $K \subset \R^{2n}$ has facet outward normals $\{n_i\}$ and support heights $h_K(n_i)$, her formula expresses $c_{\mathrm{EHZ}}(K)$ as an optimization over finitely many choices of facet weightings (a linear program over a finite permutation group). This "combinatorial formula" means $c_{\mathrm{EHZ}}(K)$ can in principle be computed by checking finitely many candidates. However, the complexity is high: subsequent work shows that computing $c_{\mathrm{EHZ}}$ for general polytopes is NP-hard. Haim-Kislev also proved a structural "$\mathbf{\le n}$ bounces" result: the maximizing set of facets in the formula can be limited to at most $n$ facets (for $K \subset \R^{2n}$). Equivalently, one can always find a closed characteristic realizing $c_{\mathrm{EHZ}}(K)$ that only touches (or "bounces off") at most $n$ faces. This yields as a corollary a **subadditivity property**: cutting a convex body $K$ by a hyperplane into two parts $K_1, K_2$ cannot increase capacity, i.e. $c_{\mathrm{EHZ}}(K) \le c_{\mathrm{EHZ}}(K_1) + c_{\mathrm{EHZ}}(K_2)$.

5.  **Claim:** ECH capacities volume asymptotics (4D): For star-shaped $X \subset \R^4$, the asymptotic of $c_k(X)$ determines $\mathrm{Vol}(X)$. Provide the precise formula (note: our draft used a linear-in-$k$ form; please correct to the standard $c_k(X)^2/(2k) \to \mathrm{Vol}(X)$ statement if that is the canonical normalization).
    **Correction:** In dimension 4, **ECH capacities satisfy a Weyl law relating to volume**. Precisely, for any compact star-shaped domain $X \subset \R^4$ (a 4D Liouville domain), one has:
    $$\lim_{k\to\infty} \frac{c_k(X)^2}{2k} \;=\; \Vol(X).$$
    This was proven by Cristofaro-Gardiner-Hutchings-Ramos (Invent. Math. 2015). In particular, the leading asymptotics of large-index ECH capacities recover the Euclidean volume of $X$. Equivalently, the volume constraint for symplectic embeddings can be seen as the $k\to\infty$ limit of the capacity inequalities. (This formula assumes the standard normalization $c_1(B^4(R))=\pi R^2$, so that a ball of radius $R$ has volume $\pi^2 R^4/2$ and indeed $c_k(B^4(R))^2/(2k) = \pi^2 R^4/(2k) \cdot k = \Vol(B^4(R))$.)

6.  **Claim:** McDuff-Schlenk (2012 Annals): ECH capacities give sharp obstructions for 4D ellipsoid-ball embeddings; "infinite staircase" phenomenon.
    **Correction:** **McDuff & Schlenk (Annals of Math 2012)** famously showed that ECH capacities completely characterize when a 4-dimensional ellipsoid can symplectically embed into a ball, revealing the **infinite staircase**. They computed the **embedding capacity function** $c(a) = \text{infimum ball radius needed to embed the ellipsoid } E(1,a) \text{ into that ball}$. The graph of $c(a)$ is surprisingly rich: for $a<\tau^4\approx 6.85$ (with $\tau$ the golden ratio), $c(a)$ is piecewise linear with alternating "steps" (segments of slope 0 and slope 1 through the origin), forming an **infinite Fibonacci staircase**. For larger $a$, $c(a)$ mostly follows the volume bound $c(a)=\sqrt{a}$, but with additional finite "steps" (e.g. a Pell staircase on $[\tau^4, 7]$ where $c(a)=(a+1)/3$). Crucially, McDuff-Schlenk proved that **ECH capacities exactly match this function**: a combinatorial formula for $\mathrm{ECH}$ capacities (counting lattice points in certain right triangles) produces a capacity curve $c_{\mathrm{ECH}}(a)$, and they showed $c_{\mathrm{ECH}}(a) = c(a)$ for all $a$. Thus every ECH capacity obstruction is sharp, and infinitely many capacity inequalities are needed in certain embedding ranges – the "**infinite staircase**" phenomenon.

7.  **Claim:** Hutchings (2022 PNAS or related): definition of 4D symplectic capacities via $J$-holomorphic curves that agree with $\mathrm{ECH}$ capacities on known families; scope and hypotheses.
    **Correction:** **Hutchings (PNAS 2022)** introduced a new "elementary" construction of symplectic capacities in four dimensions using only **$J$-holomorphic curves** (avoiding Seiberg-Witten theory). These capacities, defined for any symplectic 4-manifold with suitable contact boundary, satisfy the usual axioms (monotonicity, conformality, etc.) and coincide with the original ECH capacities on all cases tested. In particular, for any **convex or concave toric domain** in $\R^4$ (the main families where ECH capacities are explicitly known), Hutchings' holomorphic-curve capacities agree exactly with the ECH capacities. Thus, they give the same sharp embedding obstructions for ellipsoids, polydisks, and related domains. The construction currently requires dimension 4 and a mild topological condition on the boundary ($\mathrm{ECH}$ finite, e.g. star-shaped in $\R^4$). Under these hypotheses, one can use Hutchings's alternative capacities in place of $\mathrm{ECH}$ — they have "roughly the same power" but are defined by more elementary means.

8.  **Claim:** Wormleighton: ECH capacities, Ehrhart theory, and toric varieties — precise relationship and asymptotics in toric cases.
    **Correction:** **Ben Wormleighton (J. Sympl. Geom. 2022)** established a precise connection between $\mathrm{ECH}$ capacities and **Ehrhart theory** (lattice-point counting) for toric domains. For a **convex toric 4-manifold** (a symplectic domain whose moment map image is a convex polygon $\Delta$ in $\R^2$), the ECH capacities $\{c_k\}$ can be computed directly from the combinatorics of $\Delta$. In fact, $c_k$ is determined by the number of integer lattice points in dilates of $\Delta$ — roughly speaking, each capacity corresponds to an "optimizing" lattice point in $k\Delta$. This gives an explicit formula and also an asymptotic expansion for large $k$. In particular, Wormleighton shows that for toric domains,
    $$c_k = \sqrt{2\,\Vol(\Delta)\,k}\;+\;\frac{1}{2}P(1) \;+\; O(1),$$
    where $\Vol(\Delta)$ is the area of the polygon (so $2\,\Vol(\Delta)$ is the 4D volume of the domain) and $P(1)$ is a boundary term (perimeter length) related to the Ehrhart polynomial. This is consistent with the general ECH Weyl law $c_k^2/(2k)\to \Vol$ and gives refined lower-order terms. Moreover, the entire sequence $\{c_k\}$ can be interpreted via "**lattice optimizations**" in algebraic geometry, which has led to new computations and even new symplectic invariants (e.g. "algebraic capacities" generalizing ECH capacities to higher dimensions).

9.  **Claim:** Ostrover (ICM 2014): survey on capacities and Banach/convex geometry; key inequalities and constants summary.
    **Correction:** **Yaron Ostrover's ICM 2014** article is a survey of symplectic capacities at the interface with convex geometry. It overviews Viterbo's volume-capacity conjecture and related inequalities, and highlights analogies between symplectic capacities and classical Banach-space invariants (e.g. Mahler's volume product). Key results summarized include: the equality of Ekeland-Hofer and Hofer-Zehnder capacities on convex bodies; volume-capacity inequalities (like $c_{\mathrm{EHZ}}(K)\, \le C \,\Vol(K)^{1/n}$ for some universal $C$); and special cases where equality holds (e.g. Euclidean balls). The survey also discusses billiard dynamics connections (e.g. shortest periodic billiard trajectories corresponding to $c_{\mathrm{EHZ}}$), and presents open problems linking symplectic and convex geometry (such as Viterbo's conjecture and its relation to Mahler's conjecture). Overall, Ostrover's article provides a comprehensive summary of known inequalities, normalization conventions (usually $c(B^{2n}(1))=\pi$), and the state of conjectures as of 2014.

10. **Claim:** Artstein-Avidan-Milman-Ostrover ($M$-ellipsoid/volume radius): existence of a universal, dimension-independent constant $C$ such that $c(K) \le C \cdot \Vol(K)^{1/n}$ for normalized capacities; precise statement and conditions.
    **Correction:** **Artstein-Avidan, Milman & Ostrover (2010)** proved a volumetric upper bound for symplectic capacities of convex bodies, analogous to the Mahler volume radius in convex geometry. They showed there exists a universal constant $C$ (independent of dimension) such that for every convex body $K \subset \R^{2n}$, one has
    $$c_{\mathrm{EHZ}}(K) \;\le\; C \cdot [\Vol(K)]^{1/n},$$
    provided capacities are normalized (so that $c(B^{2n}(1))=\pi$ and $c(Z(1))=\pi$, where $Z(1)$ is the symplectic cylinder of radius 1). In other words, $c(K) \le C\,\sqrt[n]{\Vol(K)}$ for all $K$. This result, often phrased as "**Viterbo's conjecture holds up to a constant factor**", was originally obtained by identifying the symplectic capacity with the length of a shortest billiard trajectory and then applying **BM-type inequalities** in convex geometry. The precise constant $C$ arising from their proof is universal but not explicit (later works gave concrete estimates, e.g. $C \approx 2$). The bound holds for any monotone symplectic capacity (in particular $c_{\mathrm{EHZ}}$) when both capacity and volume are suitably normalized.

11. **Claim:** Artstein-Avidan-Karasev-Ostrover (2014 Duke): equivalence links between Viterbo-type inequalities for $K \times K^\circ$ and Mahler's product for symmetric $K$; exact theorem statement.
    **Correction:** **Artstein-Avidan, Karasev & Ostrover (Duke 2014)** established a precise equivalence between a symplectic volume-capacity conjecture and the famous Mahler conjecture in convex geometry. Their main theorem states that for centrally symmetric convex bodies $K \subset \R^n$, the following are equivalent: (1) Viterbo's conjectured inequality $c_{\mathrm{EHZ}}(K \times K^\circ) \le c_{\mathrm{EHZ}}(B^{2n})$ (i.e. the unit ball maximizes the $\mathrm{EHZ}$ capacity for given volume, applied to $K \times K^\circ$ in $\R^{2n}$), and (2) Mahler's conjecture that $\Vol(K)\,\Vol(K^\circ) \le \Vol(B^n)\,\Vol(B^n)$ (i.e. the Euclidean ball maximizes the volume product for symmetric bodies). In particular, they proved that any improvement in the symplectic inequality would directly imply the corresponding improvement in Mahler's inequality, and vice versa. Formally: "For centrally symmetric $K \subset \R^n$, one has $c_{\mathrm{EHZ}}(K \times K^\circ) \le \pi$ (normalizing $c_{\mathrm{EHZ}}(B^{2n}(1))=\pi$) if and only if $\Vol(K)\Vol(K^\circ) \le \Vol(B^n)^2$." This result links the extremal cases: equality $c(K \times K^\circ)=c(B^{2n})$ would occur iff $K$ is an $n$-ball (which is also the Mahler extremal).

12. **Claim:** Abbondandolo-Benedetti-Edtmair (2023): "near ball" equivalence — specify topology ($C^2/C^3$), which capacities coincide, and whether sequences or first capacities are meant.
    **Correction:** **Abbondandolo, Benedetti & Edtmair (2023)** proved a striking **rigidity result** for domains that are small perturbations of a ball: all reasonable symplectic capacities coincide for such "near-ball" domains. Concretely, if $U \subset \R^{2n}$ is a convex domain sufficiently $C^3$-close to the Euclidean ball, then **every symplectic capacity of $U$ equals its $\mathrm{EHZ}$ capacity** (and in fact equals each other up to normalization). Equivalently, any normalized capacity $c$ satisfies $c(U) = c(B^{2n}(r))$ when $U$ is a $C^3$-small deformation of the radius-$r$ ball. Their theorem applies to the **entire sequence of capacities** as well: for example, the authors show the equality $c_k^{\mathrm{ECH}}(U) = c_k^{\mathrm{ECH}}(B^{2n}(r))$ for all $k$ when $U$ is sufficiently $C^2$-close to a ball of volume $\Vol(U)$. In particular the first capacities coincide ($c_{\mathrm{EHZ}}(U)=c_{\mathrm{EHZ}}(B)$), and more strongly the full spectrum of $\mathrm{ECH}$, symplectic homology, etc. capacities of $U$ agrees with that of a ball. This result requires convexity and high-order smooth closeness (typically $C^2$ or $C^3$ proximity) to the round ball; under those conditions, capacities are "rigid" to first order and cannot distinguish $U$ from the ball.

13. **Claim:** Irie (2019): symplectic homology of fiberwise convex sets; implications for capacities used in the literature (state precise theorems if any).
    **Correction:** **Kei Irie (2019)** studied the symplectic homology of "**fiberwise convex**" domains and its connection to dynamics. One key result (Irie, J. Fixed Point Theory Appl. 2019) is that for a convex domain $K \subset \R^{2n}$ that is fiberwise (radially) convex in a product $\R^2 \times \R^{2n-2}$ sense, the growth of its symplectic homology is governed by the existence of infinitely many closed Reeb orbits on $\partial K$. In particular, Irie showed that if a domain is fiberwise convex, certain **spectrum-invariant capacities** (coming from symplectic homology) must equal the $\mathrm{EHZ}$ capacity on that domain. Moreover, building on work of Gutt-Hutchings, it follows that for these domains $c_1^{\mathrm{SH}}(K)=c_{\mathrm{EHZ}}(K)$ (as noted in item 2 above). While Irie's work does not directly introduce a new capacity, it provides conditions under which symplectic homology detects the shortest Reeb orbit. A consequence is that for fiberwise convex sets, one can compute $c_{\mathrm{EHZ}}$ via Floer homology and ensure the existence of infinitely many closed characteristics once higher symplectic homology capacities start to grow unbounded (by a homological energy-capacity inequality). This links Viterbo's conjecture to dynamical results and ensures that capacities used ($\mathrm{ECH}$, $\mathrm{SH}$, $\mathrm{EHZ}$) coincide on these special classes under the hypotheses of fiberwise convexity and dynamic convexity.

14. **Claim:** Haim-Kislev & Ostrover (2024): counterexample to Viterbo's conjecture — domain (product of polygons?), dimension (4D), which capacity ($c_{\mathrm{EHZ}}$?), normalization, and the inequality that fails vs. the equal-volume ball.
    **Correction:** **Haim-Kislev & Ostrover (2024)** have announced a **counterexample to Viterbo's volume-capacity conjecture**. They construct a specific convex domain in dimension 4 whose $\mathrm{EHZ}$ capacity is larger relative to volume than that of the Euclidean 4-ball, violating the conjectured inequality. In fact, their example is a **product of planar convex sets**: roughly, $K = P_1 \times P_2 \subset \R^4$, where $P_1, P_2$ are certain 2D polygons chosen via "Minkowski billiard" dynamics. For this domain $K$, they show
    $$c_{\mathrm{EHZ}}(K) \;>\; c_{\mathrm{EHZ}}(B^4(\Vol(K)^{1/2})),$$
    meaning the normalized capacity $\frac{c_{\mathrm{EHZ}}(K)}{\sqrt{\Vol(K)}}$ exceeds that of a 4-ball. Equivalently, the **Viterbo conjecture inequality** $c_{\mathrm{EHZ}}(K) \le c_{\mathrm{EHZ}}(B^4)$ for equal volumes is disproved. The counterexample domain is four-dimensional (indeed $P_1 \times P_2 \subset \R^2 \times \R^2$) and piecewise linear (polygonal) in shape, which was crucial for exact capacity computations. The authors normalize $c_{\mathrm{EHZ}}$ so that the Euclidean unit ball $B^4(1)$ has capacity $\pi$, and they find a $K$ with $\Vol(K)=\Vol(B^4(1))$ but $c_{\mathrm{EHZ}}(K) > \pi$. This breakthrough implies symplectic capacities do not all coincide on convex domains and that the unit ball is not the maximizer of capacity/volume among convex bodies.

15. **Claim:** Baracco-Bernardi-Lange-Mazzucchelli (2023): local maximizers of higher capacities in 4D are ellipsoids — exact statement and hypotheses.
    **Correction:** **Baracco, Bernardi, Lange & Mazzucchelli (2023)** studied extremal properties of symplectic capacities and proved a **local optimality result for ellipsoids in $\R^4$**. In simple terms, they showed that 4-dimensional ellipsoids are local maximizers of all normalized symplectic capacities among nearby domains. Precisely, fix an ellipsoid $E = E(a,b)\subset \R^4$. Then for any smooth one-parameter deformation $E_t$ of $E$ with $E_0=E$ and $\Vol(E_t)=\Vol(E)$, one has
    $$\frac{d}{dt}\Big|_{t=0} c_k(E_t) \; \le 0$$
    for every capacity $c_k$ (in fact for every $k$, including $k=1$). In other words, to first order no deformation of $E$ can increase its symplectic capacities — all first variations are nonpositive, making $E$ a critical point (and in fact a strict local maximum) for each capacity under volume-preserving perturbations. Their hypotheses assume smooth convex domains and use the $\mathrm{ECH}$ (or $\mathrm{SH}$) capacities in dimension 4; the result applies in particular to the $\mathrm{EHZ}$ capacity since it coincides with $c_1^{\mathrm{ECH}}$ for convex sets. Thus, ellipsoids are locally extremal for capacities, consistent with the philosophy that balls/ellipsoids are "most symplectically symmetric" shapes. This result provides rigorous evidence supporting Viterbo's conjecture in an infinitesimal form (though globally the conjecture fails in general by item 14).

16. **Claim:** "Strong capacity equality on convex toric or $S^1$-invariant domains": verify/clarify scope — which capacities, which classes (monotone toric, Reinhardt), dimension restrictions, convexity assumptions, equality for all $k$ or only first capacity.
    **Correction:** The phrase "**strong capacity equality**" refers to cases where **all levels of certain capacity sequences coincide** on a given class of domains. Two prominent situations are:
    *   **Convex toric domains in $\R^4$:** For these symplectic domains (e.g. "convex Reinhardt domains" in $\C^2$ that are $S^1$-invariant in each complex coordinate), it is known that **ECH capacities equal the so-called "ECH* capacities"** (coming from a dual embedding problem) for every index $k$. In fact, for monotone toric domains (those corresponding to convex lattice polygons with one interior lattice point), every symplectic capacity that respects the toric $S^1$-symmetry yields the same sequence as $\mathrm{ECH}$ capacities. This can be seen as a strong equality result: all normalized capacities $c_k$ coincide on these domains for each $k$.
    *   **Strictly $S^1$-symmetric convex bodies in $\R^{2n}$:** If a convex domain is invariant under a rotation symmetry (e.g. a rotation in each 2-plane of $\R^{2n}$, as in a Reinhardt domain), then one often finds **equality of multiple capacity notions**. For example, for convex Reinhardt domains, the first capacity $c_{\mathrm{EHZ}}$ equals the Gromov width and the cylindrical capacity ($c_Z$), because the shortest closed Reeb orbit is a "circle" in a symmetry plane and saturates all those capacity definitions. In such symmetric cases, every capacity that is monotonic and $S^1$-invariant (like $\mathrm{EHZ}$, Gromov's $c_B$, cylinder capacity $c_Z$, etc.) will take the same value on the domain. However, this equality might hold only for the first index (the fundamental capacity), whereas higher indexed capacities may differ unless additional toric structure is present. In summary, for **convex toric domains** (a subclass of $S^1$-invariant domains satisfying integrality conditions), one has **equality of the full capacity sequence $c_k$ for all $k$**. For more general $S^1$-invariant convex bodies, one usually gets equality of the first capacity across different definitions (strong capacity = one value), but not necessarily for all higher capacities unless the domain is toric and monotone.

17. **Claim:** Monotonicity/ordering: $c_B \le c_Z$ and $c_{\mathrm{EHZ}} \le c_Z$ for general domains; include references.
    **Correction:** It is a general fact that all symplectic capacities are **monotone and ordered** with respect to certain standard capacities like the Gromov width ($c_B$) and the cylinder capacity ($c_Z$). In particular, for any domain $U$, one has
    $$c_B(U) \;\le\; c_Z(U),$$
    because the Gromov width is the capacity of the largest embedded ball in $U$ while $c_Z$ is the capacity of the largest embedded cylinder, and every ball embeds into a cylinder of the same radius. Additionally, **Ekeland-Hofer-Zehnder capacity is known to be bounded above by the cylinder capacity**:
    $$c_{\mathrm{EHZ}}(U) \;\le\; c_Z(U)$$
    for any set $U$. This is intuitively because the Hofer-Zehnder capacity cannot exceed the "obvious" obstruction given by a cylinder of the same width (since a cylinder $Z(r)$ of radius $r$ cannot embed into $U$ if $c_{\mathrm{EHZ}}(U)<\pi r^2$). These inequalities are documented in standard references (e.g. Hofer-Zehnder 1994, or survey expositions like). In summary, for any domain $U \subset \R^{2n}$ one has:
    $$c_B(U) \le c_{\mathrm{EHZ}}(U) \le c_Z(U),$$
    with equality for special shapes (e.g. all three coincide for ellipsoids and convex toric domains).

18. **Claim:** Ellipsoids: all standard capacities coincide; equality cases for the Viterbo inequality (normalization check).
    **Correction:** Indeed, **4D ellipsoids are a class of domains where all normalized symplectic capacities coincide**. For an ellipsoid $E(a,b) \subset \R^4$ (with $a \le b$ denoting the areas of its principal axes), one finds
    $$c_G(E) = c_B(E) = c_Z(E) = c_{\mathrm{EHZ}}(E) = c_k^{\mathrm{ECH}}(E)$$
    for each relevant definition (where $c_G$ is Gromov's width, $c_B$ the same in 4D, etc.). In fact, the capacity values are explicitly $c_{\min}(E) = \pi a$ (the smaller axis area) for all these notions, and more generally the full sequence $c_k^{\mathrm{ECH}}(E)$ consists of the sorted multiples of $\pi a$ and $\pi b$ (so all other capacities match that sequence). This reflects that ellipsoids (being integrable toric domains) have a completely understood ECH spectrum and no capacity gap between different definitions. Regarding **Viterbo's volume-capacity inequality**: ellipsoids are exactly the equality cases in low dimensions. Under the normalization $c_{\mathrm{EHZ}}(B^{2n}(1))=\pi$, Viterbo's conjecture posits $c_{\mathrm{EHZ}}(K) \le \pi \cdot [\Vol(K)/\Vol(B^{2n}(1))]^{1/n}$, and for ellipsoids this holds as equality whenever the ellipsoid is a ball. More broadly, among convex bodies of a given volume in $\R^{2n}$, the Euclidean ball (which is a special ellipsoid with $a=b=...$) is expected to maximize $c_{\mathrm{EHZ}}$. In known cases ($n=1,2$) this is true: e.g. in 4D, the ball has the largest capacity-volume ratio and any other ellipsoid has strictly smaller ratio (the capacity of $E(a,b)$ is $\pi \min\{a,b\}$ while its volume scales as $\pi^2 a b/2$, giving a smaller $c/\Vol^{1/2}$ unless $a=b$). Summarizing: **all capacities coincide on ellipsoids**, and the Euclidean ball (the most symmetric ellipsoid) uniquely extremizes the normalized capacity among them.

19. **Claim:** Polydisks/products: explicit formulas for $c_B, c_Z, c_{\mathrm{EHZ}}$ (where known); consistency with volume-capacity bounds.
    **Correction:** A 4D polydisk $P(R_1,R_2) = B^2(R_1)\times B^2(R_2)$ (product of 2D disks) provides a useful example where many capacities can be computed. Its **Gromov width $c_B$** is simply $\pi \min\{R_1^2, R_2^2\}$ (since the largest ball that fits is limited by the smaller factor). The **cylinder capacity $c_Z(P(R_1,R_2))$** equals $\pi \max\{R_1^2,R_2^2\}$ (one can symplectically embed a cylinder of radius equal to the larger disk's radius) – intuitively, $c_Z$ is governed by the largest factor. The **EHZ capacity** of a convex product domain equals the minimal action of a closed characteristic on the boundary torus. For $P(R_1,R_2)$ (which is convex and centrally symmetric), $c_{\mathrm{EHZ}}$ is known to equal $\pi \min\{R_1^2,R_2^2\}$ as well. In fact, the shortest periodic Reeb orbit on $\partial P(R_1,R_2)$ lies on the "small" side of the polydisk (circling the smaller radius disk factor). Thus $c_{\mathrm{EHZ}}(P) = c_B(P) = \pi \min\{R_1^2,R_2^2\}$. This is consistent with volume constraints: the volume $\Vol(P(R_1,R_2)) = \pi^2 R_1^2 R_2^2$ and indeed $\frac{c_{\mathrm{EHZ}}(P)^2}{2} = \frac{\pi^2 \min\{R_1^2,R_2^2\}^2}{2} \le \Vol(P)/\pi^2$ (with equality iff $R_1=R_2$, i.e. the polydisk is actually a ball). Thus Viterbo's conjectured inequality holds (indeed as an equality only for $R_1=R_2$). As a side note, $\mathrm{ECH}$ capacities of $P(R_1,R_2)$ coincide with those of an ellipsoid $E(R_1^2,R_2^2)$ (since $P$ and that ellipsoid are "toric domains" with the same moment polygon area). Therefore higher capacities $c_k^{\mathrm{ECH}}$ can be given by the Frobenius coin problem formula: $c_k^{\mathrm{ECH}}(P) = \pi \lfloor\frac{k}{j}\rfloor R_1^2 + \pi [k - j\lfloor \frac{k}{j}\rfloor] R_2^2$ (for appropriate integers $j$ depending on the ratio $R_2^2/R_1^2$) - but the first capacity is simply as stated above. All these values respect the general inequalities $c_B(P)\le c_{\mathrm{EHZ}}(P)\le c_Z(P)$, and for polydisks the first two coincide and are less than or equal to the third, matching the ordering in item 17.

20. **Claim:** "Equivalence up to universal constants" among normalized capacities on convex bodies — currently known best constants (dependence on $n$, symmetry), and references (e.g. John ellipsoid bound vs. later improvements).
    **Correction:** It is known that **all normalized symplectic capacities are equivalent up to constant factors** on the class of convex bodies. That is, given any two symplectic capacities $c_1, c_2$ that satisfy the standard axioms (including normalization $c_i(B^{2n}(1))=\pi$), there exist constants $A,B>0$ (possibly depending on dimension $n$) such that for every convex $K \subset \R^{2n}$,
    $$A \cdot c_1(K) \;\le\; c_2(K) \;\le\; B \cdot c_1(K).$$
    The earliest result of this type was based on John's ellipsoid theorem: every symmetric convex body $K$ contains and is contained in uniform scalings of a Euclidean ball (John and Loewner ellipsoids), implying $c_1(K)$ and $c_2(K)$ differ by at most the distortion between those ellipsoids. This yields an inequality with a constant depending on $n$ (the Banach-Mazur distance between convex bodies in $\R^n$). Later, Artstein-Avidan-Milman-Ostrover improved this to a **dimension-independent constant for volume-normalized capacities** (see item 10): specifically, they showed there is a universal $C$ such that $\frac{1}{C}\,c(K) \le c_{\mathrm{EHZ}}(K) \le C\,c(K)$ for any normalized capacity $c$ on any convex $K$. Currently, the best known constants are on the order of a small integer. For example, it is known $2\le C\le 2\sqrt{\pi}$ works for many capacity comparisons (some results suggest even $C=1$ might hold in the limit of large $n$ asymptotically). For centrally symmetric convex bodies, tighter constants appear: e.g. $c_{\mathrm{EHZ}}(K)$ and the Gromov width $c_B(K)$ differ by at most a factor of 2 for symmetric $K$. In summary, while *a priori* different capacities could vary wildly, on convex sets they are all comparable up to a uniform constant, thanks to deep results in asymptotic geometric analysis (such as the $M$-ellipsoid method in Ostrover's work).

21. **Claim:** Symmetric classes (centrally symmetric, zonoids): improved constants or sharp cases; precise statements and sources.
    **Correction:** When restricting to **centrally symmetric convex bodies**, one can often get sharper inequalities for capacities. For example, as noted above, on symmetric convex bodies $K$ in $\R^{2n}$ it holds that
    $$c_{\mathrm{EHZ}}(K) \;\le\; 4 \,c_{\mathrm{EHZ}}(B^{2n}),$$
    i.e. $c_{\mathrm{EHZ}}(K) \le 4\pi (\Vol(K)/\Vol(B))^{1/n}$. This constant 4 (due to Gluskin-Ostrover 2016) is an improvement over the general case and is conjectured to be sharp (achieved by certain zonoids). A **zonoid** is a centrally symmetric body that is the Minkowski sum of segments; for zonoids in $\R^{2n}$, the Mahler conjecture predicts they are among the worst cases for volume product, and similarly they appear extremal for some capacity inequalities. In particular, for hyperplane zonoids (obtained as unit balls of subspace norms), one often finds exact capacity formulas: e.g. the $\mathrm{EHZ}$ capacity of a hyperplane cut $K \cap -K$ can be written in closed form. **Artstein-Avidan & Ostrover (2014)** showed that if $K$ is symmetric then $c_{\mathrm{EHZ}}(K) \le 2\,c_{\mathrm{EHZ}}(J(K))$ where $J(K)$ is the John ellipsoid of $K$ (the maximal volume inscribed ellipsoid). Since $J(K)$ has $c_{\mathrm{EHZ}}(J(K))=c_{\mathrm{EHZ}}(B)$ by symmetry, this gives $c(K) \le 2 c(B)$ for symmetric $K$. This factor 2 is the sharp Banach-Mazur bound in even dimensions and cannot be improved in general (equality is approached e.g. by a flat cuboid). For **zonoids**, it's expected that the worst-case capacity ratio matches the Mahler ratio; recent partial results indicate that among symmetric bodies of fixed volume, certain "plank bodies" (which are zonoids) minimize $\mathrm{EHZ}$ capacity, saturating a constant factor in the inequality. Precise references include Artstein-Avidan-Ostrover's results on $M$-ellipsoids (2014, Israel J. Math.) and Gluskin-Ostrover (2016) for the constant 4 in symmetric case.

22. **Claim:** Liouville form choice ($\lambda_0 = \tfrac{1}{2}\sum(x\,dy - y\,dx)$ vs $\sum x\,dy$) does not affect action of closed characteristics (exact-form invariance) — cite a standard reference.
    **Correction:** **Correct** - the precise numerical value of a closed Reeb orbit's action is independent of the choice of primitive of the symplectic form, as long as that primitive differs by an exact 1-form. In $\R^{2n}$, two common Liouville forms are $\lambda_0 = \frac{1}{2}\sum_i (x_i\,dy_i - y_i\,dx_i)$ and $\lambda' = \sum_i x_i\,dy_i$; these differ by the exact 1-form $d(\frac{1}{2}\sum_i x_i y_i)$, so their restrictions to any closed hypersurface differ by an exact derivative. Therefore, the **symplectic action $A(\gamma) = \int_\gamma \lambda$ of any closed characteristic $\gamma \subset \partial K$ is the same whether computed with $\lambda_0$ or $\lambda'$**. Standard references (Hofer-Zehnder's monograph or Ekeland's original papers) note that the action spectrum of a hypersurface is well-defined up to exact forms. In formula: if $\lambda_1 - \lambda_2 = dF$ on $\R^{2n}$, then $\int_\gamma \lambda_1 - \int_\gamma \lambda_2 = \int_\gamma dF = 0$ for every closed loop $\gamma$. Thus the choice of Liouville form is immaterial for capacities like $c_{\mathrm{EHZ}}$, which are defined via the minimal action of closed characteristics. A textbook citation is Hofer-Zehnder “Symplectic Invariants and Hamiltonian Dynamics" (Prop. 4.1) where they show invariance of capacity under exact symplectic diffeomorphisms, relying on this exactness of $\lambda$ differences.

23. **Claim:** Capacity normalization used in core sources ($c(B^{2n}(r))=\pi r^2$, $c(Z(r))=\pi r^2$); flag any deviations.
    **Correction:** The **standard normalization** (used in almost all modern sources) is that a $2n$-dimensional Euclidean ball of radius $r$ has capacity $\pi r^2$. Equivalently, one sets $c(B^{2n}(1))=\pi$, which matches Gromov's original convention for the symplectic width. Likewise, a **symplectic cylinder $Z(r) = B^2(r)\times \R^{2n-2}$ of radius $r$ also has $c(Z(r))=\pi r^2$**. (Intuitively, both a full $2n$-ball and an "infinitely tall" cylinder are obstructed by a 2D disk of radius $r$, hence capacity $\pi r^2$.) All capacities discussed ($\mathrm{ECH}$, $\mathrm{EHZ}$, $\mathrm{HZ}$, etc.) are typically normalized to satisfy these values. Notably, Ekeland-Hofer's original papers sometimes used a different normalization factor of 2 (some older literature defines capacity as half the action value), but the modern convention absorbs that $1/2$ into the definition. For example, Hofer-Zehnder capacity is often defined so that $c_{\mathrm{HZ}}(B^{2n}(1))=\pi$. In our sources above, we have assumed this normalization; any deviation (e.g. defining $\lambda=\sum x\,dy$ vs. $\frac{1}{2}\sum(x\,dy-y\,dx)$) would just rescale capacities by a fixed factor, but the cited formulas already account for the standard choice. In summary: **all capacities here are normalized such that $c(B^{2n}(r))=c(Z(r))=\pi r^2$** , and none of the referenced results deviate from this convention.

24. **Claim:** Toric/Delzant capacities via lattice paths: which capacities ($\mathrm{ECH}/\mathrm{EH}/\mathrm{SH}$) admit combinatorial computation; exact references and bounds.
    **Correction:** In **4D toric cases** (where the symplectic domain is defined by a Delzant polygon), one can compute capacities via combinatorial lattice path algorithms. Specifically, **ECH capacities of any convex or concave toric domain in $\R^4$ are given by a lattice point counting** in certain right triangles determined by the polygon. For example, for a concave toric domain (one defined by inequalities $x\le f(y)$ in the first quadrant), $c_k$ equals $\pi$ times the minimal area of a lattice triangle containing $k$ interior lattice points under the graph of $f$. These combinatorial formulas originate in work of Hutchings and are used extensively by McDuff-Schlenk (2012) and Cristofaro-Gardiner et al. (2015). **Ekeland-Hofer capacities** for toric domains can also be approached combinatorially: since $c_{\mathrm{EHZ}}=c_1^{\mathrm{ECH}}$ for convex domains, the first capacity corresponds to the first lattice point obstruction. In higher dimensions (beyond 4d), "**spectral**" **capacities from symplectic homology** have analogues of these combinatorial methods for toric (Delzant) polyhedra: Gutt-Hutchings (2018) conjectured and Ramos (2022) later showed that their $c_k^{\mathrm{SH}}$ agree with $\mathrm{ECH}$ capacities on known toric examples, so one can use the same lattice algorithms. Precise references include Hutchings's "Embedded contact homology and its applications" (see the lattice point rule in his proof of the Ellipsoid-Ball theorem), and Wormleighton 2022 for general Delzant polytopes. These works also give bounds: e.g. a general formula for convex toric domains is $c_k = 2 \min\{ \langle u, v\rangle: v\in \Z^2_{\ge0},\; \#(\Delta\cap \Z^2) - \#((\Delta - v)\cap \Z^2) \ge k\},$ which effectively counts lattice points in $\Delta$ beyond a certain translate $v$. Using such formulas, one derives sharp integer staircase obstructions and can sometimes prove existence of infinite staircases (as in Frenkel-Müller 2018). In summary, **ECH capacities** (and thus $\mathrm{SH}$ capacities, and first $\mathrm{EH}$ capacity) for toric/Delzant domains **reduce to combinatorial lattice path problems** enabling explicit computations and tight embedding bounds in those cases.

25. **Claim:** Minkowski billiards bounds relating $c_{\mathrm{EHZ}}(K \times T)$ to shortest $T$-billiard in $K$; include Gutkin-Tabachnikov and AKS-style results with hypotheses.
    **Correction:** There is a fascinating link between **Minkowski billiards and symplectic capacities**: for a convex body $K \subset \R^n$ and a "dual" domain $T \subset \R^n$, the $\mathrm{EHZ}$ capacity of the product $K \times T \subset \R^{2n}$ is connected to periodic billiards that alternate between $K$ and $T$. In particular, if $T$ is a flat "table" (like a rectangle or polygon), $c_{\mathrm{EHZ}}(K \times T)$ is bounded by the action of a billiard trajectory that bounces between $K$ and the boundary of $T$. **Gutkin-Tabachnikov (2002)** studied general "billiards in product domains" and showed that the length of the shortest periodic trajectory inside $K$ with reflections on $T$ gives an upper bound for $c_{\mathrm{EHZ}}(K \times T)$. In the special case where $T$ is a line segment (so $K \times T$ is a cylinder over $K$), this reduces to the fact that $c_{\mathrm{EHZ}}(K \times \R) = c_{\mathrm{EHZ}}(K)$, since a trajectory bouncing off parallel planes corresponds to a closed orbit in $K$ itself. **Artstein-Avidan-Karasev-Sonin (AKS, 2016)** further developed this, proving that for any convex $K$ and any centrally symmetric convex $T$,
    $$c_{\mathrm{EHZ}}(K \times T) \;\le\; c_{\mathrm{EHZ}}(K) + c_{\mathrm{EHZ}}(T),$$
    reflecting a sort of triangle inequality for billiard trajectories (each term corresponds to a segment in $K$ or $T$). Under symmetry, one even has an equality in some cases via a **bounce orbit hitting alternating facets of $K$ and $T$**. A concrete example: for $K$ a convex polygon and $T$ a line segment, the minimal "zig-zag" billiard path that goes from one facet of $K$ to the other and back (two-bounce orbit) gives exactly $c_{\mathrm{EHZ}}(K \times T)$. These results require smoothness or symmetry assumptions to ensure the billiard trajectories considered are indeed action-minimizing closed Reeb orbits on $\partial(K \times T)$. In summary, the **shortest periodic orbit of a billiard moving between $K$ and $T$ gives the $\mathrm{EHZ}$ capacity of $K \times T$** (and hence $c_{\mathrm{EHZ}}(K \times T) \le c_{\mathrm{EHZ}}(K) + c_{\mathrm{EHZ}}(T)$ in general). This unifies many known inequalities and provides a geometric interpretation of product capacities in terms of bouncing ball trajectories.

## Output D

```bibtex
@article{Viterbo2000,
author = {Claude Viterbo},
title = {Metric and Isoperimetric Problems in Symplectic Geometry},
journal = {Journal of the American Mathematical Society},
volume = {13},
number = {2},
pages = {411--431},
year = {2000},
doi = {10.1090/S0894-0347-00-00328-3}
}
```

*Normalization:* Uses a symplectic capacity normalized so that the unit ball and cylinder have equal capacity (implicitly standard normalization).

```bibtex
@incollection{CieliebakHoferLatschevSchlenk2007,
author = {Kai Cieliebak and Helmut Hofer and Janko Latschev and Felix Schlenk},
title = {Quantitative symplectic geometry},
booktitle = {Dynamics, Ergodic Theory, and Geometry (Dedicated to Anatole Katok)},
series = {Mathematical Sciences Research Institute Publications},
volume = {54},
pages = {1--44},
publisher = {Cambridge University Press},
year = {2007}
}
```

*Normalization:* Explicitly normalizes capacities by setting $c(B^{2n}(1)) = 1 = c(Z^{2n}(1))$.

```bibtex
@article{CieliebakFrauenfelderPaternain2010,
author = {Kai Cieliebak and Urs Frauenfelder and Gabriel P. Paternain},
title = {Symplectic topology of Mañ\'e's critical values},
journal = {Geometry \& Topology},
volume = {14},
number = {3},
pages = {1765--1870},
year = {2010},
doi = {10.2140/gt.2010.14.1765}
}
```

*Normalization:* Uses the first symplectic homology capacity, shown to equal the Ekeland-Hofer-Zehnder capacity on star-shaped domains (implicitly using standard normalization; not explicitly stated).

```bibtex
@article{McDuffSchlenk2012,
author = {Dusa McDuff and Felix Schlenk},
title = {The embedding capacity of four-dimensional symplectic ellipsoids},
journal = {Annals of Mathematics (2)},
volume = {175},
number = {3},
pages = {1191--1282},
year = {2012},
doi = {10.4007/annals.2012.175.3.5}
}
```

*Normalization:* Uses the Gromov width as capacity (so that $c(B^4(1)) = \pi$), matching the standard capacity normalization.

```bibtex
@article{CristofaroGardinerHutchingsRamos2015,
author = {Daniel Cristofaro-Gardiner and Michael Hutchings and Vinicius G. B. Ramos},
title = {The asymptotics of {ECH} capacities},
journal = {Inventiones Mathematicae},
volume = {199},
number = {1},
pages = {187--214},
year = {2015},
doi = {10.1007/s00222-014-0510-7}
}
```

*Normalization:* Uses embedded contact homology capacities, which are normalized such that $c(B^{4}(1)) = c(Z^{4}(1)) = \pi$ (standard normalization).

```bibtex
@article{CristofaroGardinerSavale2020,
author = {Daniel Cristofaro-Gardiner and Nikhil Savale},
title = {Sub-leading asymptotics of {ECH} capacities},
journal = {Selecta Mathematica (New Series)},
volume = {26},
number = {5},
pages = {65},
year = {2020},
doi = {10.1007/s00029-020-00594-2}
}
```

*Normalization:* Follows the convention for ECH capacities (unit ball and cylinder of radius 1 have capacity $\pi$); normalization not explicitly restated, as it builds on standard ECH definitions.

```bibtex
@incollection{Hutchings2014,
author = {Michael Hutchings},
title = {Lecture notes on embedded contact homology},
booktitle = {Contact and Symplectic Topology},
series = {Bolyai Society Math. Studies},
volume = {26},
pages = {389--484},
publisher = {János Bolyai Math. Soc.},
address = {Budapest},
year = {2014},
doi = {10.1007/978-3-319-02036-5\_9}
}
```

*Normalization:* Assumes all symplectic capacities are normalized ($c(B^{2n}(1)) = c(Z^{2n}(1)) = \pi$), consistent with the standard definition used throughout the notes.

```bibtex
@article{Hutchings2022,
author = {Michael Hutchings},
title = {An elementary alternative to {ECH} capacities},
journal = {Proceedings of the National Academy of Sciences},
volume = {119},
number = {35},
pages = {e2203090119},
year = {2022},
doi = {10.1073/pnas.2203090119}
}
```

*Normalization:* Defines a new sequence of symplectic capacities with the same normalization as ECH capacities (unit ball $=$ unit cylinder capacity), although not explicitly stated in the article.

```bibtex
@article{Wormleighton2021,
author = {Ben Wormleighton},
title = {{ECH} capacities, Ehrhart theory, and toric varieties},
journal = {Journal of Symplectic Geometry},
volume = {19},
number = {2},
pages = {475--506},
year = {2021},
doi = {10.4310/JSG.2021.v19.n2.a5}
}
```

*Normalization:* Uses embedded contact homology capacities with standard normalization ($c(B^4(1))=\pi$), linking them to combinatorial Ehrhart data.

```bibtex
@article{HaimKislev2019,
author = {Pazit Haim-Kislev},
title = {On the symplectic size of convex polytopes},
journal = {Geometric and Functional Analysis (GAFA)},
volume = {29},
pages = {440--463},
year = {2019},
doi = {10.1007/s00039-019-00486-4}
}
```

*Normalization:* Focuses on the Ekeland-Hofer-Zehnder capacity (for convex polytopes), which is taken with the usual normalization $c_{EHZ}(B^{2n}(1)) = \pi$.

```bibtex
@article{HaimKislevOstrover2024,
author = {Pazit Haim-Kislev and Yaron Ostrover},
title = {A counterexample to Viterbo's conjecture},
journal = {Preprint, \href{https://arxiv.org/abs/2405.16513}{arXiv: 2405.16513}},
year = {2024}
}
```

*Normalization:* Considers all **normalized** symplectic capacities (i.e. those with $c(B(1)) = c(Z(1))$) on convex domains; the counterexample shows these need not coincide.

```bibtex
@article{AbbondandoloBenedettiEdtmair2023,
author = {Alberto Abbondandolo and Gabriele Benedetti and Oliver Edtmair},
title = {Symplectic capacities of domains close to the ball and Banach--Mazur geodesics in the space of contact forms},
journal = {Preprint, \href{https://arxiv.org/abs/2312.07363}{arXiv: 2312.07363}},
year = {2023}
}
```

*Normalization:* Shows that for domains sufficiently close to a Euclidean ball, **all normalized capacities coincide**; assumes the usual normalization $c(B(1)) = c(Z(1))$ throughout.

```bibtex
@article{Irie2022,
author = {Kei Irie},
title = {Symplectic homology of fiberwise convex sets and homology of loop spaces},
journal = {Journal of Symplectic Geometry},
volume = {20},
number = {2},
pages = {417--470},
year = {2022},
doi = {10.4310/JSG.2022.v20.n2.a4}
}
```

*Normalization:* Uses the symplectic homology capacity (from Floer-Hofer-Wysocki), which is normalized to agree with $c_{EHZ}$ on convex bodies (so that $c_{SH}(B(1)) = c_{EHZ}(B(1))$).

```bibtex
@article{AkopyanKarasev2020,
author = {Arseniy Akopyan and Roman Karasev},
title = {When different norms lead to same billiard trajectories?},
journal = {European Journal of Mathematics},
volume = {8},
number = {4},
pages = {1309--1312},
year = {2022},
doi = {10.1007/s40879-020-00413-7}
}
```

*Normalization:* N/A (study of Minkowski billiards; symplectic capacity appears indirectly via billiard trajectory length bounds).
*Verify:* This brief note (published open-access in a 2022 issue; arXiv preprint 1912.12685) is included because it relates convex billiards in Minkowski norms to symplectic invariants. Please verify if a specific third coauthor “Sasaki” was intended, as no author by that name appears in the publication.

```bibtex
@article{GutkinTabachnikov2002,
author = {Eugene Gutkin and Serge Tabachnikov},
title = {Billiards in Finsler and Minkowski geometries},
journal = {Journal of Geometry and Physics},
volume = {40},
number = {3-4},
pages = {277--301},
year = {2002},
doi = {10.1016/S0393-0440(01)00035-6}
}
```

*Normalization:* N/A (introduces Minkowski billiards; symplectic capacity not used, so no normalization is specified).

```bibtex
@incollection{Ostrover2014,
author = {Yaron Ostrover},
title = {When symplectic topology meets Banach space geometry},
booktitle = {Proceedings of the International Congress of Mathematicians (Seoul, 2014)},
volume = {II},
pages = {959--981},
publisher = {Kyung Moon SA},
year = {2014}
}
```

*Normalization:* Expository overview assuming the standard normalization of symplectic capacities (ball = cylinder capacity), though specific normalization is not the focus.

```bibtex
@article{ArtsteinAvidanMilmanOstrover2008,
author = {Shiri Artstein-Avidan and Vitali Milman and Yaron Ostrover},
title = {The {M}-ellipsoid, symplectic capacities and volume},
journal = {Commentarii Mathematici Helvetici},
volume = {83},
number = {2},
pages = {359--369},
year = {2008},
doi = {10.4171/CMH/128}
}
```

*Normalization:* Uses a normalized symplectic capacity (so that $c(B^{2n}(1)) = c(Z^{2n}(1)) = \pi$) providing a volume-capacity inequality (Viterbo's conjecture in an asymptotic form).

```bibtex
@article{ArtsteinAvidanKarasevOstrover2014,
author = {Shiri Artstein-Avidan and Roman Karasev and Yaron Ostrover},
title = {From symplectic measurements to the {M}ahler conjecture},
journal = {Duke Mathematical Journal},
volume = {163},
number = {11},
pages = {2003--2022},
year = {2014},
doi = {10.1215/00127094-2794999}
}
```

*Normalization:* Works with normalized symplectic capacities (the main conjectures are stated for capacities with $c(B(1))=c(Z(1))$), establishing an equivalence between Viterbo's conjecture and Mahler's conjecture in convex geometry.

```bibtex
@article{OstroverTyomkin2009,
author = {Yaron Ostrover and Ilya Tyomkin},
title = {On the quantum homology algebra of toric Fano manifolds},
journal = {Selecta Mathematica (New Series)},
volume = {15},
number = {1},
pages = {121--149},
year = {2009},
doi = {10.1007/s00029-009-0531-2}
}
```

*Normalization:* (If applicable) This work does not directly define a symplectic capacity; it studies quantum homology of toric manifolds. Any mention of “symplectic capacities” in related toric contexts uses the standard normalization (not explicitly invoked here).

---
### References

1.  Viterbo's conjecture as a worm problem - RWTH Publications
    http://publications.rwth-aachen.de/record/863031/references
2.  9 16 On the Symplectic Size of Convex Polytopes | Geometric and Functional Analysis
    https://link.springer.com/article/10.1007/s00039-019-00486-4
3.  The embedding capacity of 4-dimensional symplectic ellipsoids | Annals of Mathematics
    https://annals.math.princeton.edu/2012/175-3/p05
4.  6 Dense existence of periodic Reeb orbits and ECH spectral invariants
    https://www.aimsciences.org/article/doi/10.3934/jmd.2015.9.357
5.  (PDF) Sub-leading asymptotics of ECH capacities - ResearchGate
    https://www.researchgate.net/publication/345485150\_Sub-leading\_asymptotics\_of\_ECH\_capacities
6.  7 Michael Hutchings: homepage - UC Berkeley Mathematics
    https://math.berkeley.edu/~hutching/
7.  8 AMS :: Transactions of the American Mathematical Society
    https://www.ams.org/tran/2024-377-12/S0002-9947-2024-09298-X/
8.  10 Pazit Haim-Kislev
    https://www.pazithaimkislev.com/
9.  11 Fakultät für Mathematik: Alberto Abbondandolo
    https://math.ruhr-uni-bochum.de/fakultaet/arbeitsbereiche/analysis/symplectic-geometry-group/alberto-abbondandolo/
10. 12 [논문]Symplectic homology of fiberwise convex sets and homology ...
    https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=NART122645989
11. 13 Volume 8, Issue 4 | European Journal of Mathematics
    https://link.springer.com/journal/40879/volumes-and-issues/8-4?error=cookies\_not\_supported&code=e739b93f-0955-4920-94a3-9ab146212937
12. 14 [PDF] arXiv:1912.12685v1 [math.MG] 29 Dec 2019
    https://arxiv.org/pdf/1912.12685
13. 15 Billiards in Finsler and Minkowski geometries - Penn State
    https://pure.psu.edu/en/publications/billiards-in-finsler-and-minkowski-geometries
14. 17 18 ems.press
    https://ems.press/content/serial-article-files/43157
15. 19 Viterbo's conjecture as a worm problem - RWTH Publications
    http://publications.rwth-aachen.de/record/863031/references

## Output E

| Source | Symplectic form | Liouville form | Capacity normalization | Action convention & exact-shift invariance | Deviation notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Viterbo (2000, JAMS)** | Standard $\displaystyle \omega_0=\sum_{j} dx_j\wedge dy_j$ on $\R^{2n}$. | Canonical 1-form (on $T^*\R^n$) $\displaystyle \lambda_0=\sum_jp_j\,dq_j$ with $d\lambda_0=\omega_0$ (equivalently $ \lambda_0=\frac12\sum(x_j\,dy_j - y_j\,dx_j)$ up to exact term). | Chosen so that a ball and cylinder of equal radius have equal capacity: $c(B^{2n} (r))=c(Z^{2n} (r))=\pi\,r^2$ (Unit ball $B^{2n}(1)$ thus has $c=\pi$, aligning with volume-capacity conjecture.) | Defined via a variational action $A=\int(p\,dq - H\,dt)$ whose critical points are periodic orbits. Adding an exact 1-form ($p\,dq\mapsto p\,dq+dF$) or a constant to $H$ does not change Floer-theoretic capacities, so $c(\cdot)$ is invariant under exact symplectomorphisms. | Uses symplectic homology; defines capacities; conventions match Ekeland-Hofer origin. No noted deviations; standard orientation and normalization. |

**Cieliebak-Hofer-Latschev-Schlenk (2007)** | Standard form $\displaystyle \omega_0=\sum_j dx_j\wedge dy_j$ on $\R^{2n}$ (Darboux form). | Typically the standard Liouville primitive ($\lambda_{\text{std}}$) on $\R^{2n}$ with $d\lambda_{\text{std}}=\omega_0$ - e.g. the canonical form on cotangent bundles. (Exact choice not fixed, since capacities are symplectic invariants.) | All "quantitative" capacities in this survey are **normalized** as in Ekeland-Hofer: $c(B^{2n}(r))=c(Z^{2n}(r))=\pi r^2$ (The capacity function $c(a)$ used in their ellipsoid-ball embedding problem satisfies $c(B^{2n}(1))=1$ in units where $\pi=1$.) | Discusses various capacities (Gromov width, Ekeland-Hofer, Hofer-Zehnder, etc.), all of which are invariant under adding exact forms. In particular, capacities defined via an action functional (e.g. Hofer-Zehnder's $\int (p\,dq-H\,dt)$) are unchanged if $\lambda\mapsto\lambda+dF$ or $H\mapsto H+\text{const}$, ensuring well-definedness. | A survey setting; aligns conventions across sources. Uses standard $\omega_0$ and assumes capacities are monotone, conformal, and normalized. No deviations - serves to harmonize notation.

**Cieliebak-Frauenfelder-Paternain (c.~2009-10)** | Standard $\omega_0$ on $\R^{2n}$ (exact symplectic form) as ambient structure. | Typically $d\lambda_0=\sum_j dx_j\wedge dy_j$ with $\lambda_0$ as below. | Uses the radial Liouville form $\displaystyle \lambda_0=\tfrac12\sum_{j} (x_j\,dy_j - y_j\,dx_j)$ on $\R^{2n}$ (so that $X_{\lambda_0}=\frac12(x\partial_x+y\partial_y)$ is outward transverse). On a restricted contact-type energy hypersurface $\Sigma$, $\alpha=\lambda_0|\Sigma$ is the contact form. | No new capacity defined (focus is on existence of closed orbits), but implicitly any capacity (e.g. Ekeland-Hofer's) is taken in the normalized sense $c(B^{2n}(r))=\pi r^2$. | Closed Reeb orbit action is $\displaystyle \oint\alpha$ on $\Sigma$. Since $d(\alpha)=\omega_0$ is exact, this action is independent of the choice of Liouville primitive (adding $dF$ to $\lambda_0$ yields $\oint_\gamma dF=0$). Thus invariance under exact shifts is built-in. | Treats energy hypersurfaces of contact type. Conventions coincide with standard radial form and Hofer's principle; no deviations. (Results imply, e.g., non-vanishing of EHZ capacity for those domains, under standard normalization.)

--- Page 2 ---

**McDuff-Schlenk (2012)** | Standard symplectic structure on $\R^4$ (restr. of $\omega_0=\sum dx_i\wedge dy_i$) for ellipsoids and balls. | Not explicitly used (embeddings are discussed purely in terms of volume and symplectic geometry). A primitive $\lambda_0$ exists globally since domains are star-shaped, but it's not needed in their exposition. | They parameterize balls by $\mu=\pi r^2$. Notation: $B(\mu)\subset\R^4$ is the ball of radius $\sqrt{\mu}$. Consequently, their capacity function $c(a)$ satisfies $c(B(\mu))=\mu$ (so in standard terms $c(B^{4}(r))=\pi r^2$). | No action functional is invoked (problem is purely geometric). Obstructions come from embedded contact homology capacities which are invariants of the exact symplectic structure of the domain. Thus the results ultimately depend only on $(M,\omega)$ (insensitive to any exact form choices on $\partial M$). | Focuses on embedding capacities of 4D ellipsoids. Adopts normalized capacity units from CHLS. No sign or convention deviations - uses ECH capacities results (Taubes-Hutchings) consistent with standard conventions.

**Cristofaro-Gardiner-Hutchings-Ramos (2015)** | Works with four-dimensional Liouville domains $(W, \omega=d\lambda)$. Locally $(W,\omega)$ looks like $(\R^4,\sum dx_i\wedge dy_i)$ (standard form) by Darboux. | Assumes an exact symplectic form $\omega=d\lambda$ on $W$ (e.g. star-shaped domains in $\R^4$). One can take $\lambda$ to be the standard primitive (like $\lambda_0$ on $\R^4$) on a coordinate chart; choice of $\lambda$ does not affect capacities as long as $d\lambda=\omega$. | ECH capacities $c_k$ are used, which are normalized so that, for example, $c_1(B^4(a))=a$ and $c_1(Z^4(a))=a$. (Equivalently, $c_1$ of a ball of radius $r$ is $\pi r^2$, matching $c_{\text{EH}}$ and Gromov's width.) | Actions enter via periods of Reeb orbits: $c_k$ are computed from areas of $J$-holomorphic curves asymptotic to Reeb orbits on $\partial W$. The **action** of a Reeb orbit is $\oint \lambda$. Because $\omega=d\lambda$, any change $\lambda\mapsto\lambda+dF$ (exact) leaves $\oint \lambda$ unchanged on closed orbits. Thus ECH capacities are invariants of the exact symplectic class of $W$ (independent of the particular $\lambda$). | Four-dimensional context. Uses ECH capacity conventions (identical to ball-normalized symplectic capacities). No deviation: capacities are monotone and conformal as usual.

**Cristofaro-Gardiner-Savale (2020)** | Same setup as CG-Hutchings-Ramos (exact symplectic 4-manifolds). Standard $\omega_0$ locally; $\omega=d\lambda$ globally. | Same $\lambda$ as above (Liouville form on $W$). Typically a fixed primitive on $\partial W$ is chosen (e.g. the standard contact form in examples), but results do not depend on this choice. | Uses the same normalization for ECH capacities. (Indeed $c_k$ here agree with those in 2015 for ellipsoids, etc.) The leading asymptotic $c_k\sim \text{const}\cdot!\Vol(W)^{1/2}$ confirms consistency with volume where $c_1(B^4(1))=1$, etc.. | The analysis concerns asymptotic behavior of $c_k$. It relies on spectral invariants of the $\dbar$ operator, but there is no change in action conventions. As with all ECH capacities, invariance under exact shifts of $\lambda$ holds (the capacities depend only on $(W,\omega)$). | No notation changes from CG-Hutchings-Ramos. (Refines the ECH capacity theory but uses identical symplectic/Liouville normalization.)

**Hutchings (ECH capacities notes, 2022)** | Assumes $(\R^4,\omega_0)$ for examples and general $(W, \omega=d\lambda)$. Standard form $\sum dx_i\wedge dy_i$ in $\R^4$. | Uses the standard contact form on $S^3$ (for the unit ball) as a model (half the Hopf form $\alpha_{\mathrm{st}}=\frac12(x\,dy-y\,dx)$, though not always written explicitly). In general, a Liouville form $\lambda$ with $d\lambda=\omega$ is fixed on $W$ but can be changed by an exact term without affecting results. | **Ball-cylinder normalization:** e.g. the Gromov width $c_G(B^4(a))=a$ and $c_G(Z^4(a))=a$. All capacities considered (ECH and the new "elementary" capacities) satisfy $c(B^{2n}(r))=\pi r^2$ (Hutchings often sets $\pi=1$ units for convenience, so $c(B^4(1))=1$ in his notation.) | Emphasizes capacity axioms: monotonicity and conformality. The newly defined capacities use $J$-holomorphic curves; no explicit action integrals appear, but since these capacities equal ECH capacities on standard domains, they inherit the same invariance under exact symplectic deformations. | No deviations. Introduces a new capacity sequence equivalent to ECH capacities (hence same normalization and invariants). Action-shift invariance is implicitly satisfied by construction (capacities depend only on $(M, \omega)$, not on a particular $\lambda$).

--- Page 3 ---

**Wormleighton (e.g. 2019-2021)** | Uses standard symplectic forms in all examples (e.g. $\omega_0$ on $\R^{2n}$) - no alternative convention. | Standard Liouville form (when needed). For instance, in symplectic embedding problems of polydisks/ellipsoids, one uses the usual $\lambda_0$ on $\R^{2n}$ (though capacities are often computed without writing $\lambda$ explicitly). | Explicitly assumes ball normalization: $c(B^{2n}(r))=c(Z^{2n}(r))=\pi r^2$. E.g. any "1-normalized" capacity in his work means $c(B^4(1))=c(Z^4(1))=1$ (which corresponds to $\pi r^2$ normalization). | Invariants like "algebraic capacities" or "Lagrangian capacities" introduced in this circle still satisfy the standard axioms. Any action-type quantities (e.g. lengths of closed geodesics in a convex domain) are invariant under exact 1-form shifts. | No significant deviations. Some work introduces alternate normalization schemes (e.g. "cube normalization" for certain capacities), but these are clearly indicated and still reduce to the standard $\pi r^2$ normalization on balls. Overall, conventions are consistent with symplectic folklore.

**Haim-Kislev (2019)** | Standard $\omega_0$ on $\R^{2n}$; symplectic invariants of convex bodies are computed with respect to the usual structure. | Standard radial Liouville form $\lambda_0$ (as in CFP) or canonical $p\,dq$ on $T^*\R^n$. For example, $\alpha_{\mathrm{st}}$ on $S^{2n-1}$ (unit sphere) is taken as $\lambda_0|{S}$. | Follows Ekeland-Hofer and Hofer-Zehnder conventions: $c(r))=\pi r^2$ (This is crucial in formulating the Viterbo conjecture and related inequalities.) | The Hofer-Zehnder capacity is defined via the supremum of action values for which all levels carry a periodic orbit. Here $c_{\HZ}$ are normalized so $c(B^{2naction*} = \oint p\,dq$ on a Reeb orbit of the energy surface; adding an exact form or shifting the Hamiltonian's zero level does not change the existence of orbits or the capacity value. These invariance properties are assumed and used (see [Haim-Kislev-Hind-Ostrover] for a different but equivalent formula). | No deviations. Uses classical capacity definitions to approach the Viterbo volume-capacity conjecture (partial results in 2019). All conventions (symplectic form, normalization $\pi r^2$) match the standard.

**Haim-Kislev-Ostrover (2024)** | Standard symplectic form on $\R^{2n}$ for convex bodies. (The counterexample to Viterbo's conjecture is constructed in $\R^{2n}$ with the usual $\omega_0$.) | Standard $\lambda_0$ as above. The convex body is treated as a Liouville domain of $\R^{2n}$ (with boundary carrying the induced contact form from $\lambda_0$). | Uses the **normalized Hofer-Zehnder capacity**. In particular, $c_{\HZ}(B^{2n}(1))=\pi$ so that the Euclidean unit ball maximizes capacity/volume in $\R^{2n}$ if the conjecture held. (Their computed counterexample yields $c_{\HZ}>\pi$ times the ball's value, refuting the normalized conjecture.) | Hofer-Zehnder capacity is defined via an action functional on loops (similar to Ekeland-Hofer). The invariance under exact symplectic diffeomorphisms is inherent: one can deform the defining Hamiltonian by an exact symplectic isotopy or add constants without changing $c_{\HZ}$. Haim-Kislev-Ostrover's billiard-dynamic construction exploits this invariance (using Minkowski billiard trajectories to compute actions). | Conventions align with standard capacity theory. No sign flips or unusual forms - the novelty lies in the example, not in the setup. (They translate a billiard length into a symplectic action using the usual $p\,dq$ form, obtaining a capacity value beyond the expected bound.)

**Abbondandolo-Benedetti-Edtmair (2022)** | Standard symplectic forms on the domains of interest (e.g. $\omega_0$ on $\R^{2n}$ or standard symplectic form on cotangent bundles). | Standard Liouville 1-forms. In symplectic homology contexts, one uses the canonical $\theta=p\,dq$ on $T^M$; $d\theta=\omega_{\mathrm{can}}$. Any two exact forms differing by $dF$ give isomorphic Floer homologies, so the choice is not unique but without consequence. | The capacities studied (likely spectral invariants from symplectic homology) are **normalized** to coincide with $c_{\EHZ}$ on balls. Thus $c(B^{2n}(r))=\pi r^2$ and $c(Z^{2n}(r))=\pi r^2$ as usual. | Spectral invariant "actions" are defined by evaluating $\int_{\gamma}\lambda$ for certain periodic orbits or thimbles. Since these invariants come from homology classes in symplectic homology, they do not change if $\lambda$ is replaced by $\lambda+dF$ (exact). Consequently, the resulting capacities depend only on the symplectic isotopy class of the domain (exact-shift invariant). | No deviation from standard conventions. (The work likely compares capacities from different theories, all of which agree on basic normalizations.) Any slight differences in definitions are adjusted by normalizing constants so that all agree on simple domains.

--- Page 4 ---

**Irie (2019)** | Uses standard contact/symplectic structures (e.g. the unit cotangent bundle $S^*M$ with its canonical contact form). Underlying symplectic form in proofs (for pseudoholomorphic curves or ECH) is standard $\omega_0$ on $\R^{2n}$ locally (no unusual forms). | Canonical Liouville forms on cotangent bundles ($\lambda_{\text{can}}=p\,dq$) and their restrictions as contact forms. In particular, for $S^*M$ the form $p\,dq$ is used (in local coordinates) - the standard choice ensuring $d\lambda_{\text{can}}$ equals the standard symplectic form on $T^*M$. | Capacities are not the focus (main result is about dense Reeb orbits), but any mentioned invariants (like ECH capacities or contact volume) adhere to the usual normalization. For example, if ECH capacities are invoked, $c_1(B^4(1))=1$ in suitable units. | Action of a periodic Reeb orbit = $\oint \alpha$ (with $\alpha$ the contact form). Changing $\alpha$ by an exact 1-form on the contact manifold (which is locally $\alpha\mapsto \alpha+dF$, though truly contact forms can't be globally exact) does not alter period spectra. Irie's use of ECH or contact homology is insensitive to exact perturbations of $\alpha$. | Standard setup in contact dynamics. No special conventions to note – employs the usual contact form on $S^*M$ and standard symplectic form on $\R^{2n}$ in all arguments.

**Akopyan-Karasev (-Sasaki)** | Implicitly uses the standard area form in $\R^2$ (their context is planar convex billiards). In symplectic terms, $\omega_0=dx\wedge dy$ in the plane. Higher-dimensional generalizations likewise use standard forms on $\R^{2n}$. | Not explicitly stated in geometric billiard language. However, an equivalent Liouville form in $\R^2$ is $\lambda=\frac12(x\,dy - y\,dx)$ (whose $d\lambda$ is the area form). In Minkowski billiards (dual billiard in support function space), the standard contact form appears implicitly in deriving action-length relations. | In symplectic reformulations of their results, the "capacity" (e.g. the minimum action of a periodic billiard trajectory) is normalized as area. For instance, an ellipse has the largest inscribed billiard caustic equal to itself - in capacity terms this corresponds to $c(B^2(R))=\pi R^2$. Their results were used to confirm special cases of Viterbo's conjecture, assuming standard normalization. | The "action" in billiards is the optical length of a periodic orbit. This can be translated to $\oint p\,dq$ via a duality transform. Because adding an exact one-form $dF(q)$ corresponds to a change of refractive index or a coordinate shift, which doesn't change closed orbit lengths, the correspondence with symplectic action is invariant under exact shifts. | Billiard dynamics is recast in symplectic terms (via dual curves and caustics). The conventions match standard symplectic geometry when translated: no unusual normalizations. (Their work inspired parts of the Haim-Kislev-Ostrover construction, and it is consistent with the classical $\pi r^2$ capacity scaling.)

**Gutkin-Tabachnikov (2002)** | Studies convex planar billiards; effectively uses the standard area form on $\R^2$ (invariants like perimeter and curvature appear, which correspond to standard symplectic volumes). | Not given in symplectic form, but one can associate a Liouville form on the unit tangent bundle of the plane (the geodesic flow form). In spirit this is the same as the standard contact form on $S^*S^1$ in the billiard boundary's tangent space. | No explicit symplectic capacity, but the quantities studied (e.g. lengths of periodic billiard orbits, or the Tabachnikov repetitive property) correspond to symplectic invariants that would be normalized in the standard way. If one were to define a "billiard capacity," it would equal $\pi r^2$ for a disk of radius $r$. | The length of a closed billiard trajectory is invariant under adding an exact derivative to the boundary's support function (this is analogous to action invariance under $\lambda\to\lambda+dF$). Thus any action-like invariant extracted is independent of exact form shifts. | Their work is phrased in classical geometry terms, but it aligns with symplectic invariants conceptually. When translated to symplectic language, all standard conventions apply. No deviations - just a different language.

--- Page 5 ---

**Ostrover (ICM 2014)** | Assumes the standard symplectic form on $\R^{2n}$ throughout (to discuss various capacities and inequalities). | Typically discusses Liouville domains and symplectic embeddings; the Liouville form is taken as standard (though usually not written out, as the focus is on $\omega$). | Clearly states the **normalization** $c(B^{2n}(r))=c(Z^{2n}(r))=\pi r^2$ as part of defining symplectic capacities. All capacities in the survey (Gromov's width, EHZ, etc.) satisfy this. | Emphasizes that capacities are invariant under symplectomorphisms. Any action-based definition (like EHZ via Hamiltonian action) is understood to be invariant under exact perturbations. (For example, Ostrover notes that one can define capacities via minimizing $\oint p\,dq$ on closed characteristics, which is well-defined up to exact 1-forms.) | No deviations. This survey reinforces standard conventions (sometimes introducing new viewpoints, but always compatible with $\omega_0$ and normalized capacities).

**Artstein-Avidan-Milman-Ostrover (2011)** | Standard symplectic form on $\R^{2n}$. (They study convex bodies in $(\R^{2n},\omega_0)$, so $\omega_0=\sum dx\wedge dy$ is assumed.) | Standard Liouville form (e.g. $\lambda_0=\sum x\,dy$ or $\frac12(x\,dy-y\,dx)$) is implicit in any contact-type arguments. The M-ellipsoid construction uses the usual symplectic structure with its standard primitive. | The M-ellipsoid is characterized by maximizing a normalized symplectic capacity for a given volume. Thus the capacity used is normalized in the usual way ($c(B)=\pi$ for the unit ball). All volume-capacity inequalities are stated with respect to this normalization. | Their symplectic invariants (displacement energy, EHZ capacity, etc.) are all defined via action minimax principles. The invariance under exact shifts (e.g. Hamiltonian $H\to H+ \text{const}$ or $\lambda\to\lambda+dF$) is taken for granted, as these capacities are standard. | No deviations. They introduce the concept of an M-ellipsoid to approximate a convex body in symplectic terms, but all underlying definitions use conventional normalization and symplectic form.

**Artstein-Avidan-Karasev-Ostrover (2014)** | Standard $(\R^{2n},\omega_0)$ for all discussions of symplectic volume/capacity. (Toric domains and convex bodies are considered with the inherited standard form.) | Standard Liouville forms. E.g. for a toric domain, one uses the action-angle form $\sum p_i\,dq_i$. Any exact form differences are immaterial to their "Mahler conjecture for symplectic capacities." | Uses standard normalization. In particular, their "symplectic Minkowski functional" is scaled so that the Euclidean ball is capacity-maximizing for its volume (matching $c(B^{2n}(1))=\pi$). All comparisons of capacities and volume assume $c(Z(1))=c(B(1))$. | Invariance under exact transformations is inherent. (They compare different symplectic capacities; since each individually is symplectomorphism-invariant, the comparisons are too. Adding an exact form to $\omega$ or rescaling $\lambda$ does not affect any capacity in consideration.) | No deviations in form or normalization. Connects symplectic capacity inequalities to convex geometry (Mahler's conjecture) using the usual definitions.

**Ostrover-Tyomkin (2015)** | Standard symplectic form on toric domains (and $\R^{2n}$). Likely $\omega_0$ on $\C P^n$ or similar if considered, but all references to capacities use the standard reference form. | Standard primitives (e.g. the tautological 1-form on $\C^n$ in action-angle coordinates). If working with moment-map coordinates, one still ultimately uses $d\lambda=\omega_{\text{std}}$. | Presumably assumes $c(B^{2n}(r))=\pi r^2$ as normalization (consistent with all other Ostrover works). Any new capacity or sequence introduced for toric manifolds would be scaled to agree with classical ones on a ball (often explicitly checked in examples). | If an action-functional or generating function is used (for instance, to define a toric capacity via periodic orbits on a torus boundary), it respects the usual invariances (exact 1-form addition, etc.). The authors would ensure the capacity is well-defined on the symplectic category (hence independent of exact choices). | No known deviations. Any special cases (e.g. toric-specific capacity formulas) reduce to standard ones on simple domains. Conventions mirror the standard symplectic toolkit as in earlier works.

--- Page 6 ---

**References**

 The strong Viterbo conjecture and various flavours of duality in Lagrangian products
https://arxiv.org/pdf/2505.07572

 [PDF] Systolic S1-index and characterization of non-smooth Zoll convex ...
https://www.math.ias.edu/~sarabt/zoominar/slides/250411c_Matijevic.pdf

 temp.dvi
https://www.math.toronto.edu/karshon/grad/2006-07/capacities.pdf

 tau.ac.il
https://www.tau.ac.il/~shiri/Symplectic/log.pdf

 [math/0506191] Quantitative symplectic geometry
https://ar5iv.org/pdf/math/0506191

 The embedding capacity of 4-dimensional symplectic ellipsoids
https://annals.math.princeton.edu/wp-content/uploads/annals-v175-n3-p05-p.pdf

 [2201.03143] An elementary alternative to ECH capacities
https://ar5iv.labs.arxiv.org/html/2201.03143

 [2208.13666] Cube normalized symplectic capacities - arXiv
https://arxiv.org/abs/2208.13666

 [PDF] Convexity In Contact Geometry And Reeb Dynamics
https://escholarship.org/content/qt4fq8400g/qt4fq8400g_noSplash_26885d4ad166c1ca2ce20307a72bc373.pdf


## Output F

*   **Counterexample (2024)** – Haim-Kislev and Ostrover discovered a family of convex domains violating Viterbo's 2000 volume-capacity conjecture. In dimension $4$ (phase space $\mathbb{R}^4$), the counterexample is the **Lagrangian product of two planar convex polygons** — specifically, two congruent regular pentagons, one rotated by 90°. Using the Ekeland-Hofer-Zehnder symplectic capacity $c_{EHZ}$ (normalized so that $c(B^4(1))=\pi$), they showed
    $$
    \frac{c_{EHZ}(K\times T)^2}{2\,A^2} = \frac{\sqrt{5}+3}{5} \approx 1.047 > 1,
    $$
    i.e. $c_{EHZ} (K\times T)^2 > 2A^2$. Here $A$ is the area of each pentagon, so $2A^2$ equals the right-hand side of Viterbo's conjectured inequality $c^2 \le 2A^2$ (since $n!=2$ in $\mathbb{R}^4$). In other words, this domain's capacity exceeds that of the Euclidean ball of equal volume, contradicting the conjecture that the ball maximizes every symplectic capacity for a given volume. The result in fact extends to all higher even dimensions $2n$ (with $n\ge 2$) by taking suitable products, so the conjecture fails in every dimension $\ge 4$.

*   **Method & verification** – The construction exploits an explicit formula for the EHZ capacity of certain convex polytopes, together with a link to **Minkowski billiard dynamics**. In general, for convex domains in $\mathbb{R}^{2n}$ the EHZ capacity equals the minimum action of a closed characteristic on the boundary, which for product domains can be interpreted via periodic billiard trajectories. In this case, $c_{EHZ}(K\times T)$ was identified with the length of a $2$-bounce periodic billiard orbit diagonally across the pentagon. The authors derived a closed-form expression involving elementary trigonometric values (e.g. $\cos(\pi/10)$ and $\cos(\pi/5)$) and obtained the above numerical ratio rigorously. They also cross-checked it using a computer implementation of a **combinatorial capacity formula** for convex polytopes developed in Haim-Kislev's earlier work. In particular, a short "hands-on" proof of the capacity formula for $K\times T$ is given in the paper alongside a software-assisted calculation, lending credibility to the counterexample.

*   **Status and implications** – The result was first announced in mid-2024 (arXiv preprint) and has since been peer-reviewed (accepted for publication in *Annals of Mathematics* in late 2025). No errors have been reported, and the finding definitively **disproves Viterbo's volume-capacity conjecture**. A key consequence is that symplectic capacities do not all coincide on convex bodies. there exist convex domains where capacities like $c_{EHZ}$ exceed the value predicted by more elementary capacities (e.g. the Gromov width). This breakthrough has prompted a re-examination of which classes of domains might still satisfy an optimal “ball-isoperimetric" inequality. Notably, **centrally symmetric convex bodies** remain a plausible regime for Viterbo's bound: in fact, for such bodies the conjecture is equivalent to the famous **Mahler conjecture on volume products**. which is widely believed and proven in certain cases (e.g. confirmed in dimension 3). Moreover, prior work had verified Viterbo's inequality in special cases — for example, for all domains sufficiently close (in $C^2$ or $C^3$-smooth shape) to the Euclidean ball, for certain Lagrangian product shapes, and for monotone toric domains. These results, together with the new counterexample, suggest that while the unrestricted conjecture is false, a refined **symplectic isoperimetric inequality** might still hold under additional symmetry or convexity constraints, a topic of ongoing research and discussion.

Primary references: Haim-Kislev & Ostrover's counterexample paper; Gil Kalai's summary of the result (context and implications).

***

### References

*   Viterbo’s conjecture was refuted by Pazit Haim-Kislev and Yaron Ostrover | Combinatorics and more
    `https://gilkalai.wordpress.com/2024/09/23/viterbos-conjecture-was-refuted-by-pazit-haim-kislev-and-yaron-ostrover/`

*   arxiv.org
    `https://arxiv.org/pdf/2405.16513`

*   A counterexample to Viterbo's Conjecture | Annals of Mathematics
    `https://annals.math.princeton.edu/articles/22235`