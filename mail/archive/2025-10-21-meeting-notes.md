# Meeting Notes — Weekly Sync / Demos

- Date: 2025-10-21 9:00 (Tue)
- Type: meeting notes (paraphrased; no verbatim quotes)
- Participants: Jörn, Kai

## Discussed / Presented

- discussed progress so far
- noted that the dataset fails bc it shows a systolic ratio < 1 for the Viterbo counterexample
- noted that different methods however produce the same results, so the calculations are likely correct
- discussed performance (<50ms for capacity) and that it's perfectly fine for now
- planned follow-ups to investigate further
- next steps agreed on:
  - debug dataset
  - create visualizations to show progress to Kai and Jörn more trustworthily
- follow ups:
  - create larger dataset, carry on with all the basic tests
- idea to get a proven conjecture:
  - we can use epsilon-delta balls (e.g. scaling) to estimate volume & capacity of polytopes in some parameterized space (e.g. 10 facets = (S^3 x R_>0)^10 = R^40)
  - if we have a covering of such balls wrt some metric, we can prove thus that the covered polytopes have all capacity < 1-epsilon
  - notes: some polytopes are symplectomorphic to the ball, so we need to either have fewer facets/vertices, or we need to handle the =1 cases carefully, e.g. by making a uniqueness statement inside the covering balls near sys=1 (e.g. via derivatives?)
  - note: we also need an argument to make the volume finite that we need to cover. 
  - note: it's enough to cover the vol(K)=1 surface, ofc using polytopes with vol(K)>1 and <1 as the upper and lower bounds. 
  - note: we can use symplectomorphisms to e.g. linearly transform polytopes to not be elongated; maybe that's enough with vol=~1 to get a compact domain to cover?
  - note: unsure if eps-delta are good enough, i.e. if we can work with practically tractible many balls
- since dataset was broken, no results shown
