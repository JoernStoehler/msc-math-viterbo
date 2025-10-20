---
source: arXiv:1210.2167
fetched: 2025-10-20
---
# The asymptotics of ECH capacities

                                                     The asymptotics of ECH capacities
arXiv:1210.2167v4 [math.SG] 12 Dec 2013




                                                     Daniel Cristofaro-Gardiner, Michael Hutchings,
                                                           and Vinicius Gripp Barros Ramos∗


                                                                                   Abstract
                                                      In a previous paper, the second author used embedded contact
                                                  homology (ECH) of contact three-manifolds to define “ECH capaci-
                                                  ties” of four-dimensional symplectic manifolds. In the present paper
                                                  we prove that for a four-dimensional Liouville domain with all ECH
                                                  capacities finite, the asymptotics of the ECH capacities recover the
                                                  symplectic volume. This follows from a more general theorem relat-
                                                  ing the volume of a contact three-manifold to the asymptotics of the
                                                  amount of symplectic action needed to represent certain classes in
                                                  ECH. The latter theorem was used by the first and second authors to
                                                  show that every contact form on a closed three-manifold has at least
                                                  two embedded Reeb orbits.


                                          1       Introduction
                                          Define a four-dimensional Liouville domain 1 to be a compact symplectic four-
                                          manifold (X, ω) with oriented boundary Y such that ω is exact on X, and
                                          there exists a contact form λ on Y with dλ = ω|Y . In [4], a sequence of real
                                          numbers
                                                          0 = c0 (X, ω) < c1 (X, ω) ≤ c2 (X, ω) ≤ · · · ≤ ∞
                                              ∗
                                               The first author was partially supported by NSF grant DMS-0838703. The second
                                          and third authors were partially supported by NSF grant DMS-1105820.
                                             1
                                               This definition of “Liouville domain” is slightly weaker than the usual definition, which
                                          would require that ω have a primitive λ on X which restricts to a contact form on Y .




                                                                                        1
called ECH capacities was defined. The definition is reviewed below in §1.2.
The ECH capacities obstruct symplectic embeddings: If (X, ω) symplecti-
cally embeds into (X ′ , ω ′ ), then

                               ck (X, ω) ≤ ck (X ′ , ω ′)                   (1)

for all k. For example, a theorem of McDuff [12], see also the survey [5], shows
that ECH capacities give a sharp obstruction to symplectically embedding
one four-dimensional ellipsoid into another.
    The first goal of this paper is to prove the following theorem, relating the
asymptotics of the ECH capacities to volume. This result was conjectured in
[4] based on experimental evidence; it was proved in [4, §8] for star-shaped
domains in R4 and some other examples.

Theorem 1.1. [4, Conj. 1.12] Let (X, ω) be a four-dimensional Liouville
domain such that ck (X, ω) < ∞ for all k. Then

                            ck (X, ω)2
                         lim           = 4 vol(X, ω).
                        k→∞      k
Here the symplectic volume is defined by
                                        Z
                                      1
                         vol(X, ω) =      ω ∧ ω.
                                      2 X
    In particular, when all ECH capacities are finite, the embedding ob-
struction (1) for large k recovers the obvious volume constraint vol(X, ω) ≤
vol(X ′ , ω ′ ). As we review below, the hypothesis that ck (X, ω) < ∞ for all k
is a purely topological condition on the contact structure on the boundary;
for example it holds whenever ∂X is diffeomorphic to S 3 .
    We will obtain Theorem 1.1 as a corollary of the more general Theorem 1.3
below, which also has applications to refinements of the Weinstein conjecture
in Corollary 1.4. To state Theorem 1.3, we first need to review some notions
from embedded contact homology (ECH). More details about ECH may be
found in [6] and the references therein.

1.1    Embedded contact homology
Let Y be a closed oriented three-manifold and let λ be a contact form on Y ,
meaning that λ ∧ dλ > 0. The contact form λ determines a contact structure

                                           2
ξ = Ker(λ), and the Reeb vector field R characterized by dλ(R, ·) = 0
and λ(R) = 1. Assume that λ is nondegenerate, meaning that all Reeb
orbits are nondegenerate. Fix Γ ∈ H1 (Y ). The embedded contact homology
ECH(Y, ξ, Γ) is the homology of a chain complex ECC(Y, λ, Γ, J) over Z/2
defined as follows.
     A generator of the chain complex is a finite set of pairs α = {(αi , mi )}
where the αi are distinct embedded Reeb orbits, the mi are positive in-
tegers,
P         mi = 1 whenever αi is hyperbolic, and the total homology class
   i mi [αi ] = Γ ∈ H1 (Y ). To define the chain complex differential ∂ one
chooses a generic almost complex structure J on R × Y such that J(∂s ) = R
where s denotes the R coordinate, J(ξ) = ξ with dλ(v, Jv) ≥ 0 for v ∈ ξ,
and J is R-invariant. Given another chain complex generator β = {(βj , nj )},
the differential coefficient h∂α, βi ∈ Z/2 is a mod
                                                 P 2 count of J-holomorphic
curves
P        in R × Y that converge as currents to i mi αi as s → +∞ and to
   j j j as s → −∞, and that have “ECH index” equal to 1. The defini-
     n β
tion of the ECH index is explained in [2]; all we need to know here is that
the ECH index defines a relative Z/d-grading on the chain complex, where
d denotes the divisibility of c1 (ξ) + 2 PD(Γ) in H 2 (Y ; Z) mod torsion. It is
shown in [8, §7] that ∂ 2 = 0.
     One now defines ECH(Y, λ, Γ, J) to be the homology of the chain complex
ECC(Y, λ, Γ, J). Taubes [15] proved that if Y is connected, then there is a
canonical isomorphism of relatively graded Z/2-modules
                                              −∗
                                      d
                  ECH∗ (Y, λ, Γ, J) = HM           (Y, sξ + PD(Γ)).               (2)
           −∗
      d
Here HM       denotes the ‘from’ version of Seiberg-Witten Floer cohomology
as defined by Kronheimer-Mrowka [11], with Z/2 coefficients2 , and with the
sign of the relative grading reversed. Also, sξ denotes the spin-c structure
determined by the oriented 2-plane field ξ, see e.g. [3, Ex. 8.2]. It follows
that, whether or not Y is connected, ECH(Y, λ, Γ, J) depends only on Y , ξ,
and Γ, and so can be denoted by ECH∗ (Y, ξ, Γ).
    There is a degree −2 map
                    U : ECH∗ (Y, ξ, Γ) −→ ECH∗−2 (Y, ξ, Γ).                       (3)
This map on homology is induced by a chain map which counts J-holomorphic
curves with ECH index 2 that pass through a base point in R × Y . When Y
  2
    One can define ECH with integer coefficients [9, §9], and the isomorphism (2) also
exists over Z, as shown in [17]. However Z/2 coefficients will suffice for this paper.

                                          3
is connected, the U map (3) does not depend on the choice of base point, and
agrees under Taubes’s isomorphism (2) with an analogous map on Seiberg-
Witten Floer cohomology [18]. If Y is disconnected, then there is one U map
for each component of Y .
    Although ECH is a topological invariant by (2), it contains a distinguished
class which can distinguish some contact structures. Namely, the empty set
of Reeb orbits is a generator of ECC(Y, λ, 0, J); it is a cycle by the conditions
on J, and so it defines a distinguished class
                              [∅] ∈ ECH(Y, ξ, 0),                            (4)
called the ECH contact invariant. Under the isomorphism (2), the ECH con-
tact invariant agrees with an analogous contact invariant in Seiberg-Witten
Floer cohomology [18].
    There is also a “filtered” version of ECH, which is sensitive to the contact
form and not just the contact structure. If α = {(αi , mi )} is a generator of
the chain complex ECC(Y, λ, Γ, J), its symplectic action is defined by
                                       X Z
                              A(α) =       mi   λ.                           (5)
                                       i      αi

It follows from the conditions on the almost complex structure J that if the
differential coefficient h∂α, βi =
                                 6 0 then A(α) > A(β). Consequently, for
each L ∈ R, the span of those generators α with A(α) < L is a subcomplex,
which is denoted by ECC L (Y, λ, Γ, J). The homology of this subcomplex
is the filtered ECH which is denoted by ECH L (Y, λ, Γ). Inclusion of chain
complexes induces a map
                      ECH L (Y, λ, Γ) −→ ECH(Y, ξ, Γ).                       (6)
It is shown in [10, Thm. 1.3] that ECH L (Y, λ, Γ) and the map (6) do not
depend on the almost complex structure J.
    A useful way to extract invariants of the contact form out of filtered ECH
is as follows. Given a nonzero class σ ∈ ECH(Y, ξ, Γ), define
                                 cσ (Y, λ) ∈ R
to be the infimum over L such that the class σ is in the image of the inclusion-
induced map (6). So far we have been assuming that the contact form λ is
nondegenerate. If λ is degenerate, one defines cσ (Y, λ) = limn→∞ cσ (Y, λn ),
where {λn } is a sequence of nondegenerate contact forms which C 0 -converges
to λ, cf. [4, §3.1].

                                       4
1.2    ECH capacities
Let (Y, λ) be a closed contact three-manifold and assume that the ECH con-
tact invariant (4) is nonzero. Given a nonnegative integer k, define ck (Y, λ)
to be the minimum of cσ (Y, λ), where σ ranges over classes in ECH(Y, ξ, 0)
such that Aσ = [∅] whenever A is a composition of k of the U maps associ-
ated to the components of Y . If no such class σ exists, define ck (Y, λ) = ∞.
The sequence {ck (Y, λ)}k=0,1,... is called the ECH spectrum of (Y, λ).
   Now let (X, ω) be a Liouville domain with boundary Y and let λ be a
contact form on Y with dλ = ω|Y . One then defines the ECH capacities of
(X, ω) in terms of the ECH spectrum of (Y, λ) by

                               ck (X, ω) = ck (Y, λ).

This definition is valid because the ECH contact invariant of (Y, λ) is nonzero
by [10, Thm. 1.9]. It follows from [4, Lem. 3.9] that ck (X, ω) does not depend
on the choice of contact form λ on Y with dλ = ω|Y .
    Note the volume of a Liouville domain as above satisfies
                                             1
                           vol(X, ω) =         vol(Y, λ),                  (7)
                                             2
where the volume of a contact three-manifold is defined by
                                      Z
                          vol(Y, λ) =   λ ∧ dλ.                            (8)
                                              Y

To prove (7), let λ′ be a primitive of ω on X, and use Stokes’s theorem on
X and then again on Y to obtain
               Z           Z            Z         Z
                                ′
                   ω∧ω =      λ ∧ω =       ω∧λ=      dλ ∧ λ.
                X              Y              Y             Y

   By equation (7), Theorem 1.1 is now a consequence of the following result
about the ECH spectrum:

Theorem 1.2. [4, Conj. 8.1] Let (Y, λ) be a closed contact three-manifold
with nonzero ECH contact invariant. If ck (Y, λ) < ∞ for all k, then

                             ck (Y, λ)2
                         lim            = 2 vol(Y, λ).
                         k→∞      k

                                         5
     Note that the hypothesis ck (Y, λ) < ∞ just means that the ECH contact
invariant is in the image of all powers of the U map when Y is connected,
or all compositions of powers of the U maps when Y is disconnected. The
comparison with Seiberg-Witten theory implies that this is possible only if
c1 (ξ) ∈ H 2 (Y ; Z) is torsion; see [4, Rem. 4.4(b)].
     By [4, Prop. 8.4], to prove Theorem 1.2 it suffices to consider the case
when Y is connected. Theorem 1.2 in this case follows from our main theorem
which we now state.

1.3    The main theorem
Recall from §1.1 that if c1 (ξ)+2 PD(Γ) ∈ H 2 (Y ; Z) is torsion, then ECH(Y, ξ, Γ)
has a relative Z-grading, and we can arbitrarily refine this to an absolute Z-
grading. The main theorem is now:
Theorem 1.3. [4, Conj. 8.7] Let Y be a closed connected contact three-
manifold with a contact form λ and let Γ ∈ H1 (Y ). Suppose that c1 (ξ) +
2 PD(Γ) is torsion in H 2 (Y ; Z), and let I be an absolute Z-grading of ECH(Y, ξ, Γ).
Let {σk }k≥1 be a sequence of nonzero homogeneous classes in ECH(Y, ξ, Γ)
with limk→∞ I(σk ) = ∞. Then
                              cσk (Y, λ)2
                          lim             = vol(Y, λ).                       (9)
                          k→∞   I(σk )
   The following application of Theorem 1.3 was obtained in [1]:
Corollary 1.4. [1, Thm. 1.1] Every (possibly degenerate) contact form on
a closed three-manifold has at least two embedded Reeb orbits.
    The proof of Theorem 1.3 has two parts. In §2 we show that the left
hand side of (9) (with lim replaced by lim sup) is less than or equal to the
right hand side. This is actually all that is needed for Corollary 1.4. In §3
we show that the left hand side (with lim replaced by lim inf) is greater than
or equal to the right hand side. The two arguments are independent of each
other and can be read in either order. The proof of the upper bound uses
ingredients from Taubes’s proof of the isomorphism (2). The proof of the
lower bound uses properties of ECH cobordism maps to reduce to the case
of a sphere, where (9) can be checked explicitly.
    Both arguments use Seiberg-Witten theory; in particular, Seiberg-Witten
theory is used to define ECH cobordism maps in [10]. However the proof of

                                       6
the lower bound given here would not need Seiberg-Witten theory if one
could give an alternate construction of ECH cobordism maps.


2     The upper bound
In this section we prove the upper bound half of Theorem 1.3:
Proposition 2.1. Under the assumptions of Theorem 1.3,
                                 cσk (Y, λ)2
                       lim sup               ≤ vol(Y, λ).                (10)
                         k→∞       I(σk )
    To prove Proposition 2.1, we can assume without loss of generality that
λ is nondegenerate. To see this, assume that (10) holds for nondegenerate
contact forms and suppose that λ is degenerate. We can find a sequence
of functions f1 > f2 > · · · > 1, which C 0 -converges to 1, such that fn λ is
nondegenerate for each n. It follows from the monotonicity property in [4,
Lem. 4.2] that
                            cσk (Y, λ) ≤ cσk (Y, fn λ)
for every n and k. For each n, it follows from this and the inequality (10)
for λn that
                            cσ (Y, λ)2
                     lim sup k          ≤ vol(Y, fn λ).
                       k→∞    I(σk )
Since limn→∞ vol(Y, fn λ) = vol(Y, λ), we deduce the inequality (10) for λ.
   Assume henceforth that λ is nondegenerate. In §2.1–§2.6 below we review
some aspects of Taubes’s proof of the isomorphism (2) and prove some related
lemmas. In §2.7 we use these to prove Proposition 2.1.

2.1    Seiberg-Witten Floer cohomology
The proof of the isomorphism (2) involves perturbing the Seiberg-Witten
equations on Y . To write down the Seiberg-Witten equations we first need
to choose a Riemannian metric on Y . Let J be a generic almost complex
structure on R × Y as needed to define the ECH chain complex. The almost
complex structure J determines a Riemannian metric g on Y such that the
Reeb vector field R has length 1 and is orthogonal to the contact planes ξ,
and
                             1
                   g(v, w) = dλ(v, Jw),       v, w ∈ ξy .             (11)
                             2
                                       7
Note that this metric satisfies

                            |λ| = 1,       ∗dλ = 2λ.                        (12)

One could dispense with the factors of 2 in (11) and (12), but we are keeping
them for consistency with [3] and its sequels.
    Let S denote the spin bundle for the spin-c structure sξ + PD(Γ). The
inputs to the Seiberg-Witten equations for this spin-c structure are a connec-
tion A on det(S) and a section ψ of S. The spin bundle S splits as a direct
sum
                             S = E ⊕ (E ⊗ ξ),
where E and E ⊗ ξ are, respectively, the +i and −i eigenspaces of Clifford
multiplication by λ. Here ξ is regarded as a complex line bundle using the
metric and the orientation. A connection A on det(S) is then equivalent to
a (Hermitian) connection A on E via the relation A = A0 + 2A, where A0 is
a distinguished connection on ξ reviewed in [13, §2.1].
   For a positive real number r, consider the following version of the per-
turbed Seiberg-Witten equations for a connection A on E and spinor ψ:

                   ∗FA = r(hcl(·)ψ, ψi − iλ) + i(∗dµ + ω̄)
                                                                            (13)
                   DA ψ = 0.

Here FA denotes the curvature of A; cl denotes Clifford multiplication; ω̄
denotes the harmonic 1-form such that ∗ω̄/π represents c1 (ξ) ∈ H 2 (Y ; R);
and µ is a generic 1-form such that dµ has “P-norm” less than 1, see [13,
§2.1]. Finally, DA denotes the Dirac operator determined by the connection
A on det(S) corresponding to the connection A on E.
    The group of gauge transformations C ∞ (Y, S 1 ) acts on the space of pairs
(A, ψ) by g · (A, ψ) = (A − 2g −1dg, gψ). The quotient of the space of pairs
(A, ψ) by the group of gauge transformations is called the configuration space.
The set of solutions to (13) is invariant under gauge transformations. A
solution to the Seiberg-Witten equations is called reducible if ψ ≡ 0 and
irreducible otherwise. An irreducible solution is called nondegenerate if it is
cut out transversely after modding out by gauge transformations, see [13,
§3.1].
    For fixed µ, when r is not in a certain discrete set, there are only finitely
many irreducible solutions to (13) and these are all nondegenerate. In this
case one can define the Seiberg-Witten Floer cohomology chain complex with

                                       8
Z/2 coefficients, which we denote by CMd ∗ (Y, sξ,Γ, λ, J, r). The chain complex
is generated by irreducible solutions to (13), along with additional generators
determined by the reducible solutions. The differential counts solutions to
a small abstract perturbation of the four-dimensional Seiberg-Witten equa-
tions on R × Y . In principle the chain complex differential may depend on
the choice of abstract perturbation, but since the abstract perturbation is
irrelevant to the proof of Proposition 2.1, we will omit it from the notation.

2.2    The grading
                                                                                ∗
Under our assumption that c1 (ξ)+2 PD(Γ) is torsion, the chain complex CM  d
has a noncanonical absolute Z-grading defined as follows, cf. [11, §14.4]. The
linearization of the equations (13) modulo gauge equivalence at a pair (A, ψ),
not necessarily solving the equations (13), defines a self-adjoint Fredholm
operator LA,ψ . If (A, ψ) is a nondegenerate irreducible solution to (13), then
the operator LA,ψ has trivial kernel, and one defines the grading gr(A, ψ) ∈ Z
to be the spectral flow from LA,ψ to a reference self-adjoint Fredholm operator
L0 between the same spaces with trivial kernel. The grading function gr
depends on the choice of reference operator; fix one below. To describe the
gradings of the remaining generators, recall that the set of reducible solutions
modulo gauge equivalence is a torus T of dimension b1 (Y ). As explained in
[11, §35.1], one can perturb the Seiberg-Witten equations using a Morse
function
                                  f : T → R,                                (14)
so that the chain complex generators arising from reducibles are identified
with pairs ((A, 0), φ), where (A, 0) is a critical point of f and φ is a suitable
eigenfunction of the Dirac operator DA . The grading of each such generator
is less than or equal to gr(A, 0), where the latter is defined as the spectral
flow to L0 from an appropriate perturbation of the operator LA,0 .
    We will need the following key result of Taubes relating the grading to
the Chern-Simons functional. Fix a reference connection AE on E. Given
any other connection A on E, define the Chern-Simons functional
                           Z
                cs(A) = − (A − AE ) ∧ (FA + FAE − 2i∗ω̄).                    (15)
                             Y

Note that this functional is gauge invariant because the spin-c structure
sξ + PD(Γ) is assumed torsion.

                                       9
Proposition 2.2. [13, Prop. 5.1] There exists K, r∗ > 0 such that for all
r > r∗ , if (A, ψ) is a nondegenerate irreducible solution to (13), or a reducible
solution which is a critical point of (14), then

                                        1
                          gr(A, ψ) +      2
                                            cs(A) < Kr 31/16 .                       (16)
                                       4π

2.3     Energy
Given a connection A on E, define the energy
                                    Z
                          E(A) = i λ ∧ FA .
                                             Y

Filtered ECH has a Seiberg-Witten analogue defined using the energy func-
tional as follows. Fix r such that the chain complex CM d ∗ is defined. Given
a real number L, define CM d ∗ to be the submodule of CM d ∗ spanned by gen-
                               L
erators with energy less than 2πL. It is shown in [13], as reviewed in [10,
Lem. 2.3], that if r is sufficiently large, then all chain complex generators
with energy less than 2πL are irreducible, and CM d ∗ is a subcomplex, whose
                                                      L
homology we denote by HM    d ∗ . Moreover, as shown in [13] and reviewed in
                                L
[10, Eq. (35)], if there are no ECH generators of action exactly L and if r is
sufficiently large, then there is a canonical isomorphism of relatively graded
chain complexes

                                       d −∗ (Y, sξ,Γ, λ1 , J1 , r).
                 ECC∗L (Y, λ, Γ, J) −→ CM                                            (17)
                                          L

Here (λ1 , J1 ) is an “L-flat approximation” to (λ, J), which is obtained by
suitably modifying (λ, J) near the Reeb orbits of action less than L; the
precise definition is reviewed in [10, §3.1] and will not be needed here.
    The isomorphism (17) is induced by a bijection on generators; the idea is
that in the L-flat case3 , if r is sufficiently large, then for every ECH generator
α of action less than L, there is a corresponding irreducible solution (A, ψ)
to (13) such that the zero set of the E component of ψ is close to the Reeb
   3
    In the non-L-flat case, there may be several Seiberg-Witten solutions corresponding to
the same ECH generator, and/or Seiberg-Witten solutions corresponding to sets of Reeb
orbits with multiplicities which are not ECH generators because they include hyperbolic
orbits with multiplicity greater than one. See [15, §5.c, Part 2].



                                           10
orbits in α, the curvature FA is concentrated near these Reeb orbits, and the
energy of this solution is approximately 2πA(α).
   The isomorphism of chain complexes (17) induces an isomorphism on
homology
                                    ≃  d −∗ (Y, sξ,Γ , λ1 , J1 , r),
                ECH∗L (Y, λ, Γ, J) −→ HM                                 (18)
                                           L

and inclusion of chain complexes defines a map

                  d −∗ (Y, sξ,Γ , λ1 , J1 , r) −→ HM
                  HM                              d −∗ (Y, sξ,Γ ).              (19)
                     L

Composing the above two maps gives a map

                                           d −∗ (Y, sξ,Γ).
                     ECH∗L (Y, λ, Γ, J) −→ HM                                   (20)

The isomorphism (2) is the direct limit over L of the maps (20).

2.4     Volume in Seiberg-Witten theory
The volume enters into the proof of Proposition 2.1 in two essential ways.
     The first way is as follows. It is shown in [3, §3] that for any given grading,
there are no generators arising from reducibles if r is sufficiently large. That
is, given an integer j, let sj be the supremum of all values of r such that there
exists a generator of the chain complex CM     d ∗ (Y, sξ,Γ, λ, J, r) with grading at
least −j associated to a reducible solution to (13). Then sj < ∞ for all j.
     We now give an upper bound on the number sj in terms of the volume.
                1
Fix 0 < δ < 16    . Given a positive integer j, let rj be the largest real number
such that
                                  1 2
                            j=       2
                                       rj vol(Y, λ) − rj2−δ .                    (21)
                                16π
Lemma 2.3. If j is sufficiently large, then sj < rj .
Proof. Let A0 be a connection on E with FA0 = idµ + i∗ω. Observe that
                    1
(Ared
  r , ψ) = (A0 − 2 irλ, 0) is a solution to (13). Moreover, every other re-
ducible solution is given by (A, 0), where A = Ared
                                                 r  + α for a closed 1-form
α. It follows from (15) that

                                 cs(A) = cs(Ared
                                             r )                                (22)

and
                                 1 2
                        cs(Ared
                            r ) = r vol(Y, λ) + O(r).                           (23)
                                 4
                                         11
Now suppose that j is sufficiently large that rj > r∗ where r∗ is the constant
in Proposition 2.2. Suppose that r > rj and that (A, 0) is a chain complex
generator with gr(A, 0) ≥ −j. Then equation (21) implies that
                                      1 2
                     gr(A, 0) ≥ −         r vol(Y, λ) + r 2−δ .
                                    16π 2
Combining this with (22) and (23) gives
                                   1
                     gr(A, 0) +        cs(A) ≥ r 2−δ + O(r).
                                  4π 2
This contradicts Proposition 2.2 if r is sufficiently large, which is the case if
j is sufficiently large.
    The second essential way that volume enters into the proof of Proposi-
tion 2.1 is via the following a priori upper bound on the energy:

Lemma 2.4. There is an r-independent constant C such that any solution
(A, ψ) to (13) satisfies
                                     r
                           E(A) ≤      vol(Y, λ) + C.                       (24)
                                     2
Proof. This follows from [13, Eq. (2.7)], which is proved using a priori es-
timates on solutions to the Seiberg-Witten equations. Note that there is a
factor of 1/2 in (24) which is not present in [13, Eq. (2.7)]. The reason is
that the latter uses the Riemannian volume as defined by the metric (12),
which is half of the contact volume (8) which we are using.

2.5     Max-min families
Given a connection A on E and a section ψ of S, define a functional
                                                Z
                   1
        F (A, ψ) = (cs(A) − rE(A)) + eµ (A) + r hDA ψ, ψidvol,
                   2                              Y

where                                     Z
                             eµ (A) = i        FA ∧ µ.
                                           Y

Since the spin-c structure sξ + PD(Γ) is assumed torsion, the functional F is
gauge invariant. The significance of the functional F is that the differential

                                        12
on the chain complex CM    d ∗ counts solutions to abstract perturbations of
the upward gradient flow equation for F . In particular, F agrees with an
appropriately perturbed version of the Chern-Simons-Dirac functional from
[11], up to addition of an r-dependent constant, see [10, Eq. (98)].
    A key step in Taubes’s proof of the Weinstein conjecture [13] is to use
a “max-min” approach to find a sequence (rn , ψn , An ), where rn → ∞ and
(ψn , An ) is a solution to (13) for r = rn with an n-independent bound on the
energy. We will use a similar construction in the proof of Proposition 2.1.
    Specifically, fix an integer j, and let sj be the number from §2.4. Let
      d ∗ (Y, sξ,Γ ) be a nonzero homogeneous class with grading greater than
σ̂ ∈ HM
or equal to −j. Fix r > sj for which the chain complex CM  d ∗ (Y, sξ,Γ, λ, J, r) is
defined. Since we are using Z/2-coefficients, any cycle representing the class
σ̂ has the form η = Σi (Ai , ψi ), where the pairs (Ai , ψi ) are distinct gauge
equivalence classes of solutions to (13). Define Fmin (η) = mini F (Ai , ψi ),
and
                               Fσ̂ (r) = max Fmin(η).
                                       [η]=σ̂

Note that it is natural to take the max-min here because the differential on
the chain complex increases F ; compare [3, Prop. 10.7]. Note also that Fσ̂ (r)
must be finite because there are only finitely many irreducible solutions to
(13).
    The construction in [15, §4.e] shows that for any such class σ̂, there
exists a smooth family of solutions (Aσ̂ (r), ψσ̂ (r)) to (13) with the same
grading as σ̂, defined for each r > sj for which the chain complex CM         d ∗ is
defined, such that Fσ̂ (r) = F (Aσ̂ (r), ψσ̂ (r)). This family is smooth for each
interval on which it is defined, but may not extend continuously over those
values of r for which the chain complex CM    d ∗ is not defined. Call the family
(Aσ̂ (r), ψσ̂ (r))r>sj a max-min family for σ̂. Given such a max-min family, if
r > sj is such that CM  d ∗ is defined, define Eσ̂ (r) = E(Aσ̂ (r), ψσ̂ (r)).
Lemma 2.5. (a) Fσ̂ (r) extends to a continuous and piecewise smooth func-
   tion of r ∈ (sj , ∞).
       d             1                                 d ∗ is defined.
 (b)      Fσ̂ (r) = − Eσ̂ (r) for all r > sj such that CM
       dr            2
Proof. (a) follows from [15, Prop. 4.7]; and (b) follows from [13, Eq. (4.6)],
see also [3, Lem. 10.8].
In particular, Eσ̂ (r) does not depend on the choice of max-min family.

                                        13
2.6     Max-min energy and min-max symplectic action
The numbers Eσ̂ (r) from §2.5 are related to the numbers cσ (Y, λ) from §1.2
as follows:

Proposition 2.6. Let σ be a nonzero homogeneous class in ECH(Y, ξ, Γ),
             d ∗ (Y, sξ,Γ) denote the class corresponding to σ under the
and let σ̂ ∈ HM
isomorphism (2). Then

                              lim Eσ̂ (r) = 2πcσ (Y, λ).
                             r→∞

Here, and in similar limits below, it is understood that the limit is over r
such that the chain complex CM   d ∗ is defined.
    The proof of Proposition 2.6 requires two preliminary lemmas which will
also be needed later. To state the first lemma, recall from [14, Prop. 2.8] that
in the case Γ = 0, if r is sufficiently large then there is a unique (up to gauge
equivalence) “trivial” solution (Atriv , ψtriv ) to (13) such that 1 − |ψ| < 1/2
on all of Y . If (λ, J) is L-flat with L > 0, then (Atriv , ψtriv ) corresponds to
the empty set of Reeb orbits under the isomorphism (17) with Γ = 0, see
the beginning of [16, §3]. Any solution not gauge equivalent to (Atriv , ψtriv )
will be called “nontrivial”. Let L0 denote one half the minimum symplectic
action of a Reeb orbit.

Lemma 2.7. There exists an r-independent constant c such that if r is suf-
ficiently large, then every nontrivial solution (A, ψ) to (13) satisfies E(A) >
2πL0 and
                            |cs(A)| ≤ cr 2/3 E(A)4/3 .                      (25)

Proof. The chain complex ECC∗L0 (Y, λ, Γ, J) has no generators unless Γ = 0,
in which case the only generator is the empty set of Reeb orbits. In particular,
the pair (λ, J) is L0 -flat. By (17), if r is sufficiently large then every nontrivial
solution (A, ψ) to (13) has E(A) ≥ 2πL0 . Given this positive lower bound
on the energy, the estimate (25) now follows as in [13, Eq. (4.9)]. Note that
it is assumed there that E(A) ≥ 1, but the same argument works as long as
there is a positive lower bound on E(A).
   Now fix a positive number γ such that γ < δ/4.




                                         14
Lemma 2.8. For every integer j there exists ρ ≥ 0 such if r ≥ ρ is such that
the chain complex CMd ∗ is defined, and if (A, ψ) is a nontrivial irreducible
solution to (13) of grading −j, then

                              |cs(A)| ≤ r 1−γ E(A).                         (26)

Proof. Fix j. Let (A, ψ) be a nontrivial solution to (13) of grading −j with

                              |cs(A)| > r 1−γ E(A).                         (27)

By Lemma 2.7, if r is sufficiently large then

                            |cs(A)| ≤ cr 2/3 E(A)4/3 .                      (28)

Combining (27) with (28), we conclude that E(A) ≥ c−3 r 1−3γ . Using (27)
again, it follows that
                          |cs(A)| > c−3 r 2−4γ .
But this contradicts Proposition 2.2 when r is sufficiently large with respect
to j, since δ > 4γ.
Proof of Proposition 2.6. Choose L0 > cσ (Y, λ) and let (λ1 , J1 ) be an L0 -flat
approximation to (λ, J). For r large, define f1 (r) to be the infimum over L
such that the class σ̂ is in the image of the map (19). We first claim that

                           lim (f1 (r) − cσ (Y, λ)) = 0.                    (29)
                          r→∞

This holds because for every L ≤ L0 which is not the symplectic action of
an ECH generator, in particular L 6= cσ (Y, λ), if r is sufficiently large that
the isomorphism (18) is defined, then the class σ̂ is in the image of the map
(19) if and only if L > cσ (Y, λ).
     Next define f (r) for r large to be the infimum over L such that the class
σ̂ is in the image of the inclusion-induced map

                    d ∗ (Y, sξ,Γ , λ, J, r) → HM
                    HM                        d ∗ (Y, sξ,Γ).                (30)
                       L

It follows from [10, Lem. 3.4(c)] that

                            lim (f (r) − f1 (r)) = 0.                       (31)
                            r→∞




                                         15
    By (29) and (31), to complete the proof of Proposition 2.6 it is enough
to show that
                         lim (Eσ̂ (r) − 2πf (r)) = 0.                   (32)
                            r→∞

    To prepare for the proof of (32), assume that r is sufficiently large so
that Lemma 2.7 is applicable and Lemma 2.8 is applicable to j = −gr(σ̂).
Also assume that r is sufficiently large so that all nontrivial Seiberg-Witten
solutions in grading gr(σ̂) are irreducible and have positive energy. Let (A, ψ)
be a nontrivial solution in grading gr(σ̂). Then
                               1
                     F (A, ψ) = (cs(A) − rE(A)) + eµ (A).
                               2
By [13, Eq. (4.2)] and Lemma 2.7, we have

                                  |eµ (A)| ≤ κE(A)                               (33)

where κ is an r-independent constant. The above and Lemma 2.8 imply that
                                  −2
    (1 − r −γ − 2κr −1 )E(A) ≤       F (A, ψ) ≤ (1 + r −γ + 2κr −1 )E(A).        (34)
                                   r
Also, it follows from the construction of the trivial solution in [14] that

                                          F (Atriv , ψtriv )
                     lim E(Atriv ) = lim                     = 0.                (35)
                     r→∞              r→∞        r
    Now (32) can be deduced easily from (34) and (35). The details are as
follows. Fix ε > 0 and suppose that r is sufficiently large as in the above
paragraph. By the definition of f (r), the class σ̂ is in the image of the
map (30) for L = f (r) + ε. Also, if r is sufficiently large, then by (34)
and (35), and the fact that L has an upper bound when r is large by (29)
                   P                          d L representing the class σ̂, then
and (31), if η = i (Ai , ψi ) is a cycle in CM
−2F (Ai , ψi )/r < 2π(L + ε) for each i. Consequently −2Fσ̂ (r)/r < 2π(L + ε).
By (34) and (35) again, if r is sufficiently large then Eσ̂ (r) < 2π(L+2ε), which
means that Eσ̂ (r) < f (r) + 3ε.
    By similar reasoning, if Eσ̂ (r) < f (r) − ε, then if r is sufficiently large, the
class σ̂ is in the image of the map (30) for L = f (r) − ε/2, which contradicts
the definition of f (r).



                                         16
2.7    Proof of the upper bound
We are now ready to prove Proposition 2.1. Before giving the details, here
is the rough idea of the proof. Consider an ECH class σ of grading j large,
and choose the absolute gradings so that the corresponding Seiberg-Witten
grading is −j. Consider a max-min family for the corresponding Seiberg-
Witten Floer cohomology class as r increases from rj to ∞. If (A, ψ) is the
element of the max-min family for r = rj , then heuristically we have
                                    rj               2
       4π 2 cσ (Y, λ)2   E(A)2         vol(Y, λ) + C
                       ≈       ≤ 12 2                 2−δ
                                                          ≈ 4π 2 vol(Y, λ).
               j           j         r vol(Y, λ) + rj
                                16π 2 j

The idea of the approximation on the left is that E(A) converges to 2πcσ (Y, λ)
as r → ∞ by Proposition 2.6; and since each member of the max-min family
for r > rj is irreducible by Lemma 2.3, we can apply Proposition 2.2 and some
calculations to control the change in E(A) in this family. The inequality in
the middle follows from equation (21) and Lemma 2.4. The approximations
on the left and right get better as j increases because limj→∞ rj = +∞.
Proof of Proposition 2.1. The proof has six steps.
    Step 1: Setup. If σ ∈ ECH∗ (Y, ξ, Γ) is a nonzero homogeneous class,
         d ∗ (Y, sξ,Γ) denote the corresponding class in Seiberg-Witten Floer
let σ̂ ∈ HM
cohomology via the isomorphism (2). We can choose the absolute grading I
on ECH(Y, ξ, Γ) so that the Seiberg-Witten grading of σ̂ is −I(σ) for all σ.
For Steps 1–5, fix such a class σ and write j = I(σ). We will obtain an upper
bound on cσ (Y, λ) in terms of j when j is sufficiently large, see (47) below.
    To start, we always assume that j is sufficiently large so that j > 0, the
number rj defined in (21) satisfies rj ≥ 1, Proposition 2.2 and Lemma 2.7
are applicable to r ≥ rj , Lemma 2.3 is applicable so that rj > sj , and the
trivial solution (Atriv , ψtriv ) does not have grading −j.
    Fix a max-min family (Aσ̂ (r), ψσ̂ (r)) for σ̂ as in §2.5. Let S denote the
discrete set of r for which the chain complex CM  d ∗ is not defined. Recall that
the max-min family is defined for r ∈ (sj , ∞) \ S. For such r, define
                  E(r) = Eσ̂ (r) = E(Aσ̂ (r)),
                 cs(r) = cs(Aσ̂ (r)),
                 eµ (r) = eµ (Aσ̂ (r)),
                            2Fσ̂ (r)           cs(r) 2eµ (r)
                  v(r) = −            = E(r) −      −        .              (36)
                                r                r     r

                                       17
It follows from Lemma 2.5 that v(r) extends to a continuous and piecewise
smooth function of r ∈ (sj , ∞). However the functions cs(r), E(r), and eµ (r)
might not extend continuously over S. For any equation below involving the
latter three functions, it is implicitly assumed that r ∈
                                                        / S.
    By Lemma 2.5, we have

                           dv(r)  cs(r) 2eµ (r)
                                 = 2 +          .                           (37)
                            dr     r      r2
By Proposition 2.2 we have the key estimate

                                      1
                           −j +           cs(r) < Kr 2−δ                    (38)
                                     4π 2

whenever r ≥ rj . Here we are using the fact that gr(Aσ̂ (r), ψσ̂ (r)) = −j,
because (Aσ̂ (r), ψσ̂ (r)) is irreducible by Lemma 2.3.
    Define a number r = r σ̂ as follows. We know from Lemma 2.8 that if r is
sufficiently large then
                                  |cs(r)| ≤ r 1−γ E(r).                 (39)
If (39) holds for all r ≥ rj , define r = rj . Otherwise define r to be the
supremum of the set of r for which (39) does not hold.
    Step 2. We now show that

                           lim sup E(r) ≤ v(r̄)g(r̄),                       (40)
                              r≥r̄

where                                                         
                                         r −γ + 2γκr −1
                     g(r) = exp                                    ,        (41)
                                      γ (1 − r −γ − 2κr −1 )
and κ is the constant in (33). Here and below we assume that j is sufficiently
large so that 1 − rj−γ − 2κrj−1 > 0.
    To prove (40), assume that r ≥ r̄. Then by (36), (39), and (33), as in
(34), we have
                                         1
                         E(r) ≤                  v(r).                    (42)
                                 1 − r − 2κr −1
                                      −γ

Also v(r) > 0, since r ≥ 1. By (37), (39), (33) and (42) we have

dv(r)                           r −1−γ + 2κr −2          r −1−γ + 2κr −2
      ≤ (r −1−γ +2κr −2)E(r) ≤                   v(r) ≤                     v(r).
 dr                            1 − r −γ − 2κr −1        1 − r̄ −γ − 2κr̄ −1

                                          18
Dividing this inequality by v(r) and integrating from r̄ to r gives
                           
                      v(r)      r̄ −γ + 2γκr̄ −1 − r −γ − 2γκr −1
                 ln           ≤
                      v(r̄)             γ (1 − r̄ −γ − 2κr̄ −1)
                                     r̄ + 2γκr̄ −1
                                       −γ
                              <                          .
                                γ (1 − r̄ −γ − 2κr̄ −1 )

Therefore
                                v(r) < v(r̄)g(r̄).
Together with (42), this proves (40).
   Step 3. We claim now that
                               1
                        v(r̄) ≤ rj vol(Y, λ) + C0 r̄ 1−δ .                     (43)
                               2
Here and below, C0 , C1 , C2 . . . denote positive constants which do not depend
on σ̂ or r, and which we do not need to know anything more about.
   To prove (43), use (37), (38), (33), and Lemma 2.4 to obtain

                        dv   4π 2 (j + Kr 2−δ )
                           ≤                    + C1 r −1 .
                        dr           r2
Integrating this inequality from rj to r̄ and using j > 0, we deduce that
                                                1−δ
                 4π 2 j 4π 2 j 4π 2 K(r̄ 1−δ − rj )
  v(r̄) − v(rj ) ≤      −         +                 + C1 (ln r̄ − ln rj )
                  rj       r̄         1−δ
                                                                               (44)
                 4π 2 j
               ≤        + C2 r̄ 1−δ .
                  rj

Also, by (36), (38), (33), and Lemma 2.4, we have
                                                2−δ
           1                      4π 2 (−j + Krj      ) + 2κ(rj vol(Y, λ)/2 + C)
   v(rj ) ≤ rj vol(Y, λ) + C +
           2                                              rj
           1               4π 2 j
          ≤ rj vol(Y, λ) −         + C3 rj1−δ .
           2                rj
                                                                               (45)

Adding (44) and (45) gives (43).


                                        19
   Step 4. We claim now that if j is sufficiently large then
                                               1
                                   r̄ ≤ C4 rj1−2γ .                         (46)
    To prove this, by the definition of r̄, if r̄ > rj then there exists a number
r slightly smaller than r̄ such that |cs(r)| > r 1−γ E(r). It then follows from
Lemma 2.7 that
                            r 1−γ E(r) < cr 2/3 E(r)4/3 .
Therefore
                        r 2−4γ ≤ c3 r 1−γ E(r) ≤ c3 |cs(r)|.
By (38) and the definition of rj in (21), we have
                          c3 |cs(r)| ≤ C5 rj2 + C6 r 2−δ .
Combining the above two inequalities and using the fact that r can be arbi-
trarily close to r̄, we obtain
                            r̄ 2−4γ ≤ C5 rj2 + C6 r̄ 2−δ .
Since δ > 4γ and r̄ > rj → ∞ as j → ∞, if j is sufficiently large then
                                            1
                                 C6 r̄ 2−δ ≤ r̄ 2−4γ .
                                            2
Combining the above two inequalities proves (46).
  Assume henceforth that j is sufficiently large so that (46) holds.
  Step 5. We claim now that
                                  1
                   cσ (Y, λ) ≤      rj vol(Y, λ)g(r̄) + C7 rj1−ν ,          (47)
                                 4π
               1−δ
where ν = 1 − 1−2γ  > 0.
   To prove (47), insert (46) into (43) to obtain
                               1
                        v(r̄) ≤ rj vol(Y, λ) + C8 rj1−ν .
                               2
The above inequality and (40) imply that
                                                       
                               1                    1−ν
              lim sup E(r) ≤     rj vol(Y, λ) + C8 rj     g(r̄)
                r→∞            2
                             1
                           ≤ rj vol(Y, λ)g(r̄) + C9 rj1−ν .
                             2
                                          20
It follows from this and Proposition 2.6 that (47) holds.
    Step 6. We now complete the proof of Proposition 2.1 by applying (47)
to the sequence {σk } and taking the limit as k → ∞.
    Let jk = I(σk ) and r̄k = r̄σ̂k . It then follows from (47) and the definition
of the numbers rjk in (21) that for every k sufficiently large,
             cσk (Y, λ)2   (16π 2 )−1 rj2k vol(Y, λ)2 g(r̄k )2 + C10 rj2−ν
                         ≤                                              k
                                                                             (48)
               I(σk )            (16π 2 )−1 rj2k vol(Y, λ) − rj2−δ
                                                                 k

                             vol(Y, λ)g(r̄k )2 + C11 rj−ν
                         =                             k
                                                            .
                                     1 − C12 rj−δ
                                               k

By hypothesis, as k → ∞ we have jk → ∞, and hence r̄k > rjk → ∞. It
then follows from (41) that limk→∞ g(r̄k ) = 1. Putting all this into the above
inequality proves (10).


3     The lower bound
In this last section we prove the following proposition, which is the lower
bound half of Theorem 1.3:
Proposition 3.1. Under the assumptions of Theorem 1.3,
                                    cσk (Y, λ)2
                          lim inf               ≥ vol(Y, λ).                 (49)
                           k→∞        I(σk )
   In §3.1 we review some aspects of ECH cobordism maps, and in §3.2 we
use these to prove Proposition 3.1.

3.1    ECH cobordism maps
Let (Y+ , λ+ ) and (Y− , λ− ) be closed oriented three-manifolds, not necessarily
connected, with nondegenerate contact forms. A strong symplectic cobordism
“from” (Y+ , λ+ ) “to” (Y− , λ− ) is a compact symplectic four-manifold (X, ω)
with boundary ∂X = Y+ − Y− such that ω|Y± = dλ± . Following [4], define
a “weakly exact symplectic cobordism” from (Y+ , λ+ ) to (Y− , λ− ) to be a
strong symplectic cobordism as above such that ω is exact.
    It is shown in [4, Thm. 2.3], by a slight modification of [10, Thm. 1.9],
that a weakly exact symplectic cobordism as above induces a map
             ΦL (X, ω) : ECH L (Y+ , λ+ , 0) −→ ECH L (Y− , λ− , 0)

                                          21
for each L ∈ R, defined by counting solutions to the Seiberg-Witten equa-
tions, perturbed using ω, on a “completion” of X.
    More generally, let A ∈ H2 (X, ∂X), and write ∂A = Γ+ − Γ− where
Γ± ∈ H1 (Y± ). In our weakly exact symplectic cobordism, suppose that ω has
a primitive on X which agrees with λ± on each component of Y± for which
the corresponding component of Γ± is nonzero. Then the same argument
constructs a map

          ΦL (X, ω, A) : ECH L (Y+ , λ+ , Γ+ ) −→ ECH L (Y− , λ− , Γ− ),      (50)

defined by counting solutions to the Seiberg-Witten equations in the spin-c
structure corresponding to A. As in [4, Thm. 2.3(a)], there is a well-defined
direct limit map

 Φ(X, ω, A) = lim ΦL (X, ω, A) : ECH(Y+ , ξ+ , Γ+ ) −→ ECH(Y− , ξ− , Γ− ),
                L→∞
                                                                              (51)
where ξ± = Ker(λ± ).
   The relevance of the map (51) for Proposition 3.1 is that given a class
σ+ ∈ ECH(Y+ , ξ+ , Γ+ ), if σ− = Φ(X, ω, A)σ+ , then

                          cσ+ (Y+ , λ+ ) ≥ cσ− (Y− , λ− ).                    (52)

The inequality (52) follows directly from (51) and the definition of cσ± in
§1.1, cf. [4, Lem. 4.2]. Here we interpret cσ = −∞ if σ = 0. By a limiting
argument as in [4, Prop. 3.6], the inequality (52) also holds if the contact
forms λ± are allowed to be degenerate.
    The map (50) is a special case of the construction in [7] of maps on ECH
induced by general strong symplectic cobordisms. Without the assumption
on the primitive of ω, these maps can shift the symplectic action filtration,
but the limiting map (51) is still defined.
    For computations we will need four properties of the map (51). First, if
X = ([a, b] × Y, d(es λ)) is a trivial cobordism from (Y, eb λ) to (Y, ea λ), where
s denotes the [a, b] coordinate, then

                       Φ(X, ω, [a, b] × Γ) = idECH(Y,ξ,Γ) .                   (53)

This follows for example from [10, Cor. 5.8].
   Second, suppose that (X, ω) is the composition of strong symplectic
cobordisms (X+ , ω+ ) from (Y+ , λ+ ) to (Y0 , λ0 ) and (X− , ω− ) from (Y0 , λ0 )

                                        22
to (Y− , λ− ). Let Γ0 ∈ H1 (Y0 ) and let A± ∈ H2 (X± , ∂± X± ) be classes with
∂A+ = Γ+ − Γ0 and ∂A− = Γ0 − Γ− . Then
                                                   X
             Φ(X− , ω− , A− ) ◦ Φ(X+ , ω+ , A+ ) =      Φ(X, ω, A).       (54)
                                                    A|X± =A±


This is proved the same way as the composition property in [10, Thm. 1.9].
   Third, if X is connected and Y± are both nonempty, then
                       Φ(X, ω, A) ◦ U+ = U− ◦ Φ(X, ω, A),                         (55)
where U± can be the U map associated to any of the components of Y± . This
is proved as in [4, Thm. 2.3(d)].
    Fourth, since we are using coefficients in the field Z/2, it follows from the
definitions that the ECH of a disjoint union is given by the tensor product
   ECH((Y, ξ) ⊔ (Y ′ , ξ ′), Γ ⊕ Γ′ ) = ECH(Y, ξ, Γ) ⊗ ECH(Y ′ , ξ ′, Γ′ ).       (56)
If (X, ω) is a strong symplectic cobordism from (Y+ , λ+ ) to (Y− , λ− ), and if
(X ′ , ω ′) is a strong symplectic cobordism from (Y+′ , λ′+ ) to (Y−′ , λ′− ), then it
follows from the construction of the cobordism map that the disjoint union
of the cobordisms induces the tensor product of the cobordism maps:
          Φ((X, ω) ⊔ (X ′ , ω ′), A ⊕ A′ ) = Φ(X, ω, A) ⊗ Φ(X ′ , ω ′, A′ ).      (57)

3.2     Proof of the lower bound
Proof of Proposition 3.1. The proof has four steps.
   Step 1. We can assume without loss of generality that
                                    Uσk+1 = σk                                    (58)
for each k ≥ 1. To see this, note that by the isomorphism (2) of ECH
with Seiberg-Witten Floer cohomology, together with properties of the latter
proved in [11, Lemmas 22.3.3, 33.3.9], we know that if the grading ∗ is
sufficiently large, then ECH∗ (Y, ξ, Γ) is finitely generated and
                    U : ECH∗ (Y, ξ, Γ) −→ ECH∗−2 (Y, ξ, Γ)
is an isomorphism. Hence there is a finite collection of sequences satisfying
(58) such that every nonzero homogeneous class in ECH(Y, ξ, Γ) of suffi-
ciently large grading is contained in one of these sequences (recall that we

                                          23
are using Z/2 coefficients). Thus it is enough to prove (49) for a sequence
satisfying (58). Furthermore, in this case (49) is equivalent to

                                  cσk (Y, λ)2
                        lim inf               ≥ 2 vol(Y, λ).               (59)
                         k→∞           k
   Step 2. When (Y, λ) is the boundary of a Liouville domain, the lower
bound (59) was proved for a particular sequence {σk } satisfying (58) in [4,
Prop. 8.6(a)]. We now set up a modified version of this argument.
   Fix a > 0 and consider the symplectic manifold

                           ([−a, 0] × Y, ω = d(es λ))

where s denotes the [−a, 0] coordinate. The idea is that if a is large, then
([−a, 0] × Y, ω) is “almost” a Liouville domain whose boundary is (Y, λ).
    Fix ε > 0. We adopt the notation that if r > 0, then B(r) denotes the
closed ball
                         B(r) = {z ∈ C2 | π|z|2 ≤ r},
with the restriction of the standard symplectic form on C2 . Choose disjoint
symplectic embeddings

                      {ϕi : B(ri ) → [−a, 0] × Y }i=1,...,N

such that ([−a, 0] × Y ) \ ⊔i ϕi (B(ri )) has symplectic volume less than ε. One
can find such embeddings using a covering of [−a, 0] × Y by Darboux charts.
Let
                                            GN
                    X = ([−a, 0] × Y ) \        int(ϕi (B(ri ))).
                                              i=1
                                                                       −a
Then
FN (X, ω) is a weakly exact symplectic cobordism from (Y, λ) to (Y, e λ)⊔
  i=1 ∂B(ri ). Here we P
                       can take the contact form on ∂B(ri ) to be the restric-
tion of the 1-form 2 2k=1(xk dyk − yk dxk ) on R4 ; we omit this from the
                     1

notation. Note that there is a canonical isomorphism

                             H2 (X, ∂X) = H1 (Y ).

   The symplectic form ω on X has a primitive es λ which restricts to the
contact forms on the convex boundary (Y, λ) and on the component (Y, e−a λ)



                                         24
of the concave boundary. Hence, as explained in §3.1, we have a well-defined
map
                                                    N
                                                                                  !
                                                   G
Φ = Φ(X, ω, Γ) : ECH(Y, ξ, Γ) −→ ECH (Y, ξ) ⊔          ∂B(ri ), (Γ, 0, . . . , 0)
                                                             i=1
                                                                              (60)
which satisfies (52). By (56), the target of this map is
                  N
                                              !                N
                 G                                            O
ECH (Y, ξ) ⊔        ∂B(ri ), (Γ, 0, . . . , 0) = ECH(Y, ξ, Γ)⊗   ECH(∂B(ri )).
                 i=1                                               i=1

Let U0 denote the U map on the left hand side associated to the component
Y , and let Ui denote the U map on the left hand side associated to the
component ∂B(ri ). Note that U0 or Ui acts on the right hand side as the
tensor product of the U map on the appropriate factor with the identity on
the other factors. By (55) we have
                                 Φ(U0 σ) = Ui Φ(σ)                            (61)
for all σ ∈ ECH(Y, ξ, Γ) and for all i = 0, . . . , N.
    Step 3. We now give an explicit formula for the cobordism map Φ in (60).
    Recall that ECH(∂B(ri )) has a basis {ζk }k≥0 where ζ0 = [∅] and Ui ζk+1 =
ζk . This follows either from the computation of the Seiberg-Witten Floer
homology of S 3 in [11], or from direct calculations in ECH, see [6, §4.1]. We
can now state the formula for Φ:
Lemma 3.2. For any class σ ∈ ECH(Y, ξ, Γ), we have
                    X      X
            Φ(σ) =              U0k σ ⊗ ζk1 ⊗ · · · ⊗ ζkN .
                          k≥0 k1 +...+kN =k

   Note that the sum on the right is finite because the map U0 decreases
symplectic action.
Proof of Lemma 3.2. Given σ, we can expand Φ(σ) as
                         X
               Φ(σ) =          σk1 ,...,kN ⊗ ζk1 ⊗ · · · ⊗ ζkN                (62)
                           k1 ,...,kN ≥0

where σk1 ,...,kN ∈ ECH(Y, ξ, Γ). We need to show that
                             σk1 ,...,kN = U0k1 +···+kN σ.                    (63)

                                           25
We will prove by induction on k = k1 + · · · + kN that equation (63) holds for
all σ.
    To prove (63) when k = 0, let X ′ denote the disjoint union of the trivial
cobordism ([−a − 1, −a] × Y, d(es λ)) and the balls B(ri ). Then the compo-
sition X ′ ◦ X is the trivial cobordism ([−a − 1, 0] × Y, d(es λ)) from (Y, eλ ) to
(Y, e−a−1 λ). Now each ball B(ri ) induces a cobordism map

                         ΦB(ri ) : ECH(∂B(ri )) −→ Z/2

as in (51). By (57) and (53) we have

                Φ(X ′ , Γ) = idECH(Y,ξ,Γ) ⊗ΦB(r1 ) ⊗ · · · ⊗ ΦB(rN ) .

It then follows from (53) and the composition property (54) that

                      σ = (Φ(X ′ , Γ) ◦ Φ)(σ)
                                X                         N
                                                          Y
                        =                   σk1 ,...,kN         ΦB(ri ) (ζki ).
                            k1 ,...,kN ≥0                 i=1

Now ΦB(ri ) sends ζ0 to 1 by [4, Thm. 2.3(b)], and ζm to 0 for all m > 0 by
grading considerations (the corresponding moduli space of Seiberg-Witten
solutions in the completed cobordism has dimension 2m). Therefore σ =
σ0,...,0 as desired.
     Next let k > 0 and suppose that (63) holds for smaller values of k. To
prove (63), we can assume without loss of generality that k1 > 0. Applying
U1 to equation (62) and then using equation (61) with i = 1, we obtain

                            σk1 ,...,kN = (U0 σ)k1 −1,k2 ,...,kN .

By inductive hypothesis,

                         (U0 σ)k1 −1,k2 ,...,kN = U0k−1 (U0 σ).

The above two equations imply (63), completing the proof of Lemma 3.2.
   Step 4. We now complete the proof of Proposition 3.1. Let {σk }k≥1 be a
sequence in ECH(Y, ξ, Γ) satisfying (58). By (52) we have
                                                 N
                                                          !
                                                 G
               cσk (Y, λ) ≥ cΦ(σk ) (Y, e−a λ) ⊔   ∂B(ri ) .
                                                                   i=1


                                                 26
By Lemma 3.2 and [4, Eq. (5.6)], we have
                         N
                                  !
                         G
    cΦ(σk ) (Y, e−a λ) ⊔   ∂B(ri ) =
                             i=1
                                                                      N
                                                                                        !
                                                                      X
                 max
                 ′
                                max       ′
                                              cU k′ σk (Y, e−a λ) +         cζki (∂B(ri )) .
                Uk   σk 6=0 k1 +···+kN =k       0
                                                                      i=1


Since U k−1 σk = σ1 6= 0, it follows from the above equation and inequality
that
                                           XN
                  cσk (Y, λ) ≥     max        cζki (∂B(ri )).         (64)
                                     k1 +···+kN =k−1
                                                       i=1

   Now recall from [4] that Theorem 1.3 holds for ∂B(r). In detail, we know
from [4, Cor. 1.3] that
                              cζk (∂B(r)) = dr
where d is the unique nonnegative integer such that

                                 d2 + d     d2 + 3d
                                        ≤k≤         .
                                    2          2
Consequently,
                           cζk (∂B(r))2
                      lim                 = 2r 2 = 4 vol(B(r)).          (65)
                     k→∞         k
    It follows from (64) and (65) and the elementary calculation in [4, Prop.
8.4] that
                                               X N
                               cσk (Y, λ)2
                      lim inf              ≥4       vol(B(ri )).         (66)
                        k→∞         k           i=1

By the construction in Step 2,
                N
                X
                       vol(B(ri )) ≥ vol([−a, 0] × Y, d(es λ)) − ε
                i=1                                                                            (67)
                                                 −a
                                          1−e
                                      =               vol(Y, λ) − ε.
                                            2
Since a > 0 can be arbitrarily large and ε > 0 can be arbitrarily small, (66)
and (67) imply (59). This completes the proof of Proposition 3.1.

                                               27
References
 [1] D. Cristofaro-Gardiner and M. Hutchings, From one Reeb orbit to two,
     arXiv:1202.4839.

 [2] M. Hutchings, The embedded contact homology index revisited , New per-
     spectives and challenges in symplectic field theory, 263–297, CRM Proc.
     Lecture Notes 49, Amer. Math. Soc., 2009.

 [3] M. Hutchings, Taubes’s proof of the Weinstein conjecture in dimension
     three, Bull. AMS 47 (2010), 73–125.

 [4] M. Hutchings, Quantitative embedded contact homology, J. Diff. Geom.
     88 (2011), 231–266.

 [5] M. Hutchings, Recent progress on symplectic embedding problems in four
     dimensions, Proc. Natl. Acad. Sci. USA 108 (2011), 8093–8099.

 [6] M. Hutchings, Lecture notes on embedded contact homology,
     arXiv:1303.5789, to appear in proceedings of CAST summer school, Bu-
     dapest, 2012.

 [7] M. Hutchings, Embedded contact homology as a (symplectic) field theory,
     in preparation.

 [8] M. Hutchings and C. H. Taubes, Gluing pseudoholomorphic curves along
     branched covered cylinders I , J. Symplectic Geom. 5 (2007), 43–137.

 [9] M. Hutchings and C. H. Taubes, Gluing pseudoholomorphic curves along
     branched covered cylinders II , J. Symplectic Geom. 7 (2009), 29–133.

[10] M. Hutchings and C. H. Taubes, Proof of the Arnold chord conjecture
     in three dimensions II , Geometry and Topology 17 (2013), 2601–2688.

[11] P.B. Kronheimer and T.S. Mrowka, Monopoles and three-manifolds,
     Cambridge University Press, 2008.

[12] D. McDuff, The Hofer conjecture on embedding symplectic ellipsoids, J.
     Diff. Geom. 88 (2011), 519–532.

[13] C. H. Taubes, The Seiberg-Witten equations and the Weinstein conjec-
     ture, Geom. Topol. 11 (2007), 2117-2202.

                                    28
[14] C. H. Taubes, The Seiberg-Witten equations and the Weinstein conjec-
     ture II: More closed integral curves for the Reeb vector field , Geom.
     Topol. 13 (2009), 1337-1417.

[15] C. H. Taubes, Embedded contact homology and Seiberg-Witten Floer co-
     homology I , Geometry and Topology 14 (2010), 2497–2581.

[16] C. H. Taubes, Embedded contact homology and Seiberg-Witten Floer co-
     homology II , Geometry and Topology 14 (2010), 2583–2720.

[17] C. H. Taubes, Embedded contact homology and Seiberg-Witten Floer co-
     homology III , Geometry and Topology 14 (2010), 2721–2817.

[18] C. H. Taubes, Embedded contact homology and Seiberg-Witten Floer co-
     homology V , Geometry and Topology 14 (2010), 2961–3000.




                                    29
