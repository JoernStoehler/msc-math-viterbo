---
source: arXiv:0804.0270
fetched: 2025-10-20
---
# On the quantum homology algebra of toric Fano manifolds

                                              On the quantum homology algebra of toric Fano manifolds

                                                                          Yaron Ostrover∗ and Ilya Tyomkin
arXiv:0804.0270v2 [math.SG] 15 Jun 2008




                                                                                     October 26, 2018



                                                                                          Abstract

                                                    In this paper we study certain algebraic properties of the quantum homology alge-
                                                    bra for the class of symplectic toric Fano manifolds. In particular, we examine the
                                                    semi-simplicity of the quantum homology algebra, and the more general property of
                                                    containing a field as a direct summand. Our main result provides an easily-verifiable
                                                    sufficient condition for these properties which is independent of the symplectic form.
                                                    Moreover, we answer two questions of Entov and Polterovich negatively by providing
                                                    examples of toric Fano manifolds with non semisimple quantum homology algebra, and
                                                    others in which the Calabi quasimorphism in non-unique.



                                          1        Introduction.

                                          The quantum homology algebra QH∗ (X, ω) of a symplectic manifold (X, ω) is, roughly
                                          speaking, the singular homology of X endowed with a modified algebraic structure, which
                                          is a deformation of the ordinary intersection product. It was originally introduced by the
                                          string theorists Vafa and Witten [44],[45] in the context of topological quantum field theory,
                                          followed by a rigorous mathematical construction by Ruan and Tian [39] in the symplectic
                                          setting, and by Kontsevich and Manin [26] in the algebra-geometric setting.
                                               Since its introduction in 1991, there has been a great deal of interest in the study of
                                          quantum homology from various disciplines, both by physicists and mathematicians. In
                                          particular, the quantum homology algebra plays an important role in symplectic geometry
                                          where, for example, it is ring-isomorphic to the Floer homology. Recently, the study of
                                          quantum homology had a profound impact in the realm of algebraic geometry, where ideas
                                          from string theory have led to astonishing predictions regarding enumerative geometry of
                                          rational curves. Furthermore, quantum homology naturally arises in string theory, where
                                          it is an essential ingredient in the A-model side of the mirror symmetry phenomenon. We
                                          refer the reader to [33] and [22] and the references within for detailed expositions to the
                                          theory of quantum homology.
                                              In this paper we focus on the following algebraic properties of the quantum homology
                                          algebra. Recall that a finite dimensional commutative algebra over a field is said to be
                                              ∗
                                                  The first named author was supported by NSF grant DMS-0706976.


                                                                                              1
semisimple if it decomposes into a direct sum of fields. A more general property is the
following: a finite dimensional commutative algebra A is said to contain a field as a direct
summand if it splits (as an algebra) into a direct sum A = A1 ⊕ A2 , where A1 is a field and
no assumptions on the algebra A2 are imposed. We wish to remark that there are several
different notions of semi-simplicity in the context of quantum homology (see e.g. [11], [26]).
The semi-simplicity we consider here was first examined by Abrams [1].
    Our main motivation to study the above mentioned algebraic properties of the quantum
homology algebra is the recent works by Entov and Polterovich on Calabi quasimorphisms
and symplectic quasi-states ([13], [14], [15], [16]), in which the algebraic structure of the
quantum homology plays a key role. More precisely, our prime object of interest is the sub-
algebra QH2d (X, ω), i.e. the graded part of degree 2d = dimR X of the quantum homology
QH∗ (X, ω). This subalgebra is finite dimensional over a field K↓ (see Subsection 3.1 for the
definitions). In what follows, we say that QH2d (X, ω) is semisimple if it is semisimple as a
K↓ -algebra.
   The following theorem has been originally proven in the case of monotone symplectic
manifolds in [13] (using a slightly different setting), then generalized by the first named
author in [35] to the class of rational strongly semi-positive symplectic manifolds that satisfy
some technical condition which was eventually removed in [14].

Theorem. Let (X, ω) be a rational1 strongly semi-positive symplectic manifold of dimension
2d such that the quantum homology subalgebra QH2d (X, ω) ⊂ QH∗ (X, ω) is semisimple.
Then X admits a Calabi quasimorphism and a symplectic quasi-state.

    For the definition of Calabi quasimorphisms and symplectic quasi-states, and detailed
discussion of their application in symplectic geometry we refer the reader to [13], [16].
We wish to mention here that other than demonstrating applications to Hofer’s geometry
and C 0 -symplectic topology, Entov and Polterovich used the above theorem to obtain La-
grangian intersection type results. For example, in [6] they proved (together with Biran)
that the Clifford torus in CP n is not displaceable by a Hamiltonian isotopy. In a later
work [15], they proved the non-displaceability of certain singular Lagrangian submanifolds,
a result which is currently out of reach for the conventional Lagrangian Floer homology
technique. We refer the reader to [15] for more details in this direction.
    Very recently, McDuff pointed out that the semi-simplicity assumption in the above
theorem can be relaxed to the weaker assumption that QH2d (X, ω) contains a field as a
direct summand. Moreover, she showed that in contrast with semi-simplicity, this condition
holds true for one point blow-ups of non-uniruled symplectic manifolds such as the standard
symplectic four torus T 4 (see [31] and [14] for details), consequently enlarging the class of
manifolds admitting Calabi quasimorphisms and symplectic quasi-states. Thus, in what
follows we will study not only the semi-simplicity of the quantum homology algebra, but
also the more general property of containing a field as a direct summand.
   1
    It is very plausible that the rationality assumption can be removed due to the recent works of Oh [36],
and Usher [42].



                                                    2
   A different motivation to study the semi-simplicity of the quantum homology alge-
bra is due to a work of Biran and Cornea. In [5] they showed that in certain cases the
semi-simplicity of the quantum homology implies restrictions on the existence of certain
Lagrangian submanifolds. We refer the reader to [5], Subsection 6.5 for more details.
    Finally, a third motivation comes from physics, where in the symplectic toric Fano case
the semi-simplicity of the quantum homology algebra implies that the corresponding N = 2
Landau-Ginzburg model is massive. The physical interpretation is that the theory has
massive vacua and the infrared limit of this model is trivial. See [22] and the references
within for precise definition and discussion.
    Examples of symplectic manifolds with semisimple quantum homology are CP d (see
e.g. [13]); complex Grassmannians; and the smooth complex quadric Q = {z02 + · · · +
       2
zd2 − zd+1 = 0} ⊂ CP d+1 (see [1] for the last two examples). As mentioned above, McDuff
(see [31] and [14]) provides a large class of examples of symplectic manifolds whose quantum
homology contains a field as a direct summand but is not semisimple, by considering the
one point blow-up of a non-uniruled symplectic manifold. Using the Künneth formula for
quantum homology, one can show that both semi-simplicity and the property of containing
a field as a direct summand are preserved when taking products (see [14]).
     Another class of examples are toric Fano 2-folds. Recall that up to rescaling the symplec-
tic form by a constant factor there are exactly five symplectic toric Fano 2-folds: CP 1 ×CP 1 ,
CP 2 , and the blowups of CP 2 at 1, 2 and 3 points. The following theorem is a combination
of results from [35] and [14].

Theorem. If (X, ω) is a symplectic toric Fano 2-fold then QH4 (X, ω) is semisimple.

       In view of the above, Entov and Polterovich posed the following question in [14]:
Question: Is it true that the algebra QH2d (X, ω) is semisimple for any symplectic toric
Fano manifold (X, ω)?
    It is known (see e.g. [24] Corollary 5.12, and [18] Proposition 7.6) that semi-simplicity
holds for generic toric symplectic form. For the sake of completeness, we include this
statement below. More precisely:

Theorem A. Let X be a smooth 2d-dimensional toric Fano variety. Then for a generic2
choice of a toric symplectic form ω on X, the quantum homology QH2d (X, ω) is semisimple.

   However, it turns out that the answer to the question of Entov and Polterovich is
negative. The first counter example exists in (real) dimension eight.

Proposition B. There exists a monotone3 symplectic toric Fano 4-fold (X, ω) whose quan-
tum homology algebra QH8 (X, ω) is not semisimple.
   2
     The space of toric symplectic forms has natural structure of a topological space, and generic here means
that ω belongs to a certain open dense subset in this space.
   3
     Recall that (X, ω) is called monotone if c1 = κ[ω], where κ > 0, and c1 is the first Chern class of X.




                                                     3
    Using Künneth formula we also produce examples of non-monotone symplectic Fano
manifolds (X, ω) with non semisimple quantum cohomology algebras. In particular, there
exists a non-monotone Fano 5-fold (X, ω) with a non semisimple QH10 (X, ω). Notice that
it would be interesting to construct an example of non-decomposable non-monotone sym-
plectic Fano manifold with this property.
    We wish to remark that a toric Fano manifolds X may be equipped with a distinguished
toric symplectic form ω0 , namely the normalized monotone symplectic form corresponding
to c1 (X). This is the unique symplectic form for which the corresponding moment poly-
tope is reflexive (see Section 2). Our second result shows that as far as semi-simplicity is
concerned, the symplectic form ω0 is, in a matter of speech, the worst.
Theorem C. Let X be a toric Fano manifold of (real) dimension 2d, and let ω be a toric
symplectic form on X. If QH2d (X, ω0 ) is semisimple then QH2d (X, ω) is semisimple.

   Inspired by McDuff’s observation we modify the above question of Entov and Polterovich
and ask the following:
Question: Is it true that the algebra QH2d (X, ω) contains a field as a direct summand for
any symplectic toric Fano manifold (X, ω)?
   Currently we do not have an example of a symplectic toric Fano manifold (X, ω) that
does not satisfy this property. Moreover, it seems that no such example exists in low
dimensions. We hope to return to this question in the near future. Meanwhile, we prove
the following analog of Theorem C:
Theorem D. Let X be a toric Fano manifold of (real) dimension 2d, and let ω be a
toric symplectic form on X. If QH2d (X, ω0 ) contains a field as a direct summand, then
QH2d (X, ω) contains a field as a direct summand.

    In Subsection 3.3 we show that the property of QH2d (X, ω) of having a field as a direct
summand is equivalent to the existence of a non-degenerate critical point of a certain (com-
binatorially defined) function WX , called the Landau-Ginzburg superpotential, assigned
naturally to (X, ω). McDuff’s observation and Theorem D reduce the question of the exis-
tence of Calabi quasimorphisms and symplectic quasi-states on a symplectic toric manifold
(X, ω) to the normalized monotone case (X, ω0 ), and hence to the problem of analyzing the
critical points of a function WX , depending only on X and not on the symplectic form. This
can be done easily in many cases. In particular we construct the following new examples of
symplectic manifolds admitting Calabi quasimorphisms and symplectic quasi-states:
Corollary E. Let X be one of the following manifolds: (i) a symplectic toric Fano 3-fold,
(ii) a symplectic toric Fano 4-fold, (iii) the symplectic blow up of CP d at d + 1 general
points. Then X admits a Calabi quasimorphism and a symplectic quasi-state.

    Another byproduct of our method is the following two propositions. The first one,
inspired by McDuff [32], answers a question raised by Entov and Polterovich [13] regarding
the uniqueness of the Calabi quasimorphism. We will briefly recall the definition of a Calabi
quasimorphism in Section 6. For a detailed discussion see [13], [16].

                                             4
Corollary F. Let (X, ω) be the blow up of CP 2 at one point equipped with a symplectic
form ω. If ω(L)/ω(E) < 3, where L is the class of a line on CP 2 , and E is the class of the
exceptional divisor, then there are two different Calabi quasimorphisms on (X, ω).

Remark: Other examples of symplectic manifolds for which the Calabi quasimorphism is
non-unique were constructed by Entov, McDuff, and Polterovich in [6]. We chose to include
the above example here due to the simplicity of the argument. Moreover, we remark that
Corollary F can be easily extended to other toric Fano manifolds.
    Finally, we finish this section with a folklore result, known to experts in the field and
proven in full detail by Auroux (see Theorem 6.1 in [2]). We wish to remark that the results
in [2] are more general (see Proposition 6.8 in [2]), and do not rely on Batyrev’s description
of the quantum homology algebra. However, since by using Proposition (3.3) the proof of
the claim below becomes much simpler, we felt it might be useful to include it here as well.

Corollary G. For a smooth toric Fano manifold X, the critical values of the superpo-
tential WX are the eigenvalues of the linear operator QH 0 (X, ω) → QH 0 (X, ω) given by
multiplication by q −1 c1 (X).

Structure of the paper: In Section 2 we recall some basic definitions and notations
regarding symplectic toric manifolds. In Section 3 we give three equivalent description of
the quantum cohomology of toric Fano manifolds. In section 4 we prove our main results.
For technical reasons it is more convenient for us to use quantum cohomology instead of
homology. In this setting Theorem A becomes Theorem 4.1 and Theorems C and D are
combined together to Theorem 4.3. In Section 5 we prove Proposition B and Corollary E.
In Sections 6 and 7 we prove Corollaries F and G respectively. Finally, in the Appendix we
give a short review on toric varieties.
Acknowledgement: We thank D. Auroux, L. Polterovich, P. Seidel, and M. Temkin for
helpful comments and discussions.


2     Preliminaries, notation, and conventions.

In this section we recall some algebraic definitions and collect all the facts we need regarding
symplectic toric manifolds.


2.1     Algebraic preliminaries

Convention. All the rings and algebras in this paper are commutative with unit element.


2.1.1    Semigroup algebras.

Let G be a commutative semigroup and let R be a ring. The semigroup algebra R[G] is the
R-algebra consisting of finite sums of formal monomials xg , g ∈ G, with coefficients in R,

                                               5
and equipped with the natural algebra operations. For example, if G = Zd then R[G] is the
algebra of Laurent polynomials R[x±1                   ±1               d
                                          1 , . . . , xd ], and if G = Z+ then R[G] is the polynomial
algebra R[x1 , . . . , xd ]. In this paper R will usually be either the field K or the Novikov ring
Λ which are introduced at the end of Subsection 3.1.


2.1.2    Semisimple algebras.

Among the many equivalent definitions of semisimplicity we consider the following:

Definition 2.1. Let F be a field. A finite dimensional F-algebra A is called semisimple if
it contains no nilpotent elements.

     In the language of algebraic geometry (see e.g. [12]), semisimplicity is equivalent to
the affine scheme SpecA being reduced and finite over Spec F, and in particular zero-
dimensional. Notice that a Noetherian zero-dimensional scheme is reduced if and only
if it is regular. If in addition char F = 0 this is equivalent to SpecA being geometrically
regular (i.e., SpecA ⊗F F is smooth). It follows from this geometric description that a finite
dimensional algebra A is semisimple if and only if it is a direct sum of field extensions of
F. Moreover, if char F = 0 then A is semisimple if and only if A ⊗F L is semisimple for any
field extension L/F.
    We say that F-algebra A contains a field as a direct summand if it decomposes as a
F-algebra into a direct sum A = L ⊕ A′ , where L/F is a field extension. Again, in geometric
terms this condition means that the affine scheme SpecA contains a regular point as an
irreducible component.


2.1.3    Non-Archimedean seminorms.

Let F be a field. A non-Archimedean norm is a function |·| : F → R+ satisfying the following
properties: |λµ| = |λ||µ|, |λ + µ| ≤ max{|λ|, |µ|}, and |λ| = 0 if and only if λ = 0. Notice
that the norm |·| defines a metric on F. A field F is called non-Archimedean if it is equipped
with a non-Archimedean norm such that F is complete (as a metric space). One can define
the corresponding non-Archimedean valuation ν : F → R ∪ {−∞} on F by setting4 ν(λ) :=
log |λ|. It satisfies similar properties, i.e. ν(λµ) = ν(λ) + ν(µ), ν(λ + µ) ≤ max{ν(λ), ν(µ)},
and ν(λ) = −∞ if and only if λ = 0.
    Let F be a non-Archimedean field, and let A be an F-algebra. A non-Archimedean semi-
norm on A is a function k · k : A → R+ such that kf gk ≤ kf kkgk, kf + gk ≤ max{kf k, kgk},
and kλf k = |λ|kf k for all λ ∈ F, f, g ∈ A. A seminorm is called norm if the following holds:
kf k = 0 if and only if f = 0. It is well known that if k · k is a non-Archimedean seminorm
and kf k =
         6 kgk then kf +gk = max{kf k, kgk}. Given a non-Archimedean seminorm       p k·k one
can consider the associated spectral seminorm k · ksp defined by kf ksp = limk→∞ kf k k. It
                                                                                    k


   4
     Usually one defines ν(λ) := − log |λ| and ν(0) = ∞, however, we chose the above normalization to make
it compatible with [15] and [34]



                                                    6
is easy to check that k · ksp is a non-Archimedean seminorm on A satisfying kf k ksp = kf kksp
for all k. Notice however, that k · ksp need not be a norm even if k · k is.

Lemma 2.2. Let (F, |·|) be a non-Archimedean algebraically closed field, and let A be a finite
F-algebra equipped with a non-Archimedean norm k · k. Let B ⊆ A be a local F-subalgebra,
m its maximal ideal, and eB ∈ B its unit element. Then B = FeB ⊕ m as F-modules, and
kλeB + gksp = |λ| for all λ ∈ F and g ∈ m.

Proof. The field B/m is a finite extension of F, thus B/m = F since F is algebraically
closed; the decomposition now follows. Notice that B is finite over F thus any element
g ∈ m is nilpotent,
                  qhence kgksp = 0.p Notice that keB k 6= 0 since k · k is a norm, hence
keB ksp = limk→∞ k kekB k = limk→∞ k keB k = 1. Thus kλeB ksp = |λ| > 0 = kgksp for any
0 6= λ ∈ F and g ∈ m, which implies kλeB + gksp = |λ| for all λ ∈ F and g ∈ m.

Corollary 2.3. Let F be a field, A be a finite F-algebra, and set Z = SpecA. Consider a
function f ∈ O(Z) = A and the linear operator Lf : O(Z) → O(Z), defined by Lf (a) := f a.
Then:

 (i) O(Z) = ⊕q∈Z OZ,q ,

 (ii) the set of eigenvalues of Lf is {f (q)}q∈Z , and

 (iii) if F is non-Archimedean and A is equipped with a non-Archimedean norm k · k then
       kf eq ksp = |f (q)| for any q ∈ Z, where eq denotes the unit element in OZ,q .

Proof.    (i) dimF A < ∞ implies dim Z = 0 and O(Z) = ⊕q∈Z OZ,q .

 (ii) It is sufficient to show that the operator Lf |OZ,q : OZ,q → OZ,q has unique eigenvalue
       f (q). Notice that f eq = f (q)eq + g, where g ∈ mq is a nilpotent element. Thus
       Lf |OZ,q − f (q)IdOZ,q is nilpotent, which implies the statement.

 (iii) Notice that f eq = f (q)eq + g, where g ∈ mq ; thus kf eq ksp = |f (q)| by Lemma 2.2.




2.2      Symplectic toric manifolds

Notation. Throughout the paper M denotes a lattice, i.e. a free abelian group of finite
rank d, and N = HomZ (M, Z) its dual lattice. We use the notation MR = M ⊗Z R
and NR = N ⊗Z R for the corresponding pair of dual vector spaces of dimension d. We
shall use the notation TN and TM for the algebraic tori TN = Spec F[M ] = N ⊗Z F∗ and
TM = Spec F[N ] = M ⊗Z F∗ over the base field F.

    Let T = MR /M = N ⊗Z (R/Z) be the compact torus of dimension d with lattice of
characters M and lattice of cocharacters N . A 2d−dimensional symplectic toric manifold
is a closed connected symplectic manifold (X, ω) equipped with an effective Hamiltonian
T -action, and a moment map µ : X → Lie(T )∗ = MR generating (locally) the T -action on

                                               7
X. In other words, for any g ∈ T there is x ∈ X such that g(x) 6= x, and for any ξ ∈ Lie(T )
and x ∈ X we have: dx µ(ξ) = ω(Xξ , ·), where Xξ denotes the vector field induced by ξ
under the exponential map.
     By a well known theorem of Atiyah and Guillemin-Sternberg, the image of the moment
map ∆ := µ(X) ⊂ MR is the convex hull of the images of the fixed points of the action. It
was proved by Delzant [10] that the moment polytope ∆ ⊂ MR has the following properties:
(i) there are d edges meeting at every vertex v (simplicity), (ii) the slopes of all edges
are rational (rationality), and (iii) for any vertex v the set of primitive integral vectors
along the edges containing v is a basis of the lattice M (smoothness). Such a polytope
is called a Delzant polytope. Recall that any polytope can be (uniquely) described as the
intersection of (minimal set of) closed half-spaces with rational slopes. Namely, there exist
n1 , . . . , nr ∈ N = HomZ (M, Z) and λ1 , . . . , λr ∈ R, where r is the number of facets (i.e.
faces of codimension one) of ∆ such that

                        ∆ = {m ∈ MR | (m, nk ) ≥ λk for every k}.                        (2.2.1)

     Moreover, Delzant gave a complete classification of symplectic toric manifolds in terms
of the combinatorial data encoded by a Delzant polytope. In [10] he associated to a Delzant
polytope ∆ ⊂ MR a closed symplectic manifold (X∆       2d , ω ) together with a Hamiltonian
                                                             ∆
T -action and a moment map µ∆ : X∆     2d → M such that µ(X 2d ) = ∆. He showed that
                                                R                  ∆
(X∆ 2d , ω ) is isomorphic (as Hamiltonian T -space) to (X 2d , ω), and proved that two sym-
          ∆
plectic toric manifolds are (equivariantly) symplectomorphic if and only if their Delzant
polytopes differ by a translation and an element of Aut(M ).
     The precise relations between the combinatorial data of the Delzant polytope ∆ and the
symplectic structure of X are as follows: the faces of ∆ of dimension d′ are in one-to-one
correspondence with the closed connected equivariant submanifolds of X of (real) dimension
2d′ , namely to a face α corresponds the submanifold µ−1 (α). In particular to facets of ∆
correspond submanifolds of codimension 2. Let z1 , . . . , zr ∈ H 2 (X, Z) be the Poincaré dual
of the homology classes of D1 , . . . , Dr , where Dk is the submanifold corresponding to the
facet given by (m, nk ) = λk . Then the cohomology class [ω] and the first Chern class c1 (X)
are given by
                                       Xr                        Xr
                           1
                             [ω] = −        λk zk , and c1 (X) =     zk                  (2.2.2)
                          2π
                                     i=1                        i=1

   In what follows it would be convenient for us to adopt the algebraic-geometric point of
view of toric varieties which we now turn to describe.


2.3   Algebraic Toric Varieties.

In this subsection we briefly discuss toric varieties from the algebraic-geometric point of
view. We refer the reader to the appendix of this paper for the definitions, and for a more
detailed discussion of the notions that appear below. For a complete exposition of the
subject see Fulton’s book [19] and Danilov’s survey [9].

                                               8
    Let σ ⊂ NR be a strictly convex, rational, polyhedral cone. One can assign to σ an
affine toric variety Xσ = Spec F[M ∩ σ̌], where σ̌ ⊂ MR is the dual cone and F[M ∩ σ̌] is the
corresponding commutative semigroup algebra. If τ ⊆ σ is a face then Xτ ֒→ Xσ is an open
subvariety. In particular, since σ is strictly convex, the affine toric variety Xσ contains the
torus X{0} = Spec F[M ] = N ⊗Z F∗ = TN as a dense open subset. Furthermore, the action
of the torus on itself extends to the action on Xσ .
    Recall that a collection Σ of strictly convex, rational, polyhedral cones in NR is called
a fan if the following two conditions hold:

  1. If σ ∈ Σ and τ ⊆ σ is a face then τ ∈ Σ.

  2. If σ, τ ∈ Σ then σ ∩ τ is a common face of σ and τ .

A fan Σ is called complete if ∪σ∈Σ σ = NR . One-dimensional cones in Σ are called rays.
Notation. The set of cones of dimension k in Σ is denoted by Σk , and the primitive integral
vector along a ray ρ is denoted by nρ .

    Given a (complete) fan Σ ⊂ NR one can construct a (complete) toric variety XΣ =
∪σ∈Σ Xσ by gluing Xσ and Xτ along Xσ∩τ . Recall that XΣ has only orbifold singularities
if and only if all the cones in Σ are simplicial (in this case it is called quasi-smooth); and
XΣ is smooth if and only if for any cone σ ∈ Σ the set of primitive integral vectors along
the rays of σ forms a part of a basis of the lattice N .
    The torus TN acts on XΣ and decomposes it into a disjoint union of orbits. To a cone
σ ∈ Σ one can assign an orbit Oσ ⊂ Xσ , canonically isomorphic to Spec F[M ∩ σ ⊥ ]. This
defines a one-to-one order reversing correspondence between the cones in Σ and the orbits
in XΣ . In particular orbits of codimension one correspond to rays ρ ∈ Σ and we denote
their closures by Dρ . Thus {Dρ }ρ∈Σ1 is the set of TN -equivariant primitive Weil divisors on
the variety XΣ . We remark that the set {Dρ }ρ∈Σ1 coincides with the set {Di }1≤i≤r in the
setting of the previous subsection.
    For a polytope ∆ ⊂ MR of dimension d one can assign a complete fan Σ and a piecewise
linear strictly convex function F on Σ in the following way: To a face γ ⊆ ∆ we assign the
cone σ being the dual cone to the inner angle of ∆ at γ (see [9] §5.8); and if m is a vertex
of ∆ and σm ∈ Σ is the corresponding cone then F|σm := m. Vice versa, to a pair (Σ, F )
one can assign a polytope

                      ∆F = {m ∈ MR | (m, nρ ) ≥ F (nρ ), for every ρ}.                  (2.3.3)

This gives a bijective correspondence between polytopes of dimension d in MR and pairs
(Σ, F ) as above. It is known (see the Appendix for details) that choosing a piecewise linear
strictly convex function F on Σ as above is equivalent to introducing a symplectic structure
ω on XΣ (such that the torus action is Hamiltonian) together with a moment map. Under
this identification, the polytope ∆F (2.3.3 ) coincides with the polytope ∆ (2.2.1 ) of the
symplectic manifold (XΣ , ω) with the corresponding moment map. As mentioned before,


                                              9
in what follows, it will be more convenient for us to adopt the algebraic point of view and
to consider the pair (XΣ , F ) instead of the symplectic toric manifold (X, ω).
    For a real/rational/integral piecewise linear function F on a fan Σ one can associate a
                                                   P
TN −equivariant R/Q/Z−Cartier divisor D = − ρ∈Σ1 F (nρ )Dρ . Moreover, any R/Q/Z−
Cartier divisor is equivalent to a TN −equivariant R/Q/Z−Cartier divisor of this form.
Integral TN −equivariant Cartier divisors are called T −divisors. It is well known that strictly
convex piecewise linear functions F correspond to ample divisors. Moreover, if F is integral
                                 P
then the Cartier divisor D = − ρ∈Σ1 F (nρ )Dρ corresponds to an invertible sheaf (i.e., a
line bundle) L = OXΣ (D) together with a trivialization φ : L|T → OTN defined up-to the
                                                                  N
natural action of F∗ .
Remark 2.4. For an integral function F as above, the trivialization φ identifies the global
sections of OXΣ (D) with functions on TN , furthermore the following holds

                      H 0 (XΣ , OXΣ (D)) ≃ Span{xm }m∈∆F ∩M ⊂ O(TN ).                     (2.3.4)

    Let F be an integral strictly convex piecewise linear function on Σ. Recall that the
orbits in XΣ ⊂ NR are in one-to-one order reversing correspondence with the cones in Σ,
hence they are in one-to-one order preserving correspondence with the faces of ∆F . Let
γ ⊂ MR be a face of ∆F , let σγ ∈ Σ be the corresponding cone, and let V = O σγ ⊂ XΣ
be the closure of the corresponding orbit. Then V has a structure of a toric variety with
respect to the action of the torus Spec C[M ∩ σγ⊥ ], and the restriction LV of L to V is
an ample line bundle on V ; however, LV has no distinguished trivialization. To define a
trivialization one must pick an integral point p in the affine space Span(γ) (e.g. a vertex
of γ) and this defines an isomorphism between LV and the line bundle associated to the
polytope γ − p ⊂ σγ⊥ .


Toric Fano Varieties and Reflexive Polytopes.

Let ∆ ⊂ MR be a polytope containing 0 in its interior. The dual polytope ∆∗ ⊂ NR is
defined to be
                 ∆∗ = {n ∈ NR | (m, n) ≥ −1, for every m ∈ ∆}.
Notice that its vertices are precisely the inner normals to the facets of ∆. The polytope
∆ ⊂ MR is called reflexive if (i) 0 is contained in its interior, and (ii) both ∆ and ∆∗ are
integral polytopes. Note that if ∆ is reflexive then 0 is the only integral point in its interior.
It is not hard to check (cf. [4]) that ∆ is reflexive if and only if its dual ∆∗ is reflexive.
    A complete algebraic variety is called Fano if its anti-canonical class is Cartier and
                                                 P           P
ample. Recall that if XΣ is Fano and K = − Dρ = − FK (nρ )Dρ is the standard
canonical T -divisor then ∆−FK = ∆F−K is reflexive, here FK is a piecewise linear function
defined by the following property: FK (nρ ) = 1 for any ρ ∈ Σ1 . Moreover, if ∆ is reflexive
                                                                                     P
then there exists a unique toric Fano variety XΣ such that ∆ = ∆FK , where K = − Dρ ,
and FK is as above.



                                               10
    Let XΣ be a toric Fano variety, ∆ = ∆F−K be the reflexive polytope assigned to the
                              P
anticanonical divisor −K =      Dρ , and ∆∗ be the dual reflexive polytope. Consider the
dual toric Fano variety XΣ∗ = XΣ∗ assigned to the polytope ∆∗ . Then the fan Σ coincides
with the fan over the faces of ∆∗ , and the fan Σ∗ is the fan over the faces of ∆.
    Let now X = XΣ and X ∗ = XΣ∗ be a pair of dual toric Fano varieties, and assume that X
is smooth. Then any maximal cone in Σ is simplicial, and is generated by a basis of N ; hence
the facets of the dual polytope ∆∗ are basic simplexes. Thus the irreducible components of
the complement of the big orbit in X ∗ are isomorphic to Pd−1 . Furthermore, the restriction
of the anticanonical linear system OX ∗ (−KX ∗ ) to such a component is isomorphic to the
anti-tautological line bundle OPd−1 (1).
Remark 2.5. Before we finish this subsection we wish to recall the following two facts:
(i) (see [19] section 3.2) the Euler characteristic of a quasi-smooth complete toric variety is
equal to |Σd |, and (ii) (Kushnirenko’s theorem, a particular case of Bernstein’s theorem -
see [19] section 5.3) if D is an ample T -divisor on a toric variety XΣ , and ∆ ⊂ MR is the
corresponding polytope, then the intersection number D d is given by D d = d!Volume(∆),
where the volume is relative to the lattice M .


3     The Quantum Cohomology

Below are three equivalent descriptions of the quantum cohomology of Fano toric varieties.


3.1   Symplectic Definition

We start with a symplectic definition of the quantum homology (and cohomology) of a
2d-dimensional symplectic manifold (X, ω), using Gromov-Witten invariants. We refer the
reader to [33] and the references within for a more detailed exposition. For simplicity,
throughout the text we assume that (X, ω) is semi-positive manifold (see e.g. Subsection
6.4 in [33]). The class of symplectic toric Fano manifolds is a particular example.
   By abuse of notation, we write ω(A) and c1 (A) for the results of evaluation of the
cohomology classes [ω] and c1 on A ∈ H2 (X; Z). Here c1 ∈ H 2 (X; Z) denotes the first
Chern class of X. We denote by K↓ the field of generalized Laurent series over C. More
precisely,
        nX                                                                        o
  K↓ =       aλ sλ | aλ ∈ C, and {λ | aλ 6= 0} is discrete and bounded above in R   (3.1.1)
         λ∈R

Similarly, we define K↑ to be the field of generalized Laurent series where the set {λ | aλ 6= 0}
is discrete and bounded from below in R. In the definition of the quantum homology we shall
use the Novikov ring Λ↓ := K↓ [q, q −1 ]. and in the definition of the quantum cohomology we
use the “dual” ring Λ↑ := K↑ [q, q −1 ]. By setting deg(s) = 0 and deg(q) = 2 we introduce
the structure of graded rings on Λ↓ and Λ↑ .



                                               11
    As a graded module the quantum homology (cohomology) algebra of (X, ω) is defined
to be
             QH∗ (X, ω) = H∗ (X, Q) ⊗Q Λ↓ , QH ∗ (X, ω) = H ∗ (M, Q) ⊗Q Λ↑ .
The grading on QH∗ (X, ω) (respectively on QH ∗ (X, ω)) is given by deg(a⊗sλ q j ) = deg(a)+
2j, where deg(a) is the standard degree of the class a in the homology (cohomology) of
(X, ω). Next we define the quantum product (cf [33]). We start with the quantum homology
QH∗ (X, ω). For a ∈ Hi (X, Q) and b ∈ Hj (X, Q), define (a ⊗ 1) ∗ (b ⊗ 1) ∈ QHi+j−2d (X, ω)
by                                       X
                   (a ⊗ 1) ∗ (b ⊗ 1) =        (a ∗ b)A ⊗ s−ω(A) q −c1 (A) ,
                                        A∈H2S (X)

where (a ∗ b)A ∈ Hi+j−2d+2c1(A) (M, Q) is defined by the requirement that

                     (a ∗ b)A ◦ c = GWA (a, b, c), for all c ∈ H∗ (X, Q).

Here ◦ is the usual intersection index and GWA (a, b, c) denotes the Gromov-Witten invariant
that, roughly speaking, counts the number of pseudo-holomorphic spheres representing
the class A and intersecting with generic representative of each a, b, c ∈ H∗ (X, Q) (see
e.g. [33], [38], and [39] for the precise definition). The product ∗ is extended to the whole
QH∗ (X, ω) by linearity over Λ↓ . Thus, one gets a well-defined commutative, associative
product operation ∗ respecting the grading on QH∗ (X, ω), which is a deformation of the
classical cap-product in singular homology (see [33], [38], [39] [28], and [45]). Note that
the fundamental class [X] is the unity with respect to the quantum multiplication ∗, and
that QH∗ (X, ω) is a finite-rank module over Λ↓ . Moreover, if a, b ∈ QH∗ (X, ω) have graded
degrees deg(a) and deg(b) respectively, then deg((a ⊗ 1) ∗ (b ⊗ 1)) = deg(a) + deg(b) − 2d.
    Due to some technicalities and although the above definition is more geometric, in what
follows we shall mainly use the quantum cohomology. The quantum product in this case is
defined using Poincaré duality i.e., for α, β ∈ H ∗ (X, Q) with Poincaré duals a = PD(α), b =
PD(β) we define
                                                   X
           (α ⊗ 1) ∗ (β ⊗ 1) = PDq (a ∗ b) :=             PD((a ∗ b)A ) ⊗ sω(A) q c1 (A) ,
                                              A∈H2S (X)

where the quantum Poincaré dual map PDq : QH ∗ (X, ω) → QH∗ (X, ω) is the obvious
variation of the standard Poincaré dual given by PDq (α ⊗ sλ q j ) = PD(α) ⊗ s−λ q −j .
     As mentioned in the introduction, our main object of study is the subalgebra QH2d (X, ω),
which is the graded component of degree 2d in the quantum homology algebra QH∗ (X, ω).
It is not hard to check that it is a commutative algebra of finite rank over the field K↓ . The
above mentioned (quantum) Poincaré duality induces an isomorphism between the quan-
tum homology and cohomology (see [33] remark 11.1.16). Hence, in what follows we will
work with the algebra QH 0 (X, ω) over the field K↑ instead of the algebra QH2d (X, ω) over
K↓ .
Convention. From this point on we set K := K↑ and use the Novikov ring Λ := K[q, q −1 ].



                                              12
Remark 3.1. Notice that the field K is a non-Archimedean field with respect to the
                                P
non-Archimedean norm               aλ sλ := 10− inf{λ | aλ 6=0} . It is known that K is algebraically
                                                                                    P
closed. Notice also that the map k · k : QH ∗ (X, ω) → R+ defined by k λ,j aλj sλ q j k =
10− inf{λ | ∃ aλj 6=0} , where aλj ∈ H ∗ (X, C), is a non-Archimedean norm on the quantum co-
homology algebras QH ∗ (X, ω) and QH 0 (X, ω).


3.2    Batyrev’s Description of the Quantum Cohomology

In [3], Batyrev proposed a combinatorial description of the quantum cohomology algebra
of toric Fano manifolds, using a “quantum” version of the “classical” Stanley-Reisner ideal.
This was later proved by Givental in [20], [21]. For a different approach to the proof we
refer the reader to McDuff-Tolman [34] and Cieliebak-Salamon [7].
   Before describing Batyrev’s work let us first briefly recall the definition of the classical
cohomology of toric Fano manifolds. The complete details can be found in [9] §10,11,12,
and [19] section 3.2 and Chapter 5.
    Let Σ be a simplicial fan, and let XΣ be the corresponding toric variety over C. It is
known that any cohomology class has an equivariant representative. Thus, H 2k (XΣ , Q) is
generated as a vector space by the closures of k-dimensional orbits. Notice that any such
closure V is an intersection of some equivariant divisors Dρ with appropriate multiplicity
that depends on the singularity of the XΣ along V . To be more precise, if V = Oσ , σ ∈ Σk ,
                                                   1    Qk
and ρ1 , . . . , ρk are the rays of σ then V = mult(σ)    i=1 Dρi , where mult(σ) denotes the
covolume of the sublattice spanned by nρ1 , . . . , nρk in the lattice Span(σ) ∩ N . Thus we
have a surjective homomorphism of algebras ψ : Q[zρ ]ρ∈Σ1 → H 2∗ (XΣ , Q), where Q[zρ ]ρ∈Σ1
is the polynomial algebra in free variables zρ indexed by the rays ρ ∈ Σ1 .
                                                                        P
    Let xm ∈ C[M ] be a rational function on XΣ . Then div(xm ) = ρ∈Σ1 (m, nρ )Dρ . Thus
P
   ρ∈Σ1 (m, nρ )z     ∈ Ker(ψ) for any m ∈ M . We denote by P (XΣ ) ⊂ Q[zρ ]ρ∈Σ1 the ideal
                   Pρ
generated by ρ∈Σ1 (m, nρ )zρ , m ∈ M . Notice that if ρ1 , . . . , ρk do not generate a cone in
                                   Q
Σ then ∩ki=1 Dρi = ∅, and thus ki=1 zρi ∈ Ker(ψ). We denote by SR(XΣ ) ⊂ Q[zρ ]ρ∈Σ1 the
                                                     Q
Stanley-Reisner ideal, i.e. the ideal generated by ki=1 zρi where ρ1 , . . . , ρk do not generate
a cone in Σ. It is well known that Ker(ψ) = P (XΣ ) + SR(XΣ ), and hence

                                                       Q[zρ ]ρ∈Σ1
                                 H 2∗ (XΣ , Q) =                     .
                                                   P (XΣ ) + SR(XΣ )

    We turn now to Batyrev’s description of the quantum cohomology. We say that the set
of rays ρ1 , . . . , ρk is a primitive collection if ρ1 , . . . , ρk do not generate a cone in Σ while any
                                                                                         Q
proper subset does generate a cone in Σ. Notice that the set of monomials ki=1 zρi assigned
to primitive collections forms a minimal set of generators of SR(XΣ ). The quantum version
of the Stanley-Reisner ideal QSR(XΣ ) is generated by the quantization of the minimal set
of generators above.
    More precisely, assume that we are given a smooth Fano toric variety XΣ , and a piecewise
linear strictly convex function F on Σ defining an ample R-divisor on XΣ . Let C be a


                                                   13
                                   P
primitive collection of rays. Then ρ∈C nρ belongs to a cone σC ∈ Σ, and we assume that
σC is the minimal cone containing it. It is not hard to check that σC does not contain ρ
                                            P            P
for all ρ ∈ C (cf [3]). Since XΣ is smooth, ρ∈C nρ = ρ⊆σC aρ nρ , where aρ are strictly
                                                              Q
positive integers. We define the quantization of the generator ρ∈C zρ to be
                          Y                           Y
                                q −1 s−F (nρ ) zρ −          (q −1 s−F (nρ ) zρ )aρ
                          ρ∈C                         ρ⊆σC

The quantum version of SR(XΣ ) is the ideal QSR(XΣ , F ) ⊂ Λ[zρ ]ρ∈Σ1 generated by the
quantization of the minimal set of generators. We define Batyrev’s quantum cohomology
to be
                          ∗                        Λ[zρ ]ρ∈Σ1
                       QHB  (XΣ , F ; Λ) :=                        ,
                                            P (XΣ ) + QSR(XΣ , F )
and
                       ∗                   ∗
                     QHB (XΣ , F ; K) := QHB (XΣ , F ; Λ) ⊗Λ Λ / hq − 1i.
As mentioned above, the following result was originally proposed by Batyrev [3] and proved
by Givental [20, 21]. For a proof using notation and conventions similar to ours see [34].
Recall that (X, ω) and (XΣ , F ) represents the same symplectic toric Fano manifold as
explained in Subsection 2.3 above.

Theorem 3.2. For a symplectic toric Fano manifold (X, ω) = (XΣ , F ) there is a ring
isomorphism
                         QH ∗ (X, ω) ≃ QHB∗
                                            (XΣ , F ; Λ)                      (3.2.2)

    We wish to remark that the identification (3.2.2 ) may fail without the Fano assumption
(see [8] example 11.2.5.2 and [34]).


3.3   The Landau-Ginzburg Superpotential

Here we present an analytic description of the quantum cohomology algebra for symplectic
toric Fano varieties which arose from the study of the corresponding Landau-Ginzburg
model in Physics [29], [44], [23]. We will follow the works of Batyrev [3], Givental [20], Hori-
Vafa [23], Fukaya-Oh-Ohta-Ono [18], and describe an isomorphism between the quantum
cohomology algebra of a symplectic toric Fano manifold X and the Jacobian ideal of the
superpotential corresponds to the Landau-Ginzburg mirror model of X.
   Let XΣ be a smooth Fano toric variety, and let F be a piecewise linear strictly con-
vex function on Σ defining an ample R-divisor on XΣ . Consider the Landau-Ginzburg
superpotential                          X
                               WF,Σ :=      sF (nρ ) xnρ
                                                 ρ∈Σ1

defined on the torus SpecK[N ]. This function can be considered also as a section of the anti-
canonical line bundle on the dual toric Fano variety XΣ∗ over the field K (see Remark 2.4).
One assigns to such a function the Jacobian ring K[N ]/JWF,Σ , where JWF,Σ denotes the
Jacobian ideal, i.e. the ideal generated by all partial (log-)derivatives of WF,Σ .

                                                      14
Proposition 3.3. If (XΣ , F ) is a rational smooth symplectic toric Fano variety, and WF,Σ
as above then
                     QH ∗ (X, ω) ∼ = QHB ∗
                                           (XΣ , F ; Λ) ∼
                                                        = Λ[N ]/JWF,Σ ,
and in particular
                        QH 0 (X, ω) ∼   ∗
                                    = QHB (XΣ , F ; K) ∼
                                                       = K[N ]/JWF,Σ .

    For the proof of Proposition 3.3 we shall need the following lemma.
Lemma 3.4. Let X = XΣ be a smooth toric Fano variety over the base field K, F be a
piecewise linear strictly convex function on Σ, and W = WF,Σ be the corresponding Landau-
                                                     P
Ginzburg superpotential, or more generally, a section ρ∈Σ1 bρ xnρ of the anticanonical bun-
dle on X ∗ with all bρ 6= 0. Let ZW ⊂ X ∗ = XΣ∗ be the subscheme defined by the ideal sheaf
JW ⊂ OX ∗ , where JW (−KX ∗ ) ⊂ OX ∗ (−KX ∗ ) is generated by all log-derivatives of W .
Then ZW is a projective subsheme of the big orbit TM ⊂ X ∗ of degree |Σd |. In particular it
is zero dimensional, O(ZW ) = K[N ]/JW , and dim O(ZW ) = |Σd |.

Proof of Lemma 3.4. Since XΣ is smooth each irreducible component of XΣ∗ \ TM is
isomorphic to Pd−1 . Recall that such components are in one-to-one correspondence with
the rays of the dual fan Σ∗ , or equivalently with the maximal cones in Σ. Furthermore, if
σ ∈ Σd is a cone and Dσ∗ ≃ Pd−1 is the corresponding component then the restriction of
the anticanonical linear system to such a component OX ∗ (−KX ∗ ) ⊗ ODσ∗ is isomorphic to
OPd−1 (1), and the homogeneous coordinates on Dσ∗ are naturally parameterized by the rays
ρ ⊂ σ. We denote these coordinates by yρ .
    We consider W and its log-derivatives as sections of OX ∗ (−KX ∗ ). Then, ∂m W =
P                    nρ and its restriction to D ∗ is given by
                                                               P
   ρ∈Σ1 (m, nρ )bρ x                            σ                ρ⊂σ (m, nρ )bρ yρ . Clearly the
set of these equations for m ∈ M has no common roots, hence ZW ⊂ TM . But ZW ⊂ XΣ∗
is closed, hence a projective scheme. Thus ZW is zero dimensional.
   By definition ZW is the scheme-theoretic intersection of d sections of OX ∗ (−KX ∗ ), hence
by Kushnirenko’s theorem
                                                     X
        deg ZW = (−KX ∗ )d = d!V olume(∆∗ ) = d!         V olume(∆∗ ∩ σ) = |Σd |,
                                                        σ∈Σd

since ∆∗ ∩ σ is a primitive simplex for any σ ∈ Σd .

Proof of Proposition 3.3. Consider the natural homomorphism

                    ψ : Λ[zρ ]ρ∈Σ1 → Λ[N ], defined by ψ(zρ ) = qsF (nρ ) xnρ .          (3.3.3)

Since XΣ is smooth and projective (hence complete) the fan Σ is complete, and any n ∈ N
is an integral linear combination of vectors nρ , ρ ∈ Σ1 . Thus, ψ is surjective.
    Next we claim that the quantum Stanley-Reisner ideal QSR(XΣ , F ) lies in the kernel
of ψ. Indeed, let C be a primitive collection and let
                         Y                      Y
                            q −1 s−F (nρ ) zρ −   (q −1 s−F (nρ ) zρ )aρ ,
                          ρ∈C                   ρ⊆σC


                                                15
be the corresponding quantum generator. It follows from the definition of ψ that:
         Y                       Y                          P          P
                                                                              a n
       ψ      q −1 s−F (nρ ) zρ −   (q −1 s−F (nρ ) zρ )aρ = x ρ∈C nρ − x ρ⊆σC ρ ρ = 0.
              ρ∈C                     ρ⊆σC
                                                                P
Moreover, ψ sends the ideal P (XΣ ) into JWF,Σ . Indeed, let ρ∈Σ1 (m, nρ )zρ , m ∈ M be a
generator of P (XΣ ). Then:
               X                    X
            ψ(     (m, nρ )zρ ) = q   (m, nρ )sF (nρ ) xnρ = q∂logm WF,Σ ∈ JWF,Σ .
                    ρ∈Σ1                  ρ∈Σ1

                                            ∗ (X , F ; Λ) → Λ[N ]/J
Thus, ψ defines a surjective homomorphism QHB   Σ                  WF,Σ .

   Notice that both algebras QHB ∗ (X , F ; Λ) and Λ[N ]/J
                                     Σ                    WF,Σ are free modules over Λ, and
thus to complete the proof all we need to do is to compare the ranks. On one side:
                             ∗
                     rankΛ QHB (XΣ , F ; Λ) = dimK H ∗ (XΣ , K) = χ(XΣ ) = |Σd |.

On the other side the rank of Λ[N ]/JWF,Σ over Λ is equal to dimK K[N ]/JWF,Σ , which by
Lemma 3.4 equals |Σd |. The proof is now complete.
                                     P
Lemma 3.5. Let X, X ∗ , and W = ρ∈Σ1 bρ xnρ be as in Lemma 3.4. Then the support of
                                                            P
ZW coincides with the set of critical points of the function ρ∈Σ1 bρ xnρ on the torus TM .
Furthermore, a critical point p is non-degenerate if and only if the scheme ZW is reduced
at p.

Proof. We already proved in Lemma 3.4 that ZW is a zero-dimensional subscheme of the
torus TM . Thus p ∈ ZW ⊂ TM if and only if all log-derivatives of W vanish at p if and only
if p is a critical point of W . Notice that p is a non-degenerate critical point of W if and
only if the Hessian is non-degenerate at p, or equivalently, if and only if the differentials
of the log-derivatives of W generate the cotangent space Tp∗ TM . It remains to show that
the latter condition is equivalent to the following: the log-derivatives of W generate the
maximal ideal of p ∈ TM locally, i.e. mp = JW,p = JW OTM ,p , where mp ⊂ OTM ,p denotes the
maximal ideal. Clearly if mp = JW,p then the differentials of the log-derivatives generate5
Tp∗ TM = mp /m2p . To prove the opposite direction we will need Nakayama’s lemma. Indeed,
if the differentials of the log-derivatives of W generate Tp∗ TM then mp = JW,p + m2p , thus
mp · (mp /JW,p ) = mp /JW,p , hence, by Nakayama’s lemma, mp /JW,p = 0, or equivalently
mp = JW,p .
                                           P
Corollary 3.6. For X = XΣ , X ∗ , W = ρ∈Σ1 bρ xnρ , and ZW ⊂ X ∗ = XΣ∗ as in the lemma
the following hold:

 (i) O(ZW ) is semisimple if and only if W has only non-degenerate critical points.

 (ii) O(ZW ) contains a field as a direct summand if and only if W has a non-degenerate
      critical point.
  5
      Recall that if f ∈ OTM ,p then dp f is nothing but the class of f − f (p) modulo m2p .



                                                        16
4    Proof of The Main Results

In this section we prove our main results. We start with Theorem A which follows from the
quantum Poincaré duality described in Subsection 3.1 and the following theorem:

Theorem 4.1. Let XΣ be a smooth toric Fano variety. Then for a generic choice of a toric
symplectic form ω on XΣ the quantum cohomology QH 0 (XΣ , ω) is semisimple.

    The proof follows the arguments in [24] Corollary 5.12, and [18] Proposition 7.6.

Proof of Theorem 4.1. Let X ∗ = XΣ∗ be the dual Fano toric variety and let OX ∗ (−KX ∗ )
be the anti-canonical linear system. Following Remark 2.4 we consider the subspace of
sections Span{xnρ }ρ∈Σ1 ⊂ H 0 (X ∗ , OX ∗ (−KX ∗ )). It has codimension one since XΣ is Fano
and smooth, moreover H 0 (X ∗ , OX ∗ (−KX ∗ )) is generated by Span{xnρ } and the section x0 .
    Consider a strictly convex piecewise linear function F and the associated potential
        P
WF,Σ = ρ∈Σ1 sF (nρ ) xnρ . Let ZWF,Σ be the subscheme of X ∗ defined by the log-derivatives
of WF,Σ as in Lemma 3.4. Then QH 0 (XΣ , ω) is semisimple if and only if the scheme ZWF,Σ
is reduced by Corollary 3.6 and Proposition 3.3.
    Recall that OX ∗ (−KX ∗ ) is ample, furthermore it is easy to see that for any p ∈ TM ⊂ X ∗
the differentials of the global sections of OX ∗ (−KX ∗ ) generate the cotangent space at p.
Thus for a general choice of W ∈ H 0 (X ∗ , OX ∗ (−KX ∗ )) the critical points of W are non-
degenerate, hence ZW is reduced by Lemma 3.5. Moreover the same is true for a general
section W ∈ Span{xnρ } since log-derivatives of x0 are zeroes. Thus there exists a non-zero
                                                                       P
polynomial P ∈ C[Bρ ]ρ∈Σ1 such that ZW is reduced for any W = bρ xnρ with P (bρ ) 6= 0.
    Let now ω be any toric symplectic form on XΣ , and let F be a corresponding piecewise
linear function on Σ. Notice that by varying ω we vary F (nρ ), and any simultaneous small
variation of F (nρ ) is realized by a toric symplectic form. Indeed, the fan Σ is simplicial
thus any simultaneous variation of F (nρ ) is realized by a piecewise linear function, and
since F is strictly convex any small variation gets rise to a strictly convex function. Thus
for a general variation ω ′ of ω all the monomials of P will have different degrees in s, hence
      ′
P (sF (nρ ) ) 6= 0, and we are done.

    By a similar argument one can prove the following lemma:

Lemma 4.2. Let X = XΣ be a (smooth) toric Fano variety, and let X ∗ be the dual toric
Fano variety over the field K. Let V ⊂ H 0 (X ∗ , OX ∗ (−K ∗ )) be a locally closed subvariety
                               P F (nρ ) nρ
defined over C. Assume that       s     x ∈ V for some strictly convex piecewise linear
function F on the fan Σ. Then there exists a rational strictly convex piecewise linear
                                    P F ′ (nρ ) nρ
function F ′ on the fan Σ such that   s        x ∈V.

Proof. The variety V is given by a system of polynomial equations P1 (bρ ) = ... = Pk (bρ ) = 0
for some P1 , . . . , Pk ∈ C[Bρ ]ρ∈Σ1 , and V ⊆ V is open.



                                              17
     Consider a collection of real numbers (Fρ )ρ∈Σ1 . Then Pi (sFρ ) is a formal finite sum of
(real) monomials with coefficients in C. Assume now that Pi (sFρ ) = 0. Then there exists
a system Li of linear equations with integral coefficients such that (Fρ )ρ∈Σ1 is a solution of
                ′
Li , and Pi (sFρ ) = 0 for any solution (Fρ′ )ρ∈Σ1 of the system Li .
           P F (nρ ) nρ
     Since    s      x ∈ V there exists a system L = ∪Li of linear equations with integral
coefficients such that (F (nρ ))ρ∈Σ1 is a solution of L and for any solution (Fρ′ )ρ∈Σ1 the
                   P F ′ nρ
following holds:      s ρ x ∈ V . Thus there exists a rational solution of system L obtained
from the given one by a small perturbation. Similarly to the proof of Theorem 4.1, any
such solution is of the form (F ′ (nρ ))ρ∈Σ1 where F ′ is a rational strictly convex piecewise
                                              ′                        P F ′ (nρ ) nρ
linear function on the fan Σ. Thus Pi (sF (nρ ) ) = 0 for all i, hence   s        x ∈V.

    Recall that when XΣ is Fano toric manifold then there exists a distinguished toric
symplectic form ω0 on XΣ with moment map µ0 , namely the symplectic form corresponding
to c1 (XΣ ), i.e. to the piecewise linear function F0 satisfying F0 (nρ ) = −1 for all ρ ∈ Σ1 . It
is the unique symplectic form for which the corresponding moment polytope is reflexive.
   Using the quantum Poincaré duality once again, Theorems C and D follow from:

Theorem 4.3. Let XΣ be a smooth toric Fano manifold, and let ω be a toric symplectic
form on XΣ . Then
   (i) If QH 0 (XΣ , ω0 ) is semisimple then QH 0 (XΣ , ω) is semisimple.
   (ii) If QH 0 (XΣ , ω0 ) contains a field as a direct summand then so is QH 0 (XΣ , ω).

Proof of Theorem 4.3: Let F and F0 be the piecewise linear strictly convex functions
corresponding to ω and ω0 , and let W and W0 be the Landau-Ginzburg superpotentials
assigned to F and F0 . From Proposition 3.3 it follows that QH 0 (XΣ , ω) ∼       = O(ZW ) =
K[N ]/JW and QH 0 (XΣ , ω0 ) ∼ = O(ZW0 ) = K[N ]/JW0 , where W = WF,Σ and W0 = WF0 ,Σ .
Notice that the loci of sections W ′ ∈ H 0 (XΣ∗ , O(−KXΣ∗ )) for which ZW ′ is zero dimensional
and is not reduced/does not contain a reduced point are locally closed and defined over C.
Thus, by Lemma 4.2, it is sufficient to prove the theorem only for rational symplectic forms
ω. Furthermore, notice that sa 7→ sak is an automorphism of the field K, hence without
loss of generality we may assume that ω is integral. Thus W, W0 ∈ C[s±1 ][N ].
    Next, let Y = SpecC[s±1 ][N ]/JW and Y0 = SpecC[s±1 ][N ]/JW0 , and consider the nat-
ural projections to SpecC[s±1 ] = C∗ . Notice that the fibers of Y and Y0 over s = 1 are
canonically isomorphic since W s=1 = W0 s=1 . We denote these fibers by Yc (“c” stands for
closed). By Lemma 3.4

                        dimC (O(Yc )) = dimK K[N ]/JW0 = |Σd | < ∞,
                                             P
and Y0 = SpecC[s±1 ] × Yc since W0 = s−1         ρ∈Σ1   xnρ and s is invertible in C[s±1 ].
   Consider now the algebras of functions O((Y0 )η ) and O(Yη ) on the generic fibers of
Y0 → SpecC[s±1 ] and Y → SpecC[s±1 ], i.e.

    O((Y0 )η ) := O(Y0 ) ⊗C[s±1] C(s) ≃ O(Yc ) ⊗C C(s), and O(Yη ) := O(Y ) ⊗C[s±1] C(s).

                                               18
Notice that dimC(s) (O(Yη )) = dimC (O(Yc )) = |Σd | < ∞ by Lemma 3.4. Notice also that
QH 0 (XΣ , ω0 ) = O((Y0 )η ) ⊗C(s) K, and QH 0 (XΣ , ω) = O(Yη ) ⊗C(s) K. Thus QH 0 (XΣ , ω0 ) is
semisimple over K (contains a field as a direct summand) if and only if O((Y0 )η ) is semisimple
over C(s) (contains a field as a direct summand) if and only if O(Yc ) is semisimple over C
(contains a field as a direct summand), and QH 0 (XΣ , ω) is semisimple over K (contains a
field as a direct summand) if and only if O(Yη ) is semisimple over C(s) (contains a field as
a direct summand).
    To summarize: all we want to prove is that (i) if O(Yc ) is semisimple over C then O(Yη )
is semisimple over C(s), and (ii) if O(Yc ) contains a field as a direct summand then O(Yη )
contains a field as a direct summand; or geometrically (i) if Yc is reduced then Yη is reduced,
and (ii) if Yc contains a reduced point the Yη contains a reduced point.
   We remark that since we are interested only in the fibers over the generic point and
over s = 1, we can replace C[s±1 ] by its localization at s = 1 denoted by R. By abuse of
notation Y ×SpecC[s±1 ] SpecR will still be denoted by Y . To complete the proof, we shall
need the following two observations:
Claim 4.4. The map Y → SpecR is flat and finite.
Lemma 4.5. Let Y be flat finite scheme over SpecR, and let Yc and Yη be its fibers over
the closed and generic points of SpecR. Then
    (i) If Yc is reduced then Yη is reduced.
    (ii) If Yc contains a reduced point then Yη contains a reduced point.

The theorem now follows.

Proof of Claim 4.4. First let us show flatness. It is sufficient to check flatness locally on
Y . Let us fix a closed point p ∈ Y ⊂ Spec R[N ], hence p ∈ Yc . We denote Spec R[N ] by T .
Let m1 , . . . , md be a basis of M = HomZ (N, Z) and let ∂i be the log derivations defined by
mi . Then JW is generated by {∂i W }di=1 . We claim that the sequence ∂1 W, . . . , ∂d W, s − 1
is a sequence of parameters in the maximal ideal mp ⊂ OT,p . Indeed dimp T = d + 1
and dim Spec(OT,p /(∂1 W, . . . , ∂d W, s − 1)) = 0, since dimC (OT,p /(∂1 W, . . . , ∂d W, s − 1)) =
dimC OYc ,p ≤ dimC O(Yc ) < ∞. Notice that T is regular, hence Cohen Macaulay, thus
∂1 W, . . . , ∂d W, s − 1 is an OT,p −sequence by [30] Theorem 17.4 (iii). Then the local algebra
OY,p = OT,p /(∂1 W, . . . , ∂d W ) is Cohen-Macaulay by [30] Theorem 17.3 (ii), and dimp Y = 1.
Flatness at p now follows from [30] Theorem 23.1, indeed Y is Cohen-Macaulay at p of
dimension 1, SpecR is regular of dimension 1, and the fiber over s = 1 has dimension 0.
   Remark that flatness of Y over SpecR implies (and in fact is equivalent to) the following
equivalent properties: s − 1 is not a zero divisor in O(Y ), and the natural map O(Y ) →
O(Yη ) is an embedding. In what follows we shall use these properties many times.
   Next we turn to show that O(Y ) is finite R-module. Let g1 , . . . , gl , l = |Σd |, be a basis
of O(Yc ) and let f1 , . . . , fl be its lifting to O(Y ) ⊂ O(Yη ). We claim that f1 , . . . , fl freely
                                                                                        P
generate O(Y ) as an R-module. Let λi (s) ∈ C(s) be elements such that 0 =                   λi (s)fi ∈


                                                  19
O(Yη ). If not all λi (s) are equal to zero, then there exists k such that µi (s) = (s−1)k λi (s) ∈
                                                     P                     P
R for all i and µi (1) 6= 0 for some i. Then            µi (s)fi = 0 hence   µi (1)gi = 0 which is a
contradiction. Thus f1 , . . . , fl ∈ O(Yη ) are linearly independent, and since dimC(s) O(Yη ) =
dimC O(Y1 ) = |Σd | = l, they form a basis of O(Yη ) over C(s). It remains to show that
f1 , . . . , fl generate O(Y ) as R-module. Let 0 6= f ∈ O(Y ) ⊂ O(Yη ) be any element then
        P
f =          λi (s)fi for some λi (s) ∈ C(s). As before, if not all the coefficients λi (s) ∈ R then
there exists k > 0 such that µi (s) = (s − 1)k λi (s) ∈ R for all i and µi (1) 6= 0 for at least
                    P
one i. Thus             µi (1)gi 6= 0 is the class of (s − 1)k f in O(Yc ) which is zero. This is a
contradiction, hence O(Y ) is a flat finite R-module.

Proof of Lemma 4.5. First, notice that the natural map O(Y ) → O(Yη ) is an embedding
since O(Y ) is flat over R. Second, recall that a flat finite module over a local ring is free,
thus O(Y ) ≃ Rl as an R-module, and O(Yη ) ≃ C(s)l as a C(s)-module (vector space); hence
for any 0 6= f ∈ O(Yη ) there exists a minimal integer k such that (s − 1)k f ∈ O(Y ).
    (i): Assume by contradiction that Yη is not reduced. Then there exists a nilpotent
element 0 6= f ∈ O(Yη ). Let k be the minimal integer such that (s − 1)k f ∈ O(Y ). Then
0 6= (s − 1)k f is a nilpotent and its class in O(Yc ) is not zero. Thus, we constructed a
non-zero nilpotent in O(Yc ), which is a contradiction.
    (ii): Recall that if Z = SpecA, and A is a finite dimensional algebra over a field then
A = O(Z) = ⊕q∈Z OZ,q as algebras, where OZ,q is the localization of O(Z) at q (Chinese
remainder theorem). Furthermore any element in the maximal ideal mZ,q ⊂ OZ,q is nilpo-
tent. Thus O(Yc ) = ⊕q∈Yc OYc ,q as algebras, and O(Yη ) = ⊕ǫ∈Yη OYη ,ǫ as algebras (hence as
O(Y )-modules).
    Assume that q ∈ Yc is a reduced point. Then q ∈ Y is a closed point and OY,q →
OYc ,q = C is a surjective homomorphism from a local ring with kernel generated by s − 1,
hence mY,q = (s − 1)OY,q . Tensoring O(Yη ) = ⊕ǫ∈Yη OYη ,ǫ with OY,q over O(Y ) we obtain
the following decomposition: O(Yη ) ⊗O(Y ) OY,q = ⊕ǫ∈Yη (OYη ,ǫ ⊗O(Y ) OY,q ). To finish the
proof it is sufficient to show that (a) O(Yη ) ⊗O(Y ) OY,q is a field, and (b) OYη ,ǫ ⊗O(Y ) OY,q
is either zero or OYη ,ǫ .
     For (a), notice that by Nakayma’s lemma ∩k∈N mkY,q = 0. Thus, any element in OY,q is
of the form u(s − 1)k for some integer k ≥ 0 and some invertible element u ∈ OY,q . Next,
note that s − 1 ∈ OY,q is not a nilpotent element since otherwise it would be a zero divisor
in O(Y ), and this contradicts the flatness of Y . Thus OY,q is an integral domain (in fact
it is a DVR) with field of fractions (OY,q )s−1 (localization of OY,q with respect to s − 1).
Hence O(Yη ) ⊗O(Y ) OY,q = C(s) ⊗R OY,q = (OY,q )s−1 is a field.
    For (b), let m ⊂ O(Y ) ⊂ O(Yη ) be the maximal ideal of q ∈ Y and let n ⊂ O(Yη )
be the maximal ideal of ǫ. If q belongs to the closure of ǫ then O(Y ) \ m ⊆ O(Yη ) \ n,
hence OYη ,ǫ ⊗O(Y ) OY,q = OYη ,ǫ . If q does not belong to the closure of ǫ then there exists
f ∈ O(Y ) \ m such that f ∈ n. Thus 1 ⊗ f ∈ OYη ,ǫ ⊗O(Y ) OY,q must be invertible and
nilpotent at the same time (any element in nOYη ,ǫ is nilpotent!), hence OYη ,ǫ ⊗O(Y ) OY,q = 0
and we are done.


                                                20
5     Examples and Counter-Examples

In this section we prove Proposition B and Corollary E. We first provide an example of a
polytope ∆ such that the quantum homology subalgebra QH8 (X∆ , ω0 ) of the corresponding
(complex) 4-dimensional symplectic toric Fano manifold X∆ is not semisimple. Here ω0 is
the distinguished (normalized) monotone symplectic form on X∆ .
     We start by making the identification
                       Lie(T )∗ = MR ≃ Rd , Lie(T ) = NR ≃ (Rd )∗ ≃ Rd                            (5.1)
For technical reasons, it would be easier for us to describe the vertices of the dual polytope
∆∗ , that are the inward-pointing normals to the facets of ∆. Let
          ∆∗ = Conv{e1 , e2 , e3 , e4 , −e1 + e4 , −e2 + e4 , e2 − e4 , −e2 , −e4 , −e3 − e4 },
where {e1 , e2 , e3 , e4 } is the standard basis of R4 . A straightforward computation, whose
details we omit (see remark below), shows that ∆ is a Fano Delzant polytope. We denote
by (X∆ , ω0 ) the corresponding symplectic toric Fano manifold equipped with the canonical
symplectic form ω0 .
Remark 5.1. Toric Fano 4-manifolds are completely classified (see e.g., [3], [40]). We
refer the reader to the software package “PALP ” [27] with which all the combinatorial
data of the 124 Toric Fano 4-dimensional polytopes can be explicitly computed. The above
example ∆ is the unique reflexive 4-dimensional polytope with 10 vertices, 24 Facets, 11
integer points, and 59 dual integer points (the “PALP” search command is: “class.x -di
x -He EH:M11V10N59F24L1000”), and it is listed among the 124 examples in the web-
page “http://hep.itp.tuwien.ac.at/∼kreuzer/CY/math/0702890/”. In Batyrev’s classifica-
tion [3], X∆ appears under the notation U8 as example number 116 in section 4.

     The corresponding Landau-Ginzburg super potential W : C[x±    ± ± ±
                                                              1 , x2 , x3 , x4 ] → C is given
by
                                                1   1   1    x4 x4 x2
                 W = x1 + x2 + x3 + x4 +          +   +    +   +  +
                                                x2 x4 x3 x4 x1 x2 x4
The partial derivatives are
            x4            x4 + 1 1            1              1  1 x2 x3 + x3 + 1
Wx1 = 1−     2 , Wx2 = 1−    2  + , Wx3 = 1−     2 , Wx4 = 1+ + −
            x1              x2   x4          x4 x3           x1 x2     x3 x24
It is easy to check that x0 = (−1, −1, −1, 1) is a        critical point of W . On the other hand,
the Hessian of W at the point x0
                                                                          
                                              −2          0  0 −1
                                            0            −4 0 −2          
                                                                          
                         Hess(WX∆ (z0 )) =                                
                                            0            0 −2 1           
                                              −1          −2 1 −2
has rank 3 and hence x0 is a degenerate critical point. Thus, it follows from Proposition 3.3
and Corollary 3.6 (i) that QH 0 (X∆ , ω0 ) is not semisimple. From the quantum Poincaré
duality described in Subsection 3.1 we deduce that QH8 (X∆ , ω0 ) is not semisimple and
complete the proof of Proposition B.

                                                   21
Remark 5.2. By taking the product X∆ ×P1 equipped with the symplectic form ω0 ⊗αωP1 ,
where α >> 1 and ωP1 is the standard symplectic form on P1 , we obtain non-monotone
symplectic manifolds with non semisimple quantum homology subalgebra QH10 (X∆ × P1 ).

    We now turn to sketch of proof of Corollary E. Note that the combination of McDuff’s
observation, Theorem D, and Corollary 3.6 (ii) reduces the question of the existence of a
Calabi quasimorphism and symplectic quasi-states on a symplectic toric manifold (X, ω) to
finding a non-degenerate critical point of the Landau-Ginzburg superpotential correspond-
ing to (X, ω0 ), where ω0 is the canonical symplectic form on X.
    We start with the case of the symplectic blow up of Pd at d + 1 general points. After
choosing homogeneous coordinates in an appropriate way we may assume that the d + 1
points are the zero-dimensional orbits of the natural torus action, hence the blow up admits
a structure of a toric variety. The corresponding superpotential (in the monotone case) is
given by
                                 Xd       Xd       Yd       Yd
                                               1                1
                            W =      xi +        +     xi +
                                              xi                xi
                                    j=1       j=1        j=1     j=1

It is easy to check that (−1, . . . , −1) is a non-degenerate critical point.
   Similarly, for toric Fano 3-folds and 4-folds, one can directly check (preferably using a
computer) that the corresponding superpotentials have non-degenerate critical points.6


6       Calabi quasimorphisms

The group-theoretic notion quasimorphism was originally introduced with connection to
bounded cohomology theory and since then became an important tool in geometry, topology
and dynamics (see e.g. [25]). In the context of symplectic geometry, Entov and Polterovich
constructed certain homogeneous quasimorphisms, called “Calabi quasimorphism”, and
showed several applications to Hofer’s geometry, C 0 -symplectic topology, and Lagrangian
intersection theory (see e.g. [13], [16]).
    We recall that a real-valued function Π on a group G is called a homogeneous quasimor-
phism if there is a universal constant C > 0 such that for every g1 , g2 ∈ G:

        |Π(g1 g2 ) − Π(g1 ) − Π(g2 )| ≤ C, and Π(g k ) = kΠ(g) for every k ∈ Z and g ∈ G.

                                                                                       ]
    In [13], Entov and Polterovich constructed quasimorphisms on the universal cover Ham(X)
of the group of Hamiltonian diffeomorphisms of a symplectic manifold X using Floer the-
ory. More precisely, by using spectral invariants which were defined by Schwarz [41] in the
aspherical case, and by Oh [37] for general symplectic manifolds (see also Usher [42]). These
                                                 ]
invariants are given by a map c : QH ∗ (X) × Ham(X)        → R. We refer the reader to [37]
and [33] for the precise definition of the spectral invariants and their properties.
    6
    The combinatorial data required to preform such a computation can be found e.g. within the database
“http://hep.itp.tuwien.ac.at/∼kreuzer/CY/math/0702890/”


                                                    22
                                                                         ]
    Following [13], for an idempotent e ∈ QH 0 (X, ω) we define Qe : Ham(X)        → R by
                                               ek )
                         e                c(e, φ       e  ]
Qe = c(e, ·), where c(e, φ) = lim inf k→∞ k for all φ ∈ Ham(X). Entov and Polterovich
showed that if eQH 0 (X, ω) is a field then Qe is a homogenous quasimorphism (see [13] for
the monotone case and [35], [16] for the general case). Moreover, Qe satisfies the so called
Calabi property, which means, roughly speaking, that “locally” it coincides with the Calabi
homomorphism (see [13] for the precise definition and proof). A natural question raised
in [13] asking whether such a quasimorphism is unique.
    Our goal in this section is to prove Corollary F which shows that the answer to the
question above is negative. For this we will need some preparation. We start with the
following general property of the spectral invariants (see [37],[13],[33]): for every 0 6= a ∈
                                       ]
QH ∗ (X, ω) and γ ∈ π1 (Ham(X)) ⊂ Ham(X)        the following holds: c(a, γ) = c(a∗S(γ), 1l) =
log ka ∗ S(γ)k , where S(γ) ∈ QH 0 (X, ω) is the Seidel element of γ (see e.g. [33] for the
definition), and k · k is the non-Archimedean norm discussed in Remark 3.1. Thus, for every
idempotent e ∈ QH 0 (X, ω) and γ ∈ π1 (Ham(X)), we have

                                  Qe (γ) = log ke ∗ S(γ)ksp ,                              (6.1)

where k·ksp is the corresponding non-Archimedean spectral seminorm (cf. subsection 2.1.3).
   Let now (XΣ , ω) be a symplectic toric Fano manifold, and F be a corresponding strictly
convex piecewise linear function on Σ. Consider the homomorphisms ι : N → K[N ]/JW ,
W = WF,Σ , given by the composition
                                                   S
              N = π1 (TN ) −→ π1 (Ham(XΣ )) −→ QH 0 (XΣ , ω) ≃ K[N ]/JW ,

where S is the Seidel map (see e.g [33]). By translating a result of McDuff and Tolman (see
Theorem 1.10 and Section 5.1 in [34] and [33] page 441) to the Landau-Ginzburg model
using (3.3.3 ), one obtains an explicit formula for ι, namely ι(n) = xn . To any critical
point p ∈ ZW one can assign the unit element ep ∈ OZW ,p , which is an idempotent in
O(ZW ) ≃ QH 0 (XΣ , ω). Furthermore, ep O(ZW ) = OZW ,p is a field if and only if p is a
non-degenerate critical point of W . Thus it is sufficient to find two non-degenerate critical
points of the superpotential p, p′ ∈ ZW and n ∈ N such that |xn (p)| =   6 |xn (p′ )|, thanks to
Corollary 2.3 and (6.1 ).
    Let (XΣ , F ) be the blow up of P2 at one point equipped with a strictly convex piecewise
linear function F , or equivalently, with a symplectic form ω and a moment map µ. After
adding a global linear function to F (this operation changes µ, but does not change ω) we
may assume that F (1, 0) = 0, F (0, 1) = 0, F (0, −1) = β − α, and F (−1, −1) = −α, where
α > β > 0. It is easy to check that QH 0 (XΣ , ω0 ) is semisimple, since the superpotential W0
has only non-degenerate critical points. Thus QH 0 (XΣ , ω) is semisimple by Theorem C,
and W has only non-degenerate critical points.
   Recall that the fan Σ has four rays generated by (1, 0), (0, 1), (0, −1), (−1, −1). Set
x1 = x(−1,0) and x2 = x(0,1) . Then W = x−1 1 + x2 + s
                                                        β−α x−1 + s−α x x−1 , and the scheme
                                                              2        1 2
ZW of its critical points is given by −x−1
                                         1  + s −α x x−1 = x − sβ−α x−1 − s−α x x−1 = 0, or
                                                    1 2      2        2          1 2
               4    α
equivalently, x1 − s x1 − s  (β+α)                −α  2
                                   = 0 and x2 = s x1 . Assume for simplicity that α, β ∈ Q.

                                              23
Notice that the Newton diagram of x41 − sα x1 − s(β+α) = 0 has two faces if and only if
α < 3β; otherwise it has unique face. It is classically known that solutions of such equation
correspond to the faces of the Newton diagram; each solution can be written as a Puiseux
series in s with non-Archimedean valuation −l, where l is the slope of the corresponding
face; and the number of solutions (counted with multiplicities) corresponding to a given
face is equal to the change of x1 along the face. Thus if α < 3β and n = (−1, 0) then there
exist non-degenerate critical points p, p′ ∈ ZW such that |xn (p)| = 103/α 6= 101/β = |xn (p′ )|.
Notice that ω(L)/ω(E) = α/β. Corollary F now follows.


7    The Critical Values of the Superpotential

Let (XΣ , F ) be a smooth toric Fano variety equipped with a strictly convex piecewise
linear function F , or equivalently, with a symplectic form ω and a moment map µ. Recall
                                                                                 P
that c1 (XΣ ) in Batyrev’s description of the (quantum) cohomology is given by ρ∈Σ1 zρ .
Thus, using (3.3.3 ) to identify Batyrev’s description with the Landau-Ginzburg model, one
                                           P
obtains the following formula: c1 (XΣ ) = ρ∈Σ1 qsF (nρ ) xnρ = qW , W = WF,Σ ; hence

                         q −1 c1 (XΣ ) = W ∈ K[N ]/JW = QH 0 (XΣ , ω).

Thus the set of critical values of the superpotential W is equal to the set of eigenvalues of
multiplication by q −1 c1 (XΣ ) on QH 0 (XΣ , ω) by Corollary 2.3; which proves Corollary G.


Appendix: Toric varieties.

Here we shortly summarize the part of the theory of toric varieties relevant to our paper.
We recall the basic definitions and some fundamental results (without proofs). The detailed
development of the theory can be found in Fulton’s book [19] and in Danilov’s survey [9].
    As before, throughout the appendix M denotes a lattice of rank d, N = HomZ (M, Z)
denotes the dual lattice, and MR = M ⊗Z R and NR = N ⊗Z R denote the corresponding
pair of dual vector spaces.


Definition of toric varieties and orbit decomposition.

The references for this subsection are [9] §1, 2, 5, and [19] sections 1.2-1.4, 2.1, 2.2, 2.4, 3.1.
    A subset σ ⊂ NR is called a rational, polyhedral cone if σ is a positive span of finitely
many vectors ni ∈ N , i.e. σ = SpanR+ {n1 , . . . , nk }, ni ∈ N . It is not hard to check that σ
is a rational, polyhedral cone if and only if there exist m1 , . . . , ml ∈ M ⊂ Hom(NR , R) such
that σ = ∩li=1 m−1
                 i (R+ ). A rational, polyhedral cone σ is called strictly convex if it contains
no lines, i.e. σ ∩ (−σ) = {0}. For a rational, polyhedral cone σ ⊂ NR we define the dual
cone σ̌ to be σ̌ = {m ∈ MR |(m, n) ≥ 0 ∀n ∈ σ}, which is again rational and polyhedral.
A face τ of a rational, polyhedral cone σ ⊂ NR is defined to be the intersection of σ with


                                                24
a supporting hyperplane, i.e. τ = σ ∩ Ker(m) for some m ∈ MR . It is easy to see that
a face of a (strictly convex) rational, polyhedral cone is again a (strictly convex) rational,
polyhedral cone. Faces of codimension one are called facets.
    For a strictly convex, rational, polyhedral cone σ one can assign the commutative semi-
group M ∩ σ̌. Notice that since σ̌ is rational and polyhedral this semigroup is finitely
generated, hence the semigroup algebra F[M ∩ σ̌] is also finitely generated. We define affine
toric variety Xσ over F to be Xσ = Spec F[M ∩ σ̌]. If τ ⊆ σ is a face then Xτ ֒→ Xσ is an
open subvariety. In particular, since σ is strictly convex, the affine toric variety Xσ contains
the torus X{0} = Spec F[M ] = N ⊗Z F∗ = TN as a dense open subset. Furthermore, the
action of torus on itself extends to the action on Xσ .
    A collection Σ of strictly convex, rational, polyhedral cones in NR is called a fan if the
following two conditions hold:

   1. If σ ∈ Σ and τ ⊆ σ is a face then τ ∈ Σ.

   2. If σ, τ ∈ Σ then σ ∩ τ is a common face of σ and τ .

A fan Σ is called complete if ∪σ∈Σ σ = NR . The set of cones of dimension k in Σ is denoted
by Σk , and one-dimensional cones in Σ are called rays. The primitive integral vector along
a ray ρ is denoted by nρ .
     Given a (complete) fan Σ one can construct a (complete) toric variety XΣ = ∪σ∈Σ Xσ
by gluing Xσ and Xτ along Xσ∩τ . Recall that XΣ has only orbifold singularities if and only
if all the cones in Σ are simplicial (in this case it is called quasi-smooth); and XΣ is smooth
if and only if for any cone σ ∈ Σ the set of primitive integral vectors along the rays of σ
forms a part of a basis of the lattice N .
    The torus TN acts on XΣ and decomposes it into a disjoint union of orbits. To a cone
σ ∈ Σ one can assign an orbit Oσ ⊂ Xσ , canonically isomorphic to Spec F[M ∩ σ ⊥ ]. This
defines a one-to-one order reversing correspondence between the cones in Σ and the orbits
in XΣ . In particular orbits of codimension one correspond to rays ρ ∈ Σ and we denote
their closures by Dρ . Thus {Dρ }ρ∈Σ1 is the set of TN -equivariant primitive Weil divisors7
on the variety XΣ .


Line bundles on toric varieties.

The references for this subsection are [9] §6, 5.8, and [19] sections 3.4, and 1.5.
   7
     Recall that if X is a singular variety then one must distinguish between Weil divisors (i.e. formal
finite sums of irreducible subvarieties of codimension one) and Cartier divisors (i.e. global sections of
             ∗    ∗
the sheaf KX   /OX  , or equivalently, invertible subsheaves(=line subbundles) of K, where K denotes the
sheaf of rational functions on X). There is a natural homomorphism Cartier(X) → W eil(X) and the
corresponding homomorphism between the class groups of divisors P ic(X) → Cl(X), however these maps
in general need not be surjective or injective, but for smooth varieties these are isomorphisms. For any
toric variety X these maps are injective, since X is normal. If in addition X is quasi-smooth then at least
P ic(X) ⊗Z Q → Cl(X) ⊗Z Q is an isomorphism.


                                                    25
    Let Σ be a fan in NR and let XΣ be the corresponding toric variety. Let L be an algebraic
line bundle on XΣ . By a trivialization of L we mean an isomorphism φ : L|T → OTN
                                                                                  N
considered up-to the natural action of F∗ . Recall that any algebraic line bundle on a torus
is trivial, hence any algebraic line bundle L on XΣ can be equipped with a trivialization.
To a pair (L, φ) one can assign a piecewise linear integral function F on the fan Σ (i.e. a
function F such that F|σ is linear for any σ ∈ Σ and F (N ) ⊂ Z). This defines a bijective
homomorphism between the group (with respect to the tensor product) of pairs (L, φ) and
the additive group of piecewise linear functions F as above:
                                             X
                                  F ←→ O(−        F (nρ )Dρ ).
                                              ρ∈Σ1

Furthermore, a change of the trivialization corresponds to adding a global integral linear
function to F . In the language of divisors one can rephrase the above correspondence as
follows: real/rational/integral piecewise linear functions on the fan Σ are in one-to-one
correspondence with R/Q/Z-Cartier TN -equivariant divisors. Such divisors will be called
T -divisors.
    Let (L, φ) be a T −divisor, and let F be a corresponding function. Then L is globally
generated if and only if F is convex (i.e. F (tn + (1 − t)n′ ) ≥ tF (n) + (1 − t)F (n′ ) for all
n, n′ ∈ NR and 0 ≤ t ≤ 1), and L is ample if and only if F is strictly convex (i.e. F is
convex, and its maximal linearity domains are cones in Σ). Let us now describe the global
sections of L in terms of F . Any section is completely determined by its restriction to the
big orbit which can be identified by φ with an element of F[M ]. Under this identification
the set of global sections of L is canonically isomorphic (up-to the action of F∗ ) to the vector
space SpanF {xm }m∈M ∩∆F where ∆F = ∆(L,φ) = {m ∈ MR | (m, nρ ) ≥ F (nρ ) for every ρ}.
If one changes the trivialization then ∆F is translated by the corresponding element of M .
    Notice that if L is ample then one can reconstruct the fan Σ from the polytope ∆F .
Namely, cones in Σ are in one-to-one order reversing correspondence with the faces of ∆F .
To a face γ ⊆ ∆F we assign the cone σ being the dual cone to the inner angle of ∆F at γ
(see [9] §5.8). Furthermore, if m is a vertex of ∆F and σm ∈ Σ is the corresponding cone,
then F|σm = m. Thus F can also be reconstructed from the polytope ∆F .
    Recall that the orbits in XΣ are in one-to-one order reversing correspondence with the
cones in Σ, hence they are in one-to-one order preserving correspondence with the faces of
∆F . Let γ be a face of ∆F , let σγ ∈ Σ be the corresponding cone, and let V = O σγ be the
closure of the corresponding orbit. Then V has a natural structure of a toric variety, and the
restriction of L to V is an ample line bundle on V defined by the polytope γ −p ⊂ σγ⊥ , where
p ∈ γ is any fixed vertex (the restriction of a trivialized bundle is no longer a trivialized
bundle, this is the reason why one must choose p).


Symplectic structure.

Throughout this subsection the base field is F = C. Given an ample T -divisor (L, φ) on a
toric variety XΣ , one can assign to it a symplectic form ωL,φ in the following way: first notice

                                               26
that φ defines a distinguished (up-to the action of a symmetric group and up-to a common
multiplicative factor) basis in H 0 (XΣ , L⊗r ) for any r. Let XΣ ֒→ P = P(H 0 (XΣ , L⊗r )∗ )
be the natural embedding (where r is assumed to be large enough). Recall that projective
spaces have canonical symplectic structures provided by the Fubini-Study forms. Now we
simply pull back the Fubini-Study symplectic form of volume 1 from P to XΣ , and since it
is invariant under the action of the symmetric group, we get a well defined symplectic form
on XΣ . To make this construction independent of r and to make the moment polytope
compatible with ∆(L,φ) all we have to do is to multiply the form by 2π    r . We denote this
normalized symplectic form by ωL,φ or ωF if F is the strictly convex piecewise linear function
associated to (L, φ). Thus (L, φ) defines the structure of a symplectic toric manifold on XΣ .
Furthermore, the action of the compact torus T = N ⊗Z S 1 ⊂ N ⊗Z C∗ = TN is Hamiltonian.
Such a manifold admits a moment map µωF : X → Lie(T )∗ = MR . In our case µωF is defined
by                                         P          m    2
                                             m∈∆ |x (p)| m
                               µωF (p) = P F           m     2
                                                               ,
                                               m∈∆F |x (p)|

and its image is the polytope ∆F = ∆(L,φ) (cf. [19] sections 4.1 and 4.2).


Differential Log-forms and the Canonical Class.

The references for this subsection are [9] §15, and [19] sections 4.3.
    Let Σ be a fan in NR and let XΣ be the corresponding toric variety. By a log-form we
mean a rational differential 1-form having at worst simple poles along the components of
XΣ \ TN . Recall that the sheaf Ω1XΣ (log) of log-forms is trivial vector bundle canonically
                                                               m
isomorphic to M ⊗Z OXΣ (we assign to m ∈ M the form dx       xm ). Moreover there exists an
exact sequence 0 → Ω1XΣ → Ω1XΣ (log) → ⊕ρ∈Σ1 ODρ → 0, where the last map is the sum of
                                                                  P
residues. It follows from the exact sequence above that KΣ = − ρ∈Σ1 Dρ is the canonical
(Weil) divisor on XΣ . If canonical divisor is Q−Cartier (e.g. XΣ is quasi-smooth) then the
canonical divisor corresponds to the rational piecewise linear function FK defined by the
following property: FK (nρ ) = 1 for any ρ ∈ Σ1 .
    The dual notion to a log-differential form is a log-derivative. Log-vector fields also
form a trivial vector bundle canonically isomorphic to N ⊗Z OXΣ , namely to any n ∈ N
corresponds the log-derivative ∂n defined by ∂n xm = (m, n)xm . The notion of log-derivative
will be useful in this paper to make proofs coordinate free.


References

[1] Abrams, L. The quantum Euler class and the quantum cohomology of the Grassmanni-
   ans, Israel J. Math. 117 (2000), 335-352.

[2] Auroux, D. Mirror symmetry and T-duality in the complement of an anticanonical di-
   visor, J. Gökova Geom. Topol. 1 (2007), 51-91.



                                              27
[3] Batyrev, V.V. Quantum cohomology rings of toric manifolds, Journées de Géométrie
   Algébrique d’Orsay (Orsay, 1992). Astérisque No. 218 (1993), 9-34.

[4] Batyrev, V.V. Dual polyhedra and mirror symmetry for Calabi-Yau hypersurfaces in
   toric varieties, J. Algebr. Geom. 3 (1994), 493-535.

[5] Biran, P., Cornea, O. Quantum structures for Lagrangian submanifolds, arXiv:0708.4221

[6] Biran, P., Entov, M., Polterovich L. Calabi quasimorphisms for the symplectic ball,
   Commun. Contemp. Math. 6 (2004), no. 5, 793-802.

[7] Cieliebak, K., Salamon, D. Wall crossing for symplectic vortices and quantum cohomol-
   ogy, Math. Ann. 335 (2006), no. 1, 133-192.

[8] Cox, D. A., Katz, S. Mirror symmetry and algebraic geometry. Mathematical Surveys
   and Monographs, 68. American Mathematical Society, Providence, RI, 1999.

[9] Danilov, V. I. The geometry of toric varieties. (Russian) Uspekhi Mat. Nauk 33 (1978),
   no. 2(200), 85–134, 247. English translation: Russian Math. Surveys 33 (1978), no. 2,
   97–154.

[10] Delzant, T. Hamiltoniens périodiques et images convexes de ľapplication moment, Bull.
   Soc. Math. France 116 (1988), 315-339.

[11] Dubrovin, B. Geometry of 2d topological field theories, Integrable systems and quantum
   groups (Montecatini Terme, 1993), 120-348, Lecture Notes in Math., 1620, Springer,
   Berlin, 1996.

[12] Eisenbud, D., Harris, J. The geometry of schemes. (English summary) Graduate Texts
   in Mathematics, 197. Springer-Verlag, New York, 2000.

[13] Entov, M., Polterovich L. Calabi quasimorphism and quantum homology, Inter. Math.
   Res. Not. (2003), 1635-1676.

[14] Entov, M., Polterovich L. Symplectic quasi-states and semi-simplicity of quantum ho-
   mology, arXiv:0705.3735

[15] Entov, M., Polterovich L. Rigid subsets of symplectic manifolds, arXiv:0704.0105.

[16] Entov, M., Poltreovich L. Quasi-states and symplectic intersections, Comment. Math.
   Helv. 81 (2006), no. 1, 75-99.

[17] Entov, M., McDuff, D., Poltreovich L. Private communication.

[18] Fukaya, K., Oh, Y.-G., Ohta, H., Ono, K. Lagrangian Floer theory on compact toric
   manifolds I, arXiv:0802.1703.

[19] Fulton, William Introduction to toric varieties. Annals of Mathematics Studies, 131.
   The William H. Roever Lectures in Geometry. Princeton University Press, Princeton,
   NJ, 1993. xii+157 pp.

                                            28
[20] Givental, A. A mirror theorem for toric complete intersections, Topological field theory,
   primitive forms and related topics (Kyoto, 1996), Progr. Math. 160, Birkhäuser, Boston,
   1998, pp. 141-175.

[21] Givental, A. Equivariant Gromov-Witten invariants, Int. Math. Res. Not. 13 1996 613-
   663.

[22] Hori, K., Katz, S., Klemm, A., Pandharipande, R., Thomas, R., Vafa, C., Vakil,
   R.,Zaslow, E.Mirror symmetry. Clay Mathematics Monographs, 1. American Mathemat-
   ical Society, Providence, RI; Clay Mathematics Institute, Cambridge, MA, 2003

[23] Hori, K., Vafa, C. Mirror symmetry, preprint hep-th/0002222.

[24] Iritani, H. Convergence of quantum cohomology by quantum Lefschetz, (English sum-
   mary) J. Reine Angew. Math. 610 (2007), 29-69.

[25] Kotschick, D. What is: a quasi-morphism?, Not. Amer. Math. Soc. 51 (2004) 208-209.

[26] Kontsevich, M., and Manin, Y. Gromov-Witten classes, quantum cohomology, and
   enumerative geometry. Comm. Math. Phys., 164:525562, 1994.

[27] Kreuzer, M., Skarke, H. PALP: a package for analysing lattice polytopes with
   applications to toric geometry, Comput. Phys. Comm. 157 (2004), no. 1, 87–106.
   http://hep.itp.tuwien.ac.at/∼kreuzer/CY.

[28] Liu, G. Associativity of quantum multiplication, Comm. Math. Phys. 191:2 (1998),
   265-282.

[29] Lerche, W., Vafa, C., Warner, N.P. Chiral rings in N = 2 superconformal theories,
   Neuclear Physics B324 (1989), 427-474.

[30] Matsumura H. Commutative ring theory, Translated from the Japanese by M. Reid.
   Second edition. Cambridge Studies in Advanced Mathematics, 8. Cambridge University
   Press, Cambridge, 1989.

[31] McDuff, D. Hamiltonian S 1 manifolds are uniruled, preprint, arXiv:0706.0675

[32] McDuff, D. Private communication.

[33] McDuff, D. and Salamon D. J-holomorphic curves and symplectic topology., American
   Mathematical Society, Providence, (2004).

[34] McDuff, D., Tolman, S. Topological properties of Hamiltonian circle actions, Int. Math.
   Res. Pap. 2006, 72826, 1-77.

[35] Ostrover, Y. Calabi quasi-morphisms for some non-monotone symplectic manifolds,
   Algebr. Goem. Topol. 6 (2006), 405-434.

[36] Oh, Y.-G. Floer minimax theory, the Cerf diagram and spectral invariants, preprint:
   mathSG/0406449, to appear in J. Korean Math. Soc.

                                             29
[37] Oh, Y.-G. Construction of spectral invariants of Hamiltonian diffeomorphisms on gen-
   eral symplectic manifolds, in “The breadth of symplectic and Poisson geometry”, 525-570,
   Birkhäuser, Boston, 2005.

[38] Ruan, Y., Tian, G. A mathematical theory of quantum cohomology, Math. Res. Lett.
   1:2 (1994), 269-278.

[39] Ruan, Y., Tian, G. A mathematical theory of quantum cohomology, J. Diff. Geom. 42:2
   (1995), 259-367.

[40] Sato, H. Toward the classification of higher-dimensional toric Fano varieties, Tohoku
   Math. J. 52 (2000), 383413.

[41] Schwarz, M. On the action spectrum for closed symplectically aspherical manifolds,
   Pacific J. Math. 193:2 (2000), 419-461.

[42] Usher, M. Spectral numbers in Floer theories, arXiv:math/0709.1127

[43] Vafa, C. Topological mirrors and quantum rings, Essays on mirror manifolds (S.-T.
   Yau ed.), 96-119; International Press, Hong-Kong (1992).

[44] Vafa, C. Topological Landau-Ginzburg model, Mod. Phys. Lett. A 6 (1991), 337-346.

[45] Witten, E. Two-dimensional gravity and intersection theory on moduli space, Surveys
   in Diff. Geom. 1 (1991), 243-310.



Yaron Ostrover
Department of Mathematics, M.I.T, Cambridge MA 02139, USA
e-mail: ostrover@math.mit.edu

Ilya Tyomkin
Department of Mathematics, M.I.T, Cambridge MA 02139, USA
e-mail: tyomkin@math.mit.edu




                                            30
