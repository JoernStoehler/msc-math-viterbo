---
source: arXiv:math/0506191
fetched: 2025-10-20
---
# Quantitative symplectic geometry

                                                           Quantitative symplectic geometry
arXiv:math/0506191v1 [math.SG] 10 Jun 2005




                                                       K. Cieliebak, H. Hofer, J. Latschev and F. Schlenk                      ∗


                                                                               February 1, 2008


                                             A symplectic manifold (M, ω) is a smooth manifold M endowed with a non-
                                             degenerate and closed 2-form ω. By Darboux’s Theorem such a manifold looks
                                             locally like an open set in some R2n ∼
                                                                                  = Cn with the standard symplectic form
                                                                                       n
                                                                                       X
                                                                               ω0 =          dxj ∧ dyj ,                              (1)
                                                                                       j=1

                                             and so symplectic manifolds have no local invariants. This is in sharp contrast to
                                             Riemannian manifolds, for which the Riemannian metric admits various curva-
                                             ture invariants. Symplectic manifolds do however admit many global numerical
                                             invariants, and prominent among them are the so-called symplectic capacities.
                                             Symplectic capacities were introduced in 1990 by I. Ekeland and H. Hofer [19, 20]
                                             (although the first capacity was in fact constructed by M. Gromov [40]). Since
                                             then, lots of new capacities have been defined [16, 30, 32, 44, 49, 59, 60, 90, 99]
                                             and they were further studied in [1, 2, 8, 9, 17, 26, 21, 28, 31, 35, 37, 38, 41,
                                             42, 43, 46, 48, 50, 52, 56, 57, 58, 61, 62, 63, 64, 65, 66, 68, 74, 75, 76, 88, 89,
                                             91, 92, 94, 97, 98]. Surveys on symplectic capacities are [45, 50, 55, 69, 97].
                                             Different capacities are defined in different ways, and so relations between ca-
                                             pacities often lead to surprising relations between different aspects of symplectic
                                             geometry and Hamiltonian dynamics. This is illustrated in § 2, where we discuss
                                             some examples of symplectic capacities and describe a few consequences of their
                                             existence. In § 3 we present an attempt to better understand the space of all
                                             symplectic capacities, and discuss some further general properties of symplectic
                                             capacities. In § 4, we describe several new relations between certain symplectic
                                             capacities on ellipsoids and polydiscs. Throughout the discussion we mention
                                             many open problems.
                                             As illustrated below, many of the quantitative aspects of symplectic geome-
                                             try can be formulated in terms of symplectic capacities. Of course there are
                                             other numerical invariants of symplectic manifolds which could be included in
                                                ∗ The research of the first author was partially supported by the DFG grant Ci 45/2-1. The

                                             research of the second author was partially supported by the NSF Grant DMS-0102298. The
                                             third author held a position financed by the DFG grant Mo 843/2-1. The fourth author held
                                             a position financed by the DFG grant Schw 892/2-1.



                                                                                          1
a discussion of quantitative symplectic geometry, such as the invariants derived
from Hofer’s bi-invariant metric on the group of Hamiltonian diffeomorphisms,
[44, 81, 84], or Gromov-Witten invariants. Their relation to symplectic capaci-
ties is not well understood, and we will not discuss them here.
We start out with a brief description of some relations of symplectic geometry
to neighbouring fields.


1    Symplectic geometry and its neighbours
Symplectic geometry is a rather new and vigorously developing mathematical
discipline. The “symplectic explosion“ is described in [22]. Examples of sym-
plectic manifolds are open subsets of R2n , ω0 , the torus R2n /Z2n endowed
                                               

with the induced symplectic form, surfaces equipped with an area form, Kähler
manifolds like complex projective space CPn endowed with their Kähler form,
and cotangent bundles with their canonical symplectic form. Many more exam-
ples are obtained by taking products and through more elaborate constructions,
such as the symplectic blow-up operation. A diffeomorphism ϕ on a symplectic
manifold (M, ω) is called symplectic or a symplectomorphism if ϕ∗ ω = ω.
A fascinating feature of symplectic geometry is that it lies at the crossroad of
many other mathematical disciplines. In this section we mention a few examples
of such interactions.

Hamiltonian dynamics. Symplectic geometry originated in Hamiltonian dy-
namics, which originated in celestial mechanics. A time-dependent Hamiltonian
function on a symplectic manifold (M, ω) is a smooth function H : R × M → R.
Since ω is non-degenerate, the equation

                               ω (XH , ·) = dH(·)

defines a time-dependent smooth vector field XH on M . Under suitable assump-
tion on H, this vector field generates a family of diffeomorphisms ϕtH called the
Hamiltonian flow of H. As is easy to see, each map ϕtH is symplectic. A
Hamiltonian diffeomorphism ϕ on M is a diffeomorphism of the form ϕ1H .
Symplectic geometry is the geometry underlying Hamiltonian systems. It turns
out that this geometric approach to Hamiltonian systems is very fruitful. Ex-
plicit examples are discussed in § 2 below.

Volume geometry. A volume form Ω on a manifold M is a top-dimensional
nowhere vanishing differential form, and a diffeomorphism ϕ of M is volume
preserving if ϕ∗ Ω = Ω. Ergodic theory studies the properties of volume pre-
serving mappings. Its findings apply to symplectic mappings. Indeed, since a
symplectic form ω is non-degenerate, ω n is a volume form, which is preserved
under symplectomorphisms. In dimension 2 a symplectic form is just a volume
form, so that a symplectic mapping is just a volume preserving mapping. In


                                       2
dimensions 2n ≥ 4, however, symplectic mappings are much more special. A
geometric example for this is Gromov’s Nonsqueezing Theorem stated in § 2.2
and a dynamical example is the (partly solved) Arnol’d conjecture stating that
Hamiltonian diffeomorphisms of closed symplectic manifolds have at least as
many fixed points as smooth functions have critical points. For another link
between ergodic theory and symplectic geometry see [83].

Contact geometry. Contact geometry originated in geometrical optics. A
contact manifold (P, α) is a (2n − 1)-dimensional manifold P endowed with a
1-form α such that α ∧ (dα)n−1 is a volume form on P . The vector field X on
P defined by dα(X, ·) = 0 and α(X) = 1 generates the so-called Reeb flow. The
restriction of a time-independent Hamiltonian system to an energy surface can
sometimes be realized as the Reeb flow on a contact manifold. Contact manifolds
also arise naturally as boundaries of symplectic manifolds. One can study a
contact manifold (P, α) by symplectic means by looking at its symplectization
(P × R, d(et α)), see e.g. [47, 23].

Algebraic geometry. A special class of symplectic manifolds are Kähler man-
ifolds. Such manifolds (and, more generally, complex manifolds) can be studied
by looking at holomorphic curves in them. M. Gromov [40] observed that some
of the tools used in the Kähler context can be adapted for the study of sym-
plectic manifolds. One part of his pioneering work has grown into what is now
called Gromov-Witten theory, see e.g. [73] for an introduction.
Many other techniques and constructions from complex geometry are useful in
symplectic geometry. For example, there is a symplectic version of blowing-
up, which is intimately related to the symplectic packing problem, see [67, 71]
and 4.1.2 below. Another example is Donaldson’s construction of symplectic
submanifolds [18]. Conversely, symplectic techniques proved useful for study-
ing problems in algebraic geometry such as Nagata’s conjecture [5, 6, 71] and
degenerations of algebraic varieties [7].

Riemannian and spectral geometry. Recall that the differentiable struc-
ture of a smooth manifold M gives rise to a canonical symplectic form on its
cotangent bundle T ∗ M . Giving a Riemannian metric g on M is equivalent to
prescribing its unit cosphere bundle Sg∗ M ⊂ T ∗ M , and the restriction of the
canonical 1-form from T ∗ M gives S ∗ M the structure of a contact manifold. The
Reeb flow on Sg∗ M is the geodesic flow (free particle motion).
In a somewhat different direction, each symplectic form ω on some manifold M
distinguishes the class of Riemannian metrics which are of the form ω(J·, ·) for
some almost complex structure J.
These (and other) connections between symplectic and Riemannian geometry
are by no means completely explored, and we believe there is still plenty to be
discovered here. Here are some examples of known results relating Riemannian
and symplectic aspects of geometry.
1. Lagrangian submanifolds. A middle-dimensional submanifold L of (M, ω) is

                                       3
called Lagrangian if ω vanishes on T L.
(i) Volume. Endow complex projective space CPn with the usual Kähler metric
and the usual Kähler form. The volume of submanifolds is taken with respect
to this Riemannian metric. According to a result of Givental-Kleiner-Oh, the
standard RPn in CPn has minimal volume among all its Hamiltonian deforma-
tions [77]. A partial result for the Clifford torus in CPn can be found in [39].
The torus S 1 × S 1 ⊂ S 2 × S 2 formed by the equators is also volume minimizing
among its Hamiltonian    deformations, [51]. If L is a closed Lagrangian subman-
ifold of R2n , ω0 , there exists according to [100] a constant C depending on L
                 

such that

          vol (ϕH (L)) ≥ C      for all Hamiltonian deformations of L.           (2)

(ii) Mean curvature. The mean curvature form of a Lagrangian submanifold L
in a Kähler-Einstein manifold can be expressed through symplectic invariants
of L, see [15].
2. The first eigenvalue of the Laplacian. Symplectic methods can be used to
estimate the first eigenvalue of the Laplace operator on functions for certain
Riemannian manifolds [82].
3. Short billiard trajectories. Consider a bounded domain U ⊂ Rn with smooth
boundary. There exists a periodic billiard trajectory on U of length l with

                                  ln ≤ Cn vol(U )                                (3)

where Cn is an explicit constant depending only on n, see [100, 31].


2     Examples of symplectic capacities
In this section we give the formal definition of symplectic capacities, and discuss
a number of examples along with sample applications.


2.1    Definition
Denote by Symp 2n the category of all symplectic manifolds of dimension 2n,
with symplectic embeddings as morphisms. A symplectic category is a subcat-
egory C of Symp 2n such that (M, ω) ∈ C implies (M, αω) ∈ C for all α > 0.
Throughout the paper we will use the symbol ֒→ to denote symplectic
embeddings and → to denote morphisms in the category C (which may
be more restrictive).
Let B 2n (r2 ) be the open ball of radius r in R2n and Z 2n (r2 ) = B 2 (r2 ) × R2n−2
the open cylinder (the reason for this notation will become apparent below).
Unless stated otherwise, open subsets of R2n are always equipped with the


                                          4
                                Pn
canonical symplectic form ω0 = j=1 dyj ∧dxj . We will suppress the dimension
2n when it is clear from the context and abbreviate

                        B := B 2n (1),       Z := Z 2n (1).

Now let C ⊂ Symp 2n be a symplectic category containing the ball B and the
cylinder Z. A symplectic capacity on C is a covariant functor c from C to the
category ([0, ∞], ≤) (with a ≤ b as morphisms) satisfying

(Monotonicity) c(M, ω) ≤ c(M ′ , ω ′ ) if there exists a morphism (M, ω) →
    (M ′ , ω ′ );
(Conformality) c(M, αω) = α c(M, ω) for α > 0;
(Nontriviality) 0 < c(B) and c(Z) < ∞.

Note that the (Monotonicity) axiom just states the functoriality of c. A sym-
plectic capacity is said to be normalized if

(Normalization) c(B) = 1.

As a frequent example we will use the set Op 2n of open subsets in R2n . We make
it into a symplectic category by identifying (U, α2 ω0 ) with the symplectomorphic
manifold (αU, ω0 ) for U ⊂ R2n and α > 0. We agree that the morphisms in this
category shall be symplectic embeddings induced by global symplectomorphisms
of R2n . With this identification, the (Conformality) axiom above takes the form

(Conformality)’ c(αU ) = α2 c(U ) for U ∈ Op 2n , α > 0.


2.2    Gromov radius [40]
In view of Darboux’s Theorem one can associate with each symplectic manifold
(M, ω) the numerical invariant

                cB (M, ω) := sup α > 0 | B 2n (α) ֒→ (M, ω)
                                

called the Gromov radius of (M, ω), [40]. It measures the symplectic size of
(M, ω) in a geometric way, and is reminiscent of the injectivity radius of a
Riemannian manifold. Note that it clearly satisfies the (Monotonicity) and
(Conformality) axioms for a symplectic capacity. It is equally obvious that
cB (B) = 1.
                                                          R
If M is 2-dimensional and connected, then πcB (M, ω) = M ω, i.e. cB is pro-
portional to the volume of M , see [91]. The following theorem from Gromov’s
seminal paper [40] implies that in higher dimensions the Gromov radius is an
invariant very different from the volume.
Nonsqueezing Theorem (Gromov, 1985). The cylinder Z ∈ Symp 2n sat-
isfies cB (Z) = 1.

                                         5
In particular, the Gromov radius is a normalized symplectic capacity on Symp 2n .
Gromov originally obtained this result by studying properties of moduli spaces
of pseudo-holomorphic curves in symplectic manifolds.
It is important to realize that the existence of at least one capacity c with c(B) =
c(Z) also implies the Nonsqueezing Theorem. We will see below that each of the
other important techniques in symplectic geometry (such as variational methods
and the global theory of generating functions) gave rise to the construction of
such a capacity, and hence an independent proof of this fundamental result.
It was noted in [19] that the following result, originally established by Eliashberg
and by Gromov using different methods, is also an easy consequence of the
existence of a symplectic capacity.
Theorem (Eliashberg, Gromov) The group of symplectomorphisms of a
symplectic manifold (M, ω) is closed for the compact-open C 0 -topology in the
group of all diffeomorphisms of M .


2.3    Symplectic capacities via Hamiltonian systems
The next four examples of symplectic capacities are constructed via Hamiltonian
systems. A crucial role in the definition or the construction of these capacities is
played by the action functional of classical mechanics. For simplicity, we assume
that (M, ω) = (R2n , ω0 ). Given a Hamiltonian function H : S 1 ×R2n → R which
is periodic in the time-variable t ∈ S 1 = R/Z and which generates a global flow
ϕtH , the action functional on the loop space C ∞ (S 1 , R2n ) is defined as
                                  Z         Z 1
                                                          
                       AH (γ) =      y dx −     H t, γ(t) dt.                   (4)
                                   γ         0

Its critical points are exactly the 1-periodic orbits of ϕtH . Since the action
functional is neither bounded from above nor from below, critical points are
saddle points. In his pioneering work [85, 86], P. Rabinowitz designed special
minimax principles adapted to the hyperbolic structure of the action functional
to find such critical points. We give a heuristic argument why this works.
Consider the space of loops
                                 (                                    )
                 1/2  1   2n           2   1   2n
                                                   X             2
         E = H (S , R ) = z ∈ L S ; R                    |k| |zk | < ∞
                                                       k∈Z

                    2πktJ
                          zk , zk ∈ R2n , is the Fourier series of z and J is the
           P
where z =     k∈Z e
standard complex structure of R2n ∼  = Cn . The space E is a Hilbert space with
inner product                                 X
                     hz, wi = hz0 , w0 i + 2π     |k| hzk , wk i,
                                              k∈Z

and there is an orthogonal splitting E = E ⊕ E 0 ⊕ E + , z = z − + z 0 + z + , into
                                             −

the spaces of z ∈ E having nonzero Fourier coefficients zk ∈ R2n only for k < 0,

                                         6
k = 0, k > 0. The action functional AH : C ∞ (S 1 , R2n ) → R extends to E as
                                             Z 1
                                2          2
              AH (z) = 12 z + − 12 z −        −       H(t, z(t)) dt.         (5)
                                                      0

Notice now the hyperbolic structure of the first term A0 (x), and that the second
term is of lower order. Some of the critical points z(t) ≡ const of A0 should
thus persist for H 6= 0.

2.3.1     Ekeland-Hofer capacities [19, 20]

The first constructions of symplectic capacities via Hamiltonian systems were
carried out by Ekeland and Hofer [19, 20]. They considered the space F of
time-independent Hamiltonian functions H : R2n → [0, ∞) satisfying

   • H|U ≡ 0 for some open subset U ⊂ R2n , and
   • H(z) = a|z|2 for |z| large, where a > π, a 6∈ Nπ.

Given k ∈ N and H ∈ F, apply equivariant minimax to define the critical value
                (                                                    )
        cH,k := inf    sup AH (γ) | ξ ⊂ E is S 1 -equivariant and ind(ξ) ≥ k
                       γ∈ξ

of the action functional (5), where ind(ξ) denotes a suitable Fadell-Rabinowitz
index [27, 20] of the intersection ξ ∩ S + of ξ with the unit sphere S + ⊂ E + .
The k th Ekeland-Hofer capacity cEHk   on the symplectic category Op 2n is now
defined as

        cEH
                       
          k (U ) := inf cH,k | H vanishes on some neighborhood of U

if U ⊂ R2n is bounded and as

                      cEH
                                     EH
                       k (U ) := sup ck (V ) | V ⊂ U bounded

in general. It is immediate from the definition that cEH
                                                       1  ≤ cEH
                                                              2  ≤ cEH
                                                                     3 ≤ ...
form an increasing sequence. Their values on the ball and cylinder are
                                     
                               k+n−1
                   cEH
                    k  (B) =            π,     cEH
                                                k (Z) = kπ,
                                 n

where [x] denotes the largest integer ≤ x. Hence the existence of cEH1   gives
an independent proof of Gromov’s Nonsqueezing Theorem. Using the capacity
cEH
 n , Ekeland and Hofer [20] also proved the following nonsqueezing result.

Theorem (Ekeland-Hofer, 1990) The cube P = B 2 (1) × · · · × B 2 (1) ⊂ Cn
can be symplectically embedded into the ball B 2n (r2 ) if and only if r2 ≥ n.
Other illustrations of the use of Ekeland-Hofer capacities in studying embedding
problems for ellipsoids and polydiscs appear in § 4.

                                          7
2.3.2   Hofer-Zehnder capacity [49, 50]

Given a symplectic manifold (M, ω) we consider the class S(M ) of simple Hamil-
tonian functions H : M → [0, ∞) characterized by the following properties:

   • H = 0 near the (possibly empty) boundary of M ;
   • The critical values of H are 0 and max H.

Such a function is called admissible if the flow ϕtH of H has no non-constant
periodic orbits with period T ≤ 1.
The Hofer-Zehnder capacity cHZ on Symp 2n is defined as

             cHZ (M ) := sup {max H | H ∈ S(M ) is admissible}

It measures the symplectic size of M in a dynamical way. Easily constructed
examples yield the inequality cHZ (B) ≥ π. In [49, 50], Hofer and Zehnder
applied a minimax technique to the action functional (5) to show that cHZ (Z) ≤
π, so that
                          cHZ (B) = cHZ (Z) = π,
providing another independent proof of the Nonsqueezing Theorem. Moreover,
for every symplectic manifold (M, ω) the inequality πcB (M ) ≤ cHZ (M ) holds.
The importance of understanding the Hofer-Zehnder capacity comes from the
following result proved in [49, 50].
Theorem (Hofer-Zehnder, 1990) Let H : (M, ω) → R be a proper autonomous
Hamiltonian. If cHZ (M ) < ∞, then for almost every c ∈ H(M ) the energy level
H −1 (c) carries a periodic orbit.
Variants of the Hofer-Zehnder capacity which can be used to detect periodic
orbits in a prescribed homotopy class where considered in [60, 90].

2.3.3   Displacement energy [44, 56]

Next, let us measure the symplectic size of a subset by looking at how much
energy is needed to displace it from itself. Fix a symplectic manifold (M, ω).
Given a compactly supported Hamiltonian H : [0, 1] × M → R, set
                         Z 1                            
                kHk :=          sup H(t, x) − inf H(t, x) dt.
                           0   x∈M           x∈M

The energy of a compactly supported Hamiltonian diffeomorphism ϕ is

                      E(ϕ) := inf kHk | ϕ = ϕ1H .
                                  

The displacement energy of a subset A of M is now defined as

                    e(A, M ) := inf {E(ϕ) | ϕ(A) ∩ A = ∅}

                                      8
if A is compact and as

                e(A, M ) := sup {e(K, M ) | K ⊂ A is compact}

for a general subset A of M .
Now consider the special case (M, ω) = (R2n , ω0 ). Simple explicit examples
show e(Z, R2n ) ≤ π. In [44], H. Hofer designed a minimax principle for the
action functional (5) to show that e(B, R2n ) ≥ π, so that

                          e(B, R2n ) = e(Z, R2n ) = π.

It follows that e(·, R2n ) is a symplectic capacity on the symplectic category Op 2n
of open subset of R2n .
One important feature of the displacement energy is the inequality

                                cHZ (U ) ≤ e(U, M )                              (6)

holding for open subsets of many (and possibly all) symplectic manifolds, in-
cluding (R2n , ω0 ). Indeed, this inequality and the Hofer-Zehnder Theorem imply
existence of periodic orbits on almost every energy surface of any Hamiltonian
with support in U provided only that U is displaceable in M . The proof of this
inequality uses the spectral capacities introduced in § 2.3.4 below.
As a specific application, consider a closed Lagrangian submanifold L of (R2n , ω0 ).
Viterbo [100] used an elementary geometric construction to show that
                                                     2/n
                            e L, R2n ≤ Cn (vol(L))
                                     

for an explicitconstant Cn . By a result of Chekanov [12], e L, R2n > 0. Since
                                                                     

e ϕH (L), R2n = e L, R2n for every Hamiltonian diffeomorphism of L, we
obtain Viterbo’s inequality (2).

2.3.4   Spectral capacities [32, 46, 50, 78, 79, 80, 88, 99]

For simplicity, we assume again (M, ω) = (R2n , ω0 ). Denote by H the space
of compactly supported Hamiltonian functions H : S 1 × R2n → R. An action
selector σ selects for each H ∈ H the action σ(H) = AH (γ) of a “topologically
visible” 1-periodic orbit γ of ϕtH in a suitable way. Such action selectors were
constructed by Viterbo [99], who applied minimax to generating functions, and
by Hofer and Zehnder [46, 50], who applied minimax directly to the action
functional (5). An outline of their constructions can be found in [31].
Given an action selector σ for (R2n , ω0 ), one defines the spectral capacity cσ on
the symplectic category Op 2n by

              cσ (U ) := sup σ(H) | H is supported in S 1 × U .
                            

It follows from the defining properties of an action selector (not given here)
that cHZ (U ) ≤ cσ (U ) for any spectral capacity cσ . Elementary considerations

                                         9
also imply cσ (U ) ≤ e(U, R2n ), see [31, 46, 50, 99]. In this way one in particular
obtains the important inequality (6) for M = R2n .
Another application of action selectors is
Theorem (Viterbo, 1992) Every   non-identical compactly supported Hamilto-
nian diffeomorphism of R2n , ω0 has infinitely many non-trivial periodic points.

Moreover, the existence of an action selector is an important ingredient in
Viterbo’s proof of the estimate (3) for billiard trajectories.
Using the Floer homology of (M, ω) filtered by the action functional, an ac-
tion selector can be constructed for many (and conceivably for all) symplectic
manifolds (M, ω), [32, 78, 79, 80, 88]. This existence result implies the energy-
capacity inequality (6) for arbitrary open subsets U of such (M, ω), which has
many applications [89].


2.4    Lagrangian capacity [16]
In [16] a capacity is defined on the category of 2n-dimensional symplectic man-
ifolds (M, ω) with π1 (M ) = π2 (M ) = 0 (with symplectic embeddings as mor-
phisms) as follows. The minimal symplectic area of a Lagrangian submanifold
L ⊂ M is
                            Z                    Z       
           Amin (L) := inf       ω σ ∈ π2 (M, L), ω > 0 ∈ [0, ∞].
                               σ                     σ

The Lagrangian capacity of (M, ω) is defined as

    cL (M, ω) := sup {Amin (L) | L ⊂ M is an embedded Lagrangian torus} .

Its values on the ball and cylinder are

                          cL (B) = π/n,        cL (Z) = π.

As the cube P = B 2 (1) × · · · × B 2 (1) contains the standard Clifford torus
T n ⊂ Cn , and is contained in the cylinder Z, it follows that cL (P ) = π. To-
gether with cL (B) = π/n this gives an alternative proof of the nonsqueezing
result of Ekeland and Hofer mentioned in § 2.3.1. There are also applications
of the Lagrangian capacity to Arnold’s chord conjecture and to Lagrangian
(non)embedding results into uniruled symplectic manifolds [16].


3     General properties and relations between sym-
      plectic capacities
In this section we study general properties of and relations between symplectic
capacities. We begin by introducing some more notation. Define the ellipsoids

                                          10
and polydiscs

                                                        |z1 |2         |zn |2
                                                                                  
         E(a) := E(a1 , . . . , an ) :=    z ∈ Cn              + ··· +        <1
                                                         a1             an
         P (a) := P (a1 , . . . , an ) := B 2 (a1 ) × · · · × B 2 (an )

for 0 < a1 ≤ · · · ≤ an ≤ ∞. Note that in this notation the ball, cube and
cylinder are B = E(1, . . . , 1), P = P (1, . . . , 1) and Z = E(1, ∞, . . . , ∞) =
P (1, ∞, . . . , ∞).
Besides Symp 2n and Op 2n , two symplectic categories that will frequently play
a role below are

Ell 2n : the category of ellipsoids in R2n , with symplectic embeddings induced
       by global symplectomorphisms of R2n as morphisms,
Pol 2n : the category of polydiscs in R2n , with symplectic embeddings induced
       by global symplectomorphisms of R2n as morphisms.


3.1    Generalized symplectic capacities
From the point of view of this work, it is convenient to have a more flexible no-
tion of symplectic capacities, whose axioms were originally designed to explicitly
exclude such invariants as the volume. We thus define a generalized symplectic
capacity on a symplectic category C as a covariant functor c from C to the cat-
egory ([0, ∞], ≤) satisfying only the (Monotonicity) and (Conformality) axioms
of § 2.1.
Now examples such as the volume capacity on Symp 2n are included into the
discussion. It is defined as
                                                   1/n
                                          vol(M, ω)
                         cvol (M, ω) :=                  ,
                                           vol(B)

where vol(M, ω) := M ω n /n! is the symplectic volume. For n ≥ 2 we have
                       R
cvol (B) = 1 and cvol (Z) = ∞, so cvol is a normalized generalized capacity but
not a capacity. Many more examples appear below.


3.2    Embedding capacities
Let C be a symplectic category. Every object (X, Ω) of C induces two generalized
symplectic capacities on C,

                c(X,Ω) (M, ω) := sup {α > 0 | (X, αΩ) → (M, ω)} ,
                c(X,Ω) (M, ω) := inf {α > 0 | (M, ω) → (X, αΩ)} ,



                                              11
Here the supremum and infimum over the empty set are set to 0 and ∞, respec-
tively. Note that
                                                −1
                    c(X,Ω) (M, ω) = c(M,ω) (X, Ω)    .                   (7)

Example 1. Suppose that (X, αΩ) → (X, Ω) for some α > 1. Then c(X,Ω) (X, Ω) =
∞ and c(X,Ω) (X, Ω) = 0, so that
                           ∞ if (X, βΩ) → (M, ω) for some β > 0,
                       
      c(X,Ω) (M, ω) =
                           0 if (X, βΩ) → (M, ω) for no β > 0,

                           0 if (M, ω) → (X, βΩ) for some β > 0,
                       
      c(X,Ω) (M, ω) =
                           ∞ if (M, ω) → (X, βΩ) for no β > 0.

The following fact follows directly from the definitions.
Fact 1. Suppose that there exists no morphism (X, αΩ) → (X, Ω) for any α > 1.
Then c(X,Ω) (X, Ω) = c(X,Ω) (X, Ω) = 1, and for every generalized capacity c with
0 < c(X, Ω) < ∞,
                           c(M, ω)
         c(X,Ω) (M, ω) ≤           ≤ c(X,Ω) (M, ω)        for all (M, ω) ∈ C.
                           c(X, Ω)
In other words, c(X,Ω) (resp. c(X,Ω) ) is the minimal (resp. maximal) generalized
capacity c with c(X, Ω) = 1.

Important examples on Symp 2n arise from the ball B = B 2n (1) and cylinder
Z = Z 2n (1). By Gromov’s Nonsqueezing Theorem and volume reasons we have
for n ≥ 2:
          cB (Z) = 1,       cZ (B) = 1,         cB (Z) = ∞,     cZ (B) = 0.
In particular, for every normalized symplectic capacity c,
       cB (M, ω) ≤ c(M, ω) ≤ c(Z) cZ (M, ω)          for all (M, ω) ∈ Symp 2n .   (8)
Recall that the capacity cB is the Gromov radius defined in § 2.2. The capacities
cB and cZ are not comparable on Op 2n : Example 3 below shows that for every
k ∈ N there is a bounded starshaped domain Uk of R2n such that
                  cB (Uk ) ≤ 2−k          and      cZ (Uk ) ≥ πk 2 ,
see also [43].
We now turn to the question which capacities can be represented as embedding
capacities c(X,Ω) or c(X,Ω) .
Example 2. Consider the subcategory C ⊂ Op 2n of connected open sets. Then
every generalized capacity c on C can be represented as the capacity c(X,Ω) of
embeddings into a (possibly uncountable) union (X, Ω) of objects in C.
For this, just define (X, Ω) as the disjoint union of all (Xι , Ωι ) in the category
C with c(Xι , Ωι ) = 0 or c(Xι , Ωι ) = 1.

                                          12
Problem 1. Which (generalized) capacities can be represented as c(X,Ω) for a
connected symplectic manifold (X, Ω)?
Problem 2. Which (generalized) capacities can be represented as the capacity
c(X,Ω) of embeddings from a symplectic manifold (X, Ω)?
Example 3. Embedding capacities give rise to some curious generalized capac-
ities. For example, consider the capacity cY of embeddings into the symplectic
manifold Y := ∐k∈N B 2n (k 2 ). It only takes values 0 and ∞, with cY (M, ω) = 0
iff (M, ω) embeds symplectically into Y , cf. Example 1. If M is connected,
vol(M, ω) = ∞ implies cY (M, ω) = ∞. On the other hand, for every ε > 0 there
exists an open subset U ⊂ R2n , diffeomorphic to a ball, with vol(U ) < ε and
cY (U ) = ∞. To see this, consider for k ∈ N an open neighbourhood Uk of volume
< 2−k ε of the linear cone over the Lagrangian torus ∂B 2 (k 2 ) × · · · × ∂B 2 (k 2 ).
The Lagrangian capacity of Uk clearly satisfies cL (Uk ) ≥ πk 2 . The open set
U := ∪k∈N Uk satisfies vol(U ) < ε and cL (U ) = ∞, hence U does not embed
symplectically into any ball. By appropriate choice of the Uk we can arrange
that U is diffeomorphic to a ball, cf. [88, Proposition A.3].                        ♦

Special embedding spaces.

Given an arbitrary pair of symplectic manifolds (X, Ω) and (M, ω), it is a diffi-
cult problem to determine or even estimate c(X,Ω) (M, ω) and c(X,Ω) (M, ω). We
thus consider two special cases.
1. Embeddings of skinny ellipsoids. Assume that (M, ω) is an ellipsoid
E(a, . . . , a, 1) with 0 < a ≤ 1, and that (X, Ω) is connected and has finite
volume. Upper bounds for the function

                 e(X,Ω) (a) = c(X,Ω) (E(a, . . . , a, 1)) ,   a ∈ (0, 1],

are obtained from symplectic embedding results of ellipsoids into (X, Ω), and
lower bounds are obtained from computing other (generalized) capacities and
using Fact 1. In particular, the volume capacity yields
                                      n
                            e(X,Ω) (a)      vol(B)
                                         ≥           .
                               an−1        vol(X, Ω)

The only known general symplectic embedding results for ellipsoids are obtained
via multiple symplectic folding. The following result is part of Theorem 3 in
[88], which in our setting reads
Fact 2. Assume that (X, Ω) is a connected 2n-dimensional symplectic manifold
of finite volume. Then
                                      n
                            e(X,Ω) (a)      vol(B)
                       lim               =           .
                       a→0     an−1        vol(X, Ω)


                                            13
For a restricted class of symplectic manifolds, Fact 2 can be somewhat improved.
The following result is part of Theorem 6.25 of [88].
Fact 3. Assume that X is a bounded domain in R2n , ω0 with piecewise smooth
                                                           

boundary or that (X, Ω) is a compact connected 2n-dimensional symplectic man-
ifold. If n ≤ 3, there exists a constant C > 0 depending only on (X, Ω) such
that                   n
             e(X,Ω) (a)              vol(B)                         1
                           ≤                            for all a < n .
                an−1
                                                   
                             vol(X, Ω) 1 − Ca1/n                   C

These results have their analogues for polydiscs P (a, . . . , a, 1). The analogue of
Fact 3 is known in all dimensions.
2. Packing capacities. Given an object (X, Ω) of C and k ∈ N, we denote by
`
  k (X, Ω) the disjoint union of k copies of (X, Ω) and define
                                   (                             )
                                             a
            c(X,Ω;k) (M, ω) := sup α > 0        (X, αΩ) ֒→ (M, ω) .
                                                       k

If vol(X, Ω) is finite, we see as in Fact 1 that
                                                   1
                    c(X,Ω;k) (M, ω) ≤              `         cvol (M, ω).                  (9)
                                          cvol (   k (X, Ω))

We say that (M, ω) admits a full k-packing by (X, Ω) if equality                 holds
                                                                                       in (9).
For k1 , . . . , kn ∈ N a full k1 · · · kn -packing of B 2n (1) by E k11 , . . . , k1n is given
in [96]. Full k-packings by balls and obstructions to full k-packings by balls are
studied in [3, 4, 40, 54, 66, 71, 88, 96].
Assume now that also vol(M, ω) is finite. Studying the capacity c(X,Ω;k) (M, ω)
is equivalent to studying the packing number
                                                `         
                                           vol ( k (X, αΩ)
                     p(X,Ω;k) (M, ω) = sup
                                        α      vol (M, ω)
                                                      `
where the supremum is taken over all α for which k (X, αΩ) symplectically
embeds into (M, ω). Clearly, p(X,Ω;k) (M, ω) ≤ 1, and equality holds iff equality
holds in (9). Results in [71] together with the above-mentioned full packings of
a ball by ellipsoids from [96] imply
Fact 4. If X is an ellipsoid or a polydisc, then

                              p(X,k) (M, ω) → 1 as k → ∞

for every symplectic manifold (M, ω) of finite volume.

Note that if the conclusion of Fact 4 holds for X and Y , then it also holds for
X ×Y.


                                              14
Problem 3. For which bounded convex subsets X of R2n is the conclusion of
Fact 4 true?

In [71] and [3, 4], the packing numbers p(X,k) (M ) are computed for X = B 4
and M = B 4 or CP 2 . Moreover, the following fact is shown in [3, 4]:
Fact 5. If X = B 4 , then for every closed connected symplectic 4-manifold
(M, ω) with [ω] ∈ H 2 (M ; Q) there exists k0 (M, ω) such that
                          p(X,k) (M, ω) = 1 for all k ≥ k0 (M, ω).
Problem 4. For which bounded convex subsets X of R2n and which connected
symplectic manifolds (M, ω) of finite volume is the conclusion of Fact 5 true?


3.3     Operations on capacities
We say that a function f : [0, ∞]n → [0, ∞] is homogeneous and monotone if
                 f (αx1 , . . . , αxn ) = αf (x1 , . . . , xn )               for all α > 0,
           f (x1 , . . . , xi , . . . , xn ) ≤ f (x1 , . . . , yi , . . . , xn )      for xi ≤ yi .
If f is homogeneous and monotone and c1 , . . . , cn are generalized capacities, then
f (c1 , . . . , cn ) is again a generalized capacity. If in addition 0 < f (1, . . . , 1) < ∞
and c1 , . . . , cn are capacities, then f (c1 , . . . , cn ) is a capacity. Compositions and
pointwise limits of homogeneous monotone functions are again homogeneous
and monotone. Examples include max(x1 , . . . , xn ), min(x1 , . . . , xn ), and the
weighted (arithmetic, geometric, harmonic) means
                                                                                       1
               λ1 x1 + · · · + λn xn ,              xλ1 1 · · · xλnn ,         λ1             λn
                                                                               x1   + ··· +   xn

with λ1 , . . . , λn ≥ 0, λ1 + · · · + λn = 1.
There is also a natural notion of convergence of capacities. We say that a
sequence cn of generalized capacities on C converges pointwise to a generalized
capacity c if cn (M, ω) → c(M, ω) for every (M, ω) ∈ C.
These operations yield lots of dependencies between capacities, and it is natural
to look for generating systems. In a very general form, this can be formulated
as follows.
Problem 5. For a given symplectic category C, find a minimal generating sys-
tem G for the (generalized) symplectic capacities on C. This means that every
(generalized) symplectic capacity on C is the pointwise limit of homogeneous
monotone functions of elements in G, and no proper subcollection of G has this
property.

This problem is already open for Ell 2n and Pol 2n . One may also ask for gener-
ating systems allowing fewer operations, e.g. only max and min, or only positive
linear combinations. We will formulate more specific versions of this problem
below. The following simple fact illustrates the use of operations on capacities.

                                                        15
Fact 6. Let C be a symplectic category containing B (resp. P ). Then every
generalized capacity c on C with c(B) 6= 0 (resp. c(P ) 6= 0) is the pointwise limit
of capacities.

Indeed, if c(B) 6= 0 (resp. c(P ) 6= 0), then c is the pointwise limit as k → ∞ of
the capacities                                              
                     ck = min (c, k cB ) resp. min (c, k cP ) .

Example 4. (i) The generalized capacity c ≡ 0 on Op 2n is not a pointwise limit
of capacities, and so the assumption c(B) 6= 0 in Fact 6 cannot be omitted.
(ii) The assumption c(B) 6= 0 is not always necessary:
(a) Define a generalized capacity c on Op 2n by
                              
                                 0       if vol(U ) < ∞,
                      c(U ) =
                                 cB (U ) if vol(U ) = ∞.

Then c(B) = 0 and c(Z) = 1, and c is the pointwise limit of the capacities

                           ck = max c, k1 cB .
                                              


(b) Define a generalized capacity c on Op 2n by
                                
                                   0 if cB (U ) < ∞,
                        c(U ) =
                                   ∞ if cB (U ) = ∞.

Then c(B) = 0 = c(Z) and c(R2n ) = ∞, and c = limk→∞ k1 cB .
(iii) We do not know whether the generalized capacity cR2n on Op 2n is the
pointwise limit of capacities.
Problem 6. Given a symplectic category C containing B or P and Z, charac-
terize the generalized capacities which are pointwise limits of capacities.


3.4    Continuity
There are several notions of continuity for capacities on open subsets of R2n ,
see [1, 19]. For example, consider a smooth family of hypersurfaces (St )−ε<t<ε
in R2n , each bounding a compact subset with interior Ut . S0 is said to be of
restricted contact type if there exists a vector field v on R2n which is transverse
to S0 and whose Lie derivative satisfies Lv ω0 = ω0 . Let c be a capacity on
Op 2n . As the flow of v is conformally symplectic, the (Conformality) axiom
implies (cf. [50, p. 116])
Fact 7. If S0 is of restricted contact type, the function t 7→ c(Ut ) is Lipschitz
continuous at 0.



                                        16
Fact 7 fails without the hypothesis of restricted contact type. For example, if
S0 possesses no closed characteristic (such S0 exist by [33, 34, 36]), then by
Theorem 3 in Section 4.2 of [50] the function t 7→ cHZ (Ut ) is not Lipschitz
continuous at 0. V. Ginzburg [35] presents an example of a smooth family of
hypersurfaces (St ) (albeit not in R2n ) for which the function t 7→ cHZ (Ut ) is not
smoother than 1/2-Hölder continuous. These considerations lead to
Problem 7. Are capacities continuous on all smooth families of domains boun-
ded by smooth hypersurfaces?


3.5    Convex sets
Here we restrict to the subcategory Conv 2n ⊂ Op 2n of convex open subsets of
R2n , with embeddings induced by global symplectomorphisms of R2n as mor-
phisms. Recall that a subset U ⊂ R2n is starshaped if U contains a point p
such that for every q ∈ U the straight line between p and q belongs to U . In
particular, convex domains are starshaped.
Fact 8. (Extension after Restriction Principle [19]) Assume that ϕ : U ֒→ R2n
is a symplectic embedding of a bounded starshaped domain U ⊂ R2n . Then for
any compact subset K of U there exists a symplectomorphism Φ of R2n such
that Φ|K = ϕ|K .

This principle continues to hold for some, but not all, symplectic embeddings
of unbounded starshaped domains, see [88]. We say that a capacity c defined
on a symplectic subcategory of Op 2n has the exhaustion property if

                    c(U ) = sup{ c(V ) | V ⊂ U is bounded }.                    (10)

The capacities introduced in § 2 all have this property, but the capacity in
Example 3 does not. By Fact 8, all statements about capacities defined on a
subcategory of Conv 2n and having the exhaustion property remain true if we
allow all symplectic embeddings (not just those coming from global symplecto-
morphisms of R2n ) as morphisms.
Fact 9. Let U and V be objects in Conv 2n . Then there exists a morphism
αU → V for every α ∈ (0, 1) if and only if c(U ) ≤ c(V ) for all generalized
capacities c on Conv 2n .

Indeed, the necessity of the condition is obvious, and the sufficiency follows by
observing that αU → U for all α ∈ (0, 1) and 1 ≤ cU (U ) ≤ cU (V ). What
happens for α = 1 is not well understood, see § 3.6 for related discussions.
The next example illustrates that the conclusion of Fact 9 is wrong without the
convexity assumption.
Example 5. Consider the open annulus A = B(4) \ B(1) in R2 . If 43 < α2 < 1,
then αA cannot be embedded into A by a global symplectomorphism. Indeed,


                                         17
volume considerations show that any potential such global symplectomorphism
would have to map A homotopically nontrivially into itself. This would force
the image of the ball αB(1) to cover all of B(1), which is impossible for volume
reasons.                                                                       ♦

Assume now that c is a normalized symplectic capacity on Conv 2n . Using John’s
ellipsoid, Viterbo [100] noticed that there is a constant Cn depending only on n
such that
                    cZ (U ) ≤ Cn cB (U ) for all U ∈ Conv 2n
and so, in view of (8),

            cB (U ) ≤ c(U ) ≤ Cn c(Z) cB (U )     for all U ∈ Conv 2n .        (11)

In fact, Cn ≤ (2n)2 and Cn ≤ 2n on centrally symmetric convex sets.
Problem 8. What is the optimal value of the constant Cn appearing in (11)?
In particular, is Cn = 1?

Note that Cn = 1 would imply uniqueness of capacities satisfying c(B) = c(Z) =
1 on Conv 2n . In view of Gromov’s Nonsqueezing Theorem, Cn = 1 on Ell 2n
and Pol 2n . More generally, this equality holds for all convex Reinhardt domains
[43]. In particular, for these special classes of convex sets

                     πcB = cEH
                            1  = cHZ = e(·, R2n ) = πcZ .


3.6    Recognition
One may ask how complete the information provided by all symplectic capacities
is. Consider two objects (M, ω) and (X, Ω) of a symplectic category C.
Question 1. Assume c(M, ω) ≤ c(X, Ω) for all generalized symplectic capacities
c on C. Does it follow that (M, ω) ֒→ (X, Ω) or even that (M, ω) → (X, Ω)?
Question 2. Assume c(M, ω) = c(X, Ω) for all generalized symplectic capacities
c on C. Does it follow that (M, ω) is symplectomorphic to (X, Ω) or even that
(M, ω) ∼
       = (X, Ω) in the category C?

Note that if (M, αω) → (M, ω) for all α ∈ (0, 1) then, under the assumptions
of Question 1, the argument leading to Fact 9 yields (M, αω) → (X, Ω) for all
α ∈ (0, 1).
Example 6. (i) Set U = B 2 (1) and V = B 2 (1) \ {0}. For each α < 1 there
exists a symplectomorphism of R2 with ϕ (αU ) ⊂ V , so that monotonicity and
conformality imply c(U ) = c(V ) for all generalized capacities c on Op 2 . Clearly,
U ֒→ V , but U 9 V , and U and V are not symplectomorphic.
(ii) Set U = B 2 (1) and let V = B 2 (1) \ {(x, y) | x ≥ 0, y = 0} be the slit disc.
As is well-known, U and V are symplectomorphic. Fact 8 implies c(U ) = c(V )

                                        18
for all generalized capacities c on Op 2 , but clearly U 9 V . In dimensions
2n ≥ 4 there are bounded convex sets U and V with smooth boundary which
are symplectomorphic while U 9 V , see [25].
(iii) Let U and V be ellipsoids in Ell 2n . The answer to Question 1 is unknown
even for Ell 4 . For U = E(1, 4) and V = B 4 (2) we have c(U ) ≤ c(V ) for
all generalized capacities that can presently be computed, but it is unknown
whether U ֒→ V , cf. 4.1.2 below. By Fact 10 below, the answer to Question 2
is “yes” on Ell 2n .
(iv) Let U and V be polydiscs in Pol 2n . Again, the answer to Question 1 is
unknown even for Pol 4 . However, in this dimension the Gromov radius together
with the volume capacity determine a polydisc, so that the answer to Question 2
is “yes” on Pol 4 .                                                          ♦


Problem 9. Are two polydiscs in dimension 2n ≥ 6 with equal generalized
symplectic capacities symplectomorphic?

To conclude this section, we mention a specific example in which c(U ) = c(V )
for all known (but possibly not for all) generalized symplectic capacities.
Example 7. Consider the subsets

            U = E(2, 6) × E(3, 3, 6)         and V = E(2, 6, 6) × E(3, 3)

of R10 . Then c(U ) = c(V ) whenever c(B) = c(Z) by the Nonsqueezing Theorem,
the volumina agree, and cEH          EH
                           k (U ) = ck (V ) for all k by the product formula (14).
It is unknown whether U ֒→ V or V ֒→ U or U → V . Symplectic homology as
constructed in [29, 95] does not help in these problems because a computation
based on [30] shows that all symplectic homologies of U and V agree.


3.7    Hamiltonian representability
Consider a bounded domain U ⊂ R2n with smooth boundary of restricted con-
tact type (cf. § 3.4 for the definition). A closed characteristic γ on ∂U is an
embedded circle in ∂U tangent to the characteristic line bundle

            LU = {(x, ξ) ∈ T ∂U | ω0 (ξ, η) = 0 for all η ∈ Tx ∂U } .

If ∂U is represented as a regular energy surface x ∈ R2n | H(x) = const of a
                                                  

smooth function H on R2n , then the Hamiltonian vector field XH restricted to
∂U is a section of LU , and so the traces of the periodic orbits of XH on ∂U are
the closed characteristics on ∂U . The action A (γ) of a closed characteristic γ
                              R
on ∂U is defined as A (γ) = γ y dx . The set
                 
      Σ (U ) =       kA (γ) | k = 1, 2, . . . ; γ is a closed characteristic on ∂U


                                             19
is called the action spectrum of U . This set is nowhere dense in R, cf. [50,
Section 5.2], and it is easy to see that Σ(U ) is closed and 0 ∈  / Σ(U ). For
many capacities c constructed via Hamiltonian systems, such as Ekeland-Hofer
capacities cEH
            k   and spectral capacities cσ , one has c(U ) ∈ Σ(U ), see [20, 42].
Moreover,

              cHZ (U ) = cEH
                          1 (U ) = min (Σ(U ))          if U is convex.        (12)

One might therefore be tempted to ask
Question 3. Is it true that πc(U ) ∈ Σ(U ) for every normalized symplectic
capacity c on Op 2n and every domain U with boundary of restricted contact
type?

The following example due to D. Hermann [43] shows that the answer to Ques-
tion 3 is “no”.
Example 8. Choose any U with boundary of restricted contact type such that

                                 cB (U ) < cZ (U ).                            (13)
Examples are bounded starshaped domains U with smooth boundary which
contain the Lagrangian torus S 1 × · · · × S 1 but have small volume: According
to [93], cZ (U ) ≥ 1, while cB (U ) is as small as we like. Now notice that for each
t ∈ [0, 1],
                                ct = (1 − t)cB + tcZ
is a normalized symplectic capacity on Op 2n . By (13), the interval

                      {ct (U ) | t ∈ [0, 1]} = [cB (U ), cZ (U )]

has positive measure and hence cannot lie in the nowhere dense set Σ(U ).         ♦

D. Hermann also pointed out that the argument in Example 8 together with
(12) implies that the question “Cn = 1?” posed in Problem 8 is equivalent to
Question 3 for convex sets.


3.8    Products
Consider a family of symplectic categories C 2n in all dimensions 2n such that

         (M, ω) ∈ C 2m , (N, σ) ∈ C 2n =⇒ (M × N, ω ⊕ σ) ∈ C 2(m+n) .
                                    2n
We say that a collection c : ∐∞
                              n=1 C    → [0, ∞] of generalized capacities has the
product property if

                   c(M × N, ω ⊕ σ) = min{c(M, ω), c(N, σ)}



                                          20
for all (M, ω) ∈ C 2m , (N, σ) ∈ C 2n . If R2 ∈ C 2 and c(R2 ) = ∞, the product
property implies the stability property

                          c(M × R2 , ω ⊕ ω0 ) = c(M, ω)

for all (M, ω) ∈ C 2m .
Example 9. (i) Let Σg be a closed surface of genus g endowed with an area
form ω. Then
                                 cB (Σg , ω) = π1 ω (Σg ) if g = 0,
                               (
        cB Σg × R2 , ω ⊕ ω 0 =
                            
                                 ∞                        if g ≥ 1.

While the result for g = 0 follows from Gromov’s Nonsqueezing Theorem, the
result for g ≥ 1 belongs to Polterovich [72, Exercise 12.4] and Jiang [53]. Since
                                                              2n
cB is the smallest normalized symplectic capacity on Symp `      , we find that no
                                                                 ∞
collection c of symplectic capacities defined on the family n=1 Symp 2n with
c (Σg , ω) < ∞ for some g ≥ 1 has the product or stability property.
                               `∞
(ii) On the family of polydiscs n=1 Pol 2n , the Gromov radius, the Lagrangian
capacity and the unnormalized Ekeland-Hofer capacities cEH  k   all have the prod-
uct property (see Section 4.2). The volume capacity is not stable.
(iii) Let U ∈ Op 2m and V ∈ Op 2n have smooth boundary of restricted contact
type (cf. § 3.4 for the definition). The formula

                  cEH                        EH
                                                 (U ) + cEH
                                                               
                   k (U × V ) = mini+j=k ci              j (V ) ,       (14)

in which we set cEH
                  0  ≡ 0, was conjectured by Floer and Hofer [97] and has been
proved by Chekanov [13] as an application of his equivariant Floer homology.
                                                                      2ni
Consider the collection of sets U1 × · · · × UPl , where each Ui ∈ Op     has smooth
                                                   l
boundary of restricted contact type, and i=1 ni = n. We denote by RCT 2n
the corresponding category with symplectic embeddings induced by global sym-
plectomorphisms of R2n as morphisms. If vi are vector fields on R2ni with
Lvi ω0 = ω0 , then Lv1 +···+vl ω0 = ω0 on R2n . Elements of RCT 2n can therefore
be exhausted by elements of RCT 2n with smooth boundary of restricted contact
type. This and the exhaustion property (10) of the cEH      k  shows that (14) holds
for all U ∈ RCT 2m and V ∈ RCT`      2n
                                        , implying in particular that Ekeland-Hofer
                                      ∞
capacities are stable on RCT := n=1 RCT 2n . Moreover, (14) yields that

                    cEH                       EH         EH
                                                                
                     k (U × V ) ≤ min ck (U ) , ck (V ) ,

and it shows that cEH
                    1   on RCT has the product property. Using (14) together
with an induction over the number of factors and cEH
                                                  2 (E(a1 , . . . , an )) ≤ 2a1 we
also see that cEH
               2  has the product property on products of ellipsoids. For k ≥ 3,
however, the Ekeland-Hofer capacities cEH
                                        k   on RCT do not have the product
property. As an example, for U = B 4 (4) and V = E(3, 8) we have

                 cEH                          EH     EH
                                                             
                  3 (U × V ) = 7 < 8 = min c3 (U ), c3 (V ) .



                                        21
Problem 10. Characterize the collections of (generalized) capacities on poly-
discs that have the product (resp. stability) property.

Next consider a collection c of generalized capacities on open subsets Op 2n .
In general, it will not be stable. However, we can stabilize c to obtain stable
generalized capacities c± : ∞        2n
                            `
                              n=1 Op    → [0, ∞],
        c+ (U ) := lim sup c(U × R2k ),        c− (U ) := lim inf c(U × R2k ).
                    k→∞                                   k→∞
                                                      `∞
Notice that c(U ) = c+ (U ) = c− (U ) for all U ∈ n=1 Op 2n if and only if c is
stable. If c consists of capacities and there exist constants a, A > 0 such that
                                         
               a ≤ c B 2n (1) ≤ c Z 2n (1) ≤ A         for all n ∈ N,

then c± are collections of capacities. Thus there exist plenty of stable capacities
on Op 2n . However, we have
Problem 11. Decide stability of specific collections of capacities on Conv 2n or
Op 2n , e.g.: Gromov radius, Ekeland-Hofer capacity, Lagrangian capacity, and
the embedding capacity cP of the unit cube.
             12. Does there exist a collection of capacities on ∞          2n
                                                               `
Problem
`∞                                                                n=1 Conv    or
          2n
  n=1  Op     with the product property?


3.9    Higher order capacities ?
Following [45], we briefly discuss the concept of higher order capacities. Consider
a symplectic category C ⊂ Symp 2n containing Ell 2n and fix d ∈ {1, . . . , n}. A
symplectic d-capacity on C is a generalized capacity satisfying

(d-Nontriviality) 0 < c(B) and
                     (
                        c B 2d (1) × R2(n−d) < ∞,
                                             

                        c B 2(d−1) (1) × R2(n−d+1) = ∞.
                                                  


For d = 1 we recover the definition of a symplectic capacity, and for d = n the
volume capacity cvol is a symplectic n-capacity.
Problem 13. Does there exist a symplectic d-capacity on a symplectic category
C containing Ell 2n for some d ∈ {2, . . . , n − 1}?

Problem 13 on Symp 2n is equivalent to the following symplectic embedding
problem.
Problem 14. Does there exist a symplectic embedding
                 B 2(d−1) (1) × R2(n−d+1) ֒→ B 2d (R) × R2(n−d)                  (15)
for some R < ∞ and d ∈ {2, . . . , n − 1}?

                                          22
Indeed, the existence of such an embedding would imply that no symplectic
d-capacity can exist on Symp 2n . Conversely, if no such embedding exists, then
the embedding capacity cZ2d into Z2d = B 2d (1)×R2(n−d) would be an example of
a d-capacity on Symp 2n . The Ekeland-Hofer capacity cEHd  shows that R ≥ 2 if a
symplectic embedding (15) exists. The known symplectic embedding techniques
are not designed to effectively use the unbounded factor of the target space in
(15). E.g., multiple symplectic
                              √ folding only shows that there exists a function
f : [1, ∞) → R with f (a) < 2a + 2 such that for each a ≥ 1 there exists a
symplectic embedding
                      B 2 (1) × B 2 (a) × R2 ֒→ B 4 (f (a)) × R2
of the form ϕ × id2 , see [88, Section 4.3.2].


4       Ellipsoids and polydiscs
In this section we investigate generalized capacities on the categories of ellipsoids
Ell 2n and polydiscs Pol 2n in more detail. All (generalized) capacities c in
this section are defined on some symplectic subcategory of Op 2n containing
at least one of the above categories and are assumed to have the exhaustion
property (10).


4.1     Ellipsoids
4.1.1    Arbitrary dimension

We first describe the values of the capacities introduced in § 2 on ellipsoids.
The values of the Gromov radius cB on ellipsoids are
                                          
                    cB E(a1 , . . . , an ) = min{a1 , . . . , an }.
More generally, monotonicity implies that this formula holds for all symplectic
capacities c on Op 2n with c(B) = c(Z) = 1 and hence also for π1 cEH     1
                                                                     1 , π cHZ ,
1       2n      Z
π e(·, R ) and c .
The values of the Ekeland-Hofer capacities on the ellipsoid E(a1 , . . . , an ) can
be described as follows [20]. Write the numbers m ai π, m ∈ N, 1 ≤ i ≤ n, in
increasing order as d1 ≤ d2 ≤ . . . , with repetitions if a number occurs several
times. Then
                           cEH
                                                  
                            k  E(a1 , . . . , an ) = dk .
The values of the Lagrangian capacity on ellipsoids are presently not known. In
[17], Cieliebak and Mohnke expect to prove the following
Conjecture 1.
                                                        π
                      cL E(a1 , . . . , an ) =                       .
                                                 1/a1 + · · · + 1/an


                                            23
                             
Since vol E(a1 , . . . , an ) = a1 · · · an vol(B), the values of the volume capacity
on ellipsoids are
                          cvol E(a1 , . . . , an ) = (a1 · · · an )1/n .
                                                  


In view of conformality and the exhaustion property, a (generalized) capacity
on Ell 2n is determined by its values on the ellipsoids E(a1 , . . . , an ) with 0 <
a1 ≤ · · · ≤ an = 1. So we can view each (generalized) capacity c on ellipsoids
as a function
                    c(a1 , . . . , an−1 ) := c (E(a1 , . . . , an−1 , 1))
on the set {0 < a1 ≤ · · · ≤ an−1 ≤ 1}. By Fact 7, this function is continuous.
This identification with functions yields a notion of uniform convergence for
capacities on Ell 2n .
For what follows, it is useful to have normalized versions of the Ekeland-Hofer
capacities, so in dimension 2n we define

                                             cEH
                                              k
                                 c̄k :=                .
                                          [ k+n−1
                                              n   ]π

Proposition 1. As k → ∞, for every n ≥ 2 the normalized Ekeland-Hofer
capacities c̄k converge uniformly on Ell 2n to the normalized symplectic capacity
c∞ given by
                                                        n
                     c∞ (E(a1 , . . . , an )) =                     .
                                                1/a1 + · · · + 1/an
Remark. Note that Conjecture 1 asserts that c∞ agrees with the normalized
Lagrangian capacity c̄L = ncL /π on Ell 2n .
Proof of Proposition 1. Fix ε > 0. We need to show that |c̄k (a) − c∞ (a)| ≤ ε
for every vector a = (a1 , . . . , an ) with 0 < a1 ≤ a2 ≤ · · · ≤ an = 1 and all
sufficiently large k. Abbreviate δ = ε/n.
Case 1. a1 ≤ δ. Then

                 cEH
                  k (a) ≤ kδπ,       c̄k (a) ≤ nδ,         c∞ (a) ≤ nδ

from which we conclude |c̄k (a) − c∞ (a)| ≤ nδ = ε for all k ≥ 1.
Case 2. a1 > δ. Let k ≥ 2 n−1
                           δ + 2. For the unique integer l with

                           πl an ≤ cEH
                                    k (a) < π(l + 1)an

we then have l ≥ 2. In the increasing sequence of the numbers m ai (m ∈ N,
1 ≤ i ≤ n), the first [l an /ai ] multiples of ai occur no later than l an . By the
description of the Ekeland-Hofer capacities on ellipsoids given above, this yields
the estimates
        (l − 1) an        (l − 1) an     (l + 1) an         (l + 1) an
                   + ···+            ≤k≤            + ··· +            .
            a1                an             a1                 an


                                          24
With γ := an /a1 + · · · + an /an this becomes

                             (l − 1)γ ≤ k ≤ (l + 1)γ.

Using γ ≥ n, we derive the inequalities
                        
                k+n−1         k          (l + 1)γ + n   (l + 2)γ
                            ≤ +1≤                     ≤          ,
                    n         n                n            n
                k+n−1              (l − 1)γ
                        
                              k
                            ≥ ≥              .
                    n         n         n

With the definition of c̄k and the estimate above for cEH
                                                       k , we find

                   n l an               cEH
                                         k (a)   n(l + 1)an
                           ≤ c̄k (a) = k+n−1   ≤            .
                  (l + 2)γ             [ n ]π     (l − 1)γ

Since c∞ (a) = n an /γ, this becomes

                         l                     l+1
                            c∞ (a) ≤ c̄k (a) ≤     c∞ (a),
                        l+2                    l−1
which in turn implies
                                                   2c∞ (a)
                            |c̄k (a) − c∞ (a)| ≤           .
                                                    l−1
Since a1 > δ we have
                                n                  k   kδ
                           γ≤     ,     l+1≥         ≥    ,
                                δ                  γ   n
from which we conclude
                                             2       2n
                     |c̄k (a) − c∞ (a)| ≤       ≤         ≤ε
                                            l−1   kδ − 2n
for k sufficiently large.                                                        
We turn to the question whether Ekeland-Hofer capacities generate the space
of all capacities on ellipsoids by suitable operations. First note some easy facts.
Fact 10. An ellipsoid E ⊂ R2n is uniquely determined by its Ekeland-Hofer
capacities cEH      EH
            1 (E), c2 (E), . . . .


Indeed, if E(a) and E(b) are two ellipsoids with ai = bi for i < k and ak < bk ,
then the multiplicity of ak in the sequence of Ekeland-Hofer capacities is one
higher for E(a) than for E(b), so not all Ekeland-Hofer capacities agree.
Fact 11. For every k ∈ N there exist ellipsoids E and E ′ with cEH       EH  ′
                                                                i (E) = ci (E )
               EH        EH   ′
for i < k and ck (E) 6= ck (E ).




                                            25
For example, we can take E = E(a) and E ′ = E(b) with a1 = b1 = 1, a2 =
k − 1/2, b2 = k + 1/2, and ai = bi = 2k for i ≥ 3. So formally, every generalized
capacity on ellipsoids is a function of the Ekeland-Hofer capacities, and the
Ekeland-Hofer capacities are functionally independent. However, Ekeland-Hofer
capacities do not form a generating system for symplectic capacities on Ell 2n
(see Example 10 below), and on bounded ellipsoids each finite set of Ekeland-
Hofer capacities is determined by the (infinitely many) other Ekeland-Hofer
capacities:
Lemma 1. Let d1 ≤ d2 ≤ . . . be an increasing sequence of real numbers obtained
from the sequence cEH          EH
                    1 (E) ≤ c2 (E) ≤ . . . of Ekeland-Hofer capacities of a
                         2n
bounded ellipsoid E ∈ Ell by removing at most N0 numbers. Then E can be
recovered uniquely.

Proof. We first consider the special case in which E = E(a1 , . . . , an ) is such that
ai /aj ∈ Q for all i, j. In this case, the sequence d1 ≤ d2 ≤ . . . contains infinitely
many blocks of n consecutive equal numbers. We traverse the sequence until
we have found N0 + 1 such blocks, for each block dk = dk+1 = · · · = dk+n−1
recording the number gk := dk+n − dk . The minimum of the gk for the N0 + 1
first blocks equals a1 . After deleting each occurring positive integer multiple of
a1 once from the sequence d1 ≤ d2 ≤ . . . , we can repeat the same procedure to
determine a2 , and so on.
In general, we do not know whether or not ai /aj ∈ Q for all i, j. To reduce to the
previous case, we split the sequence d1 ≤ d2 ≤ . . . into (at most n) subsequences
of numbers with rational quotients. More precisely we traverse the sequence,
grouping the di into increasing subsequences s1 , s2 , . . . , where each new number
is added to the first subsequence sj whose members are rational multiples of it.
Furthermore, in this process we record for each sequence sj the maximal length
lj of a block of consecutive equal numbers seen so far. We stop when

  (i) the sum of the lj equals n, and
 (ii) each subsequence sj contains at least N0 + 1 blocks of lj consecutive equal
      numbers.

Now the previously described procedure in the case that ai /aj ∈ Q for all i, j
can be applied for each subsequence sj separately, where lj replaces n in the
above argument.
Remark. If the volume of E is known, one does not need to know N0 in Fact 1.
The proof of this is left to the interested reader.                       ♦
The set of Ekeland-Hofer capacities does not form a generating system for sym-
plectic capacities on Ell 2n . Indeed, the volume capacity cvol is not the pointwise
limit of homogeneous monotone functions of Ekeland-Hofer capacities:




                                          26
Example 10. Consider the ellipsoids E = E(1, . . . , 1, 3n +1) and F = E(3, . . . , 3)
in Ell 2n . As is easy to see,

                         cEH       EH
                          k (E) < ck (F )          for all k.                     (16)

Assume that fi is a sequence of homogeneous monotone functions of Ekeland-
Hofer capacities which converge pointwise to cvol . By (16) and the monotonicity
of the fi we would find that cvol (E) ≤ cvol (F ). This is not true.


Problem 15. Do the Ekeland-Hofer capacities together with the volume capacity
form a generating system for symplectic capacities on Ell 2n ?

If the answer to this problem is “yes”, this is a very difficult problem as Lemma 2
below illustrates.

4.1.2   Ellipsoids in dimension 4

A generalized capacity
                      on ellipsoids in dimension 4 is represented by a function
c(a) := c E(a, 1) of a single real variable 0 < a ≤ 1. This function has the
following two properties.

(i) The function c(a) is nondecreasing.
(ii) The function c(a)/a is nonincreasing.

The first property follows directly from the (Monotonicity) axiom. The second
propertyfollows from (Monotonicity) and (Conformality): For a ≤ b, E(b, 1) ⊂
E ab a, ab , hence c(b) ≤ ab c(a). Note that property (ii) is equivalent to the
estimate
                                c(b) − c(a)   c(a)
                                            ≤                              (17)
                                   b−a         a
for 0 < a < b, so the function c(a) is Lipschitz continuous at all a > 0. We will
restrict our attention to normalized (generalized) capacities, so the function c
also satisfies

(iii) c(1) = 1.

An ellipsoid E(a1 , . . . , an ) embeds into E(b1 , . . . , bn ) by a linear symplectic
embedding only if ai ≤ bi for all i, see [50]. Hence for normalized capacities
on the category LinEll 4 of ellipsoids with linear embeddings as morphisms,
properties (i), (ii) and (iii) are the only restrictions on the function c(a). On
Ell 4 , nonlinear symplectic embeddings (”folding”) yield additional constraints
which are still not completely known; see [88] for the presently known results.
By Fact 1, the embedding capacities cB and cB are the smallest, resp. largest,
normalized capacities on ellipsoids. By Gromov’s Nonsqueezing Theorem, cB (a) =


                                          27
c̄1 (a) = a. The function cB (a) is not completely known. Fact 1 applied to c̄2
yields
            cB (a) = 1 if a ∈ 12 , 1     and cB (a) ≥ 2a if a ∈ 0, 12 ,
                                                                   
                                                √
and Fact 1 applied to cvol yields cB (a) ≥ a. Folding constructions provide
upper bounds for cB (a). Lagrangian folding [96] yields cB (a) ≤ l(a) where
                                              1              1
                       (
                           (k + 1)a for k(k+1)    ≤ a ≤ (k−1)(k+1)
                l(a) =         1              1            1
                               k       for k(k+2) ≤ a ≤ k(k+1)

and multiple symplectic folding [88] yields cB (a) ≤ s(a) where the function s(a)
is as shown in Figure 1. While symplectically folding once yields cB (a) ≤ a+1/2
for a ∈ (0, 1/2], the function s(a) is obtained by symplectically folding “infinitely
many times”, and it is known that

                                 cB 12 − cB 21 − ε
                                                    
                                                         8
                         lim inf                       ≥ .
                          ε→0+             ε             7


                                   c̄2
    1               l(a)
                                                           √
                  s(a)                        cvol (a) =       a



    1
    2                                    cB (a) = a
    1
    3

    1
    6

                                                                               a
            1 1 1        1   1            1
           12 8 6        4   3            2                            1


                  Figure 1: Lower and upper bounds for cB (a).

Let us come back to Problem 15.
Lemma 2. If the Ekeland-Hofer capacities and the volume capacity          form a
generating system for symplectic capacities on Ell 2n , then cB 41 = 12 .
                                                                  


We recall that cB 41 = 12 means that the ellipsoid E(1, 4) symplectically em-
                       

beds into B 4 (2 + ε) for every ε > 0.
Proof of Lemma 2. We can assume that all capacities are normalized. By as-
sumption, there exists a sequence fi of homogeneous and monotone functions


                                              28
in the c̄k and in cvol forming normalized      capacities
                                                      which pointwise converge        to
                               1                 4 1                             1
cB . As is easy
                                     
                to see, c̄ k E 4 , 1    ≤ c̄ k B   2    for all k, and  c vol E  4 , 1     =
cvol B 4 21 . Since the fi are monotone and                                           1
            
                                                   converge    in  particular at E 4   , 1
and B 4 12 to cB , we conclude that cB 41 = cB E 41 , 1 ≤ cB B 4 21 = 21 ,
                                                                

which proves Lemma 2.                                                                      

In view of Lemma 2, the following problem is a special case of Problem 15.
Problem 16. Is it true that cB 14 = 12 ?
                                  


The best upper bound for cB 14 presently known is s 41 ≈ 0.6729. Answering
                                                         

Problem 16 in the affirmative means
                                    to construct for each ε > 0 a symplectic
embedding E 41 , 1 → B 4 12 + ε . We do not believe that such embeddings
can be constructed “by hand”. A strategy for studying symplectic embeddings
of 4-dimensional ellipsoids by algebro-geometric tools is proposed in [6].

Our next goal is to represent the (normalized) Ekeland-Hofer capacities as em-
bedding capacities. First we need some preparations.
From the above discussion of cB it is clear that capacities and folding also yield
bounds for the functions cE(1,b) and cE(1,b) . We content ourselves with noting
Lemma 3. Let N ∈ N be given. Then for N ≤ b ≤ N + 1 we have
                              1
                                  for N1+1 ≤ a ≤ 1b ,
               cE(1,b) (a) =   b                                                        (18)
                               a for 1b ≤ a ≤ 1

and
                                                        0 < a ≤ 1b ,
                                        (
                                            a    for
                        cE(1,b) (a) =       1           1         1
                                                                                        (19)
                                            b    for    b   ≤a≤   N,

see Figure 2.
Remark. Note that (19) completely describes cE(1,b) on the whole interval (0, 1]
for 1 ≤ b ≤ 2.

Proof. As both formulas are proved similarly, we only prove (18). The first
Ekeland-Hofer capacity gives the lower bound cE(1,b) (a) ≥ a for all a ∈ (0, 1].
Note that for a ≥ 1b this bound is achieved by the standard embedding, so that
the second claim follows.
For N1+1 ≤ a ≤ N1 we have c̄N +1 (E(a, 1)) = 1 and c̄N +1 (E(1, b)) = b. Hence by
Fact 1 we see that cE(1,b) ≥ 1b on this interval, and this bound is again achieved
by the standard embedding. This completes the proof of (18).

Remark. Consider the functions

                        eb (a) := cE(1,b) (a),       a ∈ (0, 1], b ≥ 1.


                                                29
           1


                                           cE(1,b) (a)

           2
                                                          ?
           5
                     ?
                           cE(1,b) (a)

                                                                              a
                                1   2       1
                                3   5       2                          1


           Figure 2: The functions cE(1,b) (a) and cE(1,b) (a) for b = 25 .


Notice that e1 = cB . By Gromov’s Nonsqueezing Theorem and monotonicity,
              a = cB (a) = cZ (a) ≤ eb (a) ≤ cB (a), a ∈ (0, 1], b ≥ 1.
                                −1
Since eb (a) = cE(a,1) E(1, b)       by equation (7), we see that for each a ∈ (0, 1]
the function b 7→ eb (a) is monotone decreasing and continuous. By (18), it
satisfies eb (a) = a for a ≥ 1/b. In particular, we see that the family of graphs
 graph eb | 1 ≤ b < ∞ fills the whole region between the graphs of cB and

 B
c , cf. Figure 1.                                                                  ♦
The normalized Ekeland-Hofer capacities are represented by piecewise linear
functions c̄k (a). Indeed, c̄1 (a) = a for all a ∈ (0, 1], and for k ≥ 2 the following
formula follows straight from the definition
Lemma 4. Setting m := k+1
                                   
                                 2    , the function c̄k : (0, 1] → (0, 1] is given by
                             ( k+1−i              i−1               i
                                   m a for k+1−i ≤ a ≤ k+1−i
                   c̄k (a) =        i              i              i
                                                                                       (20)
                                   m        for k+1−i   ≤ a ≤ k−i     .
Here i takes integer values between 1 and m.

Figure 3 shows the first six of the c̄k and their limit function c∞ according to
Proposition 1.
In dimension 4, the uniform convergence c̄k → c∞ is very transparent, cf. Fig-
ure 3: One readily checks that c̄k −c∞ ≥ 0 if k is even, in which case kc̄k − c∞ k =
  1                                                                             m−1
k+1 , and that c̄k − c∞ ≤ 0 if k = 2m − 1 is odd, in which case kc̄k − c∞ k = mk
if k ≥ 3. Note that the sequences of the even (resp. odd) c̄k are almost, but not
quite, decreasing (resp. increasing). We still have
Corollary 1. For all r, s ∈ N, we have
                                         c̄2rs ≤ c̄2r .


                                                30
                    1
                                                                               c∞
                                                             c̄4   c̄6
                                                 c̄2
                    3
                    4                                              c̄5

                    1                                  c̄3
                    2


                    1                      c̄1
                    4



                                                                                                      a
                                       1                 1                 3
                                       4                 2                 4             1


                                 Figure 3: The first six c̄k and c∞ .


This will be a consequence of the following characterization of Ekeland-Hofer
capacities.
Lemma 5. Fix k ∈ N and denote by [al , bl ] the interval on which c̄k has the
         l
value [ k+1 ]
              . Then
         2



(a) c̄k ≤ c for every capacity c satisfying c̄k (al ) ≤ c(al ) for all l = 1, 2, . . . , [ k+1
                                                                                            2 ].

(b) c̄k ≥ c for every capacity c satisfying c̄k (bl ) ≥ c(bl ) for all l = 1, 2, . . . , [ k2 ]
    and
                                      c(a)        k
                                 lim       ≤  k+1  .
                                 a→0 a
                                                   2

Proof. Formula (17) and Lemma 4 show that where a normalized Ekeland-Hofer
capacity grows, it grows with maximal slope. In particular, going left from the
left end point al of a plateau a normalized Ekeland-Hofer capacity drops with
the fastest possible rate until it reaches the level of the next lower plateau
and then stays there, showing the minimality. Similarly, going right from the
right end point bl of some plateau a normalized Ekeland-Hofer capacity grows
with the fastest possible rate until it reaches the next higher level, showing the
maximality.

Proof of Corollary 1: The right end points of plateaus for c̄2r are given by
       i
bi = 2r−i . Thus we compute
                                                                                                        
                           i          i  is                               is                           i
             c̄2r                    = =    = c̄2rs                                   = c̄2rs
                        2r − i        r  rs                            2rs − is                     2r − i


                                                              31
and the claim follows from the characterization of c̄2r by maximality.                 

Lemma 3 and the piecewise linearity of the c̄k suggest that they may be repre-
sentable as embedding capacities into a disjoint union of finitely many ellipsoids.
This is indeed the case.
Proposition 2. The normalized Ekeland-Hofer capacity c̄k on Ell 4 is the ca-
pacity cXk of embeddings into the disjoint union of ellipsoids

                                   m        [ k2 ]                  
                                              a                 m m
                          Xk = Z          ∐            E          ,        ,
                                     k        j=1
                                                               k−j j
             k+1 
where m =      2      .

Proof. The proposition  clearly holds for k = 1. We thus fix k ≥ 2. Recall from
                                                                      j
Lemma 4 that c̄k has k2 plateaus, the j th of which has height m
                          
                                                                        and starts at
         j                         j          th
aj := k+1−j and ends at bj := k−j . The j ellipsoid in Proposition 2 is found
as follows: In view of (18) we first select an ellipsoid E(1, b) so that the point 1b
corresponds to bj . This ellipsoid is then rescaled to achieve the correct height
 j                                                  E(α,αb)
m of the plateau (note that by conformality, αc             = cE(1,b) for α > 0). We
obtain the candidate ellipsoid
                                                   
                                           m m
                              Ej = E            ,     .
                                          k−j j

The slope of c̄k following its j th plateau and the slope of cEj after its plateau
both equal k−j
            m . The cylinder is added to achieve the correct      behaviour near
                                                              k
a = 0. We are thus left with showing that for each 1 ≤ j ≤ 2 ,

                          c̄k (a) ≤ cEj (a)       for all a ∈ (0, 1].
                                                                               k
According to Lemma 5 (a) it suffices to show that for each 1 ≤ j ≤              2    and
each 1 ≤ l ≤ k2 we have
             

                                              l
                                c̄k (al ) =     ≤ cEj (al ),                         (21)
                                              m
For l > j, the estimate (21) follows from the fact that c̄k = cEj near bj and from
the argument given in the proof of Lemma 5 (a), and for l = j the estimate (21)
follows from (18) of Lemma 3 by a direct computation. We will deal with the
other cases                                   
                                               k
                                 1≤l<j≤
                                               2
by estimating cEj (al ) from below, using Fact 1 with c = cvol and c = c̄2 .



                                              32
                                        √
Fix j and recall that cvol (E(x, y)) =     xy, so that
                                                    s
              Ej          cvol (E(al , 1))                lj(k − j)
             c (al ) ≥                      =
                                                       (k + 1 − l)m2
                            
                                  m m
                       cvol E k−j     , j
                                                          s
                                                     l        j(k − j)
                                                =       ·
                                                     m      (k + 1 − l)l

gives the desired estimate (21) if j(k − j) ≥ −l2 + (k + 1)l. Computing the roots
l± of this quadratic inequality in l, we find that this is the case if
                            1           p                    
                   l ≤ l− =    k + 1 − 1 + 2k + (k − 2j)2 .
                            2
Computing the normalized second Ekeland-Hofer capacity under the assumption
                                                      2l
that al ≤ 12 , we find that c̄2 (E(al , 1)) = 2al = k+1−l and c̄2 (Ej ) ≤ m
                                                                          j , so that

                       c̄ (E(a , 1))         2l       j   l      2j
      cEj (al ) ≥      2  l         ≥           ·   =   ·         ,
                             m m
                    c̄2 E k−j , j         k + 1 − l   m   m   k + 1−l

which gives the required estimate (21) if

                                    l ≥ k + 1 − 2j.
                1
Note that for   2   ≤ al ≤ 1 we have c̄2 (E(al , 1)) = 1 and hence

                                c̄ (E(a , 1))      j   l
                                2  l         ≥   >
                                      m m
                             c̄2 E k−j , j         m   m

trivially, because we only consider l < j.
So combining the results from the two capacities,
                                                    we find that the desired esti-
mate (21) holds provided either l ≤ l− = 12 k + 1 − 1 + 2k + (k − 2j)2 or
                                                         p

l ≥ k + 1 − 2j. As we only consider l < j, it suffices to verify that
                                       1         p                   
            min(j − 1, k + 1 − 2j) ≤      k + 1 − 1 + 2k + (k − 2j)2
                                       2
for all positive integers j and k satisfying 1 ≤ j ≤ k2 . This indeed follows from
                                                     

another straightforward computation, completing the proof of Proposition 2.

Using the results above, we find a presentation of the normalized capacity c∞ =
limk→∞ c̄k on Ell 4 as embedding capacity into a countable disjoint union of
ellipsoids. Indeed, the space X4r appearing in the statement of Proposition 2 is
obtained from X2r by adding r more ellipsoids. Combined with Proposition 1
this yields the presentation

                                 c∞ = cX        on Ell 4 ,

                                           33
            `∞
where X = r=1 X2r is a disjoint union of countably many ellipsoids. Together
with Conjecture 1, the following conjecture suggests a much more efficient pre-
sentation of c∞ as an embedding capacity. The following result should also be
proved in [17].
Conjecture 2. The restriction of the normalized Lagrangian capacity c̄L to Ell 4
equals the embedding capacity cX , where X is the connected subset B(1) ∪ Z( 12 )
of R4 .

For the embedding capacities from ellipsoids, we have the following analogue of
Proposition 2.
Proposition 3. The normalized Ekeland-Hofer capacity c̄k on Ell 4 is the max-
imum of finitely many capacities cEk,j of embeddings of ellipsoids Ek,j ,
                 c̄k (a) = max { cEk,j (a) | 1 ≤ j ≤ m },     a ∈ (0, 1],
where                                                    
                                              m    m
                             Ek,j = E            ,
                                            k+1−j j
            k+1 
with m =     2       .

Proof. The ellipsoids Ek,j are determined using (19) in Lemma 3. According      to
Lemma 5 (b), this time it suffices to check that for all 1 ≤ j ≤ l ≤ k2 the values
                                                                       
                                                                   l
of the corresponding capacities at the right end points bl = k−l     of plateaus of
c̄k satisfy
                                          l
                            cEk,j (bl ) ≤    = c̄k (bl ).                      (22)
                                          m
The case l = j follows from (19) in Lemma 3 by a direct computation. For the
remaining cases                                
                                               k
                                 1≤j<l≤
                                               2
we use three different methods, depending on the value of j. If j ≤ k−1       3 , then
Fact 1 with c = cvol gives (22) by a computation similar to the one in the proof
                                                 j
of Proposition 2. If j ≥ k+1                              1
                             3 , then aj = k+1−j ≥ 2 , so that (19) in Lemma 3
shows that cEk,j is constant on [aj , 1], proving (22) in this case. Finally, if j = k3
                                   2m
and l ≥ j + 1, then c̄2 (Ek,j ) = k+1−j   and c̄2 (bl ) = 1, so that with Fact 1
                                                k+1−j
                                cEk,j (bl ) ≤         ,
                                                  2m
which is smaller than ml for the values of j and l we consider here. This completes
the proof of Proposition 3.

Here is the corresponding conjecture for the normalized Lagrangian capacity.
Conjecture 3. The restriction of the normalized Lagrangian capacity √         c̄L to
Ell 2n equals the embedding capacity cP (1/n,...,1/n) of the cube of radius 1/ n.


                                            34
4.2     Polydiscs
4.2.1    Arbitrary dimension

Again we first describe the values of the capacities in § 2 on polydiscs.
The values of the Gromov radius cB on polydiscs are
                                           
                    cB P (a1 , . . . , an ) = min{a1 , . . . , an }.
As for ellipsoids, this also determines the values of cEH                     2n      Z
                                                              1 , cHZ , e(·, R ) and c .
According to [20], the values of Ekeland-Hofer capacities on polydiscs are
                    cEH
                                             
                     k   P (a1 , . . . , an ) = kπ min{a1 , . . . , an }.
Using Chekanov’s result [11] that Amin (L) ≤ e(L, R2n ) for every closed La-
grangian submanifold L ⊂ R2n , one finds the values of the Lagrangian capacity
on polydiscs to be
                                                 
                          cL P (a1 , . . . , an ) = π min{a1 , . . . , an }.
                                                                             n
Since vol P (a1 , . . . , an ) = a1 · · · an · π n and vol(B 2n ) = πn! , the values of the
                              

volume capacity on polydiscs are

                         cvol P (a1 , . . . , an ) = (a1 · · · an · n!)1/n .
                                                  

As in the case of ellipsoids, a (generalized) capacity c on Pol 2n can be viewed
as a function
                    c(a1 , . . . , an−1 ) := c (P (a1 , . . . , an−1 , 1))
on the set {0 < a1 ≤ · · · ≤ an−1 ≤ 1}. Directly from the definitions and the
computations above we obtain the following easy analogue of Proposition 1.
Proposition 4. As k → ∞, the normalized Ekeland-Hofer capacities c̄k con-
verge on Pol 2n uniformly to the normalized Lagrangian capacity c̄L = ncL /π.

Propositions 4 and 1 (together with Conjecture 1) give rise to
Problem 17. What is the largest subcategory of Op 2n on which the normalized
Lagrangian capacity is the limit of the normalized Ekeland-Hofer capacities?

4.2.2    Polydiscs in dimension 4

Again, a normalized (generalized) capacity
                                              on polydiscs in dimension 4 is repre-
sented by a function c(a) := c P (a, 1) of a single real variable 0 < a ≤ 1, which
has the properties (i), (ii), (iii). Contrary to ellipsoids, these properties are not
the only restrictions on a normalized capacity on 4-dimensional polydiscs even if
one restricts to linear symplectic embeddings as morphisms. Indeed, the linear
symplectomorphism
                                         1
                           (z1 , z2 ) 7→ √ (z1 + z2 , z1 − z2 )
                                          2

                                            35
of R4 yields a symplectic embedding
                                  a+b √    a+b √
                                                  
                  P (a, b) ֒→ P      + ab,    + ab
                                   2        2
for any a, b > 0, which implies
Fact 12. For any normalized capacity c on LinPol 4 ,
                                         1 a √
                               c(a) ≤     + + a.
                                         2 2

Still, we have the following easy analogues of Propositions 2 and 3.
Proposition 5. The normalized Ekeland-Hofer capacity c̄k on Pol 4 is the ca-
pacity cYk , where                          !
                                    [ k+1
                                       2  ]
                          Yk = Z              ,
                                       k
as well as the capacity cYk′ , where
                                                           !
                                                  [ k+1
                                                     2 ]
                                Yk′    = B                     .
                                                     k

Corollary 2. The identity c̄k = cXk of Proposition 2 extends to Ell 4 ∪ Pol 4 .

Proof. Note that Yk is the first component of the space Xk of Proposition 2. It
thus remains to show that for each of the ellipsoid components Ej of Xk ,

                    c̄k (P (a, 1)) ≤ cEj (P (a, 1)) ,              a ∈ (0, 1].

This follows at once from the observation that for each j we have cEH
                                                                   k (Ej ) =
[ k+1
   2  ]π, whereas c EH
                    k  (P (a, 1)) = kaπ.
Problem 18. Does the equality c̄k = cXk hold on a larger class of open subsets
of R4 ?


References
   [1] S. Bates, Some simple continuity properties of symplectic capacities, The
       Floer memorial volume, 185–193, Progr. Math. 133, Birkhäuser, Basel,
       1995.
   [2] S. Bates, A capacity representation theorem for some non-convex do-
       mains, Math. Z. 227, 571–581 (1998).
   [3] P. Biran, Symplectic packing in dimension 4, Geom. Funct. Anal. 7, 420–
       437 (1997).

                                             36
 [4] P. Biran, A stability property of symplectic packing, Invent. Math. 136,
     123–155 (1999).
 [5] P. Biran, Constructing new ample divisors out of old ones, Duke
     Math. J. 98, 113–135 (1999).
 [6] P. Biran. From symplectic packing to algebraic geometry and back, Eu-
     ropean Congress of Mathematics, Vol. II (Barcelona, 2000), 507–524,
     Progr. Math. 202, Birkhäuser, Basel, 2001.
 [7] P. Biran, Geometry of symplectic intersections, Proceedings of the Inter-
     national Congress of Mathematicians, Vol. II (Beijing, 2002), 241–255,
     Higher Ed. Press, Beijing, 2002.
 [8] P. Biran and K. Cieliebak, Symplectic topology on subcritical manifolds,
     Comment. Math. Helv. 76, 712–753 (2001).
 [9] P. Biran, L. Polterovich and D. Salamon, Propagation in Hamiltonian
     dynamics and relative symplectic homology, Duke Math. J. 119, 65–118
     (2003).
[10] F. Bourgeois, Ya. Eliashberg, H. Hofer, K. Wysocki and E. Zehnder,
     Compactness results in symplectic field theory, Geom. Topol. 7, 799–888
     (2003).
[11] Y. Chekanov, Hofer’s symplectic energy and Lagrangian intersections,
     Contact and Symplectic Geometry, ed. C. B. Thomas, Publ. Newton
     Inst. 8, 296–306 Cambridge University Press (1996)

[12] Y. Chekanov, Lagrangian intersections, symplectic energy, and areas of
     holomorphic curves, Duke Math. J. 95, 213–226 (1998).
[13] Y. Chekanov, talk on a hike on Üetliberg on a sunny day in May 2004.
[14] K. Cieliebak, A. Floer and H. Hofer, Symplectic homology. II. A general
     construction, Math. Z. 218, 103–122 (1995).
[15] K. Cieliebak and E. Goldstein, A note on mean curvature, maslov class
     and symplectic area of Lagrangian immersions, J. Symplectic Geom. 2,
     261–266 (2004).
[16] K. Cieliebak and K. Mohnke, Punctured holomorphic curves and La-
     grangian embeddings, preprint 2003.
[17] K. Cieliebak and K. Mohnke, The Lagrangian capacity, in preparation.
[18] S. Donaldson, Symplectic submanifolds and almost-complex geometry.
     J. Differential Geom. 44, 666–705 (1996).
[19] I. Ekeland and H. Hofer, Symplectic topology and Hamiltonian dynamics,
     Math. Z. 200, 355-378 (1989).

                                    37
[20] I. Ekeland and H. Hofer, Symplectic topology and Hamiltonian dynamics
     II, Math. Z. 203, 553-567 (1990).
[21] I. Ekeland and S. Mathlouthi, Calcul numérique de la capacité symplec-
     tique, Progress in variational methods in Hamiltonian systems and ellip-
     tic equations (L’Aquila, 1990), 68–91, Pitman Res. Notes Math. Ser. 243,
     Longman Sci. Tech., Harlow, 1992.
[22] Y. Eliashberg, Symplectic topology in the nineties, Symplectic geometry.
     Differential Geom. Appl. 9, 59–88 (1998).
[23] Y. Eliashberg, A. Givental and H. Hofer, Introduction to symplectic field
     theory, GAFA 2000 (Tel Aviv, 1999), Geom. Funct. Anal. 2000, Special
     Volume, Part II, 560–673.
[24] Y. Eliashberg and M. Gromov, Convex symplectic manifolds, Several
     complex variables and complex geometry, Part 2 (Santa Cruz, CA, 1989),
     135–162, Proc. Sympos. Pure Math. 52, Part 2, Amer. Math. Soc., Prov-
     idence, RI (1991).
[25] Y. Eliashberg and H. Hofer, Unseen symplectic boundaries, Manifolds
     and geometry (Pisa, 1993) 178–189, Sympos. Math. XXXVI. Cambridge
     Univ. Press 1996.
[26] Y. Eliashberg and H. Hofer, An energy-capacity inequality for the sym-
     plectic holonomy of hypersurfaces flat at infinity, Symplectic geome-
     try, 95–114, London Math. Soc. Lecture Note Ser. 192, Cambridge
     Univ. Press, Cambridge, 1993.
[27] E. Fadell and P. Rabinowitz, Generalized cohomological index theories
     for Lie group actions with an application to bifurcation questions for
     Hamiltonian systems, Invent. Math. 45, 139–173 (1978).
[28] A. Floer, H. Hofer and C. Viterbo, The Weinstein conjecture in P × Cl ,
     Math. Z. 203, 469–482 (1990).
[29] A Floer, H. Hofer, Symplectic homology. I. Open sets in Cn , Math. Z.
     215, 37–88 (1994).
[30] A Floer, H. Hofer and K. Wysocki, Applications of symplectic homology.
     I, Math. Z. 217, 577–606 (1994).
[31] U. Frauenfelder, V. Ginzburg and F. Schlenk, Energy capacity inequali-
     ties via an action selector, math.DG/0402404.
[32] U. Frauenfelder and F. Schlenk, Hamiltonian dynamics on convex sym-
     plectic manifolds, math.SG/0303282.
[33] V. Ginzburg, An embedding S 2n−1 → R2n , 2n − 1 ≥ 7, whose Hamilto-
     nian flow has no periodic trajectories, Internat. Math. Res. Notices 1995,
     83–97.

                                     38
[34] V. Ginzburg, A smooth counterexample to the Hamiltonian Seifert con-
     jecture in R6 , Internat. Math. Res. Notices 1997, 641–650.
[35] V. Ginzburg, The Weinstein conjecture and theorems of nearby and al-
     most existence, The breadth of symplectic and Poisson geometry, 139–
     172, Progr. Math. 232, Birkhäuser Boston, Boston, MA, 2005.
[36] V. Ginzburg and B. Gürel, A C 2 -smooth counterexample to the Hamil-
     tonian Seifert conjecture in R4 , Ann. of Math. 158, 953–976 (2003).
[37] V. Ginzburg and B. Gürel, Relative Hofer–Zehnder capacity and periodic
     orbits in twisted cotangent bundles, Duke Math. J. 123, 1–47 (2004).
[38] V. Ginzburg and E. Kerman, Periodic orbits in magnetic fields in
     dimensions greater than two, Geometry and topology in dynamics
     (Winston-Salem, NC, 1998/San Antonio, TX, 1999), 113–121, Con-
     temp. Math. 246, Amer. Math. Soc., Providence, RI, 1999.
[39] E. Goldstein, Some estimates related to Oh’s conjecture for the Clifford
     tori in CPn , math.DG/0311460.
[40] M. Gromov, Pseudo holomorphic curves in symplectic manifolds, In-
     vent. Math. 82, 307-347 (1985).
[41] D. Hermann, Holomorphic curves and Hamiltonian systems in an open
     set with restricted contact-type boundary, Duke Math. J. 103, 335–374
     (2000).
[42] D. Hermann, Inner and outer hamiltonian capacities, Bull. Soc. Math.
     France 132, 509-541 (2004).
[43] D. Hermann,      Symplectic    capacities   and   symplectic   convexity,
     Preprint 2005.
[44] H. Hofer, On the topological properties of symplectic              maps,
     Proc. Roy. Soc. Edinburgh Sect. A 115, 25–38 (1990).
[45] H. Hofer, Symplectic capacities, Geometry of low-dimensional manifolds,
     2 (Durham, 1989), 15–34, London Math. Soc. Lecture Note Ser. 151,
     Cambridge Univ. Press, Cambridge, 1990.
[46] H. Hofer, Estimates for the energy of a symplectic map, Com-
     ment. Math. Helv. 68, 48–72 (1993).
[47] H. Hofer, Pseudoholomorphic curves in symplectizations with applica-
     tions to the Weinstein conjecture in dimension three, Invent. Math. 114,
     515–563 (1993).
[48] H. Hofer and C. Viterbo, The Weinstein conjecture in the presence of
     holomorphic spheres, Comm. Pure Appl. Math. 45, 583–622 (1992).


                                    39
[49] H. Hofer and E. Zehnder, A new capacity for symplectic manifolds, Anal-
     ysis, et cetera, 405–427, Academic Press, Boston, MA, 1990.
[50] H. Hofer and E. Zehnder, Symplectic Invariants and Hamiltonian Dy-
     namics, Birkhäuser, Basel (1994).
[51] H. Iriyeh, H. Ono and T. Sakai, Integral Geometry and Hamiltonian
     volume minimizing property of a totally geodesic Lagrangian torus in
     S 2 × S 2 , math.DG/0310432.
[52] M.-Y. Jiang, Hofer-Zehnder symplectic capacity for two-dimensional
     manifolds, Proc. Roy. Soc. Edinburgh Sect. A 123, 945–950 (1993).
[53] M.-Y. Jiang, Symplectic embeddings from R2n into some manifolds,
     Proc. Roy. Soc. Edinburgh Sect. A 130, 53–61 (2000).
[54] B. Kruglikov, A remark on symplectic packings, Dokl. Akad. Nauk 350,
     730–734 (1996).
[55] F. Lalonde, Energy and capacities in symplectic topology, Geometric
     topology (Athens, GA, 1993), 328–374, AMS/IP Stud. Adv. Math. 2.1,
     Amer. Math. Soc., Providence, RI, 1997.
[56] F. Lalonde and D. Mc Duff, The geometry of symplectic energy, Ann. of
     Math. 141, 349–371 (1995).
[57] F. Lalonde and D. Mc Duff, Hofer’s L∞ -geometry: energy and stability
     of Hamiltonian flows. I, II, Invent. Math. 122, 1–33, 35–69 (1995).
[58] F. Lalonde and C. Pestieau, Stabilisation of symplectic inequalities and
     applications, Northern California Symplectic Geometry Seminar, 63–71,
     AMS Transl. Ser. 2, 196, AMS, Providence, RI, 1999.
[59] F. Lalonde and M. Pinsonnault, The topology of the space of symplectic
     balls in rational 4-manifolds, Duke Math. J. 122, 347–397 (2004).
[60] G. Lu, The Weinstein conjecture on some symplectic manifolds contain-
     ing the holomorphic spheres, Kyushu J. Math. 52, 331–351 (1998) and
     54, 181–182 (2000).
[61] G. Lu, Symplectic capacities of toric manifolds and combinatorial in-
     equalities, C. R. Math. Acad. Sci. Paris 334, 889–892 (2002).
[62] L. Macarini, Hofer–Zehnder capacity and Hamiltonian circle actions,
     math.SG/0205030.
[63] L. Macarini, Hofer–Zehnder semicapacity of cotangent bundles and sym-
     plectic submanifolds, math.SG/0303230.
[64] L. Macarini, Hofer-Zehnder capacity of standard cotangent bundles,
     math.SG/0308174.

                                    40
[65] L. Macarini and F. Schlenk, A refinement of the Hofer–Zehnder theorem
     on the existence of closed trajectories near a hypersurface, Bull. London
     Math. Soc. 37, 297-300 (2005).
[66] F. Maley, J. Mastrangeli, L. Traynor, Symplectic packings in cotangent
     bundles of tori, Experiment. Math. 9, 435–455 (2000).
[67] D. Mc Duff, Blowing up and symplectic embeddings in dimension 4,
     Topology 30, 409–421 (1991).
[68] D. Mc Duff, Symplectic manifolds with contact type boundaries, Invent.
     Math. 103, 651–671 (1991).
[69] D. McDuff, Symplectic topology and capacities, Prospects in mathematics
     (Princeton, NJ, 1996), 69–81, Amer. Math. Soc., Providence, RI, 1999.
[70] D. McDuff, Geometric variants of the Hofer norm, J. Symplectic
     Geom. 1, 197–252 (2002).
[71] D. Mc Duff and L. Polterovich, Symplectic packings and algebraic geom-
     etry, Invent. Math. 115, 405–429 (1994).
[72] D. Mc Duff and D. Salamon, Introduction to symplectic topology, Sec-
     ond edition. Oxford Mathematical Monographs. The Clarendon Press,
     Oxford University Press, New York, 1998.
[73] D. Mc Duff and D. Salamon, J-holomorphic curves and symplectic topol-
     ogy, AMS Colloquium Publications 52, American Mathematical Society,
     Providence, RI, 2004.
[74] D. McDuff and J. Slimowitz, Hofer-Zehnder capacity and length mini-
     mizing Hamiltonian paths, Geom. Topol. 5, 799–830 (2001).
[75] D. McDuff and L. Traynor, The 4-dimensional symplectic camel and re-
     lated results, Symplectic geometry, 169–182, London Math. Soc. Lecture
     Note Ser. 192, Cambridge Univ. Press, Cambridge (1993).
[76] E. Neduv, Prescribed minimal period problems for convex Hamiltonian
     systems via Hofer-Zehnder symplectic capacity, Math. Z. 236, 99–112
     (2001).
[77] Y.-G. Oh, Second variation and stabilities of minimal Lagrangian sub-
     manifolds in Kähler manifolds, Invent. Math. 101, 501–519 (1990).
[78] Y.-G. Oh, Chain level Floer theory and Hofer’s geometry of the Hamil-
     tonian diffeomorphism group, Asian J. Math. 6, 579–624 (2002).
[79] Y.-G. Oh, Mini-max theory, spectral invariants and geometry of the
     Hamiltonian diffeomorphism group, math.SG/0206092.



                                    41
[80] Y.-G. Oh, Spectral invariants and length minimizing property of Hamil-
     tonian paths, math.SG/0212337.
[81] L. Polterovich, Gromov’s K-area and symplectic rigidity, Geom. Funct.
     Anal. 6, 726–739 (1996).
[82] L. Polterovich, Symplectic aspects of the first eigenvalue, J. Reine Angew.
     Math. 502, 1–17 (1998).
[83] L. Polterovich, Hamiltonian loops from the ergodic point of view, J. Eur.
     Math. Soc. 1, 87–107 (1999).
[84] L. Polterovich, The geometry of the group of symplectic diffeomorphisms,
     Lectures in Mathematics ETH Zürich, Birkhäuser Verlag, Basel, 2001.
[85] P. Rabinowitz, Periodic solutions of Hamiltonian systems, Comm. Pure
     Appl. Math. 31, 157–184 (1978).
[86] P. Rabinowitz, Periodic solutions of a Hamiltonian system on a pre-
     scribed energy surface, J. Differential Equations 33, 336–352 (1979).
[87] F. Schlenk, Symplectic embedding of ellipsoids, Israel J. of Math. 138,
     215–252 (2003).
[88] F. Schlenk, Embedding problems in symplectic geometry, De Gruyter Ex-
     positions in Mathematics 40, Walter de Gruyter Verlag, Berlin (2005).
[89] F. Schlenk, Applications of Hofer’s geometry to Hamiltonian dynamics,
     To appear in Comment. Math. Helv.
[90] M. Schwarz, On the action spectrum for closed symplectically aspherical
     manifolds Pacific J. Math. 193 419–461 (2000).
[91] K.-F. Siburg, Symplectic capacities in two dimensions, Manuscripta
     Math. 78, 149–163 (1993).
[92] J.-C. Sikorav, Systèmes Hamiltoniens et topologie symplectique, Dipar-
     timento di Matematica dell’ Università di Pisa, 1990, ETS EDITRICE
     PISA.
[93] J.-C. Sikorav, Quelques propriétés des plongements lagrangiens, Anal-
     yse globale et physique mathématique (Lyon, 1989), Mém. Soc. Math.
     France 46, 151–167 (1991).
[94] T. Tokieda, Isotropic isotopy and symplectic null sets, Proc. Nat. Acad.
     Sci. U.S.A. 94 13407–13408 (1997).
[95] L. Traynor,    Symplectic homology          via    generating   functions,
     Geom. Funct. Anal. 4, 718–748 (1994).
[96] L. Traynor, Symplectic packing constructions, J. Differential Geom. 42,
     411–429 (1995).

                                     42
 [97] C. Viterbo, Capacités symplectiques et applications (d’après Ekeland-
      Hofer, Gromov), Séminaire Bourbaki, Vol. 1988/89. Astérisque 177-178
      (1989), Exp. No. 714, 345–362.
 [98] C. Viterbo, Plongements lagrangiens et capacités symplectiques de tores
      dans R2n , C. R. Acad. Sci. Paris Sér. I Math. 311, 487–490 (1990).
 [99] C. Viterbo, Symplectic topology as the geometry of generating functions,
      Math. Ann. 292, 685–710 (1992).
[100] C. Viterbo, Metric and isoperimetric problems in symplectic geometry,
      J. Amer. Math. Soc. 13, 411–431 (2000).




                                     43
