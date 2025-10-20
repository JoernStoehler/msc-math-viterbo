---
source: arXiv:2208.13666
fetched: 2025-10-20
---
# Cube normalized symplectic capacities

                                                         Cube normalized symplectic capacities
                                                     Jean Gutt             Miguel Pereira                 Vinicius G. B. Ramos


                                                                                        Abstract

                                                    We introduce a new normalization condition for symplectic capacities, which
                                                 we call cube normalization. This condition is satisfied by the Lagrangian capacity
arXiv:2208.13666v1 [math.SG] 29 Aug 2022




                                                 and the cube capacity. Our main result is an analogue of the strong Viterbo
                                                 conjecture for monotone toric domains in all dimensions. Moreover, we give a
                                                 family of examples where standard normalized capacities coincide but not cube
                                                 normalized ones. Along the way, we give an explicit formula for the Lagrangian
                                                 capacity on a large class of toric domains.


                                           1     Introduction
                                           The study of symplectic embeddings is at the core of symplectic geometry. One of the
                                           most important tools in this study are symplectic capacities. A symplectic capacity
                                           is a function which assigns to each symplectic manifold (X, ω) of a fixed dimension 2n,
                                           possibly in some restricted class, a number c(X, ω) satisfying the following conditions:
                                            (1) If there exists an embedding ϕ : X1 ,→ X2 such that ϕ∗ ω2 = ω1 , then

                                                                                     c(X1 , ω1 ) ≤ c(X2 , ω2 ).

                                            (2) If r > 0, then
                                                                                   c(X, r · ω) = r · c(X, ω).

                                           After Gromov’s seminal work on symplectic embeddings [Gro85], many capacities were
                                           defined. The majority of these satisfy a normalization condition based on Gromov’s
                                           non-squeezing. More precisely, let B 2n (r) ⊂ Cn denote the ball of radius r and let
                                           Z 2n (r) = B 2 (r) × Cn−1 . As usual, the standard symplectic form on Cn (= R2n ) is defined
                                           by
                                                                                          n
                                                                                          X
                                                                                   ω0 =         dxi ∧ dyi .
                                                                                          i=1

                                           We say that a symplectic capacity is ball normalized1 if
                                            (3) c(B 2n (r), ω0 ) = c(Z 2n (r), ω0 ) = πr2 .
                                              1
                                                A capacity satisfying condition 3) is usually called normalized in the literature. We add the word
                                           “ball” in this paper because we will define another normalization condition below.




                                                                                                1
The central question about ball normalized capacities is the following conjecture, which
apparently has been folkore since the 1990s.
Conjecture 1 (strong Viterbo conjecture). If X is a convex domain in R2n , then all
normalized symplectic capacities of X are equal.
We refer to [GHR20] for a presentation of known results around Conjecture 1. The strong
Viterbo conjecture is, in particular, proven for all monotone toric domains in dimension
4.
Some of the main examples of symplectic capacities that do not satisfy this ball nor-
malization 3) come in sequences, see [EH90, Hut11]. For all of these sequences, the
first capacity is still ball normalized. Two other capacities stand alone not satisfying
3), namely the Lagrangian capacity and the cube capacity, defined by Cieliebak–
Mohnke [CM18] and Gutt–Hutchings [GH18], respectively. In this paper we will intro-
duce a new normalization condition (which we call cube normalization) which is satisfied
by these latter capacities. Our main result is an equivalent of the strong Viterbo con-
jecture for cube normalized capacities.
Theorem 2. All cube normalized symplectic capacities coincide on all monotone toric
domain in any dimension.
This paper is organized as follows. In Section 2, we define the cube normalization and
prove Theorem 2. In Section 3, we provide an explicit formula for the Lagrangian capacity
on a large class of toric domains encompassing monotone toric domains. In Section 4,
we study cube normalized capacities of an interesting class of examples of non-monotone
toric domains and we show that for some parameters, ball normalized capacities coincide
while cube normalized do not. Finally, in Section 5, we find an upper bound for the cube
capacity of a large class of weakly convex toric domains, which is used in Section 4.


2         A new normalization condition
Given a domain2 Ω ⊂ Rn≥0 , define the toric domain
                                        n                                                       o
                    XΩ = µ−1 (Ω) = (z1 , . . . , zn ) ∈ Cn | (π|z1 |2 , . . . , π|zn |2 ) ∈ Ω

where the map µ : Cn → [0, +∞)n : (z1 , . . . , zn ) 7→ (π|z1 |2 , . . . , π|zn |2 ) is the periodic
moment map. We let

                      ∂+ Ω = {p = (p1 , . . . , pn ) ∈ ∂Ω | pi > 0 for i = 1, . . . , n.} .

Recall from [GHR20] that a monotone toric domain is a compact toric domain with
smooth boundary such that for every p ∈ ∂+ Ω, the outward pointing normal vector at
p, ν = (ν1 , . . . , νn ) satifies νi ≥ 0 for i = 1, . . . , n. Note that a monotone toric domain
is the limit of toric domains XΩ0 where Ω0 is bounded by the coordinate hyperplanes
and the graph of a function whose partial derivatives are all negative, see the proof of
[GHR20, Lemma 3.2].
    2
        In this article, a domain is the closure of a non-empty open set.


                                                        2
Consider the following examples of toric domains:

  The   Ball     Bn (a) := µ−1 (ΩBn (a) ), ΩBn (a)   := {x ∈ Rn≥0   | x1 + · · · + xn ≤ a},
  The   Cylinder Zn (a) := µ−1 (ΩZn (a) ), ΩZn (a)   := {x ∈ Rn≥0   | x1 ≤ a},
  The   Cube     Pn (a) := µ−1 (ΩPn (a) ), ΩPn (a)   := {x ∈ Rn≥0   | ∀i = 1, . . . , n : xi ≤ a},
  The   NDUC     Nn (a) := µ−1 (ΩNn (a) ), ΩNn (a)   := {x ∈ Rn≥0   | ∃i = 1, . . . , n : xi ≤ a}.

Here, NDUC stands for non-disjoint union of cylinders. Within those toric domains, the

             a                a
                   ΩB2 (a)                                                     ΩN2 (a)
                                   ΩP2 (a)           ΩZ2 (a)         a

                       a                     a             a               a

           Figure 1: The domains Ω for the aforementioned domains for n = 2

ball normalization condition reformulates as
                                                    
                                  c Bn (1) = c Zn (1) = 1.

This normalization stemmed out of Gromov’s non-squeezing theorem [Gro85] asserting
that there exists a symplectic embedding Bn (a) ,→ Zn (b) if and only if a ≤ b. The
first examples of normalized capacities are the Gromov width cB and the cylindrical
capacity cZ defined for any symplectic manifold (X, ω).

        cB (X, ω) := sup{a | there exists a symplectic embedding Bn (a) −→ X},
        cZ (X, ω) := inf {a | there exists a symplectic embedding X −→ Zn (a)},

Additional examples of normalized symplectic capacities are the Hofer-Zehnder capacity
cHZ defined in [HZ11] and the Viterbo capacity cSH defined in [Vit99]. There are also
useful families of symplectic capacities parametrized by a positive integer k including
the Ekeland-Hofer capacities cEHk  defined in [EH89, EH90] using calculus of variations;
the “equivariant capacities” cGH
                              k   defined in [GH18] using positive equivariant symplectic
homology; and in the four-dimensional case, the ECH capacities cECHk    defined in [Hut11]
using embedded contact homology. For each of these families, the k = 1 capacities cEH  1 ,
cCH
 1  , and c ECH
            1   are normalized. For more   about symplectic capacities in general we refer
to [CHLS07, Sch18] and the references therein.
We now introduce a new normalization based on a “non-squeezing theorem” for the cube.
Theorem 3 ([GH18, Proposition 1.20]). There exists a symplectic embedding Pn (a) ,→
Nn (b) if and only if a ≤ b.
This theorem, together with the previous discussion, motivates the following definition.
Definition 4. We say that a symplectic capacity c is cube normalized if

                                  c(Pn (1)) = c(Nn (1)) = 1.

                                                 3
We now wish to present examples of cube normalized symplectic capacities.
The first example is the cube capacity cP [GH18]

       cP (X, ω) := sup{a | there exists a symplectic embedding Pn (a) −→ X},

A second example is the NDUC capacity cN

       cN (X, ω) := inf{a | there exists a symplectic embedding X −→ Nn (a)}

The first non immediate example of a cube normalized symplectic capacity was intro-
duced by Cieliebak and Mohnke [CM18] and proved to be cube normalized by the second
author in his PhD [Per22]. Let (X, ω) be a symplectic manifold and let L ⊂ X be a
Lagrangian submanifold. The minimal area of L is given by
                                     Z                      Z       
                   Amin (L) := inf        ω σ ∈ π2 (X, L),        ω>0 .
                                      σ                       σ

The Lagrangian capacity of (X, ω) is defined as

           cL (X, ω) := sup{Amin (L) | L is an embedded Lagrangian torus}.

Theorem 5 ([Per22]).
                             cL (Pn (1)) = cL (Nn (1)) = 1.

The second author actually proved a stronger result. For any toric domain XΩ ⊂ Cn ,
define its diagonal to be

                            δΩ := sup{a | (a, . . . , a) ∈ Ω}.

Theorem 6 ([Per22, Theorem 7.65]). If XΩ is a convex or concave toric domain then

                                      cL (XΩ ) = δΩ .

Remark 7. The proof of Theorem 6 uses linearized contact homology, and this result
is stated under some assumptions about this theory. For a more detailed discussion on
these assumptions see [Sie20, Disclaimer 1.11] and [Per22, Section 7.1].
Remark 8. The proof of Theorem 6 uses other symplectic capacities, namely
 (1) the Gutt–Hutchings capacities from [GH18], denoted by cGH
                                                            k ;

 (2) the higher symplectic capacities from [Sie20], denoted by g≤1
                                                                k ;

 (3) the McDuff–Siegel capacities from [MS22], denoted by g̃≤1
                                                            k .

Inspecting the proof of this theorem, one sees that the proof extends word for word for
any monotone toric domain, and that moreover
                         g̃≤1
                           k (XΩ )       g≤1 (XΩ )       cGH (XΩ )
          cL (XΩ ) = lim           = lim k         = lim k         = δΩ
                    k→+∞      k     k→+∞     k      k→+∞     k
for any monotone toric domain XΩ .

                                             4
One can therefore define cube normalized symplectic capacities as follows.
Definition 9. For a nondegenerate Liouville domain (X, λ), let
                                                    cGH
                                                      k (X)
                               cGH
                                inf (X)   := lim inf         ,
                                               k        k
                                                    g≤1
                                                      k (X)
                               g≤1
                                 inf (X) := lim inf          ,
                                               k        k
                                 ≤1                 g̃≤1 (X)
                               g̃inf (X) := lim inf k        .
                                               k        k
                                              ≤1        ≤1
By Remark 8 the symplectic capacities cGH
                                       inf , ginf and g̃inf are cube normalized.

Using the main result of [GR] asserting that for all k ≥ 1 cGH
                                                            k  = cEH
                                                                  k , we have another
cube normalized symplectic capacity
                                                       cEH
                                                        k (X)
                               cEH
                                inf (X) := lim inf            .
                                                  k       k
Note that the main result of [GR] together with Remark 8 shows that for any monotone
toric domain XΩ
                                                cEH (XΩ )
                               cL (XΩ ) = lim k           .
                                           k→+∞     k
This answers (for the monotone toric case) a Question by Cieliebak-Mohnke [CM18] who
asks whether this equality holds for all convex domains in R2n .
The following theorem, which is an analogue of Viterbo’s strong conjecture is our main
result:
Theorem 10. All cube normalized capacities coincide on monotone toric domains in
R2n .

Proof. Let c be a cube normalized symplectic capacity and let XΩ be a monotone toric
domain in R2n . We are going to show that then the value of c(XΩ ) is determined. The
monotonicity of XΩ ensures that

                                Pn (δΩ ) ⊂ XΩ ⊂ Nn (δΩ ).

Then,

                  δΩ = c(Pn (δΩ ))   [since c is cube normalized]
                     ≤ c(XΩ )        [by monotonicity]
                     ≤ c(Nn (δΩ ))   [by monotonicity]
                     = δΩ            [since c is cube normalized].

As a corollary of Theorem 10, we have the following formula for the value of cube
normalized symplectic capacities on monotone toric domains.
Theorem 11. Let c be a cube normalized symplectic capacity and let XΩ be a monotone
toric domain in R2n . Then
                                  c(XΩ ) = δΩ .

                                              5
In view of Theorem 10, it is reasonable to conjecture the following:
Conjecture 12. All cube normalized capacities coincide on convex domains in Cn .
We wish now to make a few comments on what precedes:
Remark 13. The link between monotone toric and convex is studied intensively and
is, at the moment, unclear. All monotone toric domains are dynamically convex3 toric
domains; however the converse is only true in R4 . Examples of monotone toric domains
not symplectomorphic to a convex domain where produced recently [DGZ, CE].
Remark 14. If c is a cube normalized symplectic capacity, then c is not normalized in
the usual sense. Indeed, by Theorem 10, if c is cube normalized then c(Bn (1)) = 1/n
and c(Zn (1)) = 1. We have the following inequalities (for any 2n-dimensional symplectic
manifold (X, ω)):
                          cP (X, ω) ≤ cB (X, ω) ≤ ncP (X, ω).
Those inequalities come from the optimal embeddings
                                     Bn (a) ⊂ Pn (a) ⊂ Bn (na)
We also have
                                       cN (X, ω) ≤ cZ (X, ω)
coming from the inclusion Zn (a) ⊂ Nn (a).
Conjecture 15.
                                      cZ (X, ω) ≤ ncN (X, ω).

The conjecture is true for n = 2. This is the main technical point of [GHR20]. This
amounts to prove that there exists a symplectic embedding
                                        Nn (a) ,→ Zn (na).

Remark 16. The minimal area of a Lagrangian torus, Amin (L), is not continuous in
L. Indeed on a toric domain XΩ , µ−1 (x) is a Lagrangian torus for x = (x1 , . . . , xn ) ∈
(intΩ ∪ ∂+ Ω). By Lemma 17,
                                
                   Amin µ−1 (x) = inf{k1 x1 + · · · + kn xn | k1 , . . . kn ∈ Z}.                 (1)


3       Computing of the Lagrangian capacity for a more
        general family of toric domains
In this section, we will see how one can use Theorem 6 to compute the Lagrangian
capacity for a larger class of toric domains which are not necessarily monotone (see
Theorem 18 below). For a toric domain XΩ , define
                                   ηΩ := inf{a | XΩ ⊂ Nn (a)}.
    Convexity is not a symplectically invariant property. This was already pointed out a long time ago
    3

but only a few symplectic substitutions have been suggested. The most prominent one is dynami-
cal convexity, introduced in [HWZ98], where they show that strict convexity guarantees dynamical
convexity.


                                                  6
Notice that if XΩ is convex or concave, then δΩ = ηΩ . To prove Theorem 18, we will
make use of the following lemma:
Lemma 17 ([Per22, Lemma 6.16]). Let (X, λ) be an exact symplectic manifold and
L ⊂ X be a Lagrangian submanifold. If π1 (X) = 0, then

                       Amin (L) = inf {λ(ρ) | ρ ∈ π1 (L), λ(ρ) > 0} .

Proof. The diagram
                                        ∂                 0
                            π2 (X, L)        π1 (L)           π1 (X)
                                                  λ
                                        ω
                                               R

commutes, where ∂([σ]) = [σ|S 1 ], and the top row is exact.

Theorem 18. Let XΩ be a toric domain. If (ηΩ , . . . , ηΩ ) ∈ ∂Ω then

                                        cL (XΩ ) = ηΩ .

Proof. By definition of ηΩ , we have XΩ ⊂ Nn (ηΩ ). Define T := µ−1 (ηΩ , . . . , ηΩ ). Then
T is an embedded Lagrangian torus in XΩ (see Fig. 2 for an illustration of ηΩ , T , Ω and
ΩNn (ηΩ ) ). Therefore,

                       ηΩ = Amin (T )       [by   Lemma 17]
                          ≤ cL (XΩ )        [by   definition of cL ]
                          ≤ cL (Nn (ηΩ ))   [by   monotonicity]
                          ≤ ηΩ              [by   Theorem 6].

Note that Theorem 18 extends mutatis mutandis, using Eq. (1), to the following
Theorem 19. Let XΩ ⊂ Nn (ηΩ ) be a toric domain in R2n such that there exist a point
x ∈ ∂+ Ω ∩ ∂+ Nn (ηΩ ) of the form x = (k1 ηΩ , . . . , kn ηΩ ) where the ki ∈ N (see Fig. 2).
Then,
                                     cL (XΩ ) = ηΩ .


4     An interesting nonexample
We now study a family of examples coming from [GHR20] of non-monotone toric do-
mains, and we determine when they satisfy the conclusion of Theorem 10.
For 0 < a < 1/2, let Ωa be the convex polygon with corners (0, 0), (1 − 2a, 0), (1 − a, a),
(a, 1 − a) and (0, 1 − 2a), and write Xa = XΩa ; see Fig. 3. Then Xa is a weakly convex
(but not monotone) toric domain.




                                              7
                                        ΩNn (ηΩ )

                                                    ΩT
                       ηΩ                           •


                               Ω


                                       ηΩ           2ηΩ


          Figure 2: Example of XΩ satisfying the assumption in Theorem 19

                                   1
                            1 − 2a




                                                         1 − 2a   1

                               Figure 3: The domain Ωa

Proposition 20. The cubic, Lagrangian and NDUC capacities of Xa are given as fol-
lows.
                                                       1
                                                        
                                cP (Xa ) = min 1 − 2a,     ,
                                                       2
                                           1
                     cL (Xa ) = cN (Xa ) = .
                                           2

Remark 21. It follows from Proposition 20 that cP (Xa ) 6= cN (Xa ) for a > 1/4. But
in [GHR20] it was shown that cB (Xa ) = cZ (Xa ) for all a ≤ 1/3. So for 1/4 < a ≤ 1/3,
the Gromov and cylindrical capacities of Xa coincide, but not the cubic and NDUC
capacities.

Proof. We note that ηΩa = 1/2 for all a ≤ 1/2 and that (1/2, 1/2) ∈ Ωa . So it follows
from Theorem 18 that cL (Xa ) = 1/2. Since Xa ⊂ N2 (a), it follows that
                              1                        1
                                = cL (Xa ) ≤ cN (Xa ) ≤ .
                              2                        2
So cN (Xa ) = 1/2.



                                              8
                                              Ω




                         Figure 4: A weakly convex toric domain XΩ

To compute the cubic capacities, we first observe that
                                    1
                                      
                                Pn      ⊂ Xa , for        0 < a ≤ 1/4,
                                    2
                            Pn (1 − 2a) ⊂ Xa , for      1/4 ≤ a < 1/2.

So cP (Xa ) ≥ min(1 − 2a, 1/2). Since cP (Xa ) ≤ cN (Xa ) = 1/2, it follows that cP (Xa ) =
1/2 = min(1 − 2a, 1/2) for 0 < a ≤ 1/4.
The fact that cP (Xa ) ≤ 1 − 2a for 1/4 < a < 1/2 follows from Theorem 22 below.


5      The cubic capacity of some weakly convex toric
       domains
In this section we obtain an upper bound for the cubic capacity of some non-monotone
toric domains, which will not in general coincide with their NDUC capacity.
A four-dimensional toric domain XΩ is said to be weakly convex4 if Ω ⊂ R2≥0 is convex
and ∂+ Ω is a piecewise smooth curve connecting the two coordinate axes, see Figure 4.
With an extra assumption, we can compute an upper bound for the cubic capacity of
XΩ .
Theorem 22. Let XΩ be a weakly convex toric domain, where ∂+ Ω is parametrized by
the curve (x, y) : [0, 1] → R2≥0 such that y(0) = 0 and x(1) = 0. Suppose that

                                         x0 (0) y 0 (1)
                                                          !
                                     max 0 , 0                ≤ 1.
                                         y (0) x (1)

Then
                                                   x(0) + y(1)
                                      cP (XΩ ) ≤               .
                                                        2
The proof of Theorem 22 uses embedded contact homology. Namely, we need a version
of [Hut16, Theorem 1.20] for weakly convex toric domains. We now explain the context
  4
    Cristofaro-Gardiner defined this to be a convex toric domain in [CG19], but usually a convex toric
domain is defined to be a particular case of this, see [GHR20], for example.


                                                   9
and the modifications that need to be made in the proof of [Hut16, Theorem 1.20] for
our purposes here.
We need some definitions to state a more general version of [Hut16, Theorem 1.20]. Let
XΩ be a weakly convex toric domain. We define a combinatorial Reeb orbit to be a pair
(v, s), where v = (xv , yv ) is a primitive vector in Z2 and s = {0, 1} such that xv ≥ 0 or
yv ≥ 0. A combinatorial orbit set is a finite formal product
                                                     k
                                                          (vi , si )mi ,
                                                     Y
                                             α=
                                                     i=1

where (vi , si ) are distinct combinatorial Reeb orbits and mi ∈ Z≥1 such that mi = 1
whenever si = 0. We define the following numbers.
                       k
                       X
              x(α) =         mi xvi ,                                                                  (2)
                       i=1
                       Xk
              y(α) =         mi yvi ,                                                                  (3)
                       i=1
                                             k
                                             X                                         k
                                                                                       X
              I(α) = x(α) + y(α) +                   mi mj max(xvi yvj , xvj yvi ) +         si mi ,   (4)
                                             i,j=1                                     i=1
                       k
                       X
             m(α) =          mi ,                                                                      (5)
                       i=1
                       Xk
              h(α) =         (1 − si ).                                                                (6)
                       i=1

We note that none of those numbers depend on Ω. The number I(α) is called the
combinatorial ECH index of α. We define the combinatorial action of α to be
                                          k
                                          X
                              AΩ (α) =          mi max{vi · p | p ∈ ∂+ Ω}.
                                          i=1

We now state a version of [Hut16, Definition 1.18] for weakly convex toric domains.
Definition 23. Let XΩ and XΩ0 be weakly convex toric domains and let α and α0 be
combinatorial orbit sets. We write α ≤Ω,Ω0 α0 if the following conditions hold:
  (i) I(α) = I(α0 ),
 (ii) AΩ (α) ≤ AΩ0 (α0 ),
(iii) x(α) + y(α) − h(α)/2 ≥ x(α0 ) + y(α0 ) + m(α0 ) − 1.
The version of [Hut16, Theorem 1.20] that we need is the following result.
Theorem 24. Let XΩ and XΩ0 be weakly convex toric domains such that XΩ ,→ XΩ0 .
Let α0 be an orbit set such that I(α0 ) > 0 and h(α0 ) = 0. Then there is an orbit set α
with I(α) = I(α0 ) and product decompositions
                                             l                       l
                                                             α0 =          αj0 ,
                                             Y                       Y
                                        α=         αj ,
                                             j=1                    j=1

such that:

                                                          10
 (a) αj ≤Ω,Ω0 αj0 ,
 (b) Given i, j, if αi = αj or αi0 = αj0 , then αi and αj have no combinatorial Reeb orbits
     in common with s = 1.
 (c) For any ∅ =
               6 S ⊂ {1, . . . , l},
                                                                        

                                                                         αj0  > 0.
                                           Y                       Y
                                   I            αj  = I 
                                           j∈S                     j∈S


Proof. The proof is essentially the same as the one of [Hut16, Theorem 1.20]. As in the
proof of [GHR20, Theorem 5.6], we first approximate Ω by a domain Ω      e ⊂ Ω such that
∂+ Ω
   e is a smooth curve and the slopes of the tangent lines at the intersections with the
x-axis and y-axis are ε and ε−1 . We observe that for a given orbit set α and δ > 0,
we can define Ωe so that |A (α) − A (α)| < δ. We define Ω   e 0 ⊃ Ω0 satisfying the same
                            Ω         Ω
                                      e
properties as above, c.f. [Hut16, Lemma 5.4]. In particular XΩe ,→ XΩe0 .
We now briefly recall the embedded contact homology (ECH) chain complex. Let (x, y) :
[0, 1] → R2 be a parametrization of ∂+ Ω        e such that y(0) = x(1) = 0. So y 0 (0)/x0 (0) =
x (1)/y (1) = ε. We assume that ε is a small irrational number and that (x00 (t), y 00 (t)) 6= 0
  0     0

for t ∈ [0, 1]. Then the standard Liouville form on R4 restricts to a contact form λ0 on
∂XΩ whose Reeb flow foliates µ−1 ((x(t), y(t)) for each t ∈ [0, 1]. Then for each t ∈]0, 1[
such that x0 (t)/y 0 (t) ∈ Q ∪ {∞}, there is a unique (p, q) ∈ Z2 such that p and q are
relatively prime and
                             (x0 (t), y 0 (t)) = c · (p, q), for c > 0.
So the torus Tp,q := µ−1 ((x(t), y(t)) is foliated by closed Reeb orbits. Note that T(p,q) is
uniquely determined by (p, q) since XΩ is weakly convex. For a Reeb orbit γ ∈ Tp,q , its
symplectic action is defined by                   Z
                                       AΩe (γ) := λ0 .
                                                               γ
It is straight-forward to check that this action doesn’t depend on γ. Indeed for every
γ ∈ Tp,q , it follows from a simple calculation that

                      AΩe (γ) = max{(p, q) · x | x ∈ ∂+ Ω}
                                                        e = A ((p, q), 1).
                                                             Ω
                                                             e

The only other Reeb orbits of λ0 are the two circles µ−1 ((x(0), y(0)) and µ−1 ((x(1), y(1)).
One can check that λ0 is Morse–Bott. Given L > 0, we can perturb the contact form
in neighborhoods of the tori Tp,q for which AΩe (γ) < L for γ ∈ Tp,q , thus obtaining an
elliptic and a hyperbolic Reeb orbit, denoted by e(p,q) and h(p,q) , respectively. This is
explained in more detail in [Hut11] and [CCGF+ 14], for example. Let λ        e denote the
pertubed contact form. The only other closed Reeb orbits of λ with action less than L
are the two circles fibering above (x(0), 0) and (0, y(1)), which are elliptic. We denote
them by e0 and e1 .
An orbit set is a finite formal product α = i αimi , where αi is a simple Reeb orbit and
                                                           Q

mi is positive integer. We always assume that αi 6= αj if i 6= j and mi = 1 if αi is
hyperbolic. The action of an orbit set is defined by
                                                   X
                                    AΩe (α) =              mi AΩe (αi ).
                                                       i


                                                    11
The filtered ECH chain complex ECC L (∂XΩe , λ)
                                             e is the Z/2 vector space generated by
all orbit sets α such that
                                 AΩe (α) < L.
Under the indentification ep,q = ((p, q), 1) and hp,q = ((p, q), 1), we can see orbit sets
as combinatorial orbit sets and their symplectic actions coincide5 . The differential of
ECC L (∂XΩe , λ)
               e is obtained by counting pseudo-holomorphic curves in R × ∂X whose
                                                                                   Ω
                                                                                   e
ECH index is 1. We will not define the ECH index here. Instead it suffices to recall that
in this setting the ECH index gives rise to an absolute index such that for each orbit set
α, I(α) is simply the combinatorial ECH index defined in (4). The fact that the original
definition and the combinatorial definition coincide follows from very similar calculations
to the one in the proof of [Hut16, Lemma 5.4], which uses previous calculations from the
proof of [CCGF+ 14, Lemma 3.3]. Here we have a max instead of a min, because of the
opposite concavity, as in [Hut16, Lemma 5.4]. It is worth noting that the calculation of
the first Chern class [CCGF+ 14, (3.14)] is almost identical and in our case it gives
                                      cτ (α) = x(α) + y(α)                                       (7)
as defined in (2) and (3).
The rest of the argument uses the cobordism map in ECH and the J0 -invariant. It is
identical to the proof of [Hut16, Theorem 1.20] using (7), where we note that the original
and the combinatorial definitions of h and m coincide.

We can now prove Theorem 22.

Proof of Theorem 22. Suppose that P2 (a) ,→ XΩ . We can find a weakly convex toric
domain XΩ0 ⊃ XΩ such that the tangent lines to the curve ∂+ Ω0 at the x and y axes
have slopes 1 − δ and 1 + δ for some small δ > 0, respectively. For each L > 0 sufficiently
large and ε > 0, we can choose XΩ0 so that
                  |AΩ0 (e1,−1 ) − x(0)| < ε         and      |AΩ0 (e−1,1 ) − y(1)| < ε,          (8)
and that
                                   |AΩ (ep,q ) − AΩ0 (ep,q )| < ε,
for all (p, q) such that AΩ (ep,q ) < L.
Now let α0 = ed1,−1 ed−1,1 e21,1 . It follows from Theorem 24 that there exists an orbit set α
and factorizations
                                           l                    l
                                                         α0 =         αj0 ,
                                           Y                    Y
                                    α=           αj ,
                                           j=1                  j=1

satisfying (a), (b) and (c). Using (b) and (c), we conclude that l ≤ 3 and that αi =
ed1,−1
   i
       ed−1,1
          i
              ek1,1 for some k ∈ {0, 1, 2} such that di ≥ d/3. Using (a), it follows from
properties (ii) and (iii) from Definition 23 that
           3k + 2di − 1 = x(αi0 ) + y(αi0 ) + m(αi0 ) − 1 ≤ x(αi ) + y(αi )
                          AP2 (a) (αi )     AΩ0 (αi )   (di (x(0) + y(1)) + k)(1 + ε)
                        =               ≤             <                               .
                                a             a                       a
   5
     To be precise, the symplectic actions with respect to the perturbed contact form is bounded from
the combinatorial action by a small constant which can be as small as desired for a given L


                                                        12
Hence
                              (di (x(0) + y(1)) + k)(1 + ε)
                           a<                               .
                                       2di + 3k − 1
Taking the limit as d → ∞ and then as ε → 0, it follows that
                                        x(0) + y(1)
                                   a≤               .
                                             2
Therefore
                                             x(0) + y(1)
                                cP (XΩ ) ≤               .
                                                  2



Bibliography
[CCGF+ 14] K. Choi, D. Cristofaro-Gardiner, D. Frenkel, M. Hutchings, and V. G. B.
           Ramos. Symplectic embeddings into four-dimensional concave toric do-
           mains. J. Topol., 7:1054–1076, 2014.
        [CE] Julian Chaidez and Oliver Edtmair. The ruelle invariant and convexity in
             higher dimensions. Preprint arXiv:2205.00935.
    [CG19] Dan Cristofaro-Gardiner. Symplectic embeddings from concave toric do-
           mains into convex ones. Journal of Differential Geometry, 112(2):199–232,
           Jun 2019.
 [CHLS07] Kai Cieliebak, Helmut Hofer, Janko Latschev, and Felix Schlenk. Quan-
          titative symplectic geometry. In Dynamics, ergodic theory, and geometry,
          volume 54 of Math. Sci. Res. Inst. Publ., pages 1–44. Cambridge Univ.
          Press, Cambridge, 2007.
    [CM18] Kai Cieliebak and Klaus Mohnke. Punctured holomorphic curves and La-
           grangian embeddings. Inventiones mathematicae, 212(1):213–295, April
           2018.
     [DGZ] Julien Dardennes, Jean Gutt, and Jun Zhang. Symplectic non-convexity of
           toric domains. Preprint arXiv:2203.05448.
    [EH89] Ivar Ekeland and Helmut Hofer. Symplectic topology and Hamiltonian
           dynamics. Math. Z., 200(3):355–378, 1989.
    [EH90] Ivar Ekeland and Helmut Hofer. Symplectic topology and Hamiltonian
           dynamics. II. Math. Z., 203(4):553–567, 1990.
    [GH18] Jean Gutt and Michael Hutchings. Symplectic Capacities from Positive
           S 1 -Equivariant Symplectic Homology. Algebraic & Geometric Topology,
           18(6):3537–3600, October 2018.
  [GHR20] Jean Gutt, Michael Hutchings, and Vinicius Gripp Barros Ramos. Examples
          around the strong Viterbo conjecture. accepted for publication in Journal
          of Fixed Point Theory and Applications, 2020.

                                             13
    [GR] Jean Gutt and Vinicius Gripp Barros Ramos. The equivalence of ekeland-
         hofer and equivariant symplectic homology capacities. Preprint available
         upon request.

 [Gro85] Mikhail Gromov. Pseudoholomorphic curves in symplectic manifolds. In-
         vent. Math., 82:307–347, 1985.

 [Hut11] Michael Hutchings. Quantitative embedded contact homology. J. Differen-
         tial Geom., 88(2):231–266, 2011.

 [Hut16] Michael Hutchings. Beyond ECH capacities. Geom. Topol., 20(2):1085–
         1126, 2016.

[HWZ98] H. Hofer, K. Wysocki, and E. Zehnder. The dynamics on three-dimensional
        strictly convex energy surfaces. Ann. of Math. (2), 148(1):197–289, 1998.

  [HZ11] Helmut Hofer and Eduard Zehnder. Symplectic Invariants and Hamiltonian
         Dynamics. Springer Basel, 2011.

 [MS22] Dusa McDuff and Kyler Siegel. Symplectic Capacities, Unperturbed Curves,
        and Convex Toric Domains. arXiv:2111.00515 [math], February 2022.

 [Per22] Miguel Pereira. Equivariant Symplectic Homology, Linearized Contact Ho-
         mology and the Lagrangian Capacity. PhD thesis, University of Augsburg,
         May 2022.

 [Sch18] Felix Schlenk. Symplectic embedding problems, old and new. Bull. Amer.
         Math. Soc., 55:139–182, 2018.

  [Sie20] Kyler Siegel. Higher Symplectic Capacities. arXiv:1902.01490 [math-ph],
          February 2020.

  [Vit99] Claude Viterbo. Functors and computations in Floer homology with appli-
          cations. I. Geom. Funct. Anal., 9(5):985–1033, 1999.




                                      14
