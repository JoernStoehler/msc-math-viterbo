---
source: arXiv:1409.5110
fetched: 2025-10-20
---
# Some optimal embeddings of symplectic ellipsoids

                                           Some optimal embeddings of symplectic ellipsoids
                                                                              R. Hind

                                                                           June 15, 2018
arXiv:1409.5110v1 [math.SG] 17 Sep 2014




                                                                              Abstract
                                                    We construct symplectic embeddings in dimension 2n ≥ 6 of ellip-
                                                soids into the product of a 4-ball or 4-dimensional cube with Euclidean
                                                space. A sequence of the embeddings are seen to be optimal by a quan-
                                                titative version of the embedding obstructions introduced in [8].
                                                    In the limiting case when our ellipsoids approach a cylinder we
                                                recover an embedding of Guth, [7]. However for compact ellipsoids
                                                our embedding gives sharper results. At the other end of the scale,
                                                for certain convergent sequences of ellipsoids we reproduce estimates
                                                given by stabilizing 4-dimensional embeddings of McDuff and Schlenk,
                                                [16], in the ball case and Frenkel and Müller, [5], in the cube case.


                                          1    Introduction
                                          Recently there has been much progress on the problem of symplectic embed-
                                          dings, particularly in dimension 4. McDuff and Schlenk in [16] completely
                                          classified ellipsoid embeddings into balls, and Frenkel and Müller in [5] clas-
                                          sified ellipsoid embeddings into cubes. Meanwhile Hutchings has developed
                                          the Embedded Contact Homoloy capacities, see [14], which provide a com-
                                          plete set of invariants in the cases mentioned above, see [18], [15], together
                                          with many others, see [2].
                                               In higher dimensions much less is known. In the present paper we con-
                                          struct a new embedding of ellipsoids using a version of multiple symplectic
                                          folding, see [21]. Furthermore, results from [8] show that a sequence of such
                                          embeddings are optimal. This is slightly surprising since in dimension 4, at
                                          least in the case of ellipsoid embeddings into a ball, we see from [16] and
                                          [21] that folding never gives optimal embeddings (except in the trivial case
                                          when the inclusion is optimal).
                                               We can compare our results with known constructions. In the limiting
                                          case when the ellipsoids approach a cylinder (becoming infinitely thin) we
                                          recover an optimal embedding result of Guth, [7], which is related to the
                                          nonexistence of higher order embedding capacities, but in the compact case
                                          our embedding gives strictly sharper results, see section 1.1 for a discussion



                                                                                  1
of this. This is the case at least for ellipsoids, for polydisks our construction
does not seem to give anything new.
    On the other hand, for certain convergent sequences of rounder ellipsoids
our embedding gives the same result as stabilizing 4-dimensional embeddings
of McDuff and Schlenk, [16], and Frenkel and Müller, [5]. This is discussed
in section 1.2. We emphasize that even though our embeddings and these
stabilizations have the same domain and range, the embeddings themselves
are very different. The McDuff–Schlenk and Frenkel–Müller embeddings are
constructed indirectly using holomorphic curves and Seiberg–Witten theory.
The embeddings constructed here are very concrete.
    Before stating our results we fix some definitions. We study symplectic
embeddings into Euclidean space R2n , with coordinates  Pn     xj , yj , 1 ≤ j ≤ n,
equipped with its standard symplectic form ω = j=1 dxj ∧ dyj . Often it
is convenient to identify R2n with Cn by setting zj = xj + iyj . The basic
domains for symplectic embedding problems are ellipsoids E and polydisks
P which we define as follows.
                                              X π|zj |2
                       E(a1 , . . . , an ) = {          ≤ 1};
                                                  aj
                                                 j

                     P (a1 , . . . , an ) = {π|zj |2 ≤ aj for all j}.
    These are subsets of Cn and so inherit the symplectic structure. A ball
of capacity R is simply an ellipsoid B 2n (R) = E(R, . . . , R). A disk of area
a will sometimes be written D(a) = B 2 (a). It is also convenient to write
λE(a1 , . . . , an ) for E(λa1 , . . . , λan ).
    The following notation will be useful.

Definition 1.1.
                         E(a1 , . . . , an ) ֒→ E(b1 , . . . , bn )
will mean that for all 0 < ǫ < 1 there exists a symplectic embedding (1 −
ǫ)E(a1 , . . . , an ) → E(b1 , . . . , bn ).

Remark 1.2. It is of course true that if there exists a symplectic embed-
ding of the interior E̊(a1 , . . . , an ) → E(b1 , . . . , bn ) then we can also write
E(a1 , . . . , an ) ֒→ E(b1 , . . . , bn ). In dimension 4, that is when n = 2, the
converse is also true. This follows from a theorem of McDuff, [17] Corollary
1.6, saying that the space of ellipsoid embeddings into an ellipsoid is path
connected. It is unknown if the converse is true in higher dimensions. How-
ever we can get an embedding of the interior in any dimension if there exists
a family of symplectic embeddings (1 − ǫ)E(a1 , . . . , an ) → E(b1 , . . . , bn ) de-
pending smoothly on ǫ. This is a consequence of a result of Pelayo and Ngo.c,
[19], Theorem 4.1. It is likely that the main construction in the present pa-
per can be carried out smoothly in a family, but we do not discuss that
here.

                                             2
   Given the notation above, we can state our main theorem as follows.

Theorem 1.3. Let n ≥ 3 and a2 ≥ 1. Then
                                                      3a2
                   E(1, a2 , . . . , an ) ֒→ B 4 (          ) × Cn−2 .
                                                     a2 + 1
Moreover, if a2 = 3d − 1 for a positive integer d and a3 , . . . , an ≥ a2 then
the embedding is optimal, that is, if R < a3a    2
                                               2 +1
                                                     then there are no symplectic
                                     4
embeddings E(1, a2 , . . . , an ) → B (R) × C n−2  .

    An analogous result in dimension 4 would discuss symplectic embeddings
of an ellipsoid into a 4-ball, and this problem has been completely solved
by McDuff and Schlenk in [16]. Relations to the 4-dimensional situation
are discussed in section 1.2. It turns out that the first part of Theorem
1.3, the existence
                 √ of an embedding, is also true in dimension 4 provided
        4    7+3 5
a2 ≤ τ =        2   but is false when a2 > τ 4 . Here τ here is the golden
ratio. The embedding is optimal when n = 2 provided a2 = ggn+2   N
                                                                   , a ratio
of odd Fibonacci numbers (see again section 1.2 for details). This includes
the cases a2 = 3d − 1 and d = 1, 2. For d = 3 we have E(1, 8) ֒→ B 4 (R) if
and only if R ≥ 17                 1                     4
                  6 and for a ≥ 8 36 we have E(1, a) ֒→ B (R) if and only if
     √
R ≥ a.
    There is a similar statement for embeddings into products of a bidisk
and Euclidean space.

Theorem 1.4. Let n ≥ 3 and a2 ≥ 1. Then
                                                2a2    2a2
               E(1, a2 , . . . , an ) ֒→ P (         ,       ) × Cn−2 .
                                               a2 + 1 a2 + 1
Moreover, if a2 = 2d + 1 for a positive integer d and a3 , . . . , an ≥ a2 then
the embedding is optimal, that is, if R < a2a      2
                                                 2 +1
                                                      then there is no symplectic
embedding E(1, a2 , . . . , an ) → P (R, R) × C n−2 .

    Now, by work of Frenkel and Müller, [5], we see that the embedding
result is true also
                  √ in dimension 4 (giving embeddings   into a cube) provided
        2                                   2
a2 ≤ σ = 3 + 2 2, but is false for a2 > σ . Here σ is the silver ratio. The
embedding is optimal if a2 is a certain ratio Pell numbers which includes
the cases a2 = 2d + 1 for d = 1, 2. This is discussed in section 1.2.
    Outline of the paper.
    The embeddings claimed in Theorems 1.3 and 1.4 are constructed in
section 2, and we describe the obstructions which imply sharpness in section
3. However before this we discuss relations to earlier work. In section
1.1 we show why our embedding improves estimates coming from Guth’s
construction. In section 1.2 we describe some intriguing similarities with
optimal 4-dimensional embeddings.


                                               3
1.1    Higher order capacities and Guth’s embedding.
The study of symplectic embeddings was initiated in Gromov’s seminal pa-
per [6], and in particular by his nonsqueezing theorem.
Theorem 1.5. (Gromov [6] Corollary 0.3.A) Suppose a2 , . . . , an ≥ a1 and
b2 , . . . , bn ≥ b1 . Then

                       E(a1 , . . . , an ) ֒→ E(b1 , . . . , bn )

only if a1 ≤ b1 .
    We can replace either or both of the ellipsoids in Theorem 1.5 by poly-
disks and the statement still holds.
    Several years later, Hofer asked in [9] whether the size of the second
factor similarly influences symplectic embeddings. In particular he asked
the following.
    Question. (Hofer, [9], page 17) Does there exist a symplectic embedding

 D(1) × Cn−1 := E(1, ∞, . . . , ∞) ֒→ E(R, R, ∞ . . . , ∞) := B 4 (R) × Cn−2

for any R?
    A positive answer implies that a sort of infinite squeezing in the second
factor is possible, and so a negative answer was expected. It was surprising
then when an ingenious construction of Guth [7] produced embeddings at
least of compact subsets of the domain ellipsoid. Guth’s construction has
since been quantified by Hind and Kerman in [8] and then extended to the
whole interior by Pelayo and Ngo.c. The conclusion is as follows.
Theorem 1.6. (Pelayo-Ngo.c, [20] Theorem 1.2, [19] Theorem 3.3) There
exist symplectic embeddings

                       D̊(1) × Cn−1 → B 4 (3) × Cn−2

and
                      D̊(1) × Cn−1 → P (2, 2) × Cn−2 .
    By letting a3 , . . . , an → ∞ we see that our main Theorems 1.3 and 1.4
recover these embeddings, at least of compact subsets of D̊(1) × Cn−1 . The
existence of the embeddings of Theorem 1.6 imply that there are no higher
order symplectic capacities as defined in [9], see [19].
    Let us briefly recall the construction of Theorem 1.6, at least applied to
large compact subsets. It relies on two lemmas. Here we let Σ(δ) denote a
once punctured 2-torus with a symplectic form of area δ.
Lemma 1.7. For all large S and all δ > 0 there exists a symplectic embed-
ding
                   φ1 : B 2(n−1) (S) → Σ(δ) × Cn−2 .

                                           4
Lemma 1.8. For all ǫ > 0 there exists a δ > 0 such that we have symplectic
embeddings
                     φ2 : D(1 − ǫ) × Σ(δ) → B 4 (3)
and
                         φ′2 : D(1 − ǫ) × Σ(δ) → P (2, 2).
    The compositions (φ2 × id) ◦ (id × φ1 ) and (φ′2 × id) ◦ (id × φ1 ) give
symplectic embeddings D(1 − ǫ) × B 2(n−1) (S) → B 4 (3) × Cn−2 and D(1 −
ǫ) × B 2(n−1) (S) → P (2, 2) × Cn−2 respectively.
    Now, it was shown in [8] that Guth’s construction is optimal in the
following sense.
Theorem 1.9. (Hind-Kerman [8] Theorem 1.2 and Theorem 1.5) If R < 3
then there exist ǫ, S > 0 such that there does not exist a symplectic embedding

                   D(1 − ǫ) × B 2(n−1) (S) → B 4 (R) × Cn−2 .

If R1 < 2 or R2 < 2 then there exist ǫ, S > 0 such that there does not exist
a symplectic embedding

                 D(1 − ǫ) × B 2(n−1) (S) → P (R1 , R2 ) × Cn−2 .

    It follows that the embeddings of Lemma 1.8 must also be optimal, that
is, we have the following.
Theorem 1.10. If R < 3 and δ > 0, then for sufficiently small ǫ > 0 there
does not exist a symplectic embedding

                            D(1 − ǫ) × Σ(δ) → B 4 (R).

If min(R1 , R2 ) < 2 and δ > 0, then for sufficiently small ǫ > 0 there does
not exist a symplectic embedding

                          D(1 − ǫ) × Σ(δ) → P (R1 , R2 ).

    In conclusion, this kind of construction, that is, factoring through φ1 ,
will not give symplectic embeddings even of compact ellipsoids E(1, S, . . . , S)
into a B 4 (R) × Cn−2 with R < 3 or a P (R1 , R2 ) × Cn−2 with R1 < 2. The
purpose of this paper is to show that nevertheless many such embeddings
do exist.

1.2   4-dimensional embeddings and capacities.
A complete understanding of our embedding problem, in the case of ellip-
soid embeddings into products of a ball and Euclidean space, amounts to
computing the capacity function

         f (a2 , . . . , an ) = inf{R|E(1, a2 , . . . , an ) ֒→ B 4 (R) × Cn−2 }.

                                            5
    We know very little about such functions. For example, Figure 1 in [1]
describes our sparse knowledge of embeddings of 6-dimensional ellipsoids
into balls.
    As far as obstructions are concerned (that is, lower bounds on f ), the
only known invariants in dimension 2n ≥ 6 besides those discussed in section
3 are the Ekeland-Hofer capacities, see [3]. These imply the following.

Proposition 1.11. Suppose 1 ≤ a2 ≤ · · · ≤ an . If a2 ≤ 2 then f (a2 , . . . , an ) =
a2 . If a2 ≥ 2 then f (a2 , . . . , an ) ≥ 2.

Proof. First suppose a2 ≤ 2. Then the second Ekeland-Hofer capacity of the
ellipsoid E = E(1, a2 , . . . , an ) is c2 (E) = a2 . The corresponding capacity of
the product Z(R) = B 4 (R) × Cn−2 is c2 (Z(R)) = R and so by monotonicity
of the capacities we have f (a2 , . . . , an ) ≥ a2 . Since the inclusion map gives
an embedding E → Z(a2 ), this is in fact an equality.
     Next suppose a2 ≥ 2. Now the second Ekeland-Hofer capacity of the
ellipsoid E is c2 (E) = 2 and so we see that f (a2 , . . . , an ) ≥ 2.

   For constructions (that is, upper bounds on f ) we note that if we have a
symplectic embedding φ : E(1, a2 ) → B 4 (R) then the product (z1 , z2 , . . . , zn ) 7→
(φ(z1 , z2 ), z3 , . . . , zn ) of course gives an embedding E(1, a2 , . . . , an ) ֒→ B 4 (R)×
Cn−2 . Hence we see immediately that we have the bound

                                f (a2 , . . . , an ) ≤ cB (a2 )                        (1)

where cB is the analogous 4-dimension capacity

                         cB (a) = inf{R|E(1, a) ֒→ B 4 (R)}.

It turns out that the function cB has been completely worked out by McDuff
and Schlenk in [16]. One conclusion is that cB (a) = 2 when 2 ≤ a ≤ 4.
Together with Proposition 1.11 this implies that equation (1) is an equality
when 2 ≤ a2 ≤ 4 provided a3 , . . . , an ≥ a2 . In other words we have the
following.

Corollary 1.12. Suppose 2 ≤ min(a2 , . . . , an ) ≤ 4. Then f (a2 , . . . , an ) = 2.

    A consequence of our main theorem, together with our knowledge of
the function cB from [16], is that inequality (1) is strict when a2 > τ 4 but
conceivably could be an equality when a2 ≤ τ 4 provided a3 , . . . , an ≥ a2 .
    Let us describe a part of the function cB which relates nicely to our
construction. Let g0 = 1 and {gn }∞  n=0 be the sequence of odd Fibonacci
numbers, that is, the sequence beginning 1, 2, 5, 13, 34, . . . . Then we can√
                                    gn+2                             4   7+3 5
define a sequence {bn }∞
                       n=0 by bn = gn . We have limn→∞ bn = τ =            2   .
Given this, Theorem 1.1.2 in [16], together with a little manipulation of
Fibonacci numbers, implies the following.


                                              6
Theorem 1.13. (McDuff-Schlenk, [16], Theorem 1.1.2) For all n ≥ 0 we
have cB (bn ) = ggn+1
                  n+2
                      = b3bn
                         n +1
                              .

    We compare all of this with our main theorem, which can be stated as
follows.

Theorem 1.14.
                                            3a2
                                                 .
                                 f (a2 , . . . , an ) ≤
                                          a2 + 1
Moreover, we have equality in the case when a2 = 3d−1 for a positive integer
d and a3 , . . . , an ≥ a2 .

   Note that f (a2 , . . . , an ) is clearly a nondecreasing function of a2 if the
remaining variables are held fixed. Therefore, assuming a3 , . . . , an ≥ a2 ≥ 2,
the equality part of the statement of the theorem gives
                                                             1
                          f (a2 , . . . , an ) ≥ 3 −                .
                                                       ⌊(a2 + 1)/3⌋
We observe that this matches the bound from Proposition 1.11 when 2 ≤
a2 < 5 (which is sharp when a2 ≤ 4), but improves that bound when
a2 ≥ 5. For 5 ≤ a2 < 8 our bound gives f (a2 , . . . , an ) ≥ 25 . Since we also
have cB (a) = 25 when 5 ≤ a ≤ 25
                               4 inequality (1) implies the following.
                                                                25
Corollary 1.15. Suppose 5 ≤ min(a2 , . . . , an ) ≤             4 .   Then f (a2 , . . . , an ) =
5
2.

    Next we see that if a2 = bn for some n then Theorem 1.14 reproduces
precisely the bound (1) coming from Theorem 1.13. In other words, for
these values of a2 our folding construction gives exactly the same result
as the product map above stabilizing a 4-dimensional embedding. These
embeddings are optimal if n = 0, 1 by the second part of Theorem 1.14, and
so one might naturally conjecture the following.

Conjecture 1.16. Suppose a3 , . . . , an ≥ a2 = bn . Then the product embed-
ding E(1, bn , a3 , . . . , an ) ֒→ B 4 ( ggn+2
                                            n+1
                                                )×Cn−2 is optimal, that is, f (bn , a3 , . . . , an ) =
cB (bn ).

    In general, examining the McDuff-Schlenk function cB (a) in more detail,
we see that the product embedding gives a better result than the construc-
tion of Theorem 1.14 when a2 < τ 4 and a2 6= bn for all n. Our construction
strictly improves on the product map when a2 > τ 4 .
    There is a similar story for embeddings into products of a cube and
Euclidean space. Define

          g(a2 , . . . , an ) = inf{R|E(1, a2 , . . . , an ) ֒→ P (R, R) × Cn−2 }.

Then Theorem 1.4 can be stated as follows.

                                                7
Theorem 1.17.
                                            2a2
                              g(a2 , . . . , an ) ≤
                                                 .
                                          a2 + 1
Moreover, we have equality in the case when a2 = 2d+1 for a positive integer
d and a3 , . . . , an ≥ a2 .
   A consequence is that if a3 , . . . , an ≥ a2 ≥ 3 then
                                                         1
                     g(a2 , . . . , an ) ≥ 2 −                    .
                                                 ⌊(a2 − 1)/2⌋ + 1
In this case the Ekeland-Hofer capacities imply only that g(a2 , . . . , an ) ≥
min(1, a2 , . . . , an ), which we know already from Gromov’s Theorem 1.5.
    The corresponding 4-dimensional embedding capacity is

                      cP (a) = inf{R|E(1, a) ֒→ P (R, R)}

and this was worked out by Frenkel and Müller in [5].
   To describe a connection to Theorem 1.17 we must introduce the se-
quences of Pell numbers {Pn }∞                                        ∞
                             n=0 and half companion Pell numbers {Hn }n=0 .
These are defined by the recursion relations

                    P0 = 0, P1 = 1,         Pn = 2Pn−1 + Pn−2 ,

                   H0 = 1, H1 = 1,         Hn = 2Hn−1 + Hn−2 .
Next we define a sequence      {βn }∞
                                    n=0    by
                                  (
                                      Hn+2
                                       Hn        if n is even
                           βn =       Pn+2
                                       Pn        if n is odd

We have limn→∞ βn = σ 2 . A part of Theorem 1.3 of [5], together with some
manipulations of the recursion relations for Pell and half companion Pell
numbers, gives the following.
Theorem 1.18. (Frenkel-Müller, [5], Theorem 1.3) For all n ≥ 0, cP (βn ) =
 2βn
βn +1 .

   Hence when a2 = βn our embedding result matches the embedding given
by stabilizing a 4-dimensional embedding. By the second statement of The-
orem 1.17 these embeddings are optimal if n = 0, 1 and it is natural to
conjecture that they are optimal for all n.
Conjecture 1.19. Suppose a3 , . . . , an ≥ a2 = βn . Then the product em-
bedding E(1, βn , a3 , . . . , an ) ֒→ P ( β2β n
                                                 , 2βn ) × Cn−2 is optimal, that is
                                            n +1 βn +1
g(βn , a3 , . . . , an ) = cP (βn ).
   In general, we see that the product embedding gives a better result than
the construction of Theorem 1.17 when a2 < σ 2 and a2 6= βn for all n. Our
construction strictly improves on the product map when a2 > σ 2 .

                                             8
2    Main construction
Let S, T ≥ 1. Reordering the first and second factors the goal of this section
is to produce an embedding
                                                        
                                  4  3S         2S     2S
      E(S, 1, T, . . . , T ) ֒→ B         ∩P        ,           × Cn−2 .
                                    S+1        S+1 S+1

In fact, choosing an embedding to fix the z4 , . . . , zn coordinates, it suffices to
work in dimension 6 and construct an embedding of the ellipsoid E(S, 1, T ).
By definition, we need to symplectically
                                         embed E(S,   1, T ) in an arbitrarily
                            4   3S           2S    2S
small neighborhood of B S+1 ∩ P S+1 , S+1 × C. To simplify things
then, when we write an embedding we will always mean an embedding into
an arbitrarily small neighborhood, and subsets will mean subsets of arbi-
trarily small neighborhoods, even if we do not say so explicitly. Similarly we
will let ǫ denote a parameter which can be chosen to be arbitrarily small.
    We begin by recalling some terminology for Hamiltonian diffeomorphisms.
A compactly supported function H : [0, 1] × Cn → R will be called a Hamil-
tonian and sometimes
                  R1     written as (t, z) 7→ H t (z). We can define the norm
of H by |H| = 0 (max H t − min H t )dt. For example, in the case when
H is independent of time (which will usually be the case for us) we have
|H| = max H − min H. Associated to H is a time dependent vector field
i∇H t where ∇H t denotes the usual gradient of a function on Euclidean
space. The corresponding flow exists for all time since H is compactly sup-
ported and we write φt for the time t flow. Then φ = φ1 is called the
Hamiltonian diffeomorphism generated by H. The Hofer norm of φ is the
infimum of |H| over all H which generate φ.
    Notation.
    We set λ = S+1S
                     . As S ≥ 1 we have 12 ≤ λ ≤ 1. Then let V be the subset
of the z1 -plane given by
                                       [
                  V = ([0, 1] × [0, λ]) ([1, 2N + 1] × {0}) .

Here N is an integer of order Sǫ . There exists a symplectomorphism ψ
from D(S), the disk of area S, to an ǫ-neighborhood of V . We can choose
this symplectomorphism such that the disk D(λ) ⊂ D(S) is mapped to a
neighborhood of the square [0, 1] × [0, λ].
     Let D1 = D(λ) and D2 be another disk of area λ lying in the annulus
D(2(λ + ǫ)) \ D(λ + ǫ). We think of D1 and D2 as subsets of the z2 -plane.
     Next let Ai denote the annulus Ai = D(i(T + ǫ)) \ D((i − 1)(T + ǫ)),
which we think of as a subset of the z3 -plane. Let B1 = D(T ) and in general
let Bi be a disk of area T lying inside Ai .
     Now we can define the bidisks Pi in the (z2 , z3 )-plane by Pi = D1 × Bi
if i is odd, and Pi = D2 × Bi if i is even.


                                         9
B3

     {               P3


                                      φ
                                          2




 B
 2

     {                                          P2




                                       φ1


B1

     {                P1
               {
              {

                     D1                        D2

         Figure 1: Polydisks Pi and diffeomorphisms φi .




                               10
   The first few polydisks Pi are illustrated in Figure 1, and the diffeomor-
phisms φi are described in the following lemma.

Lemma 2.1. There exist Hamiltonian diffeomorphisms φi of the (z2 , z3 )
plane, generated by time-independent, compactly supported, Hamiltonian
functions Gi of norm bounded by λ + ǫ, such that φi (Pi ) = Pi+1 .
    Moreover, for i odd and 0 ≤ t ≤ 1, we have

                 φti (Pi ) ⊂ D((1 + t)(λ + ǫ)) × (Ai ∪ Ai+1 ).

For i even we have

                 φti (Pi ) ⊂ D((2 − t)(λ + ǫ)) × (Ai ∪ Ai+1 ).

Proof. We prove existence for the φi when i is odd, the even case is similar.
     There exists a Hamiltonian diffeomorphism ψ2 of the z2 -plane of norm
bounded by λ + ǫ such that the corresponding Hamiltonian is supported in
D(2(λ + ǫ)) and ψ2 (D1 ) = D2 . It is not hard to produce a flow such that
we have ψ2t (D1 ) ⊂ D((1 + t)(λ + ǫ)). By abuse of notation let us also write
ψ2 for the Hamiltonian diffeomorphism ψ2 × id. of the (z2 , z3 )-plane.
     Also, there exists a Hamiltonian diffeomorphism ψ3 of the z3 -plane such
that the corresponding Hamiltonian is supported in Ai ∪ Ai+1 and ψ3 (Bi ) =
Bi+1 . We may assume that both ψ2 and ψ3 are generated by time-independent
Hamiltonian functions, say H2 and H3 respectively.
     Let χ be a cut-off function with χ(x) = 0 when x ≤ λ and χ(x) = 1
when x ≥ λ+ ǫ. Then consider the Hamiltonian function χ(π|z2 |2 )H3 (z3 ) on
the (z2 , z3 )-plane. This generates a diffeomorphism ψ˜3 which is the identity
when restricted to D(λ) × C (where χ(π|z2 |2 ) is identically 0) but with
ψ˜3 (z2 , z3 ) = (z2 , ψ3 (z3 )) when π|z2 |2 ≥ λ + ǫ.
     Hence, since D1 = D(λ) and D2 ⊂ D(2(λ+ǫ))\D(λ+ǫ), the composition
                       −1
φi = ψ˜3 ◦ ψ2 ◦ ψ˜3 maps Pi to Pi+1 and it remains to check that it has the
required properties.
     The diffeomorphism φi is the time-1 map of a Hamiltonian flow ψ˜3 ◦ ψ2t ◦
    −1                                                                      −1
ψ˜3 . This flow is generated by a Hamiltonian function Gi = H2 ◦ ψ˜3 ,
where H2 is a Hamiltonian generating ψ2 , see for instance [13] Proposition
1, page 144. Therefore we may assume φi has norm bounded by λ + ǫ (the
bound on the norm of H2 ).
     Second, since ψ2t (Pi ) ⊂ D((1 + t)(λ + ǫ)) × C, and ψ˜3 preserves π|z2 |2
(since its Hamiltonian commutes with π|z2 |2 ) we also have φti (Pi ) ⊂ D((1 +
t)(λ + ǫ)) × C as required.

   With all of this in place, we can begin the construction.




                                      11
     Step 1. Repositioning the domain.
     Recall the symplectomorphism ψ mapping D(S) to a neighborhood of
V . We apply the map φ0 = ψ × id. to E(S, 1, T ) and think of the image
F0 = φ0 (E(S, 1, T )) as fibered over (a neighborhood of) V with projection
π1 : (z1 , z2 , z3 ) 7→ z1 .
     The fibers of π1 are all contained in the bidisk P (1, T ). However, note
that if (z1 , z2 , z3 ) ∈ E(S, 1, T ) and π|z1 |2 > λ, then π|z2 |2 < 1 − Sλ = λ.
Hence, since ψ maps D(λ) to the square [0, 1] × [0, λ], the fibers of π1 over
points in the interval [1, 2N + 1] × {0} lie inside the smaller bidisk P (λ, T ) =
P1 .
     Step 2. Displacing fibers.
     Let χi be a cut-off function with χi (x) = 0 when x ≤ 2i and χi (x) = 1
when x ≥ 2i + 1. Further, we may assume 0 ≤ χ′i (x) ≤ 1 + ǫ. (Recall
throughout that ǫ is any quantity which can be arbitrarily small.)
     Let Gi be the Hamiltonian function, with norm bounded by λ + ǫ, gen-
erating the diffeomorphism φi described in Lemma 2.1. Then we can define
σi to be the Hamiltonian diffeomorphism generated by χi (Rez1 )Gi (z2 , z3 )
and set
                             Fi = σi (σi−1 . . . (σ1 (F0 )) . . . ).
   The following is a key lemma. Recall that by a subset we really mean a
subset of a small neighborhood.
Lemma 2.2. Let Wi = [2i, 2i + 1] × [min Gi , max Gi ] be a subset of the
z1 -plane. Then
                                              N
                                              [                              N
                                                                             [
             π1 (FN ) ⊂ ([0, 1] × [0, λ])           ([2i − 1, 2i] × {0})           Wi .
                                              i=1                            i=1

    The fibers of FN over [0, 1] × [0, λ] lie in P (1, T ).
    The fibers of FN over [2i − 1, 2i] × {0} lie in Pi .
    The fibers of FN over Wi lie in D(2(λ + ǫ)) × (Ai ∪ Ai+1 ).
    More precisely, and as usual up to an error of order ǫ, the fibers of FN
over a point z1 with Re(z1 ) = 2i + t for some 0 ≤ t ≤ 1 lie in D((1 + t)(λ +
ǫ)) × (Ai ∪ Ai+1 ) if i is odd and D((2 − t)(λ + ǫ)) × (Ai ∪ Ai+1 ) if i is even.
   Note that the union in the statement of the lemma is connected, because
the Gi can be chosen to have compact support.

Proof. We will work by induction and show that
                               k
                               [                            k
                                                            [          [
π1 (Fk ) ⊂ ([0, 1] × [0, λ])         ([2i − 1, 2i] × {0})         Wi       ([2k + 1, 2N + 1] × {0})
                               i=1                          i=1

and that the fibers are as described in the statement of the lemma over points
z1 with Re(z1 ) ≤ 2k + 1. Furthermore, the fibers over [2k + 1, 2N + 1] × {0}
lie in Pk+1 .

                                               12
    As noted at the end of Step 1, this is the case for F0 (where we have
π1 (F0 ) ⊂ V ) so we assume that some Fk−1 has this property.
    The diffeomorphism σk is the identity on {Rez1 ≤ 2k} (where χk = 0)
and acts by (z1 , z2 , z3 ) 7→ (z1 , φk (z2 , z3 )) on {Rez1 ≥ 2k + 1}. Therefore the
fibers of Fk over [2k + 1, 2N + 1] × {0} lie in Pk+1 and it remains to check
the image under σk of points with z1 = 2k + t for some 0 ≤ t ≤ 1.
    Using the complex notation above, these points flow according to the
time independent vector field

            Xk = i(χk (Rez1 )∇Gk (z2 , z3 ) + ∇χk (Rez1 )Gk (z2 , z3 )).

The flow preserves Rez1 and when Rez1 = 2k + t is given by Xk = i(χk (2k +
t)∇Gk (z2 , z3 ) + ∇χk (2k + t)Gk (z2 , z3 )). We have that χk (k + t) is roughly
equal to t and ∇χk (k + t) is bounded by 1 + ǫ. Therefore the component
of the flow in the z1 -plane is parallel to the imaginary axis and has velocity
bounded above and below by max Gk and min Gk respectively, up to terms
of order ǫ. The component in the (z2 , z3 )-plane is roughly it∇Gk and since
Gk is time-independent the time-1 flow is equivalent to flowing by i∇Gk for
time t. By Lemma 2.1 this has image in D((1 + t)(λ + ǫ)) × (Ak ∪ Ak+1 ) if
k is odd and D((2 − t)(λ + ǫ)) × (Ak ∪ Ak+1 ) for k even.

    Step 3. Folding.
    Let τ : C → C be a symplectic immersion of a neighborhood of π1 (FN )
in the z1 -plane with the following properties. The immersion τ restricts to
an embedding on the square W0 = [0, 1] × [0, λ], on each Wi for i ≥ 1, and
on each interval [2i − 1, 2i] × {0}. Moreover, we assume that the images of
the intervals [2i − 1, 2i] × {0} are disjoint from the images of the Wi , and
that the images of the Wi for i odd are disjoint from the images of the Wi
for i even. As the Wi have area roughly λ (the norm of the Gi ), we may
assume that τ maps π1 (FN ) to a neighborhood of D(2λ). In fact, we will
assume that τ maps the Wi for i even into a neighborhood of D(λ) and the
Wi for i odd close to D(2λ) \ D(λ). Such an immersion is illustrated in
Figure 2. Moreover, using [21], Lemma 3.1.5, we can further assume that
points z1 ∈ Wi for i odd with Rez1 = 2i + t are mapped close to D((2 − t)λ).
(Comparing with Lemma 2.2, this means the points whose fibers lie in the
largest bidisks are mapped closer to the center of D(2λ).)
    Finally consider the symplectic immersion φ = τ × id. We claim that
this restricts to give an embedding of FN and furthermore satisfies φ(FN ) ⊂
(B 4 (3λ) ∩ P (2λ, 2λ)) × C. This will complete our construction.
    Firstly, φ|FN is an embedding since by Lemma 2.2, if i 6= j are either
both even or both odd then the fibers of FN over Wi and the fibers over Wj
are disjoint. Also, the fibers over the interval [2i − 1, 2i] × {0} lie in Pi and
so the fibers over different intervals are also disjoint.
    It remains to check the image. Note that the coordinate z2 ∈ D(2λ) (or
really a small neighborhood) for all points in our images. We use here that

                                         13
W0                                                            W3   . . . .
           W1
                                       W2




     W0                                 W2




                     W1                                  W3




                 D(λ)      W0,2,...




          D(2λ) \ D(λ)      W1,3,...



           Figure 2: The immersion τ of the z1 -plane.
                               14
λ ≥ 21 to justify this for the images of points in the fibers over W0 . Therefore
if (z1 , z2 , z3 ) ∈ φ(FN ) and z1 ∈ D(λ) we have (z1 , z2 ) ∈ B 4 (3λ) ∩ P (2λ, 2λ)
as required. Suppose then that π|z1 |2 = (1+t)λ for some 0 ≤ t ≤ 1. Then by
the construction of τ we may assume that (z1 , z2 , z3 ) is the image of a point
(z1′ , z2 , z3 ) in a Wi for i odd with Rez1′ ≤ 2i + 1 − t. Hence by Lemma 2.2 we
have z2 ∈ D((2−t)λ) ⊂ D(2λ). Thus π|z1 |2 +π|z2 |2 ≤ (1+t)λ+(2−t)λ = 3λ
and again we see that (z1 , z2 ) ∈ B 4 (3λ) ∩ P (2λ, 2λ) as required.


3    Embedding obstructions
In this section we show that our embeddings are optimal. This is an almost
immediate consequence of Propositions 3.4 and 3.14 in [8], but to state the
conclusion we need to recall some notation. The open ball B̊ 4 (R) can be
compactified by the complex projective plane CP 2 (R) equipped with a sym-
plectic form where lines have area R. Then we study symplectic embeddings
of an ellipsoid

                    E → B̊ 4 (R) × Cn−2 ⊂ CP 2 (R) × Cn−2 .

Identifying E with its image, the symplectic manifold X = CP 2 (R) × Cn−2 \
                                                                             

E admits a tame almost-complex structure, which, in a neighborhood of
∂E, is biholomorphic to ∂E × (−∞, 0) with a translation invariant almost-
complex structure J, see [4] and [8], section 3.1. Here ∂E carries a contact
form α coming from the standard Liouville form on Cn . This induces a con-
tact structure ξ = {α = 0} on ∂E and a corresponding Reeb vector field v.
Our almost-complex structure structure can be chosen so that J preserves ξ
               ∂
and J(v) = − ∂t  , where t is the coordinate on (−∞, 0). We denote by γ the
closed Reeb orbit {zj = 0, j ≥ 2} ∩ ∂E. Assuming E = E(a1 , . . . , an ) with
aj ≥ a1 for all j, this is the orbit of shortest action. Let rγ be the r times
cover of γ.
    Given all of this, we can study finite energy J-holomorphic curves in X
asymptotic to closed Reeb orbits on ∂E. For the foundations of finite energy
curves see [10], [11], [12]. The key existence result for finite energy curves
which gives our embedding obstructions is the following, see [8], section 3.4.

Theorem 3.1. (Hind-Kerman, [8], Propositions 3.4 and 3.14) Let E ⊂
B̊ 4 (R) × Cn−2 be the image of an ellipsoid E(1, S2 , . . . , Sn ) with Sj ≥ 3d − 1
for all j. Then there exists a finite energy plane of degree d in X asymptotic
to (3d − 1)γ.

    The degree of a finite energy curve in X can be taken to be its inter-
section number with CP 1 (∞) × Cn−2 , where CP 1 (∞) is the line at infinity
in CP 2 (R) and the intersection number is just the number of intersections
counted with multiplicity.


                                        15
   Finite energy curves have positive symplectic area. Since lines in CP 2 (R)
have area R and γ has action 1, a computation using Stokes’ Theorem of
the area of the planes given by Theorem 3.1 therefore implies
                              dR − (3d − 1) > 0
or
                                            1
                                   R>3− .
                                            d
                         3a2        1
If a2 = 3d − 1 then a2 +1 = 3 − d and so this bound on R implies that the
corresponding embedding of Theorem 1.3 is optimal.
      There is a similar construction of finite energy planes into products
which gives obstructions to embeddings of an ellipsoid E into a product
P̊ (R1 , R2 ) × Cn−2 . Now we compactify P̊ (R1 , R2 ) to the product of spheres
S 2 (R1 ) × S 2 (R2 ), where S 2
                               (R) denotes a sphere of area R. Setting Y =
    2          2
 S (R1 ) × S (R2 ) × C    n−2  \ E, we can find a tame almost-complex struc-
ture as before and study finite energy holomorphic curves in Y . The relevant
existence result is then as follows.
Theorem 3.2. Let E ⊂ P̊ (R1 , R2 ) × Cn−2 be the image of an ellipsoid
E(1, S2 , . . . , Sn ) with Sj ≥ 2d + 1 for all j and R1 ≤ R2 . Then there exists
a finite energy plane of bidegree (d, 1) in Y asymptotic to (2d + 1)γ.
    We say that the bidegree of a finite energy curve in Y is (k, l) if its
intersection number with ∞×S 2 (R2 )×Cn−2 is k and its intersection number
with S 2 (R1 ) × ∞ × Cn−2 is l.
    Now applying Stokes’ Theorem to the planes in Theorem 3.2 gives
                              dR1 + R2 > 2d + 1
and setting R1 = R2 this shows that the embedding of Theorem 1.4 is
optimal as claimed when a2 = 2d + 1.


References
 [1] O. Buse and R. Hind, Symplectic embedding of ellipsoids in dimension
     greater than four, Geom. Top., 15 (2011), 2091–2110.
 [2] K. Choi, D. Cristofaro-Gardiner, D. Frenkel, M. Hutchings and V. G.
     B. Ramos, Symplectic embeddings into four-dimensional concave toric
     domains, J. Topol., to appear.
 [3] I. Ekeland and H. Hofer, Symplectic topology and Hamiltonian dynam-
     ics II, Math. Z., 203 (1990), 553–567.
 [4] Y. Eliashberg, A. Givental and H. Hofer, Introduction to symplectic
     field theory, GAFA 2000 (Tel Aviv, 1999), Geom. Funct. Anal., 2000,
     Special Volume, Part II, 560–673.

                                       16
 [5] D. Frenkel and D. Müller, Symplectic embeddings of 4-dimensional el-
     lipsoids into cubes, arXiv:1210.2266.

 [6] M. Gromov, Pseudo-holomorphic curves in symplectic manifolds, Inv.
     Math., 82 (1985), 307–347.

 [7] L. Guth, Symplectic embeddings of polydisks, Inv. Math, 172 (2008),
     477–489.

 [8] R. Hind and E. Kerman, New obstructions to symplectic embeddings,
     Inv. Math., 196 (2014), 383–452.

 [9] H. Hofer, Symplectic capacities. In: Geometry of Low-dimensional
     Manifolds 2 (Durham, 1989). Lond. Math. Soc. Lect. Note Ser., vol.
     151, pp. 15–34. Cambridge Univ. Press, Cambridge (1990).

[10] H. Hofer, K. Wysocki and E. Zehnder, Properties of pseudoholomorphic
     curves in symplectisations I: Asymptotics, Ann. Inst. H. Poincaré Anal.
     Non Lineaire, 13 (1996) , 337–379.

[11] H. Hofer, K. Wysocki and E. Zehnder, Properties of pseudoholomor-
     phic curves in symplectisations II: Embedding controls and algebraic
     invariants, Geom. Funct. Anal., 5 (1995), 337–379.

[12] H. Hofer, K. Wysocki and E. Zehnder, Properties of pseudoholomor-
     phic curves in symplectisations III: Fredholm theory, Topics in nonlin-
     ear analysis, 381-475, Prog. Nonlinear Differential Equations Appl., 35,
     Birkhäuser, Basel, 1999.

[13] H. Hofer and E. Zehnder, Symplectic invariants and Hamiltonian dy-
     namics, Birkhäuser, Basel, 1994.

[14] M. Hutchings, Quantitative embedded contact homology, J. Diffl.
     Geom. 88 (2011), 231-–266.

[15] M. Hutchings, Recent progress on symplectic embedding problems in
     four dimensions, Proc. Natl. Acad. Sci. USA, 108 (2011), 8093-–8099.

[16] D. McDuff and F. Schlenk, The embedding capacity of 4-dimensional
     symplectic ellipsoids, Ann. of Math., 175 (2012), 1191–1282.

[17] D. McDuff, Symplectic embeddings of 4-dimensional ellipsoids, J.
     Topol., 2 (2009), 1–22.

[18] D. McDuff, The Hofer conjecture on embedding symplectic ellipsoids,
     J. Diffl. Geom., 88 (2011), 519-–532.

[19] A. Pelayo and S. V. Ngo.c, The Hofer question on intermediate sym-
     plectic capacities, preprint, arXiv:1210.1537.

                                     17
[20] A. Pelayo and S. V. Ngo.c, Sharp symplectic embeddings of cylinders,
     preprint, arXiv:1304.5250.

[21] F. Schlenk, Embedding problems in symplectic geometry De Gruyter
     Expositions in Mathematics 40. Walter de Gruyter Verlag, Berlin. 2005.




                                    18
