---
source: arXiv:1101.1069
fetched: 2025-10-20
---
# Recent progress on symplectic embedding problems in four dimensions

                                          Recent progress on symplectic embedding problems
arXiv:1101.1069v2 [math.SG] 15 Feb 2011




                                                          in four dimensions
                                                                        Michael Hutchings


                                                                             Abstract
                                                    We survey some recent progress on understanding when one four-
                                                dimensional symplectic manifold can be symplectically embedded into
                                                another. In 2010, McDuff established a number-theoretic criterion for
                                                the existence of a symplectic embedding of one four-dimensional el-
                                                lipsoid into another. This is related to previously known criteria for
                                                when a disjoint union of balls can be symplectically embedded into a
                                                ball. The new theory of “ECH capacities” gives general obstructions to
                                                symplectic embeddings in four dimensions which turn out to be sharp
                                                in the above cases.

                                              Recall that a symplectic manifold is a pair (X, ω), where X is an oriented
                                          smooth manifold of dimension 2n for some integer n, and ω is a closed 2-
                                          form on X such that the top exterior power ω n > 0 on all of X. The basic
                                          example of a symplectic manifold is Cn = R2n with coordinates zj = xj + iyj
                                          for j = 1, . . . , n, with the standard symplectic form
                                                                                  n
                                                                                  X
                                                                         ωstd =         dxj dyj .                        (1)
                                                                                  j=1

                                          If (X0 , ω0 ) and (X1 , ω1 ) are two symplectic manifolds of dimension 2n, it is
                                          interesting to ask whether there exists a symplectic embedding φ : (X0 , ω0 ) →
                                          (X1 , ω1 ), i.e. a smooth embedding φ : X0 → X1 such that φ∗ ω1 = ω0 .
                                                It turns out that the answer to this question is unknown, or only re-
                                          cent known, even for some very simple examples such as the following. If
                                          a1 , . . . , an > 0, define the ellipsoid
                                                                                                                    
                                                                                                          n      2
                                                                                                         X |zj |    
                                                          E(a1 , . . . , an ) = (z1 , . . . , zn ) ∈ Cn π          ≤1 .
                                                                                                            aj      
                                                                                                    j=1




                                                                                   1
In particular, define the ball

                               B(a) = E(a, . . . , a).

Also define the polydisk

             P (a1 , . . . , an ) = (z1 , . . . , zn ) ∈ Cn π|zj |2 ≤ aj .
                                   


In these examples the symplectic form is taken to be the restriction of ωstd .
    An obvious necessary condition for the existence of a symplectic embed-
ding φ : (X0 , ω0 ) → (X1 , ω1 ) is the volume constraint

                           vol(X0 , ω0 ) ≤ vol(X1 , ω1 ),                    (2)

where the volume of a symplectic manifold is defined by
                                      1
                                        Z
                         vol(X, ω) =        ωn.
                                     n! X

However in dimension greater than two, the volume constraint (2) is far
from sufficient, even for convex subsets of R2n , as shown by the famous:
Gromov nonsqueezing theorem [Gr] There exists a symplectic embedding
B(r) → P (R, ∞, . . . , ∞) if and only if r ≤ R.
     Let us now restrict to the case of dimension four. As we will see below,
symplectic embedding problems are more tractable in four dimensions than
in higher dimensions, among other reasons due to the availability of Seiberg-
Witten theory. But symplectic embedding problems in four dimensions are
still hard. For example, the question of when one four-dimensional ellipsoid
can be symplectically embedded into another was answered only in 2010, by
McDuff. (The analogous question in higher dimensions remains open.) To
state the result, if a and b are positive real numbers, and if k is a positive
integer, define (a, b)k to be the kth smallest entry in the array (am+bn)m,n∈N ,
counted with repetitions. Denote the sequence ((a, b)k+1 )k≥0 by N (a, b). For
example,
                   N (1, 2) = (0, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, . . .).
If {ck }k≥0 and {c′k }k≥0 are two sequences of real numbers indexed by non-
negative integers, the notation {ck } ≤ {c′k } means that ck ≤ c′k for every
k ≥ 0.

Theorem 1 (McDuff [M3]). There exists a symplectic embedding int(E(a, b)) →
E(c, d) if and only if N (a, b) ≤ N (c, d).

                                          2
    Note that given specific real numbers a, b, c, d, it can be a nontrivial
number-theoretic problem to decide whether N (a, b) ≤ N (c, d). For exam-
ple, consider the special case where c = d, i.e. the problem of symplectically
embedding an ellipsoid into a ball. By scaling, we can encode this problem
into a single function f defined as follows: If a is a positive real number,
define f (a) to be the infimum of the set of c ∈ R such that there exists a
symplectic embedding int(E(a, 1)) → B(c). Note that vol(E(a, b)) = ab/2,
                                                 √
so the volume constraint implies that f (a) ≥ a. By Theorem 1,

                    f (a) = inf{c ∈ R | N (a, 1) ≤ N (c, c)}

McDuff-Schlenk [MS] computed f explicitly (without using Theorem 1) and
found in particular that:
                      √
   • If 1 ≤ a ≤ ((1 + 5)/2)4 , then f is piecewise linear.
                          √
   • The interval [((1 + 5)/2)4 , (17/6)2 ] is partitioned into finitely many
                                                              √
     intervals, on each of which either f is linear or f (a) = a.
                                 √
   • If a ≥ (17/6)2 then f (a) = a.

    The starting point for the proof of Theorem 1 is that in [M2], the ellipsoid
embedding problem is reduced to an instance of the ball packing problem:
Given positive real `
                    numbers a1 , . . . , am and a, when does there exist a sym-
plectic embedding m   i=1 int(B(ai )) → B(a)? (Here and below, all of our
balls are four dimensional.) It turns out that the answer to this problem
has been understood in various forms since the 1990’s. The form that is the
most relevant for our discussion is the following:

Theorem 2. There exists a symplectic embedding
                             m
                             a
                                   int(B(ai )) → B(a)
                             i=1

if and only if
                                    m
                                    X
                                          di ai ≤ da                        (3)
                                    i=1

whenever (d1 , . . . , dm , d) are nonnegative integers such that
                             m
                             X
                               (d2i + di ) ≤ d2 + 3d.                       (4)
                             i=1


                                            3
    For example, consider the special case where all of the ai ’s are equal, and
say a = 1. Define ν(m) to be the supremum, over all symplectic embeddings
of the disjoint union of m equal balls into B(1), of the fraction of the volume
of B(1) that is filled.

Proposition 3. (a) (McDuff-Polterovich [MP]) The first 9 values of ν are
    1, 1/2, 3/4, 1, 20/25, 24/25, 63/64, 288/289, 1.

 (b) (Biran [B1]) ν(m) = 1 for all m ≥ 9.

Proof. The upper bounds on ν(m) for m = 2, 3, 5, 6, 7, 8 follow from The-
orem 2 by taking (d1 , . . . , dm , d) to be (1, 1, 1), (1, 1, 0, 1), (1, 1, 1, 1, 1, 2),
(1, 1, 1, 1, 1, 0, 2), (2, 1, 1, 1, 1, 1, 1, 3), and (3, 2, 2, 2, 2, 2, 2, 2, 6) respectively.
For the proof that these upper bounds are sharp see [MP].
    To prove (b), by Theorem            2, it is enough to show that if (d1 , . . . , dm , d)
                          √ P
satisfy
  P (4) then          (1/ m) i di ≤ d. To do so, consider two              P cases. First,
                                                                                         √
if i d2i ≤P     d2 then the Cauchy-Schwarz inequality gives m                  i=1 di ≤ d m.
Second, if i d2i ≥ d2 then
                        m              m                         m
                                                                        !
                    1 X            1X               1           X
                  √         di ≤           di ≤ d +      d2 −        d2i ≤ d.
                    m              3                3
                      i=1          i=1                        i=1




Remark. Traynor [Tr] gives an explicit construction of a maximal sym-
plectic packing of B(1) by m equal open balls when m is a perfect square
or when m ≤ 6, compare Proposition 7 below and see [S] for an extensive
discussion. Explicit constructions for m = 7, 8 are given by Wieck [W].
However explicit maximal packings are not known in general, and the proof
of Theorem 2 is rather indirect.


Axioms for ECH capacities
We now explain how the “only if” parts of Theorems 1 and 2 can be recovered
from the general theory of “ECH capacities”.
   In general, a symplectic capacity is a function c, defined on some class of
symplectic manifolds, with values in [0, ∞], with the following properties:

    • (Monotonicity) If there exists a symplectic embedding (X0 , ω0 ) →
      (X1 , ω1 ), then c(X0 , ω0 ) ≤ c(X1 , ω1 ).

    • (Conformality) If α is a nonzero real number, then c(X, αω) = |α|c(X, ω).

                                              4
See [CHLS] for a review of many symplectic capacities.
   In [H3] a new sequence of symplectic capacities was introduced in four
dimensions, called ECH capacities. If (X, ω) is a symplectic four-manifold
(not necessarily closed or connected), its ECH capacities are a sequence of
numbers
               0 = c0 (X, ω) < c1 (X, ω) ≤ c2 (X, ω) ≤ · · · ≤ ∞.       (5)
We denote the entire sequence by c• (X, ω) = (ck (X, ω))k≥0 . The following
are some basic properties of the ECH capacities:

Theorem 4 ([H3]). The ECH capacities satisfy the following axioms:

   • (Monotonicity) ck is monotone for each k.

   • (Conformality) ck is conformal for each k.

   • (Ellipsoid) c• (E(a, b)) = N (a, b).

   • (Disjoint Union)
                  m
                              !      (m                   m
                                                                     )
                  a                   X                   X
             ck     (Xi , ωi ) = max    cki (Xi , ωi )          ki = k .
                  i=1                       i=1           i=1


    In particular, the ECH capacities give sharp obstructions to symplecti-
cally embedding one ellipsoid into another, or a disjoint union of balls into
a ball:

Corollary 5. (a) If there is a symplectic embedding E(a, b) → E(c, d),
      then N (a, b) ≤ N (c, d).

 (b) If there is a symplectic embedding m
                                       `
                                         i=1 B(ai ) → B(a), then the in-
      equalities (3) hold.

Proof. (a) This follows immediately from the Monotonicity and Ellipsoid
axioms in Theorem 4.
    (b) First note that by the Ellipsoid axiom in Theorem 4, we have ck (B(a)) =
d, where d is the unique nonnegative integer such that

                            d2 + d     d2 + 3d
                                   ≤k≤         .
                               2          2
Now suppose there is a symplectic embedding m
                                                      `
                                                        i=1 B(ai ) → B(a), and
suppose (d1 , . . . , dm , d) are nonnegative integers satisfying (4). Let ki =


                                       5
                        Pm
(d2i + di )/2 and k =     i=1 ki   and k′ = (d2 + 3d)/2. By hypothesis, k ≤ k′ .
We then have
                 m               m                             m
                                                                          !
                 X               X                             a
                       di ai =         cki (B(ai )) ≤ ck             B(ai )
                 i=1             i=1                           i=1
                          ≤ ck (B(a)) ≤ ck′ (B(a)) = da,

where the first inequality holds by the Disjoint Union axiom, the second by
Monotonicity, and the third by (5).


Ball packing
We now review the proof of Theorem      2, and related criteria for the existence
of a symplectic embedding m
                              `
                                i=1 int(B(a i )) → B(a). By scaling, we may
assume that a = 1.
    The first step is to show that the existence of a ball packing is equivalent
                                                                   2
to the existence of a certain symplectic form on CP2 #mCP . There is a
standard symplectic form ω on CP2 such that hL, ωi = 1, where L denotes
the homology class of a line. With this symplectic form, vol(CP2 ) = 1/2,
                                                         2
                                  `m int(B(1)) → CP . Now suppose there
and there is a symplectic embedding
exists a symplectic embedding
                          `m         i=1 B(ai ) → int(B(1)). We then have a
symplectic embedding i=1 B(ai ) → CP2 . We can now perform the “sym-
plectic blowup” along (the image of) each of the balls B(ai ). This amounts
to removing the interior of B(ai ), and then collapsing the fibers of the Hopf
fibration on ∂B(ai ) to points, so that ∂B(ai ) is collapsed to the ith ex-
                                                                         2
ceptional divisor. The result is a symplectic form ω on CP2 #mCP whose
cohomology class is given by
                                                  m
                                                  X
                                 PD[ω] = L −            ai Ei ,               (6)
                                                  i=1

where Ei denotes the homology class of the ith exceptional divisor, and PD
denotes Poincaré duality. Also the canonical class for this symplectic form
(namely −c1 of the tangent bundle as defined using an ω-compatible almost
complex structure) is given by
                                                     m
                                                     X
                             PD(K) = −3L +                    Ei .            (7)
                                                        i=1




                                              6
                                                                            2
    To proceed, define Em to be the set of classes in H2 (CP2 #mCP ) that
have square −1 and can be represented by a smoothly embedded sphere that
is symplectic with respect to some symplectic form ω obtained from blowing
up CP2 . Elements of Em are called “exceptional classes”. One can show that
the set Em does not depend on the choice of ω as above. In fact, Li-Li [LiLi]
used Seiberg-Witten theory to show that Em consists of the set of classes A
such that A2 = A · K = −1 and A is representable by a smoothly embedded
sphere.

Proposition 6. Let a1 , . . . , am > 0. Then the following are equivalent:

 (a) There exists a symplectic embedding
                                 m
                                 a
                                       B(ai ) → int(B(1)).
                                 i=1

                                                               2
 (b) There exists a symplectic form ω on CP2 #mCP satisfying (6) and
     (7).
     Pm 2              Pm
 (c)   i=1 ai < 1, and  i=1 di ai < d whenever

                                          m
                                          X
                                  dL −          di Ei ∈ Em .
                                          i=1

       Pm
 (d)      i=1 di ai < d whenever (d1 , . . . , dm , d) are nonnegative integers, not
       all zero, satisfying (4).

Proof. (a) ⇒ (b) follows from the blowup construction described above.
    (b) ⇒ (a): It is shown in [MP] that if (6) holds and if ω is homotopic
through symplectic forms to a form obtained by blowing up CP2 along small
balls, then one can “blow down” to obtain a ball packing. And it is shown
                                                            2
in [LiLiu] that any two symplectic forms on CP2 #mCP satisfying (7) are
homotopic through symplectic forms.
    (b) ⇒ (c) because ω 2 > 0 and ω has positive pairing with every excep-
tional class.
    (c) ⇒ (b) is proved in [LiLiu]. Actually, by Lemma 1 below, it is enough
to prove the slightly weaker statement that if (c) holds then (a1 , . . . , am ) is
in the closure of the set of tuples satisfying (b). This last statement follows
from earlier work of McDuff [M1, Lem. 2.2] and Biran [B2, Thm. 3.2]. The
idea of the argument is as follows. Without loss of generality, a1 , . . . , am


                                           7
                                   P
are rational. Write A = L − i ai Ei . Let ω0 be a symplectic form on
             2
CP2 #mCP obtained by blowing up CP2 along small balls. We will see
below that for some positive integer n there exists a connected, embedded,
ω0 -symplectic surface C representing the class nA. Then, since A · A > 0
(by the first condition in (c)), the “inflation” procedure in [M1, Lem. 1.1]
allows one to deform ω0 in a neighborhood of C to obtain a symplectic form
ω with cohomology class [ω] = ω0 + r PD(A) for any r > 0. By taking r
large and scaling, this gives a symplectic form whose cohomology class is
arbitrarily close to PD(A).
     To find a surface C as above, choose a generic ω0 -compatible almost com-
plex structure J. By the wall crossing formula for Seiberg-Witten invariants
                                                                               2
[KM1] and Taubes’s “SW⇒Gr” theorem [Ta1], if α ∈ H2 (CP2 #mCP ) is
any class with α2 − K · α ≥ 0 and ω0 · (K − α) < 0, then there exists a J-
holomorphic curve C in the class α. This works for α = nA when n is large.
It turns out that the resulting holomorphic curve C is a connected embed-
ded symplectic surface as desired, unless it includes an embedded sphere Σ
of self-intersection −1 (or a multiple cover thereof) which does not intersect
the rest of C. In this last case, Σ would represent an exceptional class with
A · Σ < 0, contradicting the second condition in (c).
     (d) ⇒ (c): Assume that (d) holds. The first part of (c) follows by an
easy calculus exercise, see [H3], and is also a special case of a general relation
between ECH capacities and symplectic volume discussed
                                                     P          at the end of this
article. To prove the rest of (c),     let A =  dL −     d E
                                                          i i  be  an exceptional
                2
                                   P 2       2
class. Since A = −1, we have          di = d + 1. Furthermore
                                                         P         the adjunction
formula implies that K · AP   = −1, so by (7) we have       di = 3d − 1. Adding
these two equations gives (d2i + di ) = d2 + 3d. We must have d ≥ 0, since
A is represented by an embedded surface which is symplectic with respect
to
P ω0 . If all of the integers di are nonnegative as well, then (d) implies that
    di ai < d as desired. If any of the integers di are negative, then replace
them by 0 and apply (d) to obtain an even stronger inequality.
     (a) ⇒ (d): This follows by invoking the ECH capacities as in Corollary 5.
One can also prove this without using ECH capacities as follows. Suppose
there is a ball packing as in (a), let ω be a symplectic form as in (b) and let J
be a generic ω-compatible almost complex structure. Let (d1 , . . . , dP  m , d) be
nonnegative integers (not all zero) satisfying (4), and write α = dL− i di Ei .
Then α2 − K · α ≥ 0 by (4), and ω · (K − α) < 0 since d1 , . . . , dm , d are
nonnegative, so as in the proof of (c) ⇒ (b) there exists a J-holomorphic
curve C representing the class α. Since this must have positive symplectic
area with respect to ω, we obtain the inequality (3).


                                        8
    As an alternative to the above paragraph, McDuff [M3] proves that (c)
⇒ (d) by an algebraic argument using the explicit description of Em in
[LiLi].

    Theorem 2 now follows from Proposition 6, together with the following
technical lemma:
                                                 `m
Lemma 1. There is a symplectic     `m embedding i=1 int(B(ai )) → B(a) if
there is a symplectic embedding i=1 B(λai ) → int(B(a)) for every λ < 1.

Proof. It is shown in [M1] that any two symplectic embeddings m
                                                               `
                                                                 i=1 B(ai ) →
int(B(a)) are equivalent via a symplectomorphism      of int(B(a)). Conse-
quently, if there is a symplectic embedding m
                                              `
                                                i=1 B(λa i ) → int(B(a)) for
every `λ < 1, then we can obtain a sequence of symplectic embeddings
φn : m  i=1 B((1 − 1/n)ai ) → int(B(a)) such that φn is the restriction of
φn+1 . The `  direct limit of the maps φn then gives the desired symplectic
embedding m     i=1 int(B(ai )) → B(a).


Ellipsoid embeddings
We now explain McDuff’s proof of Theorem 1 using Theorem 2. By a
continuity argument as in Lemma 1, we can assume without loss of generality
that a/b and c/d are rational.
   If a and b are positive real numbers with a/b rational, the weight ex-
pansion W (a, b) is a finite list of real numbers (possibly repeated) defined
recursively as follows:

   • If a < b then W (a, b) = (a) ∪ W (a, b − a).

   • W (a, b) = W (b, a).

   • W (a, a) = (a).

For
`m example, W (5, 3) = (3, 2, 1, 1). If W (a, b) = (a1 , . . . , am ), write B(a, b) =
  i=1 B(ai ). Ellipsoid embeddings are then related to ball packings as fol-
lows:

Proposition 7 (McDuff [M2]). Suppose a/b and c/d are rational with c < d.
Then there is a symplectic embedding int(E(a, b)) → E(c, d) if and only if
there is a symplectic embedding

                       int (B(a, b) ⊔ B(d − c, d)) → B(d).                        (8)


                                          9
Proof. We will only explain the easier direction, namely why an ellipsoid
embedding gives rise to a ball packing. For this purpose consider the mo-
ment map µ : C2 → R2 defined by µ(z1 , z2 ) = π(|z1 |2 , |z2 |2 ). Call two
subsets of R2 “affine equivalent” if one can be obtained from the other by
the action of SL(2, Z) and translations. Note that if U1 , U2 are affine equiv-
alent open sets in the positive quadrant of R2 , then µ−1 (U1 ) and µ−1 (U2 )
are symplectomorphic.
    If a, b > 0, let ∆(a, b) denote the triangle in R2 with vertices (0, 0), (a, 0),
and (0, b). Then E(a, b) = µ−1 (∆(a, b)). If a < b, then ∆(a, b) is the union
(along a line segment) of ∆(a, a) and a triangle which is affine equivalent
to ∆(a, b − a). It follows by induction that if W (a, b) = (a1 , . . . , am ), then
∆(a, b) is partitioned into m triangles, such that the ith triangle is affine
equivalent to ∆(ai , ai ). By Traynor [Tr], there is a symplectic embedding of
int(B(ai )) into µ−1 (int(∆(ai , ai ))). Hence there is a symplectic embedding
                           int(B(a, b)) → int(E(a, b)).                         (9)
Likewise, int(∆(d, d))\∆(c, d) is affine equivalent to int(∆(d−c, d)), so there
is a symplectic embedding
                       int(B(d − c, d)) → B(d) \ E(c, d),
and hence a symplectic embedding
                       int(E(c, d) ⊔ B(d − c, d)) → B(d).                      (10)
If there is a symplectic embedding int(E(a, b)) → E(c, d), then composing
this with the embeddings (9) and (10) gives a symplectic embedding as in
(8).

    The idea of the proof of Theorem 1 is to use the fact that the existence
of an ellipsoid embedding is equivalent to the existence of a ball packing,
and the fact that ECH capacities give a sharp obstruction to the existence
of ball packings, to deduce that ECH capacities give a sharp obstruction to
the existence of ellipsoid embeddings. To proceed with the details, if a• and
a′• are sequences of real numbers indexed by nonnegative integers, define
another such sequence a• #a′• by
                           (a• #a′• )k = max (ai + a′j ).
                                        i+j=k

Note that the operation # is associative, and the Disjoint Union axiom of
ECH capacities can be restated as
                        c• (X1 ⊔ X2 ) = c• (X1 )#c• (X2 ).

                                        10
Proof of Theorem 1. The “only if” part follows from Corollary 5(a). To
prove the “if” part, assume without loss of generality that a/b and c/d are
rational and c < d, and suppose that N (a, b) ≤ N (c, d). By Proposition 7,
we need to show that there exists a symplectic embedding as in (8). By
Theorem 2 and the calculation in Corollary 5(b), it is enough to show that

                     c• (B(a, b) ⊔ B(d − c, d)) ≤ c• (B(d)).

To prove this, first note that applying the Monotonicity axiom to the em-
bedding (9) and using our hypothesis gives

                    c• (B(a, b)) ≤ c• (E(a, b)) ≤ c• (E(c, d)).

By the Disjoint Union axiom and the fact that the operation ‘#’ respects
inequality of sequences, it follows that

             c• (B(a, b) ⊔ B(d − c, d)) ≤ c• (E(c, d) ⊔ B(d − c, d)).

On the other hand, applying Monotonicity to the embedding (10) gives

                     c• (E(c, d) ⊔ B(d − c, d)) ≤ c• (B(d)).

By the above two inequalities we are done.                                       

Remark. The above is McDuff’s original proof of Theorem 1. Her sub-
sequent proof in [M3] avoids using the monotonicity of ECH capacities (a
heavy piece of machinery) as follows. The idea is to define the ECH ca-
pacities of any union of balls or ellipsoids by the Ellipsoid and Disjoint
Union axioms, and then to algebraically justify all invocations of Mono-
tonicity in the proof. For example, in the above argument for the ‘if’
part of Theorem 1, in the first step one needs to show that if a/b is ra-
tional then c• (B(a, b)) ≤ c• (E(a, b)). In fact one can show algebraically
that c• (B(a, b)) = c• (E(a, b)). To do so, by induction and the associa-
tivity of #, it is enough to show that if a/b is rational and a < b then
N (a, b) = N (a, a)#N (a, b − a). The proof of this may be found in [M3].

Remark. The proof of Theorem 1 generalizes to show that ECH capacities
give a sharp obstruction to symplectically embedding any disjoint union of
finitely many ellipsoids into an ellipsoid.

Remark. Theorem 1 does not directly generalize to higher dimensions.
That is, if one defines N (a1 , . . . , an ) to be the sequence of nonnegative inte-
ger linear combinations of a1 , . . . , an in increasing order, then when n > 2 it

                                        11
is not true that int(E(a1 , . . . , an )) symplectically embeds into E(a′1 , . . . , a′n )
if and only if N (a1 , . . . , an ) ≤ N (a′1 , . . . , a′n ). In particular, Hind-Kerman
[HK] used methods of Guth [Gu] to show that E(1, R, R) symplectically
embeds into E(a, a, R2 ) whenever a > 3. However if R is sufficiently large
with respect to a then N (1, R, R) 6≤ N (a, a, R2 ).


Embedded contact homology
In the ball packing story above, an important role was played by Taubes’s
“SW=Gr” theorem, which relates Seiberg-Witten invariants of symplectic
4-manifolds to holomorphic curves. The ECH capacities are defined using
an analogue of “SW=Gr” for contact 3-manifolds.
     Let Y be a closed oriented 3-manifold. Recall that a contact form on Y
is a 1-form λ on Y such that λ ∧ dλ > 0 everywhere. The contact form λ
determines a Reeb vector field R characterized by dλ(R, ·) = 0 and λ(R) = 1.
A Reeb orbit is a closed orbit of R, i.e. a map γ : R/T Z → Y for some T > 0,
modulo reparametrization, such that γ ′ (t) = R(γ(t)). The contact form λ
is called “nondegenerate” if all Reeb orbits are cut out transversely in an
appropriate sense. This holds for generic contact forms λ.
     If λ is a nondegenerate contact form on Y as above, and if Γ ∈ H1 (Y ),
the embedded contact homology ECH∗ (Y, λ, Γ) is defined as follows. It is
the homology of a chain complex ECC∗ (Y, λ, Γ) which is freely generated
over Z/2 (it can also be defined over Z but this will not be needed here). A
generator is a finite set of pairs α = {(αi , mi )} where the αi ’s are distinct
embedded Reeb orbits, the mi ’s are positive integers, mi = 1 whenever αi is
hyperbolic
P             (i.e. the linearized Reeb flow around αi has real eigenvalues), and
   i mi [αi ] = Γ ∈ H1 (Y ). The chain complex has a relative grading which is
defined in [H1]; the details of this are not important here.
     To define the differential

                      ∂ : ECC∗ (Y, λ, Γ) → ECC∗−1 (Y, λ, Γ)

one chooses a generic almost complex structure J on R×Y with the following
properties: J is R-invariant, J(∂s ) = R where s denotes the R coordinate,
and J sends Ker(λ) to itself, rotating positively in the sense that dλ(v, Jv) >
0 for 0 6= v ∈ Ker(λ). If α = {(αi , mi )} and β = {(βj , nj )} are two chain
complex generators, then the differential coefficient h∂α, βi ∈ Z/2 is a mod
2 count of J-holomorphic curves in R ×   PY which have “ECH index”     P equal
to 1 and which converge as currents to i mi αi as s → +∞ and to j nj βj
as s → −∞. Holomorphic curves with ECH index 1 have various special

                                           12
properties, one of which is that they are embedded (except that they may
include multiply covered R-invariant cylinders), hence the name “embedded
contact homology”. For details see [H2] and the references therein. It is
shown in [HT1] that ∂ 2 = 0. Although the differential usually depends on
the choice of J, the homology of the chain complex does not. In fact it
is shown by Taubes [Ta2] that ECH∗ (Y, λ, Γ) is isomorphic to a version of
Seiberg-Witten Floer cohomology of Y as defined in [KM2].
    The definition of ECH capacities only uses the case Γ = 0.
    To define the ECH capacities we need to recall four additional structures
on embedded contact homology:
    (1) There is a chain map
                   U : ECC∗ (Y, λ, Γ) → ECC∗−2 (Y, λ, Γ)
which counts J-holomorphic curves of ECH index 2 passing through a generic
point in z ∈ R × Y , see [HT2]. The induced map on homology
                   U : ECH∗ (Y, λ, Γ) → ECH∗−2 (Y, λ, Γ)
does not depend on z when Y is connected. If Y has n components, then
there are n different versions of the U map.
   (2) If α = {(αi , mi )} is a generator of the ECH chain complex, define its
symplectic action by
                                   X      Z
                          A(α) =       mi     λ ∈ R≥0 .
                                   i        αi

It follows from the conditions on J that the differential decreases the sym-
plectic action, i.e. if h∂α, βi 6= 0 then A(α) > A(β). Hence for each L ∈ R
we can define ECH L (Y, λ, Γ) to be the homology of the subcomplex spanned
by generators with action less than L. It is shown in [HT3] that this does
not depend on J, although unlike the usual ECH it does depend strongly
on λ.
    (3) The empty set of Reeb orbits is a legitimate generator of the ECH
chain complex, and by the above discussion it is a cycle. Thus we have a
canonical element
                               [∅] ∈ ECH∗ (Y, λ, 0).
   (4) Let (Y+ , λ+ ) and (Y− , λ− ) be closed oriented 3-manifolds with nonde-
generate contact forms. A weakly exact symplectic cobordism from (Y+ , λ+ )
to (Y− , λ− ) is a compact symplectic 4-manifold (X, ω) such that ∂X =
Y+ − Y− , the form ω on X is exact, and ω|Y± = dλ± . The key result which
enables the definition of the ECH capacities is the following:

                                       13
Theorem 8. A weakly exact symplectic cobordism (X, ω) as above induces
maps
         ΦL (X, ω) : ECH∗L (Y+ , λ+ , 0) → ECH∗L (Y− , λ− , 0)
for each L ∈ R with the following properties:
 (a) ΦL [∅] = [∅].

 (b) If U+ is any of the U maps for Y+ , and if U− is any of the U maps for
      Y− corresponding to the same component of X, then ΦL ◦U+ = U− ◦ΦL .
Idea of proof. This theorem follows from a slight modification of the main re-
sult of [HT3], as explained in [H3]. The first step is to define a “completion”
X of X by attaching cylindrical ends [0, ∞) × Y+ to the positive boundary
and (−∞, 0]× Y− to the negative boundary. One chooses an almost complex
structure J on X which is ω-compatible on X and which on the ends agrees
with almost complex structures as needed to define the ECH of (Y± , λ± ).
    One would like to define a chain map ECC∗ (Y+ , λ+ , 0) → ECC∗ (Y− , λ− , 0)
by counting J-holomorphic curves in X with ECH index 0. Considering ends
of ECH index 1 moduli spaces would prove that this is a chain map. The
conditions on J and the fact that we are restricting to Γ = 0 imply that
this map would respect the symplectic action filtrations and satisfy property
(a). To prove property (b) one would choose a path ρ in X from a positive
cylindrical end to a negative cylindrical end. Counting ECH index 1 curves
that pass through ρ would then define a chain homotopy as needed to prove
that ΦL ◦ U+ = U− ◦ ΦL .
    Unfortunately, it is not currently known how to define ΦL by counting
holomorphic curves as above, due to technical difficulties caused by multiply
covered holomorphic curves with negative ECH index, see [H1, §5]. However
one can still define ΦL and prove properties (a) and (b) by passing to Seiberg-
Witten theory, using arguments from [Ta2].                                    


Definition of ECH capacities
Let Y be a closed oriented 3-manifold with a contact form λ, and suppose
that [∅] 6= 0 ∈ ECH∗ (Y, λ, 0). We then define a sequence of real numbers

                0 = c0 (Y, λ) < c1 (Y, λ) ≤ c2 (Y, λ) ≤ · · · ≤ ∞

as follows. Suppose first that λ is nondegenerate. If Y is connected, define
                        n                                        o
         ck (Y, λ) = inf L ∈ R ∃η ∈ ECH L (Y, λ, 0) : U k η = [∅] .

                                       14
If Y is disconnected, define ck (Y, λ) the same way, but replace the condition
U k η = [∅] with the condition that every k-fold composition of U maps send
η to [∅]. Finally, if λ is degenerate, one defines ck (Y, λ) by approximating λ
by nondegenerate contact forms.
     Moving back up to four dimensions, define a (four dimensional) Liouville
domain to be a compact symplectic four manifold (X, ω) such that ω is exact,
and there exists a contact form λ on ∂X with dλ = ω|∂X . In other words,
(X, ω) is a weakly exact symplectic cobordism from a contact three-manifold
to the empty set. For example, any star-shaped subset of R4 is a Liouville
domain. Here “star-shaped” means that the boundary is transverse to the
radial vector field, and we take the standard symplectic form (1) as usual.
     If (X, ω) is a Liouville domain, define its ECH capacities by
                              ck (X, ω) = ck (∂X, λ)
where λ is any contact form on ∂X with dλ = ω|∂X . Note here that
ck (∂X, λ) is defined because it follows from Theorem 8(a) that [∅] 6= 0 ∈
ECH∗ (∂X, λ, 0). Also, ck (X, ω) does not depend on λ, because changing
λ will not change the ECH chain complex, and the fact that we restrict to
Γ = 0 implies that changing λ does not affect the symplectic action filtration.
Proof of Theorem 4 (for Liouville domains). The Conformality axiom fol-
lows directly from the definition. The Ellipsoid and Disjoint Union axioms
are proved by direct calculations in [H3]. To prove the Monotonicity ax-
iom, let (X0 , ω0 ) and (X1 , ω1 ) be Liouville domains and let φ : (X0 , ω0 ) →
(X1 , ω1 ) be a symplectic embedding. Let λi be a contact form on ∂Xi with
dλi = ω|∂Xi for i = 0, 1. By a continuity argument we can assume without
loss of generality that φ(X0 ) ⊂ int(X1 ) and that the contact forms λi are
nondegenerate. Then (X1 \φ(int(X1 )), ω1 ) defines a weakly exact symplectic
cobordism from (∂X1 , λ1 ) to (∂X0 , λ0 ). It follows immediately from The-
orem 8 that ck (∂X1 , λ1 ) ≥ ck (∂X0 , λ0 ), because the maps ΦL preserve the
set of ECH classes η with U k η = [∅].
    More generally, ck of an arbitrary symplectic manifold (X, ω) is defined
to be the supremum of ck (X ′ , ω ′ ), where (X ′ , ω ′ ) is a Liouville domain that
can be symplectically embedded into (X, ω).


More examples of ECH capacities
Theorem 9. [H3] The ECH capacities of a polydisk are
      ck (P (a, b)) = min{am + bn | m, n ∈ N, (m + 1)(n + 1) ≥ k + 1}.

                                        15
    It turns out that ECH capacities also give a sharp obstruction to symplec-
tically embedding an ellipsoid into a polydisk. The proof uses the following
analogue of Proposition 7:
Proposition 10 (Müller [Mu]). Let a, b, c, d > 0 with a/b rational. Then
there is a symplectic embedding int(E(a, b)) → P (c, d) if and only if there is
a symplectic embedding
                   int(B(a, b) ⊔ B(c) ⊔ B(d)) → B(c + d).
     As in Proposition 7, the “only if” direction in Proposition 10 follows from
an explicit construction (together with (9)). Namely, the triangle ∆(c +
d, c + d) is partitioned into a rectangle of side lengths c and d together with
translates of ∆(c, c) and ∆(d, d), so there is a symplectic embedding
                   int(P (c, d) ⊔ B(c) ⊔ B(d)) → B(c + d).                 (11)
Corollary 11. There is a symplectic embedding int(E(a, b)) → P (c, d) if
and only if c• (E(a, b)) ≤ c• (P (c, d)).
Proof. Copy the above proof of Theorem 1, using Proposition 10 and (11)
in place of Proposition 7 and (10).

Remark. ECH capacities do not always give sharp obstructions to sym-
plectically embedding a polydisk into an ellipsoid. For example, it is easy
to check that c• (P (1, 1)) = c• (E(1, 2)). Thus ECH capacities give no ob-
struction to symplectically embedding P (1, 1) into E(a, 2a) whenever a > 1.
However the Ekeland-Hofer capacities (see [CHLS]) show that P (1, 1) does
not symplectically embed into E(a, 2a) whenever a < 3/2. And the lat-
ter bound is sharp, since according to our definitions P (1, 1) is a subset of
E(3/2, 3).
    Theorem 9 is deduced in [H3] from the following more general calculation,
proved using results from [HS]. Let k · k be a norm on R2 , regarded as a
translation-invariant norm on T T 2 . Let k·k∗ denote the dual norm on T ∗ T 2 .
Define
                       Tk·k∗ := ζ ∈ T ∗ T 2 kζk∗ ≤ 1 ,
                                

with the canonical symplectic form on T ∗ T 2 .
Theorem 12. [H3] If k · k is a norm on R2 , then
           ck Tk·k∗ = min ℓk·k (Λ) |PΛ ∩ Z2 | = k + 1 .
                             
                                                                           (12)
Here the minimum is over convex polygons Λ in R2 with vertices in Z2 , and
PΛ denotes the closed region bounded by Λ. Also ℓk·k (Λ) denotes the length
of Λ in the norm k · k.

                                      16
   Finally, we remark that in all known examples, the ECH capacities
asymptotically recover the symplectic volume, via:

Conjecture 13. [H3] Let (X, ω) be a four-dimensional Liouville domain
such that ck (X, ω) < ∞ for all k. Then

                            ck (X, ω)2
                         lim           = 4 vol(X, ω).
                        k→∞      k

Of course, it is the deviation of ck (X, ω)2 /k from 4 vol(X, ω) that gives rise
to nontrivial symplectic embedding obstructions.

Acknowledgments. I thank Paul Biran and Dusa McDuff for patiently
explaining to me the ball packing and ellipsoid embedding stories. This
work was partially supported by NSF grant DMS-0806037.


References
[B1]     P. Biran, Symplectic packing in dimension 4 , Geom. Funct. Anal.
         7 (1997), 420–437.

[B2]     P. Biran, From symplectic packing to algebraic geometry and back ,
         European Congress of Mathematics, Vol. II (Barcelona, 2000), 507–
         524, Progr. Math. 202, Birkhäuser, 2001.

[CHLS] K. Cieliebak, H. Hofer, J. Latschev, and F. Schlenk, Quantitative
       symplectic geometry, Dynamics, ergodic theory, and geometry, 1-44,
       Math. Sci. Res. Inst. Publ. 54, Cambridge University Press, 2007.

[Gr]     M. Gromov, Pseudoholomorphic curves in symplectic manifolds, In-
         vent. Math. 82 (1985), 307–347.

[Gu]     L. Guth, Symplectic embeddings of polydisks, Invent. Math. 172
         (2008), 477-489.

[HK]     R. Hind and E. Kerman, New obstructions to symplectic embed-
         dings, arXiv:0906.4296.

[H1]     M. Hutchings, The embedded contact homology index revisited, New
         perspectives and challenges in symplectic field theory, 263–297,
         CRM Proc. Lecture Notes, 49, AMS, 2009.



                                      17
[H2]     M. Hutchings, Embedded contact homology and its applications,
         Proceedings of the ICM 2010, Hyderabad, vol. II, 1022–1041.

[H3]     M. Hutchings, Quantitative embedded contact homology,
         arXiv:1005.2260, to appear in J. Differential Geometry.

[HS]     M. Hutchings and M. Sullivan, Rounding corners of polygons and
         the embedded contact homology of T 3 , Geometry and Topology 10
         (2006), 169–266.

[HT1]    M. Hutchings and C. H. Taubes, Gluing pseudoholomorphic curves
         along branched covered cylinders I , J. Symplectic Geom. 5 (2007),
         43–137.

[HT2]    M. Hutchings and C. H. Taubes, The Weinstein conjecture for stable
         Hamiltonian structures, Geometry and Topology 13 (2009), 901–
         941.

[HT3]    M. Hutchings and C. H. Taubes, Proof of the Arnold chord conjec-
         ture in three dimensions II , in preparation.

[KM1] P.B. Kronheimer and T.S. Mrowka, The genus of embedded surfaces
      in the projective plane, Math. Res. Lett. 1 (1994), 797–808.

[KM2] P.B. Kronheimer and T.S. Mrowka, Monopoles and three-manifolds,
      Cambridge University Press, 2007.

[LiLi]   Bang-He Li and T-J. Li, Symplectic genus, minimal genus and dif-
         feomorphisms, Asian J. Math 6 (2002), 123-144.

[LiLiu] T-J. Li and A-K. Liu, Uniqueness of symplectic canonical class,
        surface cone and symplectic cone of 4-manifolds with B + = 1, J.
        Diff. Geom. 58 (2001), 331–370.

[M1]     D. McDuff, From symplectic deformation to isotopy, Topics in sym-
         plectic 4-manifolds (Irvine, CA, 1996), 85–99, First Int. Press Lect.
         Ser. I, Int. Press, Cambridge MA, 1998.

[M2]     D. McDuff, Symplectic embeddings of 4-dimensional ellipsoids, J.
         Topology 2 (2009), 1–22.

[M3]     D. McDuff, The Hofer conjecture on embedding symplectic ellip-
         soids, arXiv:1008.1885, v2.



                                     18
[MP]    D. McDuff and L. Polterovich, Symplectic packings and algebraic
        geometry, Invent. Math. 115 (1994), 405–429.

[MS]    D. McDuff and F. Schlenk, The embedding capacity of 4-dimensional
        symplectic ellipsoids, arXiv:0912.0532, v2.

[Mu]    D. Müller, Symplectic embeddings of ellipsoids into polydiscs, Univ.
        Neuchatel PhD thesis, in preparation.

[S]     F. Schlenk, Embedding problems in symplectic geometry, de Gruyter
        Expositions in Mathematics 40, Walter de Gruyter, Berlin, 2005.

[Ta1]   C. H. Taubes, Seiberg-Witten and Gromov Invariants for Symplectic
        4-manifolds, International Press, Somerville MA 2000.

[Ta2]   C. H. Taubes, Embedded contact homology and Seiberg-Witten Floer
        homology I , arXiv:0811.3985.

[Tr]    L. Traynor, Symplectic packing constructions, J. Differential Geom-
        etry 42 (1995), 411–429.

[W]     I. Wieck, Explicit symplectic packings: symplectic tunnelling and
        new maximal constructions, PhD thesis, Universität zu Köln,
        Shaker-Verlag, Aachen (2009).




                                    19
