---
source: arXiv:2111.09177
fetched: 2025-10-20
---
# Remarks on symplectic capacities of p-products

                                            Remarks on symplectic capacities of p-products
arXiv:2111.09177v1 [math.SG] 17 Nov 2021




                                                                    P. Haim-Kislev, Y. Ostrover

                                                                          November 18, 2021


                                                                                 Abstract

                                                    In this note we study the behavior of symplectic capacities of convex
                                                 domains in the classical phase space with respect to symplectic p-products.
                                                 As an application, by using a “tensor power trick”, we show that it is
                                                 enough to prove the weak version of Viterbo’s volume-capacity conjecture
                                                 in the asymptotic regime, i.e., when the dimension is sent to infinity.
                                                 In addition, we introduce a conjecture about higher-order capacities of
                                                 p-products, and show that if it holds, then there are no non-trivial p-
                                                 decompositions of the symplectic ball.



                                           1    Introduction and Results

                                           Symplectic capacities are numerical invariants that serve as a fundamental tool
                                           in the study of symplectic rigidity phenomena. Most known examples are closely
                                           related to Hamiltonian dynamics, and can moreover be used to study the exis-
                                           tence and the behavior of periodic orbits of certain Hamiltonian systems. We
                                           refer the reader to [5, 20] for more information on symplectic capacities, their
                                           properties, and the role that they play in symplectic topology.
                                              In this note we consider symplectic capacities of convex subsets of the classical
                                           phase space (R2n , ωn ), and study their behavior with respect to “symplectic
                                           p-products” (cf. [19, 21]). More precisely, recall that the Cartesian product
                                           M × N of two symplectic manifolds (M, ωM ), (N, ωN ) has a natural symplectic
                                                                        ∗         ∗
                                           structure given by ω = πM      ωM + πN   ωN , where πM , πN are the two natural
                                           projections. For two convex bodies K ⊂ R2n and T ⊂ R2m , which contain
                                           the origin of their respective ambient spaces, and 1 ≤ p ≤ ∞, we consider the
                                           well-known p-product operation
                                                                      [                         
                                                         K ×p T :=         (1 − t)1/p K × t1/p T ⊂ R2n × R2m .
                                                                     0≤t≤1

                                           Since the above definition of the p-product of two bodies is naturally applicable
                                           only when the bodies contain the origins of their respective ambient spaces,

                                                                                     1
from now on we will assume that all convex bodies contain the origin unless
specifically stated otherwise. Note that K ×∞ T = K × T , and that K ×1 T =
conv{(K × 0) ∪ (0 × T )}, where conv stands for the convex hull of a set. The
1-product K ×1 T is also known as the free sum of K and T , and is related
to the product operation via duality (see e.g., [14]). We remark that the norm
naturally associated with K ×p T satisfies

   k(x, y)kpK×p T = kxkpK + kykpT , and k(x, y)kK×∞ T = max{kxkK , kykT }.

Moreover, let hL : Rd → R be the support function associated with a convex
body L in Rd (see the notations given at the end of this section). Then, the
support function of K ×p T satisfies

      hK×p T (x, y)q = hK (x)q + hT (y)q , and hK×1 T = max{hK (x), hT (y)},
          1       1
where     p   +   q   = 1 and p, q ≥ 1.


1.1       The systolic ratio of symplectic p-products

For a convex domain K ⊂ R2n , it is known that many normalized symplectic ca-
pacities, including the first Ekeland-Hofer capacity c1EH [8, 9], the Hofer-Zehnder
capacity cHZ [16], the symplectic homology capacity cSH [25], and the first Gutt-
Hutchnigs capacity c1GH [12], coincide. Moreover, when K is smooth, all these
capacities are given by the minimal action among all the closed characteristics
on the boundary ∂K ∗ . The above claims follow from a combination of results
from [1, 11, 12, 16, 18]. In what follows, for a convex domain K in R2n we shall
denote the above mentioned coinciding capacities by cEHZ (K). The systolic ratio

                                                    cEHZ (K)
                                   sysn (K) :=
                                                 (n!Vol(K))1/n

of K is defined as the ratio of this capacity of K to the normalized ω-volume of
K. Note that sysn (B) = 1, for any Euclidean ball B in R2n .
  Recall the following weak version of Viterbo’s volume-capacity conjecture [24].
Conjecture 1.1 (Viterbo). If K ⊂ R2n is a convex domain, then

                                          sysn (K) ≤ 1.

  Our first result concerns the systolic ratio of symplectic p-products. We show
that if two convex bodies K ⊂ R2n and T ⊂ R2m fulfill Conjecture 1.1, then
the same is true for the p-product of K and T . More precisely,
   ∗ Ifthe boundary of K is not smooth, the above capacities coincide with the minimal
action among “generalized closed characteristics”, as explained, e.g., in [2].




                                                 2
Theorem 1.2. For convex bodies K ⊂ R2n , T ⊂ R2m , and 1 ≤ p ≤ ∞

                 sysn+m (K ×p T )m+n ≤ sysn (K)n sysm (T )m ,

where equality holds if and only if cEHZ (K) = cEHZ (T ) and p = 2.
Remark 1.3. It is not hard to deduce that sysn+m (K ×p T ) < sysn+m (K ×2 T )
for p > 2 from the proof of Theorem 1.2 below. On the other hand, for p = 1,
consider a convex body K ⊂ R2n and choose a convex body T ⊂ R2 with
c(T ) = nc(K). Then, using Proposition 1.5 below, one gets that
                                    
           sysn+1 (K ×1 T )      n               1
                            =          (2n + 1) n+1 > 1, for n ≥ 2.
           sysn+1 (K ×2 T )     n+1

  One application of Theorem 1.2 is a “tensor power trick” (see, e.g., the dis-
cussion in Section 1.9.4 of [23]), which shows that it is enough to prove Conjec-
ture 1.1 above for convex domains in the asymptotic regime n → ∞.
Corollary 1.4. If Conjecture 1.1 holds in dimension 2n for some n > 1, then
it also holds in dimension 2m for every m ≤ n. Moreover, if there exists a
                n→∞
sequence α(n) −−−−→ 1 such that for every convex body K ⊂ R2n one has

                                sysn (K) ≤ α(n),

then Conjecture 1.1 holds in every dimension n.

   We remark that Corollary 1.4 was already known to experts [17]. Another ap-
plication of Theorem 1.2 concerns obstructions on the possibility of a “p-product
decomposition” of the Euclidean ball. This will be discussed in Section 1.3 below
(see Corollary 1.12).
  It is well known (see e.g., [5]) that cEHZ satisfies the Cartesian product prop-
erty i.e., for two convex bodies K ⊂ R2n and T ⊂ R2m ,

                    cEHZ (K × T ) = min{cEHZ (K), cEHZ (T )}.

An important ingredient in the proof of Theorem 1.2 above, which might be
also of independent interest, is the following generalization of this formula for
the symplectic p-product of two convex bodies (cf. Lemma 2.2 below).
Proposition 1.5. For two convex bodies K ⊂ R2n , T ⊂ R2m , and 1 ≤ p ≤ ∞,
                    
                    min{cEHZ (K), cEHZ (T )},           if 2 ≤ p ≤ ∞,
    cEHZ (K ×p T ) =          p               p
                                                  p−2
                                                    p
                     c (K) p−2 + c (T ) p−2           , if 1 ≤ p < 2.
                       EHZ            EHZ




                                        3
1.2        Higher-order capacities of symplectic p-products

Consider the family of Ekeland-Hofer capacities {ckEH }∞  k=1 defined in [8, 9]. We
recall that in Eq. (3.8) from [5] it is asserted that if X1 ⊂ R2n and X2 ⊂ R2m
are two compact star-shaped domains, then

                     ckEH (X1 × X2 ) = min {ciEH (X1 ) + cjEH (X2 )},                   (1)
                                            i+j=k


where i and j are non-negative integers, and by definition c0EH = 0. Motivated
by Proposition 1.5, we conjecture the following generalization of (1).
Conjecture 1.6. For star-shaped domains X1 ⊂ R2n , X2 ⊂ R2m , and 1 ≤ p,
                        h                                 i p−2
                   
                           i
                                      p
                                             j          p     p
                    min  c
                   i+j=k EH  (X 1 ) p−2 + c
                                             EH (X 2 ) p−2       ,    if p ≥ 2,
  k                         h                                 i
 cEH (X1 ×p X2 ) =                       p                 p
                                                                p−2
                   
                     max ciEH (X1 ) p−2 + cjEH (X2 ) p−2
                                                                 p
                                                                    , if 1 ≤ p ≤ 2.
                   
                   i+j=k+1
                            i,j6=0

Remark 1.7. (I) In the case p = 2, the two expressions in Conjecture 1.6 agree,
and a combinatorial argument shows that they are equal to the k th term in the
sequence {ciEH (X1 ), cjEH (X2 )} arranged in non-decreasing order with repetitions.
For completeness, we provide a proof of this fact in the Appendix of the paper
(see Lemma 5.1).
   (II) It is conjectured that the family {ckGH }∞
                                                 k=1 of Gutt-Hutchings symplectic
capacities introduced in [12] coincide with the family of Ekeland-Hofer capacities
{ckEH }∞
       k=1 on the class of compact star-shaped domains of R
                                                               2n
                                                                  (see Conjecture
1.9 in [12]). Thus, Conjecture 1.6 above could be stated for {ckGH }∞k=1 as well.


  Using the remarkable fact that for convex/concave toric domains there are
“combinatorial” formulas to compute the Gutt-Hutchings capacities {ckGH }∞     k=1
(see Theorems 1.6 and 1.14 in [12]), we can show that Conjecture 1.6, when
stated for the family {ckGH }∞
                             k=1 , holds in some special cases. More precisely,

Theorem 1.8. For convex toric domains K ⊂ R2n , T ⊂ R2m , and p ≥ 2†
                                     h         p               p
                                                                  i p−2
                                                                     p
                ckGH (K ×p T ) = min ciGH (K) p−2 + cjGH (T ) p−2       .
                                      i+j=k


For concave toric domains K ⊂ R2n , T ⊂ R2m , and 1 ≤ p ≤ 2
                                                h          p               p
                                                                              i p−2
                                                                                 p
               ckGH (K ×p T ) =       max        ciGH (K) p−2 + cjGH (T ) p−2       .
                                     i+j=k+1
                                       i,j6=0
   † See   Remark 1.7 (I) above about the case p = 2.




                                                   4
   Theorem 1.8 generalizes certain calculations of ckGH in some known cases. In
particular, it is known that the Cartesian product property holds for convex toric
domains (see Remark 1.10 in [12]), and recently the Gutt-Hutchings capacities
of p-product of discs were calculated in [19] (see also [21]).
  We end this subsection with a result about the limit of the normalized Ekeland-
Hofer capacities. Following [5], we set

                                                    ckEH
                                         c¯k :=          ,
                                                      k
                                                                                                ck
and denote the limit of c¯k as k → ∞ by c∞ . Denote also cGH   ∞ := limk→∞ k
                                                                               GH


to be the same limit for the Gutt-Hutchings capacities. In [5, Problem 17] the
authors raise the question whether c∞ coincides with the Lagrangian capacity
cL defined in [6]. In [12] it is shown that cGH∞ (K) ≤ cL (K) for a convex or
concave toric domain K, and moreover, in this case cGH   ∞ (K) = c (K), where
the cube capacity c (K) is the largest area of the 2-faces of a cube that can be
symplectically embedded into K. In [5], it was proved that
                                                             1
                      c∞ (E(a1 , . . . , an )) =                        ,
                                                    1/a1 + · · · + 1/an
                                                         π|z1 |2            π|zn |2
for the ellipsoids E(a1 , . . . , an ) := {z ∈ Cn :        a1      + ···+     an      < 1}, and that

                        c∞ (P (a1 , . . . , an )) = min{a1 , . . . , an },

for the polydiscs P (a1 , . . . , an )p:= B 2 [a1 ] × · · · × B 2 [an ], where B n [r] ⊂ Rn
is the Euclidean ball of radius r/π. A generalization of the above formulas,
assuming Conjecture 1.6 holds, is the following.
Theorem 1.9. Assume Conjecture 1.6 holds. Then, for 1 ≤ p ≤ ∞, and convex
domains K1 , . . . , Km ⊂ R2n such that c∞ (K1 ), . . . , c∞ (Km ) exist, one has
                                           −p                   −p
                                                                     −2
                                                                      p
          c∞ (K1 ×p · · · ×p Km ) = c∞ (K1 ) 2 + · · · + c∞ (Km ) 2      .                       (2)

Remark 1.10. (I) As ellipsoids and polydiscs are p-product of discs with p = 2
and p = ∞, respectively, Theorem 1.9 does indeed generalize the above men-
tioned formulas for c∞ (E(a1 , . . . , an )) and c∞ (P (a1 , . . . , an )).
  (II) Theorem 1.8 gives us that (2) holds for cGH   ∞ , and for convex/concave
toric domains with p ≥ 2 and 1 ≤ p ≤ 2, respectively. Moreover, this can be
easily recovered by using the fact, shown in [12], that for convex/concave toric
domains
                                                     1
                      cGH
                       ∞ (XΩ ) = c (XΩ ) =
                                             k(1, . . . , 1)kΩ
(see Section 3 below for the relevant definitions and notations).



                                                5
1.3    Symplectic p-decomposition of convex bodies

In this section we consider the following general question:
Question 1.11. Which convex bodies are symplectomorphic to a symplectic
p-product configuration of convex bodies of lower dimensions?

  An immediate corollary of Theorem 1.2 is the following claim that relates
the p-decomposition property of symplectic images of the Euclidean ball and
Viterbo’s volume-capacity conjecture.
Corollary 1.12. If for some symplectic image of the Euclidean ball B       e in
R 2n+2m                                  2n             2m          e
        there exist convex bodies X ⊂ R and Y ⊂ R such that B = X ×p Y
for some p 6= 2, then Conjecture 1.1 is false. Moreover, if B  e = X ×2 Y , and
sysn (X) 6= 1 or sysm (Y ) 6= 1, then Conjecture 1.1 is false.

  We finish the Introduction with the following consequence of Conjecture 1.6
that answers Question 1.11 for the case of the Euclidean ball.
Theorem 1.13. Assume Conjecture 1.6 holds. Then, for any pair of con-
vex bodies X ⊂ R2n , Y ⊂ R2m , and p 6= 2, the product X ×p Y is not sym-
plectomorphic to a Euclidean ball. Moreover, if a symplectic image of the
ball Be 2(n+m) [r] ⊂ R2(n+m) can be written as B
                                               e 2(n+m) [r] = X ×2 Y for some
convex bodies X ⊂ R , Y ⊂ R , then one has ckEH (X) = ckEH (B 2n [r]) and
                          2n         2m

ckEH (Y ) = ckEH (B 2m [r]) for every k ≥ 1.

Notations: Denote by Km the class of convex bodies in Rm , i.e., compact
convex sets with non-empty interior. For a smooth convex body, we denote
by nK (x) the unit outer normal to ∂K at the point x ∈ ∂K. The support
function hK : Rm → R associated with a convex body K ∈ Km is given
by hK (u) = sup{hx, ui | x ∈ K}. The phase space R2n is equipped with the
standard symplectic structure ωn = dq ∧ dp, and the standard linear complex
structure J : R2n → R2n . The symplectic
                                      R         action of a closed curve γ in R2n
is defined as the integral A(γ) := γ λ, where λ is a primitive of ωn . Finally,
W 1,2 (S 1 , R2n ) is the Banach space of absolutely continuous 2π-periodic func-
tions whose derivatives belong to L2 (S 1 , R2n ).
Organization of the paper: In Section 2 we prove Theorem 1.2, Corollary 1.4
and Proposition 1.5. In Section 3 we prove Theorem 1.8 and Theorem 1.9. In
Section 4 we prove Corollary 1.12 and Theorem 1.13. The paper is concluded
with an appendix including a proof of Lemmas 5.1.
Acknowledgements: We wish to thank Viktor Ginzburg, Başak Gürel, and
Marco Mazzucchelli for helpful comments on higher-order symplectic capaci-
ties, and to Shiri Artstien-Avidan for many stimulating discussions and useful
remarks. The authors received funding from the European Research Council
grant No. 637386. Y.O. is partially supported by the ISF grant No. 667/18.

                                       6
2    The EHZ-capacity of symplectic p-products

In this section we prove Proposition 1.5, Theorem 1.2, and Corollary 1.4. We
will give two separate proofs of Proposition 1.5. The first is an analytic proof
based on Clarke’s dual action principle (see [7]), and the second is a geometric
proof based on an analysis of the dynamics of characteristics on p-products.
  We start with a few preliminary definitions and remarks. Consider a convex
body K in R2n with a smooth boundary. The restriction of the symplectic form
ω to the hypersurface ∂K canonically defines a one-dimensional sub-bundle,
ker ω|∂K , whose integral curves form the characteristic foliation of ∂K. Recall
that cEHZ (K) is defined to be the minimal symplectic action among the closed
characteristics on ∂K, or equivalently, the minimal period of a closed charac-
teristic γ, where γ is parametrized by
                                         2JnK (γ(t))
                              γ̇(t) =                  .
                                        hK (nK (γ(t)))
We remark that although the above definition of cEHZ (K) was given only for the
class of convex bodies with smooth boundary, it can be naturally generalized to
the class of convex sets in R2n with nonempty interior (see, e.g., [2]). Moreover,
since all the symplectic capacities considered in this paper are known to be
continuous with respect to the Hausdorff metric on the class of convex domains,
in what follows we can without loss of generality assume that all the convex
domains are smooth and strictly convex.
  Next we describe the dynamics of a characteristic on the p-product K ×p T ,
for two convex bodies K and T . Let x ∈ ∂K and y ∈ ∂T . Let (αx, βy) ∈
∂(K ×p T ) be a point on the boundary of the p-product of K and T , i.e., such
that αp + β p = 1 and α, β ≥ 0. A direct computation gives
           nK×p T (αx, βy)               nK (x)              nT (y) 
                                 = αp−1             , β p−1             .       (3)
        hK×p T (nK×p T (αx, βy))        hK (nK (x))         hT (nT (y))
This equation shows that the two natural projections of the characteristic di-
rections in K ×p T are the characteristic directions in K and T , respectively.
  The main ingredient in the first proof of Proposition 1.5 is the following
formula based on Clark’s dual action principle: for a convex body K ⊂ R2n and
p ≥ 1, one has
                                               Z 2π
                      p                      1
             cEHZ (K) 2 = π p     min               hpK (ż(t))dt                (4)
                              z∈En , A(z)=1 2π 0
                                                        Z 2π
                                             1        p
                        = πp      min          A(z)− 2        hpK (ż(t))dt,     (5)
                              z∈En , A(z)>0 2π            0
                                      R 2π
where En = {z ∈ W 1,2 (S 1 , R2n ) | 0 z(t)dt = 0}, and A(z) is the symplectic
action of z. Here (4) follows, e.g., from Proposition 2.1 in [3], and (5) follows by

                                           7
rescaling. Moreover, following the proof of Proposition 2.1 in [3] the minimizer
of (4) coincides, up to translation and rescaling, with a closed characteristic on
the boundary ∂K.
   For the proof of Proposition 1.5 we also need the following lemma, the proof
of which is a simple exercise.
Lemma 2.1. For a, b > 0 one has
                                          
                                          min{a, b},                if 1 ≤ q ≤ 2,
               min axq/2 + b(1 − x)q/2   =  2         2
                                                           2−q
                                                             2
              x∈[0,1]                      a 2−q + b 2−q       ,    if q > 2.


Proof of Proposition 1.5. Let K ⊂ R2n and T ⊂ R2m be two convex bodies
with smooth boundaries, and let 1 < p < ∞. Note that the proof of the two
cases p = 1 and p = ∞ follows from the above case by a standard continuation
argument. Let γ ⊂ En+m be a minimizer of (4) for the body K ×p T and denote
by γ1 and γ2 the projections of γ on R2n and R2m , respectively. From the fact
that γ is a rescaling and translation of a closed characteristic on ∂(K ×p T ), its
velocity γ̇(t) is equal to a positive constant times JnK×p T (γ(t)). Using (3) and
the convexity of K and T , we get that A(γ1 ), A(γ2 ) > 0. Moreover, the actions
of γ1 and γ2 satisfy A(γ1 ) + A(γ2 ) = 1. Note moreover that from the definition
of K ×p T it follows that

                              hK×p T (x, y)q = hK (x)q + hT (y)q ,
      1       1
for   p   +   q   = 1. Using (4) for K ×p T and (5) for K and for T gives

                                Z 2π                   2/q
                               1 q      q
          cEHZ (K ×p T ) = π           hK×p T (γ̇(t))dt
                              2π 0
                                Z 2π                         Z 2π                2/q
                             q 1        q                 q 1       q
                         = π           hK (γ̇1 (t))dt + π          hT (γ̇2 (t))dt
                              2π 0                         2π 0
                          h                                               i2/q
                         ≥ A(γ1 )q/2 cEHZ (K)q/2 + A(γ2 )q/2 cEHZ (T )q/2      .

For p ≥ 2, using Lemma 2.1 one has that

                           cEHZ (K ×p T ) ≥ min{cEHZ (K), cEHZ (T )}.

Next, since for every symplectic subspace E and any convex body P one has
cEHZ (P ) ≤ cEHZ (πE P ), where πE stands for the projection operation, we get
that cEHZ (K ×p T ) ≤ min{cEHZ (K), cEHZ (T )}, and hence

                           cEHZ (K ×p T ) = min{cEHZ (K), cEHZ (T )}.




                                               8
For 1 ≤ p < 2, Lemma 2.1 gives
                                                            2−q 2/q
                                          q               q     2
               cEHZ (K ×p T ) ≥ cEHZ (K) 2−q + cEHZ (T ) 2−q


                                               1                 1
                                                                      1−2/p
                                   = cEHZ (K) 1−2/p + cEHZ (T ) 1−2/p        .

For the other direction, take the minimizers γ1 ∈ En and γ2 ∈ Em of (4) for K
and T , respectively. Recall that
                                they are qnormalized so that       A(γ1 ) = A(γ2 ) = 1.
                                                                 q
Consider now the curve γ = cEHZ (K)     2(2−q) γ1 , cEHZ (T ) 2(2−q) γ2 . Then,

cEHZ (K ×p T ) ≤
              "           Z   2π
                      1                                 q
       A(γ)−1 π q                  hqK (cEHZ (K) 2(2−q) γ̇1 (t))dt
                     2π   0

                                                                Z                                            #2/q
                                                                    2π
                                                         1                                  q
                                                     +π     q
                                                                          hqT (cEHZ (T ) 2(2−q) γ̇2 (t))dt
                                                        2π      0
                                                    "
                    q                   q
                                              −1                    q2                q
  =        cEHZ (K) 2−q + cEHZ (T ) 2−q              cEHZ (K) 2(2−q) cEHZ (K) 2
                                                                                             #2/q
                                                                      q2                 q
                                                    +cEHZ (T )      2(2−q)   cEHZ (T )   2



                    q                   q
                                              2/q−1                1                 1
                                                                                           1−2/p
  =        cEHZ (K) 2−q + cEHZ (T ) 2−q                 = cEHZ (K) 1−2/p + cEHZ (T ) 1−2/p        ,

and the proof of the proposition is now complete.

  Now let us examine closed characteristics on K ×p T from a dynamical point of
view. Note that (3) implies that if γ1 (t) ⊂ ∂K and γ2 (t) ⊂ ∂T are characteristics
with
                        2JnK (γ1 (t))                   2JnT (γ2 (t))
             γ˙1 (t) =                   and γ˙2 (t) =
                       hK (nK (γ1 (t)))                hT (nT (γ2 (t)))
then
                              γ(t) = (αγ1 (αp−2 t), βγ2 (β p−2 t))
is a characteristic on ∂(K ×p T ) with the parametrization

                                               2JnK×p T (γ(t))
                               γ̇(t) =                              .
                                             hK×p T (nK×p T (γ(t)))

Assume first that α 6= 0 and β 6= 0. Note that the curve γ is closed if and only
if γ1 and γ2 are closed and there exists t0 ∈ R such that γ1 (αp−2 t0 ) = γ1 (0)
and γ2 (β p−2 t0 ) = γ2 (0), hence t1 := αp−2 t0 and t2 := β p−2 t0 are periods of γ1



                                                    9
and γ2 , respectively, which satisfy t1 /αp−2 = t2 /β p−2 . As αp + β p = 1, one gets
that                                           p

                                  p         t1p−2
                                α = p              p ,

                                        t1p−2 + t2p−2
and                                                   p          p    p−2
                         t0 = t1 /αp−2 = (t1p−2 + t2p−2 )              p    .
In the case that either α = 0 or β = 0 one gets that γ = (0, γ2 ) or γ = (γ1 , 0) and
t0 = t2 or t0 = t1 , respectively. Finally, recall that in the above parametrization
the minimal period gives the capacity, and hence

  cEHZ (K ×p T ) = min{t0 : t0 ∈ P(K ×p T )}
                                  p           p       p−2
                 = min{min((t1p−2 + t2p−2 )            p    , t1 , t2 ) : t1 ∈ P(K), t2 ∈ P(T )}
                                       p                        p    p−2
                 = min((cEHZ (K)      p−2   + cEHZ (T ) p−2 )         p    , cEHZ (K), cEHZ (T )),

where P(Q) is the set of periods of all the closed characteristics on ∂Q for a
convex body Q ⊂ R2n . Note that if p > 2, then
                                      p           p       p−2
                   ∀x, y > 0, (x p−2 + y p−2 )             p    > max{x, y},

and if 1 ≤ p < 2, then
                                      p           p       p−2
                    ∀x, y > 0, (x p−2 + y p−2 )            p    < min{x, y}.

Hence when p > 2,

                    cEHZ (K ×p T ) = min{cEHZ (K), cEHZ (T )},

and when 1 ≤ p < 2,
                                                      p                         p   p−2
                cEHZ (K ×p T ) = (cEHZ (K) p−2 + cEHZ (T ) p−2 )                     p    .

This observation reproves Proposition 1.5 in a somewhat different way.
  For the proof of Theorem 1.2 we need the following lemma,
                                                         R       which is a sim-
                                                    1           −kxkp
ple corollary of the well-known formula Vol(K) = Γ(1+ n
                                                        ) R n e     K dx for the
                                                                            p
volume of a convex body K ⊂ Rn (see e.g. [22]).
Lemma 2.2. For convex K ⊂ Rn and T ⊂ Rm one has
                                 Γ( np + 1)Γ( m
                                              p + 1)
                Vol(K ×p T ) =                                      Vol(K)Vol(T ).
                                          Γ( m+n
                                              p + 1)


Proof of Theorem 1.2. First let p ≥ 2. Then, from Lemma 2.2 and the fact
that K ×2 T ⊆ K ×p T it follows that
                                                            m!n!
            Vol(K ×p T ) ≥ Vol(K ×2 T ) =                          Vol(K)Vol(T ),
                                                          (m + n)!

                                             10
with equality only when p = 2. Next, from Proposition 1.5 it follows that
                                                                           n              m
     cEHZ (K ×p T ) = min{cEHZ (K), cEHZ (T )} ≤ cEHZ (K) m+n cEHZ (T ) m+n ,

with equality only when cEHZ (K) = cEHZ (T ). Hence

                                      cEHZ (K)n cEHZ (T )m
     sysm+n (K ×p T )m+n ≤                                 = sysn (K)n sysm (T )m ,
                                      n!Vol(K) m!Vol(T )

with equality only when p = 2 and when cEHZ (K) = cEHZ (T ).
  Next, let 1 ≤ p < 2. Using Proposition 1.5 and the following inequality,
which is a simple consequence of the inequality of the weighted arithmetic and
geometric means, one has:
                                      p               p
                                                           p−2
                                                             p
            cEHZ (K ×p T ) = cEHZ (K) p−2 + cEHZ (T ) p−2                                     (6)
                                                     p−2
                                       n+m              p
                                                                       n            m
                            ≤          m     n               cEHZ (K) n+m cEHZ (T ) n+m
                                    m n+m n n+m
Plugging (6) in the formula for the systolic ratio gives
                                                  p               p
                                                                       p−2
                                                                         p (m+n)
                                         cEHZ (K) p−2 + cEHZ (T ) p−2
        sysm+n (K ×p T )m+n =
                                                  (m + n)!Vol(K ×p T )
                                                        p−2
                                               (n+m)m+n    p

                                                 mm nn        cEHZ (K)n cEHZ (T )m
                                    ≤
                                     (n + m)! Γ(1+2n/p)Γ(1+2m/p)
                                                Γ(1+(2n+2m)/p) Vol(K)Vol(T )
                                        
                                        1 cEHZ (K)n cEHZ (T )m
                                    =g                          ,
                                        p n!Vol(K) m!Vol(T )

where
                                       1−2x
                        (n + m)m+n                 Γ(1 + (2n + 2m)x)    n!m!
         g(x) :=                                                               .
                           mm n n                Γ(1 + 2nx)Γ(1 + 2mx) (m + n)!

Hence, to finish the proof it remains to check that

                                            g(x) ≤ 1,
            1
for every   2   < x ≤ 1. As in [15, Lemma 4.5], a direct computation shows that

                                         d2
                                            ln[g(x)] > 0,
                                        dx2
and hence
                                g(x) ≤ max{g(1/2), g(1)},


                                                 11
for every x ∈ [1/2, 1]. As g(1/2) = 1 it remains to show that

                                    mm n n  (2n + 2m)! n!m!
                   1 ≥ g(1) =           n+m
                                                                .
                                 (n + m)    (2n)!(2m)! (n + m)!
One is able to show this directly using, e.g., the following bounds:
               √                  1           √                 1
                 2πnn+1/2 e−n e 12n+1 < n! < 2πnn+1/2 e−n e 12n .

Alternatively, note that g(1) is the systolic ratio of a 1-product of Euclidean balls
of different radii. Indeed, choose K ⊂ R2n to be the Euclidean ball of capacity
                                                                             n
1, and choose T ⊂ R2m to be the Euclidean ball of capacity c(T ) = m           c(K).
Now one has equality in (6) above when p = 1, and hence

             sysm+n (K ×1 T )m+n = g(1)sysn (K)n sysm (T )m = g(1).

Since K ×1 T is S 1 -symmetric, cEHZ (K ×1 T ) = cGr (K ×1 T ) and the largest sym-
plectic ball inside K ×1 T is a Euclidean ball (cf. Proposition 1.4 in [13]). This
                                                                    d2
gives g(1) = sysm+n (K ×1 T )m+n < 1, as required. Moreover, dx       2 ln[g(x)] > 0

implies that g(x) < g(1/2) for x ∈ (1/2, 1]. Therefore, the inequality is strict
whenever 1 ≤ p < 2.

Proof of Corollary 1.4. Assume without loss of generality that m ≥ n. For
a convex K ⊂ R2n take a ball B 2(m−n) [cEHZ (K)] with the same capacity as K.
Now Theorem 1.2 gives

              sysn (K)n = sysn (K)n sysm−n (B 2(m−n) [cEHZ (K)])m−n
                           = sysm (K ×2 B 2(m−n) [cEHZ (K)])m .

Thus, Conjecture 1.1 in dimension 2m implies Conjecture 1.1 in dimension 2n
for any n ≤ m. Moreover, if we know that sysn (K) ≤ α(n) for any K and n,
                                         n→∞
for some function α that satisfies α(n) −−−−→ 1, then we can take a product of
K with itself m-times to conclude that

                  sysnm (K ×2 K ×2 · · · ×2 K)mn = (sysn (K)n )m ,

and since sysnm (K ×2 · · · ×2 K) ≤ α(nm), letting m → ∞ we get

                                       sysn (K) ≤ 1.

This completes the proof of the corollary.


3     Capacities of p-products of toric domains

In this section we prove Theorem 1.8 and Theorem 1.9. Define the moment map
µ : Cn → Rn+ by µ(z1 , . . . , zn ) = π(|z1 |2 , . . . , |zn |2 ). For a domain Ω ⊂ Rn+ define

                                             12
the corresponding toric domain by XΩ = µ−1 (Ω) ⊂ Cn and the continuation of
Ω to an unconditional domain by Ωb = {(x1 , . . . , xn ) ∈ Rn |(|x1 |, . . . , |xn |) ∈ Ω}.
We say XΩ is a convex (concave) toric domain if Ω     b is a convex (concave) set.
For more information on toric domains see, e.g., Section 2 of [13]. For the proof
of the theorem mentioned above we shall need the following two simple lemmas.
Lemma 3.1. Let XΩ1 ⊂ Cn and XΩ2 ⊂ Cm be two toric domains. Then, for
every 1 ≤ p ≤ ∞ one has XΩ1 ×p XΩ2 = XΩ1 ×p/2 Ω2 .

Proof of Lemma 3.1. First note that kµ(x)kΩ = kxk2XΩ . Indeed, since µ(αx) =
α2 µ(x), one has µ(x)       x                                       n    m
                  λ2 ∈ Ω ⇐⇒ λ ∈ XΩ for λ ∈ R+ . Next, for (x, y) ∈ C × C ,

               (x, y) ∈ XΩ1 ×p/2 Ω2 ⇐⇒ (µ(x), µ(y)) ∈ Ω1 ×p/2 Ω2
                                                    p/2            p/2
                                      ⇐⇒ kµ(x)kΩ1 + kµ(y)kΩ2 ≤ 1
                                      ⇐⇒ kxkpXΩ + kykpXΩ ≤ 1
                                                    1          2

                                      ⇐⇒ (x, y) ∈ XΩ1 ×p XΩ2 ,

which completes the proof.
Lemma 3.2. For convex domains Ω1 ⊂ Rn and Ω2 ⊂ Rm , the domain Ω1 ×p Ω2
is convex for p ≥ 1, and for concave domains Ω1 ⊂ Rn and Ω2 ⊂ Rm , the
domain Ω1 ×p Ω2 is concave for 0 < p ≤ 1.

Proof of Lemma 3.2. For (x, y), (w, z) ∈ Ω1 ×p Ω2 and p ≥ 1, it follows from
Minkowski inequality that
                                                               1/p
     k(x, y) + (w, z)kΩ1 ×p Ω2 = kx + wkpΩ1 + ky + zkpΩ2
                                                                                 1/p
                                 ≤ ((kxkΩ1 + kwkΩ1 )p + (kykΩ2 + kzkΩ2 )p )
                                                   1/p                   1/p
                                 ≤ kxkpΩ1 + kykpΩ2      + kwkpΩ1 + kzkpΩ2
                                 = k(x, y)kΩ1 ×p Ω2 + k(w, z)kΩ1 ×p Ω2 ,

which proves the first part of the lemma. In a similar manner, a direct compu-
tation shows that for 0 < p ≤ 1, using now the reverse Minkowski inequality,
one has

           k(x, y) + (w, z)kΩ1 ×p Ω2 ≥ k(x, y)kΩ1 ×p Ω2 + k(w, z)kΩ1 ×p Ω2 ,

which proves the second part of the lemma.

   Next, consider the family {ckGH }∞
                                    k=1 of the Gutt-Hutchings symplectic capac-
ities introduced in [12]. The proofs of Theorem 1.8 is based on the followin:




                                            13
Theorem 3.3 (Gutt, Hutchings [12]). For a convex toric domain XΩ ⊂ Cn ,
                      (                                    n
                                                                   )
                                                           X
        k                                               n
       cGH (XΩ ) = min hΩ (v) v = (v1 , . . . , vn ) ∈ N ,   vi = k ,
                                                                             i=1

and for a concave toric domain XΩ ⊂ Cn ,
                           (                                                          )
                                                          X
           ckGH (XΩ )     = max [v]Ω v ∈         Nn>0 ,         vi = k + n − 1 ,
                                                            i
                
where [v]Ω = min hv, wi w ∈ Σ , and Σ is the clouser of the set ∂Ω ∩ Rn>0 .

Proof of Theorem 1.8. Consider first the case of convex toric domains K
and T , and p > 2. As before, the case p = 2 will follow from a standard
continuation argument. Then, using Lemma 3.1 and Lemma 3.2, one has that
K ×p T = Xµ(K)×p/2 µ(T ) is a convex toric domain, and hence, by Theorem 3.3,

       ckGH (K ×p T ) = ckGH (Xµ(K)×p/2 µ(T ) )
                  (                                             n+m
                                                                               )
                                                                X
                                                     n+m
          = min hµ(K)×p/2 µ(T ) (v) v ∈ N                   ,         vi = k
                                                                i=1
                  (
                           p                          p                           p−2
                                                                                     p
                           p−2                        p−2
          = min           hµ(K) (v1 , . . . , vn ) + hµ(T ) (vn+1 , . . . , vn+m )

                                                 n+m
                                                                  )
                                                 X
                                   v ∈ Nn+m ,            vi = k
                                                 i=1
                      "        (                          n
                                                                         )
                                     p                    X
                                    p−2              n
          =   min         min hµ(K) (v) v ∈ N ,                 vr = i
              i+j=k
                                                          r=1
                                          (                           m
                                                                                      ) # p−2
                                               p                      X                    p

                                   + min hµ(T ) (v) v ∈ Nm ,
                                              p−2
                                                                             vr = j
                                                                      r=1
                  h         p               p
                                               i p−2
                                                  p
          =   min ciGH (K) p−2 + cjGH (T ) p−2       .
              i+j=k

For the case of concave domains K and T and 1 ≤ p ≤ 2 one has the following.
                                                 p         p     p−2
                                                                    p
                                                 p−2       p−2
                [(x, y)]µ(K)×p/2 µ(T )      = [x]µ(K) + [y]µ(T )       ,




                                               14
and hence, using similar arguments as above, one has

    ckGH (K ×p T ) = ckGH (Xµ(K)×p/2 µ(T ) )
               (                                         n+m
                                                                                       )
                                                         X
      =    max [v]µ(K)×p/2 µ(T ) v ∈          Nn+m
                                               >0 ,             vi = k + m + n − 1
                                                         i=1
                 (
                                          p                              p     p−2
                                                                                  p
                                          p−2                            p−2
      =    max        [(v1 , . . . , vn )]µ(K) + [(vn+1 , . . . , vn+m )]µ(T )

                                             n+m
                                                                             )
                                             X
                               v∈   Nn+m
                                     >0 ,           vi = k + m + n − 1
                                              i=1
                      "       (                          n
                                                                                )
                                     p                   X
      =      max                    p−2
                          min [v]µ(K) v ∈       Nn>0 ,         vr = i + n − 1
           i+j=k+1
            i>0,j>0                                      r=1
                                      (                          m
                                                                                        ) # p−2
                                              p                  X                           p
                                             p−2
                              + min [v]µ(T ) v ∈         Nm
                                                          >0 ,         vr = j + m − 1
                                                                 r=1
                      h          p               p
                                                    i p−2
                                                       p
      =      max       ciGH (K) p−2 + cjGH (T ) p−2       ,
           i+j=k+1
            i>0,j>0

and the proof of the theorem is complete.

Proof of Theorem 1.9. Note that it is enough to prove (2) for the product
of two convex domains. Namely, let K ⊂ R2n and T ⊂ R2m be convex domains
                     ck (K)     ck (T )
such that the limits EHk    and EHk     exist. Then we wish to prove that
                                         −p          −p
                                                          −p
                                                            2
                     c∞ (K ×p T ) = c∞ (K) 2 + c∞ (T ) 2      .

We will prove this in the case where p > 2. The proof for 1 ≤ p < 2 is similar.
One can derive the case p = 2 by continuity of the capacities and the inclusions

                           K ×2−ε T ⊂ K ×2 T ⊂ K ×2+ε T.

First we will show that
                                         −p          −p
                                                          −p
                                                            2
                     c∞ (K ×p T ) ≤ c∞ (K) 2 + c∞ (T ) 2      .

Denote                                                    p
                                               c∞ (T ) 2
                                  w=             p           p
                                          c∞ (K) 2 + c∞ (T ) 2
and choose
                      ik = ⌈kw⌉,            jk = k − ik = ⌊k(1 − w)⌋.



                                                   15
By the definition of c∞ , for every ε > 0 there exists a large enough k such that
                                                                                1
            ciEH
               k
                 (K) < ik (c∞ (K) + ε),     cjEH
                                               k
                                                 (T ) < jk (c∞ (T ) + ε),         < ε.
                                                                                k
Using Conjecture 1.6 we get
                h          p               p
                                              i p−2
                                                 p
ckEH (K ×p T ) ≤ ciEH
                    k
                      (K) p−2 + cjEH
                                   k
                                     (T ) p−2
                   h p                   p       p                 p
                                                                      i p−2
                                                                         p
                  < ikp−2 (c∞ (K) + ε) p−2 + jkp−2 (c∞ (T ) + ε) p−2
                                                                                                          p−2
                        p        1 − {wk} p−2 p                p          p          p                 p     p
                  ≤ k  p−2 (w +            )    (c∞ (K) + ε)  p−2  +k   p−2 (1 − w) p−2 (c∞ (T ) + ε) p−2
                                     k
                     h                                              i p−2        
                            p          p              p           p     p
                  ≤k w     p−2 c∞ (K) p−2  + (1 − w) p−2 c∞ (T ) p−2        + O(ε) ,

By substituting w one gets
     ckEH (K ×p T )      c∞ (K)c∞ (T )            p         p
                                                                p−2
                                                                  p
                    <        p         p  c∞ (T ) 2 + c∞ (K) 2       + O(ε)
            k         c∞ (K) + c∞ (T )
                             2         2

                              p           p
                                             − p2
                    = c∞ (K)− 2 + c∞ (T )− 2       + O(ε).                  (7)


   For the other direction, denote by îk , ĵk the minimizers in the formula for
ckEH (K ×p T ) from Conjecture 1.6. We consider two cases. The first is that both
îk and ĵk goes to infinity as k goes to infinity, and the second is that one of
the indices, say îk , is bounded. In the first case, since îk → ∞ and ĵk → ∞ as
k → ∞, the definition of c∞ implies that for every ε > 0 there exists a large
enough k such that

                 cîEH
                     k
                       (K) > îk (c∞ (K) − ε),      cĵEH
                                                        k
                                                          (T ) > ĵk (c∞ (T ) − ε).

Hence, one has
                          h           p                p
                                                          i p−2
                                                             p
          ckEH (K ×p T ) = cîEH
                               k
                                 (K) p−2 + cĵEH
                                               k
                                                 (T ) p−2
                          h p                   p        p                 p
                                                                              i p−2
                                                                                 p
                         > îkp−2 (c∞ (K) − ε) p−2 + ĵkp−2 (c∞ (T ) − ε) p−2       .

Without the restriction that îk , ĵk ∈ N, the right hand side is minimized when
                                   p                                                  p
                 k(c∞ (T ) − ε) 2                                   k(c∞ (K) − ε) 2
 îk :=                p                 p ,        ĵk :=                p                 p ,
          (c∞ (K) − ε) 2 + (c∞ (T ) − ε) 2                   (c∞ (K) − ε) 2 + (c∞ (T ) − ε) 2
and hence,
                                              p                  p
                                                                    − p2
              ckEH (K ×p T ) > k (c∞ (K) − ε)− 2 + (c∞ (T ) − ε)− 2       .


                                               16
Consequently,
               ckEH (K ×p T )         p            p
                                                      − p2
                             > c∞ (K)− 2 + c∞ (T )− 2       + O(ε).
                      k
Suppose now that îk ≤ i0 , which means ĵk ≥ k − i0 , for every k. For every
ε > 0, take a large enough k so that
                h           p                p
                                                i p−2
                                                   p
                                                       h        p                               p
                                                                                                   i p−2
                                                                                                      p
ckEH (K ×p T ) = cîEH
                     k
                       (K) p−2 + cĵEH
                                     k
                                       (T ) p−2       > c1 (K) p−2 + ((k − i0 ) (c∞ (T ) − ε)) p−2       .

Hence
                                      ckEH (K ×p T )
                    c∞ (K ×p T ) = lim               > c∞ (T )
                                  k→∞        k
                                               p           p − 2
This contradicts (7), since c∞ (T ) > c∞ (K)− 2 + c∞ (T )− 2 p . The proof of
the theorem is now complete.


4     On the p-decomposition of a symplectic ball

In this section we prove Corollary 1.12 and Theorem 1.13. We start with the
proof of the former.

Proof of Corollary 1.12. Assume that Conjecture 1.1 holds. The first part
of the corollary follows immediately from the fact that if p 6= 2 then

                sysm+n (K ×p T )m+n < sysn (K)n sysm (T )m ≤ 1,

so the systolic ratio is strictly smaller than that of the ball. Similarly, if one has
Be = X ×2 Y and sysn (X) < 1 or sysm (Y ) < 1 then

                            e m+n ≤ sysn (K)n sysm (T )m < 1,
                    sysm+n (B)

which completes the proof of the corollary.

  The following lemma, whose proof is a direct consequence of Theorem 1.2 and
Lemma 3.4 in [10], is key in the proof of Theorem 1.13.
Lemma 4.1. For any convex K ⊂ R2n , and any i ∈ N,

                                 ciEH (K) < cn+i
                                             EH (K).


Proof of Theorem 1.13. Assume without loss of generality that r = 1 and
n ≤ m. Assume, by contradiction, that B 2n+2m [1] = X ×p Y . First consider
the case p > 2. From Conjecture 1.6 one has
                     h         p               p
                                                  i p−2
                                                     p
ckEH (X ×p Y ) = min ciEH (X) p−2 + cjEH (Y ) p−2       ≤ min{ckEH (X), ckEH (Y )},
                  i+j=k


                                           17
and hence ckEH (X) ≥ 1, ckEH (Y ) ≥ 1 for any k ≤ m + n. Consider k = m + 1.
Using Conjecture 1.6 again, we get that
                                              h          p               p
                                                                            i p−2
                                                                               p
         1 = cm+1
              EH (X ×p Y ) =          min      ciEH (X) p−2 + cjEH (Y ) p−2       .
                                    i+j=m+1

Note that for 0 < i, j ≤ m one has
                     h          p               p
                                                   i p−2
                                                      p
                      ciEH (X) p−2 + cjEH (Y ) p−2       > ciEH (X) ≥ 1,

and using Lemma 4.1 for i = m + 1, j = 0, and i = 0, j = m + 1, one has

                cm+1       1
                 EH (X) > cEH (X) ≥ 1,            cm+1        1
                                                   EH (Y ) > cEH (Y ) ≥ 1,

which is a contradiction. Next, for 1 ≤ p < 2, assume that i, j are maximizers
in the formula from Conjecture 1.6 for the index k = m + n, i.e.,
                                    h           p               p
                                                                   i p−2
                                                                      p
                                                       j
              1 = cm+n
                   EH  (X × p Y ) =   c i
                                        EH (X) p−2 + c
                                                       EH (Y ) p−2       .

Since i + j = m + n + 1 we may assume without loss of generality that i > n,
and hence Lemma 4.1 gives ciEH (X) > c1EH (X). In addition note that
                                   h          p               p
                                                                 i p−2
                                                                    p
               1 = c1EH (X ×p Y ) = c1EH (X) p−2 + c1EH (Y ) p−2       .

Thus, we conclude that
         h          p               p
                                       i p−2
                                          p
                                               h         p               p
                                                                            i p−2
                                                                               p
      1 = ciEH (X) p−2 + cjEH (Y ) p−2       > c1EH (X) p−2 + cjEH (Y ) p−2
         h          p               p
                                       i p−2
                                          p
        ≥ c1EH (X) p−2 + c1EH (Y ) p−2       = 1,

which is again a contradiction. This completes the first part of the theorem.
  Finally, when p = 2, note that ckEH (X ×2 Y ) is the k th term in the sequence
{ciEH (X), cjEH (Y
                )} arranged in non-decreasing order with repetitions. Denote
the elements in this sequence by Mk . From the fact that ckEH (X ×2 Y ) = 1 for
1 ≤ k ≤ m + n, one has that M1 = M2 = . . . = Mm+n = 1. By means of
Lemma 4.1 we get that the only way this is possible is if

        c1EH (X) = . . . = cnEH (X) = 1, and c1EH (Y ) = . . . = cm
                                                                  EH (Y ) = 1.

Next, using the fact that Mm+n+1 = . . . = M2m+2n = 2 we get again by
invoking Lemma 4.1 that

       cn+1               2n              m+1                2m
        EH (X) = . . . = cEH (X) = 2 and cEH (Y ) = . . . = cEH (Y ) = 2.

Continuing this argument by induction completes the proof.

                                             18
5      Appendix

Denote by Mk (K, T ) the k-th term in the sequence A(K, T ) := {ciEH (K), cjEH (T )},
arranged in non-decreasing order with repetitions.
Lemma 5.1. The two expressions in Conjecture 1.6 coincide for p = 2, and
moreover they are equal to Mk (K, T ), i.e.

    min max{ciEH (K), cjEH (T )} =     max min{ciEH (K), cjEH (T )} = Mk (K, T ).
    i+j=k                            i+j=k+1


Proof of Lemma 5.1. First, we may assume that all elements in A(K, T ) are
unique. Otherwise, one can assign an arbitrary strict total order ≺ to A(K, T )
that satisfies that if a, b ∈ A(K, T ) and a < b then a ≺ b, and ciEH (K) ≺ cjEH (K)
and ciEH (T ) ≺ cjEH (T ) for all i < j. Then one can continue the proof where each
time we consider inequalities between elements of A(K, T ) one can consider
instead the same inequalities with ≺.
   We start with the case mini+j=k max{ciEH (K), cjEH (T )} ≥ Mk (K, T ). Let i, j
be the minimizers in the above inequality, and assume without loss of generality
that ciEH (K) ≥ cjEH (T ). Then the maximum is ciEH (K), and there are at least k
elements in {clEH (K)}∞             l       ∞                          i
                            l=1 ∪ {cEH (T )}l=1 that are smaller than cEH (K). Indeed,
c1EH (K), . . . , ciEH (K) are i such elements, and because ciEH (K) ≥ cjEH (T ), one
has that c1EH (T ), . . . cjEH (T ) ≤ ciEH (K) and hence there are additional j such
elements and overall there are at least i + j = k such elements. Now for the case
mini+j=k max{ciEH (K), cjEH (T )} ≤ Mk (K, T ), assume without loss of generality
that the k-th element is ciEH (K) for some i. Obviously one has that i ≤ k. Put
j = k − i. It is enough to show that ciEH (K) ≥ cjEH (T ). If cjEH (T ) > ciEH (K)
then there are less than k elements that are smaller than ciEH (K). Indeed,
                                                                                j−1
ci+1             i                                                  1
 EH (K) > cEH (K), and there are at most j − 1 elements cEH (T ), . . . , cEH (T )
                          i
that are less than cEH (K) and hence there are at most i + j − 1 < k such
elements. Hence ciEH (K) ≥ cjEH (T ) as required.

   The next case to consider is maxi+j=k+1 min{ciEH (K), cjEH (T )} ≤ Mk (K, T ).
Let i, j be the maximizers in the above inequality, and assume without loss of
generality that ciEH (K) ≤ cjEH (T ). Then there are at most k − 1 elements that
are strictly smaller than ciEH (K). Indeed, c1EH (K), . . . , ci−1 EH (K) are i − 1 such
elements, and c1EH (T ), . . . , cj−1
                                    EH  (T ) are j − 1 such elements, so overall there are
at most i − 1 + j − 1 = k − 1 such elements and ciEH (K) ≤ Mk (K, T ). Finally let
us deal with the case maxi+j=k+1 min{ciEH (K), cjEH (T )} ≥ Mk (K, T ). Assume
without loss of generality that the k-th element is Mk (K, T ) = ciEH (K) for some
i. Put j = k + 1 − i. Then cjEH (T ) ≥ ciEH (K), otherwise there are j elements
c1EH (T ), . . . , cjEH (T ) that are strictly smaller than ciEH (K) plus i − 1 elements
c1EH (K), . . . , ci−1                                        i
                     EH (K) that are strictly smaller than cEH (K), and hence there are
i − 1 + j = k elements that are strictly smaller than ciEH (K), which means that



                                           19
ciEH (K) > Mk (K, T ). Therefore,

 max min{ciEH (K), cjEH (T )} ≥ min{ciEH (K), cjEH (T )} = ciEH (K) = Mk (K, T ),
i+j=k+1

and the proof of the lemma is thus complete.


References
 [1] Abbondandolo, A., Kang, J. Symplectic homology of convex domains and
     Clarke’s duality, arXiv:1907.07779.
 [2] Artstein-Avidan, S., Ostrover, Y. Bounds for Minkowski billiard trajectories
     in convex bodies, IMRN 2014, 165–193.
 [3] Artstein-Avidan, S., Ostrover, Y. A Brunn-Minkowski inequality for sym-
     plectic capacities of convex domains, IMRN 2008.
 [4] de Bruijn, N. G., Erdös, P. Some linear and some quadratic recursion for-
     mulas. II, Nederl. Akad. Wetensch. Proc. Ser. A. 55 = Indagationes Math.
     1952, 152–163.
 [5] Cieliebak, K., Hofer, H., Latschev, J., Schlenk, F. Quantitative symplectic
     geometry, in Dynamics, Ergodic Theory, and Geometry, 1-44, Math. Sci.
     Res. Inst. Publ.54, Cambridge University Press, 2007.
 [6] Cieliebak, K., Mohnke, K., Punctured holomorphic curves and Lagrangian
     embeddings, Invent. Math. 212 (2018), no. 1, 213–295.
 [7] Clarke, F. H. A classical variational principle for periodic Hamiltonian
     trajectories, Proc. Amer. Math. Soc. 76 (1979), 186–188.
 [8] Ekeland, I., Hofer, H. Symplectic topology and Hamiltonian dynamics,
     Math. Z. 200 (1989), 355–378.
 [9] Ekeland, I., Hofer, H. Symplectic topology and Hamiltonian dynamics II,
     Math. Z. 203 (1990), 553–567.
[10] Ginzburg, V. L., Gürel, B. Z., Mazzucchelli, M. On the spectral charac-
     terization of Besse and Zoll Reeb flows, Ann. Inst. H. Poincaré Anal. Non
     Linéaire 38 (2021), 549–576
[11] Ginzburg, V., Shon, J. On the filtered symplectic homology of prequantiza-
     tion bundles, Int. J. Math. 29 (2018), 1850071, 35pp.
[12] Gutt, J., Hutchings, M. Symplectic capacities from positive S 1 -equivariant
     symplectic homology, Algebr. Geom. Topol. 18 (2018), 3537–3600.
[13] Gutt, J., Hutchings, M., Ramos, V. G. B. Examples around the strong
     Viterbo conjecture, arXiv:2003.10854.

                                       20
[14] Henk, M., Richter-Gebert, J., and Ziegler, G. M. Basic properties of convex
     polytopes. In Handbook of Discrete and Computational Geometry, CRC
     Press Ser. Discrete Math. Appl., pages 243–270. CRC, Boca Raton, FL,
     1997.
[15] Henze, M. The Mahler Conjecture, Diploma Thesis. Otto-von-Guericke-
     Universität Magdeburg, (2008).
[16] Hofer, H., Zehnder, E. A new capacity for symplectic manifolds, In Analysis,
     et cetera, 405–427, Academic Press, Boston MA (1990).
[17] Hutchings, M. Private Communication.
[18] Irie, K. Symplectic homology of fiberwise convex sets and homology of loop
     spaces, arXiv:1907.09749.
[19] Kerman, E., Liang, Y. On symplectic capacities and their blind spots,
     arXiv:2109.01792, 2021.
[20] McDuff, D. Symplectic Topology Today, AMS Joint Mathematics Meeting,
     Baltimore, Colloquium Lectures, 2014.
[21] Ostrover, Y., Ramos, G.B.V. Symplectic embeddings of the ℓp sum of two
     discs, J. Topol. Anal., https://doi.org/10.1142/S1793525321500242.
[22] Schneider, R. Convex Bodies: the Brunn-Minkowski Theory, Encyclopedia
     of Mathematics and its Applications, 44. Cambridge University Press, 1993.
[23] Tao, T. Structure and Randomness. Pages from Year One of a Mathemat-
     ical Blog, American Mathematical Society, Providence, RI, 2008.
[24] Viterbo, C. Metric and isoperimetric problems in symplectic geometry, J.
     Am. Math. Soc., 13(2):411–431, 2000.
[25] Viterbo, C. Functors and computations in Floer homology with applications
     I, Geom. Funct. Anal. 9 (1999), 985–1033.

Pazit Haim-Kislev
School of Mathematical Sciences, Tel Aviv University, Israel
e-mail: pazithaim@mail.tau.ac.il

Yaron Ostrover
School of Mathematical Sciences, Tel Aviv University, Israel
e-mail: ostrover@tauex.tau.ac.il




                                       21
