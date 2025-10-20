---
source: arXiv:2407.08512
fetched: 2025-10-20
---
# Anchored symplectic embeddings

                                                                     Anchored symplectic embeddings
                                                           Michael Hutchings, Agniva Roy, Morgan Weiler, and Yuan Yao


                                                                                            Abstract
arXiv:2407.08512v1 [math.SG] 11 Jul 2024




                                                     Given two four-dimensional symplectic manifolds, together with knots in their boundaries,
                                                 we define an “anchored symplectic embedding” to be a symplectic embedding, together with a
                                                 two-dimensional symplectic cobordism between the knots (in the four-dimensional cobordism
                                                 determined by the embedding). We use techniques from embedded contact homology to deter-
                                                 mine quantitative critera for when anchored symplectic embeddings exist, for many examples
                                                 of toric domains. In particular we find examples where ordinarily symplectic embeddings exist,
                                                 but they cannot be upgraded to anchored symplectic embeddings unless one enlarges the target
                                                 domain.


                                           1     Introduction
                                           1.1    Basic definitions
                                           Let pX, ωq and pX 1 , ω 1 q be compact symplectic manifolds with boundary of the same dimension. A
                                           symplectic embedding of pX, ωq into pX 1 , ω 1 q is a smooth embedding φ : X Ñ X 1 such that φ˚ ω 1 “ ω.
                                           Since Gromov proved the celebrated nonsqueezing theorem [13] in 1985, there has been much work
                                           studying when symplectic embeddings exist; see e.g. the surveys [6, 22, 37].
                                              In this paper we consider a “relative” version of this question in the four dimensional case.
                                           Suppose that pX, ωq and pX 1 , ω 1 q are compact symplectic four-manifolds with boundary Y and Y 1 .
                                           Let γ and γ 1 be oriented knots in Y and Y 1 respectively.

                                           Definition 1.1. An anchored symplectic embedding

                                                                                   pX, ω, γq ÝÑ pX 1 , ω 1 , γ 1 q

                                           is a pair pφ, Σq, where
                                                                                 φ : pX, ωq ÝÑ pintpX 1 q, ω 1 q
                                           is a symplectic embedding, and
                                                                                      Σ Ă X 1 zφpintpXqq
                                           is a smoothly embedded symplectic surface, which we call an “anchor”, such that

                                                                                        BΣ “ γ 1 ´ φpγq.                                          (1.1)

                                           We require that Σ is tranverse to BX 1 Y φpBXq.

                                           Remark 1.2. In the examples we consider, γ will be a closed characteristic in BX, i.e. a closed leaf
                                           of the characteristic foliation Kerpω|T BX q, and likewise γ 1 will be a closed characteristic in BX 1 . In
                                           this case the surface Σ is automatically transverse to BX 1 Y φpBXq because it is symplectic.

                                                                                                 1
                                     γ1




                                                                                 Σ




                                                                ϕpX, ωq


                                                                          ϕpγq

                      pX 1 , ω 1 q




   Figure 1: A schematic of an anchored symplectic embedding pϕ, Σq : pX, ω, γq Ñ pX 1 , ω 1 , γ 1 q.


Remark 1.3. A related notion of “symplectic hat” is studied in [12]. There one starts with pX, ω, γq
and looks for an anchored symplectic embedding as above, except that pX 1 , ω 1 q is a closed symplectic
manifold (possibly much larger than pX, ωq), and γ 1 “ H.

Remark 1.4. The definition of “anchored symplectic embedding” also makes sense for symplectic
manifolds of higher dimension. However this case is less interesting, because as pointed out in [12],
there is an h-principle for symplectic submanifolds of codimension greater than two [11, Thm. 12.1.1],
which allows a smooth embedding Σ satisfying (1.1) to be isotoped to a symplectic embedding by
a C 0 -small isotopy under mild hypotheses. See also Remark 1.21 below.

    The goal of this paper is to study quantitative obstructions to anchored symplectic embeddings.
In particular, we will see examples where there does not exist an anchored symplectic embedding
pX, ω, γq Ñ pX 1 , ω 1 , γ 1 q, although there does exist a symplectic embedding pX, ωq Ñ pX 1 , ω 1 q, and
moreover there exists an anchored symplectic embedding pX, ω, γq Ñ pX 1 , rω, γ 1 q if r ą 1 is suffi-
ciently large.

Remark 1.5. There is also an “orthogonal” story giving quantitative obstructions to Lagrangian
cobordisms between Legendrian knots or between transverse knots; see e.g. [9, 10, 34, 35]. In
addition, from the perspective of C 0 symplectic geometry, there have been a few rigidity results
concerning how symplectic homeomorphisms may act on codimension two symplectic submanifolds;
see e.g. [2, 16, 17].



                                                    2
                       (a) Convex                                                  (b) Concave

Figure 2: Regions Ω and Ω1 in R2ě0 determining convex and concave toric domains respectively.
Neither is smooth.


1.2    Toric domains
The examples of symplectic four-manifolds that we will consider are all “toric domains”, which we
now review.

Definition 1.6. Let Ω be a compact region in R2ě0 . Assume that 0 P intpΩq and that BΩ consists
of the line segment from p0, 0q to pa, 0q for some positive real number a “ apΩq, the line segment
from p0, 0q to p0, bq for some positive real number b “ bpΩq, and a continuous curve B` Ω in R2ě0
from pa, 0q to p0, bq which intersects the axes only at its endpoints. We define the toric domain

                                             XΩ “ µ´1 pΩq

where µ : C2 Ñ R2ě0 is given by pz1 , z2 q ÞÑ π|z1 |2 , π|z2 |2 . We equip XΩ with the restriction of the
                                             `                 ˘

standard symplectic form on R4 “ C2 . Note that if the curve B` Ω is smooth, then BXΩ is a smooth
hypersurface in R4 , and in this case we say that XΩ is smooth.
   We say that XΩ is a convex toric domain 1 if the set
                                         p “ tµ P R2 | p|µ1 |, |µ2 |q P Ωu
                                         Ω

is convex. We say that XΩ is a concave toric domain if the set R2ě0 zΩ is convex. See Figure 2.

Example 1.7. The following are some basic examples of convex toric domains.

  (i) If Ω is the triangle with vertices p0, 0q, pr, 0q, and p0, rq, then XΩ is the ball

                                             B 4 prq “ tz P C2 | π|z|2 ď ru.
   1
    A “convex toric domain” is not the same as a toric domain that is convex; see the discussion in [15, §2]. Different
and broader notions of “convex toric domain” are studied in [7, 8].


                                                          3
 (ii) More generally, if Ω is the triangle with vertices p0, 0q, pa, 0q, and p0, bq, then XΩ is the
      ellipsoid
                                                              2   π|z2 |2
                                      "               ˇ                     *
                                                    2 ˇ π|z1 |
                                                      ˇ
                            Epa, bq “ pz1 , z2 q P C ˇ          `         ď1 .
                                                          a         b
 (iii) If Ω is the rectangle r0, as ˆ r0, bs, then XΩ is the polydisk
                                                         ˇ
                              P pa, bq “ pz1 , z2 q P C2 ˇ π|z1 |2 ď a, π|z2 |2 ď b .
                                         ␣                                         (

      Note that the ellipsoid Epa, bq is smooth, but the polydisk P pa, bq is not smooth.
Remark 1.8. If XΩ is a smooth toric domain, then there are two distinguished closed characteristics
in BXΩ , given by the circles pπ|z1 |2 “ apΩq, z2 “ 0q and pz1 “ 0, π|z2 |2 “ bpΩqq. We denote these
closed characteristics by e1,0 “ e1,0 pXΩ q and e0,1 “ e0,1 pXΩ q respectively.
Remark 1.9. If Ω Ă intpΩ1 q, then there is an anchored symplectic embedding
                                    pφ, Σq : pXΩ , e1,0 q ÝÑ pXΩ1 , e1,0 q.
Here φ is given by the inclusion map XΩ Ñ XΩ1 , while Σ is the annulus
                                     ˇ
                              z P C2 ˇ apΩq ď π|z1 |2 ď apΩ1 q, z2 “ 0 .
                            ␣                                         (


1.3    Statement of main results
We now consider examples where the inclusion in Remark 1.9 is optimal.
Theorem 1.10. (proved in §3.2) If a ą 1, then there exists an anchored symplectic embedding
pP pa, 1q, e1,0 q Ñ pB 4 pcq, e1,0 q if and only if c ą a ` 1, or equivalently P pa, 1q Ă intpB 4 pcqq.
Remark 1.11. The statement of Theorem 1.10 makes sense, even though P pa, 1q is not smooth,
because e1,0 pP pa, 1qq is contained in the smooth part of BP pa, 1q.
Remark 1.12. If 1 ď a ď 2, then already the existence of a symplectic embedding P pa, 1q Ñ B 4 pcq
implies2 that c ě a ` 1, by [24, Thm. 1.3]. If a ą 2, then better symplectic embeddings are possible.
In particular, if a ą 2 then one can use symplectic folding [36, §4.4.2] to construct a symplectic
embedding P pa, 1q Ñ P pa{2 ` ϵ, 2 ` ϵq for ϵ ą 0 small. It follows that there exists a symplectic
embedding φ : P pa, 1q Ñ B 4 pcq whenever3 c ą 2 ` a{2.
    The following is a much larger family of examples (which however does not include Theorem 1.10
as a special case).
Theorem 1.13. (proved in §3.2) Let XΩ and XΩ1 be convex toric domains in R4 . Suppose that
                                               apΩq ą bpΩ1 q.                                            (1.2)
If there exists an anchored symplectic embedding
                                         pXΩ , e1,0 q ÝÑ pXΩ1 , e1,0 q                                   (1.3)
then Ω1 Ă Ω2 .
  2
   See also [18, Thm. 1] for a stronger result about Lagrangian embeddings
                                                                    ?
                                                                           which implies this.
  3
   This lower bound on c is known to be optimal when 2 ď a ď 5`3 7 « 2.54 by [5]. When a ą 6, better symplectic
embeddings are possible using multiple symplectic folding [36, §4.3.2].


                                                      4
Remark 1.14. The existence of an anchored symplectic embedding (1.3) forces apΩq ă apΩ1 q, for
the simple reason that the anchor Σ must have positive symplectic area, and this symplectic area
would be apΩ1 q ´ apΩq.

Remark 1.15. There are many examples of pairs of convex toric domains XΩ and XΩ1 for which a
symplectic embedding XΩ Ñ intpXΩ1 q exists, but an anchored symplectic embedding pXΩ , e1,0 q Ñ
pXΩ1 , e1,0 q is obstructed by Theorem 1.13, even though the positive area condition apΩq ă apΩ1 q
holds. For example, one can use symplectic folding as in Remark 1.12 to construct a symplectic
embedding P p8, 2q Ñ Ep11, 7q. However Theorem 1.13 shows that such an embedding cannot be
upgraded to an anchored symplectic embedding pP p8, 2q, e1,0 q Ñ pEp11, 7q, e1,0 q.

   Without the hypothesis (1.2), we can prove a similar result for “2-anchored symplectic embed-
dings”. Suppose that pX, ωq and pX 1 , ω 1 q are compact symplectic four-manifolds with boundary Y
and Y 1 . Let γ1 and γ2 be disjoint knots in Y , and let γ11 and γ11 be disjoint knots in Y 1 .

Definition 1.16. A 2-anchored symplectic embedding

                                    pX, ω, γ1 , γ2 q ÝÑ pX 1 , ω 1 , γ11 , γ21 q

is a triple pφ, Σ1 , Σ2 q, where
                                       φ : pX, ωq ÝÑ pintpX 1 q, ω 1 q
is a symplectic embedding, and
                                          Σ1 , Σ2 Ă X 1 zφpintpXqq
are disjoint smoothly embedded symplectic surfaces such that

                                              BΣi “ γi1 ´ φpγi q.

We also require that Σi is transverse to BX 1 Y φpBXq.

Theorem 1.17. (proved in §3.2) Let XΩ and XΩ1 be convex toric domains in R4 . If there exists a
2-anchored symplectic embedding

                                   pXΩ , e1,0 , e0,1 q ÝÑ pXΩ1 , e1,0 , e0,1 q,

then Ω Ă Ω1 .

    We also have an analogous result for concave toric domains:

Theorem 1.18. (proved in §3.3) Let XΩ and XΩ1 be concave toric domains in R4 . If there exists
a 2-anchored symplectic embedding

                                   pXΩ , e1,0 , e0,1 q ÝÑ pXΩ1 , e1,0 , e0,1 q,

then Ω Ă Ω1 .




                                                         5
1.4   More about anchors
To get a better understanding of the difference between a symplectic embedding and an anchored
symplectic embedding, fix a symplectic embedding φ : pX, ωq Ñ pintpX 1 q, ω 1 q and knots γ Ă BX
and γ 1 Ă BX 1 . When can the symplectic embedding φ be upgraded to an anchored symplectic
embedding? That is, when does there exist an embedded symplectic surface Σ Ă W “ X 1 zφpintpXq
satisfying the requirements in Definition 1.1?
    There are three basic necessary conditions. To describe them, let H2 pW, γ 1 , φpγqq denote the set
of 2-chains Z in W with BX “ γ 1 ´φpγq, modulo boundaries of 3-chains. This is an affine space over
H2 pW q. An anchor Σ as above determinesş 1 a “relative homology class” Z “ rΣs P H2 pW, γ , φpγqq.
                                                                                               1

Since Σ is symplectic, we must have Z ω ą 0. In addition, the relative adjunction formula (see
e.g. [20, §4.4]) determines the genus of Σ in terms of the relative homology class Z, and this genus
must be nonnegative. And, of course, there must exist a smoothly embedded surface in the relative
homology class Z of the correct genus.
    However, the existence of a relative homology class Z satisfying these conditions is not sufficient,
as shown by the following simple example.

Theorem 1.19. (proved in §3.2) Let XΩ , XΩ1 Ă R4 be convex toric domains with Ω Ă Ω1 . Then
the inclusion map ı : XΩ Ñ XΩ1 can be upgraded to an anchored symplectic embedding

                                   pı, Σq : pXΩ , e1,0 q ÝÑ pXΩ1 , e0,1 q,

if and only if
                                            bpΩ1 q ą x0 ` y0 ,                                      (1.4)
where px0 , y0 q P BΩ is a point where the tangent line to BΩ has slope ´1.

Remark 1.20. The “if” part of Theorem 1.19 is proved as follows. Let η be a smooth path in
Ω1 z intpΩq from p0, bpΩ1 qq to papΩq, 0q. The path η lifts to an embedded cylinder Σ Ă XΩ1 z intpXΩ q
with BΣ “ e0,1 pΩ1 q ´ e1,0 pΩq, such that for each point px, yq in the interior of η, the intersection of
Σ with the 2-torus µ´1 px, yq is a geodesic in the homology class p1, 1q. The cylinder Σ is symplectic
if and only if the function x ` y is strictly decreasing along the path η. The existence of a path η
as above along which the function x ` y is strictly decreasing is equivalent to the inequality (1.4).
See Figure 3.

Remark 1.21. In Remark 1.20, suppose that x ` y is not strictly decreasing along the path
η, so that the cylinder Σ is not symplectic. If we assume that bpΩ1 q ą apΩq (this is a weaker
condition than the inequality (1.4)), then one can use h-principle arguments to show that Σ has
a C 0 -small regular homotopy rel boundary to an immersed symplectic cylinder in XΩ1 z intpXΩ q.
However Theorem 1.19 tells us that if condition (1.4) does not hold, then the self-intersections of
this cylinder cannot be cancelled within XΩ1 zXΩ while keeping the cylinder symplectic (although
there does exist an embedded symplectic cylinder in XΩ1 ).

1.5   Idea of the proofs
We will prove the main results in §3, after assembling necessary background in §2. The basic principle
is as follows. Let pX, ωq and pX 1 , ω 1 q be smooth star-shaped domains in R4 . Given a symplectic


                                                     6
Figure 3: An example of regions Ω and Ω1 (in blue and red, respectively) illustrating Remark 1.20.
The tangent line relevant to (1.4) is the dotted black line, and evidently intersects the y-axis below
bpΩ1 q; a possible path η of slope less than ´1 is in green.


embedding φ : pX, ωq Ñ pintpX 1 q, ω 1 q, we obtain a symplectic cobordism W “ X 1 zφpintpXqq. There
is an associated cobordism map on embedded contact homology (ECH),

                              Φ “ ΦpW q : ECHpBX 1 q ÝÑ ECHpBXq,                                 (1.5)

defined in [32], which in this case is an isomorphism. For a suitable almost complex structure J on
the “symplectic completion” W of W , one can find a chain map ϕ inducing Φ, such that whenever
a coefficient of ϕ is nonzero, there is a corresponding “broken J-holomorphic current” in W . Since
any holomorphic curve has positive symplectic area, the existence of such a broken holomorphic
current leads to an inequality involving the symplectic actions of the Reeb orbits to which its ends
are asymptotic.
    Now given an anchor Σ, after straightening Σ near the boundary and “completing” it to an
appropriate surface Σ, we can choose J such that Σ is holomorphic. Any other holomorphic curve
in W must have positive intersections with Σ. This leads to restrictions on which components of the
chain map ϕ can be nonzero. (Compare [25, Rem. 7.1].) These restrictions sharpen the inequalities
described in the previous paragraph, leading to the proofs of the main theorems.
    We remark that instead of using ECH cobordism maps, we could instead find the holomorphic
curves in W that we need using more “elementary” arguments; see [26, Rem. 11].

Acknowledgments. M.H. was partially supported by NSF grant DMS-2005437. A.R. was par-
tially supported by NSF grants DMS-2203312 and DMS-1907654. M.W. was partially supported by
NSF grant DMS-2103245. Y.Y. was partially supported by ERC Starting Grant No. 851701. A.R.
and Y.Y. are grateful to the Kylerec 2022 workshop. M.W. and Y.Y. also thank the 2023 Cornell
Topology Festival. We further thank Ko Honda, Cagatay Kutluhan, Steven Sivek, John Etnyre,
and Jen Hom for helpful conversations.




                                                  7
2        ECH of perturbations of convex and concave toric domains
In this section we prepare for the proofs of the main theorems. In §2.1 we review what we need to
know about embedded contact homology (ECH). In §2.2 and 2.3 we combinatorially describe the
ECH chain complex (not including the differential) for certain “nice” perturbations of convex and
concave toric domains, along with additional information about the chain complex generators such
as symplectic action and linking numbers. Finally, in §2.4 and §2.5 we introduce what we need to
know about ECH cobordism maps.

2.1        Embedded contact homology
Let Y be a closed oriented three-manifold (not necessarily connected). Let λ be a contact form on
Y , and assume that λ is nondegenerate (see below). To simplify the discussion4 , assume that

                                                 H1 pY q “ H2 pY q “ 0.                           (2.1)

(In the cases relevant to the proofs of the main results, Y will be the boundary of a smooth toric
domain and thus diffeomorphic to S 3 .) We now review how to define the embedded contact homology
of pY, λq with Z{2 coefficients5 , which we denote by ECH˚ pY, λq. This is a Z-graded Z{2-module
(the definition of the Z-grading will use the homological assumption (2.1)).

Contact geometry. The contact form λ determines the contact structure ξ “ ker λ, as well as
the Reeb vector field R characterized by

                                             dλpR, ¨q “ 0,           λpRq “ 1.

A Reeb orbit is a smooth map

                                    γ : R{T Z ÝÑ Y        with        γ 1 ptq “ Rpγptqq.          (2.2)

We declare two Reeb orbits to be equivalent if they differ by reparametrization of the domain. We
say that the Reeb orbit γ is simple if the map γ is an embedding.
    Let tψt : Y öutPR denote the flow of the Reeb vector field R. Let γ be a Reeb orbit as above.
The derivative of the time t Reeb flow restricts to a symplectic linear map

                                           dψt : pξγp0q , dλq ÝÑ pξγptq , dλq,                    (2.3)

and dψT is called the linearized return map of γ. We say that γ is nondegenerate if 1 is not
an eigenvalue of dψT . In this case we say that γ is elliptic if the eigenvalues of dψT are on the
unit circle, positive hyperbolic if the eigenvalues of dψT are positive, and negative hyperbolic if the
eigenvalues of dψT are negative. We say that the contact form λ is nondegenerate if all Reeb orbits
are nondegenerate, and we assume below that this is the case.
    If γ is a Reeb orbit γ, its symplectic action Apγq is the integral of the contact form
                                                       ż
                                              Apγq “ λ,
                                                                 γ

or equivalently the period T in (2.2).
    4
        See [23] for the definition of ECH without the homological assumption (2.1).
    5
        It is also possible to define ECH with Z coefficients [31, §9].


                                                             8
The chain module.

Definition 2.1. An orbit set is a finite set of pairs α “ tpαi , mi qu, where the αi are distinct simple
Reeb orbits and mi P Zą0 . An ECH generator is an orbit set as above such that mi “ 1 whenever
αi is hyperbolic.

   We sometimes use the multiplicative notation
                                    ź m
                                       αi i ÐÑ tpαi , mi qu.
                                         i

We define ECC˚ pY, λq to be the free Z{2-module generated by the ECH generators.
   The module ECC˚ pY, λq has a Z-grading by the ECH index, which is defined as follows.

Definition 2.2. If α is an ECH generator (or more generally an orbit set), its ECH index is defined
by
                              Ipαq “ cτ pαq ` Qτ pαq ` CZIτ pαq P Z
where

   • τ is a symplectic trivialization of ξ over each of the simple orbits αi in α.

   • cτ is the relative first Chern number, defined as follows. For each i, since H1 pY q “ 0, we can
     choose a surface Σi in Y with BΣi “ αi . Then cτ pαi q “ c1 pξ|Σi , τ q, where the right hand
     side is the signed count of zeroes of a generic section of ξ|Σi which over αi “ BΣi is constant
     and nonvanishing with respect to τ . This count does not depend on the choice of Σi since
     H2 pY q “ 0. Finally,                       ÿ
                                        cτ pαq “    mi cτ pαi q P Z.                             (2.4)
                                                    i

   • Qτ is the relative intersection pairing defined by
                                       ÿ                 ÿ
                             Qτ pαq “     m2i Qτ pαi q `   mi mj ℓpαi , αj q P Z.                  (2.5)
                                         i                i‰j

      Here if i ‰ j then ℓpαi , αj q denotes the linking number of αi with αj , while Qτ pαi q denotes
      the linking number of αi with a pushoff of αi via the framing τ .

   • We define
                                                     mi
                                                    ÿÿ
                                       CZIτ pαq “           CZτ pαik q P Z.
                                                    i k“1

      Here αik denotes the k th iterate of αi , and CZτ denotes the Conley-Zehnder index; see e.g. the
      review in [23, §3.2].

    The ECH index does not depend on the choice of τ , even though the individual terms in it do;
see e.g. [20, §2.8].




                                                    9
The differential.
Definition 2.3. An almost complex structure J on R ˆ Y is λ-compatible if:
       • JBs “ R, where s denotes the R coordinate on R ˆ Y ;

       • J maps the contact structure ξ to itself, rotating positively with respect to dλ; and

       • J is invariant under translation of the R factor on R ˆ Y .
       Fix a λ-compatible almost complex structure J. We consider J-holomorphic curves

                                            u : pC, jq ÝÑ pR ˆ Y, Jq

where the domain is a punctured compact Riemann surface. We assume that for each puncture,
there exists a Reeb orbit γ, such that in a neighborhood of the puncture, u is asymptotic to R ˆ γ
as either s Ñ `8 (in which case we say that this is a “positive puncture”) or s Ñ ´8 (a “negative
puncture”). If u is somewhere injective, then u is determined by its image in R ˆ Y , which by abuse
of notation we still denote by C.
Definition 2.4. A J-holomorphic current is a finite sum C “ k dk Ck where the Ck are distinct
                                                                ř
somewhere injective J-holomorphic curves as above, and the dk are positive integers. If α and β are
orbit sets, we define MJ pα, βq to be the set of J-holomorphic currents C such that limsÑ`8 pC X
ptsu ˆ Y qq “ α and limsÑ´8 pC X ptsu ˆ Y qq “ β as currents6 .
    Since J is R-invariant, R acts on MJ pα, βq by translation of the R coordinate on R ˆ Y . By
[23, Prop. 3.7], if J is generic and Ipαq ´ Ipβq “ 1, then MJ pα, βq{R is a finite set; moreover, for
a given ECH generator α, there are only finitely many ECH generators β with Ipαq ´ Ipβq “ 1 for
which MJ pα, βq is nonempty. We can then define the differential

                                     BJ : ECC˚ pY, λq ÝÑ ECC˚´1 pY, λq

as follows. If α is an ECH generator, then
                                                  ÿ               MJ pα, βq
                                     BJ α “                 #Z2             ¨ β.
                                                                     R
                                              Ipαq´Ipβq“1

Here β is an ECH generator, and #Z2 denotes the mod 2 count.
    It is shown in [30] that BJ2 “ 0. We denote the homology of the chain complex pECC˚ pY, λq, BJ q
by ECH˚ pY, λq or ECH˚ pY, ξq. It is shown in [39] that this homology is canonically isomorphic to
a version of Seiberg-Witten Floer cohomology depending only on Y and ξ and not on λ or J.

ECH of S 3 . As explained in [23, §3.7], we have
                                               #
                                                            Z2      if ˚ “ 0, 2, 4, . . .
                                ECH˚ pS 3 , ξstd , Jq “
                                                            0       else.

Here ξstd denotes the standard tight contact structure on S 3 .
   6
   This means, for example, that if the pair pαi , mi q appears in α, then  ř of the curves Ck have positive punctures
                                                                       ř some
asymptotic to covers of αi , say of multiplicities qi,k,l , and we have k dk l qi,k,l “ mi .


                                                          10
Action filtration. Let α “ tαi , mi u denote an orbit set. We define its symplectic action to be
                                        ÿ    ż        ÿ
                             Apαq “       mi     λ“      mi Apγq.
                                              i      αi         i

Given L P R, let ECC˚L pY, λq denote the subset of ECCpY, λq spanned by ECH generators α with
Apαq ă L. If J is a λ-compatible almost complex structure, then it follows from the second bullet
in Definition 2.3 that if MJ pα, βq ‰ H then Apαq ě Apβq. Consequently pECC˚L pY, λq, BJ q is a
subcomplex of pECC˚ pY, λq, BJ q. The homology of this subcomplex is the filtered ECH which we
denote by ECH˚L pY, λq. It is shown in [32, Thm. 1.3] that filtered ECH does not depend on J
(although it does depend on λ), and furthermore the maps
                                                                    1
                                     ECH L pY, λq ÝÑ ECH L pY, λq

for L ă L1 and
                                     ECH L pY, λq ÝÑ ECHpY, ξq
induced by inclusion of chain complexes are also independent of J.

J0 index There is an important variant of the ECH index I, denoted by J0 (not an almost complex
structure). If α “ tpαi , mi qu is an ECH generator, we define
                                                        ÿ
                               J0 pαq “ Ipαq ´ 2cτ pαq ´ CZτ pαimi q P Z.
                                                            i

According to [24, Prop. 3.2], the J0 index bounds topological complexity of holomorphic curves as
follows. Let J be a λ-compatible almost complex structure, let α “ tpαi , mi qu and β “ tpβj , nj qu
be ECH generators, and let C P MJ pα, βq be somewhere injective with connected domain. Then
                                 ÿ`          ˘ ÿ´ ´          ¯
                   2gpCq ´ 2 `       2n`
                                       i ´ 1  `     2n j ´ 1   ď J0 pαq ´ J0 pβq.             (2.6)
                                 i                    j

Here gpCq denotes the genus of C, while n` i denotes the number of positive punctures of C asymp-
totic to covers of αi , and n´
                             j denotes the number of negative punctures of C asymptotic to covers
of βj .

2.2   Reeb dynamics on the boundary of a toric domain
We now discuss the Reeb dynamics on the boundary of a toric domain. The following is a con-
solidation and review of material from [4, §3.2] which discusses concave toric domains and [24, §5]
which discusses convex toric domains, with updated notational conventions from [14].
    Let XΩ Ă R4 be a smooth toric domain as in Definition 1.6, and assume that B` Ω is transverse
to the radial vector field on R2 (which holds for example for convex and concave toric domains).
Then BXΩ is a star-shaped hypersurface in R4 . As such, the standard Liouville form
                                                 2
                                              1ÿ
                                       λ0 “         pxi dyi ´ yi dxi q                         (2.7)
                                              2 i“1

restricts to a contact form on BXΩ .

                                                     11
   As in Remark 1.8, there are two distinguished Reeb orbits e1,0 and e0,1 in BXΩ , where z2 “ 0
and z1 “ 0 respectively. We now discuss the Reeb dynamics on the rest of BXΩ where z1 , z2 ‰ 0.
Here we use coordinates pr1 , θ1 , r2 , θ2 q where z1 “ r1 eiθ1 and z2 “ r2 eiθ2 .
   Let px, yq P BΩ with x, y ą 0. Let pa, bq be an outward normal vector to BΩ at px, yq. On the
two-torus µ´1 px, yq, the Reeb vector field is given by
                                                     ˆ               ˙
                                               2π         B       B
                                      R“               a      `b       .                    (2.8)
                                             ax ` by     Bθ1     Bθ2

(See [14, §2.2] for more general computations.) Using (2.8) we obtain the following information
about the Reeb orbits in BXΩ .

Simple orbits. If a{b P Q Y t8u, then the torus µ´1 px, yq is foliated by Reeb orbits (and all simple
    Reeb orbits in BXΩ other than e1,0 and e0,1 arise this way). In this case, our convention is to
    rescale the normal vector pa, bq so that a, b are relatively prime nonnegative integers.

Symplectic action. In the above situation, if γ is a simple Reeb orbit in µ´1 px, yq, then its
   symplectic action is
                                        Apγq “ ax ` by.                                  (2.9)
      We also have
                                    Ape1,0 q “ apΩq,        Ape0,1 q “ bpΩq.                       (2.10)

Linking numbers. The linking numbers of simple Reeb orbits in BXΩ » S 3 are given as follows.
     First, e1,0 and e0,1 form a Hopf link, and in particular

                                               ℓpe1,0 , e0,1 q “ 1.                                (2.11)

      Second, if γ is a simple Reeb orbit distinct from e1,0 and e0,1 , and if pa, bq is the associated
      integer vector as above, then as one traverses γ, the coordinate θ1 winds a times and the
      coordinate θ2 winds b times. Consequently

                                                ℓpγ, e1,0 q “ b,
                                                                                                   (2.12)
                                                ℓpγ, e0,1 q “ a,

      Finally, let γ 1 be a simple Reeb orbit distinct from e1,0 , e0,1 , and γ, and let pa1 , b1 q be the
      associated integer vector as above. Orient the curve B` Ω from pa, 0q to p0, bq, and suppose
      that µpγq precedes µpγ 1 q along this curve or that µpγq “ µpγ 1 q. Then we can homotope γ to
                          1
      ea1,0 and γ 1 to eb0,1 without crossing, so

                                                ℓpγ, γ 1 q “ ab1 .                                 (2.13)

Relative first Chern number. We can find a section s of ξ over BXΩ such that s´1 p0q “ e1,0 Y
     e0,1 , and s takes values in the Kerpdµq. The section s determines a trivialization τ of ξ over all
     simple Reeb orbits other than e1,0 and e0,1 . If γ is such a Reeb orbit with associated integer
     vector pa, bq, then it follows from (2.12) after an orientation check that

                                                cτ pγq “ a ` b.                                    (2.14)


                                                   12
   To describe the ECH of BXΩ at the chain level, we need to perturb XΩ so that the contact form
on the boundary becomes nondegenerate. We now describe a “nice” way to do so for convex toric
domains.
   Suppose that XΩ Ă R4 is a convex toric domain. If pa, bq P Z2ě0 , define

                                }pa, bq}˚Ω “ maxtax ` by | px, yq P Ωu.                             (2.15)

Note that the maximum in (2.15) is realized by a point px, yq P B` Ω where pa, bq is an outward
normal vector.

Lemma 2.5. Let XΩ Ă R4 be a convex toric domain and let L ą maxpapΩq, bpΩqq. Then there
exists a smooth star-shaped domain X Ă R4 with the following properties:

   • The C 0 distance between BXΩ and BX is at most L´1 .

   • The contact form λ0 |BX is nondegenerate.

   • The simple Reeb orbits in BX with symplectic action less than L consist of, for each pair of
     relatively prime nonnegative integers pa, bq with }pa, bq}˚Ω ă L, an elliptic simple Reeb orbit
     ea,b , and a positive hyperbolic simple Reeb orbit ha,b when a, b ą 0.

   • We can arrange that either X Ă XΩ or XΩ Ă X, and that this inclusion upgrades to a
     2-anchored symplectic embedding pX, e1,0 , e0,1 q Ñ pXΩ , e1,0 , e0,1 q or vice versa.

   • If a, b are relatively prime nonnegative integers with }pa, bq}˚Ω ă L, and if γa,b denotes either
     ea,b or ha,b when a, b ą 0, then

                                         |Apγa,b q ´ }pa, bq}˚Ω | ă L´1 .                           (2.16)

   • If pa1 , b1 q is another pair of relatively prime nonnegative integers with }pa1 , b1 q}˚Ω ă L, and if
     γa1 ,b1 is distinct from γa,b , then the linking number in BX » S 3 is

                                        ℓpγa,b , γa1 ,b1 q “ maxpab1 , a1 bq.                       (2.17)

   • There is a trivialization τ of ξ over all of the simple Reeb orbits with symplectic action less
     than L such that

                                     cτ pγa,b q “ a ` b,                                            (2.18)
                                    Qτ pγa,b q “ ab,                                                (2.19)
                                   CZτ pem
                                         a,b q   “ 1,        if   m}pa, bq}˚Ω   ă L,                (2.20)
                                   CZτ pha,b q “ 0.                                                 (2.21)

Proof of Lemma 2.5. To start, by a C 0 -small perturbation of B` Ω, we can arrange that (i) B` Ω is
smooth; (ii) B` Ω is strictly convex; and (iii) B` Ω is nearly perpendicular to the axes. More precisely,
there is a small irrational ϵ ą 0 such that where B` Ω meets the y axis, its slope is ´ϵ, and where
B` Ω meets the x axis, its slope is ´ϵ´1 . In particular, for each pair of relatively prime nonnegative
integers pa, bq with }pa, bq}˚Ω ă L, there is a unique point px, yq P B` Ω at which pa, bq is an outward
normal vector to B` Ω.

                                                        13
    Given px, yq as above, since B` Ω is strictly convex, the circle of Reeb orbits in µ´1 px, yq is
Morse-Bott. Similarly7 to [28, §3.1] (see [1, §3.1] for a more general situation), the contact form can
be perturbed in a neighborhood of µ´1 px, yq (which corresponds to perturbing XΩ ), so that this
circle of Reeb orbits is reduced to two nondegenerate simple Reeb orbits: an elliptic orbit ea,b , for
which the linearized return map is a slight positive rotation, and a positive hyperbolic orbit ha,b .
The perturbation may also create new Reeb orbits of action greater than L. We can further perturb
the contact form to arrange that the Reeb orbits with action greater than L are nondegenerate8 .
    The above implies the first four bullet points in the lemma. The action estimate (2.16) now
follows from (2.9) and (2.10). The linking number formula (2.17) follows from (2.11), (2.12), and
(2.13).
    To prove the last bullet point, over the simple Reeb orbits ea,b and ha,b with a, b ą 0, choose the
trivialization τ as in (2.14). Then for a, b ą 0, equation (2.18) follows from (2.14), equation (2.19)
follows similarly to (2.17), and equations (2.20) and (2.21) follow from the definition of the Conley-
Zehner index in e.g. [23, §3.2]. The trivialization τ has an extension over e1,0 and e0,1 satisfying
equations (2.18), (2.19), and (2.20), as in [23, §3.7].
    A slight modification of the above lemma holds for concave toric domains. To state it, if XΩ Ă R4
is a concave toric domain, and if pa, bq P Z2ě0 , define
                                  rpa, bqsΩ “ mintax ` by | px, yq P B` Ωu.
Lemma 2.6. Let XΩ be a concave toric domain and let L ą maxpapΩq, bpΩqq. Then there exists a
smooth star-shaped domain X Ă R4 such that:
    • The first five bullet points in Lemma 2.5 hold, with } ¨ }˚Ω replaced by r¨sΩ ă L.
    • If pa, bq and pa1 , b1 q are pairs of relatively prime nonnegative integers with rpa, bqsΩ , rpa1 , b1 qsΩ ă
      L, and if γa1 ,b1 is distinct from γa,b , then the linking number in BX » S 3 is
                                            ℓpγa,b , γa1 ,b1 q “ minpab1 , a1 bq.

    • There is a trivialization τ of ξ over all of the simple Reeb orbits with symplectic action less
      than L such that
                                 cτ pγa,b q “ a ` b,
                                   Qτ pγa,b q “ ab,
                                  CZτ pem
                                        a,b q “ ´1,         if   m}pa, bq}˚Ω ă L    and   a, b ą 0,
                                  CZτ pha,b q “ 0,
                      CZτ pe1,0 q, CZτ pe0,1 q ą L.

Proof. To start, by a C 0 -small perturbation of B` Ω, we can arrange that (i) B` Ω is smooth; (ii)
B` Ω is strictly concave; and (iii) B` Ω is nearly tangent to the axes. More precisely, there is a small
irrational ϵ ą 0 such that where B` Ω meets the y axis, its slope is ´ϵ´1 , and where B` Ω meets the
x axis, its slope is ´ϵ. The rest of the argument follows the proof of Lemma 2.5.
Remark 2.7. We are using a different notational convention from [4, 24]; a Reeb orbit ea,b or ha,b
here corresponds to eb,a or hb,a in those references.
   7
     The picture in [28, §3.1] corresponds to a concave toric domain, and for our convex case the directions of the
arrows should be reversed.
   8
     This is not actually necessary to define the filtered embedded contact homology ECH L .


                                                         14
2.3     Combinatorial ECH generators for convex and concave toric domains
We now review how to combinatorially describe the ECH generators, their ECH indices, and their
approximate symplectic actions, for “nice” perturbations of convex and concave toric domains. This
is based on [24] and [4] with some minor notational changes.

2.3.1    Convex toric domains
Definition 2.8. [24, Def. 1.9] A convex integral path is a path Λ in the plane such that:

   • The endpoints of Λ are p0, ypΛqq and pxpΛq, 0q where xpΛq and ypΛq are nonnegative integers.

   • Λ is the graph of a piecewise linear concave function f : r0, xpΛqs Ñ r0, ypΛqs with f 1 p0q ď 0,
     possibly together with a vertical line segment at the right.

   • The vertices of Λ (the points at which its slope changes, and the endpoints) are lattice points.

Notation 2.9. If v is an edge of a convex integral path (a line segment between consecutive vertices),
then the vector from the upper left endpoint to the lower right endpoint of v has the form pb, ´aq
where a, b are nonnegative integers. Write v K “ pa, bq, and define the multiplicity of v, which we
denote by mpvq, to be the greatest common divisor of a and b.

Definition 2.10. If XΩ is a convex toric domain and Λ is a convex integral path, define the Ω-action
of Λ to be                                            ÿ      › K ›˚
                               AXΩ pΛq “ AΩ pΛq “            ›v › .
                                                                  Ω
                                                       vPEdgespΛq

Definition 2.11. [24, Def. 1.10] A convex generator is a convex integral path Λ, together with a
labeling of each edge by ‘e’ or ‘h’ (we omit the labeling from the notation). Horizontal and vertical
edges are required to be labeled ‘e’.

Notation 2.12. If Λ is a convex generator, let hpΛq denote the number of edges that are labeled
‘h’. Let epΛq denote the number of edges that are labeled ‘e’, or that are labeled ‘h’ and have
multiplicity greater than one.

Definition 2.13. [24, Def. 1.11] If Λ is a convex generator, define the combinatorial ECH index of
Λ by                                         ´         ¯
                                   IpΛq
                                   p     “ 2 LpΛq
                                               p   ´ 1 ´ hpΛq.                              (2.22)

Here LpΛq
      p     denotes the number of lattice points in the polygon bounded by Λ, the line segment
from p0, 0q to pxpΛq, 0q, and the line segment from p0, 0q to p0, ypΛqq, including lattice points on the
boundary. Also, define the combinatorial J0 index of Λ by

                               Jp0 pΛq “ IpΛq
                                         p    ´ 2xpΛq ´ 2ypΛq ´ epΛq.                            (2.23)

Lemma 2.14. (cf. [24, Lem. 5.4]) Let XΩ Ă R4 be a convex toric domain, and let L ą maxpapΩq, bpΩqq.
Then a perturbation X of XΩ as in Lemma 2.5 can be chosen so that there is a bijection
                    "                     *     "                             *
                      convex generators Λ          ECH generators α in BX
                 ı:                         ÝÑ                                           (2.24)
                        with AΩ pΛq ă L                 with Apαq ă L

                                                  15
such that if ıpΛq “ α, then

                                        |Apαq ´ AΩ pΛq| ă L´1 ,                                      (2.25)
                                                  Ipαq “ IpΛq,
                                                          p                                          (2.26)
                                                   J0 pαq “ Jp0 pΛq.                                 (2.27)

Proof. The bijection ı is defined as follows. If Λ is a convex generator, then ıpΛq is the product over
the edges of Λ of the following contributions. Let v be an edge of Λ and write v K “ pma, mbq where
a, b ě 0 are relatively prime and m is the multiplicity of v. If v is labeled ‘e’, then the contribution is
 a,b . If v is labeled ‘h’, then the contribution is ea,b ha,b . It follows from (2.16) that, possibly after
                                                      m´1
em
choosing inputting a larger value of L to Lemma 2.5, ı is a well-defined bijection (2.24) satisfying
(2.25). The formulas (2.26) and (2.27) for I and J0 follow from equations (2.17)–(2.21) as in [24,
§5.3, Step 4].

Remark 2.15. Under the bijection (2.24), the total number of simple Reeb orbits that appear in
α equals epΛq ` hpΛq.

2.3.2   Concave toric domains
A variant of the above story holds for concave toric domains.

Definition 2.16. A concave integral path is a path Λ in the plane such that:

   • The endpoints of Λ are p0, ypΛqq and pxpΛq, 0q where xpΛq and ypΛq are positive integers.

   • Λ is the graph of a piecewise linear convex function f : r0, xpΛqs Ñ r0, ypΛqs with f 1 p0q ă 0
     and f pxpΛqq “ 0.

   • The vertices of Λ are lattice points.

Definition 2.17. If XΩ is a concave toric domain and Λ is a concave integral path, define the
Ω-action of Λ to be                          ÿ     “ K‰
                                 AΩ pΛq “           v Ω.
                                                  vPEdgespΛq

Definition 2.18. A concave generator is a concave integral path Λ, together with a labeling of
each edge by ‘e’ or ‘h’ (we omit the labeling from the notation). We define hpΛq and epΛq as before.

Definition 2.19. If Λ is a concave generator, define the combinatorial ECH index of Λ by
                                           ´          ¯
                                  IpΛq “ 2 LpΛq ´ 1 ` hpΛq.
                                  q          q                                           (2.28)

Here LpΛq
      q     denotes the number of lattice points in the polygon bounded by Λ, the line segment
from p0, 0q to pxpΛq, 0q, and the line segment from p0, 0q to p0, ypΛqq, including lattice points on the
boundary, except not including lattice points on Λ itself. Also, define the combinatorial J0 index of
Λ by
                                Jq0 pΛq “ IpΛq
                                          q    ´ 2xpΛq ´ 2ypΛq ` epΛq.                            (2.29)

   The following is a special case of [4, Lem. 3.3].

                                                    16
Lemma 2.20. Let XΩ Ă R4 be a concave toric domain, and let L ą maxpapΩq, bpΩqq. Then a
perturbation X of XΩ as in Lemma 2.6 can be chosen so that there is a bijection
               "                           *    "                               *
                 concave generators Λ with        ECH generators α in BX with
            ı:                               ÝÑ
                 AΩ pΛq ă L and IpΛq
                                 q    ăL             Apαq ă L and Ipαq ă L

such that if ıpΛq “ α, then

                                         |Apαq ´ AΩ pΛq| ă L´1 ,
                                                   Ipαq “ IpΛq,
                                                           q
                                                     J0 pαq “ Jq0 pΛq.                                   (2.30)

Proof. This follows from Lemma 2.6, similarly to the proof of Lemma 2.14.

Definition 2.21. If XΩ Ă R4 is a convex or concave toric domain, we say that a star-shaped
domain X provided by Lemma 2.14 or Lemma 2.20 respectively is an L-nice perturbation of XΩ .

Remark 2.22. There is also a combinatorial formula for the ECH differential BJ for suitable J on
the ECH generators as described above for L-nice perturbations of convex and concave toric domains,
similar to the differential for the ECH of T 3 [29] or the PFH of a Dehn twist [28] respectively. In
principle this is proved in [3], although certain details are not fully explained. The formula in the
convex case is stated in [24, Conj. A.3], and more details in the concave case are provided in [40].
Morse-Bott theory needed for this is worked out in [42, 43].

2.4     Cobordism maps on ECH
We now review cobordism maps on embedded contact homology and some of their properties in the
special case that we need.

Definition 2.23. Let pY` , λ` q and pY´ , λ´ q be contact three-manifolds. A strong symplectic
cobordism from9 pY` , λ` q to pY´ , λ´ q is a compact symplectic four-manifold pW, ωq such that
BW “ Y` ´ Y´ and ω|Y˘ “ dλ˘ .

     Given a cobordism as above, one can find a neighborhood N´ of Y´ in W , identified with
r0, ϵq ˆ Y´ for some ϵ ą 0, in which ω “ es λ´ , where s denotes the r0, ϵq coordinate. Likewise,
one can choose a neighborhood N` » p´ϵ, 0s ˆ Y` of Y` in W in which ω “ es λ` . Fix a choice
of neighborhoods N´ and N` . Using the neighborhood identifications, we can glue to form the
symplectic completion

                             W “ pp´8, 0s ˆ Y´ q YY´ W YY` pr0, 8q ˆ Y` q.

Definition 2.24. An almost complex structure J on W is cobordism-admissible if:

      • On W , the almost complex structure J is ω-compatible.

      • On p´8, 0s ˆ Y´ and r0, 8q ˆ Y` , the almost complex structure J agrees with the restrictions
        of λ˘ -compatible almost complex structures J˘ on R ˆ Y˘ .
  9
    This usage of the words “from” and “to” is natural from the perspective of symplectic geometry, but opposite
from most topology literature.


                                                      17
    Assume now that the contact forms λ˘ are nondegenerate. For a cobordism-admissible almost
complex structure J as above, if α` is an orbit set for λ` and α´ is an orbit set for λ´ , then we define
a moduli space MJ pα` , α´ q of J-holomorphic currents in W analogously to the symplectization
case in §2.1.
    More generally, we define a broken J-holomorphic current from α` to α´ to be a tuple pCN´ , . . . , CN` q
where N´ ď 0 ď N` , for which there exist orbit sets α´ “ α´ pN´ q, . . . , α´ p0q in Y´ and orbit sets
α` p0q, . . . , α` pN` q “ α` in Y` , such that:
   • Ci P MJ´ pα´ pi ` 1q, α´ piqq{R for i “ N´ , . . . , ´1.
   • C0 P MJ pα` p0q, α´ p0qq.
   • Ci P MJ` pα` piq, α` pi ´ 1qq{R for i “ 1, . . . , N` .
   • If i ‰ 0, then Ci is not R-invariant.
We denote the set of such broken J-holomorphic currents by MJ pα` , α´ q.
Proposition 2.25. (special case of [24, Thm. 3.5]) Let pW, ωq be a strong symplectic cobordism
from pY` , λ` q to pY´ , λ´ q and assume that the contact forms λ˘ are nondegenerate. Assume also10
that
                                    H1 pY˘ q “ H2 pY˘ q “ H2 pW q “ 0.                       (2.31)
Then for each L P R there is a well-defined cobordism map
                            ΦL pW, ωq : ECH˚L pY` , λ` q ÝÑ ECH˚L pY´ , λ´ q                            (2.32)
with the following properties:
 (a) If L ă L1 , then the diagram
                                                      ΦL pW,ωq
                                  ECH˚L pY` , λ` q ÝÝÝÝÝÑ ECH˚L pY´ , λ´ q
                                       §                       §
                                       §                       §
                                       đ                       đ
                                                           1
                                       1              ΦL pW,ωq           1
                                  ECH˚L pY` , λ` q   ÝÝÝÝÝÝÑ ECH˚L pY´ , λ´ q
      commutes. In particular, we have a well-defined direct limit
                       ΦpW, ωq “ lim ΦL pX, ωq : ECH˚ pY` , λ` q ÝÑ ECH˚ pY´ , λ´ q.                    (2.33)
                                   LÑ8

 (b) If W is diffeomorphic to a product r0, 1s ˆ Y , then the map (2.33) is an isomorphism.
 (c) Let J˘ be generic λ˘ -compatible almost complex structures on R ˆ Y˘ , and let J be any
     cobordism-admissible almost complex structure on W extending J˘ . Then for each L, the
     cobordism map (2.32) is induced by a (noncanonical) chain map
                             ϕ : pECC L pY` , λ` q, BJ` q ÝÑ pECC L pY´ , λ´ q, BJ´ q
      with the following property: If α˘ are ECH generators in Y˘ , and if the coefficient xϕα` , α´ y ‰
      0, then there exists a broken J-holomorphic current pCN´ , . . . , CN` q P MJ pα` , α´ q.
  10
     The homological assumptions (2.31) can be dropped, if one restricts to the subspace of ECH generated by
nullhomologous ECH generators and assumes that the cobordism pW, ωq is “weakly exact”; see [24, §3.10]. In this
case the sense in which the cobordism map respects the grading needs to be stated more carefully.


                                                      18
Remark 2.26. The homological assumptions (2.31) imply that in part (c), if xϕα` , α´ y ‰ 0, or
more generally if MJ pα` , α´ q ‰ H, then Apα` q ě Apα´ q, with equality only if α` “ H.

2.5   Special properties of ECH cobordism maps for toric domains
The construction in [32] of the cobordism map (2.32) does not directly count holomorphic currents,
due to difficulties with multiple covers, but rather uses Seiberg-Witten theory; see [23, §5.5]. This
is why in part (c), we only obtain a broken holomorphic current. However in some special cases,
namely for “L-tame” cobordisms defined in [24, §4.1], we obtain actual holomorphic currents.
    In particular, suppose that XΩ´ , XΩ` Ă R4 are convex or concave toric domains. Suppose
further that XΩ´ is a convex toric domain or XΩ` is a concave toric domain (i.e. we are not in
the case where XΩ´ is a concave toric domain and XΩ` is a convex toric domain, which is studied
in [7]). Suppose there exists a symplectic embedding φ : XΩ´ Ñ intpXΩ` q. By Lemma 2.14
and/or Lemma 2.20, we can find L-nice approximations X´ Ă XΩ´ and X` Ą XΩ` . Let W “
X` zφpintpX´ qq; this is a symplectic cobordism from pBX` , λ` q to pBX´ , λ´ q, where λ˘ is the
restriction to BX˘ of the standard Liouville form (2.7).

Lemma 2.27. In the above situation, if J is a generic cobordism-admissible almost complex struc-
ture on W , then:

 (a) If Apα` q ă L and C P MJ pα` , α´ q is a J-holomorphic current, then Ipα` q ě Ipα´ q.

 (b) In Proposition 2.25(c), the broken J-holomorphic current pCN´ , . . . , CN` q P MJ pα` , α´ q sat-
      isfies N´ “ N` “ 0, so that we have a J-holomorphic current C0 P MJ pα` , α´ q.

Proof. If XΩ´ and XΩ` are both convex toric domains, then W is an “L-tame” cobordism in the
sense of [24, Def. 4.3], as shown in [24, §6]. The same is true when XΩ` is a concave toric domain,
by a similar argument. Assertion (a) now follows from [24, Prop. 4.6(a)]. Assertion (b) follows from
(a) together with the fact that every non-R-invariant J˘ -holomorphic current in R ˆ Y˘ has positive
ECH index, as reviewed in [23, Prop. 3.7].

   We will also need the following lemma regarding linking numbers.

Lemma 2.28. Let XΩ` and XΩ´ be convex toric domains, and let φ : XΩ´ Ñ intpXΩ` q be a
symplectic embedding. Let L, X` , X´ , W be as above. Let J be any cobordism-admissible almost
complex structure on W . Let α` and α´ be convex generators with Apα` q ă L. Suppose there
exists a holomorphic current C P MJ pα` , α´ q.

 (a) If there exists a J-holomorphic curve C1 P MJ pe1,0 , e1,0 q, then ypα` q ě ypα´ q.

 (b) If there exists a J-holomorphic curve C2 P MJ pe0,1 , e0,1 q, then xpα` q ě xpα´ q.

Proof. We follow the proof of [25, Lem. 5.1] with minor modifications.
   We first prove assertion (a). We can assume without loss of generality that C consists of a single
somewhere injective curve C which is distinct from C1 . Let s` ąą 0 and let

                                      η` “ C X pts` u ˆ BX` q.

By standard results on asymptotics of holomorphic curves, see e.g. [38, Cor. 2.5, 2.6], if s` is
sufficiently large then η` is cut out transversely and disjoint from the Reeb orbits in α` . Likewise,

                                                  19
let s´ ăă 0 and let η´ “ C Xpts´ uˆBX´ q; if |s´ | is sufficiently large then η´ is cut out transversely
and disjoint from the Reeb orbits in α´ .
    Now observe that
                            ℓpη` , e1,0 q ´ ℓpη´ , e1,0 q “ #pC X C1 q ě 0.                       (2.34)
Here ‘#’ denotes the algebraic intersection number. The equality on the left holds by the definition
of linking number, and the inequality on the right holds by intersection positivity for J-holomorphic
curves.
     To start to analyze the left hand side of (2.34), we can write
                                                    ž
                                           η` “           ηγ`
                                                  pγ,mqPα`

where ηγ` is a link in a tubular neighborhood of γ which, in this tubular neighborhood, is homologous
to mγ. By the definition of linking number, we have
                                                     ÿ
                                     ℓpη` , e1,0 q “     ℓpηγ` , e1,0 q.
                                                  pγ,mqPα`

If pa, bq ‰ p1, 0q and γ “ ea,b or γ “ ha,b , then it follows from equation (2.17) that

                                           ℓpηγ` , e1,0 q “ mb.

If γ “ e1,0 , then it follows from the winding number bounds from [19, §3], which are reviewed in
our notation in [23, Lem. 5.3(b)], that

                                           ℓpηe`1,0 , e1,0 q ď 0.

Combining the above three lines, we conclude that

                                         ℓpη` , e1,0 q ď ypα` q.                                 (2.35)

    A similar calculation shows that

                                         ℓpη´ , e1,0 q ě ypα´ q.                                 (2.36)

(In fact, if e1,0 appears in α´ , then the inequality (2.36) is strict.) Combining (2.34), (2.35), and
(2.36) completes the proof of (a). Assertion (b) is proved by the same argument.


3     Proofs of the main theorems
We now prove the main results. Theorems 1.10, 1.13, 1.17, and 1.19 are proved in §3.2, and
Theorem 1.18 is proved in §3.3.

3.1   Preliminary lemmas
We begin with a lemma concerning the following geometric setup. Let XΩ´ and XΩ` be convex
toric domains, and suppose there exists a symplectic embedding

                                        φ : XΩ´ ÝÑ intpXΩ` q.

                                                    20
Let
                                  L ą maxpapΩ´ q, apΩ` q, bpΩ´ q, bpΩ` qq
and let X´ Ă XΩ´ and X` Ą XΩ` be L-nice approximations provided by Lemma 2.14. Write
Y˘ “ BX˘ and let λ˘ denote the induced contact form on Y˘ . By Lemma 2.14, ECH generators in
Y´ or Y` with symplectic action less than L can be identified with convex generators with Ω´ -action
or Ω` -action less than L, respectively, via the bijection ı, and we omit ı from the notation. Let W
be the symplectic cobordism from pY` , λ` q to pY´ , λ´ q given by X` zφpintpX´ qq as in §2.4. Let J
be a cobordism-admissible almost complex structure on W as in Definition 2.24.
Lemma 3.1. Let a, b         ě 0 be relatively prime nonnegative integers, not both zero, and assume
that L ą AΩ` pea,b q.      Suppose there exist J-holomorphic curves C1 P MJ pe1,0 , e1,0 q and C2 P
MJ pe0,1 , e0,1 q. Let Λ   be an ECH generator in Y´ with IpΛq
                                                             p     “ Ipe
                                                                      p a,b q. Suppose there exists a
J-holomorphic current            J
                           C P M pea,b , Λq. Then Λ “ ea,b .
Proof. Since ea,b is a simple Reeb orbit, the current C consists of a single somewhere injective curve
C with multiplicity one11 . It then follows from the J0 bound (2.6), equation (2.27), and Remark 2.15
that
                             Jp0 pea,b q ´ Jp0 pΛq ě 2gpCq ´ 1 ` epΛq ` hpΛq.                     (3.1)
Since IpΛq
      p      p a,b q, it follows from equation (2.23) that
           “ Ipe

                         Jp0 pea,b q ´ Jp0 pΛq “ 2pxpΛq ´ a ` ypΛq ´ bq ` epΛq ´ 1.                     (3.2)

By Lemma 2.28, we have
                                                 xpΛq ď a                                               (3.3)
and
                                                 ypΛq ď b.                                              (3.4)
Combining (3.1), (3.2), (3.3), and (3.4), we obtain

                                            2gpCq ` hpΛq ď 0,

with equality only if xpΛq “ a and ypΛq “ b. We conclude that

                           gpCq “ 0,     hpΛq “ 0,        xpΛq “ a,       ypΛq “ b.

Since IpΛq
      p      p a,b q, the index formulas (2.22) and (2.26) imply that
           “ Ipe

                                              LpΛq
                                              p    “ Lpe
                                                     p a,b q.

Since the path underlying Λ is convex and has the same endpoints as the line segment corresponding
to ea,b , it follows that Λ “ ea,b . (We also get that C is a cylinder, although we do not need this.)

Proposition 3.2. Let XΩ be a convex toric domain and let a, b ě 0 be relatively prime nonnegative
integers. Let X be an L-nice perturbation of XΩ where L is large with respect to a, b, and Ω. Let
Y “ BX and let λ denote the induced contact form on Y . Let J´ be a generic λ-compatible almost
complex structure on R ˆ Y . Then ea,b is a cycle in pECC˚L pY, λq, BJ´ q which represents a nonzero
homology class in ECH˚ pY, λq.
  11
    Since the symplectic form on W is exact, every nonconstant J-holomorphic curve in W must have at least one
positive end, as in Remark 2.26.


                                                     21
Proof. We proceed in four steps.
     Step 1: We first prove the proposition in the special case where a, b ą 0 and XΩ is the ellipsoid
Epac, bcq where c ą 0 is a positive real number.
     By [24, Lem. 2.1(a)], ea,b is “minimal” for Epac, bcq in the sense of [24, Def. 1.15]. Minimality
means that ea,b uniquely minimizes Ω-action among all convex generators with the same ECH index
as ea,b and with all edges labeled ‘e’. The proposition in this case now follows from [24, Lem. 5.5].
     Step 2: We now set up the proof of the proposition in the general case where a, b ą 0.
     Let c ą 0 be a positive real number which is sufficiently large that XΩ Ă intpEpac, bcqq. Let
X´ Ă XΩ and X` Ą Epac, bcq be L-nice perturbations of XΩ and Epac, bcq respectively, where L
is large with respect to a, b, c, and Ω. Let Y˘ and W be as in the statement of Lemma 3.1. Let J
be a cobordism-admissible almost complex structure on W which restricts to J´ on p´8, 0s ˆ Y´ ,
and let J` denote the restriction of J to r0, 8q ˆ Y` .
     Define a surface C1 Ă W to be the union of W X pC ˆ t0uq with the “trivial half-cylinders”
p´8, 0s ˆ e1,0 in p´8, 0s ˆ Y´ and r0, 8q ˆ e1,0 in r0, 8q ˆ Y` . Likewise, define C2 Ă W to be the
union of W X pt0u ˆ Cq with the trivial half-cylinders over e0,1 in p´8, 0s ˆ Y´ and r0, 8q ˆ Y` . By
the definition of λ˘ -compatible almost complex structure, C1 and C2 are necessarily J-holomorphic
in p´8, 0s ˆ Y´ and r0, 8q ˆ Y` . We can choose J so that C1 and C2 are J-holomorphic in W as
well.
     Next, perturb J to be generic as needed for Lemma 2.27. By standard automatic transversality
arguments, the holomorphic cylinders C1 and C2 persist under a sufficiently small perturbation.
(See [41] for a general treatment of automatic transversality, and [27, Lem. 4.1] for a simple special
case which is sufficient for the present situation.) We use the same notation J, C1 , and C2 for the
new almost complex structure and holomorphic cylinders.
     Step 3. We now complete the proof of the proposition in the general case where a, b ą 0.
     Let
                          ϕ : pECC˚L pY` , λ` , q, J` q ÝÑ pECC˚L pY´ , λ´ q, J´ q
be a chain map as provided by Proposition 2.25(c). Since ea,b is a cycle representing a nontrivial
homology class in ECHpY` , λ` q by Step 1, it follows from Proposition 2.25(a),(b) that
                                            ϕpea,b q P ECC˚L pY´ , λ´ q
is a cycle representing a nontrivial homology class in ECHpY´ , λ´ q.
    Let Λ be an ECH generator for pY´ , λ´ q, and suppose that xϕpea,b q, Λy ‰ 0. By Lemma 2.27(b),
there exists a J-holomorphic current C P MJ pea,b , Λq. By Lemma 3.1, we have Λ “ ea,b .
    It follows from the previous paragraph that ϕpea,b q equals either ea,b or zero. But we know that
ϕpea,b q represents a nontrivial homology class in ECHpY´ , λ´ q, so we must have ϕpea,b q “ ea,b .
    Step 4. It remains to prove that e1,0 and e0,1 are cycles which represent the nonzero homology
class in ECH2 pY, λq.
    We will restrict attention to e1,0 , as the proof for e0,1 follows by a symmetric argument. As in
Step 1, the claim holds if XΩ “ Epc, bcq where b ą 1 and c ą 0. The claim for general XΩ then
follows by repeating Steps 2 and 3.
Remark 3.3. For suitable almost complex structures J´ , Proposition 3.2 would also follow from
the combinatorial formula for the ECH differential12 in [24, Conj. A.3], together with algebraic
calculations as in [29, Prop. 5.9].
  12
     A combinatorial formula for the ECH differential on a different (not L-nice) perturbation of the ellipsoid Epa, bq
is computed in [33].


                                                          22
Lemma 3.4. Under the assumptions of Lemma 3.1, let

                          ϕ : pECCpY` , λ` q, BJ` q ÝÑ pECCpY´ , λ´ q, BJ´ q

be a chain map as provided by Proposition 2.25(c). Then

                                               ϕpea,b q “ ea,b .

Proof. By Proposition 3.2, ea,b is a cycle in the chain complex pECCpY` , λ` , BJ` q representing
a nontrivial element of ECHpY` , λ` q. We now repeat Step 3 of the proof of Proposition 3.2 to
conclude that ϕpea,b q “ ea,b .

3.2   Results for convex toric domains
Proof of Theorem 1.10. The “if” part of the theorem follows from Remark 1.9, so we just need to
prove the “only if” part. Assume that a ą 1 and that there exists an anchored symplectic embedding

                               pφ, Σq : pP pa, 1q, e1,0 q ÝÑ pB 4 pcq, e1,0 q.

We need to show that c ą a ` 1. By straightening Σ near the boundary (see below), we can find
an anchored symplectic embedding into pB 4 pc ´ εq, e1,0 q for some ε ą 0. Thus it is enough to show
that c ě a ` 1.
    Fix L ąą a, c. Let X´ Ă P pa, 1q and X` Ą B 4 pcq be L-nice approximations, and let W be the
resulting cobordism from pY` , λ` q to pY´ , λ´ q as at the beginning of §3.2.
    Note that the anchor Σ is necessarily transverse to BP pa, 1q and BB 4 pcq. We can then perform a
symplectic isotopy to straighten Σ near the boundary. Using the fourth bullet point in Lemma 2.5,
we can then construct an embedded symplectic surface C1 in W such that

                               C1 X pp´8, 0s ˆ Y´ q “ p´8, 0s ˆ e1,0 ,
                                  C1 X pr0, 8q ˆ Y` q “ r0, 8q ˆ e1,0 .

   As in the proof of Proposition 3.2, Step 2, we can choose a generic cobordism-admissible almost
complex structure J on W such that a perturbation of C1 , which we still denote by C1 , is J-
holomorphic. Let
                         ϕ : pECCpY` , λ` q, BJ` q ÝÑ pECCpY´ , λ´ q, BJ´ q
be a chain map as provided by Proposition 2.25(c).
   By Proposition 3.2, e1,1 is a cycle in pECCpY` , λ` q, BJ` q which represents the nonzero class
in ECH4 pY` , λ` q. Then ϕpe1,1 q represents the nontrivial class in ECH4 pY´ , λ´ q. In particular,
ϕpe1,1 q ‰ 0. The only ECH generators in Y´ with ECH index 4 are e21,0 , e1,1 , and e20,2 , so at least
one of these must appear in ϕpe1,1 q.
   We must have xϕe1,1 , e20,1 y “ 0 as in (3.4). Therefore e21,0 or e1,1 appears in ϕpe1,1 q.
   By Remark 2.26, the chain map ϕ respects the symplectic action filtration. Recall from (2.25)
that the actions of convex generators are approximated by the Ω-action in Definition 2.10. In
particular, we compute that
                                            AB 4 pcq pe1,1 q “ c
and
                            AP pa,1q pe21,0 q “ 2a,     AP pa,1q pe1,1 q “ a ` 1.

                                                      23
Therefore, by (2.25) and Remark 2.26, we have
                                  c ` 2L´1 ą minp2a, a ` 1q “ a ` 1.
Since L can be chosen arbitrarily large, we conclude that c ě a ` 1.

Lemma 3.5. Let XΩ´ and XΩ` be convex toric domains in R4 . Suppose that for every pair of
positive relatively prime integers a, b ą 0, we have
                                          AΩ´ pea,b q ď AΩ` pea,b q.                               (3.5)
Then Ω´ Ă Ω` .
Proof. If XΩ is a convex toric domain and a, b ą 0 are relatively prime positive integers, let Ωa,b Ă R2
denote the closed half-space to the lower left of the tangent line to B` Ω with slope ´b{a. Then
                                                      č
                                         Ω “ R2ě0 X Ωa,b                                            (3.6)
                                                            a,b

where the intersection is over pairs pa, bq of relatively prime positive integers. By Definition 2.10,
the hypothesis (3.5) is equivalent to the statement that
                                                 Ωa,b  a,b
                                                  ´ Ă Ω` .                                         (3.7)
It follows from (3.6) and (3.7) that Ω´ Ă Ω` .

Proof of Theorem 1.17. Suppose there exists a 2-anchored symplectic embedding
                          pφ, Σ1 , Σ2 q : pXΩ´ , e1,0 , e0,1 q ÝÑ pXΩ` , e1,0 , e0,1 q.
We need to show that Ω´ Ă Ω` . By Lemma 3.5, it is enough to show that if a, b ą 0 are relatively
prime positive integers, then the action inequality (3.5) holds.
    Fix L ąą a, b. Let X˘ , Y˘ , and W be as in Lemma 3.1. As in the proof of Theorem 1.10, from
the anchors we can construct disjoint embedded symplectic cylinders C1 , C2 in W , such that
                                C1 X pp´8, 0s ˆ Y´ q “ p´8, 0s ˆ e1,0 ,
                                  C1 X pr0, 8q ˆ Y` q “ r0, 8q ˆ e1,0 ,
                                C2 X pp´8, 0s ˆ Y´ q “ p´8, 0s ˆ e0,1 ,
                                  C2 X pr0, 8q ˆ Y` q “ r0, 8q ˆ e0,1 .
As in the proof of Proposition 3.2, Step 3, we can choose a generic cobordism-admissible almost
complex structure J on W so that perturbations of C1 and C2 , which we still denote by C1 and C2 ,
are J-holomorphic. In particular, C1 P MJ pe1,0 , e1,0 q, and C2 P MJ pe0,1 , e0,1 q.
    Now let
                           ϕ : ECCpY` , λ` , J` q ÝÑ ECCpY´ , λ´ , J´ q
be a chain map as provided by Proposition 2.25(c). By Lemma 3.4, we have
                                               ϕpea,b q “ ea,b .
By (2.25) and Remark 2.26, we have
                                    AX´ pea,b q ă AX` pea,b q ` 2L´1 .
Since L can be chosen arbitrarily large, the desired inequality (3.5) follows.

                                                       24
Proof of Theorem 1.13. This is a slight variation on the proof of Theorem 1.17. Suppose there exists
an anchored symplectic embedding

                                   pφ, Σq : pXΩ´ , e1,0 q ÝÑ pXΩ` , e1,0 q.

Suppose also that apΩ´ q ą bpΩ` q, or equivalently

                                         AΩ´ pe1,0 q ą AΩ` pe0,1 q.                                    (3.8)

We need to show that Ω´ Ă Ω` . By Lemma 3.5, it is enough to show that if a, b ą 0 are relatively
prime positive integers, then the inequality (3.5) holds.
   To prove (3.5), let L ąą a, b, and let X˘ , Y˘ , and W be as in the statement of Lemma 3.1.
From the anchor we can construct an embedded symplectic surface C1 in W such that
                                 C1 X pp´8, 0s ˆ Y´ q “ p´8, 0s ˆ e1,0 ,
                                   C1 X pr0, 8q ˆ Y` q “ r0, 8q ˆ e1,0 .

We can choose a generic cobordism-admissible almost complex structure J on W so that a pertur-
bation of C1 , which we still denote by C1 , is J-holomorphic. In particular, C1 P MJ pe1,0 , e1,0 q.
   Now let
                             ϕ : ECCpY` , λ` , J` q ÝÑ ECCpY´ , λ´ , J´ q
be a chain map as provided by Proposition 2.25. We know from Proposition 3.2 that e0,1 is a cycle in
pECCpY` , λq, BJ` q which represents the nonzero class in ECH2 pY` , λ` q. Then ϕpe0,1 q is a cycle in
pECCpY´ , λ´ q, BJ´ q which represents the nonzero class in ECH2 pY´ , λ´ q. The only possibilities are
either ϕpe0,1 q “ e0,1 or ϕpe0,1 q “ ϕpe1,0 q. If L is chosen sufficiently large, then the latter possibility
is ruled out by the action hypothesis (3.8), so ϕpe0,1 q “ e0,1 .
    By Lemma 2.27(b), there exists a J-holomorphic curve C2 P MJ pe0,1 , e0,1 q. Given the J-
holomorphic curves C1 and C2 , we now complete the proof as in the last paragraph of the proof of
Theorem 1.17.

Proof of Theorem 1.19. By Remark 1.20, we just need to prove the “only if” part of the theorem.
Suppose there exists a symplectic surface Σ Ă XΩ1 with

                                       BΣ “ e0,1 pXΩ1 q ´ e1,0 pXΩ q.

We need to prove the inequality (1.4). As in the proof of Theorem 1.10, it is sufficient to prove the
non-strict version of this inequality. In the notation of Definition 2.10, this inequality is equivalent
to
                                         AΩ1 pe0,1 q ě AΩ ph1,1 q.                                 (3.9)
    Choose L ą AΩ1 pe0,1 q. By Lemma 2.14, we can choose L-nice approximations X´ Ă XΩ and
X` Ă XΩ1 with X´ Ă X` . Let W be the symplectic cobordism at the beginning of §3.2.
    As in the proof of Theorem 1.10, from the surface Σ we can construct a surface C1 in W ,
and we can choose a cobordism-admissible almost complex structure J1 on W , such that C1 P
MJ1 pe0,1 , e1,0 q. By the relative adjunction formula (e.g. as packaged by the J0 index), C1 must be
a cylinder. Then, as in the proof of Proposition 3.2, the curve C1 satisfies automatic transversality.
    Likewise, as in the proof of Proposition 3.2, from the surface W X pt0u ˆ Cq we can construct
a surface C2 P W , and we can choose a cobordism-admissible almost complex structure J2 on W ,
such that C2 P MJ2 pe0,1 , e0,1 q.

                                                     25
    By Lemma 2.28(b), we have MJ2 pe0,1 , e1,0 q “ H.
    Let tJτ u1ďτ ď2 be a generic one-parameter family of cobordism-admissible almost complex struc-
tures on W interpolating between J1 and J2 above. Since MJ1 pe0,1 , e1,0 q is nonempty, and since
automatic transversality holds for the moduli spaces MJτ pe0,1 , e1,0 q, some breaking must occur for
τ P p1, 2q. That is, there exists τ P p1, 2q and a broken holomorphic current

                                  pCN´ , . . . , CN` q P MJτ pe0,1 , e1,0 q

as in §2.4 with N` ą N´ .
    We claim that N` “ 0. Suppose to the contrary that N` ą 0. Let Jτ` denote the almost
complex structure on R ˆ BX` determined by the restriction of Jτ to r0, 8q ˆ BX` . Then CN`
                                                       `
is a somewhere-injective holomorphic curve in MJτ pe0,1 , αq, and it follows from the ECH index
inequality in [20, Thm. 4.15] (see also the exposition in [23, §3.4]) that Ipαq ď 2. The only
possibility is that α “ e1,0 . But such a holomorphic curve cannot exist by automatic transversality
as in the proof of Proposition 3.2, Step 2.
    Since N` “ 0, it follows that C0 is a somewhere injective curve in MJτ pe0,1 , αq, and by the ECH
index inequality (loc. cit.) we have Ipαq ď 3. The only possibility is that α “ h1,1 . Then by (2.25)
and Remark 2.26, we have
                                    AΩ` pe0,1 q ě AΩ´ ph1,1 q ´ 2L´1 .
Since we can choose L arbitrarily large, this implies the desired inequality (3.9).

3.3   Results for concave toric domains
To finish up, we now prove Theorem 1.18 on 2-anchored symplectic embeddings of concave toric
domains. The proof is very similar, and in some sense “dual”, to the proof of Theorem 1.17 for
convex toric domains.
    Sinilarly to the beginning of §3.2, let XΩ´ and XΩ` be concave toric domains, and suppose there
exists a symplectic embedding φ : XΩ´ Ñ intpXΩ` q. Let L ą maxpapΩ´ q, apΩ` q, bpΩ´ q, bpΩ` qq
and let X´ Ă XΩ´ and X` Ą XΩ` be L-nice approximations provided by Lemma 2.20. Assume
also that the Reeb orbits e1,0 and e0,1 in X´ and X` all have the same (large) rotation number (we
will need this for automatic transversality below). Write Y˘ “ BX˘ and let λ˘ denote the induced
contact form on Y˘ . By Lemma 2.20, ECH generators in Y´ or Y` with symplectic action and
ECH index less than L can be identified with concave generators with combinatorial ECH index
and Ω´ -action or Ω` -action less than L, respectively, via the bijection ı, and we omit ı from the
notation. Let W be the symplectic cobordism from pY` , λ` q to pY´ , λ´ q given by X` zφpintpX´ qq
as in §2.4. Let J be a cobordism-admissible almost complex structure on W as in Definition 2.24.
Lemma 3.6. Let a, b ą 0 be relatively prime positive integers, and let Λ be a concave generator
with IpΛq
     q        q a,b q. Assume that L ą AΩ pΛq. Suppose there exist J-holomorphic cylinders
           “ Ipe                                  `
C1 P M pe1,0 , e1,0 q and C2 P MJ pe0,1 , e0,1 q and a J-holomorphic current C P MJ pΛ, ea,b q. Then
         J

Λ “ ea,b .
Proof. To start, we claim that C consists of a single somewhere injective curve C with multiplicity
1. To prove this, let C0 P MJ pΛ0 , ea,b q denote the component of C with a negative end asymptotic
to ea,b , and write C1 “ C ´ C0 P MJ pΛ1 , Hq. Thus Λ “ Λ0 Λ1 . We need to show that C1 “ 0. If not,
then Λ1 ‰ H. It then follows using the index formula (2.28) that IpΛ
                                                                   q 0 q ă IpΛq
                                                                            q      q a,b q. Then the
                                                                                “ Ipe
existence of C contradicts Lemma 2.27(a).

                                                     26
    It follows from the J0 bound (2.6), equation (2.30), and Remark 2.15 that

                               Jq0 pΛq ´ Jq0 pea,b q ě 2gpCq ´ 1 ` epΛq ` hpΛq.                             (3.10)

    Since IpΛq
          q      q a,b q, it follows from equation (2.29) that
               “ Ipe

                          Jq0 pΛq ´ Jq0 pea,b q “ 2pa ´ xpΛq ` b ´ ypΛqq ` epΛq ´ 1.                        (3.11)

    Since C has positive intersections with C2 , as in Lemma 2.28 we have

                                                   xpΛq ě a.                                                (3.12)

(This is simpler than the convex case in Lemma 2.28 because C does not have any ends asymptotic
to e0,1 .) Likewise, since C has positive intersections with C1 , we have

                                                    ypΛq ě b.                                               (3.13)

    Combining (3.10), (3.11), (3.12), and (3.13), we obtain

                                              2gpCq ` hpΛq ď 0,

with equality only if xpΛq “ a and ypΛq “ b. The rest of the argument proceeds as in the proof of
Lemma 3.1.

Proposition 3.7. Let XΩ be a concave toric domain and let a, b ą 0 be positive integers. Let X be
an L-nice perturbation of XΩ where L is large with respect to a, b, and Ω. Let Y “ BX and let λ
denote the induced contact form on Y . Let J` be a generic λ-compatible almost complex structure
on R ˆ Y . Let x be a cycle in pECC L pY, λq, J` q representing the nonzero class in ECH˚ pY, λq with
grading Ipe
        q a,b q. Then ea,b is a summand in x.

Proof. We first prove the proposition when XΩ is an ellipsoid Epac, bcq for c ą 0. In this case,
similarly to [24, Lem. 2.1(a)], ea,b is “maximal” in the sense that it uniquely maximizes Ω-action
among all concave generators with the same ECH index as ea,b and with all edges labeled ‘e’. The
proposition then follows similarly13 to [24, Lem. 5.5].
     To prove the proposition for a general concave toric domain XΩ , choose c ą 0 sufficiently small
so that Epac, bcq Ă intpXΩ q. Let X´ Ă Epac, bcq and X` Ą XΩ be L-nice perturbations, where L
is large with respect to a, b, and Ω. Let Y˘ and W be as in the statement of Lemma 3.6. Let J be
a cobordism-admissible almost complex structure on W which restricts to J` on r0, 8q ˆ Y` , and
let J´ denote the restriction of J to p8, 0s ˆ Y´ .
     As in Step 2 of the proof of Proposition 3.2, we can choose J to be generic as needed for
Lemma 2.27 and so that there exist J-holomorphic cylinders C1 P MJ pe1,0 , e1,0 q and C2 P MJ pe0,1 , e0,1 q.
     Let
                         ϕ : pECC˚L pY` , λ` , q, J` q ÝÑ pECC˚L pY´ , λ´ q, J´ q
be a chain map as provided by Proposition 2.25(c). By Proposition 2.25(a),(b), ϕpxq is a cycle
representing the nonzero generator in ECHpY´ , λ´ q with grading Ipe
                                                                 q a,b q. By the ellipsoid case, ea,b
is a summand in ϕpxq. Therefore x contains a summand Λ with xϕpΛq, ea,b y ‰ 0. By Lemma 2.27(b),
there exists a J-holomorphic current C P MJ pΛ, ea,b q. By Lemma 3.6, Λ “ ea,b .
  13
     The proof of [24, Lem. 5.5], which applies to convex toric domains, uses the formula for the ECH capacities of
convex toric domains in [24, Lem. 5.6]. In the present situation we instead need to use the formula for the ECH
capacities of concave toric domains in [4, Thm. 1.21].


                                                        27
Remark 3.8. For suitable J` , Proposition 3.7 can also be deduced from the formula for the ECH
differential in [40, Prop. 3.3], together with some algebraic calculations similar to [29].

Lemma 3.9. Under the assumptions of Lemma 3.6, let

                         ϕ : pECCpY` , λ` q, BJ` q ÝÑ pECCpY´ , λ´ q, BJ´ q

be a chain map as provided by Proposition 2.25(c). Then

                                          xϕpea,b q, ea,b y ‰ 0.

Proof. Let x be a cycle in pECCpY` , λ` q, BJ` q representing the nonzero homology class of grading
q a,b q. It follows from Proposition 2.25(a),(b) that ϕpxq is a cycle representing the nonzero class
Ipe
in ECHpY´ , λ´ q of the same grading. We know from Proposition 3.7 that ea,b is a summand in
ϕpxq. Therefore there is a summand Λ in x with xϕpΛq, ea,b y ‰ 0. It follows from Lemma 2.27(b)
and Lemma 3.6 that Λ “ ea,b .

Lemma 3.10. Let XΩ´ and XΩ` be concave toric domains in R4 . Suppose that for every pair of
positive relatively prime integers a, b ą 0, we have

                                       AΩ´ pea,b q ď AΩ` pea,b q.                                        (3.14)

Then Ω´ Ă Ω` .

Proof. This is a slight modification of the proof of Lemma 3.5.

Proof of Theorem 1.18. Suppose there exists a 2-anchored symplectic embedding

                           pφ, Σ1 , Σ2 q : pXΩ´ , γ1 , γ2 q ÝÑ pXΩ` , γ1 , γ2 q.

We need to show that Ω1 Ă Ω2 . By Lemma 3.10, it is enough to show that if a, b ą 0 are relatively
prime positive integers, then the action inequality (3.14) holds. This follows from Lemma 3.9 by
the same argument as in the proof of Theorem 1.17.


References
 [1] F. Bourgeois, A Morse-Bott approach to contact homology, in Symplectic and Contact Topol-
     ogy: Interactions and Perspectives, Fields Institute Communications 35 (2003), 55–77.

 [2] L. Buhovsky and E. Opshtein, Some quantitative results in C 0 symplectic geometry, Invent.
     Math. 205 (2016), 1–56.

 [3] K. Choi, Combinatorial        embedded      contact    homology      for      toric   contact   manifolds,
     arXiv:1608.07988.

 [4] K. Choi, D. Cristofaro-Gardiner, D. Frenkel, M. Hutchings, and V.G.B. Ramos, Symplectic
     embeddings into four-dimensional concave toric domains, J. Topol. 7 (2014), 1054–1076.

 [5] K. Christianson and J. Nelson, Symplectic embeddings of 4-dimensional polydisks into balls,
     Alg. Geom. Topol. 18 (2018), 2151–2178.

                                                    28
 [6] K. Cieliebak, H. Hofer, J. Latschev, and F. Schlenk, Quantitative symplectic geometry, Math.
     Sci. Res. Inst. Publ. 54 (2007), 1–44.

 [7] D. Cristofaro-Gardiner, Symplectic embeddings from concave toric domains into convex ones,
     J. Diff. Geom. 112 (2019), 199–232.

 [8] D. Cristofaro-Gardiner, T.S. Holm. A. Mandini, and A.R. Pires, On infinite staircases in toric
     symplectic four-manifolds, arXiv:2004.13062.

 [9] I. Datta, Lagrangian cobordisms between enriched knot diagrams, J. Symplectic Geom. 21
     (2023), 159–234.

[10] G. Dimitroglou Rizell and M. Sullivan, An energy-capacity inequality for Legendrian submani-
     folds, J. Topol. Anal. 12 (2020), 547–623.

[11] Y. Eliashberg and N. Mishachev, Introduction to the h-principle, Graduate Studies in Mathe-
     matics 48, American Mathematical Society (2002).

[12] J. Etnyre and M. Golla, Symplectic hats, J. Topology 15 (2022), 2216–2269.

[13] M. Gromov, Pseudoholomorphic curves in symplectic manifolds, Invent. Math. 82 (1985), 307–
     347.

[14] J. Gutt and M. Hutchings, Symplectic capacities from positive S 1 -equivariant symplectic ho-
     mology, Alg. Geom. Topol. 18 (2018), 3537–3600.

[15] J. Gutt, M. Hutchings, and V. Ramos, Examples around the strong Viterbo conjecture, J. Fixed
     Point Theory and Applications 24 (2022), Paper No. 41.

[16] P. Haim-Kislev, R. Hind, and Y. Ostrover, On the existence of symplectic barriers, Selecta
     Math. 30.4 (2024), 1–11.

[17] P. Haim-Kislev, R. Hind, and Y. Ostrover, Quantitative results on symplectic barriers,
     arXiv:2404.19396.

[18] R. Hind and O. Opshtein, Squeezing Lagrangian tori in dimension 4 , Comment. Math. Helv.
     95 (2020), 535–567.

[19] H. Hofer, K. Wysocki, and E. Zehnder, Properties of pseudo-holomorphic curves in symplecti-
     zations. II. Embedding controls and algebraic invariants, Geom. Func. Anal. 5 (1995), 270–328.

[20] M. Hutchings, The embedded contact homology index revisited , New perspectives and challenges
     in symplectic field theory, CRM Proc. Lecture Notes 49 (2009), 263–297.

[21] M. Hutchings, Quantitative embedded contact homology, J. Diff. Geom. 88 (2011), 231–266.

[22] M. Hutchings, Recent progress on symplectic embedding problems in four dimensions, PNAS
     108 (2011), 8093–8099.

[23] M. Hutchings, Lecture notes on embedded contact homology, Contact and symplectic topology,
     Bolyai Math. Stud. 26 (2014), 389–484.


                                                29
[24] M. Hutchings, Beyond ECH capacities, Geom. Topol. 20 (2016), 1085–1126.
[25] M. Hutchings, Mean action and the Calabi invariant, J. Modern Dynamics 10 (2016), 511–539.
[26] M. Hutchings, An elementary alternative to ECH capacities, PNAS vol. 119, No. 35,
     e2203090119 (2022).
[27] M. Hutchings and J. Nelson, Cylindrical contact homology for dynamically convex contact forms
     in three dimensions, J. Symplectic Geom. 14 (2016), 983–1012.
[28] M. Hutchings and M. Sullivan, The periodic Floer homology of a Dehn twist, Algebr. Geom.
     Topol. 5 (2005), 301–354.
[29] M. Hutchings and M. Sullivan, Rounding corners of polygons and the embedded contact homol-
     ogy of T 3 , Geom. Topol. 10 (2006), 169–266.
[30] M. Hutchings and C.H. Taubes, Gluing pseudoholomorphic curves along branched covered cylin-
     ders I , J. Symplectic Geom. 5 (2007), 43–137.
[31] M. Hutchings and C.H. Taubes, Gluing pseudoholomorphic curves along branched covered cylin-
     ders II , J. Symplectic Geom. 7 (2009), 29–133.
[32] M. Hutchings and C.H. Taubes, Proof of the Arnold chord conjecture in three dimensions II ,
     Geom. Topol. 17 (2013), 2601–2688.
[33] J. Nelson and M. Weiler, Torus knotted Reeb dynamics and the Calabi invariant,
     arXiv:2310.18307.
[34] J. Sabloff and L. Traynor, Obstructions to the existence and squeezing of Lagrangian cobordisms,
     J. Topol. Anal. 2 (2010), 203–232.
[35] J. Sabloff and L. Traynor, The minimal length of a Lagrangian cobordism between Legendrians,
     Select Math. 23 (2017), 1419–1448.
[36] F. Schlenk, Embedding problems in symplectic geometry, De Gruyter, 2005.
[37] F. Schlenk, Symplectic embedding problems, old and new , Bull. Amer. Math. Soc. 55 (2018),
     139–182.
[38] R. Siefring, Relative asymptotic behavior of pseudoholomorphic half-cylinders, Comm. Pure
     Appl. Math. 61 (2008), 1631–1684.
[39] C. H. Taubes, Embedded contact homology and Seiberg-Witten Floer cohomology I , Geom.
     Topol. 14 (2010), 2497–2581.
[40] J. Trejos, Symplectic embeddings of toric domains with boundary a lens space, arXiv:2312.15374.
[41] C. Wendl, Automatic transversality and orbifolds of punctured holomorphic curves in dimension
     four , Comment. Math. Helv. 85 (2010), 347–407.
[42] Y. Yao, From cascades to J-holomorphic curves and back , arXiv:2206.04334.
[43] Y. Yao, Computing embedded contact homology in Morse-Bott settings, arXiv:2211.13876.


                                                 30
