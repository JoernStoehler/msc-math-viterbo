---
source: arXiv:math/9903086
fetched: 2025-10-20
---
# On symplectic folding

arXiv:math/9903086v1 [math.SG] 15 Mar 1999
                        On symplectic folding
                                  Felix Schlenk

                               10. January 1999


                                     Abstract
     We study the rigidity and flexibility of symplectic embeddings of sim-
     ple shapes. It is first proved that under the condition rn2 ≤ 2r12 the
     symplectic ellipsoid E(r1 , . . . , rn ) with radii r1 ≤ · · · ≤ rn does not
     embed in a ball of radius strictly smaller than rn . We then use sym-
     plectic folding to see that this condition is sharp and to construct some
     nearly optimal embeddings of ellipsoids and polydiscs into balls and
     cubes. It is finally shown that any connected symplectic manifold of
     finite volume may be asymptotically filled with skinny ellipsoids or
     polydiscs.


Contents
1 Introduction                                                                                 3

2 Rigidity                                                                                     7

3 Flexibility                                                                                 11
  3.1 The folding construction . . . . . . . . . . . . . . .          .   .   .   .   .   .   14
  3.2 Folding in four dimensions . . . . . . . . . . . . . .          .   .   .   .   .   .   22
      3.2.1 The folding construction in four dimensions               .   .   .   .   .   .   22
      3.2.2 Multiple folding . . . . . . . . . . . . . . .            .   .   .   .   .   .   23
      3.2.3 Embeddings into balls . . . . . . . . . . . .             .   .   .   .   .   .   26
      3.2.4 Embeddings into cubes . . . . . . . . . . .               .   .   .   .   .   .   31
  3.3 Folding in higher dimensions . . . . . . . . . . . .            .   .   .   .   .   .   36
      3.3.1 Embeddings of polydiscs . . . . . . . . . . .             .   .   .   .   .   .   38
      3.3.2 Embeddings of ellipsoids . . . . . . . . . . .            .   .   .   .   .   .   39
  3.4 Lagrangian folding . . . . . . . . . . . . . . . . . .          .   .   .   .   .   .   62
  3.5 Symplectic versus Lagrangian folding . . . . . . . .            .   .   .   .   .   .   69
  3.6 Summary . . . . . . . . . . . . . . . . . . . . . . .           .   .   .   .   .   .   69

                                          2
4 Packings                                                              70
  4.1 Asymptotic packings . . . . . . . . . . . . . . . . . . . . . . . 73
  4.2 Refined asymptotic invariants . . . . . . . . . . . . . . . . . . 80
  4.3 Higher order symplectic invariants . . . . . . . . . . . . . . . 86

5 Appendix                                                                  87


1    Introduction
                              R
Let U be an open subset of n which is diffeomorphic to a ball, endow
U with the Euclidean volume form Ω0 , and let (M, Ω) be any connected
n-dimensional volume manifold. Then U embeds into M via a volume
preserving map if and only if Vol (U, Ω0 ) ≤ Vol (M, Ω). (A proof of this
“folk-theorem” is given below.)
               P
                                                                    R
     Let ω0 = ni=1 dxi ∧ dyi be the standard symplectic form on 2n and
                             R
equip any open subset U of 2n with this form. An embedding ϕ : U ֒→
R  2n is called symplectic, if ϕ∗ ω = ω . In particular, every symplectic
                                   0      0
embedding preserves the volume and the orientation. In dimension two,
the converse holds true. In higher dimensions, however, strong symplectic
rigidity phenomena appear. A spectacular example for this is Gromov’s
Nonsqueezing Theorem [12], which states that a ball B 2n (r) of radius r
symplectically embeds in the standard symplectic cylinder B 2 (R) × 2n−2R
if and only if r ≤ R. This and many other rigidity results for symplectic
maps could later be explained via symplectic capacities which arose from
the variational study of periodic orbits of Hamiltonian systems (see [14] and
the references therein).
     On the other hand, the flexibility of symplectic codimension 2 embed-
dings of open manifolds [13, p. 335] implies that given any symplectic ball
         R
B 2n−2 in 2n−2 and a symplectic manifold (M 2n , ω), there exists an ǫ > 0
such that B 2n−2 × B 2 (ǫ) symplectically embeds in M (see [10, p. 579] for
details).
    The aim of this work is to investigate the zone of transition between
rigidity and flexibility in symplectic topology. Unfortunately, symplectic
capacities can be computed only for very special sets, and there is still not
much known about what one can do with a symplectic map. We thus look
at a model situation. Let
                                 (                       n
                                                                     )
                                                         X π|zi |2
                                            C
            E(a1 , . . . , an ) = (z1 , . . . , zn ) ∈ n
                                                             ai
                                                                   <1
                                                 i=1


                                     3
                                                   p
be the open symplectic ellipsoid with radii ai /π, and write D(a) for the
open disc of area a and P (a1 , . . . , an ) for the polydisc D(a1 ) × · · · × D(an ).
Since a permutation of the symplectic coordinate planes is a (linear) sym-
plectic map, we may assume ai ≤ aj for i < j. Finally, denote the ball
E 2n (a, . . . , a) by B 2n (a) and the “n-cube” P 2n (a, . . . , a) by C 2n (a). We call
any of these sets a simple shape. We ask:

“Given a simple shape S, what is the smallest ball B and what is the smallest
cube C such that S symplectically fits into B and C?”

Observe that embedding S into a minimal ball amounts to minimizing its
diameter, while embedding S into a minimal cube amounts to minimizing
its symplectic width.
   Our main rigidity result states that for “round” ellipsoids the identity
provides already the optimal embedding.

Theorem 1 Let an ≤ 2a1 and a < an . Then E(a1 , . . . , an ) does not
embed symplectically in B 2n (a).

An ordinary symplectic capacity only shows that if a < a1 , there is no
symplectic embedding of E(a1 , . . . , an ) into B 2n (a). Our proof uses the first
n Ekeland-Hofer capacities. For n = 2, Theorem 1 was proved in [10] as
an early application of symplectic homology, but the argument given here is
much simpler and works in all dimensions.
    Our first flexibility result states that Theorem 1 is sharp.

Theorem 2A          Given any ǫ > 0 and a > 2π, there exists a symplectic
embedding
                                                       a       
                      E 2n (π, . . . , π, a) ֒→ B 2n        +π+ǫ .
                                                        2


Lalonde and McDuff observed in [18] that their technique of symplectic
folding can be used to prove Theorem 2A for n = 2. The symplectic folding
construction considers a 4-ellipsoid as a fibration of discs of varying size over
a disc and applies the flexibility of volume preserving maps to both the base
and the fibres. It is therefore purely four dimensional in nature. We refine
the method in such a way that it will nevertheless be sufficient to prove the
result for arbitrary dimension.

                                             4
                                                        PSfrag replacements
                                                                           1

       A
       π




     6
                       inclusion         lEB
                                          π
     5
                              sEB
                               π

     4
                                                volume condition
     3

     2
                                                  cEH
                                                                            a
           2   4   6     8          12     15           20     folding onceπ
                                                               24


               Figure 1: What is known about E(π, a) ֒→ B 4 (A)


    Theorem 1 and Theorem 2A shed some light on the power of Ekeland-
Hofer capacities: As soon as these invariants cease to imply that there is no
better embedding than the identity, there is indeed a better embedding.
    For embeddings of ellipsoids into cubes, the same procedure yields a sim-
ilarly sharp result, but for embeddings of polydiscs into balls and cubes the
result is less satisfactory. In four dimensions, the precise result is as follows.

Theorem 2B Let ǫ be any positive number.
  (i) Let a > π. Then there is no symplectic embedding of E(π, a) into
      C 4 (π), but E(π, a) symplectically embeds in C 4 ( a+π
                                                           2 + ǫ).

 (ii) Let a > 2π. Then P (π, a) symplectically embeds in B 4 ( a2 + 2π + ǫ) as
      well as in C 4 ( a2 + π + ǫ).

Question 1 Does P (π, 2π) symplectically embed in B 4 (A) for some A < 3π
or in C 4 (A) for some A < 2π?

Both, Theorem 2A and Theorem 2B as well as its higher dimensional version
can be substantially improved by multiple folding. Let us discuss the result

                                          5
in case of embeddings of 4-ellipsoids into 4-balls (cf. Figure 1). Let sEB (a)
be the function describing the best embeddings obtainable by symplectic
folding. It turns out that

                                 sEB (2π + ǫ) − 2π  3
                       lim sup                     = .
                        ǫ→0+             ǫ          7

Question 2 Let fEB (a) = inf{A | E(π, a) symplectically embeds in B 4 (A)}.
How does fEB look like near 2π? In particular,

                                 fEB (2π + ǫ) − 2π  3
                      lim sup                      < ?
                        ǫ→0+             ǫ          7

 Moreover, as a → ∞ the image of E(π, a) fills up an arbitrarily large per-
centage of the volume of B 4 (sEB (a)). This can also be seen via a Lagrangian
folding method, which was developed by Traynor in [31] and yielded the best
previously known results for the above embedding problem (see the curve
lEB in Figure 1). Symplectic folding, however, may be used to prove that
any connected symplectic manifold (M, ω) of finite volume can be asymp-
totically filled by skinny ellipsoids and polydiscs: For a > π set

                                      Vol (E 2n (απ, . . . , απ, αa))
               pE   2n
                a (M , ω) = sup                                       ,
                                  α           Vol (M, ω)

where the supremum is taken over all α for which E 2n (απ, . . . , απ, αa) sym-
plectically embeds in (M, ω), and define pPa (M, ω) in a similar way.

Theorem 3 lima→∞ pE                    P
                  a (M, ω) and lima→∞ pa (M, ω) exist and equal
1.

This result exhibits that in the limit symplectic rigidity disappears. We
finally give estimates of the convergence speed from below.
   Appendix A provides computer programs necessary to compute the op-
timal embeddings of ellipsoids into a 4-ball and a 4-cube obtainable by our
methods, and in Appendix B we give an overview on known results on the
Gromov width of closed symplectic manifolds.

Acknowledgement. I greatly thank Dusa McDuff for her fine criticism
on an earlier more complicated attempt towards Theorem 2A, which gave
worse estimates, and for having explained to me the main point of the folding
construction.

                                          6
   I also thank Leonid Polterovich for suggesting to me to look closer at
Lagrangian folding.



2    Rigidity
Throughout this paper, if there is no explicit mention to the contrary, all
maps will be assumed to be symplectic. In dimension two this just means
that they preserve the orientation and the area.
     Denote by O(n) the set of bounded
                                    Pn domains in    R
                                                     2n endowed with the

standard symplectic structure ω0 = i=1 dxi ∧ dyi . Given U ∈ O(n), write
|U | for the volume of U with respect to the Euclidean volume form Ω0 =
 1 n
n! ω0 . Let D(n) be the group of symplectomorphisms of       R
                                                            2n and D (n)
                                                                      c
                   R
respectively Sp(n; ) the subgroups of compactly supported respectively
                               R
linear symplectomorphisms of 2n . Define the following relations on O(n):


     U ≤1 V                                    R
               ⇐⇒ There exists a ϕ ∈ Sp(n; ) with ϕ(U ) ⊂ V .
     U ≤2 V    ⇐⇒ There exists a ϕ ∈ D(n) with ϕ(U ) ⊂ V .
     U ≤3 V    ⇐⇒ There exists a symplectic embedding ϕ : U ֒→ V .

Of course, ≤1 ⇒ ≤2 ⇒ ≤3 , but all the relations are different: That ≤1 and
≤2 are different is well known (see (2) below and Traynor’s theorem stated at
the beginning of section 3). The construction of sets U and V ∈ O(n) with
U ≤3 V but U 6≤2 V relies on the following simple observation. Suppose
that U and V not only fulfill U ≤3 V but are symplectomorphic, whence, in
particular, |U | = |V |. Thus, if U ≤2 V and ϕ is a map realizing U ≤2 V , no
        C
point of n \ U can be mapped to V , and we conclude that ϕ(∂U ) = ∂V .
In particular, the characteristic foliations on ∂U and ∂V are isomorphic,
and if ∂U is of contact type, then so is ∂V (see [14] for basic notions in
Hamiltonian dynamics).
   Let now U = B 2n (π), let

                    SD = D(π) \ {(x, y) | x ≥ 0, y = 0}

be the slit disc and set V = B 2n (π) ∩ (SD × · · · × SD). Traynor proved in
[31] that for n ≤ 2, V is symplectomorphic to B 2n (π). But ∂U and ∂V are
not even diffeomorphic. For n ≥ 2 very different examples were found in [8]
and [4]. Theorem 1.1 in [8] and its proof show that there exist U, V ∈ O(n)
with smooth convex boundaries such that U and V are symplectomorphic

                                     7
and C ∞ -close to B 2n (π), but the characteristic foliation of ∂U contains an
isolated closed orbit while the one of ∂V does not. And Corollary A in [4]
and its proof imply that given any U ∈ O(n), n ≥ 2, with smooth boundary
∂U of contact type, there exists a symplectomorphic and C 0 -close V ∈ O(n)
whose boundary is not of contact type.
    We in particular see that even for U being a ball, ≤3 does not imply ≤2 .

In order to detect some rigidity via the above relations we therefore must
pass to a small subcategory of sets:
   Let E(n) be the collection of symplectic ellipsoids described in the intro-
duction

                       E(n) = {E(a) = E(a1 , . . . , an )}

and write 4i for the restrictions of the relations ≤i to E(n).
   Notice again that

                           41 =⇒ 42 =⇒ 43 .

42 and 43 are actually very similar: Since ellipsoids are starlike, we may
apply Alexander’s trick to prove the extension after restriction principle (see
[6] for details), which tells us that given any embedding ϕ : E(a) ֒→ E(a′ )
and any δ ∈ ]0, 1[ we can find a ψ ∈ D(n) which coincides with ϕ on E(δa);
hence

        E(a) 43 E(a′ ) =⇒ E(δa) 42 E(a′ )           for all δ ∈]0, 1[ .    (1)

It is, however, not clear whether 42 and 43 are the same: While Theorem 2.2
proves this under an additional condition, the folding construction of section 3
suggests that they are different in general. But let us first prove a general
and common rigidity property of these relations:

Proposition 2.1 The relations 4i are partial orderings on E(n) .

Proof. The relations are clearly reflexive and transitive, so we are left with
identitivity. Of course, the identitivity of 43 implies the one of 42 which,
in its turn, implies the one of 41 . We still prefer to give independent proofs
which use tools whose difficulty is about proportional to the depth of the
results.
    It is well known from linear symplectic algebra [14, p. 40] that

                  E(a) 41 E(a′ ) ⇐⇒ ai ≤ a′i          for all i,           (2)


                                       8
in particular 41 is identitive.
    Given U ∈ O(n) with smooth boundary ∂U , the spectrum σ(U ) of U is
defined to be the collection of the actions of closed characteristics on ∂U . It
is clearly invariant under D(n), and for an ellipsoid it is given by
                                                   def
  σ(E(a1 , . . . , an )) = {d1 (E) ≤ d2 (E) ≤ . . . } = {kai | k ∈   N, 1 ≤ i ≤ n}.
Let now ϕ be a map realizing E(a) 42 E(a′ ). E(a) 42 E(a′ ) 42 E(a) gives
in particular |E(a)| = |E(a′ )|, and we conclude as above that ϕ(∂E(a)) =
∂E(a′ ). This implies σ(E(a)) = σ(E(a′ )) and the claim for 42 follows.
    To prove identitivity of 43 recall that Ekeland-Hofer capacities [7] pro-
                                                                           C
vide us with a whole family of symplectic capacities for subsets of n . They
are invariant under Dc (n), and for an ellipsoid E they are given by the
spectrum:

              {c1 (E) ≤ c2 (E) ≤ . . . } = {d1 (E) ≤ d2 (E) ≤ . . . }.            (3)

First observe that in the proof of the extension after restriction principle the
generating Hamiltonian can be chosen to vanish outside a large ball, so the
extension can be assumed to be in Dc (n). This shows that in the definition
of 42 we may replace D(n) by Dc (n) without changing the relation, and
that Ekeland-Hofer capacities may be applied to 42 . Next observe that for
any i ∈ {1, 2, 3} and α > 0

                    E(a) 4i E(a′ ) =⇒ E(αa) 4i E(αa′ ),                           (4)

just conjugate the given map ϕ with the dilatation by α−1 . Applying this
and (1) we see that for any δ1 , δ2 ∈ ]0, 1[ the assumed relations

                             E(a) 43 E(a′ ) 43 E(a)

imply

                         E(δ2 δ1 a) 42 E(δ1 a′ ) 42 E(a),

and now the monotonicity of all the ci = di immediately gives a = a′ .                ✷


   It is well known (we refer again to the beginning of section 3) that 42
does not imply 41 in general. However, a suitable pinching condition guar-
antees that “linear” and “non linear” coincide:

Theorem 2.2 Let κ ∈ ] π2 , π[. Then the following statements are equivalent:

                                          9
  (i) B 2n (κ) 41 E(a) 41 E(a′ ) 41 B 2n (π)

 (ii) B 2n (κ) 42 E(a) 42 E(a′ ) 42 B 2n (π)

(iii) B 2n (κ) 43 E(a) 43 E(a′ ) 43 B 2n (π) .

Theorem 1 follows from Theorem 2.2, (2) and (4). For n = 2, Theorem 2.2
was proved in [10]. That proof uses a deep result by McDuff, namely that
the space of symplectic embeddings of a ball into a larger ball is unknotted,
and then applies the isotopy invariance of symplectic homology. However,
Ekeland-Hofer capacities provide an easy proof. The crucial point is that as
true capacities they have - very much in contrast to symplectic homology -
the monotonicity property.

Proof of Theorem 2.2. (ii) ⇒ (i): By assumption we have B 2n (κ) 42
E(a) 42 B 2n (π), so the first Ekeland-Hofer capacity c1 gives

                                     κ ≤ a1 ≤ π                               (5)

and cn gives

                                 κ ≤ cn (E(a)) ≤ π.                           (6)

(5) and κ > π2 imply 2a1 > π, whence the only elements in σ(E(a))
possibly smaller than π are a1 , . . . , an . It follows therefore from (6) that
an = cn (E(a)), whence ci (E(a)) = ai (1 ≤ i ≤ n). Similarly we find
ci (E(a′ )) = a′i (1 ≤ i ≤ n), and from E(a) 42 E(a′ ) we conclude ai ≤ a′i .
     (iii) ⇒ (i) follows now by a similar reasoning as in the proof of the
identitivity of 43 : Starting from

                      B 2n (κ) 43 E(a) 43 E(a′ ) 43 B 2n (π),

(1) shows that for any δ1 , δ2 , δ3 ∈ ]0, 1[

               B 2n (δ3 δ2 δ1 κ) 42 E(δ2 δ1 a) 42 E(δ1 a′ ) 42 B 2n (π) .
                                                     π
Choosing δ1 , δ2 , δ3 so large that δ3 δ2 δ1 κ >     2   we may apply the already
proved implication to see

               B 2n (δ3 δ2 δ1 κ) 41 E(δ2 δ1 a) 41 E(δ1 a) 41 B 2n (π),

and since δ1 , δ2 , δ3 may be chosen arbitrarily close to 1, (2) shows that we
are done.                                                                    ✷


                                          10
3     Flexibility
As it was pointed out in the introduction, the flexibility of symplectic codi-
mension 2 embeddings of open manifolds implies that a condition as in
Theorem 1 is necessary for rigidity. An explicit necessary condition was
first obtained by Traynor in [31]. Her construction may be extended in an
obvious way (see subsection 3.4, in particular Corollary 3.18 (i)E ) to prove

Theorem (Traynor, [31, Theorem 6.4]) For all k ∈                           N
                                                               and ǫ > 0 there
exists a symplectic embedding
                                            
                        π
                   E       , π, . . . , π, kπ ֒→ B 2n (π + ǫ).
                       k+1

However, neither this theorem nor any refined version yielded by the La-
grangian method used in its proof can decide whether Theorem 1 is sharp
(cf. Figure 1). Our first flexibility result states that this is indeed the case:

Theorem 3.1 Let a > 2π and ǫ > 0. Then E 2n (π, . . . , π, a) embeds sym-
plectically in B 2n ( a2 + π + ǫ).

    For n = 2, this theorem together with Theorem 1 gives a complete an-
swer to our question in the introduction, whereas for arbitrary n it only
states that Theorem 1 is sharp. We indeed cannot expect a much better
result since (as is seen using Ekeland-Hofer capacities) E 2n (π, 3π, . . . , 3π)
does not embed in any ball of capacity strictly smaller than 3π.

Proof of Theorem 3.1. We will construct an embedding
                                         a      
                      Φ : E(a, π) ֒→ B 4    +π+ǫ
                                          2
satisfying

                        a     π 2 |z1 |2
    π|Φ(z1 , z2 )|2 <     +ǫ+            + π|z2 |2        for all (z1 , z2 ) ∈ E(a, π).   (7)
                        2          a
The composition of the linear symplectomorphism

                         E 2n (π, . . . , π, a) → E 2n (a, π, . . . , π)

with the restriction of Φ × id2n−4 to E 2n (a, π, . . . , π) is then the desired
embedding.

                                              11
    The great flexibility of 2-dimensional area preserving maps is basic for
the construction of Φ. We now make sure that we may describe such a map
by prescribing it on an exhausting and nested family of loops.

Definition A family L of loops in a simply connected domain U ⊂ 2 is     R
called admissible if there is a diffeomorphism β : D(|U |) \ {0} → U \ {p} for
some point p ∈ U such that

  (i) concentric circles are mapped to elements of L

 (ii) in a neighbourhood of the origin β is an orientation preserving isome-
      try.


Lemma 3.2 Let U and V be bounded and simply connected domains in 2           R
of equal area and let LU respectively LV be admissible families of loops in
U respectively V . Then there is a symplectomorphism between U and V
mapping loops to loops.

Remark. The regularity condition (ii) imposed on the families taken into
consideration can be weakened. Some condition, however, is necessary as
is seen from taking LU a family of concentric circles and LV a family of
rectangles with round corners and width larger than a positive constant. ✸

Proof of Lemma 3.2. We may assume that (U, LU ) = (D(πR2 ), {reiφ }),
and after reparametrizing the r-variable by a diffeomorphism of ]0, R[ which
is the identity near 0 we may assume that β maps the loop C(r) of radius r
to the loop L(r) in LV which encloses the area πr 2 .
    We now search for a family h(r, ·) of diffeomorphisms of S 1 such that the
map α given by α(reiφ ) = β(reih(r,φ) ) is a symplectomorphism. With other
words, we look for a smooth h : ]0, R[×S 1 → S 1 which is a diffeomorphism
for r fixed and solves the initial value problem
                         ∂h                     ′   ih(r,φ) )
                  (∗)       ∂φ (r, φ) = 1/ det β (re
                           h(r, 0)    = 0

View φ for a moment as a real variable. The existence and uniqueness
theorem for ordinary differential equations with parameter yields a smooth
               R R
map h : ]0, R[ × → satisfying (∗). Thus, h(r, ·) is a diffeomorphism of ,    R
and it remains to check that it is 2π-periodic. But this holds since the map
α : reiφ 7→ β(reih(r,φ) ) locally preserves the volume and α(C(r)) is contained
in the loop L(r).

                                      12
   Finally, α is an isometry in a punctured neighbourhood of the origin and
thus extends to all of D(πR2 ).                                           ✷

    While Traynor’s construction relies mainly on considering a 4-ellipsoid
as a Lagrangian product of a rectangle and a triangle, we view it as a trivial
fibration over a symplectic disc with symplectic discs of varying size as fibres:
                                 C
More generally, define for U ⊂ open and f : U → >0         R
             F(U, f ) = {(z1 , z2 ) ∈   C2 | z1 ∈ U, π|z2 |2 < f (z1)}.
This is the trivial fibration over U with fiber over z1 the disc of capacity
                R
f (z1 ). For λ ∈ set

                          Uλ = {z1 ∈ U | f (z1 ) ≥ λ}.

Given two such fibrations F(U, f ) and F(V, g), an embedding ψ : U ֒→ V
defines an embedding ψ × id : F(U, f ) ֒→ F(V, g) if and only if f (z1 ) ≤
g(ψ(z1 )) for all z1 ∈ U , and under the assumption that all the sets Uλ and
Vλ are connected, we see from Lemma 3.2 that inequalities

                        area Uλ < area Vλ          for all λ

are sufficient for the existence of an embedding F(U, f ) ֒→ F(V, g).

Example ([19, p. 54]) Let T (a, b) = F(R(a), g) with

                 R(a) = {z1 = (u, v) | 0 < u < a, 0 < v < 1}

and g(z1 ) = g(u) = b − u be the trapezoid. We think of T (a, b) as depicted
in Figure 2.                                                               ✸


Lemma 3.3 For all ǫ > 0,

 (i) E(a, b) embeds in T (a + ǫ, b)

(ii) T (a, b) embeds in E(a + ǫ, b).
                                                                          2
Proof. E(a, b) is described by U = D(a) and f (z1 ) = b (1 − π|za1 | ). For (i)
look at α and for (ii) at ω in Figure 3. The symplectomorphism ω is defined
on a round neighbourhood of R(a).                                            ✷


                                          13
fibre capacity



        b



                                                                 PSfrag replacements



                                                                                      u
                                                             a


                         Figure 2: The trapezoid T (a, b)


    Lemma 3.3 and its proof readily imply that in order to construct for any
a > 2π and ǫ > 0 an embedding Φ satisfying (7) it is enough to find for
any a > 2π and ǫ > 0 an embedding Ψ : T (a, π) ֒→ T ( a2 + π + ǫ, a2 + π + ǫ),
(u, v, z2 ) 7→ (u′ , v ′ , z2′ ) satisfying

                        a     πu
      u′ + π|z2′ |2 <     +ǫ+    + π|z2 |2    for all (u, v, z2 ) ∈ T (a, π).   (8)
                        2      a

3.1   The folding construction
The idea in the construction of an embedding Ψ satisfying (8) is to separate
the small fibres from the large ones and then to fold the two parts on top of
each other.

Step 1. Following [19, Lemma 2.1] we first separate the “low” regions
over R(a) from the “high” ones:
   Let δ > 0 be small. Let F be described by U and f as in Figure 4 and
write
                                  n    a      o
                        P1 = U ∩ u ≤ + δ ,
                                      2         
                                     a+π
                      P2 = U ∩ u ≥          + 9δ
                                       2
                           L = U \ (P1 ∪ P2 ).


                                       14
                                                      v
       D(a)               z1                      1

                                      α
                                                                                          u
                                                                                   a+ǫ

                                                                       PSfrag replacements

                                                      v
   D(a + ǫ)                z1                 1

                                      ω
                                                                                          u
                                                                                   a




                Figure 3: The first and the last base deformation

It is clear from the discussion at the beginning of the proof that there is an
embedding β × id : T (a, π) ֒→ F with

                                                                       π            
           β |{u< a −δ} = id    and       β |{u> a +δ} = id +                + 10δ, 0 .    (9)
                 2                                    2                  2

Step 2. We next map the fibers into a convenient shape:
   Let σ be a symplectomorphism mapping D(π) to Re and D( π2 ) to Ri as
specified in Figure 5. We require that for z2 ∈ D( π2 )
                                              π        
                          2
                    π|z2 | + 2δ > y(σ(z2 )) − − − 2δ ,
                                                2
i.e.
                                             π                         π
                     y(σ(z2 )) < π|z2 |2 −                for z2 ∈ D         .            (10)
                                             2                          2
       Write for this bundle of round squares
                                           a       a
                    (id × σ)F = S = S(P1 )    S(L)   S(P2 ).

                                             15
                                                                                  PSfrag replacements
           v
                                                                          U
       1
                          P1                          L                           P2
                                             δ
                                                                                                           u

           f

      π
       π
       2

                                                                                                           u
                                        a             a+π                                     π
                                        2   +δ         2           + 9δ                  a+   2   + 10δ


           Figure 4: Separating the low fibres from the large fibres


   In order to fold S(P2 ) over S(P1 ) we first move S(P2 ) along the y-axis
and then turn it in the z1 -direction over S(P1 ).

Step 3. To move S(P2 ) along the y-axis we follow again [18, p. 355]:
                         RR                  R
   Let c ∈ C ∞ ( , ) with c( ) = [0, 1 − δ] and

                               
                                   0,     t ≤ a2 + 2δ and t ≥ a+π
                                                                2 + 8δ
                    c(t) =
                                   1 − δ, a2 + 3δ ≤ t ≤ a+π
                                                         2  + 7δ.

Put I(t) =
                 Rt
                    0   c(s) ds and define ϕ ∈            C∞(R4 , R4 ) by
                                                                                         
                                                           1
               ϕ(u, x, v, y) =          u, x, v + c(u) x +                        , y + I(u) .                 (11)
                                                           2

We then find
                                        
                                            I2    0
                                                                                 
                                                                                        ∗  c(u)
                                                                                                   
               dϕ(u, x, v, y) =
                                            A    I2            with A =
                                                                                      c(u)   0
                                                                                                       ,

whence ϕ is a symplectomorphism. Moreover, with I∞ = I( a+π
                                                         2 + 8δ),

     ϕ |{u≤ a +2δ} = id              and         ϕ |{u≥ a+π +8δ} = id + (0, 0, 0, I∞ ),                        (12)
                2                                              2



                                                          16
                              z2                         PSfrag replacements
                                                           y
                                                                    1
                                        σ         − 12              2
                                                                              x
                                                                        −δ


                                             Ri                         − π2 − 2δ

                                             Re
                                                                        −π − δ



                       Figure 5: Preparing the fibres

                         1
and assuming that δ <   10   we compute
                         π            π
                           + 2δ < I∞ < + 5δ.                                   (13)
                         2            2
The first inequality in (13) implies

                      ϕ(P2 × Ri ) ∩ (  R2 × Re) = ∅.                           (14)

Remark. ϕ is the crucial map of the construction; in fact, it is the only
truly symplectic, i.e. not 2-dimensional map. ϕ is just the map which sends
the lines {v, x, y constant} to the characteristics of the hypersurface
                                                   
                                                  1
                     (u, x, y) 7→ u, x, c(u) x +      ,y ,
                                                  2
which generates (the cut off of) the obvious flow separating Ri from Re . ✸


Step 4. From (11), Figure 4 and Figure 5 we read off that the projec-
tion of ϕ(S) onto the (u, v)-plane is contained in the union of U with the
open set bounded by the graph of u 7→ δ + c(u) and the u-axis. Observe
that δ + c(u) ≤ 1.
    Define a local embedding γ of this union into the convex hull of U as
follows: On P1 the map is the identity and on P2 it is the orientation pre-
serving isometry between P2 and P1 which maps the right edge of P2 to the

                                        17
               c(t)


                 1−δ
                                               PSfrag replacements




                                                                   t
                         a                            a+π
                         2   + 2δ                      2    + 8δ


                              Figure 6: The cut off c

left edge of P1 . In particular, we have for z1 = (u, v) ∈ P2
                                             π
                         u(γ(z1 )) = a +       + 10δ − u.                 (15)
                                             2
On the remaining domain γ looks as follows: In a 4δ -collar of the line from
a to b the map is the identity and on a 4δ -collar of the line from c to d the
linear extension of the map on P2 , and we require
                        a     π                a     
         u′ (γ(u, v)) −    + δ < + 8δ − u −           + δ + 2δ,
                         2       2                 2
i.e.
                                             π
                      u′ (γ(u, v)) < −u +      + a + 12δ.                 (16)
                                             2
(14) shows that γ × id is one-to-one on ϕ(S).

Step 5. We finally adjust the fibers:
    First of all observe that the projection of ϕ(S) onto the z2 -plane is con-
tained in a tower shaped domain T (cf. Figure 8) and that by the second
inequality in (13) we have T ⊂ {y < π2 + 4δ}.
    We define a symplectomorphism τ from a neighbourhood of T to a disc
by prescribing the preimages of concentric circles as in Figure 8: We require

                                π               π
             • π|τ (z2 )|2 < y +  + 3δ for y ≥ − − 2δ                     (17)
                                2               2
                         2     −1     2 π
             • π|τ (z2 )| < π|σ (z2 )| + + 8δ for z2 ∈ Re .               (18)
                                        2

                                        18
               1




                                                   PSfrag replacements
                   b   ●                                   ●   c
                   a
                       ●                                   ●
                                                               d            u
                                          γ
                   d′
               1       ●
                   c′ ●




                   b′ ●
                      ●                                                     u′
                   a′


                                    Figure 7: Folding

   This finishes the construction. We think of the result as depicted in
Figure 9.

                                                1                ǫ
   Let now ǫ > 0 arbitrary and choose δ = min{ 10 ,             14 }.   It remains to check
that
                                  def
                                Ψ = (γ × τ ) ◦ ϕ ◦ (β × σ)
satisfies (8). So let z = (z1 , z2 ) = (u, v, x, y) ∈ T (a, π) and write Ψ(z) =
(u′ , v ′ , z2′ ). We have to show that
                              πu                       a
                       u′ −      + π|z2′ |2 − π|z2 |2 < + 14δ.                        (19)
                               a                       2

Case 1. β(z1 ) ∈ P1 :
   (9) implies u < a2 + δ, and by (12) and step 4 we have ϕ = id and γ = id,
whence (9) and (18) give
                                u′ = u′ (β(u, v)) < u + 2δ,
                                                               π
                       π|z2′ |2 = π|τ (σ(z2 ))|2 < π|z2 |2 +     + 8δ.
                                                               2

                                              19
                                       y


                                                    π
                                                    2   + 4δ

                       T

                                                           x

                                      PSfrag replacements

                                                    − π2 − 2δ



                                                    −π − δ



                      Figure 8: Mapping the tower to a disc

Therefore
                   πu                              π       π
            u′ −      + π|z2′ |2 − π|z2 |2 < u 1 −     + 2δ + + 8δ
                    a                               a       2
                                             a     π     π
                                           <     1−    + + 11δ
                                             2      a     2
                                             a
                                           =   + 11δ.
                                             2
Case 2. β(z1 ) ∈ P2 :
   Step 2 shows σ(z2 ) ∈ Ri , by (12) we have ϕ = id + (0, 0, 0, I∞ ), and (9)
implies u > a2 − δ and u(β(z1 )) + 2δ ≥ u + π2 + 10δ, whence by (15)
                                     π
         u′ = u′ (γ(β(z1 ))) = a +     + 10δ − u(β(z1 )) ≤ a − u + 2δ.
                                     2
Moreover, from (17), (10) and (13) we see
                     π|z2′ |2 = π|τ (σ(z2 ) + (0, I∞ ))|2
                                                   π
                              < y(σ(z2 )) + I∞ + + 3δ
                                                   2
                                      2   π π             π
                              < π|z2 | − + + 5δ + + 3δ
                                          2    2          2
                                      2   π
                              < π|z2 | + + 8δ.
                                          2

                                        20
                A




                                                PSfrag replacements
                π


                                                                   u
                                                A        a


                    Figure 9: Folding an ellipsoid into a ball


   Therefore

              πu                                  π       π
       u′ −      + π|z2′ |2 − π|z2 |2 < a − u 1 +     + 2δ + + 8δ
               a                                   a        2
                                            a     π π
                                      < a−      1+    + + 12δ
                                            2      a     2
                                        a
                                      =   + 12δ.
                                        2

Case 3. β(z1 ) ∈ L:
    By construction we have σ(z2 ) ∈ Ri , and using the definition of ϕ,
inequality (16) implies

                                                    π
                        u′ < −u(β(u, v)) +            + a + 12δ.
                                                    2

Next (17), (10) and the estimate I(t) < (1 − δ)(t − ( a2 + 2δ)) give

     π|z2′ |2 < π|τ (x(σ(z2 )), y(σ(z2 )) + I(u(β(u, v))))|2
                                             π
              < y(σ(z2 )) + I(u(β(u, v)) + + 3δ
                                            2                π
                      2   π                           a
              < π|z2 | − + (1 − δ) u(β(u, v)) − − 2δ + + 3δ.
                          2                           2        2
                       a              a                                  a
Moreover, (9) shows    2   −δ < u <   2   + δ, whence u(β(u, v)) > u >   2   − δ, and

                                           21
therefore
       πu                                     π             π a    
u′ −      + π|z2′ |2 − π|z2 |2 < −u(β(u, v)) + + a + 12δ −       −δ
        a                                     2          a a 2 a
                                              a
                                 +u(β(u, v)) − − 2δ − δ     − δ + δ + 2δ2 + 3δ
                                              2           2       2
                                 a         π      2
                               =   + 13δ + δ + 3δ
                                 2         a
                                 a
                               <   + 14δ.
                                 2
                                                                            ✷



3.2     Folding in four dimensions
In four dimensions we may exploit the great flexibility of symplectic maps
which only depend on the fibre coordinates to provide rather satisfactory
embedding results for simple shapes.
    We first discuss a modification of the folding construction described in
the previous section, then explain multiple folding and finally calculate the
optimal embeddings of ellipsoids and polydiscs into balls and cubes which
can be obtained by these methods.
    Not to disturb the exposition furthermore with δ-terms we skip them in
the sequel. Since all sets considered will be bounded and all constructions
will involve only finitely many steps, we won’t lose control of them.

3.2.1    The folding construction in four dimensions
The map σ in step 2 of the folding construction given in the previous section
was dictated by the estimate (19) necessary for the n-dimensional result. As
a consequence, the map γ had to disjoin the z2 -projection of P2 from the
one of P1 , and we ended up with the unused white sandwiched triangle in
Figure 9. In order to use this room as well we modify the construction as
follows:
    Replace the map σ of step 2 by the map σ given by Figure 10. If we
define ϕ as in (11), the z2 -projection of the image of ϕ will almost coincide
with the image of σ. Choose now γ as in step 4 and define the final map τ on
a neighbourhood of the image of ϕ such that it restricts to σ −1 on the image
of σ. If all the δ’s were chosen appropriately, the composite map Ψ will be
one-to-one, and the image Ψ will be contained in T (a/2 + π + ǫ, a/2 + π + ǫ)
for some small ǫ. We think of the result as depicted in Figure 11.

                                     22
                                                                y
                             z2
                                                         PSfrag replacements

                                                                         π
                                        σ                                2




                                                                             x
                                                − 21                 1
                                                                     2



                       Figure 10: The modified map σ



                   A


                                        PSfrag replacements
                   π

                                                                u
                                            A     a


                   Figure 11: Folding in four dimensions


3.2.2   Multiple folding

Neither Theorem 2 nor Traynor’s theorem stated at the beginning of section 3
tells us if E(π, 4π) embeds in B 4 (a) for some a ≤ 3π (cf. Figure 1). Mul-
tiple folding, which is explained in this subsection, will provide better em-
beddings. To understand the general construction it is enough to look at a
3-fold: The folding map Ψ is the composition of maps explained P in Figure 12.
                                                          R
Here are the details: Pick reasonable u1 , . . . , u4 ∈ >0 with 4j=1 ui = a
and put

                                  i
                                πX
                       li = π −     uj ,         i = 1, 2, 3.                (20)
                                a
                                  j=1



                                        23
                             β × id                                          id × σ1
                                                                     PSfrag replacements



                                      ϕ2 ◦ ϕ1                                       γ1 × id



            S2
              F2        S1      γ2 × id                               id × σ2
            F1


                                F3 S3       F4

                   ϕ3                                γ3 × id




                              Figure 12: Multiple folding


Step 1. Define β : R(a) → U by Figure 13.

Step 2. For l1 = π/2 the map σ1 is given by Figure 10, and in general it is
defined to be the symplectomorphism from D(π) to the left round rectangle
in Figure 14.
                                                                             Rt
Step 3. Choose cut offs ci over Li , i = 1, 2, put Ii (t) =                     0 ci (s) ds   and
define ϕi on β × σ1 (T (a, π)) by
                                                                            
                                                         1
            ϕi (u, x, v, y) =       u, x, v + ci (u) x +           , y + Ii (u) .
                                                         2

The effect of ϕ2 ◦ ϕ1 on the fibres is explained by Figure 14.

Step 4. γ1 is essentially the map γ of the folding construction: On P1
it is the identity, for u1 ≤ u ≤ u1 + l1 it looks like the map in Figure 7, and
for u > u1 + l1 it is an isometry. Observe that by construction, the slope of
the stairs S2 is 1, while the one of the upper edge of the floor F1 is less than
1. S2 and F1 are thus disjoint.

                                                24
                                                         PSfrag replacements


          v
                                  U
      1
              P1                P2            P3        P4
                                                                           u
                        L1             L2          L3




     π

     l1
     l2
     l3
                                                                           u
               u1        l1      u2        l2 u3 l3     u4



                                Figure 13: β



Step 5. γ2 × id is not really a global product map, ` `   but restricts to a
product on certain pieces of its domain: It fixes F1 S1 F2 , and it is the
product γ2 ×id on the remaining domain where γ2 restricts to an isometry on
u1 ≤ 0 and looks like the map given by Figure 15 on the z1 -projection of S2 .

For further reference, we summarize the result of the two preceding steps in
the

Folding Lemma. Let S be the stairs connecting two floors of minimal
respectively maximal height l.

  (i) If the floors have been folded on top of each other by folding on the
      right, S is contained in a trapezoid with horizontal lower edge of length
      l and left respectively right edge of length 2l respectively l.

 (ii) If the floors have been folded on top of each other by folding on the
      left, S is contained in a trapezoid with horizontal upper edge of length
      l and left respectively right edge of length l respectively 2l.

                                      25
                                                        PSfrag replacements
                                                                y
                                      y          2l1 + l2
              y               2l1


    l1                   ϕ1                        ϕ2
                         l1 − l2
                         x                         x                       x
    − 21            1
                    2



                   Figure 14: The first and the second lift


The remaining three maps are restrictions to the relevant parts of already
considered maps.

Step 6. On {y > 2l1 } the map σ2 is the automorphism whose image is
described by the same scheme as the image of σ1 , and id × σ2 restricts to
the identity everywhere else.

Step 7. On {y > 2l1 } the map ϕ3 restricts to the usual lift, and it is
the identity everywhere else.

Step 8. Finally, γ3 × id turns F4 over F3 . It is an isometry on F4 , looks like
the map given by Figure 7 on S3 and restricts to the identity everywhere else.

This finishes the multiple folding construction.


3.2.3      Embeddings into balls

In this subsection we use multiple folding to construct good embeddings of
ellipsoids into balls, and we also look at embeddings of polydiscs into balls.


3.2.3.1 Embedding ellipsoids into balls We now choose the uj ’s op-
timal.
   Fix u1 > 0. As proposed in Figure 31, we assume that the second floor
F2 touches the boundary of T (A, A) and that all the other uj ’s are chosen

                                      26
                                          γ2




                                      PSfrag replacements



                       Figure 15: Folding on the left


maximal. In other words, A is given by
                                                    
                                                  2π
                 A(a, u1 ) = u1 + 2 l1 = 2π + 1 −      u1 ,               (21)
                                                   a

and we proceed as follows: If the remaining length r1 = a − u1 is smaller
than u1 , i.e. u1 ≥ a/2, we are done; otherwise we try to fold a second time.
By the Folding Lemma, this is possible if and only if l1 < u1 , i.e.
                                          aπ
                                  u1 >       .                            (22)
                                         a+π
If (22) does not hold, the embedding attempt fails; if (22) holds, the Folding
Lemma and the maximality of u2 imply u2 = u1 − l2 , whence by (20)
                                  a+π       aπ
                           u2 =       u1 −     .
                                  a−π      a−π
If the upper left corner of F3 lies outside T (A, A), the embedding attempt
fails, otherwise we go on.
    In general, assume that we folded already j times and that j is even. If
the length of the remainder rj = rj−1 − uj is smaller than uj , we are done; if

                                         27
      A


                                                       PSfrag replacements


          l2l2
            l2
      π                    l1

                           l1
                                                                            u
                         u1     u1 + l1 A                           a


                                Figure 16: A 12-fold


not, we try to fold again: The Folding Lemma and the maximality of uj+1
imply uj+1 + 2lj+1 = uj , and substituting lj+1 = lj − uj+1 π/a we get
                                        a
                           uj+1 =           (uj − 2lj ).
                                     a − 2π
If uj ≤ 2lj , the embedding attempt fails, otherwise we go on: If the length
of the new remainder rj+1 = rj − uj+1 is smaller than uj+1 + lj , we are done;
otherwise we try to fold again: The Folding Lemma and the maximality of
uj+2 imply uj+2 + lj+2 = uj+1 + lj , whence by (20)

                                         a+π
                                uj+2 =       uj+1 .
                                         a−π
The embedding attempt fails here if and only if the upper left corner of the
floor Fj+3 lies outside T (A, A); if this does not happen, we may go on as
before.
     First of all note that whenever the above embedding attempt succeeds, it
indeed describes an embedding of E(π, a) into T (A(a, u1 ), A(a, u1 )). In fact,
it is enough to define the fiber adjusting map τ on a small neighbourhood of
the resulting tower T in such a way that for any z2 = (x, y), z2′ = (x′ , y ′ ) ∈ T
we have

                       y ≤ y ′ =⇒ |τ (z2 )|2 < |τ (z2′ )|2 .

                                         28
    (21) shows that we have to look for the smallest u1 for which the above
embedding attempt succeeds. Call it u0 = u0 (a). As we have seen above,
u0 lies in the interval
                                         
                                     aπ a
                               Ia =     ,   .                               (23)
                                    a+π 2

Moreover, if the embedding attempt succeeds for u1 , the same clearly holds
true for any u′1 > u1 . Hence, given u1 ∈ Ia , the corresponding embedding
attempt succeeds if and only if u1 ≥ u0 . Appendix A1 provides a com-
puter program calculating u0 , and the result sEB (a) = 2π + (1 − 2π/a)u0
is discussed and compared with the one yielded by Lagrangian folding in
subsection 3.5.

Remarks. 1. Simple geometric considerations show that our choices in
the above algorithm are optimal, i.e. sEB (a) provides the best estimate for
an embedding of E(π, a) into a ball obtainable by multiple folding.
    2. Let u1 > u0 and let N (u1 ) be the number of folds needed in the above
embedding procedure determined by u1 . Then N (u1 ) → ∞ as u1 ց u0 , i.e.
the best embeddings are obtained by folding arbitrarily many times. This
follows again from an easy geometric reasoning.
    3. Fix N and let AN (a) be the function describing the optimal embedding
obtainable by folding N times. Then {AN }n∈N is a monoton decreasing
family of rational functions on [2π, ∞[. For instance,

                              1                                     a+π
        A1 (a) = 2π + (a − 2π) ,         A2 (a) = 2π + (a − 2π)
                              2                                    3a + π

and

                                             (a + π)(a + 2π)
                  A3 (a) = 2π + (a − 2π)                       .
                                             4(a2 + aπ + π 2 )

So, A′1 (2π) = 21 and A′2 (2π) = A′3 (2π) = 73 . One can show that A′N (2π) =   3
                                                                                7
for all N ≥ 3. Thus

                                 sEB (2π + ǫ) − 2π  3
                       lim sup                     = .
                        ǫ→0+             ǫ          7

                                                                                ✸


                                        29
3.2.3.2    Embedding polydiscs into balls

Proposition 3.4 Let a > 2π and ǫ > 0. Then P (π, a) embeds in B 4 (sP B (a) + ǫ),
where sP B is given by

               a − 2π
  sP B (a) =          + (k + 2)π,        2(k2 − k + 1) < a/π ≤ 2(k2 + k + 1).
                 2k

Proof.    Let N = 2k − 1, k ∈   N, be odd.         From Figure 17 we read off that


                   A3


                                          PSfrag replacements



                    π
                                                                  u
                                    u1        A3            10π


           Figure 17: The optimal embedding P (π, 10π) ֒→ B 4 (A)

under the condition u1 > N π the optimal embedding by folding N times is
described by

          a = π + 2(u1 − π) + 2(u1 − 3π) + · · · + 2(u1 − N π) + π
               = 2π + 2ku1 − 2k2 π

and AN (a) = u1 + 2π; hence

                                    a − 2π
                         AN (a) =          + (k + 2)π,
                                      2k

provided that AN (a) − 2π > (2k − 1)π. This condition translates to a >
2(k2 − k + 1)π, and the claim follows.                                ✷

Remark. sP B is the optimal result obtainable by multiple folding. In fact,
a simple geometric argument or a similar calculation as in the proof shows
that folding 2k times yields worse estimates.                            ✸


                                         30
                                         PSfrag replacements

                         A
                         π



                                                inclusion
                     7
                                         sP B
                                          π
                     5

                     3                      volume condition
                     2                      cEH
                                                                a
                                                                π
                         1 2            6             10


             Figure 18: What is known about P (π, a) ֒→ B 4 (A)

                                        √
Remark 3.5 Let dP B (a) = sP B (a) − 2πa be the difference between sP B
and the volume condition. dP B attains √  its local maxima at ak = 2(k2 −
k + 1)π, where dP B (ak ) = (2k + 1)π − 2π k2 − k + 1. This is an increasing
sequence converging to 2π.                                                ✸



3.2.4    Embeddings into cubes

                               C
Given an open set U in n , call the orthogonal projections of U onto the
n symplectic coordinate planes the shadows of U . As pointed out in [10, p.
580], symplectic capacities measure to some extent the areas of the shad-
ows of a set. Of course, this can not be made rigorous since the areas of
shadows are no symplectic invariants, but for sufficiently regular sets these
areas indeed are symplectic capacities: As remarked before, the capacities
a1 , . . ., an of the ellipsoid E(a1 , . . . , an ) are symplectic capacities and, more
generally, given any bounded U with connected smooth boundary ∂U of
restricted contact type and with a shadow whose boundary is the shadow
of a closed characteristic on ∂U which lies in a single symplectic coordinate
direction, this shadow is a capacity of U [7, Proposition 2]. Moreover, the
smallest shadow of a polydisc and of a symplectic cylinder are capacities.
      Instead of studying embeddings into minimal balls, i.e. to reduce the
diameter of a set, it is therefore a more symplectic enterprise to look for
minimal embeddings into a polydisc C 2n (a), i.e. to reduce the maximal
shadow.

                                          31
    The Non-Squeezing Theorem states that the smallest shadow of simple
sets (like ellipsoids, polydiscs or cylinders) can not be reduced. We therefore
call obstructions to the reduction of the maximal shadow highest order rigid-
ity. (More generally, calling an ellipsoid or a polydisc given by a1 ≤ · · · ≤ an
                                                                   R
i-reducible if there is an embedding into C 2i (a′ ) × 2n−2i for some a′ < ai ,
one might explore i-th order rigidity.)

The disadvantage of this approach to higher order rigidity is that for a
polydisc there are no good higher invariants available, in fact, Ekeland-
Hofer-capacities see only the smallest shadow [7, Proposition 5]:

                               cj (P (a1 , . . . , an )) = ja1 .

Many of the polydisc-analogues of the rigidity results for ellipsoids proved
in section 2 are therefore either wrong or much harder to prove. It is for
instance not true that P (a1 , . . . , an ) embeds linearly in P (a′1 , . . . , a′n ) if and
only if ai ≤ a′i for all i, for a long enough 4-polydisc may be turned into the
diagonal of a cube of smaller maximal shadow:
                                √
Lemma 3.6 Let r > 1 + 2. Then P 2n (π, . . . , π, πr 2 ) embeds linearly in
C 2n (a) for some a < πr 2 .

Proof. It is clearly enough to prove the lemma for n = 2. Consider the
linear symplectomorphism given by
                                                  1
                    (z1 , z2 ) 7→ (z1′ , z2′ ) = √ (z1 + z2 , z1 − z2 ).
                                                   2

For (z1 , z2 ) ∈ P (π, πr 2 ) we have for i = 1, 2

                       1                                 1 r2
              |zi′ |2 ≤ (|z1 |2 + |z2 |2 + 2|z1 ||z2 |) ≤ +   + r,                     (24)
                       2                                 2  2
and the √right hand side of (24) is strictly smaller than r 2 provided that
r > 1 + 2.                                                               ✷

Similarly, we don’t know how to prove the full analogue of Proposition 2.1:
   Let P(n) be the collection of polydiscs

                               P(n) = {P (a1 , . . . , an )}

and write i for the restrictions of the relations ≤i to P(n). Again 2
and 3 are very similar, again all the relations i are clearly reflexive and

                                             32
transitive, and again the identitivity of 2 , which again implies the one of
1 , follows from the equality of the spectra, which is implied by the equality
of the volumes. (Observe that, even though the boundary of a polydisc is
not smooth, its spectrum is still well defined.) For n=2 the identitivity of
3 is seen by using any symplectic capacity, which determines the smallest
shadow, and the equality of the volumes; but for arbitrary n we don’t know
a proof.

    While the lack of convenient invariants made it impossible to get good
rigidity results for embeddings into polydiscs, the folding construction pro-
vides us with rather satisfactory flexibility results.

3.2.4.1 Embedding ellipsoids into cubes We again use the notation
of section 3.2, fold first at some reasonable u1 and then choose the subsequent
uj ’s maximal (see Figure 19). Let w(a, u1 ) = u1 + l1 = π + (1 − π/a)u1 be


        A

                                                 PSfrag replacements

        π


                                                                      u
                       u1    A                                   7π


            Figure 19: The optimal embedding E(π, 7π) ֒→ C 4 (A)

the width of the image and h = h(a, u1 ) its height.
   Let’s first see what we get by folding once: The only condition on u1 is
a/2 ≤ u1 , whence h(a, u1 ) = π < π + (1 − π/a)u1 = w(a, u1 ). The optimal
choice of u1 is thus u1 = a/2.
   Suppose now that we fold at least twice. The only condition on u1 is
then again l1 < u1 , i.e.
                                         aπ
                                 u1 >       .
                                        a+π
Observe that h(a, u1 ) diverges if u1 approaches aπ/(a+ π). Note also that w
is increasing in u1 while h is decreasing. Thus, w(a, u1 ) and h(a, u1 ) inter-
sect exactly once, namely in the optimal u1 , which we call u0 . In particular,

                                        33
we see that folding only once never yields an optimal embedding. Write
sEC (a) = π + (1 − π/a)u0 for the resulting estimate. It is computed in
Appendix A2. Again, it is easy to see that our choices in the above proce-
dure are optimal, i.e. sEC (a) provides the best estimate for an embedding
of E(π, a) into a cube obtainable by symplectic folding.

Example. If we fold exactly twice, we have h = 2l1 + l2 , or, since l2
satisfies a = u1 + u2 + (a/π)l2 and u2 = u1 − l2 ,

                                    2π      π(a − 2u1 )
                         h = 2π −      u1 +             .
                                     a        a−π
Thus, provided that l2 + (a/π)l2 ≤ w, the equation h = w yields

                                        aπ(2a − π)
                             u0 =                    .                  (25)
                                    a2   + 2aπ − π 2
Indeed, u0 satisfies (25) whenever a > π. Finally, l2 + (a/π)l2 ≤ w holds if
and only if π ≤ a ≤ 3π.                                                  ✸

                     A
                     π                     PSfrag replacements


                 4
                                 inclusion
                 3                             sEC
                                                π

                 2

                                           volume condition
                 1
                                         cEH
                                                              a
                                                              π
                         2   3      4     5    6     7


            Figure 20: What is known about E(π, a) ֒→ C 4 (A)

     In fact, (25) also holds true for all a for which the optimal embedding
of E(π, a) obtainable by multiple folding is a 3-fold for which the height is
still described by h = 2l1 + l2 , i.e. for which u4 ≤ u3 . This happens for

                                          34
3 < a/π < 4.2360 . . . , whence
                          aπ(3a − π)                    a
             sEC (a) =                        for 1 ≤     ≤ 4.2360 . . . .
                         a2+ 2aπ − π 2                  π
In general, sEC is a piecewise rational function. Its singularities are those
a for which uN (a) = uN (a)+1 , where we wrote N (a) for the number of folds
determined by u0 (a).
                                        p
Remark 3.7 Let dEC (a) = sEC (a) − πa/2 be the difference between sEC
and the volume condition. The set of local minima of dEC coincides with
its singular set, i.e. with the singular set of sEC . On the other hand, dEC
attains its local maxima at those a for which the point of FN (a)+1 touches
the boundary of T (A, A). Computer calculations suggest that on this set,
dEC is increasing, but bounded by (2/3)π.                                  ✸


3.2.4.2   Embedding polydiscs into cubes
Proposition 3.8 Let a > 2π and ǫ > 0. Then P (π, a) embeds in C 4 (sP C (a) + ǫ),
where sP C is given by
                     
                       (N + 1)π, (N − 1)N + 2 < πa ≤ N 2 + 1
          sP C (a) =    a+2N π
                         N +1 ,  N 2 + 1 < πa ≤ N (N + 1) + 2 .

Proof. The optimal embedding by folding N times is described by

                          2u1 + (N − 1)(u1 − π) = a,

whence u1 = a+(N   −1)π
                N +1    ; in fact, by the assumption on a, the only condition
u1 > π for N ≥ 2 is satisfied. Thus AN (a) = max{ a+2N     π
                                                       N +1 , (N + 1)π}, and
the proposition follows.                                                    ✷

                                               √
Remark 3.9 The difference dP C (a) = sP C (a) − πa between sP C and the
                                                     2
                       √ its local maxima at aN = (N − N + 2)π, where
volume condition attains
                           2
dP C (aN ) = (N + 1)π − N − N + 2 π. This is an increasing sequence con-
verging to (3/2)π.                                                    ✸


    Since for a ≤ 2π folding cannot reduce P (π, a) and since we believe that
for small a folding is essentially the only way to achieve a reduction (see also
[20]), we state:

                                         35
                                                     PSfrag replacements


              π
                                                                                u
                               u1 u1 + π                              a


                     Figure 21: Folding P (π, a) three times


Conjecture 3.10 The polydisc-analogue of Theorem 1’ holds. In particu-
lar,
P 2n (π, . . . , π, a) embeds in C 2n (A) for some A < a if and only if a > 2π.

3.3    Folding in higher dimensions
Even though symplectic folding is an essentially four dimensional process,
we may still use it to get good embeddings in higher dimensions as well. The
point is that we may fold into different symplectic directions of the fiber.
In view of the applications of higher dimensional folding in subsection 4.1
and 4.2 we will concentrate on embedding skinny polydiscs into cubes and
skinny ellipsoids into balls and cubes.

   Given domains U ⊂        R2n and V, W ⊂ Rn and given α > 0, we set
         αU = {αz ∈      R2n | z ∈ U }        and        αV × W = α(V × W ).

As in the four dimensional case we may view an ellipsoid E(a1 , . . . , an ) as
fibered over the disc D(an ) with ellipsoids γE(a1 , . . . , an−1 ) of varying size
as fibres. By deforming the base D(an ) to a rectangle as in Figure 3 we may
get rid of the y1 -coordinate. It will be convenient to get rid of the other
                               R                R
yi -coordinates too. Write 2n (x, y) = n (x) × n (y) and set   R
                                                         n
                                                         X
          △(a1 , . . . , an ) = {0 < x1 , . . . , xn |
                                                           xi
                                                               ai
                                                                    < 1} ⊂   Rn(x),
                                                         i=1
              ✷(b1 , . . . , bn ) = {0 < yi < bi , 1 ≤ i ≤ n} ⊂           Rn(y).

                                             36
Lemma 3.11 For all ǫ > 0,

  (i) E(a1 − ǫ, . . . , an − ǫ) embeds in △(a1 , . . . , an ) × ✷n (1) in such a way
      that for all α ∈ ]0, 1], αE(a1 − ǫ, . . . , an − ǫ) is mapped into (α +
      ǫ)△(a1 , . . . , an ) × ✷n (1).

 (ii) △(a1 − ǫ, . . . , an − ǫ) × ✷n (1) embeds in E(a1 , . . . , an ) in such a way
      that for all α ∈ ]0, 1], α△(a1 − ǫ, . . . , an − ǫ) × ✷n (1) is mapped into
      (α + ǫ)E(a1 , . . . , an ).

Proof. By Lemma 3.2 we find embeddings αi : D(ai − ǫ) ֒→ ✷(ai , 1) satis-
fying

                                    ǫ    a1
     xi (αi (zi )) < π|zi |2 +                             for zi ∈ D(ai − ǫ), 1 ≤ i ≤ n
                                    n max(1, an )

(cf. Figure 3). Given (z1 , . . . , zn ) ∈ E(a1 − ǫ, . . . , an − ǫ) we then find

         n
         X                           n
                                     X
           xi (αi (zi ))               π|zi |2         1 ǫ a1
                                <                  +
                  ai                        ai         ai n an
         i=1                          i=1
                                            ai − ǫ    ǫ       ǫ   ǫ
                                < max              +    = 1−    +   = 1,
                                       i      ai     an      an an

and given (z1 , . . . , zn ) ∈ αE(a1 − ǫ, . . . , an − ǫ) we find

                n
                X                            n
                                             X
                  xi (αi (zi ))                π|zi |2           a1 ǫ
                                        <                    +        < α + ǫ.
                           ai                         ai         ai n
                 i=1                         i=1


    The proof of (ii), which uses products of maps ωi as in Figure 3, is sim-
ilar.                                                                       ✷

    Forgetting about all the ǫ’s, we may thus view an ellipsoid as a La-
grangian product of a simplex and a cube. In the setting of symplectic
folding, however, we will still rather think of E(a1 , . . . , an ) as fibered over
the base ✷(an , 1). By Lemma 3.11(i) we may assume that the fiber over
(xn , yn ) is (1 − x1 /an )△(a1 , . . . , an−1 ) × ✷n−1 (1).
    Similarly, by mapping the discs D(ai ) symplectomorphically to the rect-
angles ✷(ai , 1) and then looking at the Lagrangian instead of the symplectic
splitting, we may think of P (a1 , . . . , an ) as ✷(a1 , . . . , an ) × ✷n (1).

                                                 37
3.3.1    Embeddings of polydiscs
We fold a polydisc P (a1 , . . . , an ) by folding a four dimensional factor P (ai , aj )
for some i 6= j ∈ {1, . . . , n} and leaving the other factor alone. An already
folded polydisc may be folded again by restricting the folding process to a
component containing no stairs. The choice of i and j is only restricted by
the condition that the new image should still be embedded.

3.3.1.1 Embedding polydiscs into cubes In view of an application in
subsection 4.1 we are particularly interested in embedding thin polydiscs into
cubes. So fix P 2n (a, π, . . . , π) and let A be reasonably large. As explained
above, we think of P 2n (a, π, . . . , π) as ✷n (a, π, . . . , π)× ✷n (1) and of C 2n (A)
as ✷n (A) × ✷n (1). The base direction will thus be the z1 -direction. Folding
into the zi -direction for some i ∈ {2, . . . , n}, we will always lift into the
xi -direction.
     We describe the process for n = 3: First, fill a z1 -z2 -layer as well as
possible by lifting N times into the x2 -direction (cf. Figure 21). Then lift
once into the x3 -direction and fill a second z1 -z2 -layer . . . . If u1 is chosen
appropriately, we will fold N times into the x3 -direction and fill N + 1 z1 -
z2 -layers.
     The following proposition generalizes Proposition 3.8 to arbitrary dimen-
sion.
Proposition 3.12 Let a > 2π and ǫ > 0. Then P 2n (π, . . . , π, a) embeds in
C 2n (s2n                    2n
       P C (a) + ǫ), where sP C is given by
            (
 2n                (N + 1)π,     (N − 1)N n−1 < πa − 2 ≤ (N − 1)(N + 1)n−1
sP C (a) =        a−2π
                (N +1)n−1
                          + 2π, (N − 1)(N + 1)n−1 < πa − 2 ≤ N (N + 1)n−1 .

Proof. The optimal embedding by folding N times in each z1 -z2 -layer is
described by
                      2u1 + ((N + 1)n−1 − 2)(u1 − π) = a,
whence
                                  a + ((N + 1)n−1 − 2)π
                           u1 =                         .
                                        (N + 1)n−1
Thus
                                                                 
                                 a + 2((N + 1)n−1 − 1)π
             AN (a) = max                               , (N + 1)π ,
                                       (N + 1)n−1
and the proposition follows.                                                          ✷


                                           38
3.3.2   Embeddings of ellipsoids
We will concentrate on embedding ellipsoids E 2n (π, . . . , π, a) with a very
large.

3.3.2.1 Embedding ellipsoids into cubes Studying embeddings
E 2n (π, . . . , π, a) ֒→ C 2n (A) of skinny ellipsoids into minimal cubes, we face
the problem of filling the fibers ✷n−1 (A)×✷n−1 (1) of the cube by many small
fibers γ△n−1 (π) × ✷n−1 (1) of the ellipsoid. Forget about the irrelevant y-
factors. Since a is very large, γ decreases very slowly. We are thus essentially
left with the problem of filling n − 1-cubes by equal n − 1-simplices. This is
trivial for n − 1 = 1 and n − 1 = 2, but impossible for n − 1 ≥ 3. Indeed,
only 2m−1 m-simplices △m (π) fit into ✷m (π), whence we only get

                         |E 2n (π, . . . , π, a)|    2n−2
                       lim                        ≥          .                (26)
                      a→∞ |C 2n (s2n (a))|          (n − 1)!
                                   EC

    We describe now the embedding process for n − 1 = 2 in more detail
(cf. Figure 22). We first fill almost half of the “first column” of the cube
                          x3
                               ν1
                     A

                     µ1              δ1
                                          PSfrag replacements


                                    ..
                                     .



                                                             x2
                                                        A


           Figure 22: Filling the cube fibres by the ellipsoid fibres

fiber, move the ellipsoid fibre out of this first column (µ1 ), deform it to
its complementary fiber (δ1 ), move this fiber back to the first column (ν1 ),
and fill almost all of the remaining room in the first column. We then
pass to the second column and proceed as before. The deformations δi are
performed by applying 2-dimensional maps to both symplectic directions

                                           39
of the ellipsoid fibers (see Figure 25 in 3.3.2.2 and the text belonging to it
for more details). In order to guarantee that different stairs do not inter-
sect, we arrange the stairs arising from folding in such a way that the z1 -
projections of “upward-stairs” lie in {0 < y1 < 1/2} while the z1 -projections
of “downward-stairs” lie in {1/2 < y1 < 1}, and we arrange the stairs arising
from moving in such a way that the z1 -projections of the µi - respectively
νi -stairs lie in {0 < y1 < 1/4} respectively {1/4 < y1 < 1/2} if i is odd and
in {1/2 < y1 < 3/4} respectively {3/4 < y1 < 1} if i is even (cf. Figure 7).
The x1 -intervals used for folding respectively moving will then be double
respectively four times as large as usual, but this will not affect (26).

Remark. We will prove in subsection 4.1 that the left hand side of (26) is
1 for any n.                                                            ✸


3.3.2.2 Embedding ellipsoids into balls If we try to fill the fibers
△n−1 (A) × ✷n−1 (1) of a ball by many small fibers γ△n−1 (π) × ✷n−1 (1) of
a skinny ellipsoid, we end up with a result for s2n EB (a) as in (26). In the
problem of embedding a skinny ellipsoid into a minimal ball, however, both
the fibers of the ellipsoid and the fibers of the ball are balls. This may be
used to prove

Proposition 3.13 For any n,

                                |E 2n (π, . . . , π, a)|
                             lim                         = 1.
                             a→∞ |B 2n (s2n (a))|
                                          EB

Proof.   The idea of the proof is very simple: Instead of packing a large
simplex by small simplices, we will leave the simplices alone and pack the
cubes by small cubes, a trivial problem.
                                 N
   So pick a very large l ∈ , write
                                               
                 2n         (i − 1)A         iA
          Pi = B (A) ∩               < x1 <       ,    1 ≤ i ≤ l,
                                l             l

and set
                                            A − A/l
                                     k1 =           ,
                                               π
where A is again a parameter which will be fixed later on. After applying
the diagonal map diag [k1 , . . . , k1 , 1/k1 , . . . , 1/k1 ] to the fibers, the ellipsoid

                                            40
                   A
                       T1
            A − A/l
                              T2

                                               PSfrag replacements




                                                          Tl         x1
                            A/l 2A/l                           A


            Figure 23: Embedding a skinny ellipsoid into a ball


is contained in ✷(a, 1) × △n−1 (k1 π) × ✷n−1 (1/k1 ). We will embed some part
✷(b1 , 1) × △n−1 (k1 π) × ✷n−1 (1/k1 ) of this set into P1 by fixing the simplices
and moving the cubes along the yi -directions (2 ≤ i ≤ n) (see Figure 23 and
Figure 24).
    We want to fill as much of ✷n−1 (1) by cubes ✷n−1 (1/k1 ) as possible.
However, in order to use also the space in P2 optimally, we will have to
deform the ellipsoid fibers before passing to P2 , and for this we will have to
use some space in ✷n−1 (1). Assume that we fold N1′ times in each z1 -z2 -
layer and by this embed ✷(b′1 , 1) × △n−1 (k1 π) × ✷n−1 (1/k1 ) into P1 . The
maximal ellipsoid fiber over P2 will then be
                                                     
                           b′1     n−1            n−1    1
                        1−       △     (k1 π) × ✷            .
                            a                           k1

We want to deform this fiber to a fiber
                                                  
                         b′1     n−1 ′         n−1  1
                    1−         △    (k2 π) × ✷
                         a                          k2′

fitting into the minimal ball fiber △n−1 (A − 2A/l) × ✷n−1 (1) over P2 . We
thus define k2′ by (1 − b′1 /a)k2′ π = A − 2A/l. As we shall see below, the

                                       41
                           y3


                       1                  PSfrag replacements
                                                          C ′′
                                                 C′
                 N1 /k1
                                            C




                   1/k1                                     1/k2 , 1/k )
                                                          max(1/k 1   2
                                                               y2
                                1/k1        N1 /k1    1


                 Figure 24: Filling the y-factor of the fibers

appropriate ellipsoid fiber deformation can then be achieved in ✷n−1 (1) \
✷n−1 (1 − max(1/k1 , 1/k2 )).
   The optimal choice of N1′ and k2′ is the solution of the system
                     n                                         o 
       N1 = max N ∈
                         . 
                                N
                               N even, kN1 < 1 − max k11 , k12
                                          
                                                                  
                                                                    .
      k2 π =     A − 2A l     1 − b1 (N
                                     a
                                        )                         

By folding N1 times in each z1 -z2 -layer we fill nearly all of ✷n−1 (1−max(1/k1 , 1/k2 ))
and indeed stay away from ✷n−1 (1)\✷n−1 (1−max(1/k1 , 1/k2 )) (cf. Figure 24).
    The deformation of the ellipsoid fibres is achieved as follows: We first
move the cube C along all yi -directions, i ≥ 2, by 1 − max(1/k1 , 1/k2 ) −
(N1 − 1)/k1 − ǫ for some ǫ ∈ ]0, 1 − max(1/k1 , 1/k2 ) − N1 /k1 [. This can
be done whenever A/l > nπ. We then deform the translate C ′ to C ′′ .
This deformation is the restriction to (1 − b1 /a)△n−1 (k1 π) × ✷n−1 (1) of a
product of n − 1 two-dimensional symplectic maps αi which are explained
in Figure 25: On yi ≤ N1 /k1 , αi is the identity, and on yi ≥ 1 − 1/k2 − ǫ it
is an affine map with linear part
                                                    
                                          k2     k1
                          (xi , yi ) 7→      xi , y i .
                                          k1     k2
   Assume that we can choose A such that proceeding in this way, we
successively fill a large part of all the Pi , 1 ≤ i ≤ l − 1, and leave Pl

                                       42
                                               PSfrag replacements


                         yi
                                                                      C
                     1                                               C′
                                                  αi                 C ′′
                  1/k2
                                                       αi     1/k1
                   ǫ                                   αi
               N1 /k1


                                                                        xi
                                             b1               b1
                                      (1 −   a )k2 π   (1 −   a )k1 π



                          Figure 25: Rescaling the fibers

untouched, i.e. the embedding process ends exactly when passing from Pl−1
to Pl (cf. Figure 23). The process is then described by the equations for the
pairs (Ni , ki+1 ), 1 ≤ i ≤ l − 2,
                        n                                                o 
        Ni = max N ∈          N    N even, N
                                           ki  <   1 −  max      1
                                                                    , 1
                                                                 ki ki+1
                                                                 
                                                                            
                                                                            
                               .        P i−1                               , (27)
                                            j=1 bj (Nj )+bi (N )
    ki+1 π =         A − (i+1)A
                            l         1 −           a
                                                                            
                                                                            

where bj (Nj ) is the x1 -length of the part embedded into Pj , and by

                   Nl−1 = max{n ∈      N | N even, N < kl−1}.
We finally observe that, in reality, the system (27) splits. Indeed, the second
line in (27) readily implies that ki < 2ki+1 whenever i ≤ l − 2. Thus, the
                                              N
first line in (27) reads Ni = max{N ∈ | N even, N/ki < 1 − 1/ki }, and
the embedding process is described by

                                  N
            Ni = max{N ∈ 2 | N < ki − 1}                                     (28.1)
                               ,     Pi             !
                       (i + 1)A          j=1 bj (Nj )
         ki+1 π =   A−              1−                                       (28.2)
                           l                a
          Nl−1 = max{N ∈ 2        N | N < kl−1 }.                            (28.3)

We now argue that such an A indeed exists, and that it is the minimal A
for which the above embedding process succeeds.
    Observe first that such a minimal A, which we denote by A0 , indeed
exists, for clearly, if A was chosen very large, the embedding process will

                                        43
end at some Pi with i < l − 1, and if A was chosen very small, it won’t
succeed at all.
    Suppose now that the embedding process for A0 ends before passing
from Pl−1 to Pl . Pick A′ < A0 and write ki and Ni respectively ki′ and Ni′
for the embedding parameters belonging to A0 respectively A′ . If A0 − A′
is small, k1 − k1′ is small too; thus, by (28.1), N1 = N1′ whenever A0 − A′
is small enough. But then, b1 (N1 ) − b′1 (N1 ) is small, whence (28.2) shows
that k2 − k2′ is small. Arguing by induction, we assume that Nj = Nj′ and
that bj (Nj ) − b′j (Nj ) and kj+1 − kj+1
                                       ′  are small for j ≤ i. Then, by (28.1) or
                                      ′
(28.3), and after choosing A0 − A even smaller if necessary, we may assume
                   ′ . If i + 2 ≤ l − 1, b               ′
that Ni+1 = Ni+1                          j+1 (Nj+1 ) − bj+1 (Nj+1 ) is then small
                                            ′
too, whence (28.2) shows that ki+2 − ki+2 is small.
    We hence may assume that all differences bi − b′i are arbitrarily small.
But then the embedding process for A′ will succeed as well, a contradiction.
   Recall that A0 = A0 (a, l) still depends on l. The best embedding result
provided by the above procedure is thus

                           s2n
                            EB (a) = min{A0 (a, l)}.
                                        l∈N
Set
                                        |E 2n (π, . . . , π, a)|
                        q(a, l) = 1 −
                                         |B 2n (A0 (a, l))|
and
                                      |E 2n (π, . . . , π, a)|
                         q(a) = 1 −                            .
                                        |B 2n (s2n
                                                EB (a))|

In order to prove the proposition, we have to show that

                                  lim q(a) = 0.                              (29)
                                 a→∞

    Given any a and l, the region in B 2n (A0 (a, l)) which is not covered by
the image of E 2n (π, . . . , π, a) is the disjoint union of four types of regions
Rh (a, l), 1 ≤ h ≤ 4.
      R1 (a, l) is the union of the “triangles” Ti (a, l) (see Figure 23).
      R2 (a, l) is the space needed for folding (see Figure 28).
      R3 (a, l) is the union of the space needed to deform the ellipsoid fibers
      and the space caused by the fact that the Ni have to be integers (see
      Figure 24).

                                         44
      R4 (a, l) is the image of the difference set of the embedded set and
      E 2n (π, . . . , π, a) (see Figure 26).

Detailed descriptions of these sets are given below.
   Let ǫ > 0 be small. We will find aǫ and lǫ such that

                        |Rh (a, lǫ )|
                        2n
                                          <ǫ        for all a ≥ aǫ ,           (30.h)
                      |B (A0 (a, lǫ ))|

1 ≤ h ≤ 4. Since the sets Rh (a, l) are disjoint and q(a) ≤ q(a, l), (30.h),
1 ≤ h ≤ 4, imply (29).
    Set Rh,i (a, l) = Rh (a, l) ∩ Pi (a, l). We first of all observe that the ratio
|R1 (a, l)|/|B 2n (A0 (a, l))| depends only on l and can be made arbitrarily small
by taking l large. We thus find l1 such that

                     |R1 (a, l)|
                     2n
                                       <ǫ        for all a and l ≥ l1 .
                   |B (A0 (a, l))|

Moreover, notice that given ζ > 0 we can choose l1 such that for all a and
l ≥ l1

             |R1,i (a, l)|
                           <ζ    whenever i is not too near to l − 1.            (31)
              |Pi (a, l)|

Here and in the sequel, “i too near to l − 1” stands for “1 − i/(l − 1) smaller
than a constant which can be made arbitrarily small by taking first l and
then also a large”.
    Next, our construction clearly shows that given ζ as above and l being
fixed we may find a1 such that for a ≥ a1 and for all i ∈ {1, . . . , l − 1}

                    |R2,i (a, l)|                   |R3,i (a, l)|
                                  <ζ      and                     < ζ.           (32)
                     |Pi (a, l)|                     |Pi (a, l)|

In particular, given any lǫ ≥ l1 , we find aǫ such that (30.1), (30.2) and (30.3)
hold true.
    Recall that the embedding ϕa,l : E 2n (π, . . . , π, a) ֒→ B 2n (A0 (a, l)) is de-
fined on a larger domain with piecewise constant fibres. Set

                     Xi (a, l) = ϕ−1
                                  a,l (Pi (a, l)),
                     Yi (a, l) = Xi (a, l) \ E 2n (π, . . . , π, a),
                     Zi (a, l) = Xi (a, l) ∩ E 2n (π, . . . , π, a)


                                            45
     π




                                                              PSfrag replacements

                                                                                a


                             Figure 26: Y (a, 8) ⊂ X(a, 8)

                    `                               `l−1                        `l−1
and X(a, l) = l−1     i=1 Xi (a, l), Y (a, l) =        i=1 Yi (a, l), Z(a, l) =  i=1 Zi (a, l)
(cf. Figure 26), and recall that we denoted the u-width of Xi (a, l) by bi (a, l).
Assume now that ζ is small. Then (31) and (32) show that for a ≥ aǫ
and i not too near to lǫ − 1, |Xi (a, lǫ )|/|Pi (a, lǫ )| is near to 1. Thus, a
simple volume comparison shows that if lǫ is large, bi (a, lǫ )/a and hence also
|R4,i (a, lǫ )|/|Pi (a, lǫ )| = |Yi (a, lǫ )|/|Pi (a, lǫ )| is small for these a and i. In
particular, we may choose lǫ and aǫ such that (30.4) holds true too.
    This completes the proof of Proposition 3.13. For later purposes, we
state that given ζ > 0, we may find l0 and a0 such that for all a ≥ a0 and i
not too near to l0 − 1
                           |Rh,i (a, l0 )|
                                           < ζ,      1 ≤ h ≤ 4.                         (33)
                            |Pi (a, l0 )|
                                                                                           ✷


   The above proof gives no information about the convergence speed in
(29). The remainder of this paragraph is devoted to the proof of

Proposition 3.14 Given ǫ > 0 there is a constant C(n, ǫ) such that for all
a
                           |E 2n (π, . . . , π, a)|              1
                      1−        2n   2n             < C(n, ǫ)a− 2n +ǫ .
                             |B (sEB (a))|

Proof. The proposition follows from the existence of a pair (a0 , l0 ) such
                                                      N
that for a ∈ Ik (a0 ) = [4kn a0 , 4(k+1)n a0 [, k ∈ 0 ,

                          (2 − ǫ)q(4n a, 2k+1 l0 ) < q(a, 2k l0 ).                      (34)


                                             46
                                                          1
Indeed, choose C(n, ǫ) so large that C(n, ǫ)a− 2n +ǫ > q(a) for a < a0 and
                                 1
                     C(n, ǫ)a− 2n > q(a, l0 )        for a ∈ I0 (a0 ).                  (35)

Then, if a ∈ Ik (a0 ) for some k ∈      N,
                                 (34)                  a       
         q(a) ≤ q(a, 2k l0 )     <      (2 − ǫ)−k q        , l0
                                                       4kn
                                 (35)                                    1
                                 <      (2 − ǫ)−k C(n, ǫ)2k a−ǫ a− 2n +ǫ
                                                                                 1
                                 ≤      (2 − ǫ)−k C(n, ǫ)2k 4−ǫkn a−ǫ
                                                                   0 a
                                                                      − 2n +ǫ
                                                                             1
                                 <      (2 − ǫ)−k 2k 4−ǫkn C(n, ǫ)a− 2n +ǫ
                                                      1
                                 <      C(n, ǫ)a− 2n +ǫ .

    So let’s prove (34). Fix (a0 , l0 ) and â ∈ I0 (a0 ) and set ak = 4kn a0 ,
âk = 4kn â, lk = 2k l0 and
                                        A0 (âk+1 , lk+1 )
                                ρk =                       ,
                                          A0 (âk , lk )

     N
k ∈ 0 . Given a specified subset S(a, l) of B 2n (A0 (a, l)) and a parameter
p(a, l) belonging to the embedding ϕa,l : E 2n (π, . . . , π, a) ֒→ B 2n (A0 (a, l)),
we write k S and k p instead of S(âk , lk ) and p(âk , lk ). Moreover, we write k S ′
for the rescaled subset ρ1k S(âk+1 , lk+1 ) of ρ1k B 2n (A0 (âk+1 , lk+1 )) and k p′ for
the parameter belonging to the rescaled embedding ρ1k E 2n (π, . . . , π, âk+1 ) ֒→
 1   2n                                                  ′       ′
ρk B (A0 (âk+1 , lk+1 )). Finally, write ρ, S, S , p, p instead of ρ0 , 0 S,
   ′          ′              2n                 ′      1 2n
0 S , 0 p, 0 p , set E = E (π, . . . , π, â), E = ρ E (π, . . . , π, â1 ) and B =
B 2n (A0 (â, l0 )), and observe that B = B ′ .
                                                                             N
     We claim that we can find (a0 , l0 ) such that for all k ∈ 0 , âk ∈ Ik (a0 )
and i not too near to lk − 1
                                       ′
                            (4 − ǫ)|k Rh,2i(−1) | < |k Rh,i |,                       (36.h.k)

1 ≤ h ≤ 4. We will first prove (36.h.0) and will then check that the con-
ditions valid for (â, l0 ) which allowed us to conclude (36.h.0) are also valid
for (âk , lk ) provided that (36.h.m) holds true for m ≤ k − 1. Arguing by
induction, we thus see that (36.h.k) holds true for all k ∈ 0 .          N
                                                          N
    Set ǫ1 = ǫ/16 and observe that for all k ∈ 0 and i not too near to lk − 1
                                                       
                            ′            ′        1
                        |k P2i−1 | > |k P2i | >     − ǫ1 |k Pi |.           (37)
                                                  2


                                             47
We conclude that for k ∈            N0, âk ∈ Ik (a0 ) and i not too near to lk − 1
                     
                 3ǫ       |Rh,2i(−1) (âk+1 , lk+1 )|   |Rh,i (âk , lk )|
              2−                                      <                    ,    (38.h.k)
                 4         |P2i(−1) (âk+1 , lk+1 )|     |Pi (âk , lk )|

1 ≤ h ≤ 4. In particular, there is (a0 , l0 ) such that for all â ∈ I0 (a0 ),

                             |Rh (âk+1 , lk+1 )|        |Rh (âk , lk )|
               (2 − ǫ)                                < 2n                  ,
                          |B 2n (A0 (âk+1 , lk+1 ))|  |B (A0 (âk , lk ))|

1 ≤ h ≤ 4. Since Rh (a, l) are disjoint, this implies (34).
                           `
    (R1) Let R1 (a, l) = li=1 Ti (a, l) be the union of the “triangles” Ti (a, l) ⊂
B 2n (A0 (a, l)) (see Figure 23). R1,2i(−1)
                                   ′                                            ′
                                            is a subset of R1,i , and |R1,i |/|R1,2i(−1) |=



                              T1′
                                     T2′
                  T1


                                                    PSfrag replacements


                                                              T2l′ 0 −1
                                                                   T2l′ 0

                                                        Tl0


                                    Figure 27: R1 and R1′
        ′
|Ti |/|T2i(−1)                                                            ′
               | depends only on l0 (see Figure 27). Clearly, 4 − |Ti |/|T2i(−1) | is
small if |Ti |/|Pi | is small enough. By taking l0 large, we may make |Ti |/|Pi |
arbitrarily small for i not too near to l0 − 1. Thus, (36.1.0) holds true
whenever l0 is large enough. Observe finally that (36.1.0) implies (36.1.k),
k∈ .N
    (R2) Recall that the x1 -length of the space needed for folding equals the
fiber capacity at the place where we fold. The staircases needed for folding

                                               48
                                 `
are thus contained in R2 (a, l) = l−1
                                   i=1 R2,i (a, l), where R2,i (a, l) equals
            (                   Pi−1 !                              Pi−1 !)
              (i − 1)A           j=1 bj              iA                j=1 bj
Qi (a, l) \            +π 1−              < x1 <        −π 1−                 .
                  l               a                   l                 a

Here, we put

                              Qi (a, l) = Pi (a, l) \ Ti (a, l).




                                                PSfrag replacements


                                                                      x1
                        A0     A0
                        2l0    l0



                                 Figure 28: R2 and R2′

     Observe that for i not too near to lk − 1, |k Q′2i−1 ∩ k Ti |/|k Q′2i−1 | → 0
as lk → ∞ (cf. Figure 27). Hence, also |k R2,2i−1      ′                  ′
                                                             ∩ k Ti |/|k R2,2i−1 | → 0 as
                                        ′
lk → ∞. We may thus neglect k R2,2i−1 ∩ k Ti and prove (36.2.k) with k R2,2i−1       ′
                    ′
replaced by k R2,2i−1 \ k Ti (which we denote again by k R2,2i−1 ).  ′
               P                             Pi−1 ′
     If ui = i−1  j=1 bj respectively ui =
                                          ′
                                                 j=1 bj is the x1 -coordinate at which
the   image    of E   respectively E ′ enters P , then the volume embedded into
`i−1                                              i
   j=1  Pj  is
                                                      "                     n #
     π n−1  n               n
                                             π n−1      â1 n       â1     ′
              â − (â − ui )      resp.                       −         − ui       , (39)
   ân−1 n!                                 â1n−1 n!     ρ            ρ


                                             49
and the fiber capacity at ui respectively u′i is
                                                                                
                      π                                        π        â1
                  ci = (â − ui )         resp.       c′i   =                  ′
                                                                            − ui .           (40)
                      â                                      â1        ρ
Thus, c1 = ρc′1 . We claim that

    ci > (1 − ǫ1 )ρc′2i(−1)
             whenever â is large enough and i is not too near to l0 − 1.
                                                                         (41)

Since c′2i−1 > c′2i , it suffices to show that

   ci > (1 − ǫ1 )ρc′2i−1       for â large enough and i not too near to l0 − 1.
                                                                              (41’)

So assume that there is an i violating the inequality in (41’) and set

                 i0 = min{1 ≤ i ≤ l0 − 1 | ci ≤ (1 − ǫ1 )ρc′2i−1 }.

Let ζ > 0 be so small that

                                            ζ < ǫ1                                           (42)

and set
                           |Zi (a, l)|                                   |Z(a, l)|
             zi (a, l) =                  and        z(a, l) =                           .
                           |Pi (a, l)|                              |B 2n (A0 (a, l))|
By the definition of ρ, z and z ′ ,
                                                     z
                                         ρn = 4n        .                                    (43)
                                                     z′
By (33), for any large enough l0 there is a0 such that for all â ∈ I0 (a0 ) and
i not too near to l0 − 1

                                         zi > 1 − ζ.                                         (44)

We have seen in (R1 ) that for all i ∈ {1, . . . , l0 }
                                      ′
                                    |R1,2i(−1) | < |R1,i |.                                  (45)

Moreover, if ζ is small enough, we clearly have that for i not too near to
l0 − 1

                                         ci > c′2i(−1) .                                     (46)


                                                50
This implies that for these i
                                     ′
                                   |R2,2i(−1) | < |R2,i |.                         (47)

We now assume that a0 is so large compared to l0 that

                                   A0 (a0 , l0 ) > 12l0 π.                         (48)

Then, A0 (â, l0 ) > 12l0 π > 12l0 ci , i.e.

                        A0 (â, l0 )
                                     > 12ci ,            1 ≤ i ≤ l0 − 1.           (49)
                            l0

                                        ′
                                      |R3,2i(−1) | < |R3,i |                       (50)

now follows from (46) in the same way as (73) will follow from (41). Finally,
for ζ small enough and i not too near to l0 − 1 we clearly have that
                                     ′
                                   |R4,2i(−1) | < |R4,i |.                         (51)

We conclude from (45), (47), (50) and (51) and (37) that
                         ′
                       |Rh,2i(−1) |          |Rh,i |
                           ′            <3           ,          1 ≤ h ≤ 4.
                         |P2i(−1) |           |Pi |

This shows that

                                         zi′ > 1 − 3ζ.                             (52)

Set
                            `i−1                                    `i−1   ′
                             j=1 Zj                        ′          j=1 Zj
                  z<i =     `i−1              and         z<i   = `i−1     ′
                                                                               .   (53)
                             j=1 Pj                                   j=1 Pj


By (44) and (52), we may assume that for all i ∈ {1, . . . , l0 − 1}

                                                           ′
                         z<i > 1 − ζ          and         z<i > 1 − 3ζ.

In particular,

                            z >1−ζ            and         z ′ > 1 − 3ζ             (54)


                                                51
and
                                                     ′
                      z<i0 > 1 − ζ           and    z<i 0
                                                          > 1 − 3ζ.                   (55)
                                                        `i0 −1
Comparing the two volumes embedded into                Pj , we get from (39) that
                                                          j=1
                                                "                           n #
          n−1                             n−1         n
  ′     π                                 π       â 1         â 1
 z<i 0 n−1
                ân − (â − ui0 )n = z<i0 n−1              −        − u′2i0 −1     .
      â    n!                           â1 n!    ρ            ρ
                                                                                      (56)

By (40), ci0 ≤ (1 − ǫ1 )ρc′2i0 −1 translates to

                                         4n
                         u′2i0 −1 ≤             (ui − ǫ1 â).                         (57)
                                      (1 − ǫ1 )ρ 0
Plugging (57) into (56), we find
            n         !                      n       !
            4        ′     n             4
      z<i0        − z<i0 â ≥ z<i0                    ′
                                                   − z<i0 (â − ui0 )n ,
            ρ                        ρ(1 − ǫ1 )

and using (43) and dividing by z<i0 we get
             ′    ′           ′             ′    
             z    z<i       n   z     1       z<i
                −     0
                          â ≥              −     0
                                                      (â − ui0 )n .                  (58)
              z   z<i0          z (1 − ǫ1 )n z<i0

By (54) and (55), |1 − z ′ /z| and |1 − z<i
                                         ′ /z
                                            0 <i0 | can be made arbitrarily small
by taking ζ small. (58) thus shows that for ζ small enough, 1 − ui0 /â must
be small, i.e. i0 must be near to l0 − 1. This concludes the proof of (41’).
   Putting everything together, we see that l0 and a0 may be chosen such
that for i not too near to l0 − 1
          (41)
                         ′
                                        (43),(54)             p          ′
   |R2,i | > (1 − ǫ1 )ρ|R2,2i(−1) |          >      (1 − ǫ1 )4 n 1 − ζ |R2,2i(−1) |
                                             (42)
                                              >     4(1 − ǫ1 )2 |R2,2i(−1)
                                                                  ′
                                                                           |
                                                             ′
                                             >      (4 − ǫ)|R2,2i(−1) |.

This proves (36.2.0).
   Suppose now that (36.h.m), 1 ≤ h ≤ 4, and hence also (38.h.m) hold
true for m ≤ k − 1. (38.h.m) and (44) imply that for i not too near to lk − 1

                                      k zi   > 1 − ζ.                                 (59)


                                              52
The reasoning which implied (46) thus also shows that for i as in (46)

                                     k c2k i    > k c′2k−1 i .                 (60)

Since l0 is large and ζ is small, k c2k−1 i − k c2k i is small. We thus see that for
i not too near to l0 − 1

                                      k ci    > k c′2i(−1)                     (61)

almost holds true, and hence also
                                   ′
                               |k R2,2i(−1) || < |k R2,i |                     (62)

almost holds true. Next, observe that (44) and (59) imply that A0 (ak , lk )/A0 (a0 , l0 )
is near to 4k . This and (48) show that

                                   A0 (ak , lk ) > 12lk π,                     (63)

and in the same way as we derived (50) from (46) and (49) we may derive
from (61) and (63) that
                                   ′
                               |k R3,2i(−1) | < |k R3,i |                      (64)

almost holds true. Finally, by (59), we also have that for i not too near to
lk − 1
                                   ′
                               |k R4,2i(−1) | < |k R4,i | .                    (65)

We infer from (37), (62), (64) and (65) that
                        ′
                    |k Rh,2i(−1) |            |k Rh,i |
                         ′           <3                 ,        1 ≤ h ≤ 4,
                     |k P2i(−1) |              |k Pi |

i.e.
                                          ′
                                       k yi    > 1 − 3ζ.

Proceeding exactly as in the case k = 0 we thus get that for i not too near
to lk − 1

                            k ci    > (1 − ǫ1 )ρk k c′2i(−1) ,                 (66)

from which (36.2.k) follows in the same way as for k = 0.

                                                 53
   (R3) Set

                             Di (a, l) = ✷n−1 (1) \ ✷n−1 (Ni ki )

and
              # i−1                i
                                                  "                    Pi−1                 !
                X                  X                                     j=1 bj (a, l)
Wi (a, l) =           bj (a, l),         bj (a, l) ×]0, 1[× 1 −                                 △n−1 (π),
                                                                               a
               j=1                 j=1
                                                                                                      (67)

1 ≤ i ≤ l − 1. Moreover, let Ci be the cube in the y-factor of the fibers
which will be deformed and let Ki be the extra space in Pi needed to move
Ci along the yj -directions, j ≥ 2. Then,

                                          l−1
                                                           !                   l−2
                                          a                                    a
               R3 (a, l) = ϕa,l                 Wi (a, l)      × Di (a, l) ∪         Ki .
                                          i=1                                  i=1

    We first of all observe that Ki ⊂ ϕa,l (Wi (a, l))×Ci and that |Ci |/|Di (a, l)|
is small for i not too near to l − 1 and a large, since then ki (a, l) is large.
We thus may forget about the Ki . Next, as in (R2 ), notice that for i not
too near to lk − 1,
                     ′                   ′
                 |k R3,2i−1 ∩ k Ti |/|k R3,2i−1 |→0                  as lk → ∞,

                             ′
whence we may neglect k R3,2i−1                                      ′
                                   ∩ k Ti and prove (36.3.k) with k R3,2i−1 re-
             ′                                           ′
placed by k R3,2i−1 \ k Ti (which we denote again by k R3,2i−1 ).
    By (28.1),
                         
                             ki − 2,        (ki even)
         Ni (a, l) =                                             for 1 ≤ i ≤ l − 2.                   (68)
                             ki − 3,        (ki odd)

This and Figure 24 show that for these i,
                                                                           
          3                   Ni (a, l)                               Ni (a, l)
  1−              (n − 1) 1 −             < |Di (a, l)| < (n − 1) 1 −             .
      ki (a, l)               ki (a, l)                               ki (a, l)
                                                                              (69)

Observe now that ci ki = c′2i k2i
                               ′ < c′     ′
                                    2i−1 k2i−1 . Hence, by (41),

                                      ′
                                     k2i(−1) > (1 − ǫ1 )ρki                                           (70)


                                                      54
if i is not too near to l0 − 1. (68) and (70) imply that for these i

                                1 − Ni /ki        2
                                  ′       ′      > (1 − ǫ1 )ρ.                         (71)
                            1 − N2i(−1) /k2i(−1)  3

Using again that for i not too near to l − 1, ki (a, l) is large whenever a is
large, we conclude from (69) and (71) that for a0 large enough and i not too
near to l0 − 1,

                                      |Di |     2
                                       ′       > (1 − 2ǫ1 )ρ.                          (72)
                                    |D2i(−1) |  3

    We conclude that for such a0 and i

          ′
                        (49),(72)       52               10
|R3,i |/|R3,2i(−1) |       >        2      (1 − 2ǫ1 )ρ >    (1 − 2ǫ1 )4(1 − ǫ1 ) > 4 − ǫ.
                                        63                9
                                                                                     (73)

This proves (36.3.0).
   Suppose again that (36.h.m), 1 ≤ h ≤ 4, holds true for m ≤ k − 1. Then
(66) implies
                                       ′
                                    k k2i(−1)   > (1 − ǫ1 )ρk k ki

if i is not too near to lk − 1, and proceeding as before we obtain (36.3.k).

    (R4) Recall that R4 (a, l) = ϕa,l (Y (a, l)) (cf. Figure 26).
                             `l−1            2n
    To any partition Z̄ = `     i=1 Z̄i of E (π, . . . , π, ā) looking as in Figure 26
associate the set X(Z̄) = Xi (Z̄) which is obtained from Z̄ by replacing
each fiber in Z̄i by the`maximal fiber in Z̄i (see Figure 26). Set Yi (Z̄) =
X  (Z̄)\ Z̄i and Y (Z̄) = Yi (Z̄).`
`il−1                                  Clearly, if the partitions E 2n (π, . . . , π, ā) =
                 2n            ¯        l−1 ¯
  i=1 Z̄i and E (π, . . . , π, ā) =    i=1 Z̄ i are similar to each other, then

                                         |Yi (Z̄)|   |Yi (Z̄¯)|
                                                   =            .                      (74)
                                           |Z̄i |      |Z̄¯i |
                  `l
Let B 2n (Ā) =        i=1 P̄i   be a partition as in Figure 28 and assume that

              |Z̄i |                            |Z̄¯i |
                     >1−ζ               and             >1−ζ        for 1 ≤ i ≤ i0 .
              |P̄i |                            |P̄i |

                                                   55
Clearly, if ζ is small enough and i0 is large enough, Z̄ and Z̄¯ are almost
similar. (74) thus shows that given i1 not too large we may find ζ and i0
such that for i ≤ i1

                              |Yi (Z̄)|            |Yi (Z̄¯)|
                                        < (1 + ǫ1 ) ¯ .                              (75)
                                |Z̄i |               |Z̄ i |

    Given âm ∈ Im (a0 ), m ∈     N0, and 1 ≤ i ≤ l0 − 1, set
                                            2m
                                            ai
                         Zi (âm ) =                  Zj (âm , lm ),
                                       j=2m (i−1)+1

            `                           `2m i
Z(âm ) = Zi (âm ), P (Zi (âm )) = j=2      m (i−1)+1 P (Zj (âm , lm )) and z(Zi (âm )) =

|Zi (âm )|/|P (Zi (âm ))|. For a0 large and i as above we clearly have that for
         N
all m ∈ 0 and âm ∈ Im (a0 )
                    `2m i
                       j=2m (i−1)+1    Y (Zj (âm , lm ))       |Yi (â, l0 )|
                                                            ≤                  .     (76)
                              |P (Zi (âm ))|                   |Pi (â, l0 )|

Assume now that for some m, i not too near to l0 − 1 and 2m (i − 1) + 1 ≤
j ≤ 2m

              Rh,j (âm , lm )       1    |Rh,i (â, l0 )|
                               ≤        m
                                                           ,            1 ≤ h ≤ 3.   (77)
               Pj (âm , lm )    (2 − ǫ) |Pi (â, l0 )|

(76) and (77) in particular imply that for these i

                                    z(Zi (âm )) ≥ zi .                              (78)

(78) and (75) imply that l0 and a0 may be chosen such that for all âm , âm′
satisfying (77) and i not too near to l0 − 1

                       |Yi (Z(âm ))|             |Yi (Z(âm′ ))|
                                      < (1 + ǫ1 )                 .                  (79)
                         |Zi (âm )|                |Zi (âm′ )|

Suppose now that (36.h.m), 1 ≤ h ≤ 4, holds true for m ≤ k − 1. We then
have shown in (Rh ), 1 ≤ h ≤ 3, that (77) holds true for m ≤ k + 1. (79)
thus implies that for i not too near to l0 − 1

                       |Yi (Z(âk+1 ))|             |Yi (Z(âk ))|
                                        < (1 + ǫ1 )                ,
                         |Zi (âk+1 )|                |Zi (âk )|

                                              56
and (78) with m = k now shows that for these i
                       |Yi (Z(âk+1 ))|    1 + ǫ1 |Yi (Z(âk ))|
                                         <                       .                 (80)
                       |P (Zi (âk+1 ))|   1 − ζ |P (Zi (âk ))|
Pick ǫ2 so small that
                                ǫ  1 + ǫ2 1 + ǫ1
                              1−                   < 1.                            (81)
                                 4 1 − ǫ2 1 − ζ
This is possible since
                        ǫ  1 + ǫ1 (42)  ǫ  1 + ǫ1
                      1−             < 1−             < 1.
                         4 1−ζ             4 1 − ǫ1
We will show that l0 and a0 can be chosen such that for any âm satisfying
(78), i not too near to l0 − 1 and 2m (i − 1) + 1 ≤ j ≤ 2m i

      (1 − ǫ2 )|Y (Zi (âm ))| < 4m |Yj (âm , lm )| < (1 + ǫ2 )|Y (Zi (âm ))|.   (82)

The second inequality in (82) with m = k + 1, (80), the first inequality in
(82) with m = k and (81) then imply (36.4.k).
   In order to prove (82), pick some small ζ0 = ζ and assume l0 and a0 to
be so large that for all â ∈ I0 (a0 ), zi (â, l0 ) > 1 − ζ0 whenever i is not too
                            ¯ for any a ≥ a1 which satisfies (78). Then
near to l0 − 1. Write ā or ā

                                  z(Zi (ā)) > 1 − ζ0                              (83)

if i is not too near to l0 − 1. Fix once and for all such an i. Given
                       N
âm ∈ Im (a0 ), m ∈ , which satisfies (78), set d = u2m i − u2m (i−1) , uM =
u2m (i−1) + d/2 and δ = u2m (i−1)+2m−1 − uM , and write Z0 = Zi (âm ),
       `2m (i−1)+2m−1
Z1 = j=2m (i−1)+1 Zj (âm , lm ) and Z2 = Z0 \ Z1 . Also write Xj = X(Zj ),
Yj = Y (Zj ) and Pj = P (Zj ), j = 0, 1, 2 (see Figure 29). Finally, define
Rh (Zj ), 1 ≤ h ≤ 4, in the obvious way.
    Define α, β and γ1 by
                            |X1 |           d/2 + δ
                                  = (1 + α)         ,                              (84)
                            |X2 |           d/2 − δ

                       |Xj | ≤ (1 + β)|Zj |,        j = 1, 2,                      (85)

and

                                 |P1 | = (1 + γ1 )|P2 |.                           (86)


                                            57
                                                                  PSfrag replacements
                                     Y1              Y0
                                                          Y2


                            Z1                       Z2



                                            δ
                                                                                    u
              u2m (i−1)                   uM u2m (i−1)+2m−1 u2m i


                                      Figure 29: X0

We assume that β is chosen minimal, and we observe that γ1 is independent
of âm and is small since i is not too near to l0 − 1 and l0 is large. By (83),
|Z0 | > (1 − ζ0 )|P0 |. This and (86) readily imply that
                     |Zj | > (1 − (2 + γ1 )ζ0 )|Pj |,          j = 1, 2.           (87)
Thus, since γ1 < 1,
                     d/2 + δ   (84)   |X1 | (85)     |Z1 |
          (1 + α)                =           ≥
                     d/2 − δ          |X2 |      (1 + β)|Z2 |
                                                                                   (88)
                                            (87) (1 − 3ζ0 )|P1 |   1 − 3ζ0
                                             >                   >
                                                  (1 + β)|P2 |      1+β
and
         d/2 + δ   |X1 | (85) (1 + β)|Z1 |
                 <        ≤
         d/2 − δ   |X2 |          |Z2 |
                                                                                   (89)
                         (87) (1 + β)|P1 | (86) (1 + β)(1 + γ1 )
                          <                   =                  .
                              (1 − 3ζ0 )|P2 |       1 − 3ζ0
If δ < 0, by (88),
            d(α + β + αβ + 3ζ0 ) > |δ|(4 + 2α + 2β + 2αβ − 6ζ0 ),

and if δ ≥ 0, by (89),

           d(γ1 + β + γ1 β + 3ζ0 ) > δ(4 + 2γ1 + 2β + 2γ1 β − 6ζ0 ).


                                                58
Set µ = max(α, γ1 ). Then

                                        d
                                |δ| <     (µ + β + 3ζ0 )                           (90)
                                        2
if ζ0 , β and µ are small enough.
     Set c = âm − u2m (i−1) . Observe that, by (83), if ζ0 is small, d(ā)/d(ā ¯)
             ¯                  ¯             ¯
and c(ā)/c(ā) are near to ā/ā for all ā, ā. Hence, d(ā)/c(ā) is essentially
independent of ā. Let ν1 be such that d(ā)/c(ā) ≤ ν1 for all ā. Since c(ā)
is large for i not too near to l0 − 1 and since l0 is also large, ν1 is small.
Moreover, we readily compute
                                                
                               n − 1 d + 2δ     d
                            α=              +o                                     (91)
                                 2      c       c

and
                                                
                               n − 1 d + 2δ     d
                            β=              +o     .                               (92)
                                 4      c       c

Thus, α and β are dominated by ν1 , i.e. there are small constants α1 and
β1 such that α ≤ α1 and β ≤ β1 for all ā. Set µ1 = max(α1 , γ1 ).
     Next, notice that |Y1 |/|P1 | and |Y2 |/|P2 | are essentially half as large as
|Y0 |/|P0 | and hence also about half as large as |Yi (â, l0 )|/|Pi (â, l0 )|. Indeed,
                                       "              !#
                        1     πc n−1      c       d n
             |Y0 | =                     d−   1− 1−        ,
                     (n − 1)! âm           n       c

and |Y1 | respectively |Y2 | are obtained from this expression by replacing d
by d/2 + δ respectively c by c − (d/2 + δ) and d by d/2 − δ. This yields
                                                            
                |Y1 | 1          1 n−2d        δ        d     δ
                     −       =             +4 +o          +o
                |Y0 | 4          4 6 c         d        c     d
                                                                                   (93)
                            (90) n
                             <     ν1 + µ1 + β1 + 3ζ0 ,
                                 2
and since ν1 is small, it turns out that the same estimate also holds true for
|Y2 |/|Y0 |. Moreover, (86) implies that

                            |Pj |     1
                                  ≥        ,       j = 1, 2.                       (94)
                            |P0 |   2 + γ1

                                           59
If ζ0 , β1 , µ1 and ν1 and also ǫ are small enough, we hence get
               |Yj |   (93),(94)   3 |Y0 | (79) 3             |Yi (â, l0 )|
                          <                 <     (1 + ǫ1 )
               |Pj |               5 |P0 |      5             |Zi (â, l0 )|
                                                3 1 + ǫ1 |Yi (â, l0 )|
                                            <                                        (95)
                                                5 1 − ζ0 |Pi (â, l0 )|
                                                2 |Yi (â, l0 )|
                                            <                    ,       j = 1, 2.
                                                3 |Pi (â, l0 )|
We conclude that for j = 1, 2
                                                       P4
                           |Zj |                        h=1 |Rh (Zj )|
                  z(Zj ) =       = 1−
                           |Pj |                             |Pj |
                                                       P3
                                                        h=1 |Rh (Zj )|   + |Yj |
                                         > 1−
                                                                 |Pj |               (96)
                                                            P4
                                                2            h=1 |Rh,i (â, l0 )|
                                      (77),(95)
                                         >   1−
                                                3             |Pi (â, l0 )|
                                              2
                                         > 1 − ζ0 .
                                              3
In particular, ζ0 in (83) may be replaced by ζ1 = 23 ζ0 .
    We conclude that l0 and a0 may be chosen such that for all âm

               (1 − L1 )|Y0 | < 4|Yj | < (1 + L1 )|Y0 |,                 j = 1, 2.   (97)

Here, we put

               L1 = L(ζ1 , β1 , µ1 , ν1 ) = 4(µ1 + β1 + 3ζ1 ) + 2nν1 .

Observe that L is linear in ζ1 , β1 , µ1 and ν1 .
                                                                 `
    Assume now that m ≥ 2 and consider the partition Z1 = Z12 Z22 whose
components consist of 2m−2 consecutive components of Z(âm , lm ). Set d′ =
d/2 + δ and define δ′ to be the difference of the u-width of Z12 and d′ /2. If
α′ is defined by

                               |X12 |             ′
                                              ′ d /2 + δ
                                                          ′
                                      = (1 + α )            ,
                               |X22 |            d′ /2 − δ′
we have
                                                     ′
                               ′  n − 1 d′ + 2δ′     d
                              α =                +o      .                           (98)
                                    2       c        c


                                                  60
Since ζ1 is small, δ′ /d′ is small. (91) and (98) thus show that α is near to
2α′ . In particular,
                                             2
                                      α′ <     α.                              (99)
                                             3
Similarly, if β ′ is the minimal constant with

                       |Xj2 | ≤ (1 + β ′ )|Zj2 |,     j = 1, 2,

we have
                            ′                             ′
               ′ n−1         d + 2δ′      d′ − 2δ′          d
             β =       max           ,       ′      ′
                                                        +o
                   4             c     c − d /2 − δ         c
                        ′     ′
                                     ′                                      (100)
                 n − 1 d + 2|δ |      d
               =                 +o       ,
                   4      c            c
and we conclude from (92) and (100) as above that
                                             2
                                      β′ <     β.                             (101)
                                             3
A similar but simpler calculation shows that γ ′ , which is defined by P (Z12 ) =
(1 + γ ′ )P (Z22 ), satisfies
                                             2
                                      γ′ <     γ1 .                           (102)
                                             3
Next, since δ′ /d is small, we also have that
                                         d′   2
                                            < ν1 .                             (103)
                                         c    3
                                            `
Consider now the partition Z2 = Z32 Z42 . While for Z1 we had c′ = c, now,
c′′ = âm − u2m (i−1)+2m−1 = c − d′ . But c′′ /c = 1 − d′ /c is near to 1, whence
the same arguments as above show (99), (101), (102) and (103) with α′ , β ′ ,
γ ′ and c′ replaced by α′′ , β ′′ , γ ′′ and c′′ . Finally, an argument analogous to
the one which proved (96) shows z(Zj2 ) > 1 − 32 ζ1 , 1 ≤ j ≤ 4. Summing up,
we have shown that there are constants ζ2 = 23 ζ1 , β2 , µ2 and ν2 independent
of âm such that L2 = L(ζ2 , β2 , µ2 , ν2 ) satisfies L2 < 32 L1 and such that for
all âm
                                 2
            (1 − L2 )|Yj | < 4|Y2j(−1) | < (1 + L2 )|Yj |,        j = 1, 2.

   In general, let Z k (âm ), 0 ≤ k ≤ m, be the partition of Z0 whose compo-
nents consist of 2m−k consecutive components of Z(âm , lm ). Applying the

                                           61
above arguments to the components of Z k (âm ), we see by finite induction
that there are constants Lk , 1 ≤ k ≤ m, with Lk+1 < 23 Lk such that for all
âm
                                        k+1
                (1 − Lk+1 )|Yjk | < 4|Y2j(−1) | < (1 + Lk+1 )|Yjk )|,

1 ≤ j ≤ 2k , 0 ≤ k ≤ m − 1. Hence, with
                                       ∞
                                       Y           k !
                                                   2
                           π± (x) =            1±     x
                                                   3
                                       k=1

we have that for all j ∈   {1, . . . , 2m }
                                   m
                                   Y
                π− (L1 )|Y0 | <        (1 − Lk )|Y0 |
                                   k=1
                               < 4m |Yjm |                              (104)
                                 Ym
                               <     (1 + Lk )|Y0 | < π+ (L1 )|Y0 |.
                                   k=1
Let l0 and a0 be so large that for i not too near to l0 − 1, L1 is so small
that 1 − ǫ2 < π− (L1 ) and π+ (L1 ) < 1 + ǫ2 . Then (104) implies (82). This
completes the proof of Proposition 3.14.                                  ✷


3.4   Lagrangian folding
As already mentioned at the beginning of this section, there is a Lagrangian
version of folding developed by Traynor in [31]. Here, the whole ellipsoid
or the whole polydisc is viewed as a Lagrangian product of a cube and a
simplex or a cube, and folding is then simply achieved by wrapping the
base cube around the base of the cotangent bundle of the torus via a linear
map. This version has thus a more algebraic flavour. However, it yields
good embeddings only for comparable shapes, while the best embeddings of
an ellipsoid into a polydisc respectively of a polydisc into an ellipsoid via
Lagrangian folding pack less than 1/n! respectively n!/nn of the volume.
   For the convenience of the reader we review the method briefly.
            R              R             R
Write again 2n (x, y) = n (x) × n (y) and set

                               (                            )
                                                               R
           ✷(a1 , . . . , an ) = {0 < xi < ai , 1 ≤ i ≤ n} ⊂ n (x),
                                                    n
                                                    X
          △(b1 , . . . , bn ) = 0 < y1 , . . . , yn
                                                      yi
                                                      bi
                                                                   R
                                                         < 1 ⊂ n (y)
                                                   i=1


                                              62
and
                                       Tn =     Rn (x)/πZn .
The embeddings are given by the compositions of maps
                                       α
                                   E
       E(a1 − ǫ, . . . , an − ǫ) −−→ ✷n (1) × △(a1 , . . . , an )
                                        β
                                       −
                                       →         ✷(q1 π, . . . , qn π) × △( qa11π , . . . , qannπ )
                                        γ
                                       −
                                       →         T n × △n ( A
                                                            π)
                                        δ
                                        E
                                       −→ B 2n (A)
respectively
                                  α  P
              P (a1 , . . . , an ) −−→ ✷n (1) × ✷(a1 , . . . , an )
                                   β
                                  −
                                  →         ✷(q1 π, . . . , qn π) × ✷( qa11π , . . . , qannπ )
                                   γ
                                  −
                                  →         T n × ✷n ( A
                                                       π)
                                  δP
                                 −→ C 2n (A),
where ǫ > 0 is arbitrarily small and the qi are of the form ki or 1/ki for some
ki ∈ .N
    αE and αP are the map (x1 , y1 , . . . , xn , yn ) 7→ (−y1 , x1 , . . . , −yn , xn ) fol-
lowed by the maps described at the beginning of section 3.3, and β is a
diagonal linear map:
                                                               
                                                  1         1
                   β = diag q1 π, . . . , qn π,      ,...,        .
                                                q1 π       qn π
Next, let
                                            
                                   n        A     n
                           δ̃E : ✷ (π) × △     ֒→ B 2n (A)
                                            π
and
                                            
                                   n        A     n
                           δ̃P : ✷ (π) × ✷     ֒→ C 2n (A)
                                            π
be given by
                                                   √                    √
          (x1 , . . . , xn , y1 , . . . , yn ) 7→ ( y1 cos 2x1 , . . . , yn cos 2xn ,
                                                   √                      √
                                                  − y1 sin 2x1 , . . . , − yn sin 2xn ).
Notice that δ̃E respectively δ̃P extend to an embedding of T n × △n (A/π)
respectively T n × ✷n (A/π). These extensions are the maps δE and δP . We
finally come to the folding map γ.

                                                   63
Lemma 3.15          (i) If the natural numbers k1 , . . . , kn−1 are relatively prime,
   then
                                                                                          
                                                     1                             − k11
                                               
                                                         1                    0   − k12   
                                                                                           
                                                                ..                 ..     
                      M (k1 , . . . , kn−1 ) =                       .              .     
                                                                                    1
                                                                                           
                                                         0                    1 − kn−1    
                                                                                    1

      embeds ✷(π/k1 , . . . , π/kn−1 , k1 . . . kn−1 π) into T n .

 (ii) For any k2 , . . . , kn ∈   N \ {1}
                                                                                                  
                                         1 − k12
                                       
                                           1            − k13                     0               
                                                                                                   
                                                        ..           ..                           
                                                           .              .                       
                N (k2 , . . . , kn ) = 
                                                                     ..
                                                                                                   
                                                                                                   
                                                                          . −k 1                  
                                                                              n−1                 
                                           0                                  1           − k1n   
                                                                                            1

      embeds ✷(π/(k2 . . . kn ), k2 π, . . . , kn π) into T n .

Proof. ad (i). Let M x = M x′ for x, x′ ∈ ✷(1/k1 , . . . , 1/kn−1 , k1 . . . kn−1 ),
so
                          xn        x′
                   xi −      = x′i − n + li ,                 1 ≤ i ≤ n−1                              (105)
                          ki         ki

for some li ∈   Z and
                                     xn = x′n + ln ,                                                   (106)

where ln ∈   Z satisfies |ln | < k1 . . . kn−1 .         Substituting (106) into (105) we
get

                                          ln
                        xi − x′i = li +      ,           1 ≤ i ≤ n − 1.                                (107)
                                          ki

If ln = 0, we conclude x = x′ . Otherwise, |xi − x′i | < 1/ki for 1 ≤ i ≤ n − 1
and (107) imply that ln is an integral multiple of all the ki , whence by the
assumption on the ki we have |ln | ≥ k1 . . . kn−1 , a contradiction.

                                             64
    ad (ii). Let N x = N x′ for x, x′ ∈ ✷(1/(k2 . . . kn ), k2 , . . . , kn ), so

                           xi+1        x′
                    xi −        = x′i − i+1 + li ,               1 ≤ i ≤ n−1                        (108)
                           ki+1        ki+1

for some li ∈     Z and
                                           xn = x′n + ln .                                          (109)

Substituting (109) into the last equation of (108) and resubstituting the
resulting equations successively into the preceding ones, we get

                        ln          ln−1          ln−2              l2
     x1 = x′1 +                +             +              + ··· +    + l1 .                       (110)
                    k2 . . . kn k2 . . . kn−1 k2 . . . kn−2         k2

Since |x1 − x′1 | < 1/(k2 . . . kn ), equation (110) has no solution for x1 6= x′1 ,
hence x1 = x′1 , and substituting this into (108) and using |xi − x′i | < ki , 2 ≤
i ≤ n, we successively find xi = x′i .                                           ✷

    The folding map γ can thus be taken to be M × M ∗ , where M is as in
(i) or (ii) of the lemma and M ∗ denotes the transpose of the inverse of M .

Remark 3.16 For polydiscs, the construction clearly commutes with tak-
ing products. For ellipsoids, a similar compatibility holds: Let M1∗ re-
spectively M2∗ be linear injections of △(a1 , . . . , am ) into △(a′1 , . . . , a′m ) re-
spectively △(b1 , . . . , bn ) into △(b′1 , . . . , b′n ). Then M1∗ ⊕ M2∗ clearly injects
△(a1 , . . . , am , b1 , . . . , bn ) into △(a′1 , . . . , a′m , b′1 , . . . , b′n ). Thus, given (possi-
bly trivial) Lagrangian foldings λ1 and λ2 which embed E(a1 , . . . , am ) into
E(a′1 , . . . , a′m ) and E(b1 , . . . , bn ) into E(b′1 , . . . , b′n ), the Lagrangian folding
λ1 ⊕ λ2 embeds E(a1 , . . . , am , b1 , . . . , bn ) into E(a′1 , . . . , a′m , b′1 , . . . , b′n ). ✸


    In the following statements, ǫ denotes any positive number.

Proposition 3.17              (i) Let k1 < · · · < kn−1 be relatively prime and a > 0.
    Then

       (i)E E 2n (π, . . . , π, a) ֒→ B 2n (max{(kn−1 + 1)π, k1 ···akn−1 } + ǫ)
                                                                                        a
       (i)P P 2n (π, . . . , π, a) ֒→ C 2n (max{kn−1 π, (n − 1)π +                k1 ··· kn−1 }).

 (ii) Let n ≥ 3, k2 , . . . , kn ∈      N \ {1} and a2, . . . , an > 0. Then
                                                  65
      (ii)E E(π, a2 , . . . , an ) ֒→ B 2n (A+ǫ), where A is found as follows: Mul-
            tiply the first column of N ∗ by k2 · · · kn and the i th column by
            (ai /π)/ki , 2 ≤ i ≤ n. Then add to every row its smallest entry
            and add up the entries of each column. A/π is the maximum of
            these sums.
      (ii)P P (π, a2 , . . . , an ) ֒→ P (A1 , . . . , An ), where the Ai are found as
            follows: Multiply N ∗ as in (ii)E . Ai /π is the sum of the absolute
            values of the entries of the i th row.

Proof. ad (i). Write y ′ = M ∗ (k1 , . . . , kn−1 )y. We have
                                                                                               
                                                      1
                                                               1                  0             
                                                                                                
                                                                    ..                          
                   M ∗ (k1 , . . . , kn−1 ) =                            .                      .
                                                                                                
                                                                                  1             
                                                      1         1                   1
                                                      k1        k2   ...          kn−1          1

Thus, given y ∈ △(k1 , . . . , kn−1 , k1 ···a/π
                                             kn−1 ),

                           y1                          yn−1    a/π                                             yn
y1′ + · · · + yn′ = (k1 + 1)  + · · · + (kn−1 + 1)          +                                                 a/π
                           k1                         kn−1 k1 · · · kn−1
                                                                                                          k1 ··· kn−1
                                                  
                                         a/π
                   < max kn−1 + 1,                   ,
                                     k1 · · · kn−1

and given y ∈ ✷(k1 , . . . , kn−1 , k1 ···a/π
                                           kn−1 ),


                                                                         a/π
                       y ′ ∈ ✷(k1 , . . . , kn−1 , n − 1 +                         ).
                                                                     k1 · · · kn−1

ad (ii). We have
                                                                                                        
                                        1
                                  k1                      1                                        0     
                                           2                                                             
                                                                     ..                                  
                                  − k 1k                  1                  .                           
                                                          k3                                             
        N ∗ (k2 , . . . , kn ) = 
                                            2 3
                                          ..               ..         ..               ..                 .
                                          .                .                 .             .             
                                                                                                         
                                  (−1)n−1            (−1)n−2
                                                                      ...               1
                                                                                                    1     
                                  k2 ··· kn−1        k3 ··· kn−1                   kn−1                  
                                     (−1)n            (−1)n−1                        −1              1
                                     k2 ··· kn         k3 ··· kn      ...          kn−1 kn          kn   1

                                                      66
Observe that we are free to compose N ∗ with a translation. Multiplying the
columns as prescribed we get the vertices of the simplex
                                                       
                      ∗               a2 /π       an /π
                    N △ k2 . . . kn ,       ,...,         .
                                       k2          kn
Adding to the rows of this new matrix its smallest entry corresponds to
                                                                   R
translating this new simplex into the positive cone of n (y). The claim
thus follows. A similar but simpler procedure leads to the last statement.
                                                                        ✷


Proposition 3.17 leads to the number theoretic problem of finding appro-
priate relatively prime numbers k1 , . . . , kn−1 . An effective method which
solves this problem for a large is described in the proof of Proposition 4.10
(i)E .

Corollary 3.18 (i)E E 2n (π, lEB (a), . . . , lEB (a), a) ֒→ B 2n (lEB (a)+ǫ), where
                                              
                                                 (k + 1)π, (k − 1)(k + 1) ≤ a/π ≤ k(k + 1)
    lEB (a) = min max{(k + 1)π, a/k} =
              k∈N                                   a/k,     k(k + 1) ≤ a/π ≤ k(k + 2).

(i)P P 2n (π, lP C (a), . . . , lP C (a), a) ֒→ C 2n (lP C (a)), where
                                                   
                                                          kπ,      (k − 1)2 ≤ a/π ≤ k(k − 1)
     lP C (a) = min max{kπ, a/k + π} =
                 k∈N                                   a/k + π, k(k − 1) ≤ a/π ≤ k2 .

    For n ≥ 3 and any k ∈      N \ {1}
(ii)E E 2n (π, kn π, . . . , kn π) ֒→ B 2n ((kn−1 + kn−2 + (n − 2)kn−3 )π + ǫ),

(ii)P P 2n (π, (k − 1)kn−1 π, . . . , (k − 1)kn−1 π) ֒→ C 2n (kn−1 π).

Proof. In (i)E and (i)P Remark 3.16 was applied. For both (ii)E and (ii)P
choose k2 = · · · = kn = k. In (ii)E , the maximal sum is the one of the
entries of the n−1 st column, and in (ii)P all the sums are kn−1 .      ✷

Examples.

   ad (i)E and (i)P . Remark 3.16 and Proposition 3.17 (i) applied to op-
posite entries imply that for any k ∈       N
              E 2n (π, kπ, k2 π, . . . , k2l π) ֒→ B 2n ((kl + kl−1 )π + ǫ)

                                           67
                                                    PSfrag replacements
                                                                               1
                  A
                  π



                 5
                                       inclusion
                                             lP C
                 4                    sP C    π
                                       π

                 3

                 2                             volume condition

                                                                               a
                                                                               π
                      2         4        6              9


                Figure 30: What is known about P (π, a) ֒→ C 4 (A)

and

                  P 2n (π, kπ, k2 π, . . . , k 2l π) ֒→ C 2n ((kl + kl−1 )π)

if n = 2l + 1 is odd and

          E 2n (π, k2 π, k4 π, . . . , k2n−2 π) ֒→ B 2n ((kn−1 + kn−2 )π + ǫ)

and

            P 2n (π, k2 π, k4 π, . . . , k2n−2 π) ֒→ C 2n ((kn−1 + kn−2 )π)

if n is even.

   ad (ii)E . For n = 3, Proposition 3.17 yields
                                                                    
                     6                     a2                 a3
  E(π, a2 , a3 ) ֒→ B max k3 (k2 + 1)π,         (k3 + 1) + π,    +π +ǫ
                                          k2 k3               k3
for any k2 , k3 ∈     N
                      \ {1}. With (k2 , k3 ) = (k, lk − 1) we thus get for any
      N
k ∈ \ {1} and l ∈         N
                                        
                k(lk − 1)2
        E π,               π, k(lk − 1) π ֒→ B 6 (k(lk − 1)π + π + ǫ).
                                       2
                     l
   ad (ii)P . For n = 3, Proposition 3.17 yields
                                                             
                          6                      a2   a2    a3
      P (π, a2 , a3 ) ֒→ C max k2 k3 π, k3 π + , π +      +
                                                 k2  k2 k3 k3

                                              68
                   N
for any k2 , k3 ∈ \ {1}. With (k2 , k3 ) = (k, lk − l + 1) we thus get for any
      N
k ∈ \ {1} and l ∈        N
 P (π, (k − 1)k(lk − l + 1)π, l(k − 1)k(lk − l + 1)π) ֒→ C 6 (k(lk − l + 1)π).

                                                                             ✸

3.5       Symplectic versus Lagrangian folding
For small a, the estimate sEB provides the best result known. For example,
we get sEB
        π (4π) = 2.6916 . . . , whence we have proved

Fact. E(π, 4π) embeds in B 4 (2.692 π).

lEB (a) < sEB (a) happens first at a/π = 5.1622 . . . . In general, computer
calculations suggest that lEB and sEB yield alternately better estimates:
For all k ∈    Nwe seem to have that lEB < sEB on an interval around
a = k(k + 1)π and sEB < lEB on an interval around k(k + 2)π; moreover,
they suggest that

                   lim (sEB (k(k + 2)π) − lEB (k(k + 2)π)) = 0,
                   k→∞

i.e. lEB and sEB seem to be asymptotically equivalent. We checked the
above statements for k ≤ 5 000.
                                                √
Remark 3.19 The difference dEB (a) = lEB (a) − πa between lEB and the
volume condition
          p      attains local maxima at ak = k(k + 2)π, where dEB (a) =
(k + 2)π − k(k + 2) π. This is a decreasing sequence converging to π. ✸



    Figure 1 summarizes the results. The non trivial estimates from below
are provided by Ekeland-Hofer capacities, which yield A(a) ≥ a for a ∈
[π, 2π] and A(a) ≥ 2π for a > 2π.

3.6       Summary
                                                C
Given U ∈ O(n) and α > 0, set αU = {αz ∈ n | z ∈ U }.
   For U, V ∈ O(n) define squeezing constants

      s(U, V ) = inf{α | there is a symplectic embedding ϕ : U ֒→ αV }.

                                       69
Specializing, we define squeezing numbers

                      sE
                       q2 ...qn (U ) = s(U, E(1, q2 , . . . , qn ))

and

                      sPq2...qn (U ) = s(U, P (1, q2 , . . . , qn )),

and we write sB (U ) for sE               C          P
                          1...1 (U ) and s (U ) for s1...1 (U ).
   With this notation, the main results of this section read

                    sB (E(π, a)) ≤ min(sEB (a), lEB (a))                (111)
                     B
                    s (P (π, a)) ≤ sP B (a)                             (112)
                     C
                    s (E(π, a)) ≤ sEC (a)                               (113)
                     C
                    s (P (π, a)) ≤ min(sP C (a), lP C (a))              (114)

and

                         sC (P 2n (π, . . . , π, a)) ≤ s2n
                                                        P C (a)


4     Packings
In the previous section we tried to squeeze a given simple shape into a min-
imal ball and a minimal cube. This problem may be reformulated as follows:

“Given a ball B respectively a cube C and a simple shape S, what is the
largest simple shape similar to S which fits into B respectively C?”
or equivalently:
“Given a ball or a cube, how much of its volume may be symplectically packed
by a simple shape of a given shape?”

More generally, given U ∈ O(n) and any connected symplectic manifold
(M 2n , ω), define the U -width of (M, ω) by

w(U, (M, ω)) = sup{α | there is a symplectic embedding ϕ : αU ֒→ (M, ω)},
                                1
                                  R    n
and if the volume Vol(M, ω) = n!    M ω is finite, set

                                            |w(U, (M, ω))U |
                         p(U, (M, ω)) =                      .
                                               Vol(M, ω)

                                           70
In this case, the two invariants determine each other, p(U, (M, ω)) > 0 by
Darboux’s theorem, and if in addition n = 1, p(U, (M, ω)) = 1 by Theorem
4.2.
     Given real numbers 1 ≤ q2 ≤ · · · ≤ qn , we define weighted widths

                 wqE2 ...qn (M, ω) = w(E(1, q2 , . . . , qn ), (M, ω)),
                 wqP2 ...qn (M, ω) = w(P (1, q2 , . . . , qn ), (M, ω))

and packing numbers

                                                          (wqE2 ...qn (M, ω))n q2 . . . qn
 pE
  q2 ...qn (M, ω) = p(E(1, q2 , . . . , qn ), (M, ω)) =                                    ,
                                                                   n! Vol(M, ω)
                                                          (wqP2 ...qn (M, ω))n q2 . . . qn
 pPq2 ...qn (M, ω) = p(P (1, q2 , . . . , qn ), (M, ω)) =                                  .
                                                                     Vol(M, ω)
                                    E (M, ω) and p(M, ω) for pE (M, ω).
Write w(M, ω) for the Gromov width w1...1                     1...1



Example 4.1 Assume that (M, ω) = (V, ω0 ) ∈ O(n). By the very defini-
tions of squeezing constants and widths we have
                                                      1
                                  w(U, V ) =                .
                                                   s(U, V )

In particular, we see that squeezing numbers and weighted widths of simple
shapes determine each other via

                                                                   π2
        wqE2 ...qn (E(π, p2 π, . . . , pn π)) =                                          ,    (115)
                                                    sE
                                                     p2 ...pn (E(π, q2 π, . . . , qn π))
                                                                    π2
        wqP2 ...qn (P (π, p2 π, . . . , pn π)) =                                          ,   (116)
                                                    sPp2...pn (P (π, q2 π, . . . , qn π))
                                                                   π2
        wqE2 ...qn (P (π, p2 π, . . . , pn π)) =                                         ,    (117)
                                                    sPp2...pn (E(π, q2 π, . . . , qn π))
                                                                    π2
        wqP2 ...qn (E(π, p2 π, . . . , pn π)) =                                           .   (118)
                                                    sE
                                                     p2 ...pn (P (π, q2 π, . . . , qn π))

Combined with the estimates stated in subsection 3.6, these equations pro-
vide estimates of weighted widths and packing numbers of simple shapes
from below.                                                             ✸


                                             71
If (M, ω) is an arbitrary symplectic manifold whose Gromov width is known
to be large, these results may be used to estimate wqE2 ...qn (M, ω) and pE
                                                                          q2 ...qn (M, ω)
reasonably well from below.

Example. Let T 2 (π) be the 2-torus of volume π and S 2 (2π) the sphere
of volume 2π and endow M = T 2 (π) × S 2 (2π) with the split symplectic
structure. Theorem 5.2(ii) shows that p(M ) = 1. Thus, by (115) and (111)

                                (wqE (B 4 (2π)))2 q
 pE         E   4
  q (M ) ≥ pq (B (2π)) =
                                       4π 2
                                       qπ 2                      qπ 2
                            =     B              2
                                                    ≥                            .
                                (s (E(π, qπ)))        (min(sEB (qπ), lEB (qπ)))2

In particular, limq→∞ pE
                       q (M ) = 1.                                                 ✸


   On the other hand, w(U, (M, ω)) ≥ w(V, (M, ω)) whenever U ≤3 V ; in
particular, w ≥ wqE2 ...qn ≥ wqP2 ...qn for all 1 ≤ q2 ≤ · · · ≤ qn . Thus, if w(M, ω)
and the weights are small, we get good estimates of weighted widths and
packing numbers from above.

Example. Let r ≥ 1 and M = S 2 (π) × S 2 (rπ) with the split symplec-
tic structure. By the Non-Squeezing Theorem stated at the beginning of
                                                                  q
Appendix B we have w(M ) ≤ π, whence wqE (M ) ≤ π and pEq (M ) ≤ 2r . For
q ≤ r the obvious embedding E(π, qπ) ֒→ P (π, rπ) ֒→ M shows that these
inequalities are actually equalities.                                  ✸


    The knowledge of the Gromov width is thus of particular importance to
us. Recently considerable progress has been made in computing or estimat-
ing the Gromov width of closed 4-manifolds. An overview on these results
is given in Appendix B.

Remark. Since the Gromov width is the smallest symplectic capacity
we might try to estimate it from above by using other symplectic capac-
ities. However, other capacities (like the Hofer-Zehnder capacity or the first
Ekeland-Hofer capacity, Viterbo’s capacity and the capacity arising from
                                                    R
symplectic homology in the case of subsets of 2n ) are usually even harder
to compute. In fact, we do not know of any space for which a capacity
other than the Gromov width is known and finite while its Gromov width

                                         72
is unknown.                                                                ✸


4.1   Asymptotic packings
Theorem 4.2 Let M n be a connected manifold endowed with a volume form
               R
Ω and let U ⊂ n be diffeomorphic to a standard ball. Then U embeds in
M by a volume preserving map if and only if |U | ≤ Vol (M, Ω).

               R       R
Proof. Endow >0 = >0 ∪ {∞} with the topology whose base of open
                                                    R
sets is given by joining the open intervals ]a, b[ ⊂ >0 with the subsets of
                                                                R
the form ]a, ∞] = ]a, ∞[ ∪ {∞}. Denote the Euclidean norm on n by k · k
                                R
and let S1 be the unit sphere in n .

                   R
Lemma 4.3 Let n be endowed with its standard smooth structure, let
        R
µ : S1 → >0 be a continuous function and let
                                                    
            S= x∈       R
                        n
                           x = 0 or 0 < kxk < µ
                                                   x
                                                  kxk

be the starlike domain associated to µ. Then S is diffeomorphic to   Rn.
Remark. The diffeomorphism guaranteed by the lemma may be chosen
such that the rays emanating from the origin are preserved.

Proof of the lemma. If µ(S1 ) = {∞}, there is nothing to prove. For µ
bounded, the lemma was proved by Ozols [28]. If µ is neither bounded nor
µ(S1 ) = {∞}, Ozols’s proof readily extends to our situation. Using his no-
tation, the only modifications needed are: Require in addition that r0 < 1
                                                            R
and that ǫ1 < 2, and define continuous functions µ̃i : S1 → >0 by

                        µ̃i = min{i, µ − ǫi + δi /2}.

With these minor adaptations the proof in [28] applies word by word.       ✷

Next, pick a complete Riemannian metric g on M . (We refer to [16] for
basic notions and results in Riemannian geometry.) The existence of such
a metric is guaranteed by a theorem of Whitney [33], according to which
                                                        R
M can be embedded as a closed submanifold in some m . We may thus
take the induced Riemannian metric. A direct and elementary proof of the
existence of a complete Riemannian metric is given in [27]. Fix a point
p ∈ M , let expp : Tp M → M be the exponential map at p with respect to g,

                                     73
                                          e
let C(p) be the cut locus at p and set C(p)   = exp−1 p (C(p)). Let S1 be the
                                     R
unit sphere in Tp M , let µp : S1 → >0 be the function defining C(p)  e   and
                                                    e
let Sp ⊂ Tp M be the starlike domain defined by C(p).     Since g is complete,
µp is continuous [16, p. 98]. We are thus in the situation of Lemma 4.3, and
since expp (Sp ) = M \ C(p) [16, p. 100], we obtain

Corollary 4.4 Let (M n , g) be a complete Riemannian manifold. Then the
maximal normal neighbourhood M \ C(p) of any point p in M is diffeomor-
                     R
phic to the standard n .

                                                                  e
Using polar coordinates on Tp M we see from Fubini’s Theorem that C(p)
has zero measure; thus the same holds true for C(p), whence

             Vol (Sp , exp∗p Ω) = Vol (M \ C(p), Ω) = Vol (M, Ω).

Theorem 4.2 now follows from Lemma 4.3 and

Proposition 4.5 (Greene-Shiohama, [11]) Two volume forms Ω1 and Ω2
on an open manifold are diffeomorphic if and only if the total volume and
the set of ends of infinite volume are the same for both forms.

                                                                             ✷
Remark. The existence of a volume preserving embedding of a set U as
above with |U | < Vol (M, Ω) immediately follows from Moser’s deformation
technique if M is closed and from Proposition 4.5, which is itself an exten-
sion of that technique to open manifolds, if M is open. The main point in
Theorem 4.2, however, is that all of the volume of M can be filled. This is
in contrast to the full symplectic packings by k balls established in [25], [2]
and [3].                                                                     ✸

In view of the Non-Squeezing Theorem and the existence of symplectic ca-
pacities, very much in contrast to the volume-preserving case, there exist
strong obstructions to full packings by “round” simple shapes in the sym-
plectic category. (We refer to the previous sections for related results on
embeddings into simple shapes and to Appendix B for an overview on known
results on the Gromov width of closed four manifolds.)
    However, the results of section 3 show for example that for embeddings
into four dimensional simple shapes packing obstructions more and more
disappear if we pass to skinny domains. The main goal of this section is to
show that in the limit rigidity indeed disappears.

                                      74
Theorem 4.6 Let (M, ω) be a connected symplectic manifold of finite vol-
ume. Then
   pE               E
    ∞ (M, ω) = lim p1...1q (M, ω)         and     pP∞ (M, ω) = lim pP1...1q (M, ω)
                  q→∞                                          q→∞

exist and equal 1.
Remark. Remark 3.16, Proposition 3.17(i) and the theorem immediately
imply that for any (M, ω) as in the theorem
              lim pE 2 n−1 (M, ω)         and      lim pP 2 n−1 (M, ω)
             q→∞ qq ... q                         q→∞ qq ... q

exist and equal 1.                                                                   ✸


     The proof of the statement for polydiscs proceeds along the following
lines: We first fill M up to some ǫ with small disjoint closed cubes, which we
connect by lines. We already know how to asymptotically fill these cubes
with thin polydiscs, and we may use neighbourhoods of the lines to pass
from one cube to another (cf. Figure 31).
     The case of ellipsoids is less elementary. For n ≤ 3, the statement for
ellipsoids follows from the one for polydiscs and the fact that a polydisc
may be asymptotically filled by skinny ellipsoids. This is proved in the
same way as (26). In higher dimensions, however, symplectic folding alone
is not powerful enough to fill a polydisc by thin ellipsoids, since there is
no elementary way of filling a cube by balls. However, algebro-geometric
methods imply that in any dimension cubes can indeed be filled by balls.
Using this, we may almost fill (M, ω) by equal balls, which we connect again
by thin lines. The claim then readily follows from the proof of Proposition
3.13.
    We begin with the following
Lemma 4.7 (McDuff-Polterovich, [25]) Let (M, Ω) be a symplectic mani-
fold of finite volume. Then, given `
                                   ǫ > 0, there is an embedding
                                                              ` of a dis-
joint union of closed equal cubes    C(λ) into M such that | C(λ)| >
Vol (M ) − 2ǫ.
Proof. Assume first that M is compact and cover M with Darboux charts
Vi = ϕi (Ui ), i = 1, . . . , m. Pick closed cubes C 1 , . . . , C j1 ⊂ U1 of possibly
varying size such that
                             j1
                             X                           ǫ
                                   |Cj | > Vol (V1 ) −     .
                                                         m
                             j=1


                                           75
                                                              P
Proceeding by finite induction, for S         i > 1, set ki = i−1l=1 jl and pick closed
                                                i−1
cubes C ki +1 , . . . , C ki +ji ⊂ Ui \ ϕ−1
                                         i  (       V
                                                j=1 j ) such that

                           ji
                           X                             i−1
                                                         [              ǫ
                                 |Cki +j | > Vol (Vi \         Vj ) −     .
                                                                        m
                           j=1                           j=1


Choose now λ so small that`all the cubes C k , 1 ≤ k ≤ km+1 , admit an em-
bedding of a disjoint union nj=1k
                                  C(λ) such that nk |C(λ)| > |Ck | − ǫ/km+1 .
                                     Pkm+1
In this way, we get an embedding of k=1      nk closed cubes into M filling
more than Vol (M ) − 2ǫ.
    If M is not compact, choose a volume-preserving embedding ϕ :
  2n
B (Vol (M )− ǫ) ֒→ M (cf. Theorem 4.2) and apply the already proved part
to (B 2n (Vol (M ) − ǫ), ϕ∗ ω).                                           ✷

                                    αP 2n (π, . . . , π, a)




            C 1 (λ)                C 2 (λ)    L2         C 3 (λ)                  C 4 (λ)
                                                                   PSfrag replacements
                      ǫ2
            √                                                                               x1
             λ         L1                                            L3
                                                ψ



                                                                              M




                   Figure 31: Asymptotic filling by polydiscs

    We next connect the cubes by thin lines.

                                                76
                                `            `
    Pick ǫ1 > 0 and let ϕ = ki=1 ϕi : ki=1 C i (λ) ֒→ M be a corresponding
embedding guaranteed by Lemma 4.7. Extensions of the ϕi to small neigh-
bourhoods of C i (λ) are still denoted by ϕi . We may assume that the faces
of the C i (λ) are cubes and that all the C i (λ) lie in the positive cone of 2n    R
and touch the x1 -axis. Join these cubes by straight lines Li as described
in Figure 31, i.e. fixing regular parameterizations Li (t) : [0, 1] → Li we have
Li (0) ∈ ∂C i (λ), Li (1) ∈ ∂C i+1 (λ) and
                         
                            (x1 (Li (t)), 0,
                                          √ . . . , 0)√ for i odd,
                Li (t) =
                            (x1 (Li (t)), λ, . . . , λ) for i even.
          `k−1       `                `
Let now i=1     λi :    Li → M`\ ϕi (Ci (λ)) be a disjoint family of embedded
curves in M which touches             ϕi (C i (λ) only at the points λi (0) and λi (1)
and coincides with ϕi|Li respectively ϕi+1|Li+1 on a small neighbourhood
of C i (λ) respectively C i+1 (λ). Choose 1-parameter families of symplectic
frames {ej,i (t)}2n                      ′       2n
                 j=1 respectively {ej,i (t)}j=1 along Li (t) respectively λi (Li (t))
                      d                            d
such that e1,i (t) = dt λi (t) and e′1,i (t) = dt    λi (Li (t)). Let ψ̃i be an extension
of λi to a neighbourhood of Li which coincides with ϕi respectively ϕi+1 on
a neighbourhood of λi (0) respectively λi (1) and which sends the symplectic
frame along Li (t) to the one along λi (Li (t)), i.e.
                                        
                              TLi (t) ψ̃i (ej,i (t)) = e′j,i (t).

ψ̃i is thus a diffeomorphism on a neighbourhood of Li which is symplectic
along Li . Using a variant of Mosers’s method (see [26, Lemma 3.14 and its
proof ]) we see that ψ̃i may be deformed to an embedding ψi of a possibly
smaller neighbourhood of Li which still coincides with λi on Li and ϕi
respectively ϕi+1 on a neighbourhood of Li (0) respectively Li (1), but is
symplectic everywhere. Choose ǫ2 > 0 so small that for all i, ψi is defined
on          = {x1 (Li (t))}×[0, ǫ2 ]2n−1 if i is odd and on Ni (ǫ2 ) = {x1 (Li (t))}×
 √ Ni (ǫ2 ) √
[ λ − ǫ2 , λ]2n−1 if i is even.
     Summing up, we see that there exists ǫ2 > 0 such that
                                      a           a
                           N (ǫ2 ) =     Ci (λ)     Ni (ǫ2 )

symplectically embeds in M .
    It remains to show that N (ǫ2 ) may be asymptotically filled by skinny
polydiscs. We try to fill N (ǫ2 ) by αP 2n (π, . . . , π, a) with α small and
a large by packing the Ci (λ) as described in subsection 3.3.1 and using
Ni (ǫ2 ) to pass from Ci (λ) to Ci+1 (λ). Here we think of αP 2n (π, . . . , π, a) as

                                           77
α2
                                    and of C 2n (λ) as ǫ12 ✷(λ, . . . , λ)×✷(ǫ2 , . . . , ǫ2 ).
ǫ2 ✷(a, π, . . . , π)×✷(ǫ2 , . . . , ǫ2 )
Write Pi for the restriction of the image of αP 2n (π, . . . , π, a) to Ci (λ). In
order to guarantee that the “right” face of Pi and the “left” face of Ni (ǫ2 )
fit, we require that the number of folds in each z1 -z2 -layer is even and that
the component of Pi between its right face and the last stairs touches ∂C i (λ)
wherever possible. This second point may be achieved by making n−1 of
the stairs in Pi a little bit higher than necessary. The part of the image
of αP 2n (π, . . . , π, a) between Pi and Pi+1 will thus be contained in Ni (ǫ2 )
whenever α2 π < ǫ22 .
     Now, in Proposition 3.12 we have
                                           aπ n−1
                                      lim           = 1,
                                     a→∞ (s2n (a))n
                                           PC

and hence, by duality,
                                   lim pP (C 2n (λ))    = 1.                            (119)
                                  q→∞ 1...1q

(119) is clearly not affected by the two minor modifications which we re-
quired above for the packing of Ci (λ). Thus half of the theorem follows.
   As explained above, in order to prove the statement for ellipsoids we
need the following non-elementary result.

Proposition 4.8 (McDuff-Polterovich, [25, Corollary 1.5.F]) For each pos-
itive integer k, arbitrarily much of the volume of C 2n (π) may be filled by
n! kn equal closed balls.

    This proposition may be proved in two different ways, either via symplec-
tic blowing up and fibrations or via symplectic branched coverings. Com-
bining it with Lemma 4.7, we see that we may fill as much of the volume of
(M, ω) by disjoint equal closed balls as we want.
    So assume
             that (M, ω) is almost filled by m+1 disjoint equal closed balls
 B i (λ), ϕi , 0 ≤ i ≤ m. By Lemma 3.11(ii) we may think of Bi (λ) as fibered
over ]iλ + i, (i + 1)λ + i[ × ]0, 1[ with fibers γ△n−1 (λ) × ✷n−1 (1), 1 ≥ γ > 0
(cf. Figure
      `      32). Exactly as in the case of cubes we find an ǫ > 0 such that
ϕ= m    i=0 i extends to a symplectic embedding ψ of a small neighbourhood
            ϕ
of
                          a           [
                  N (ǫ) =     Bi (λ)     ]0, mλ + m[ × ]0, ǫ[2n−1 .

Let τi :   R2n → R2n, z 7→ z + i(ǫ − 1, 0, . . . , 0) and set
                  ei (ǫ) = ]iλ + (i − 2)ǫ, iλ + iǫ[ × ]0, 1[ × ]0, ǫ[2n−2
                  N

                                             78
                                                               PSfrag replacements




             B0 (λ)            B1 (λ)                                     Bm (λ)
                                             ǫ ...
                                                                                      x1
       0        λ      λ+1                                             (m + 1)λ + m


                                  Figure 32: N (ǫ)


and
                                  m
                                  a                    m
                                                      [a
                        e (ǫ) =
                        N               τi (Bi (λ))          ei (ǫ).
                                                             N
                                  i=0                  i=1


It is a simple matter to find a symplectomorphism σ of 2 such that σ ×   R
id2n−2 embeds N  e (ǫ) into an arbitrarily small neighbourhood of N (ǫ). It thus
remains to show that N   e (ǫ) may be asymptotically filled by skinny ellipsoids.
               e
We try to fill N (ǫ) by αE 2n (π, . . . , π, a) with α small and a large by packing
the Bi (λ) as in the proof of Proposition 3.13 and using N       ei (ǫ) to pass from
                                                     2n
Bi (λ) to Bi+1 (λ). To this end, think of αE (π, . . . , π, a) as fibered over
                          2
✷(α2 a, 1) with fibers βǫ △n−1 (π) × ✷n−1 (ǫ), α ≥ β > 0.
    We observe that the present packing problem is easier then the one
treated in Proposition 3.13 inasmuch as now only a part of αE 2n (π, . . . , π, a)
is embedded into a Bi (λ), whence the ellipsoid fibres decrease slowlier.
         `
    Let li=1 Pi be a partition of τ1 (B1 (λ)) as in the proof of Proposition
3.13 and let γ△n−1 (λ) × ✷n−1 (1) be the smallest fiber of Pl−1 . Assume
that l is so large that γλ < ǫ and that α is so small that α2 π < ǫγλ. The
image of the last ellipsoid fiber mapped to Pl−1 is then contained in N        e1 (ǫ),
and we may pass to τ2 (B2 (λ)). Having reached P1 (τ2 (B2 (λ))), we first of all
move the ellipsoid fiber out of the connecting floor and then deform the fiber
of the second floor to a fiber with maximal △n−1 -factor (µ1 in Figure 33).
We then fill the remaining room in P1 (τ2 (B2 (λ))) as well as possible (cf.
Figure 33) and proceed filling τ2 (B2 (λ)) as before. The above modification
in the filling of τ2 (B2 (λ)) clearly does not affect the result in Proposition
3.13. Going on in the same way, we fill almost all of N       e (ǫ). This concludes
the proof of Theorem 4.6.                                                           ✷




                                            79
                         y3


                     1
                                            δ1

                                        PSfrag replacements


                              δ2

                                                            y2
                                                   1


            Figure 33: The two deformations in P1 (τ2 (B2 (λ)))


4.2   Refined asymptotic invariants
Theorem 4.6 shows that the asymptotic packing numbers pE             P
                                                             ∞ and p∞ are
uninteresting invariants. However, we may try to recapture some symplectic
information on the target space by looking at the convergence speed. Given
(M, ω) with Vol (M, ω) < ∞ consider the function

                    [1, ∞[→        R,   q 7→ 1 − pE
                                                  1...1q (M, ω)

and define a refined asymptotic invariant by

             αE (M, ω) = sup{β | 1 − pE
                                      1...1q (M, ω) = O(q
                                                          −β
                                                             )}.

Define αP (M, ω) in a similar way.
   Let U ∈ O(n) with piecewise smooth boundary ∂U . Given a subset
S ⊂ ∂U , let

                          Ss = {x ∈ U | d(x, S) < s}

be the s-neighbourhood of S in U . We say that U is admissible, if there
exists ǫ > 0 such that U \ ∂Uǫ is connected.

Example 4.9 Let K(h, k) ⊂          R2n be a camel space:
                K(h, k) = {x1 < 0} ∪ {x1 > 0} ∪ H(h, k),

                                          80
where
                                ( n                n
                                                                          )
                                 X                 X
                  H(h, k) =              x2i   +         yi2      2
                                                               < h , x1 = k .
                                   i=2             i=1

Pick sequences (hi )i∈N and (ki )i∈N with h1 > h2 > . . . , hi → 0 and 0 =
k1 < k2 < . . . , ki → 1, let C = {−1 < x1 , . . . , xn , y1 , . . . , yn < 1} be a cube
and set
                                               ∞
                                               \
                                 U =C∩               K(hi , ki ).
                                               i=1

Then C is not admissible. Thickening the walls and smoothing the bound-
ary, we obtain non admissible sets with smooth boundaries.           ✸



Proposition 4.10 Let U ∈ O(n) be admissible and let (M 2n , ω) be a closed
symplectic manifold. Then
                     1
(i)E     αE (U ) ≥   n       if n ≤ 3 or if U ∈ E(n)
                         1
(ii)E    αE (M, ω) ≥     n      if n ≤ 3
                     1
(i)P     αP (U ) ≥   n

(ii)P    αP (M, ω) ≥ n1 .

Question. Given γ ∈ ]0, 12 [, are there sets U, V ∈ O(2) with αE (U ) =
αP (V ) = γ ? Candidates for such necessarily non admissible sets are the
sets described in Example 4.9 with (hi ), (ki ) chosen appropriately.  ✸


Proof of Proposition 4.10. ad (i)P . If U is a cube, the claim follows at
once from Proposition 3.12. If U is an arbitrary admissible set, let

                Nd = {(x1 , . . . , x2n ) ∈    R2n | xi ∈ dZ, 1 ≤ i ≤ n}
                   R
be the d-net in 2n , and let Cd be the union of all those open cubes in
R 2n
 √ \ Nd which lie entirely in U . Observe    √ that U \ Cd ⊂ ∂U s ∂Us whenever
d 2n < s. Let s0 < ǫ and d0 < s0 / 2n. Pick α0 much smaller than d0
and exhaust Cd0 with α20 P 2n (π, . . . , π, a0 ) by successively filling the cubes in

                                               81
                                    N
Cd0 . More generally, let k ∈ 0 , suppose that we almost exhausted Cd0 /2k
by α2k0 P 2n (π, . . . , π, ak ) and consider Cd0 /2k+1 . Then

                               U \ Cd0 /2k+1 ⊂ ∂U s0 /2k+1 .                           (120)

We fill the cubes in Cd0 /2k by 2αk+10
                                       P 2n (π, . . . , π, ak+1 ) in the same order as
                      α0 2n
we filled them by 2k P (π, . . . , π, ak ), but in between also fill the cubes in
Cd0 /2k+1 \ Cd0 /2k . Observe that in order to come back from a cube Ck+1 ∈
Cd0 /2k+1 to its “mother-cube” Ck ∈ Cd0 /2k , we possibly have to use some extra
                                                                α0
space in Ck , but that for the subsequent filling by 2k+2           P 2n (π, . . . , π, ak+2 )
this extra space will be halved.
     Since the ak were chosen maximal and since we exhaust more and more
of U ,
                                       ak+1
                                  lim         = 2n .                                    (121)
                                 k→∞ ak

(121), the preceding remark and the case of a cube show that for any δ > 0
there is a constant C1 (δ) such that for any k, any k′ ≤ k and any Ck′ ∈ Cd0 /2k′
                                                       
              Ck′ \ image 2αk0′ P 2n (π, . . . , π, ak )           − 1 +δ
                                                          < C1 (δ)ak n .  (122)
                              |Ck′ |
Let ∂k U be the k-dimensional components of ∂U , 0 ≤ k ≤ 2n − 1, and let
|∂k U | be their k-dimensional volume. Then there are constants ck depending
only on U such that
                                          |∂k U s |
                                    lim             = ck ,
                                   s→0+    s2n−k
whence
                              ∂U s/2         ∂2n−1 U s/2   1
                       lim            = lim               = .                          (123)
                      s→0+    |∂U s |  s→0 + |∂2n−1 U s |  2
(120), (123) and (122) imply that for any δ > 0 there is a constant C2 (δ)
such that for any k
                           α                                  − 1 +δ
                              0
            Cd0 /2k \ image k P 2n (π, . . . , π, ak ) < C2 (δ)ak n .  (124)
                            2
Next, (120), (121) and (123) show that for any δ > 0 there is a constant
C3 (δ) such that for any k
                                                                1
                                                               −n +δ
                       |U \ Cd0 /2k | ≤ |Bs0 /2k | < C3 (δ)ak          .               (125)


                                             82
(i)P now follows from (124) and (125).
     ad (ii)P . Cover M with Darboux charts (Ui , ϕi ), i = 1, . . . , m, and choose
admissible
      Sm subsets Vi of Ui such that the sets Wi = ϕi (Vi ) are disjoint
and i=1 W i = M . Choose different points pi , qi ∈ Vi , set p̃i = ϕi (pi ),
                    ei : [0, 1] → M be a family of smooth, embedded and disjoint
q̃i = ϕi (qi ), let λ
curves connecting q̃i with p̃i+1 , and set λi,j = ϕ−1       e
                                                         j (λi ), 1 ≤ i ≤ m − 1,
1 ≤ j ≤ m. We may assume that near qi respectively pi+1 , λi,i respectively
λi,i+1 are linear paths parallel to the x1 -axis. As in the proof of Theorem
4.6 we find ǫ > 0 such that the λ    ei extend to disjoint symplectic embeddings

                            ψi : [0, 1] × [−ǫ, ǫ]2n−1 → M

whose compositions ψi,i = ϕ−1  i ◦ ψi respectively ψi,i+1 = ϕi+1 ◦ ψi restrict
to translations near {0} × [−ǫ, ǫ]2n−1 respectively {1} × [−ǫ, ǫ]2n−1 . More
generally, set ψi,i = ϕ−1
                       j ◦ ψi , and given δ ≤ ǫ, set


           ψiδ = ψi |[0,1]×[−δ,δ]2n−1        and         δ
                                                        ψi,j = ψi,j |[0,1]×[−δ,δ]2n−1 .

Let α be so small that α2 π ` < 4δ2 . We may then fill M with αP 2n (π, . . . , π, a)
by successively filling Wi \ m−1             δ
                                k=1 image ψk and passing from Wi to Wi+1 with
               δ
the help of ψi .
    In order to estimate the convergence speed of the filling of Wi , let us
look at the corresponding filling of Vi instead. Set
                                                                  a
       λδi,j = {x ∈ Vi | d(x, image λi,j ) < δ} and Viδ = Vi \        λδi,j .
                                                                                   j
                                            `
Let L be a Lipschitz-constant for               i,j   ψi,j . Then

                                              δ
                                       image ψi,j ⊂ λLδ
                                                     i,j .                                (126)

With Vi also Vi0 is admissible, and so there is δ0 > 0 such that ViLδ0
is connected. This and (126) show that we may fill Vi with a part of
α0 P 2n (π, . . . , π, a0 ) by entering Vi through λLδ   i,i , filling as much of Vi
                                                             0                      Lδ0

as possible and leaving Vi through λLδ                     δ
                                           i,i+1 . Let i Cd be the union of those open
                                              0


          R
cubes in 2n \ Nd which lie entirely in ViLδ . Then

                                            m−1
                                            a             [
                                  δ0               2Lδ0
                        Vi \   i Cd0    ⊂         λi,j         (∂Vi )Lδ0                  (127)
                                            j=1



                                                  83
           √
whenever d0 2n < Lδ0 . Finally,
                                     s/2
                                   λi,j            1    1
                             lim            =          < .                      (128)
                            s→0+    λsi,j        22n−1  2

(ii)P now follows from (127), (128) and the proof of (i)P .
    ad (i)E and (ii)E . By the Folding Lemma, E(π, a) ֒→ P (π, (a + π)/2),
whence the case n = 2 follows from (i)P and (ii)P .
    Let n = 3, and let U be a cube. We fill U as described in 3.3.2.1. This
asymptotic packing problem resembles the one in the proof of Proposition
3.14. Again, for given a, the region in U not covered by the image of the
maximal ellipsoid αE(π, π, a) fitting into U decomposes into several disjoint
regions Rh (a), 2 ≤ h ≤ 4.
      R2 (a) is the space needed for folding.
      R3 (a) is the union of the space needed to deform the ellipsoid fibers
      and the space caused by the fact that the sum of the sizes of the
      ellipsoid fibres embedded into a column of the cube fibre and the x3 -
      width of the space needed to deform one of these ellipsoid fibres might
      be smaller than the size of the cube fibre.
      R4 (a) is the space caused by the fact that the size of the ellipsoid fibres
      decreases during the filling of a column of the cube fibre.
We compare Rh (a) with Rh (2n a) = Rh (8a). Let α′ E(π, π, 8a) be the max-
imal ellipsoid fitting into U . A volume comparison shows that for a large
α′ is very close to α/2. A similar but simpler analysis than in the proof of
Proposition 3.14 now shows that given ǫ > 0 there is a0 such that for any
a ≥ a0
                   (2 − ǫ) |Rh (8a)| < |Rh (a)| ,       2 ≤ h ≤ 4.
This implies the claim in case of a cube. The general case follows from this
case in the same way as (i)P and (ii)P followed from the case of a cube.
    Finally, let E = E(b1 , . . . , bn ). It follows from the description of La-
grangian folding in subsection 3.4 and from Lemma 3.15(i) that given n − 1
relatively prime numbers k1 , . . . , kn−1 there is an embedding E 2n (π, . . . , π, a) ֒→
βE(b1 , . . . , bn ) whenever
                                                              )
                        π       π        1
                       βbi + ki βbn < ki ,         1≤i≤n−1
                                 π      k1 ···kn−1 π            .                (129)
                                βbn <         a      .


                                            84
W.l.o.g. we may set bn = 1. (129) then reads
                                                                                  
                         ki π < (β − 1)bi ,      1≤i≤n−1
                                                                                      .              (130)
                            a < k1 · · · kn−1 β.

Pick some (large) constant C and define β by
                                                   n−1
                                                        
                    b1 · · · bn−1 β n = π n−1 a + Ca n .

Moreover, pick n − 1 prime numbers p1 , . . . , pn−1 , let l be the least common
multiple of {pi − pj | 1 ≤ i < j ≤ n − 1}, define mi , 1 ≤ i ≤ n − 1, by

                       mi = max{m ∈              N | mil − pi < (β − 1)bi /π}
and set ki = mi l − pi . We claim that the ki are relatively prime. Indeed,
assume that for some i 6= j

                            d | mi l − p i            and      d | mj l − p j .                      (131)

Then d divides (mi l − pi ) − (mj l − pj ) = pi − pj , and hence, by the definition
of l, d divides l. But then, by (131), d divides pi and pj , whence d = 1.
    The first n − 1 inequalities in (130) hold true by the definition of the ki ,
and since bi ≤ 1,

             π n−1 k1 · · · kn−1 β > (βb1 − l − 1) · · · (βn−1 − l − 1)β
                                                       n−1
                                                       X
                                                    n
                                   = b1 · · · bn−1 β +     (−1)i ci β n−i ,
                                                                        i=1

where the ci are positive constants depending only on b1 , . . . , bn−1 and l.
For a large enough the last expression is larger than b1 · · · bn−1 β n − c1 β n−1 ,
which equals
                                                                   n−1
                                                                        n                    n−1
             n−1                n−1                     π n−1                          n−1     n
         π             a + Ca    n        − c1                                a + Ca    n
                                                     b1 · · · bn−1

and this is larger than π n−1 a whenever a and C are large enough.
   Finally, we have that
 |E 2n (π, . . . , π, a)|     π n−1 a            1               1
                                                                        1
                                                                −n
                          = n              =         1 = 1 − Ca    + o  a− n ,
  |βE(b1 , . . . , bn )|   β b1 · · · bn−1   1 + Ca −n

from which the second claim in (i)E follows.                                                            ✷


                                                        85
Remark. Suppose that we knew that there is a natural number k such that
the cube C 2n admits a full symplectic packing by k equal balls and such that
the space of symplectic embeddings of k equal balls into C 2n is unknotted.
Combining such a result with Proposition 3.14 and the techniques used in
the proof of Theorem 4.6 and Proposition 4.10 we may derive that
                                   1                             1
                      αE (U ) ≥          and    αE (M, ω) ≥
                                  2n                            2n
for any admissible U ∈ O(n) and any closed symplectic manifold (M 2n , ω).

4.3    Higher order symplectic invariants
The construction of good higher order invariants for subsets of 2n has            R
turned out to be a difficult problem in symplectic topology. The known
such invariants are Ekeland-Hofer capacities [6, 7] and symplectic homology
[9, 10], which both rely on the variational study of periodic orbits of certain
Hamiltonian systems, and the symplectic homology constructed via gener-
ating functions [32]. We propose here some higher order invariants which
are based on an embedding approach.
    Let (M 2n , ω) be a symplectic manifold and let

        e1 (M, ω) = sup{A | B 2n (A) symplectically embeds in (M, ω)}

be the Gromov-width of (M, ω). We inductively define n−1 other invariants
by

        ei (M, ω) = sup{A | E 2n (e1 (M, ω), . . . , ei−1 (M, ω), A, . . . , A)
                                       symplectically embeds in (M, ω)}.

Similarly, given U ∈ O(n), let

             en (U ) = inf{A | U symplectically embeds in B 2n (A)}

and inductively define n − 1 other invariants ei (U ) by

ei (U ) = inf{A | U symplectically embeds in E 2n (A, . . . , A, ei+1 (U ), . . . , en (U )}.

Clearly,

                      e1 (M, ω) ≤ e2 (M, ω) ≤ · · · ≤ en (M, ω)

and

                     e1 (M, ω) ≤ e2 (M, ω) ≤ · · · ≤ en (M, ω).

                                           86
Moreover, ei (M, αω) = |α| ei (M, ω) and ei (U, αω0 ) = |α| ei (U, ω0 ) for all α ∈
R  \ {0}, and ei and ei are indeed invariants, that is ei (M, ω) = ei (N, τ ) and
e (U, ω0 ) = ei (V, ω0 ) if there are symplectomorphisms ϕ : (M, ω) → (N, τ )
 i

and ψ : (U, ω0 ) → (V, ω0 ).

Example 4.11 Ekeland-Hofer capacities show that

                       ei (E(a1 , . . . , an )) = ai ,     1 ≤ i ≤ n,

and

            ei (E(a1 , . . . , an )) = ai ,      1 ≤ i ≤ n,          if 2a1 ≥ an .

                                                                                     ✸


e1 and en are also monotone and nontrivial, and are hence symplectic ca-
pacities (see [14] for the axioms of a symplectic capacity). This, however,
does not hold true for any of the higher invariants. Indeed, let Z(π) =
        R
D(π) × 2n−2 be the standard symplectic cylinder. Then

                             ei (Z(π)) = ∞          for all i ≥ 2.

Moreover, Example 4.11 and Theorem 2A show that none of the ei , i ≥ 2,
is monotone, and the same holds true for ei , i ≤ n − 1. For instance, set
Uλ = 34 E(λ−1 π, λπ) and V = E(π, 2π). By Theorem 4.6, Uλ symplectically
embeds in V and e2 (Uλ ) is near to 34 π if λ is large. Then also e1 (Uλ ) is near
to 43 π; but e1 (V ) = π.
     Similar invariants may be constructed by looking at polydiscs instead of
ellipsoids.
     These considerations indicate that it should be difficult to construct
higher order symplectic capacities via an embedding approach.


5     Appendix
A. Computer programs
All the Mathematica programs of this appendix may be found under
            ftp://ftp.math.ethz.ch/pub/papers/schlenk/folding.m
For convenience, in the programs (but not in the text) both the u-axis and
the capacity-axis are rescaled by a factor 1/π.

                                               87
A1. The estimate sEB
As said at the beginning of 3.2.3.1 we fix a and u1 and try to embed
E(π, a) into B 4 (2π + (1 − 2π/a)u1 ) by multiple folding. If this works, we
set A(a, u1 ) = 2π + (1 − 2π/a)u1 and A(a, u1 ) = a otherwise.

A[a_, u1_] :=
  Block[{A=2+(1-2/a)u1},
     j = 2;
     uj = (a+1)/(a-1)u1-a/(a-1);
     rj = a-u1-uj;
     lj = rj/a;
     While[True,
           Which[EvenQ[j],
                    If[rj <= uj,
                        Return[A],
                        If[uj <= 2lj,
                           Return[a],
                           j++;
                           uj = a/(a-2)(uj-2lj);
                           rj = rj-uj;
                           li = lj;
                           lj = rj/a
                          ]
                      ],
                  OddQ[j],
                    If[rj <= uj+li,
                        Return[A],
                        j++;
                        uj = (a+1)/(a-1)uj;
                        rj = rj-uj;
                        lj = rj/a
                      ]
                ]
            ]
         ]

This program just does what we proposed to do in 3.2.3.1 in order to de-
cide if the embedding attempt associated with u1 succeeds or fails. Note,
however, that in the Oddq[j]-part, we did not check whether the upper left
corner of Fj+2 is contained in T (A, A). However, this negligence does not

                                    88
cause troubles, since if the left edge of Fj+2 indeed exceeds T (A, A), the
embedding attempt will fail in the subsequent EvenQ[j+1]-part. In fact,
that the left edge of Fj+2 exceeds T (A, A) means that lj+1 > uj+1 ; hence
rj+1 > uj+1 (since otherwise the embedding attempt would have succeeded
in the preceding OddQ[j]-part), but uj+1 ≤ 2lj+1 .
    Writing again u0 for the minimal u1 which leads to an embedding,
A(a, u1 ) is equal to a for u1 < u0 and it is a linear increasing function
for u1 ≥ u0 . Since, by (23), we may assume that u0 ≤ a/2, we have
A(a, u0 ) ≤ π + a/2 < a, whence u0 is found up to accuracy acc/2 by the
following bisectional algorithm.

u0[a_, acc_] :=
  Block[{},
     b = a/(a+1);
     c = a/2;
     u1 = (b+c)/2;
     While[(c-b)/2 > acc/2,
       If[A[a,u1] < a, c=u1, b=u1];
       u1 = (b+c)/2
          ];
     Return[u1]
       ]

Here the choice b = aπ/(a + π) is also based on (23). Up to accuracy acc,
the resulting estimate sEB (a) is given by

sEB[a_, acc_] := 2 + (1-2/a)u0[a,acc].

A2. The estimate sEC
Given a and u1 , we first calculate the height of the image of the correspond-
ing embedding. The following program is easily understood by looking at
Figure 19.

h[a_, u1_] :=
  Block[{l1=1-u1/a},
     j = 2;
     uj = (a+1)/(a-1)u1-a/(a-1);
     rj = a-u1-uj;
     lj = rj/a;
     hj = 2l1;

                                     89
      While[rj > u1+l1 - lj,
            j++;
            uj = (a+1)/(a-1)uj;
            rj = rj-uj;
            li = lj;
            lj = rj/a;
            If[EvenQ[j], hj = hj+2li]
           ];
      Which[EvenQ[j],
            hj = hj+lj,
            OddQ[j],
            hj = hj+Max[li,2lj]
           ];
      Return[hj]
        ]


As explained in 3.2.4.1, the optimal folding point u1 is the u-coordinate of
the unique intersection point of h(a, u1 ) and w(a, u1 ). It may thus be found
again by a bisectional algorithm.


u0[a_, acc_] :=
  Block[{},
     b = a/(a+1);
     c = a/2;
     u1 = (b+c)/2;
     While[(c-b)/2 > acc/2,
       If[h[a,u1] > 1+(1-1/a)u1, b=u1, c=u1];
       u1 = (b+c)/2
          ];
     Return[u1]
      ]


Again, the choices b = aπ/(a + π) and c = a/2 reflect that we fold at least
twice in which case u1 ≥ l1 must hold true. Up to accuracy acc, the resulting
estimate sEC (a) is given by


sEC[a_, acc_] := 1+(1-1/a)u0[a,acc].

                                     90
B. Report on the Gromov width of closed symplectic mani-
folds
Recall that given any symplectic manifold (M 2n , ω) its Gromov width is
defined by

w(M, ω) = sup{c | there is a symplectic embedding (B 2n (c), ω0 ) ֒→ (M, ω)}.

Historically, the width provided the first example of a symplectic capac-
ity. Giving the size of the largest Darboux chart of (M, ω), the width is
always positive, and in the closed case it is finite. We now restrict to closed
manifolds and define an equivalent packing invariant by
                               |B 2n (w(M, ω))|    w(M, ω)n
               p(M 2n , ω) =                    =              .
                                  Vol(M, ω)       n! Vol(M, ω)
In two dimensions the width is the volume and p = 1 (see Theorem 4.2).
The basic result to discover rigidity in higher dimensions is a version of Gro-
mov’s Non-Squeezing Theorem [22].

Non-Squeezing Theorem (compact                             2n
                          2
                                      R version) Let (M , ω) be closed,
let σ be an area form on S such that S 2 σ = 1 and assume that there is a
symplectic embedding B 2n+2 (c) ֒→ (M × S 2 , ω ⊕ aσ). Then a ≥ c.
                                                    π
Remark. More generally, let S 2 ֒→ M ⋉ S 2 −      → M be an oriented S 2 -
bundle over a closed manifold M and let ω be a symplectic form on M ⋉ S 2
whose restriction to the fibers is nondegenerate and induces the given ori-
entation. In particular, a = h[ω], [pt × S 2 ]i > 0. Then the proof of the
above Non-Squeezing Theorem also implies that c ≤ a whenever B 2n+2 (c)
symplectically embeds in (M ⋉ S 2 , ω). We will verify this below in the case
where M is 2-dimensional.                                                  ✸

     Since the theory of J-holomorphic curves works best in dimension four,
the deepest results on the Gromov-width have been proved for 4-manifolds.
Given a symplectic 4-manifold (M, ω), let c1 be the first Chern class of
(M, ω) with respect to the contractible set of almost complex structures
compatible with ω. Let C be the class of symplectic 4-manifolds (M, ω) for
                                            Z
which there exists a class A ∈ H2 (M ; ) with non-zero Gromov invariant
and c1 (A) + A2 6= 0. Recall that a symplectic 4-manifold is called rational
if it is the symplectic blow-up of   CP
                                      2 and that it is said to be ruled if it
         2
is an S -bundle over a Riemann surface. The class C consists of symplectic
blow-ups of

                                       91
   • rational and ruled manifolds;

   • manifolds with b1 = 0 and b+
                                2 = 1;

                                             Z
   • manifolds with b1 = 2 and (H 1 (M ; ))2 6= 0.

We refer to [24] for more information on the class C.
    Recall that by definition an exceptional sphere in a symplectic 4-manifold
(M, ω) is a symplectically embedded 2-sphere S of self-intersection number
S ·S = −1, and that (M, ω) is said to be minimal if it contains no exceptional
spheres. Combining the technique of symplectic blowing-up with Taubes
theory of Gromov invariants, Biran [2, Theorem 6.A] showed that for the
symplectic 4-manifolds (M, ω) in class C all packing obstructions come from
exceptional spheres in the symplectic blow-up of (M, ω) and from the volume
constraint. His result suffices to compute the Gromov-width of all minimal
manifolds in the class C.

Theorem 5.1 (Biran [2, Theorem 2.F]) Let (M, ω) be a closed symplectic
4-manifold in the class C which is minimal and neither rational nor ruled.
Then p(M, ω) = 1.

Examples of manifolds satisfying the conditions of the above theorem are
hyper-elliptic surfaces and the surfaces of Barlow, Dolgachev and Enriques,
all viewed as Kähler surfaces.

    We next look at minimal manifolds which are rational or ruled.
    Let ωSF be the unique U(3)-invariant Kähler form on    CP   2 whose integral

over CP   1 equals π. In the rational case, by a theorem of Taubes [30], (M, ω)

is symplectomorphic to (  CP   2 , aω
                                      SF ) for some a > 0, thus p(M, ω) = 1.
    Denote by Σg the Riemann surface of genus g. There are exactly two ori-
entable S 2 -bundles with base Σg , namely the trivial bundle π : Σg ×S 2 → Σg
and the nontrivial bundle π : Σg ⋉ S 2 → Σg [26, Lemma 6.25]. Such a man-
                                                                  P        C
ifold is called a ruled surface. Σg ⋉ S 2 is the projectivization (L1 ⊕ ) of
                                    C
the complex rank two bundle L1 ⊕ over Σg , where L1 is a holomorphic line
bundle of Chern index 1. A symplectic form ω on a ruled surface is called
compatible with the given ruling π if it restricts on each fiber to a symplectic
form. Such a symplectic manifold is then called a ruled symplectic manifold.
It is known that every symplectic structure on a ruled surface is diffeomor-
phic to a form compatible with the given ruling π via a diffeomorphism
which acts trivially on homology, and that two cohomologous symplectic
forms compatible with the same ruling are isotopic [21]. A symplectic form

                                        92
ω on a ruled surface is thus determined up to diffeomorphism by the class
              R
[ω] ∈ H 2 (M ; ).
     Fix now an orientation of the fibers of the given ruled symplectic mani-
fold. We say that ω is admissible if its restriction to each fiber induces the
given orientation.
     Consider first the trivial bundle Σg × S 2 with its given orientation, and
                                                            Z
let {B = [Σg × pt], F = [pt × S 2 ]} be a basis of H 2 (M ; ) (here and hence-
forth we identify homology and cohomology via Poincaré duality). Then a
cohomology class c = bB + aF can be represented by an admissible form if
and only if c(B) = a > 0 and c(F ) = b > 0. We write Σg (a) × S 2 (b) for this
ruled symplectic manifold.
                                                                        Z
     In case of the nontrivial bundle Σg ⋉S 2 a basis of H 2 (Σg ⋉S 2 ; ) is given
by {A, F }, where A is the class of a section with selfintersection number
                                                 F
−1 and F is the fiber class. Set B = A + . {B, F } is then a basis of
                                                 2
              R
H 2 (Σg ⋉ S 2 ; ) with B · B = F · F = 0 and B · F = 1. It turns out that in
case g = 0 a form c = bB + aF can be represented by an admissible form
if and only if a > 2b > 0, while in case g ≥ 1 this is possible if and only if
a > 0 and b > 0 [26, Theorem 6.27]. We write (Σg ⋉ S 2 , ωab ) for this ruled
symplectic manifold.
     Finally note that each admissible form is cohomologous to a standard
Kähler form. For the trivial bundles these are just the split forms, and for
the non-trivial bundles we refer to [17, p. 276].
Theorem 5.2 Let (M 4 , ω) be a ruled symplectic manifold, i.e. either (M, ω) =
Σg (a) × S 2 (b) or (M, ω) = (Σg ⋉ S 2 , ωab ). If (M, ω) = S 2 (a) × S 2 (b) we may
assume that a ≥ b. Then
                                                      b
  (i) p(S 2 (a) × S 2 (b)) = p(S 2 ⋉ S 2 , ωab ) =   2a
                                                         b
 (ii) p(Σg (a) × S 2 (b)) = p(Σg ⋉ S 2 , ωab ) = min{1, 2a } if g ≥ 1
The statements for the trivial bundles are proved in [2, Theorem 6.1.A],
and the ones for the non-trivial bundles are calculated in [29]. Observe
that the upper bounds predicted by the Non-Squeezing Theorem and the
volume condition are sharp in all cases. Explicit maximal embeddings are
easily found for g = 0 and for g ≥ 1 if a ≥ b [29], but no explicit maximal
embeddings are known for g ≥ 1 if a < b.
                                                   b
    Also notice that p(S 2 (b) × Σg (a)) = min{1, 2a } if g ≥ 1 implies that the
Non-Squeezing Theorem does not remain valid if the sphere is replaced by
any other closed surface.


                                         93
    If (M 4 , ω) does not belong to the class C only very few is known about
p(M, ω). Indeed, no obstructions to full packings are known. Some flexibility
results for products of higher genus surfaces were found by Jiang.

Theorem 5.3 (Jiang [15, Corollary 3.3 and 3.4]) Let Σ be any closed surface
of area a > 1.
  (i) Let T 2 be the 2-torus. There is a constant C > 0 such that p(T 2 (1) ×
      Σ(a)) ≥ C.
 (ii) Let g ≥ 2. There is a constant C(g) > 0 depending only on g such
      that w(Σg (1) × Σ(a)) ≥ C(g) log a.

Remark. If Σ = S 2 Birans sharp result in Theorem 5.2 is of course much
better.                                                              ✸


Example 5.4 Set R(a) = {(x, y) ∈            R2 | 0   < x < 1, 0 < y < a}, and
consider the linear symplectic map
ϕ : (R(a) × R(a), dx1 ∧ dy1 + dx2 ∧ dy2 ) → (         R2 × R2 , dx1 ∧ dy1 + dx2 ∧ dy2)
                          (x1 , y1 , x2 , y2 ) 7→ (x1 + y2 , y1 , −y2 , y1 + x2 ).
       R               RZ RZ
Let p : 2 → T 2 = / × / be the projection onto the standard sym-
                                                      R
plectic torus. Then p ◦ ϕ : R(a) × R(a) → T 2 × 2 is an embedding; indeed,
given (x1 , y1 , x2 , y2 ) and (x′1 , y1′ , x′2 , y2′ ) with
                        x1 + y2 ≡ x′1 + y2′          mod   Z                   (132)
                            y1 ≡      y1′     mod    Z                         (133)
                           −y2 =      −y2′                                     (134)
                        y 1 + x2 =    y1′ +   x′2                              (135)
                                                               Z
(134) gives y2 = y2′ and thus (132) implies x1 ≡ x′1 mod whence x1 = x′1 .
Moreover, (133) and (135) show that y1 − y1′ = x′2 − x2 ≡ 0 mod , hence    Z
x2 = x′2 and y1 = y1′ .
   Next observe that p ◦ ϕ(R(a) × R(a)) ⊂ T 2 ×] − a, 0[×] − a − 1, a + 1[.
Thus R(a) × R(a) embeds in T 2 (1) × Σ(2a(a + 1)), and since B 4 (a) embeds
in R(a) × R(a) and B 4 (1) embeds in T 2 (1) × Σ(a) for any a ≥ 1, we have
shown

Proposition 5.5 Let a ≥ 1. Then
                                             √
                   2             max{a + 1 − 2a + 1, 2}
               p(T (1) × Σ(a)) ≥                        .
                                           4a

                                      94
In particular, the constant C in Theorem 5.3(i) can be chosen to be C = 1/8.
                                                                          ✸


     It would be interesting to have a complete list of those symplectic 4-
manifolds with p(M, ω) = 1. As we have seen above, the minimal such
manifolds in class C are those which are not ruled, the trivial bundles Σ(a)×
S 2 (b) with g(Σ) ≥ 1 and a ≥ 2b and the nontrivial bundles (Σ ⋉ S 2 , ωab )
with g(Σ) ≥ 1 and a ≤ 0. Combining the techniques of [2] with Donaldson’s
existence result for symplectic submanifolds, Biran [3] found examples with
p(M, ω) = 1 which do not belong to C.
    In higher dimensions almost no flexibility results are known. Note how-
                                               C                C
ever that for the standard Kähler form ωSF on P n we have p( P n , ωSF ) =
1 (see e.g. [25]), and that the technique used in Example 5.4 shows
                                                                R   that given
any constant form ω on T 2n and an area form σ on Σ with Σ σ = 1 there
is a constant C > 0 such that p(T 2n × Σ, ω ⊕ aσ) ≥ C ([15, Theorem 3.1]).


References
 [1] P. Biran. The Geometry of Symplectic Packing. Ph.D. thesis. Tel-Aviv
     University, 1997.
 [2] P. Biran. Symplectic packing in dimension 4. Geom. Funct. Anal. 7(3)
     (1997) 420-437.
 [3] P. Biran. A stability property of symplectic packing. Preprint 1997.
 [4] K. Cieliebak. Symplectic boundaries: creating and destroying closed
     characteristics. Geom. Funct. Anal. 7(2) (1997) 269-321.
 [5] M. Demazure. Surfaces de del Pezzo II-V. In: Séminaire sur les Sin-
     gularités des Surfaces (1976 - 1977), Lect. Notes Math., vol 777, pp.
     23-69, Springer 1980.
 [6] I. Ekeland and H. Hofer. Symplectic topology and Hamiltonian dynam-
     ics. Math. Z. 200 (1990) 355-378.
 [7] I. Ekeland and H. Hofer. Symplectic topology and Hamiltonian dynam-
     ics II. Math. Z. 203 (1990) 553-567.
 [8] Y. Eliashberg and H. Hofer. Unseen symplectic boundaries. Manifolds
     and geometry (Pisa, 1993) 178-189. Sympos. Math. XXXVI. Cambridge
     Univ. Press 1996.

                                     95
 [9] A. Floer and H. Hofer. Symplectic Homology I: Open sets in         Cn.
     Math. Z. 215 (1994) 37-88.

[10] A. Floer, H. Hofer and K. Wysocki. Applications of symplectic homol-
     ogy I. Math. Z. 217 (1994) 577-606.

[11] R. Greene and K. Shiohama. Diffeomorphisms and volume preserving
     embeddings of non-compact manifolds. Transactions of the American
     Mathematical Society 255 (1979) 403-414.

[12] M. Gromov. Pseudo-holomorphic curves in symplectic manifolds. In-
     vent. math. 82 (1985) 307-347.

[13] M. Gromov. Partial Differential Relations. Springer 1986.

[14] H. Hofer and E. Zehnder. Symplectic Invariants and Hamiltonian Dy-
     namics. Birkhäuser 1994.

[15] M.-Y. Jiang. Symplectic embeddings of      R2n   into some manifolds.
     Preprint Peking University (1997).

[16] S. Kobayashi and K. Nomizu. Foundations of Differential Geometry.
     Volume II, Interscience, New York 1969.

[17] F. Lalonde. Isotopy of symplectic balls, Gromov’s radius and the struc-
     ture of ruled symplectic 4-manifolds. Math. Ann. 300(2) (1994) 273–
     296.

[18] F. Lalonde and D. Mc Duff. The geometry of symplectic energy. Ann.
     of Math. 141 (1995) 349-371.

[19] F. Lalonde and D. Mc Duff. Hofer’s L∞ -geometry: energy and stability
     of Hamiltonian flows, part II. Invent. math. 122 (1995) 35-69.

[20] F. Lalonde and D. Mc Duff. Local non-squeezing theorems and stability.
     Geom. Funct. Anal. 5(2) (1995) 365-386.

[21] F. Lalonde and D. Mc Duff. The classification of ruled symplectic 4-
     manifolds. Math. Res. Lett. 3(6) (1996) 769–778.

[22] F. Lalonde, D. Mc Duff and L. Polterovich. In preparation.

[23] D. Mc Duff. Blowing up and symplectic embeddings in dimension 4.
     Topology 30(3) (1991) 409-421.

                                    96
[24] D. Mc Duff. From symplectic deformation to isotopy. Topics in symplec-
     tic 4-manifolds (Irvine, CA, 1996), 85–99, First Int. Press Lect. Ser., I,
     Internat. Press, Cambridge, MA, 1998.

[25] D. Mc Duff and L. Polterovich. Symplectic packings and algebraic ge-
     ometry. Invent. math. 115 (1994) 405-429.

[26] D. Mc Duff and D. Salamon. Introduction to Symplectic Topology. Ox-
     ford Mathematical Monographs, Clarendon Press 1995.

[27] K. Nomizu and H. Ozeki. The existence of complete Riemannian met-
     rics. Proc. Amer. Math. Soc. 12 (1961) 889-891.

[28] V. Ozols. Largest normal neighborhoods. Proc. Amer. Math. Soc. 61
     (1976) 99-101.

[29] F. Schlenk. Some new explicit packings of symplectic 4-manifolds. In
     preparation.

[30] C. Taubes. SW ⇒ Gr: from the Seiberg-Witten equations to pseudo-
     holomorphic curves. J. Amer. Math. Soc. 9(3) (1996) 845-918.

[31] L. Traynor. Symplectic packing constructions. J. Differential Geom. 42
     (1995) 411-429.

[32] L. Traynor. Symplectic homology via generating functions. Geom.
     Funct. Anal. 4(6) (1994) 718-748.

[33] H. Whitney. Differentiable manifolds. Ann. of Math. 37 (1936) 645-680.

   Felix Schlenk, Mathematik, ETH Zentrum, 8092 Zürich, Switzerland
   E-mail address: felix@math.ethz.ch




                                      97
