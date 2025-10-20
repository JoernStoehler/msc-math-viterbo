---
source: arXiv:1907.09749
fetched: 2025-10-20
---
# Symplectic homology of fiberwise convex sets and homology of loop spaces

                                                 SYMPLECTIC HOMOLOGY OF FIBERWISE CONVEX SETS AND
                                                             HOMOLOGY OF LOOP SPACES

                                                                                               KEI IRIE


                                                      Abstract. For any nonempty, compact and fiberwise convex set K in T ∗ Rn , we prove
                                                      an isomorphism between symplectic homology of K and a certain relative homology of
arXiv:1907.09749v3 [math.SG] 13 Jun 2021




                                                      loop spaces of Rn . We also prove a formula which computes symplectic homology capacity
                                                      (which is a symplectic capacity defined from symplectic homology) of K using homology
                                                      of loop spaces. As applications, we prove (i) symplectic homology capacity of any convex
                                                      body is equal to its Ekeland-Hofer-Zehnder capacity, (ii) a certain subadditivity property
                                                      of the Hofer-Zehnder capacity, which is a generalization of a result previously proved by
                                                      Haim-Kislev.




                                                                                         1. Introduction

                                           1.1. Symplectic homology and the capacity cSH . Let n be a positive integer. Let
                                           us consider coordinates q1 , . . . , qn , p1 , . . . , pn on T ∗ Rn , where q1 , . . . , qn are coordinates on
                                           Rn and p1 , . . . , pn are coordinates on fibers with respect to the global frameX                dq1 , . . . , dqn .
                                           We often abbreviate (q1 , . . . , qn ) by q and (p1 , . . . , pn ) by p. Let ωn :=                    dpi dqi ∈
                                                                                                                                             1≤i≤n
                                             2    ∗   n                                                       ∗   n
                                           Ω (T R ). For any nonempty compact set K ⊂ T R and real numbers a < b, one can
                                           define a Z-graded Z/2-vector space SH [a,b)
                                                                                 ∗     (K), which is called symplectic homology (see
                                           Section 2.2 for details).
                                           Remark 1.1. Throughout this paper, all (co)homology groups are defined over Z/2Z,
                                           unless otherwise specified.

                                              When K satisfies certain nice conditions, we say that K is a restricted contact type
                                           (RCT) set (see Definition 2.6; note that our definition of RCT sets is slightly more gener-
                                           alized than the usual definition). Any compact star-shaped (in particular, convex) set is
                                           a RCT set (Lemma 2.8). For any RCT set K ⊂ T ∗ Rn and a ∈ R>0 , there exists a natural
                                           linear map
                                                                  iaK : H∗+n (T ∗ Rn , T ∗ Rn \ K) → SH [0,a)
                                                                                                        ∗     (K).
                                           See Section 2.3 for the definition of iaK . Also, as we define in Section 2.4, there exists
                                                                 T ∗ Rn
                                           a canonical element νK       ∈ H2n (T ∗ Rn , T ∗ Rn \ K). Then let us define the following
                                           numerical invariant:
                                                                                                                      ∗ Rn
                                                                                                         T
                                                                          cSH (K) := inf{a ∈ R>0 | iaK (νK                   ) = 0}.
                                           In this paper, the invariant cSH is called symplectic homology capacity.

                                              Date: June 15, 2021.
                                                                                                    1
Remark 1.2. The first symplectic capacity defined from symplectic homology was intro-
duced by Floer-Hofer-Wysocki [9], who defined a capacity (denoted by cFHW ) for arbitrary
open sets in the symplectic vector space. The above definition of cSH is due to Hermann
[14], which is based on the idea by Viterbo [23] (see Section 5.3 of [23]). Indeed, Hermann
(Proposition 5.7 of [14]) proved that (in the language of the present paper) any C ∞ -RCT
set K (see Definition 2.6) satisfies cSH (K) = cFHW (int (K)). Here int (K) denotes the
interior of K.

  Although symplectic homology and the capacity cSH are fundamental quantitative in-
variants of subsets of the symplectic vector space, they are notoriously difficult to com-
pute, or even to estimate. This is because symplectic homology is a version of Floer
homology, whose definition involves counting solutions of nonlinear PDEs (so called Floer
equations), thus it is very difficult to compute these invariants directly from definitions.
The core results of this paper, which we discuss in Section 1.2, enable us to investigate
these invariants via computations of homology of loop spaces.

1.2. Main results. The core results of this paper are Theorem 3.4 and Corollary 3.8.
Corollary 3.8 has two applications: Theorem 1.4 and Theorem 1.8. The goal of this
subsection is to describe these four results.
  Theorem 3.4 shows that, for any nonempty compact set K ⊂ T ∗ Rn which is fiberwise
convex (i.e. K ∩Tq∗ Rn is convex for every q ∈ Rn ), symplectic homology of K is isomorphic
to a certain relative homology of loop spaces of Rn . Theorem 3.4 is a version of the well-
known isomorphism between Floer homology of cotangent bundles and homology of loop
spaces. Indeed, the proof of Theorem 3.4 heavily relies on the proof by Abbondandolo-
Schwarz [3] of this isomorphism.
   Corollary 3.8, which is an easy consequence of Theorem 3.4, shows that if K is a RCT
set then cSH (K) is equal to a certain min-max value defined from homology of loop spaces.
In the rest of this subsection, we present two applications of Corollary 3.8: Theorem 1.4
and Theorem 1.8.
  To state Theorem 1.4, let us recall the definition of the Ekeland-Hofer-Zehnder capacity
(which we denoted by cEHZ ) of convex bodies. For definitions of “symplectic action” and
“closed characteristics”, see Section 2.3.
Definition 1.3. K ⊂ T ∗ Rn is called a convex body if K is compact, convex, and int (K) 6=
∅. When ∂K is a C ∞ -hypersurface, then its Ekeland-Hofer-Zehnder capacity cEHZ (K) is
defined as the minimum symplectic action of closed characteristics on ∂K. For arbitrary
convex body K, we define
cEHZ (K) := inf{cEHZ (K ′ ) | K ′ is a convex body with C ∞ -boundary such that K ⊂ K ′ }.

  Now let us state our first application of Corollary 3.8:
Theorem 1.4. cSH (K) = cEHZ (K) for any convex body K ⊂ T ∗ Rn .
Remark 1.5.       • Theorem 1.4 is also proved by Abbondandolo-Kang [1]. Their proof
     is based on an isomorphism (which is the main result of [1]) between the filtered
     Floer complex of a convex quadratic Hamiltonian on T ∗ Rn (satisfying some techni-
     cal conditions) and the filtered Morse complex of its Clarke dual action functional.
                                             2
     • Using S 1 -equivairiant symplectic homology, one can define a sequence of capacities
       (ckSH S 1 )k≥1. Felix Schlenk [21] pointed out that, assuming some standard properties
       of these capacities, Theorem 1.4 implies c1SH S 1 (K) = cSH (K) for any convex body
       K ⊂ T ∗ Rn ; see Section 2.5 for details.

  Theorem 1.4 is motivated by the following folk conjecture, which says that all symplectic
capacities on T ∗ Rn coincide for convex bodies (see Section 5 of [20] and the references
therein):
Conjecture 1.6. Let c be any symplectic capacity on T ∗ Rn ; namely, c is a map from the
set of all subsets of T ∗ Rn to [0, ∞] which satisfies the following three properties:

     • For any S ⊂ T ⊂ T ∗ Rn , there holds c(S) ≤ c(T ).
     • For any S ⊂ T ∗ Rn , a ∈ R>0 and ϕ ∈ Diff (T ∗ Rn ) such that ϕ∗ ωn = aωn , there
       holds c(ϕ(S)) = ac(S).
     • c({(q, p) ∈ T ∗ Rn | |q|2 + |p|2 ≤ 1}) = c({(q, p) ∈ T ∗ Rn | q12 + p21 ≤ 1}) = π.

Then c(K) = cEHZ (K) for any convex body K.

  Conjecture 1.6 is still widely open. As far as the author knows, Conjecture 1.6 was
verified only for the first equivariant Ekeland-Hofer capacity and the Hofer-Zehnder ca-
pacity. The result for the first equivariant Ekeland-Hofer capacity was mentioned by
Viterbo (Proposition 3.10 of [22]), and a detailed proof can be found in Section 6 of Gutt-
Hutchings-Ramos [12]. The result on the Hofer-Zehnder capacity is due to Hofer-Zehnder
[15]. Theorem 1.4 verifies Conjecture 1.6 for the symplectic homology capacity cSH .
  Our second application of Corollary 3.8 is a certain subadditivity property of the Hofer-
Zehnder capacity. Let us recall the definition of the Hofer-Zehnder capacity:
Definition 1.7. H ∈ Cc∞ (T ∗ Rn , R≥0 ) is called Hofer-Zehnder admissible if there exists a
nonempty open set U ⊂ T ∗ Rn such that H|U ≡ max H, and every nonconstant periodic
orbit of its Hamiltonian vector field XH (see the first paragraph of Section 2 for our
convention) has period strictly larger than 1. Let Had denote the set of all Hofer-Zehnder
admissible functions on (T ∗ Rn , ωn ). For any S ⊂ T ∗ Rn such that int (S) 6= ∅, its Hofer-
Zehnder capacity cHZ (S) ∈ R>0 is defined as
                     cHZ (S) := sup{max H | H ∈ Had , supp H ⊂ S}.

  Now we can state our second application of Corollary 3.8:
Theorem 1.8. Let K be any compact set in T ∗ Rn with int (K) 6= ∅, and Π be any
hyperplane in T ∗ Rn which intersects int (K). Let Π+ and Π− be distinct closed halfspaces
such that ∂Π+ = ∂Π− = Π. Then, setting K + := K ∩ Π+ and K − := K ∩ Π− , there holds
                     cHZ (K) ≤ cEHZ (conv (K + )) + cEHZ (conv (K − )),
where conv denotes the convex hull.

  Theorem 1.8 can be rephrased as follows: for any K and Π such that K + and K − are
convex, cHZ (K) ≤ cEHZ (K + ) + cEHZ (K − ). In particular, we recover the following result
by Haim-Kislev [13] as a corollary:
                                              3
Corollary 1.9 ([13] Theorem 1.8). Let K be any convex body in T ∗ Rn and Π be any
hyperplane in T ∗ Rn which intersects int (K). Then, cEHZ (K) ≤ cEHZ (K + ) + cEHZ (K − ).

  The proof in [13] uses a combinatorial formula (Theorem 1.1 of [13]) which computes
the EHZ capacity of convex polytopes, and it seems difficult to extend this proof to prove
Theorem 1.8 when K is not convex.
  Theorem 1.8 is inspired by the following conjecture by Akopyan-Karasev-Petrov [5]:
                                                                                  m
                                                                                  [
                                                                          ∗   n
Conjecture 1.10 ([5]). Let K, K1, . . . , Km be convex bodies in T R . If K ⊂           Ki , then
                                                                                  i=1
             m
             X
cEHZ (K) ≤         cEHZ (Ki ).
             i=1


  In [5], Conjecture 1.10 was verified for hyperplane cuts of round balls, which was later
generalized to hyperplane cuts of arbitrary convex bodies (Corollary 1.9). Note that the
convexity of K1 , . . . , Km is essential in Conjecture 1.10, as shown by examples in Section
5.1 of [5], for which the subadditivity fails without the convexity assumption. Let us also
mention the following Proposition 1.11, which gives another such example. The proof of
Proposition 1.11, which we explain in Section 7, is elementary.
Proposition 1.11. Let n ≥ 2 be an integer. For any bounded B ⊂ T ∗ Rn and any
ε ∈ R>0 , there are compact star-shaped sets K1 , K2 ⊂ T ∗ Rn such that B ⊂ K1 ∪ K2 and
e(K1 ), e(K2 ) < ε, where e denotes the Hamiltonian displacement energy.

  On the other hand, it seems unknown if the following conjecture, which is stronger than
Conjecture 1.10, holds true.
Conjecture 1.12. For any convex bodies K1 , . . . , Km in T ∗ Rn ,
                                       m
                                       [              m
                                                      X
                                              
                                 cHZ         Ki ≤           cEHZ (Ki ).
                                       i=1            i=1


   As far as the author knows, Theorem 1.8 is the first verification of Conjecture 1.12 in
a situation not covered by Conjecture 1.10.

1.3. Structure of this paper. Let us explain the structure of this paper. In Section
2 we review basics of symplectic homology. In particular, we recall the definition of the
capacity cSH and explain its basic properties. In Section 3, we state Theorem 3.4, and
deduce Corollary 3.8 from Theorem 3.4. Section 4 is devoted to the proof of Theorem
3.4, which is based on the “hybrid moduli space” method of Abbondandolo-Schwarz [3].
The outline of the proof is sketched in the first paragraph of Section 4. Section 4 is
the most technical section, and can be skipped at the first reading. In Section 5 we
prove Theorem 1.4, and in Section 6 we prove Theorem 1.8. Using Corollary 3.8, these
results can be proved by elementary arguments about loop spaces. In particular, the key
estimate is Lemma 5.7. In Section 7, we prove Proposition 1.11. This section can be read
independently from Sections 2–6.
                                                  4
   Acknowledgement. The author thanks Felix Schlenk for pointing out an application
discussed in Section 2.5, and his comments on an earlier version of this paper. The author
also thanks Alberto Abbondandolo and Jungsoo Kang for sharing their manuscript [1] and
having discussions about relations between their approach and the author’s. Finally, the
author thanks the referee for many comments which are very helpful to improve readability
of this paper. This research is supported by JSPS KAKENHI Grant No.18K13407 and
No.19H00636.


                  2. Symplectic homology and the capacity cSH

  For any h ∈ C ∞ (T ∗ Rn ), its Hamiltonian vector field Xh ∈ X (T ∗ Rn ) is defined by
ωn (Xh , · ) = −dh( · ). Let S 1 := R/Z. For any H ∈ C ∞ (S 1 × T ∗ Rn ) and t ∈ S 1 , we define
Ht ∈ C ∞ (T ∗ Rn ) by Ht (q, p) := H(t, q, p). Let

                 P(H) := {γ : S 1 → T ∗ Rn | γ̇(t) = XHt (γ(t)) (∀t ∈ S 1 )}.

γ ∈ P(H) is called nondegenerate if 1 is not an eigenvalue of (dϕ1H )γ(0) , where (ϕtH )0≤t≤1
denotes the Hamiltonian isotopy generated by H.
Remark 2.1. The isotopy (ϕtH )0≤t≤1 may not be globally defined, but it is defined at
least on a neighborhood of γ(0).

2.1. Filtered Floer homology. In this subsection, we review basic facts about filtered
Floer homology of (time-dependent) Hamiltonians on Cn which are compact perturbations
of quadratic functions. The results in this subsection are essentially contained in [7].
However, here we mainly follow [18], since the class of Hamiltonians we consider is slightly
different from that in [7].
  For any H ∈ C ∞ (S 1 × T ∗ Rn ) we consider the following conditions:

 (H0): Every γ ∈ P(H) is nondegenerate.
 (H1): There exist A ∈ R>0 \ πZ and B ∈ R such that the function

                     H(t, q, p) − A(|q|2 + |p|2) − B ∈ C ∞ (S 1 × T ∗ Rn )

        is compactly supported.

   In the following we assume that H ∈ C ∞ (S 1 × T ∗ Rn ) satisfies (H0) and (H1). Note
that (H1) implies that all elements of P(H) are contained in a compact subset of T ∗ Rn .
This is because on the complement of a sufficiently large compact set, every orbit of XH
                                            π
is periodic with the minimal period equal to . By A ∈ / πZ, there exists no periodic orbit
                                            A
with period 1 on the complement. Moreover (H0) implies that P(H) is discrete, thus it
is finite.
                                                        [a,b)
  For any real numbers a < b and k ∈ Z, let CF k                (H) denote the Z/2-vector space
spanned by
                       {γ ∈ P(H) | AH (γ) ∈ [a, b), ind CZ (γ) = k}.
                                               5
Here, ind CZ denotes the Conley-Zehnder index (see Section 1.3 of [7]) and AH is defined
by                               Z      ÅX       ã
                                      ∗
                       AH (γ) :=    γ      pi dqi − Ht (γ(t)) dt.
                                          S1         i


  To define a boundary operator on CF [a,b)
                                        ∗   (H), we take J = (Jt )t∈S 1 , which is a C ∞ -
                                        ∗ n
family of almost complex structures on T R with the following condition:

 (J1): For every t ∈ S 1 , Jt is compatible with respect to ωn . Namely, gJt (v, w) :=
       ωn (v, Jt w) is a Riemannian metric on T ∗ Rn .

For any J satisfying (J1) and x− , x+ ∈ P(H), we define
           MH,J (x− , x+ ) := {u : R × S 1 → T ∗ Rn | ∂s u − Jt (∂t u − XHt (u)) = 0,
                             lim us = x± }.
                            s→±∞

Here s denotes the coordinate on R, t denotes the coordinate on S 1 , and us : S 1 → T ∗ Rn
is defined by us (t) := u(s, t). We set M¯H,J (x− , x+ ) := MH,J (x− , x+ )/R, where the R
action on MH,J (x− , x+ ) is defined by
                  (r · u)(s, t) := u(s − r, t)               (u ∈ MH,J (x− , x+ ), r ∈ R).

  Let us define the standard complex structure on T ∗ Rn , which is denoted by Jstd , by
                    Jstd (∂pi ) = ∂qi ,        Jstd (∂qi ) = −∂pi           (1 ≤ i ≤ n).
Lemma 2.2. Suppose H satisfies (H0) and (H1), J satisfies (J1), and sup kJt − Jstd kC 0
                                                                                           t∈S 1
is sufficiently small. Then         sup           |u(s, t)| < ∞.
                                x− ,x+ ∈P(H)
                               u∈MH,J (x− ,x+ )
                                 (s,t)∈R×S 1


Proof. This lemma follows from Lemma 2.3 in [18]; note that conditions (H0), (J1) in
[18] are the same as (H0), (J1) in this paper, and the condition (H1) in [18] is weaker
than (H1) in this paper.                                                             

   For a generic (with respect to the C ∞ -topology) choice of J, the moduli space M¯H,J (x− , x+ )
is cut out transversally for any pair (x− , x+ ). For any such J, M¯H,J (x− , x+ ) is a finite set
if ind CZ (x+ ) = ind CZ (x− ) − 1, and the linear map
                               [a,b)
                                                                      #2 M¯H,J (x− , x+ ) · x+
                                                        X
     ∂H,J : CF [a,b)
               ∗     (H) → CF  ∗−1   (H); x− →
                                             7
                                                         ind CZ (x+ )=ind CZ (x− )−1
          2
satisfies∂H,J  = 0, where #2 denotes the cardinality modulo 2. The homology of the chain
complex (CF [a,b)
               ∗   (H), ∂H,J ) does not depend on the choice of J. This homology is denoted
by HF ∗ (H) and called filtered Floer homology of H. For any a, b, a′ , b′ ∈ R with a < b,
         [a,b)
                                                                                       ′ ,b′ )
a′ < b′ , a ≤ a′ and b ≤ b′ , one can define a natural linear map HF ∗[a,b) (H) → HF [a
                                                                                     ∗         (H).
                                                                               2
Remark 2.3. As we remarked at the beginning of this subsection, the fact ∂H,J      = 0, as
well as the independence of the homology on the choice of J, are due to [7] and references
therein.
                                                         6
  Suppose that H − , H + ∈ C ∞ (S 1 × T ∗ Rn ) satisfy (H0), (H1) and
(1)                 H − (t, q, p) < H + (t, q, p)       (∀(t, q, p) ∈ S 1 × T ∗ Rn ).
Then, for any real numbers a < b one can define a linear map (called monotonicity map)
                                   HF [a,b)
                                      ∗     (H − ) → HF [a,b)
                                                        ∗     (H + )
 as follows. First, we take J − = (Jt− )t∈S 1 and J + = (Jt+ )t∈S 1 such that J − defines a
 boundary map on CF ∗ (H − ) and J + defines a boundary map on CF ∗ (H + ). Next, we take
 a C ∞ -family of Hamiltonians H = (Hs,t )(s,t)∈R×S 1 and a C ∞ -family of almost complex
 structures J = (Js,t)(s,t)∈R×S 1 such that the following conditions hold:
                                                       ® −
                                                         H (t, q, p) (s ≤ −s0 )
(HH1): There exists s0 > 0 such that Hs,t (q, p) =
                                                         H + (t, q, p) (s ≥ −s0 ).
(HH2): ∂s Hs,t (q, p) ≥ 0 for any (s, t, q, p) ∈ R × S 1 × T ∗ Rn .
(HH3): There exist a(s), b(s) ∈ C ∞ (R) such that the following conditions hold:
           • a′ (s) ≥ 0 for any s.
           • a(s) ∈ πZ =⇒ a′ (s) > 0.
           • Setting ∆s,t (q, p) := H(s, t, q, p) − a(s)(|q|2 + |p|2 ) − b(s), there holds
                   sup k∆s,t kC 1 (T ∗ Rn ) < ∞,        sup k∂s ∆s,t kC 0 (T ∗ Rn ) < ∞.
                   (s,t)                                (s,t)
                                                ® −
                                                 Jt (s ≤ −s1 )
 (JJ1): There exists s1 > 0 such that Js,t =
                                                 Jt+ (s ≥ s1 ).
 (JJ2): For every (s, t) ∈ R × S 1 , Js,t is compatible with ωn .
Remark 2.4. For any H − and H + satisfying (H0), (H1) and (1), there exists H =
(Hs,t )(s,t)∈R×S 1 satisfying (HH1), (HH2) and (HH3), as we explained in pp.517 of [18]. Let
us repeat the explanation for the convenience of the reader. Take ρ ∈ C ∞ (R) such that
ρ|R≤0 ≡ 0, ρ|R≥1 ≡ 1 and 0 < ρ(s) < 1, ρ′ (s) > 0 for any 0 < s < 1. Then let us define
H = (Hs,t)(s,t)∈R×S 1 by
                     Hs,t (q, p) := (1 − ρ(s))H − (t, q, p) + ρ(s)H + (t, q, p).
On the other hand, the existence of J = (Js,t)(s,t)∈R×S 1 satisfying (JJ1) and (JJ2) is
straightforward from the fact that the set of almost complex structures compatible with
ωn is contractible.

  For any H = (Hs,t )(s,t)∈R×S 1 and J = (Js,t)(s,t)∈R×S 1 satisfying the above conditions,
and for any x− ∈ P(H − ) and x+ ∈ P(H + ), we consider the moduli space
 MH,J (x− , x+ ) := {u : R × S 1 → T ∗ Rn | ∂s u − Js,t (∂t u − XHs,t (u)) = 0, lim us = x± }.
                                                                                           s→±∞

Lemma 2.5. Suppose that H satisfies (HH1), (HH2) and (HH3). If J satisfies (JJ1),
(JJ2) and sup kJs,t − Jstd kC 0 is sufficiently small, then
           (s,t)∈R×S 1

                                         sup              |u(s, t)| < ∞.
                                x− ∈P(H − ),x+ ∈P(H + )
                                    u∈MH,J (x− ,x+ )
                                      (s,t)∈R×S 1


Proof. See Lemma 2.4 in [18].                                                                     
                                                    7
   For a generic choice of (H, J) which satisfies the assumptions in Lemma 2.5, MH,J (x− , x+ )
is cut out transversally for any pair (x− , x+ ). In particular, MH,J (x− , x+ ) is a finite set
if ind CZ (x+ ) = ind CZ (x− ), and the linear map
                                                        X
                      −
      Φ : CF [a,b)
               ∗   (H   ) → CF  [a,b)
                                ∗     (H +
                                           ); x− →
                                                 7                 #2 MH,J (x− , x+ ) · x+
                                                         ind CZ (x+ )=ind CZ (x− )

satisfies ∂H + ,J + ◦ Φ = Φ ◦ ∂H − ,J − . The induced map on homology
                               H∗ (Φ) : HF [a,b)
                                           ∗     (H − ) → HF ∗[a,b) (H + )
does not depend on the choice of (H, J); see Section 4.3 of [7]. This completes the
definition of the monotonicity map.
  For any H 0 , H 1, H 2 ∈ C ∞ (S 1 × T ∗ Rn ) satisfying (H0), (H1) and
             H 0 (t, q, p) < H 1 (t, q, p) < H 2 (t, q, p)             (∀(t, q, p) ∈ S 1 × T ∗ Rn ),
the diagram
                            [a,b)                                                 [a,b)
                        HF ∗        (H 0 )                                 /   HF ∗       (H 2 )
                                       ❖❖❖
                                          ❖❖❖                               ♦♦7
                                             ❖❖❖                       ♦♦♦♦♦
                                                ❖❖'                   ♦
                                                                   ♦♦♦
                                                      [a,b)
                                                 HF ∗         (H 1 )
commutes (all three maps are monotonicity maps).

2.2. Symplectic homology. For any nonempty compact set K in T ∗ Rn , let HK denote
the set of H ∈ C ∞ (S 1 × T ∗ Rn ) which satisfies (H0), (H1) and H(t, q, p) < 0 for any
(t, q, p) ∈ S 1 × K. Then HK becomes a directed set by setting H 0 < H 1 if and only if
H 0 (t, q, p) < H 1 (t, q, p) for any (t, q, p) ∈ S 1 × T ∗ Rn . For any real numbers a < b, we set
                                     SH ∗[a,b) (K) := lim HF [a,b)
                                                             ∗     (H),
                                                      −→
                                                       H∈HK

where the limit is taken by monotonicity maps.
  For any a, b, a′ , b′ ∈ R with a < b, a′ < b′ , a ≤ a′ , b ≤ b′ , and nonempty compact sets
  ′                                                                       ′ ,b′ )
K ⊂ K, one can define a natural linear map SH [a,b)     ∗    (K) → SH [a∗         (K ′ ). Also, for any
c ∈ R>0 one can define a natural isomorphism
                                 SH [a,b) (K) ∼
                                                     2   2
                                             ∗= SH [c a,c b) (cK).
                                                               ∗

                                                                     (H) ∼
                                                                                 2 a,c2 b)
This follows from an isomorphism of filtered Floer homology HF [a,b)
                                                               ∗         = HF [c
                                                                              ∗            (Hc ),
                 2
where Hc (x) := c H(x/c).

2.3. Symplectic homology of RCT sets. Let us start from our definition of RCT
(restricted contact type) sets:
Definition 2.6. Let K be a compact subset of T ∗ Rn .

      • K is called a C ∞ -RCT set, if K is connected, int K 6= ∅, ∂K is of C ∞ , and there
        exists X ∈ X (T ∗ Rn ) which satisfies the following properties:
          – LX ωn ≡ ωn ,
          – X points strictly outwards at every point on ∂K.
                                                        8
     • K is called a RCT set, if there exists a sequence (Ki )i≥1 which satisfies the following
       properties:
         – Ki is a C ∞ -RCT set for every i,
         – Ki+1 ⊂ Ki for every i,
            ∞
            \
         –     Ki = K.
            i=1

Remark 2.7. Usually, “restricted contact type domain” is defined as a domain (i.e.
connected open set) such that its closure is a C ∞ -RCT set in the above sense (see e.g.
Definition 1.3 in [14]). Thus, the above definition of RCT set is slightly more generalized
than the usual definition.

  K ⊂ T ∗ Rn is called star-shaped if there exists x ∈ K such that ty + (1 − t)x ∈ K for
any y ∈ K and t ∈ [0, 1]. In particular any convex set is star-shaped.
Lemma 2.8. Any compact and star-shaped set in T ∗ Rn is a RCT set.

Proof. Suppose that K ⊂ T ∗ Rn is compact and star-shaped. We may assume that
(0, . . . , 0) ∈ K and ty ∈ K for any t ∈ [0, 1] and y ∈ K. Let S := {(q, p) ∈ T ∗ Rn |
|q|2 + |p|2 = 1}. Then there exists a function f : S → R≥0 such that
                              K = {ty | y ∈ S, 0 ≤ t ≤ f (y)}.

 It is easy to see that f is upper semi-continuous. Thus there exists a sequence (fj )j≥1 in
 ∞
C (S, R>0 ) such that fj (y) > fj+1 (y) for every y ∈ S and j ≥ 1, and f (y) = lim fj (y).
                                                                                    j→∞
For every j ≥ 1, Kj := {ty | y ∈ S, 0 ≤ t ≤ fj (y)} is a C ∞ -RCT set, since X :=
   n
1X
      pi ∂pi + qi ∂qi satisfies LX ωn = ωn , and is transversal to ∂Kj . Then (Kj )j≥1 is a
2 i=1
                                                   \∞
                             ∞
decreasing sequence of C -RCT sets satisfying          Kj = K, thus K is a RCT set.      
                                                     j=1


   Let K be a C ∞ -RCT set in T ∗ Rn . The distribution ker(ωn |∂K ) on ∂K defines a 1-
dimensional foliation of ∂K, which is called the characteristic foliation of ∂K. Closed
characteristics are closed leaves of this foliation which are diffeomorphic to S 1 . Let
P(∂K) denote the set of closed characteristics. The distribution ker(ωn |∂K ) is oriented
so that v ∈ ker(ωn |∂K ) is positive if and only if ωn (X, v) > 0, where X is any vector on
∂K which points strictly outwards. With this orientation, for each γ ∈ P(∂K) we define
its symplectic action A (γ) by
                                            Z ÅX           ã
                                  A (γ) :=           pi dqi .
                                             γ       i

Lemma 2.9. Let K be any C -RCT set in T ∗ Rn . Then every γ ∈ P(∂K) satisfies
                                 ∞

A (γ) > 0. Moreover, there exists γ0 ∈ P(∂K) such that A (γ0 ) = inf A (γ).
                                                                       γ∈P(∂K)


Proof. By definition of C ∞ -RCT sets, there exists X ∈ X (T ∗ Rn ) which satisfies LX ωn =
ωn and points strictly outwards on ∂K. Let us define λ ∈ Ω1 (T ∗ Rn ) by λ := iX ωn . Then
                                                 9
λ is a contact form on ∂K, and when Rλ denotes its Reeb vector field (i.e. iRλ (dλ) ≡ 0
and λ(Rλ ) ≡ 1), P(∂K) is the set of simple closed orbits of Rλ . Moreover, for every
γ ∈ P(∂K), A (γ) is equal to the period of γ as an orbit of Rλ . Then inf A (γ) is
                                                                                                           γ∈P(∂K)
positive, since ∂K is compact and Rλ is nonzero at every point on ∂K. To show that
there exists a closed orbit which attains the infimum, let (γj )j≥1 be a sequence in P(∂K)
such that A (γj ) converges to the infimum as j → ∞. Let us take pj on γj for each j,
and let p be the limit of a certain subsequence of (pj )j . Then the orbit γ0 which passes
through p is closed, and A (γ0 ) is equal to the infimum.                                

  For any C ∞ -RCT set K ⊂ T ∗ Rn , we denote cmin (K) :=                                         min A (γ). When K is
                                                                                                 γ∈P(∂K)
convex, cmin (K) is also denoted by cEHZ (K) (see Definiton 1.3).
Lemma 2.10. For any C ∞ -RCT set K ⊂ T ∗ Rn and ε ∈ (0, cmin (K)), one can assign an
isomorphism SH ∗[0,ε) (K) ∼
                          = H∗+n (T ∗ Rn , T ∗Rn \ K) so that the diagram
                     H∗+n (T ∗ Rn , T ∗ Rn \ K)        /   H∗+n (T ∗ Rn , T ∗ Rn \ K ′ )
                               ∼
                               =                                                         ∼
                                                                                         =
                                                                                
                               [0,ε)                                        [0,ε′ )
                            SH ∗            (K)                      /   SH ∗            (K ′ )
commutes for any C ∞ -RCT sets K ′ ⊂ K and 0 < ε ≤ ε′ < min{cmin(K), cmin (K ′ )}.

Proof. The isomorphism SH ∗[0,ε)(K) ∼  = H∗+n (K, ∂K) ∼  = H∗+n (T ∗ Rn , T ∗ Rn \ K) follows
from the third bullet in Proposition 4.7 of [14]. The commutativity of the diagram follows
from the construction of this isomorphism.                                                 
Remark 2.11. For any convex body K and ε ∈ (0, cEHZ(K)), there exists a natural
isomorphism SH ∗[0,ε) (K) ∼   = H∗+n (T ∗ Rn , T ∗Rn \ K) obtained as
  SH ∗[0,ε) (K) ∼
                = lim SH ∗[0,ε)(K ′ ) ∼
                                      = lim H (T ∗ Rn , T ∗Rn \ K ′ ) ∼
                                                                      = H∗+n (T ∗ Rn , T ∗ Rn \ K),
                  −→′                   −→′ ∗+n
                 K                            K

where K ′ runs over all convex bodies with C ∞ boundaries such that K ′ ⊃ K. The second
isomorphism holds since cEHZ (K ′ ) > ε, which follows from the monotonicity of the EHZ
capacity cEHZ (K ′ ) ≥ cEHZ (K).

  By Lemma 2.10, for any C ∞ -RCT set K we obtain an isomorphism
                      H∗+n (T ∗ Rn , T ∗ Rn \ K) ∼
                                                 = lim SH ∗[0,ε) (K).
                                                   ←−
                                                              ε→0

Then, for any a ∈ R>0 , we can define a linear map
              iaK : H∗+n (T ∗ Rn , T ∗ Rn \ K) ∼
                                               = lim SH ∗[0,ε)(K) → SH [0,a)
                                                                       ∗     (K).
                                                 ←−
                                                   ε→0

The following diagram commutes for any C -RCT sets K ′ ⊂ K and a ≤ a′ :
                                                  ∞


(2)                  H∗+n (T ∗ Rn , T ∗ Rn \ K)        /   H∗+n (T ∗ Rn , T ∗ Rn \ K ′ )
                                                                                             ′
                               ia
                                K                                                        ia
                                                                                          K′
                                                                                    
                               [0,a)                                        [0,a′ )
                           SH ∗             (K)                  /       SH ∗            (K ′ ).
                                                  10
Also, the following diagram commutes for any c ∈ R>0 :
                                                       ∼
                                                       =
(3)                H∗+n (T ∗ Rn , T ∗ Rn \ K)                  /   H∗+n (T ∗ Rn , T ∗ Rn \ cK)
                                                                                            2
                             ia
                              K                                                           iccKa
                                                                                     
                             [0,a)                                            [0,c2 a)
                          SH ∗ (K)                    ∼
                                                                       /   SH ∗        (cK).
                                                      =


 Now let us define the map iaK : H∗+n (T ∗ Rn , T ∗ Rn \ K) → SH [0,a)
                                                                  ∗    (K) for any RCT set
K and a ∈ R>0 . Notice that there are natural isomorphisms
                H∗+n (T ∗ Rn , T ∗ Rn \ K) ∼
                                           = lim H (T ∗ Rn , T ∗ Rn \ K ′ ),
                                             −→′ ∗+n
                                                          K

                                      SH [0,a) (K)   ∼
                                                     = lim SH [0,a) (K ′ ),
                                         ∗             −→     ∗
                                                          K′

where K ′ runs over all C ∞ -RCT sets with K ′ ⊃ K. Then one can define iaK as the limit
of (iaK ′ )K ′ ⊃K .

2.4. Symplectic homology capacity cSH . To define the capacity cSH , we first need the
following definition. Recall that, in this paper all (co)homology groups are defined over
Z/2, unless otherwise specified.
Definition 2.12. For any R-vector space V of dimension d ∈ Z>0 and a compact subset
                   V
K ⊂ V , we define νK ∈ Hd (V, V \ K) in the following manner.

      • If K is convex, then Hd (V, V \ K) ∼   = Z/2. Then we define νK      V
                                                                               to be the unique
        non-zero element of Hd (V, V \ K).
      • When K is an arbitrary compact subset of V , take a compact convex set K ′ ⊂ V
        satisfying K ⊂ K ′ , and let iKK ′ : Hd (V, V \ K ′ ) → Hd (V, V \ K) be the linear map
        induced by id V : (V, V \ K ′ ) → (V, V \ K). Then it is easy to see that iKK ′ (νK  V
                                                                                               ′)

        does not depend on the choice of K ′ . Then we define νK     V             V
                                                                        := iKK ′ (νK ′ ).



  Now, for any RCT set K ⊂ T ∗ Rn , we define
                                                                              ∗ Rn
                                                         T
                          cSH (K) := inf{a ∈ R>0 | iaK (νK                           ) = 0}.
The invariant cSH will be called symplectic homology capacity. The next lemma summa-
rizes some properties of the capacity cSH . The properties (i), (ii), (iii) are (respectively)
called conformality, monotonicity, and spectrality.
Lemma 2.13.         (i): For any RCT set K and c ∈ R>0 , there holds cSH (cK) = c2 cSH (K).
  (ii): For any RCT sets K ′ ⊂ K, there holds cSH (K ′ ) ≤ cSH (K).
 (iii): For any C ∞ -RCT set K, there exist γ ∈ P(∂K) and m ∈ Z≥1 such that cSH (K) =
        m · A (γ). In particular cSH (K) ≥ cmin (K).

Proof. (i) follows from the commutativity of (3), and (ii) follows from the commutativity
of (2). (iii) is proved in Corollary 5.8 of [14] under the assumption that ∂K has a nice
action spectrum (see pp. 342 of [14] for its definition). Since ∂K has a nice action
spectrum for C ∞ -generic K (Proposition 2.5 of [14]), one can remove this assumption by
the limiting argument.                                                                 
                                                       11
2.5. S 1 -equivariant symplectic homology capacities. For any C ∞ -RCT set K ⊂
T ∗ Rn (in general, for any Liouville domain) and a ∈ R>0 , one can define the S 1 -equivariant
                                1
symplectic homology SH [0,a),S
                            ∗     (K) and a linear map
                           1      1                                    1
                     (iaK )S : H∗+n
                                S
                                    (T ∗ Rn , T ∗ Rn \ K) → SH [0,a),S
                                                               ∗       (K),
          1
where H∗S (T ∗ Rn , T ∗ Rn \ K) is the S 1 -equivariant homology with the trivial S 1 -action on
(T ∗ Rn , T ∗ Rn \ K), thus canonically isomorphic to H∗ (T ∗ Rn , T ∗ Rn \ K) ⊗ H∗ (CP ∞ ). For
each k ∈ Z≥1 , let
                                                 1    ∗ Rn
                    ckSH S 1 (K) := inf{a | (iaK )S (νK
                                                      T
                                                             ⊗ [CP k−1]) = 0}.

Let us call the invariants ckSH S 1 (k ≥ 1) equivariant symplectic homology capacities.
Remark 2.14. This construction goes back at least to Section 5.3 of Viterbo [23], where
the Floer-theoretic analogue of the equivariant Ekeland-Hofer capacities [6] was intro-
duced. This construction is revisited in recent papers such as Gutt-Hutchings [11] and
Ginzburg-Shon [10]. In particular, [11] introduced a sequence of capacities using positive
equivariant symplectic homology with rational coefficients, established basic properties of
these capacities, and gave combinatorial formulas to compute these capacities of convex
and concave toric domains. In [11] it is conjectured that the Gutt-Hutchings capacities
are equal to the equivariant Ekeland-Hofer capacities for any compact star-shaped domain
(Conjecture 1.9 of [11]).

  For any C ∞ -RCT set K, there holds the following inequalities:
(4)                            cmin (K) ≤ c1SH S 1 (K) ≤ cSH (K).
For the first inequality, see the “contractible Reeb orbits” property in Theorem 1.24 of
[11]. For the second inequality, see Lemma 3.2 of [10].
Remark 2.15. One has to be careful since [11] and [10] use Q -coefficients, while we
work over Z/2 -coefficients. Also, the definitions of equivariant capacities in these papers
use positive (equivariant) symplectic homology, and are superficially different from our
definition. However, it is straightforward to see that the proofs in these papers also work
in our setting.

  F. Schlenk [21] pointed out that Theorem 1.4, combined with (4), implies the following
corollary:
Corollary 2.16. cEHZ (K) = c1SH S 1 (K) = cSH (K) for any convex body K in T ∗ Rn .


               3. Symplectic homology and loop space homology

  Let pr : T ∗ Rn → Rn denote the natural projection map, namely pr(q, p) := q. For any
q ∈ Rn , we identify Tq∗ Rn with pr−1 (q).
Definition 3.1. K ⊂ T ∗ Rn is called fiberwise convex if Kq := K ∩ Tq∗ Rn is a convex set
in Tq∗ Rn for every q ∈ Rn .
                                                12
   Throughout this section, K denotes a nonempty, compact and fiberwise convex set in
T ∗ Rn . In Section 3.1, we state Theorem 3.4, which shows that symplectic homology of K
is isomorphic to a certain relative homology of loop spaces of Rn . The proof of Theorem
3.4 is carried out in Section 4. In Section 3.2, we deduce Corollary 3.8 from Theorem 3.4,
which shows that the capacity cSH (K) is equal to a certain min-max value defined from
homology of loop spaces. In Section 3.3, we prove some technical results about fiberwise
convex functions, which are used in Section 3.1 and in the proof of Theorem 3.4 (see
Section 4.6).

3.1. Symplectic homology and loop space homology. Let Λ denote the space of
L1,2 -maps from S 1 = R/Z to Rn , equipped with the L1,2 -topology. For each γ ∈ Λ, we
define lenK (γ) as follows:
                              Z
                                   ( max p · γ̇(t) ) dt (γ(S 1 ) ⊂ pr(K))
(5)               lenK (γ) :=    S p∈Kγ(t)
                                  1

                                −∞                       (γ(S 1 ) 6⊂ pr(K)).
                              

Example 3.2. If K is the unit disk cotangent bundle of pr(K), namely
                               K = {(q, p) ∈ T ∗ Rn | q ∈ pr(K), |p| ≤ 1},
                  Z
then lenK (γ) =            |γ̇(t)| dt for any γ ∈ Λ satisfying γ(S 1 ) ⊂ pr(K).
                      S1


  Let us summarize elementary properties of lenK .
Lemma 3.3. Let K be any nonempty, compact, and fiberwise convex set in T ∗ Rn .

   (i): (5) is well-defined. Namely, for any γ ∈ Λ satisfying γ(S 1 ) ⊂ pr(K), the function
        ργ : S 1 → R; t 7→ max p · γ̇(t) is integrable.
                                 p∈Kγ(t)
  (ii): lenK is upper semi-continuous. Namely, if a sequence (γk )k in Λ converges to
        γ ∈ Λ in the L1,2 -topology, then lenK (γ) ≥ lim sup lenK (γk ).
                                                                 k
  (iii): Suppose that ∂K is of C ∞ and strictly convex. Let γ : S 1 → int (pr(K)) be a
         C ∞ -map such that γ̇(t) 6= 0 for every t ∈ S 1 . Then, for every t ∈ S 1 there exists
         unique pγ (t) ∈ Kγ(t) such that pγ (t) · γ̇(t) = max p · γ̇(t). Moreover, γ̄ : S 1 → ∂K
                                                             p∈Kγ(t)
                                                       ∞
       defined by γ̄(t) := (γ(t), pγ (t)) is of C , and satisfies
                                             Z         n
                                                      ÅX       ã
                                                    ∗
                               lenK (γ) =        γ̄      pi dqi .
                                                  S1       i=1
                                           ∞
  (iv): Suppose that ∂K is of C and strictly convex. Then lenK is continuous on {γ ∈
        Λ | γ(S 1 ) ⊂ pr(K)} with respect to the L1,2 -topology.
   (v): Let K ′ be any nonempty, compact, and fiberwise convex set in T ∗ Rn which satisfies
        K ′ ⊂ K. Then lenK ′ (γ) ≤ lenK (γ) for any γ ∈ Λ.

Proof. (i) and (ii) are consequences of Lemmas 3.10 and 3.12. Let us take a sequence
(Hj )j≥1 as in Lemma 3.10, and let LHj denote the Legendre dual of Hj (see Lemma 3.12
for the definition of Legendre dual).
                                                    13
   Let us prove (i). Since K is compact, there exists C > 0 such that |p| ≤ C for every
(q, p) ∈ K. Then |ργ | ≤ C · |γ̇| for every γ ∈ Λ satisfying γ(S 1 ) ⊂ pr(K). Since |γ̇|
is integrable, it is sufficient to show that ργ is measurable. Lemma 3.12 (ii) says that
ργ (t) = lim LHj (γ(t), γ̇(t)) for every t ∈ S 1 . Then ργ is measurable, since LHj (γ, γ̇) is
            j→∞
obviously measurable for every j.
                                                                                 Z
  Let us prove (ii). For each j, let us define Lj : Λ → R by Lj (γ) :=                    LHj (γ, γ̇) dt.
                                                                                     S1
Then (Lj )j≥1 is a decreasing sequence of continuous functions on Λ, and lenK = lim Lj
                                                                                               j→∞
by Lemma 3.12. Then lenK is upper semi-continuous.
  Let us prove (iii). Since ∂K is of C ∞ and strictly convex, ∂Kq is of C ∞ and strictly
convex for any q ∈ int (pr(K)). Then, for any t ∈ S 1 , there exists unique pγ (t) ∈ Kγ(t)
which satisfies max p · γ̇(t) = pγ (t) · γ̇(t). Moreover, γ̄ = (γ, pγ ) is of C ∞ by the inverse
               p∈Kγ(t)
                                                          ÅX          ã
                                                        ∗
mapping theorem. The last assertion follows from γ̄             pi dqi = pγ (t) · γ̇(t) dt, which
                                                                i
is straightforward.
  Let us prove (iv). First we prove that
                          c : pr(K) × Rn → R;        (q, v) 7→ max p · v
                                                                p∈Kq

is continuous. Let (qk , vk )k≥1 be a sequence on pr(K) × Rn which converges to (q∞ , v∞ ) as
k → ∞. Then we want to show lim max p · vk = max p · v∞ . By the compactness of K
                                   k→∞ p∈Kqk            p∈Kq∞
one has lim sup max p · vk ≤ max p · v∞ , thus it is sufficient to show lim inf max p · vk ≥
            k→∞   p∈Kqk        p∈Kq∞                                           k→∞ p∈Kqk
max p · v∞ . Take p∞ ∈ Kq∞ so that p∞ · v∞ = max p · v∞ . We claim that there exists
p∈Kq∞                                                 p∈Kq∞
a sequence (pk )k such that (qk , pk ) ∈ K for every k and lim pk = p∞ . This claim can be
                                                                k→∞
verified as follows:

     • If q∞ ∈ int (pr(K)), then there exists ε > 0 such that the closed ε-neighborhood
       of q∞ is contained in pr(K). For each k ≥ 1 such that |qk − q∞ | < ε, let us define
       pk as follows:
          – If qk = q∞ , then pk := p∞ .
          – If qk 6= q∞ , then there exist (qk′ , p′k ) ∈ K and tk ∈ (0, 1) such that |q∞ −qk′ | = ε
            and qk = tk qk′ + (1 − tk )q∞ . Then pk := tk p′k + (1 − tk )p∞ .
       Then it is easy to see that lim pk = p∞ .
                                       k→∞
     • If q∞ ∈ ∂(pr(K)), then Kq∞ = {p∞ } since ∂K is strictly convex, thus any sequence
       (pk )k satisfying (qk , pk ) ∈ K (∀k) satisfies lim pk = p∞ .
                                                       k→∞

Now we can finish the proof of the continuity of c by lim inf max p · vk ≥ lim pk · vk =
                                                                k→∞ p∈Kqk             k→∞
p∞ · v∞ .
  Now suppose that (iv) does not hold. Then there exists a sequence (γk )k in {γ ∈
Λ | γ(S 1 ) ⊂ pr(K)} which converges to γ∞ in the L1,2 -topology, and inf |lenK (γk ) −
                                                                                     k
lenK (γ∞ )| > 0. By replacing (γk )k with its subsequence if necessary, we may assume
                                                14
lim γ̇k (t) = γ̇∞ (t) for almost every t ∈ S 1 . On the other hand lim γk (t) = γ∞ (t) for
k→∞                                                                                              k→∞
every t. Thus lim c(γk (t), γ̇k (t)) = c(γ∞ (t), γ̇∞ (t)) for almost every t, which implies
                 k→∞
lim lenK (γk ) = lenK (γ∞ ), contradicting our assumption.
k→∞

  Finally, (v) follows from pr(K ′ ) ⊂ pr(K) and max′ (p·v) ≤ max (p·v) for any q ∈ pr(K ′ )
                                                               p∈Kq                       p∈Kq
and v ∈ Rn .                                                                                           

  For any a ∈ R, let ΛaK := {γ ∈ Λ | lenK (γ) < a}. By Lemma 3.3 (ii), this is open in Λ
with the L1,2 -topology. Moreover, Lemma 3.3 (v) shows that if K ′ ⊂ K then ΛaK ⊂ ΛaK ′ .
Theorem 3.4. For any nonempty, compact and fiberwise convex set K ⊂ T ∗ Rn and real
numbers a < b, one can assign an isomorphism
                                  SH [a,b)
                                     ∗     (K) ∼
                                               = H∗ (ΛbK , ΛaK )
so that the diagram
                                    [a,b)         ∼
                                                  =
(6)                             SH ∗        (K)            /   H∗ (ΛbK , ΛaK )

                                                                          
                                  [a′ ,b′ )                            ′              ′
                               SH ∗ (K ′ )        ∼
                                                       /       H∗ (ΛbK ′ , ΛaK ′ )
                                                  =

commutes for any a ≤ a′ , b ≤ b′ and any fiberwise convex K ′ ⊂ K.
Remark 3.5. If the boundary of pr(K) ⊂ Rn is of C ∞ and K is the unit disk cotangent
bundle of pr(K), then Theorem 3.4 is essentially equivalent to Theorem 1.1 of [18].
Remark 3.6. It is likely that Theorem 3.4 naturally extends to any nonempty, compact
and fiberwise convex set K ⊂ T ∗ Q where Q is an arbitrary closed manifold. However,
since our main applications (Theorem 1.4 and Theorem 1.8) make sense only on symplectic
vector spaces, in this paper we work on symplectic vector spaces.

3.2. Symplectic homology capacity and loop space homology. In this subsection,
we prove a formula (Corollary 3.8) which computes cSH (K) in terms of homology of loop
spaces of Rn . Let us recall from Section 2.4 that for any RCT set K,
                                                                               ∗ Rn
                          cSH (K) = inf{a ∈ R>0 | iaK (νK
                                                        T
                                                                                      ) = 0}.

  For any a ∈ R>0 , let us consider a map
                               a
                              jK : (Rn , Rn \ pr(K)) → (ΛaK , Λ0K )
which sends each q ∈ Rn to the constant loop at q.
Lemma 3.7. Let K be any RCT set in T ∗ Rn which is fiberwise convex. Then, for any
                T ∗ Rn                                           Rn
a ∈ R>0 , iaK (νK      ) ∈ SH [0,a)
                              n
                                                            a
                                    (K) corresponds to H∗ (jK )(νpr(K) ) ∈ Hn (ΛaK , Λ0K ) via the
isomorphism SH [0,a)
                   ∗    (K) ∼
                            = H∗ (ΛaK , Λ0K ).

Proof. For any R ∈ R>0 let KR := {(q, p) ∈ T ∗ Rn | |q|, |p| ≤ R}.
                                                  15
   First notice that it is sufficient to prove the lemma for K = KR for every R. Indeed,
for any compact K ⊂ T ∗ Rn , there exists R such that K ⊂ KR . By the commutativity of
(6), we have a commutative diagram

                                       [0,a)                                     [0,a)
                                  SH ∗           (KR )                    /   SH ∗           (K)
                                       ∼
                                       =                                                     ∼
                                                                                             =
                                                                                    
                                H∗ (ΛaKR , Λ0KR )                 /       H∗ (ΛaK , Λ0K ).
                                                              ∗       n                          ∗   n
Then the upper horizontal map sends iaKR (νK  T R
                                                R
                                                    ) to iaK (νKT R
                                                                     ). Assuming that we have
                                                               a     T ∗ Rn         a      Rn
proved the lemma for KR , the left vertical map sends iKR (νKR ) to H∗ (jK            R
                                                                                        )(νpr(K R)
                                                                                                   ),
                      a    Rn
which is sent to H∗ (jK )(νpr(K) ) by the lower horizontal map. By the commutativity of
                                                  T ∗ Rn                  Rn
the diagram, the right vertical map sends iaK (νK                  a
                                                         ) to H∗ (jK )(νpr(K) ), which completes
the proof for K.
  Thus it is sufficient to consider the case K = KR . It is also sufficient to consider the
case when a is sufficiently small, since for any a < b we have a commutative digram

                                    [0,a)                                        [0,b)
                                SH ∗         (KR )                    /       SH ∗           (KR )
                                     ∼
                                     =                                                       ∼
                                                                                             =
                                                                                        
                               H∗ (ΛaKR , Λ0KR )              /   H∗ (ΛbKR , Λ0KR ).

                                             a      R                 n
   Moreover, it is sufficient to prove H∗ (jK  R
                                                 )(νpr(K R)
                                                            ) 6= 0 for sufficiently small a. Indeed,
when a is sufficiently small, Remark 2.11 implies that SH [0,a)     n    (KR ) ∼
                                                                               = Z/2 is generated by
 a    T ∗ Rn                              [0,a)       ∼
iKR (νKR ). Then the isomorphism SH ∗ (KR ) = Hn (ΛKR , ΛKR ) maps iaKR (νK
                                                                  a      0               T ∗ Rn
                                                                                                ) to the
                                                                                           R
                                                                       n
only nonzero element in Hn (ΛaKR , Λ0KR ), that is H∗ (jK    a
                                                               R
                                                                 )(ν R
                                                                    pr(KR ) ).
  The rest of the proof is essentially the same as the proof of Lemma 6.6 (2)
                                                                          Z of [18], which
we repeat here for the sake of completeness. For any γ ∈ Λ let len(γ) :=                                      |γ̇(t)| dt, and
                                                                                                         S1
for any a ∈ R>0 let U a := {γ ∈ Λ | len(γ) < a/R}. Also let BR := {q ∈ Rn | |q| ≤ R}
and VR := {γ ∈ Λ | γ(S 1 ) 6⊂ BR }. Then

                                  ΛaKR = U a ∪ VR ,                       Λ0KR = VR .

Since both U a and VR are open sets in Λ, the inclusion map

                         (U a , U a ∩ VR ) → (U a ∪ VR , VR ) = (ΛaKR , Λ0KR )

induces an isomorphism on homology. Thus it is sufficient to show that

                               caR : (Rn , Rn \ BR ) → (U a , U a ∩ VR )

which sends each q ∈ Rn to the constant loop at q, induces an injection on homology if a
is sufficiently small.
                                                         16
   Let us define ev : Λ → Rn by ev (γ) := γ(0). If a is sufficiently small, then ev maps
U a ∩ VR to Rn \ {0}, and we obtain a commutative diagram
                                                  ca
                                                   R
                                (Rn , Rn \ BR )         / (U a , U a ∩ VR )
                                            ❘❘❘
                                               ❘❘❘
                                                    ❘❘❘
                                                       ❘❘❘           ev
                                                id Rn      ❘(      
                                                       (Rn , Rn \ {0}).
The diagonal map induces an isomorphism on homology, thus H∗ (caR ) is injective. This
completes the proof.                                                                

  As an immediate corollary of Lemma 3.7, we obtain the following formula which com-
putes cSH (K) from homology of loop spaces.
Corollary 3.8. For any RCT set K ⊂ T ∗ Rn which is fiberwise convex,
                                                     a    R          n
                        cSH (K) = inf{a ∈ R>0 | H∗ (jK )(νpr(K) ) = 0}.

3.3. Technical results on fiberwise convex functions. In this subsection we prove
some preliminary results on (fiberwise) convex functions.
Definition 3.9. For any (finite-dimensional) real vector space V , f ∈ C 0 (V, R) is called
convex if f (tx+(1−t)y) ≤ tf (x)+(1−t)f (y) for any x, y ∈ V and t ∈ [0, 1]. f ∈ C 2 (V, R)
is called strictly convex, if for any x ∈ V , the Hessian of f at x (which is a symmetric
bilinear form on V ) is positive definite. f ∈ C 0 (T ∗ Rn ) is called fiberwise convex if f |Tq∗ Rn
is convex for every q ∈ Rn , and f ∈ C 2 (T ∗ Rn ) is called fiberwise strictly convex if f |Tq∗ Rn
is strictly convex for every q ∈ Rn .

  For any a ∈ R>0 , let us define Qa ∈ C ∞ (T ∗ Rn ) by Qa (q, p) := a(|q|2 + |p|2 ).
Lemma 3.10. For any nonempty, compact, and fiberwise convex set K ⊂ T ∗ Rn , there
exist sequences (aj )j≥1 and (Hj )j≥1 which satisfy the following properties:

    (i): (aj )j is a strictly increasing sequence in R>0 \ πZ.
   (ii): lim aj = ∞.
        j→∞
  (iii): (Hj )j is a strictly increasing sequence of fiberwise strictly convex C ∞ -functions on
         T ∗ Rn .
  (iv): For every j, there exists bj ∈ R such that Hj is a compact perturbation of Qaj + bj ,
         i.e. Hj − (Qaj + bj ) is compactly supported.
                           ®
                             ∞ ((q, p) ∈  / K)
   (v): lim Hj (q, p) =
         j→∞                 0    ((q, p) ∈ K).

Proof. Let us take a sequence (Uj )j of open sets in T ∗ Rn such that Uj+1 ⊂ Uj for every
       ∞
       \
j, and   Uj = K.
       j=1

  Let us consider conditions (ii’) and (v’) as follows:

  (ii’): aj > 2j for every j.
                                                  17
  (v’): The following properties hold for every j:
          • Hj (q, p) > 2j if (q, p) ∈
                                     / Uj ,
               1                     1
          • − j < Hj (q, p) < − j+1 if (q, p) ∈ K.
              2                    2

Obviously (ii’) implies (ii), and (v’) implies (v). Thus it is sufficient to construct sequences
(aj )j and (Hj )j satisfying (i), (ii’), (iii), (iv), (v’). We are going to construct such sequences
by induction on j. Suppose that we have defined a1 , . . . , aj−1 and H1 , . . . , Hj−1 satisfying
these conditions. In the following argument we construct a pair (aj , Hj ) so that these
conditions are satisfied. Let us take a ∈ R>0 \ πZ such that a > max{aj−1 , 2j }. We fix
such a in the rest of the proof.
  Step 1. For any b ∈ R≥0 , we define Fb : T ∗ Rn → R in the following way.
  For each q ∈ Rn , let F (b, q) denote the set of convex functions f : Tq∗ Rn → R satisfying
the following conditions:

     • f (p) ≤ Qa (q, p) + b for every p ∈ Tq∗ Rn .
                   3
     • f (p) ≤ − j+2 if (q, p) ∈ K.
                2

Let us define Fb by Fb (q, p) :=     sup f (p). Then, Fb |Tq∗ Rn is convex (thus continuous) for
                                   f ∈F (b,q)
every q ∈ Rn . The function Fb satisfies the following properties:

 (1-0): If q ∈
             / pr(K) then Fb (q, p) = Qa (q, p) + b.
 (1-1): Fb is a compact perturbation of Qa + b.
                       3
 (1-2): Fb (q, p) ≥ − j+2 for every (q, p) ∈ T ∗ Rn .
                     2
                       3
 (1-3): Fb (q, p) = − j+2 if (q, p) ∈ K.
                     2
 (1-4): For any ε > 0, there exists δ > 0 such that if p ∈ Tq∗ Rn satisfies dist(Kq , p) < δ,
                                                                                   3
        where dist denotes the Euclidean distance on Tq∗ Rn , then Fb (q, p) < − j+2 + ε.
                                                                                 2

(1-0) holds since Qa (q, p)+b ∈ F (b, q) if q ∈
                                              / pr(K). (1-2) and (1-3) hold since the constant
              3
function − j+2 is an element of F (b, q). (1-1) holds since if |q|2 + |p|2 is sufficiently large,
            2
the linear function

                        Tq∗ Rn → R; x 7→ Qa (q, p) + b + 2a(p · (x − p))

is an element of F (b, q). (1-4) follows from (1-3), (1-1) and the convexity of Fb |Tq∗ Rn .
  Moreover, when b is sufficiently large, the following properties hold:

 (1-5): Fb (q, p) > Hj−1(q, p) for any (q, p) ∈ T ∗ Rn .
 (1-6): Fb (q, p) > 2j if (q, p) ∈
                                 / Uj .
                                                 18
  Let us check that (1-5) holds for sufficiently large b. By the induction assumption,
          1                         1         3
Hj−1 < − j on K. Thus Hj−1 + j+2 < − j+2 on K. Since Hj−1 is a compact pertur-
          2                       2         2
                                                                           1
bation of Qaj−1 + bj−1 and aj−1 < a, when b is sufficiently large Hj−1 + j+2 ∈ F (b, q).
                                                                         2
                                1                               ∗ n
This means that Hj−1(q, p) + j+2 ≤ Fb (q, p) for any (q, p) ∈ T R , thus (1-5) holds.
                              2
  Let us check that (1-6) holds for sufficiently large b. For any (q, p) ∈    / Uj such that
Kq 6= ∅, let p′ be the unique point on Kq such that |p − p′ | = dist(Kq , p).

Remark 3.11. The uniqueness of p′ follows from the convexity of Kq . Indeed, suppose
that there exist p′ 6= p′′ in Kq satisfying |p − p′ | = |p − p′′ | = dist(Kq , p). Then p′′′ :=
(p′ + p′′ )/2 ∈ Kq by the convexity of Kq . On the other hand |p − p′′′ | < dist(Kq , p) by
p′ 6= p′′ , which contradicts p′′′ ∈ Kq .

  Let us define a linear function Hq,p on Tq∗ Rn by

                                     3                                 2j + 1 + 3/2j+2
                  Hq,p (x) := −            + (x − p′ ) · (p − p′ ) ·                   .
                                  2j+2                                     |p − p′ |2
                                               3
Then Hq,p (p) = 2j + 1 and Hq,p ≤ −                 on Kq . Also, there holds
                                             2j+2
                       S := sup ( max
                                   ∗ n
                                       Hq,p (x) − Qa (q, x)) < ∞.
                                   / j x∈Tq R
                             (q,p)∈U
                               Kq 6=∅

This can be checked as follows: for any (q, p) ∈
                                               / Uj with Kq 6= ∅,
                                                                       |∇Hq,p|2
                   max  Hq,p (x) − Qa (q, x) = Hq,p (0) +                       − a|q|2 .
                    ∗ n
                  x∈Tq R                                                 4a

Setting γ := 2j + 1 + 3/2j+2,
                                 3                                γ           3    Rγ
                 Hq,p (0) = −            − p′ · (p − p′ ) ·         ′
                                                                         ≤ − j+2 +    ,
                                2j+2                          |p − p | 2    2       δ
where R and δ are positive constants (depending only on K and Uj ) such that |p′ | ≤ R
and |p − p′ | ≥ δ. Also, there holds |∇Hq,p | = γ/|p − p′ | ≤ γ/δ. Then we can conclude
that S < ∞.
  We show that if b > max{2j , S} then (1-6) holds, i.e. Fb (q, p) > 2j if (q, p) 6∈ Uj . We
consider two cases:
                                                                                             3
     • The case Kq 6= ∅. In this case Hq,p ∈ F (b, q), because Hq,p ≤ −                            on Kq
                                                                                            2j+2
       and Hq,p (x) ≤ Qa (q, x) + S < Qa (q, x) + b for any x ∈                  Tq∗ Rn .
                                                                           Hence Fb (q, p) ≥
                   j       j
       Hq,p (p) = 2 + 1 > 2 .
     • The case Kq = ∅. In this case, Fb (q, p) = Qa (q, p) + b ≥ b > 2j .

In the rest of the proof we take and fix b so that (1-5) and (1-6) hold.
                                                      19
                                                                                 Z
  Step 2. Let us take ρ ∈         Cc∞ (Rn , R≥0 )   such that ρ(x) = ρ(−x) and            ρ(x) dx = 1.
                                                                                 Rn
For any ε > 0 let ρε (x) := ε−n ρ(x/ε). Then we define Gε : T ∗ Rn → R by
                                      Z
                            ε
                          G (q, p) :=        Fb (q, y)ρε (p − y) dy.
                                            y∈Tq∗ Rn

Then Gε satisfies the following properties:

     • For every q ∈ Rn , Gε |Tq∗ Rn is a C ∞ convex function.
                                                                                 Z
                              ε
     • If q ∈
            / pr(K) then G (q, p) = Qa (q, p) + b + a · c(ε), where c(ε) :=               |x|2 ρε (x) dx.
                                                                                     Rn
     • Gε is a compact perturbation of Qa + b + a · c(ε).
                                                                                              1
     • Gε (q, p) ≥ Fb (q, p) for any (q, p) ∈ T ∗ Rn . In particular, Gε (q, p) > max{−          , Hj−1(q, p)}
                                                                                              2j
        for any (q, p) ∈ T ∗ Rn , and Gε (q, p) > 2j for any (q, p) ∈
                                                                    / Uj .

Moreover, by (1-4), if ε is sufficiently small then
                                                                      1
                              (q, p) ∈ K =⇒ Gε (q, p) < −                  .
                                                                    2j+1
In the rest of the proof we fix such ε.
  Step 3. For each q ∈ Rn , let us define Hq : T ∗ Rn → R by
                             Hq (q ′ , p) := Gε (q, p) + a(|q ′ |2 − |q|2 ).
Then Hq |Tq∗′ Rn is a C ∞ -convex function for every q ′ ∈ Rn . Moreover, if q ∈
                                                                               / pr(K) then
Hq = Qa + b + a · c(ε).
  For every q ∈ Rn , there exists an open neighborhood of q (denoted by Uq ) such that
the following properties hold for every q ′ ∈ Uq :
                                  1
     • Hq (q ′ , p) > max{− j , Hj−1 (q ′ , p)} for every p ∈ Tq∗′ Rn .
                                 2
                          1
     • Hq (q , p) < − j+1 if (q ′ , p) ∈ K.
             ′
                        2
     • Hq (q ′ , p) > 2j if (q ′ , p) ∈
                                      / Uj .

Moreover, if q ∈
               / pr(K) then we may take Uq so that Uq ∩ pr(K) = ∅.
   Let us consider an open covering of Rn , U := {Uq }q∈Rn . Let V = {Vi }∞          i=1 be a
refinement of U which is locally finite. For every i, choose qi ∈ Rn such that Vi ⊂ Uqi .
Let (χi )i be a partition of 1 with V , i.e. χi ∈ C ∞ (Rn , [0, 1]) and supp χi ⊂ Vi for every
            X∞
i ≥ 1, and      χi ≡ 1. Then
            i=1
                                                ∞
                                                X
                                   H(q, p) :=          χi (q)Hqi (q, p)
                                                 i=1

is a C ∞ -function on T ∗ Rn , and satisfies the following properties:

     • H is a compact perturbation of Qa + b + a · c(ε).
                                                    20
     • H is fiberwise convex.
     • H(q, p) > Hj−1(q, p) for every (q, p) ∈ T ∗ Rn .
         1                    1
     • − j < H(q, p) < − j+1 if (q, p) ∈ K.
        2                   2
     • H(q, p) > 2j if (q, p) ∈
                              / Uj .

  The first property holds since Hqi 6= Qa + b + a · c(ε) only if qi ∈ pr(K), and there are
only finitely many such qi ’s. The other properties are straightforward.
  Step 4. Let us take a sufficiently small δ > 0 such that a+δ ∈
                                                               / πZ. Then Hj := H +Qδ
satisfies the following properties:

     • Hj is a compact perturbation of Qa+δ + b + a · c(ε).
     • Hj is fiberwise strictly convex.
     • Hj (q, p) > Hj−1 (q, p) for every (q, p) ∈ T ∗ Rn .
         1                      1
     • − j < Hj (q, p) < − j+1 for every (q, p) ∈ K.
        2                     2
     • Hj (q, p) > 2j if (q, p) ∈
                                / Uj .

The fourth property can be achieved by taking δ sufficiently small. The other properties
are straightforward.
  Finally, setting aj := a + δ, the pair (aj , Hj ) satisfies conditions (i), (ii’), (iii), (iv),
(v’).                                                                                          
Lemma 3.12. Let K be a compact and fiberwise convex set in T ∗ Rn , and let (Hj )j≥1
and (aj )j≥1 be sequences which satisfy the conditions in Lemma 3.10. For each j, let
LHj ∈ C ∞ (T Rn ) denote the Legendre dual of Hj , namely
                  LHj (q, v) := max
                                 ∗ n
                                     (p · v − Hj (q, p))    (q ∈ Rn , v ∈ Tq Rn ).
                               p∈Tq R

Then the following properties hold:

    (i): LHj (q, v) > LHj+1 (q, v) for any (q, v) ∈ T Rn and j ≥ 1.
                           (
                              max(p · v) (q ∈ pr(K))
   (ii): lim LHj (q, v) = p∈Kq
         j→∞
                              −∞            (q ∈
                                               / pr(K)).
              Z
  (iii): lim      LHj (γ(t), γ̇(t)) dt = lenK (γ) for any γ ∈ Λ.
        j→∞   S1


Proof. (i): For each q ∈ Rn , there exists p0 ∈ Tq∗ Rn which satisfies LHj+1 (q, v) = p0 · v −
Hj+1 (q, p0). Then
LHj (q, v) = max(p · v − Hj (q, p)) ≥ p0 · v − Hj (q, p0) > p0 · v − Hj+1 (q, p0 ) = LHj+1 (q, v).
               p


  (ii) follows from Lemma 3.13 applied to (Hj |Tq∗ Rn )j , identifying Rn and Tq∗ Rn via the
standard Riemannian metric on Rn .
  (iii): First, we consider the case γ(S 1 ) ⊂ pr(K). By Lemma 3.3 (i), ργ : S 1 →
R; t 7→ max p · γ̇(t) is integrable. On the other hand, LH1 (γ, γ̇) is integrable (since γ̇
        p∈Kγ(t)

                                                 21
is square-integrable), and (LHj (γ, γ̇))j is a decreasing sequence of integrable functions,
which converges to ργ pointwise as j → ∞. Then, by Lebesgue’s dominated convergence
theorem, we obtain
          Z                          Z                            Z
      lim     LHj (γ(t), γ̇(t)) dt =   lim LHj (γ(t), γ̇(t)) dt =   ργ (t) dt = lenK (γ).
     j→∞   S1                            S 1 j→∞                                   S1


  Next, we consider the case γ(S 1 ) 6⊂ pr(K). In this case I := γ −1 (Rn \ pr(K)) is a
nonempty open set in S 1 . Now consider an obvious inequality
         Z                         Z                         Z
            LHj (γ(t), γ̇(t)) dt ≤     LH1 (γ(t), γ̇(t)) dt + LHj (γ(t), γ̇(t)) dt.
           S1                            S 1 \I                              I

The first term on the RHS does not depend on j, and the second term goes to −∞ as
j → ∞. Thus the LHS goes to −∞.                                                
Lemma 3.13. Let K be any compact and convex set in Rn , which may be empty. Let
(aj )j≥1 and (hj )j≥1 be sequences with the following properties:

   (i): (aj )j is a strictly increasing sequence in R>0 .
  (ii): lim aj = ∞.
       j→∞
  (iii): (hj )j is a strictly increasing sequence of convex C ∞ -functions on Rn .
                                                                  2
                        ® exists bj ∈ R such that hj (x)−aj |x| −bj is compactly supported.
  (iv): For every j, there
                          ∞ (x ∈   / K)
   (v): lim hj (x) =
         j→∞              0     (x ∈ K).

Then, for any x ∈ Rn
                                                             (
                                                                max(x · y) (K 6= ∅)
                   lim maxn (x · y − hj (y)) =                   y∈K
                   j→∞    y∈R                                    −∞               (K = ∅).

Proof. First we consider the case K 6= ∅. Let H denote the set of h ∈ C ∞ (Rn ) with the
following properties:

  (a): h is convex.
  (b): There exists Q ∈ C ∞ (Rn ) of the form
                                                             X
                                Q(x1 , . . . , xn ) =             aij xi xj + b
                                                        1≤i,j≤n

        where (aij )1≤i,j≤n is a non-negative symmetric matrix, such that h(x) − Q(x) is
        compactly supported.
   (c): h(x) < 0 for any x ∈ K.

Then the sequence (hj )j is cofinal in H , which implies that for any x ∈ Rn
                                            
                 lim maxn (x · y − hj (y)) = inf (maxn (x · y − h(y))).
                   j→∞    y∈R                                h∈H y∈R

For any h ∈ H , there holds
                    maxn (y · x − h(y)) ≥ max(x · y − h(y)) ≥ max x · y.
                    y∈R                           y∈K                         y∈K

                                                        22
Thus inf (maxn (x · y − h(y))) ≥ max x · y. To complete the proof, it is sufficient to prove
      h∈H y∈R                           y∈K
the opposite inequality, i.e.        inf (max(x · y − h(y))) ≤ max x · y. To prove this, it is
                                    h∈H y∈Rn                         y∈K
sufficient to show that for any δ > 0 there exists h ∈ H such that
                                  maxn (x · y − h(y)) ≤ max x · y + δ.
                                  y∈R                    y∈K

When x = 0 it is easy to see. When x 6= 0, let Ix := {x · y | y ∈ K} and Mx := max Ix =
max x · y. It is easy to see that there exists ϕ ∈ C ∞ (R) with the following properties:
y∈K

      •   ϕ is convex.
      •   There exist a > 0 and b ∈ R such that ϕ(t) − (at2 + b) is compactly supported.
      •   −δ ≤ ϕ(t) < 0 for any t ∈ Ix .
      •   ϕ′ (Mx ) = 1.

Take such ϕ and let h(y) := ϕ(x · y). Then h ∈ H , and there holds
                maxn (x · y − h(y)) = max(t − ϕ(t)) = Mx − ϕ(Mx ) ≤ Mx + δ.
                y∈R                       t∈R

This completes the proof when K 6= ∅.
  Finally we consider the case K = ∅. Let H ′ denote the set of h ∈ C ∞ (Rn ) which
satisfies conditions (a) and (b) above. Then, the sequence (hj )j is cofinal in H ′ , which
implies that for any x ∈ Rn
                                             
                    lim maxn (x · y − hj (y)) = inf ′ (maxn (x · y − h(y))).
                      j→∞   y∈R                         h∈H    y∈R
           ′                  ′
If h ∈ H then h + c ∈ H for any c ∈ R, thus the RHS is obviously equal to −∞. This
completes the proof.                                                            


                                    4. Proof of Theorem 3.4

  The goal of this section is to prove Theorem 3.4. In Section 4.1, we summarize basic
properties of Lagrangian action functionals on the free loop space of Rn . In Section 4.2
we state Theorem 4.5, which shows an isomorphism between Hamiltonian Floer homology
on T ∗ Rn and homology of loop spaces of Rn . The proof of Theorem 4.5 occupies Sections
4.3–4.5; the plan of the proof of Theorem 4.5 is explained in the last paragraph of Section
4.2. Finally, in Section 4.6, we prove Theorem 3.4 by taking a limit of isomorphisms
obtained by Theorem 4.5.

4.1. Lagrangian action functional on the loop space. Consider the following con-
ditions (L1), (L2) for L ∈ C ∞ (S 1 × T Rn ):

 (L1): There exist a ∈ R>0 and b ∈ R such that the function on S 1 × T Rn
                                         Å 2
                                          |v|
                                                        ã
                                                    2
                            L(t, q, v) −      − a|q| + b
                                           4a
       is compactly supported.
 (L2): There exists c ∈ R>0 such that ∂v2 L(t, q, v) ≥ c for any (t, q, v) ∈ S 1 × T Rn .
                                                   23
Remark 4.1. ∂v2 L(t, q, v) ≥ c means that the symmetric matrix (∂vi ∂vj L(t, q, v)−cδij )1≤i,j≤n
                            ®
                              1 (i = j)
is nonnegative, where δij =
                              0 (i 6= j).

  Recall Λ := L1,2 (S 1 , Rn ). If L satisfies the condition (L1), then one can define the
functional SL : Λ → R by
                                         Z
                               SL (γ) :=      L(t, γ(t), γ̇(t)) dt.
                                               S1

Lemma 4.2. If L ∈ C ∞ (S 1 × T Rn ) satisfies (L1) and (L2), the functional SL satisfies
the following properties:

   (i): SL is a Fréchet C 1 -function. The differential dSL is given by
                 Z
      dSL (ξ) :=                                                          ˙ dt
                     ∂q L(t, γ(t), γ̇(t)) · ξ(t) + ∂v L(t, γ(t), γ̇(t)) · ξ(t)     (∀ξ ∈ Λ).
                   S1

        Moreover dSL is Gâteaux differentiable.
  (ii): γ ∈ Λ satisfies dSL (γ) = 0 if and only if γ ∈ C ∞ (S 1 , Rn ) and satisfies

                          ∂q L(t, γ(t), γ̇(t)) − ∂t (∂v L(t, γ(t), γ̇(t))) = 0.

  (iii): For every γ ∈ Λ, let us define DSL (γ) ∈ Λ so that

                           hDSL (γ), ξiL1,2 = dSL (γ)(ξ)        (∀ξ ∈ Λ),
                                                     Z
       where h , iL1,2   is defined by hf, giL1,2 :=   f (t) · g(t) + f˙(t) · ġ(t) dt. Then the pair
                                                         S1
       (SL , DSL ) satisfies the Palais-Smale condition. Namely, if a sequence (xk )k on
       Λ satisfies sup |SL (xk )| < ∞ and lim dSL (DSL (xk )) = 0 then (xk )k contains a
                     k                           k→∞
       convergent subsequence.


Proof. (i) and (ii) follow from Proposition 3.1 (i), (ii) of [4]. (iii) is proved as Corollary
3.4 of [18], which is based on Proposition 3.3 of [4].                                      


   Suppose that L ∈ C ∞ (S 1 × T Rn ) satisfies (L1) and (L2). Let P(L) denote the set of
critical points of SL , namely

                                 P(L) := {γ ∈ Λ | dSL (γ) = 0}.

For any γ ∈ P(L), the second Gâteaux differential d2 SL (γ) is Fredholm and has finite
Morse index (see Proposition 3.1 (iii) of [4]). The Morse index is denoted by ind Morse (γ).
We say that γ is nondegenerate if 0 is not an eigenvalue of d2 SL (γ). Let us introduce the
following condition for L ∈ C ∞ (S 1 × T Rn ):

 (L0): Every γ ∈ P(L) is nondegenerate.
                                                    24
4.2. Isomorphism between Hamiltonian Floer homology and loop space homol-
ogy. Let us consider the following condition for H ∈ C ∞ (S 1 × T ∗ Rn ):

 (H2): There exists c ∈ R>0 such that ∂p2 H(t, q, p) ≥ c for any (t, q, p) ∈ S 1 × T ∗ Rn .

For any H ∈ C ∞ (S 1 × T ∗ Rn ) which satisfies (H1) and (H2), its Legendre dual LH ∈
C ∞ (S 1 × T Rn ) is defined by
          LH (t, q, v) := max
                           ∗ n
                               (p · v − H(t, q, p))                        (t ∈ S 1 , q ∈ Rn , v ∈ Tq Rn ).
                         p∈Tq R

Lemma 4.3.    (i): If H satisfies (H1) and (H2), then LH satisfies (L1) and (L2).
    Moreover, the map
                             P(H) → P(LH ); x 7→ γx := pr ◦ x
       is a bijection, and the inverse map is
                                            P(LH ) → P(H); γ 7→ (γ, pγ )
       where pγ is characterized by
              LH (t, γ(t), γ̇(t)) = pγ (t) · γ̇(t) − H(t, γ(t), γ̇(t))                                      (∀t ∈ S 1 ).
  (ii): In the situation of (i), for any x ∈ P(H), γx is nondegenerate if and only if x is
        nondegenerate. Moreover, for any such x, there holds ind Morse(γx ) = ind CZ (x).

Proof. (i) can be checked by direct computations. (ii) follows from Theorem 1 of [19]
Section 7.3.                                                                       
Remark 4.4. Lemma 4.3 (ii) extends to Hamiltonians on arbitrary manifolds, at least
when H is a “classical” Hamiltonian (i.e. the sum of the kinetic energy and a potential
function on the base) on a Riemannian manifold M, although one needs a correction term
if the vector bundle γx∗ T M is not oriented. See Theorem 1.2 and Lemma 2.1 of [24].

  Now let us state the isomorphism between Hamiltonian Floer homology on T ∗ Rn and
homology of loop spaces of Rn :
Theorem 4.5. For any H ∈ C ∞ (S 1 × T ∗ Rn ) which satisfies (H0), (H1), (H2), and any
real numbers a < b, one can define an isomorphism
                        HF [a,b) (H) ∼
                                     = H∗ (S −1 (R<b ), S −1 (R<a ))
                                        ∗                LH                           LH
so that the following diagram commutes:
                            [a,b)                                                          [a′ ,b′ )
(7)                     HF ∗                (H)                               /       HF ∗                 (H)
                            ∼
                            =                                                                          ∼
                                                                                                       =
                                                                                                  
               H∗ (SL−1
                      H
                        (R<b ), SL−1
                                   H
                                     (R<a ))                  /       H∗ (SL−1
                                                                             H
                                                                               (R<b′ ), SL−1
                                                                                           H
                                                                                             (R<a′ ))
where a ≤ a′ and b ≤ b′ ,
                            [a,b)                                                          [a,b)
(8)                     HF ∗                (H)                                   /   HF ∗             (H ′ )
                            ∼
                            =                                                                          ∼
                                                                                                       =
                                                                                              
               H∗ (SL−1
                      H
                        (R<b ), SL−1
                                   H
                                     (R<a ))                      /   H∗ (SL−1
                                                                             H′
                                                                                (R<b ), SL−1
                                                                                           H′
                                                                                              (R<a ))
                                                         25
where H(t, q, p) < H ′ (t, q, p) (∀(t, q, p) ∈ S 1 × T ∗ Rn ).
Remark 4.6. Commutative diagrams (7) and (8) are special cases of the following com-
mutative diagram:
                             [a,b)                                         [a′ ,b′ )
(9)                      HF ∗        (H)                            /   HF ∗           (H ′)
                             ∼
                             =                                                      ∼
                                                                                    =
                                                                               
                H∗ (SL−1
                       H
                         (R<b ), SL−1
                                    H
                                      (R<a ))           /   H∗ (SL−1
                                                                   H′
                                                                      (R<b′ ), SL−1
                                                                                  H′
                                                                                     (R<a′ )),

where a ≤ a′ , b ≤ b′ and H(t, q, p) < H ′ (t, q, p) (∀(t, q, p) ∈ S 1 × T ∗ Rn ).

  The proof of Theorem 4.5, which follows the arguments in [3] and [18], occupies Sections
4.3–4.5. In Section 4.3 we recall the construction of Morse complex of Lagrangian action
functionals. In Section 4.4 we explain a chain-level construction of the isomorphism in
Theorem 4.5 and check the commutativity of the diagram (7). In Section 4.5 we prove
the commutativity of the diagram (8).

4.3. Morse theory for Lagrangian action functionals. Suppose that L ∈ C ∞ (S 1 ×
T Rn ) satisfies (L0), (L1) and (L2). The goal of this subsection is to recall the construction
of the Morse complex of SL .
  For each k ∈ Z≥0 , let CM k (L) denote the free Z/2-module generated over
                                     {γ ∈ P(L) | ind Morse(γ) = k}.
To define the boundary operator we need the following lemma. For definitions of “Morse
vector field” and “Morse-Smale condition”, see Section 2 of [4]. In the next lemma,
Λ = L1,2 (S 1 , Rn ) is equipped with a natural structure of a Hilbert manifold.
Lemma 4.7. If L ∈ C ∞ (S 1 × T Rn ) satisfies (L0), (L1), (L2), there exists a smooth
vector field X on Λ which satisfies the following conditions:

    (i): X is complete.
   (ii): SL is a Lyapunov function for X. Namely, dSL (X(γ)) < 0 if X(γ) 6= 0.
  (iii): X is a Morse vector field. X(γ) = 0 if and only if γ ∈ P(L), and the Morse
         index of X at γ is equal to the Morse index of γ as a critical point of SL .
  (iv): The pair (SL , X) satisfies the Palais-Smale condition.
   (v): X satisfies the Morse-Smale condition up to every order.

Proof. This lemma follows from Lemma 3.5 of [18] (which is essentially same as Theorem
4.1 of [4]), since the condition (L1) of [18] is weaker than the condition (L1) of this
paper.                                                                               

  Let us take a vector field X on Λ which satisfies the conditions in Lemma 4.7. Let
(ϕtX )t∈R denote the flow on Λ generated by X. For any γ ∈ P(L) let us set
                            W u (γ : X) := {x ∈ Λ | lim ϕtX (x) = γ}
                                                             t→−∞

                            W (γ : X) := {x ∈ Λ | lim ϕtX (x) = γ}.
                                 s
                                                             t→∞

                                                   26
  For any real numbers a < b, let CM [a,b)
                                     ∗     (L) denote the free Z/2-module generated over
{γ ∈ P(L) | a ≤ SL (γ) < b}. For any two generators γ and γ ′ , let
                               MX (γ, γ ′ ) := W u (γ : X) ∩ W s (γ ′ : X).
When γ 6= γ ′ , let M¯X (γ, γ ′ ) denote the quotient of MX (γ, γ ′ ) by the natural R-action.
Since X satisfies the Morse-Smale condition, the boundary operator
                                                                        #2 M¯X (γ, γ ′ ) · γ ′
                                   [a,b)
                                                         X
      ∂L,X : CM [a,b)
                  ∗   (L) → CM ∗−1 (L); γ 7→
                                                       ind Morse (γ ′ )=ind Morse (γ)−1

                               2
is well-defined and satisfies ∂L,X = 0. Homology of the chain complex (CM ∗[a,b) (L), ∂L,X )
does not depend on the choice of X, and denoted by HM ∗[a,b) (L). There exists a natural
isomorphism HM [a,b)
                  ∗   (L) ∼
                          = H∗ (SL−1 (R<b ), SL−1 (R<a )). These facts follow from Theorems
2.7, 2.8 and 2.11 in [2].
  Consider L0 , L1 ∈ C ∞ (S 1 × T Rn ) which satisfy (L0), (L1), (L2) and L0 (t, q, v) >
L1 (t, q, v) for any (t, q, v) ∈ S 1 × T Rn . We also assume that P(L0 ) ∩ P(L1 ) = ∅.
  Take vector fields X 0 , X 1 on Λ such that (L0 , X 0 ) and (L1 , X 1 ) satisfy the conditions in
Lemma 4.7. By taking small perturbations of X 0 and X 1 (note that these perturbations
do not change Morse complexes of L0 and L1 ), we can achieve the following condition:

         For any γ 0 ∈ P(L0 ) and γ 1 ∈ P(L1 ), W u (γ 0 : X 0 ) is transverse to
         W s (γ 1 : X 1 ).

If this assumption is satisfied, MX 0 ,X 1 (γ 0 , γ 1 ) := W u (γ 0 : X 0 ) ∩ W s (γ 1 : X 1 ) is a smooth
manifold of dimension ind Morse (γ 0 ) − ind Morse (γ 1 ). Then we define a chain map
                                                                   X
  Φ : CM [a,b)
          ∗    (L0
                   , X 0
                         ) → CM [a,b)
                                ∗     (L1
                                          , X 1
                                                ); γ   →
                                                       7                          ♯2 MX 0 ,X 1 (γ, γ ′ ) · γ ′ .
                                                             ind Morse (γ ′ )=ind Morse (γ)

Φ induces a linear map on homology HM [a,b)∗  (L0 ) → HM ∗[a,b) (L1 ), which does not depend
on the choices of X 0 , X 1 . Via isomorphisms between the Morse homology and the loop
space homology, this map corresponds to the map
                    H∗ (SL−1          −1                −1          −1
                           0 (R<b ), SL0 (R<a )) → H∗ (SL1 (R<b ), SL1 (R<a ))


which is induced by the inclusion map.

4.4. Isomorphism at chain level. Let us take H ∈ C ∞ (S 1 × T ∗ Rn ) satisfying (H0),
(H1), (H2). Its Legendre dual LH satisfies (L0), (L1), (L2) by Lemma 4.3. Let us
also take real numbers a < b. The goal of this subsection is to define a chain map
CM [a,b)
    ∗    (LH ) → CF [a,b)
                    ∗     (H) which induces an isomorphism HM ∗[a,b) (LH ) ∼
                                                                           = HF [a,b) (H).
  The definition of the chain map involves “hybrid moduli spaces” introduced by Abbondandolo-
Schwarz [3]. Let us take X and J as follows:

      • X is a vector field on Λ such that CM [a,b)
                                                  ∗   (LH , X) is well-defined.
      • J = (Jt )t∈S 1 is a family of almost complex structures on T ∗ Rn such that CF ∗[a,b) (H, J)
        is well-defined.
                                                      27
 For any γ ∈ P(LH ) with SLH (γ) ∈ [a, b) and x ∈ P(H) with AH (x) ∈ [a, b), let
MX,H,J (γ, x) denote the set of u ∈ L1,3 (R≥0 × S 1 , T ∗ Rn ) such that
                                   ∂s u − Jt (∂t u − XHt (u)) = 0,
                                   pr ◦ u0 ∈ W u (γ : X),
                                    lim us = x.
                                   s→∞

Here us : S 1 → T ∗ Rn is defined by us (t) := u(s, t).
Remark 4.8. The above Sobolev space L1,3 can be replaced with L1,r for any 2 < r ≤ 4;
see pp.299 of [3].
Lemma 4.9. Let γ and x be as above.

   (i): For any u ∈ MX,H,J (γ, x), there holds
                           SLH (γ) ≥ SLH (pr ◦ u0 ) ≥ AH (u0) ≥ AH (x).
         In particular, if MX,H,J (γ, x) 6= ∅ then SLH (γ) ≥ AH (x).
   (ii): If SLH (γ) = AH (x), then MX,H,J (γ, x) 6= ∅ if and only if x = pr ◦ γ. Moreover,
         the moduli space MX,H,J (γ, pr◦γ) consists of a point which is cut out transversally.

Proof. See pp.299 of [3] for (i) and the first sentence in (ii). For the second sentence in
(ii), see Proposition 3.7 of [3].                                                        
Lemma 4.10. For generic J, MX,H,J (γ, x) has a structure of a C ∞ -manifold of dimension
ind Morse (γ) − ind CZ (x) for any γ and x as above.

Proof. The case x = pr ◦ γ is discussed in Lemma 4.9 (ii). The other cases follow from
the standard argument using [8]. See pp.313 of [3].                                  

  Let us state the following C 0 -estimate. For comments on the proof see Remark 4.14.
Lemma 4.11. If sup kJt − Jstd kC 0 is sufficiently small, then for any γ and x as above
                   t∈S 1

                                         sup          |u(s, t)| < ∞.
                                    u∈MX,H,J (γ,x)
                                     (s,t)∈R≥0 ×S 1


  By these results and the standard compactness and glueing arguments (see Sections 3.3
and 3.4 of [3]), generic J which is sufficiently close to Jstd satisfies the following properties:

     • For any γ and x as above satisfying ind Morse (γ) − ind CZ (x) = 0, the moduli space
       MX,H,J (γ, x) is a finite set.
     • A linear map
                                                       X
      Ψ : CM [a,b)
              ∗    (L, X) →  CF  [a,b)
                                 ∗     (H, J); γ →
                                                 7                 #2 MX,H,J (γ, x) · x
                                                           ind CZ (x)=ind Morse (γ)

        is a chain map with respect to boundary operators ∂LH ,X and ∂H,J .
                                                      28
Finally, Lemma 4.9 implies that Ψ is an isomorphism (see Section 3.5 of [3]). In particular,
H∗ (Ψ) : HM [a,b)
            ∗     (L) → HF [a,b)
                           ∗     (H) is an isomorphism.
  For any a, b, a′ , b′ ∈ R satisfying a < b, a′ < b′ , a ≤ a′ and b ≤ b′ , the commutativity of
the following diagram is straightforward from the definition of Ψ:
                                       [a,b)                                      [a′ ,b′ )
(10)                             CM ∗               (LH )        /       CM ∗                 (LH )
                                       ∼
                                       =                                                    ∼
                                                                                            =
                                                                                       
                                       [a,b)                                      [a′ ,b′ )
                                  CF ∗              (H)              /   CF ∗                 (H).

This implies the commutativity of (7).


4.5. Commutativity of monotonicity maps. The goal of this subsection is to prove
the commutativity of (8). Let us take the following data:

       • H, H ′ ∈ C ∞ (S 1 × T ∗ Rn ) satisfying (H0), (H1), (H2) and H(t, q, p) < H ′ (t, q, p)
         for any (t, q, p) ∈ S 1 × T ∗ Rn .
       • Real numbers a < b.
       • Almost complex structures J, J ′ and vector fields X, X ′ such that chain complexes
         CF [a,b)
             ∗    (H, J), CF [a,b)
                             ∗     (H ′ , J ′ ), CM [a,b)
                                                    ∗     (LH , X), CM [a,b)
                                                                       ∗     (LH ′ , X ′ ) are defined.

  Without loss of generality, we may assume P(LH ) ∩ P(LH ′ ) = ∅. Indeed, for any H
and H ′ satisfying (H0), (H1), (H2) and H < H ′ , there exists a strictly increasing sequence
(Hj )j≥1 such that every Hj satisfies (H0), (H1), (H2), lim Hj = H, and
                                                                                            j→∞

       P(LHj ) ∩ P(LHj+1 ) = ∅,                 P(LHj ) ∩ P(LH ) = ∅,                                    P(LHj ) ∩ P(LH ′ ) = ∅
for every j ≥ 1. Then, assuming that the commutativity of (8) is proved for pairs
(Hj , Hj+1), (Hj , H) and (Hj , H ′) for every j, the commutativity of (8) for (H, H ′ ) follows
by taking limits.
  In the previous subsection we defined isomorphisms of chain complexes Ψ : CM [a,b)     ∗     (LH , X) →
   [a,b)              ′      [a,b)          ′        [a,b) ′   ′
CF ∗ (H, J) and Ψ : CM ∗ (LH ′ , X ) → CF ∗ (H , J ). We also defined chain maps
ΦL : CM [a,b)
          ∗   (LH , X) → CM [a,b)
                               ∗   (LH ′ , X ′) and ΦH : CF [a,b)
                                                             ∗    (H, J) → CF [a,b)
                                                                              ∗     (H ′, J ′ ). Our
goal is to show that the following diagram commutes up to homotopy:
                                   [a,b)                     Ψ                       [a,b)
(11)                           CM ∗        (LH , X)          ∼
                                                                             /   CF ∗               (H, J)
                                                             =
                                     ΦL                                                             ΦH
                                                                                               
                                   [a,b)                     ∼
                                                             =                     [a,b)
                              CM ∗         (LH ′ , X ′ )                 /   CF ∗           (H ′ , J ′ ).
                                                            Ψ′

This immediately implies the commutativity of the diagram (8). Since vector spaces in
the diagram (11) are generated by finitely many critical points, boundary operators and
chain maps in this diagram do not change under C ∞ -small perturbations of X, X ′ , J, J ′ .
Hence we may assume that these data are taken so that all moduli spaces which appear
in the rest of this subsection are cut out transversally.
                                                            29
  To prove that (11) commutes up to homotopy, first we define a linear map
                                                      X
  Θ : CM [a,b)
          ∗    (LH , X) → CF [a,b)
                             ∗     (H ′ , J ′ ); γ 7→          #2 MX,H ′ ,J ′ (γ, x) · x.
                                                           ind Morse (γ)=ind CZ (x)

Θ is a chain map (namely ∂H ′ ,J ′ ◦ Θ = Θ ◦ ∂LH ,X ) by the same reason that Ψ in the
previous subsection is a chain map. We are going to prove ΦH ◦ Ψ ∼ Θ ∼ Ψ′ ◦ ΦL .
 First we prove Ψ′ ◦ ΦL ∼ Θ. For any γ ∈ P(LH ) and x ∈ P(H ′ ) such that
SLH (γ), AH ′ (x) ∈ [a, b), let N 0 (γ, x) denote the set of (α, u, v) where
                   α ∈ R≥0 ,     u : [0, α] → Λ,        v ∈ L1,3 (R≥0 × S 1 , T ∗ Rn )
such that
                     u(0) ∈ W u (γ : X),       u(s) = ϕsX ′ (u(0)) (∀s ∈ [0, α]),
                     ∂s v − Jt′ (∂t v − XHt′ (v)) = 0,
                     pr ◦ v0 = u(α),      lim vs = x.
                                         s→∞


  Let us state the following C 0 -estimate:
Lemma 4.12. If sup kJt − Jstd kC 0 is sufficiently small, then for any γ and x as above
                    t∈S 1

                                         sup           |v(s, t)| < ∞.
                                   (α,u,v)∈N 0 (γ,x)
                                    (s,t)∈R≥0 ×S 1


  For generic J ′ which is sufficiently close to Jstd , N 0 (γ, x) is a finite set for any γ and
x satisfying ind CZ (x) = ind Morse (γ) + 1, and the linear map
                                 [a,b)
                                                            X
       K 0 : CM [a,b)
                 ∗    (LH ) → CF ∗+1 (H ′ ); γ 7→                         #2 N 0 (γ, x) · x
                                                          ind CZ (x)=ind Morse (γ)+1

satisfies ∂H ′ ,J ′ ◦ K 0 + K 0 ◦ ∂LH ,X = Θ − Ψ′ ◦ ΦL . For details see Section 4.3 of [18].
  Secondly we prove ΦH ◦ Ψ ∼ Θ. Let us take (Hs,t)(s,t)∈R×S 1 and (Js,t)(s,t)∈R×S 1 which
satisfy (HH1), (HH2), (HH3) and (JJ1), (JJ2). In particular there exists s2 > 0 such that
                                         ®
                                          (Ht , Jt ) (s ≤ −s2 )
                         (Hs,t , Js,t) =
                                          (Ht′ , Jt′ ) (s ≥ s2 ).
For any γ ∈ P(LH ) and x ∈ P(H ′) such that SLH (γ), AH ′ (x) ∈ [a, b), let N 1 (γ, x)
denote the set of (β, w) where
                            β ∈ R≤s2 ,       w ∈ L1,3 (R≥β × S 1 , T ∗ Rn )
such that
                 pr ◦ wβ ∈ W u (γ : X),          ∂s w − Js,t (∂t w − XHs,t (w)) = 0
                  lim ws = x.
                  s→∞


  Let us state the following C 0 -estimate:
                                                   30
Lemma 4.13. If sup kJt − Jstd kC 0 is sufficiently small, then for any γ and x as above
                     t∈S 1



                                           sup        |w(s, t)| < ∞.
                                    (β,w)∈N 1 (γ,x)
                                     (s,t)∈R≥β ×S 1



  For generic J which is sufficiently close to Jstd , N 1 (γ, x) is a finite set for any γ and x
satisfying ind CZ (x) = ind Morse (γ) + 1, and the linear map

                                   [a,b)
                                                                     X
         K 1 : CM [a,b)
                  ∗     (L) → CF ∗+1 (H ′);      γ 7→                                   #2 N 1 (γ, x) · x
                                                           ind CZ (x)=ind Morse (γ)+1



satisfies ∂H ′ ,J ′ ◦ K 1 + K 1 ◦ ∂L,X = Θ − ΦH ◦ Ψ. For details see Section 4.3 of [18].

Remark 4.14 (Proofs of C 0 -esimtates). C 0 -estimates in this section, namely Lemmas
4.11, 4.12, 4.13, are slight generalizations of Lemmas 4.8, 4.9, 4.10 in [18]. These results
in [18] are stated for Hamiltonians of special type (i.e. elements of the sequence (H m )m
defined in Section 4.1 of [18]), however the proofs of these results in [18] use only assump-
tions (JJ1), (JJ2), (HH1), (HH2), (HH3). Hence the proofs in [18] work without any
modification for Lemmas 4.11, 4.12, 4.13. Strictly speaking, the condition (HH3) in [18]
requires b(s) ≡ 0 in the condition (HH3) in this paper. Namely, if H ∈ C ∞ (R×S 1 ×T ∗ Rn )
satisfies (HH3) in this paper, then there exists b ∈ C ∞ (R) such that

(12)                            H 0 (s, t, q, p) := H(s, t, q, p) − b(s)

satisfies the condition (HH3) in [18]. However, this difference does not affect Floer equa-
                                                                                      1
                                                      0 (q, p) for any (s, t) ∈ R × S
tions, since (12) obviously implies XHs,t (q, p) = XHs,t                                and
           ∗ n
(q, p) ∈ T R .


4.6. Proof of Theorem 3.4. Now we can complete the proof of Theorem 3.4. Let K
be any nonempty, compact and fiberwise convex set in T ∗ Rn . Taking time-dependent
perturbations of Hamiltonians obtained in Lemma 3.10, there exists a sequence (Hj )j≥1
in C ∞ (S 1 × T ∗ Rn ) which satisfies the following conditions:

       • Hj satisfies (H0), (H1), (H2) for every j ≥ 1.
                            (t, q, p) for every j ≥ 1 and (t, q, p) ∈ S 1 × T ∗ Rn .
       • Hj (t, q, p) < Hj+1®
                              0 ((q, p) ∈ K)
       • lim Hj (t, q, p) =                       for any (t, q, p) ∈ S 1 × T ∗ Rn .
         j→∞                  ∞ ((q, p) ∈   / K)

  For each j, let Lj := LHj ∈ C ∞ (S 1 × T Rn ) denote the Legendre dual of Hj . Then,
(SL−1
    j
      (R<c ))j≥1 is an increasing sequence of open sets in Λ for any c ∈ R. Moreover
                                                      31
∞
[
      SL−1
         j
           (R<c ) = ΛcK by Lemma 3.12 (iii). Then we obtain
j=1

                        SH [a,b)
                           ∗     (K) = lim HF [a,b)
                                              ∗     (Hj )
                                       −→
                                       j→∞
                                    ∼
                                    = lim H (S −1 (R<b ), SL−1 (R<a ))
                                       −→ ∗ Lj               j
                                      j→∞
                                          ∞
                                         Å[               ∞            ã
                                    ∼
                                                         [
                                               −1               −1
                                    = H∗     SLj (R<b ),    SLj (R<a )
                                             j=1            j=1

                                    = H∗ (ΛbK , ΛaK ),
where the isomorphism on the second line follows from the commutativity of (8). Finally,
the commutativity of (6) follows from the commutativity of (9) and taking limits of
Hamiltonians. This completes the proof of Theorem 3.4.                                


                                  5. Proof of Theorem 1.4

   The goal of this section is to prove Theorem 1.4. Namely, we prove cSH (K) = cEHZ (K)
for any convex body K ⊂ T ∗ Rn .
  The case n = 1 can be proved by the following simple argument. For any convex
body K ⊂ T ∗ R1 , both cEHZ (K) and the Hamiltonian displacement energy of K (denoted
by e(K)) are equal to the measure of K. On the other hand, cEHZ (K) ≤ cSH (K) (by
Lemma 2.13 (iii)) and cSH (K) ≤ e(K) (second inequality in Theorem 1.4 of [14]), thus
cEHZ (K) = cSH (K) = e(K).
  Hence we assume n ≥ 2 in the rest of the proof. Let us first introduce the notion of
nice convex bodies.
Definition 5.1. A convex body K ⊂ T ∗ Rn is called nice if ∂K is of C ∞ and strictly
convex, and there exists a C ∞ -map Γ : S 1 → ∂K which satisfies the following conditions:

   (i): Γ̇(t) generates ker(ωn |TΓ(t) ∂K ) and of positive direction (i.e. ωn (X, Γ̇(t)) > 0 for
        any X ∈ TΓ(t) (T ∗ Rn ) which points strictly outwards) for every t ∈ S 1 ,
        Z         n
                 ÅX       ã
               ∗
  (ii):      Γ      pi dqi = cEHZ (K),
           S1     i=1
  (iii): pr ◦ Γ(S 1 ) ⊂ int (pr(K)).

Any curve Γ which satisfies these three conditions is called a nice curve on ∂K.
Remark 5.2. The convex body B := {(q, p) ∈ T ∗ Rn | |q|2 + |p|2 ≤ 1} is not nice. Indeed,
if Γ : S 1 → ∂B satisfies the conditions (i) and (ii) above, then Γ(S 1 ) = {(e sin t, e cos t) |
t ∈ R/2πZ}, thus pr(Γ(S 1 )) = {es | −1 ≤ s ≤ 1}. Hence pr(Γ(S 1 )) is not contained in
int (pr(B)) = {q ∈ Rn | |q| < 1}.
Lemma 5.3. When n ≥ 2, for any convex body K ⊂ T ∗ Rn , there exists a sequence of
nice convex bodies which converges to K in the Hausdorff distance.
                                                   32
Proof. It is easy to see that there exists a sequence (Kj )j such that each ∂Kj is of C ∞
and strictly convex, and lim Kj = K in the Hausdorff distance. Thus it is sufficient to
                             j→∞
show that, for any convex body C ⊂ T ∗ Rn such that ∂C is of C ∞ and strictly convex,
there exists C ′ which is nice and arbitrarily close to C. Since C is strictly convex,
                               LC := {x ∈ ∂C | pr(x) ∈ ∂(pr(C))}
is a submanifold of ∂C which is diffeomorphic to S n−1 , in particular its codimension in
∂C is n. Since n ≥ 2, there exists C ′ which is arbitrarily C ∞ -close to C, and all closed
characteristics of ∂C ′ are disjoint from LC ′ , which implies that C ′ is nice.         

  By Lemma 5.3, Theorem 1.4 is reduced to the following theorem:
Theorem 5.4. For any n ∈ Z≥2 and any nice convex body K ⊂ T ∗ Rn , there holds
cSH (K) = cEHZ (K).

  In the rest of this section we prove Theorem 5.4. Let n ∈ Z≥2 and K be any nice
convex body in T ∗ Rn . Let Γ be a nice curve on ∂K, and γ := pr ◦ Γ : S 1 → int (pr(K)).
By Lemma 3.3 (iii), there holds
                                    Z      ÅX        ã
                                         ∗
                         lenK (γ) =    Γ       pi dqi = cEHZ (K).
                                              S1          i

Lemma 5.5. γ̇(t) 6= 0 for any t ∈ S 1 .

Proof. Let ν be a unit vector which is normal to TΓ(t) (∂K). Since Γ̇(t) is parallel to
Jstd (ν), it is sufficient to show that the p-component of ν is nonzero. If the p-component
of ν is zero, then the convexity of K implies (q, p) ∈ K =⇒ q · ν ≤ γ(t) · ν, thus
γ(t) ∈ ∂(pr(K)), which contradicts the assumption γ(S 1 ) ⊂ int (pr(K)).                 
                                   ∞                        ∞  1
Lemma 5.6. LetÅ(γs )−1≤s≤1 ã be a C -family of elements of C (S , int (pr(K))) such that
             d
γ0 = γ. Then    lenK (γs )      = 0.
             ds             s=0


Proof. Since γ̇(t) 6= 0 for any t ∈ S 1 , we may assume that γ̇s (t) 6= 0 for any (s, t) ∈
[−1, 1] × S 1 . Let us define γ̄s : S 1 → ∂K as in Lemma 3.3 (iii). Namely,
                  γ̄s (t) = (γs (t), pγs (t)),           pγs (t) · γ̇s (t) = max p · γ̇s (t).
                                                                             p∈Kγs (t)
                                    Z               ÅX             ã
Then Γ = γ̄0 , and lenK (γs ) =           (γ̄s )∗         pi dqi       for every s ∈ [−1, 1]. Thus
                                     S1              i

  d                     d
     Å           ã        ÅZ            ÅX        ãã       Z
                                      ∗
      lenK (γs )      =        (γ̄s )      pi dqi        =      ωn ((∂s γ̄s )s=0 (t), Γ̇(t)) dt = 0.
  ds              s=0   ds S 1           i           s=0    S 1


                                                                                                       

  For any a ∈ R≥0 and x ∈ Rn , let us define γa,x ∈ Λ by γa,x (t) := aγ(t) + x. Let
                         T := {(a, x) ∈ R≥0 × Rn | γa,x (S 1 ) ⊂ pr(K)}.
                                                          33
It is easy to see that T is a compact convex set in R≥0 × Rn . Let us define a function
L : T → R by L(a, x) := lenK (γa,x ). Obviously L(1, 0, . . . , 0) = lenK (γ) = cEHZ (K). By
Lemma 3.3 (iv), L is continuous.
Lemma 5.7. L(a, x) ≤ L(1, 0, . . . , 0) for any (a, x) ∈ T .

Proof. By the continuity of L, it is sufficient to prove the lemma for (a, x) ∈ int T . For
any s ∈ [0, 1], let
                  γs := γsa+(1−s),sx ,           Ls := lenK (γs ) := L(sa + (1 − s), sx)
Our goal is to prove L1 ≤ L0 .
   For any s ∈ [0, 1], we have (sa + (1 − s), sx) ∈ int T . This implies that γs (S 1 ) ⊂
int (pr(K)) and sa + (1 − s) > 0, thus γ̇s (t) = (sa + (1 − s))γ̇(t) 6= 0 for any t ∈ S 1 . Let
us abbreviate pγs as ps . Then
                                      Z
                                Ls =        ps (t) · γ̇s (t) dt.
                                                     S1

By (γ0 (t), p0 (t)), (γ1 (t), p1 (t)) ∈ K and the convexity of K,
                                     (γs (t), (1 − s)p0 (t) + sp1 (t)) ∈ K.
Then
               ps (t) · γ̇s (t) = max p · γ̇s (t) ≥ ((1 − s)p0 (t) + sp1 (t)) · γ̇s (t).
                                     p∈Kγs (t)

On the other hand γ̇s (t) = (sa + (1 − s))γ̇(t), thus
                       Z
                Ls ≥       (1 + (a − 1)s)γ̇(t) · (p0 (t) + (p1 (t) − p0 (t))s) dt
                          S1
and the equality holds for s = 0. Hence
                                   Z
                      ∂s Ls |s=0 ≥   γ̇(t) · ((a − 2)p0 (t) + p1 (t)) dt.
                                             S1

On the other hand ∂s Ls |s=0 = 0 by Lemma 5.6. Then we obtain
                      Z                                Z
                           γ̇(t) · p1 (t) dt ≤ (2 − a)   γ̇(t) · p0 (t) dt.
                           S1                                   S1
Now we can finish the proof by
                       Z
            L1 − L0 =      aγ̇(t) · p1 (t) − γ̇(t) · p0 (t) dt ≤ −(a − 1)2 L0 ≤ 0.
                                S1
The first inequality follows from a ≥ 0, and the second inequality follows from L0 ≥ 0,
which is obvious since L0 = lenK (γ) = cEHZ (K) > 0.                                 

  We have proved
                                max lenK (γa,x ) = lenK (γ) = cEHZ (K).
                            (a,x)∈T

On the other hand, if (a, x) ∈
                             / T , then lenK (γa,x ) = −∞. Thus for any C > cEHZ (K), one
can define a map
                ℓC : (R≥0 × Rn , R≥0 × Rn \ T ) → (ΛC    0
                                                    K , ΛK );              (a, x) 7→ γa,x .
                                                          34
Now consider the commutative diagram
                                                        C)
                                                   Hn (jK
                          Hn (Rn , Rn \ pr(K))              / Hn (ΛC , Λ0 )
                                                                   K    K
                                                              4
                                                     ✐✐✐ ✐✐ ✐
                                                           ✐
                                                   ✐✐
                                               ✐✐✐✐
                                          ✐✐✐✐ Hn (ℓC )
                      Hn (R≥0 × Rn , R≥0 × Rn \ T )
where the vertical map is induced by the map q 7→ (0, q). Since T is bounded, the vertical
                     C
map is 0. Then H∗ (jK  ) = 0, which implies cSH (K) ≤ C. Since C is any number larger
than cEHZ (K), we obtain cSH (K) ≤ cEHZ (K). The inverse inequality cSH (K) ≥ cEHZ (K)
follows from Proposition 2.13 (iii), thus we have proved Theorem 5.4, to which Theorem
1.4 was reduced.                                                                        


                               6. Proof of Theorem 1.8

  The goal of this section is to prove Theorem 1.8. Let us recall the situation: K is
a compact set in T ∗ Rn with int (K) 6= ∅, Π is a hyperplane which intersects int (K),
Π+ and Π− are distinct closed halfspaces with ∂Π+ = ∂Π− = Π, and K + := K ∩ Π+ ,
K − = K ∩ Π− . Then our goal is to prove
                      cHZ (K) ≤ cEHZ (conv (K + )) + cEHZ (conv (K − )),
where conv denotes the convex hull.
   Let K ′ := conv (K + ) ∪ conv (K − ). Then K ′ is star-shaped, thus it is a RCT set. We
first need the following lemma:
Lemma 6.1. If C ⊂ T ∗ Rn is a RCT set satisfying int (C) 6= ∅, then cHZ (C) ≤ cSH (C).

Proof. First we need to recall Corollary 3.5 of [17]: for any 2n-dimensional Liouville
domain (W, λ) and a ∈ R>0 \ Spec(W, λ) such that the canonical map ιa : H n−∗ (W ) →
HF <a
    ∗ (W, λ) satisfies ιa (1) = 0, there holds cHZ (int W, dλ) ≤ a. Moreover, since Spec(W, λ)
is a measure zero set, the assumption a ∈    / Spec(W, λ) can be omitted.
   Now let us assume that C ⊂ T ∗ Rn is a C ∞ -RCT set with a nice action spectrum in the
sense of [14]. There exists X ∈ X (T ∗ Rn ) satisfying LX ωn ≡ ωn and X points outwards
on ∂C. Setting λ := (iX ωn )|C , (C, λ) is a Liouville domain and there exists a canonical
isomorphism HF <a          ∼     [0,a)
                                       (C) such that ιa corresponds to iaC (see Section 4, in
                 ∗ (C, λ) = SH ∗
particular Proposition 4.5 of [14]). Now, if a > cSH (C) then ιa (1) = 0, thus cHZ (C) ≤ a.
This completes the proof when C is a C ∞ -RCT set with a nice action spectrum.
  Let C be an arbitrary RCT set in T ∗ Rn . Then there exists a sequence of C ∞ -RCT sets
                                                                              \∞
(with nice action spectra) (Cj )j≥1 such that Cj+1 ⊂ Cj for every j ≥ 1 and       Cj = C.
                                                                                 j=1
Then SH [0,a)
        ∗     (C) ∼
                  = lim SH [0,a)
                           ∗     (Cj ) for every a > 0, which implies cSH (C) = lim cSH (Cj ).
                    −→                                                          j→∞
                    j→∞
On the other hand, for each j there holds cHZ (C) ≤ cHZ (Cj ) ≤ cSH (Cj ) thus we obtain
cHZ (C) ≤ lim cSH (Cj ) = cSH (C).                                                    
          j→∞

                                              35
  Now let us state the key inequality:
Lemma 6.2. cSH (K ′ ) ≤ cEHZ (conv (K + )) + cEHZ (conv (K − )).

  Assuming Lemma 6.2, we obtain
           cHZ (K) ≤ cHZ (K ′ ) ≤ cSH (K ′ ) ≤ cEHZ (conv (K + )) + cEHZ (conv (K − )),
where the first inequality follows from K ⊂ K ′ , the second inequality follows from Lemma
6.1, and the last inequality is Lemma 6.2. Hence we have reduced Theorem 1.8 to Lemma
6.2.

6.1. Proof of Lemma 6.2. The case n = 1 is easy to prove. Indeed, for any compact
S ⊂ T ∗ R1 satisfying int (S) 6= ∅, there holds cHZ (S) ≤ |S|, where | · | denotes the measure.
Also, |S| = cEHZ (S) if S is convex. Then we can prove the case n = 1 by
  cHZ (K ′ ) ≤ |K ′ | = |conv (K + )| + |conv (K − )| = cEHZ (conv (K + )) + cEHZ (conv (K − )).

  Hence   in the rest of the proof we may assume n ≥ 2. We may also assume that
Π = {q1   = 0}, since for any hyperplane Π there exists an affine map A on T ∗ Rn with
A∗ ωn =   ωn and A(Π) = {q1 = 0}. Finally, we assume that K + = K ∩ {q1 ≥ 0},
K− = K    ∩ {q1 ≤ 0}.
Lemma 6.3. K ′ is fiberwise convex.

Proof. Let q = (q1 , . . . , qn ) ∈ Rn . If q1 > 0, then Kq′ = K ′ ∩ Tq∗ Rn = conv (K + ) ∩ Tq∗ Rn ,
thus Kq′ is convex. Similarly, if q1 < 0, then Kq′ = conv (K − ) ∩ Tq∗ Rn , thus Kq′ is convex.
Finally, when q1 = 0, there holds Kq′ = conv (K + ) ∩ Tq∗ Rn = conv (K − ) ∩ Tq∗ Rn , since
conv (K + ) ∩ {q1 = 0} = conv (K ∩ {q1 = 0}) = conv (K − ) ∩ {q1 = 0}. In particular, Kq′
is convex.                                                                                       

                                             A       n   n      ′       A      0
  For any A ∈ R>0 , let us consider the map jK ′ : (R , R \ pr(K )) → (ΛK ′ , ΛK ′ ) which

maps each q ∈ Rn to the constant loop at q. By Corollary 3.8, to prove Lemma 6.2 it is
sufficient to prove the following:
(13)            A > cEHZ (conv (K + )) + cEHZ (conv (K − )) =⇒ Hn (jK
                                                                    A
                                                                      ′ ) = 0.


By Lemma 5.3, there exist nice convex bodies C + and C − such that conv (K + ) ⊂ C + ,
conv (K − ) ⊂ C − and cEHZ (C + ) + cEHZ (C − ) < A. Let Γ+ : S 1 → ∂C + be a nice curve
on C + , and Γ− : S 1 → ∂C − be a nice curve on C − . By changing parameterizations if
necessary, we may assume that the following properties hold:

       • The q1 -component of pr ◦ Γ+ : S 1 → Rn takes its minimum at 0 ∈ S 1 ,
       • The q1 -component of pr ◦ Γ− : S 1 → Rn takes its maximum at 0 ∈ S 1 .

Then there exist γ + : S 1 → R≥0 × Rn−1 and γ − : S 1 → R≤0 × Rn−1 such that γ + − pr ◦ Γ+
and γ − − pr ◦ Γ− are constant maps from S 1 to Rn .
Remark 6.4. By Lemma 5.5, γ + and γ − are nonconstant.
                                                36
Lemma 6.5.                 (i): For any a ∈ R≥0 and x ∈ R≥0 × Rn−1 ,
                                                +
                                               γa,x : S 1 → Rn ; t 7→ aγ + (t) + x
                            +
         satisfies lenK ′ (γa,x ) ≤ cEHZ (C + ).
   (ii): For any a ∈ R≥0 and x ∈ R≤0 × Rn−1 ,
                                                −
                                               γa,x : S 1 → Rn ; t 7→ aγ − (t) + x
                           −
        satisfies lenK ′ (γa,x ) ≤ cEHZ (C − ).

Proof. Since γa,x  +
                      (S 1 ) ⊂ R≥0 × Rn−1 and K ′ ∩ pr−1 (R≥0 × Rn−1 ) ⊂ C + , there holds
         +                +                                                    +
lenK ′ (γa,x ) ≤ lenC + (γa,x ). On the other hand, Lemma 5.7 implies lenC + (γa,x ) ≤ cEHZ (C + ),
which completes the proof of (i). The proof of (ii) is similar to the proof of (i).             

   For any (s, t, x2 , . . . , xn ) ∈ (R2 \ (R<0 )2 ) × Rn−1 , we define γs,t,x2,...,xn : S 1 → Rn as
follows:

     • When s ≤ 0 and t ≥ 0,
                                 ®
                                  t · γ + (2θ) + (−s, x2 , . . . , xn ) (0 ≤ θ ≤ 1/2)
           γs,t,x2,...,xn (θ) :=
                                  (−s, x2 , . . . , xn )                (1/2 ≤ θ ≤ 1).
     • When s, t ≥ 0,
                                     ®
                                      t · γ + (2θ) + (0, x2 , . . . , xn )   (0 ≤ θ ≤ 1/2)
               γs,t,x2,...,xn (θ) :=        −
                                      s · γ (2θ − 1) + (0, x2 , . . . , xn ) (1/2 ≤ θ ≤ 1).
     • When s ≥ 0 and t ≤ 0,
                                ®
                                 (t, x2 , . . . , xn )                  (0 ≤ θ ≤ 1/2)
          γs,t,x2,...,xn (θ) :=       −
                                 s · γ (2θ − 1) + (t, x2 , . . . , xn ) (1/2 ≤ θ ≤ 1).

Then, Lemma 6.5 implies
                             sup                        lenK ′ (γs,t,x2,...,xn ) ≤ cEHZ (C + ) + cEHZ (C − ) < A,
         (s,t,x2 ,...,xn   )∈(R2 \(R   <0   )2 )×Rn−1

thus one can define a map
                ℓA : (R2 \ (R<0 )2 ) × Rn−1 → ΛA
                                               K ′ ; (s, t, x2 , . . . , xn ) 7→ γs,t,x2 ,...,xn .

It is easy to check that ℓA is continuous with respect to the L1,2 -topology on Λ. For any
(x1 , . . . , xn ) ∈ Rn , let c(x1 ,...,xn ) denote the constant map from S 1 to (x1 , . . . , xn ).
Lemma 6.6.                 (i): For any r ∈ R≤0 ,
                            γr,0,x2,...,xn = c(−r,x2 ,...,xn ) ,        γ0,r,x2,,...,xn = c(r,x2 ,...,xn ) .
   (ii): There exists R ∈ R>0 such that
                  max{|s|, |t|, |(x2, . . . , xn )|} > R =⇒ lenK ′ (γs,t,x2 ,...,xn ) = −∞.
                                                                   37
Proof. (i) follows directly from the definition. To prove (ii), let us take R > 0 so that
the following conditions hold:
      B n (R) ⊃ pr(K ′ ),        R · min{diam(γ + (S 1 )), diam(γ − (S 1 ))} ≥ diam(pr(K ′ )).
Here B n (R) := {q ∈ Rn | |q| ≤ R} and diam denotes the diameter. Note that the
second condition can be achieved when R is sufficiently large, since γ + and γ − are both
nonconstant maps (see Remark 6.4).
   Let us prove that such R satisfies the required conditions: if lenK ′ (γs,t,x2,...,xn ) > −∞
(which is equivalent to γs,t,x2,...,xn (S 1 ) ⊂ pr(K ′ )) then max{|s|, |t|, |(x2, . . . , xn )|} ≤ R. It
is sufficient to consider the following three cases:

      • s ≤ 0 and t ≥ 0 : Since t · diam(γ + (S 1 )) ≤ diam(pr(K ′ )), we obtain t ≤ R. Since
        γs,t,x2,...,xn (0) = (−s, x2 , . . . , xn ) ∈ pr(K ′ ) ⊂ B n (R), we obtain |s|, |(x2, . . . , xn )| ≤
        R.
      • s, t ≥ 0 : Since t · diam(γ + (S 1 )), s · diam(γ − (S 1 )) ≤ diam(pr(K ′ )), we obtain
        t, s ≤ R. Since γs,t,x2 ,...,xn (0) = (0, x2 , . . . , xn ) ∈ pr(K ′ ) ⊂ B n (R), we obtain
        |(x2 , . . . , xn )| ≤ R.
      • s ≥ 0 and t ≤ 0 : this case is similar to the first case.

                                                                                                           

  Let us define h : Rn → (R2 \ (R<0 )2 ) × Rn−1 by
                                              ®
                                               (−x1 , 0, x2 , . . . , xn ) (x1 ≥ 0),
                   h(x1 , x2 , . . . , xn ) =
                                               (0, x1 , x2 , . . . , xn )  (x1 ≤ 0).
Then Lemma 6.6 (i) implies ℓA ◦ h(x1 , . . . , xn ) = c(x1 ,...,xn) . By Lemma 6.6 (ii), when
R ∈ R>0 is sufficiently large,
                                                               −∞
            Hn (ℓA ◦ h) : Hn (Rn , Rn \ B n (R)) → Hn (ΛA                   A      0
                                                        K ′ , ΛK ′ ) → Hn (ΛK ′ , ΛK ′ )

is zero. We may also assume that pr(K ′ ) ⊂ B n (R). Now the diagram
                                                              A )
                                                         Hn (jK ′
                                     n   n          ′        / Hn (ΛA ′ , Λ0 ′ )
                              Hn (R , R \ pr(K ))                   K      K
                                          O                     5
                                                         ❦❦ ❦❦❦
                                                      ❦❦❦
                                                   ❦❦❦
                                                ❦❦❦ Hn (ℓA ◦h)
                              Hn (Rn , Rn \ B n (R))
commutes, the vertical map is surjective (since pr(K ′ ) is star-shaped) and the diagonal
                       A
map is zero, thus Hn (jK ′ ) = 0, which completes the proof of (13).                   

                                 7. Proof of Proposition 1.11

  First let us introduce a few notations. For any S ⊂ Rn , let
            D ∗ S := {(q, p) ∈ T ∗ Rn | q ∈ S, |p| ≤ 1},
            w(S) := inf{sup h − inf h | h ∈ Cc∞ (Rn ), |dh(x)| ≥ 1 for any x ∈ S},
            r(S) := sup{r | there exists q ∈ Rn with B n (q : r) ⊂ S}.
B n (q : r) denotes the closed ball in Rn with center q and radius r.
                                                        38
  Our goal is to show that, for any bounded B ⊂ T ∗ Rn and any ε ∈ R>0 , there exist
compact star-shaped sets K1 , K2 ⊂ T ∗ Rn such that B ⊂ K1 ∪ K2 and e(K1 ), e(K2 ) < ε.
Note that, for any compact K ⊂ T ∗ Rn and a > 0, there holds e(aK) = a2 e(K). Thus we
may assume that B is a subset of D ∗ B n (1) = {(q, p) ∈ T ∗ Rn | |q|, |p| ≤ 1}.
  For any nonempty compact S ⊂ Rn , there holds
                                   e(D ∗ S) ≤ 2w(S) ≤ Cn r(S)
where Cn is a positive constant which depends only on n. The first inequality is proved
in Lemma 4 of [16], and the second inequality is proved in Section 2.2 of [16], although
notations and settings in this section are slightly different from those in [16].
   For any θ, let Rθ denote the anti-clockwise rotation of R2 with center (0, 0) and angle
θ. For any integer N ≥ 1, let
                T (N) := {(r cos θ, r sin θ) | 0 ≤ r ≤ 1, 0 ≤ θ ≤ π/N} ⊂ R2 .
Moreover, for any i ∈ {1, 2}, let us define Si (N) ⊂ R2 and S̄i (N) ⊂ Rn by
                N −1                                    ®
                 [                                        Si (N)          (n = 2)
      Si (N) :=      R (i+2j−1)π (T (N)),    S̄i (N) :=             n−2
                 j=0
                           N                              Si (N) × B (1) (n ≥ 3).

Then D ∗ S̄i (N) ⊂ T ∗ Rn is a compact star-shaped set for any i ∈ {1, 2}, and there holds
                            B ⊂ D ∗ B n (1) ⊂ D ∗ S̄1 (N) ∪ D ∗ S̄2 (N).
On the other hand, for any N and i,
                                                                       π
                           r(S̄i (N)) ≤ r(Si (N)) ≤ r(T (N)) ≤           .
                                                                      2N
             πCn
Thus, if N >      , then max e(D ∗ S̄i (N)) < ε. One can complete the proof by taking
               2ε        1≤i≤2
such N and setting Ki := D ∗ S̄i (N) (i = 1, 2).                                   

                                           References
  [1] A. Abbondandolo, J. Kang, Symplectic homology of convex domains and Clarke’s duality, arXiv:
      1907.07779.
  [2] A. Abbondandolo, P. Majer, Lectures on the Morse complex for infinite-dimensional manifolds, in
      ‘Morse theoretic methods in nonlinear analysis and in symplectic topology’ (P. Biran, O. Cornea
      and F. Lalonde, eds.), Springer, Dordrecht, 2006, 1–74.
  [3] A. Abbondandolo, M. Schwarz, On the Floer homology of cotangent bundles, Comm. Pure Appl.
      Math. 59 (2006), 254–316.
  [4] A. Abbondandolo, M. Schwarz, A smooth pseudo-gradient for the Lagrangian action functional,
      Adv. Nonlinear Stud. 9 (2009), 597–623.
  [5] A. Akopyan, R. Karasev, F. Petrov, Bang’s problem and symplectic invariants, J. Symplectic
      Geom. 17 (2019), 1579–1611.
  [6] I. Ekeland, H. Hofer, Symplectic topology and Hamiltonian dynamics II, Math. Z. 203 (1990),
      553–567.
  [7] A. Floer, H. Hofer, Symplectic Homology I, Open sets in Cn , Math. Z. 215 (1994), 37–88.
  [8] A. Floer, H. Hofer, D. Salamon, Transversality in elliptic Morse theory for the symplectic action,
      Duke Math. J. 80 (1996), 251–292.
  [9] A. Floer, H. Hofer, K. Wysocki, Application of Symplectic Homology I, Math. Z. 217 (1994),
      577–606.
                                                  39
 [10] V. Ginzburg, J. Shon, On the filtered symplectic homology of prequantization bundles, Internat. J.
      Math. 29 (2018), no. 11, 1850071.
 [11] J. Gutt, M. Hutchings, Symplectic capacities from positive S 1 -equivariant symplectic homology,
      Algebr. Geom. Topol 18 (2018), 3537–3600.
 [12] J. Gutt, M. Hutchings, V. G. B. Ramos, Examples around the strong Viterbo conjecture,
      arXiv:2003.10854
 [13] P. Haim-Kislev, On the symplectic size of convex polytopes, Geom. Funct. Anal. 29 (2019), 440–463.
 [14] D. Hermann, Holomorphic curves and Hamiltonian systems in an open set with restricted contact-
      type boundary, Duke Math. 103 (2000), 335–374.
 [15] H. Hofer, E. Zehnder, A new capacity for symplectic manifolds, Analysis et cetera, Academic Press,
      Boston, MA, 1990, pp. 405–427.
 [16] K. Irie, Displacement energy of unit disk cotangent bundles, Math. Z. 276 (2014), 829–857.
 [17] K. Irie, Hofer-Zehnder capacity of unit disk cotangent bundles and the loop product, J. Eur. Math.
      Soc. (JEMS) 16 (2014), 2477–2497.
 [18] K. Irie, Symplectic homology of disk cotangent bundles of domains in Euclidean spaces, J. Sym-
      plectic Geom. 12 (2014), 511–552.
 [19] Y. Long, Index theory for symplectic paths with applications, Progr. Math, vol.207, Birkhäuser,
      Basel, 2002.
 [20] Y. Ostrover, When symplectic topology meets Banach space geometry, Proceedings of the Interna-
      tional Congress of Mathematicians–Seoul 2014. Vol. II, 959–981, Kyung Moon Sa, Seoul, 2014.
 [21] F. Schlenk, comments in the conference “Interactions of symplectic topology and dynamics”, Cor-
      tona, Italy, June 2019.
 [22] C. Viterbo, Capacité symplectiques et applications, Séminaire Bourbaki, Vol.1988/98, Astérisque,
      177–178 (1989), 345–362.
 [23] C. Viterbo, Functors and computations in Floer homology with applications, I, Geom. Funct. Anal.
      9 (1999), 985–1033.
 [24] J. Weber, Perturbed closed geodesics are periodic orbits: Index and transversality, Math. Z. 241
      (2002), 45–81.


  Research Institute for Mathematical Sciences, Kyoto University, Kyoto 606-8502,
JAPAN

  Email address: iriek@kurims.kyoto-u.ac.jp




                                                  40
