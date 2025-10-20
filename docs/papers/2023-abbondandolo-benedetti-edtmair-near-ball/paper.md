---
source: arXiv:2312.07363
fetched: 2025-10-20
---
# Symplectic capacities of domains close to the ball and Banach–Mazur geodesics

                                           Symplectic capacities of domains close to the ball and
arXiv:2312.07363v1 [math.SG] 12 Dec 2023




                                           Banach–Mazur geodesics in the space of contact forms
                                                  Alberto Abbondandolo∗, Gabriele Benedetti†, and Oliver Edtmair‡


                                                                                         Abstract
                                                      We prove that all normalized symplectic capacities coincide on smooth domains
                                                  in Cn which are C 2 -close to the Euclidean ball, whereas this fails for some smooth
                                                  domains which are just C 1 -close to the ball. We also prove that all symplectic capac-
                                                  ities whose value on ellipsoids agrees with that of the n-th Ekeland–Hofer capacity
                                                  coincide in a C 2 -neighborhood of the Euclidean ball of Cn . These results are deduced
                                                  from a general theorem about contact forms which are C 2 -close to Zoll ones, saying
                                                  that these contact forms can be pulled back to suitable “quasi-invariant” contact
                                                  forms. We relate all this to the question of the existence of minimizing geodesics
                                                  in the space of contact forms equipped with a Banach–Mazur pseudo-metric. Using
                                                  some new spectral invariants for contact forms, we prove the existence of minimizing
                                                  geodesics from a Zoll contact form to any contact form which is C 2 -close to it. This
                                                  paper also contains an appendix in which we review the construction of exotic ellip-
                                                  soids by the Anosov–Katok conjugation method, as these are related to the above
                                                  mentioned pseudo-metric.


                                           Introduction
                                           Normalized symplectic capacities. Let
                                                                                            n
                                                                                            X
                                                                                     ω0 =           dxj ∧ dyj
                                                                                              j=1
                                                                                          n
                                           be the standard symplectic form on C , endowed with coordinates zj = xj + iyj , for
                                           j = 1, . . . , n. In this paper, by a symplectic capacity on (Cn , ω0 ) we mean a function
                                                                           c : {open subsets of Cn } → [0, +∞]
                                           satisfying the following conditions:
                                              ∗
                                                 Fakultät für Mathematik, Ruhr-Universität Bochum, Universitätsstraße 150, 44801 Bochum, Germany,
                                           alberto.abbondandolo@rub.de
                                               †
                                                 Department of Mathematics, Vrije Universiteit Amsterdam, De Boelelaan 1111, 1081 HV Amsterdam,
                                           g.benedetti@vu.nl
                                               ‡
                                                 Department of Mathematics, University of California at Berkeley, Berkeley, CA, 94720, USA,
                                           oliver edtmair@berkeley.edu

                                                                                                1
   • (monotonicity) if there exists a symplectomorphism φ : Cn → Cn such that φ(A) ⊂
     A′ , then c(A) ≤ c(A′ );

   • (conformality) c(rA) = r 2 c(A) for every r > 0;

   • (non-triviality) c(B) > 0 and c(Z) < ∞.

Here, B denotes the unit Euclidean ball in Cn and Z the cylinder {z ∈ Cn | |z1 | < 1}. The
symplectic capacity c is said to be normalized if the non-triviality condition is upgraded
by the following requirement:

   • (normalization) c(B) = c(Z) = π.

Symplectic capacities were introduced by Ekeland and Hofer in [EH89] building on Gro-
mov’s work [Gro85]. What we here call a symplectic capacity on (Cn , ω0 ) is called by
some authors a relative or extrinsic symplectic capacity, whereas a symplectic capacity
should be intrinsic, meaning that the monotonicity axiom should hold more generally for
all symplectic embeddings of A into A′ , see [MS17, Chapter 12]. With this terminology,
any symplectic capacity is a relative symplectic capacity, but there are interesting relative
symplectic capacities, such as Hofer’s displacement energy from [Hof90], which are not
symplectic capacities. In this paper, we wish to consider also these more general functions
and hence work with the above weaker definition, but, to avoid to burden the terminology,
we omit the word “relative”.
    Gromov’s non-squeezing theorem implies that the functions

      c(A) := sup{πr 2 | ∃ symplectomorphism φ : Cn → Cn such that φ(rB) ⊂ A},
      c(A) := inf{πr 2 | ∃ symplectomorphism φ : Cn → Cn such that φ(A) ⊂ rZ},

are normalized symplectic capacities. They are known as ball capacity (or Gromov width)
and cylindrical capacity, respectively. From the above axioms, one easily sees that they
are the smallest and largest normalized capacities: any normalized symplectic capacity c
satisfies the inequalities
                                         c ≤ c ≤ c.                                  (1)
The ball and cylindrical capacities are difficult to compute, as this would involve un-
derstanding symplectic embeddings, which is precisely what one would use a symplectic
capacity for. Over the last decades, several more computable symplectic capacities based
on periodic Hamiltonian orbits, pseudo-holomorphic curves, Lagrangian submanifolds, or
a combination of these ingredients have been constructed. See the survey [CHLS07] and
references therein. Many of these symplectic capacities are spectral, meaning that the sym-
plectic capacity of a bounded domain A with smooth boundary of restricted contact type
coincides with the action of some closed characteristic on ∂A (these hypersurfaces always
admit closed characteristics, as first proven in [Vit87]).
    It is important to observe that the normalization axiom does not determine c uniquely.
Indeed, different normalized symplectic capacities may have different values on domains

                                             2
belonging to the set

   A := {bounded open neighborhoods of the origin in Cn whose boundary is smooth
         and transverse to the radial direction}.

The first examples are due to Hermann, who in [Her98b] constructed Reinhardt domains
in A with arbitrarily small volume, and hence arbitrarily small ball capacity, but large
displacement energy, and hence large cylindrical capacity. Examples of domains in A
whose cylindrical capacity is strictly larger than the displacement energy are constructed
in [Ham02]. Examples of domains whose ball capacity does not coincide with any spectral
capacity are given by the domains in A with fixed volume and arbitrarily large systole
which are constructed in [ABHS18] and [Sağ21]. Here, the systole of A ∈ A is the positive
number
                  sys(A) := min{action of closed characteristics on ∂A}.
   A long standing open question is whether all normalized symplectic capacities coincide
on bounded convex domains, see e.g. [Her98b, Conjecture 1.9] and [MS17, Section 14.9,
Problem 53]. A positive answer to this question would imply that for any normalized
symplectic capacity and any bounded convex domain C the inequality

                                    c(C)n ≤ n! vol(C)                                  (2)

holds, as the above inequality trivially holds for the ball capacity. Inequality (2) was
conjectured by Viterbo in [Vit00] and, for this reason, the coincidence of all normalized
symplectic capacities on bounded convex domains is in the recent literature referred to as
the strong Viterbo conjecture. Note that the equality holds in (2) when C is symplec-
tomorphic to a ball, and Viterbo actually conjectured that these are the only bounded
convex domains for which the equality holds (this part of Viterbo’s conjecture would not
follow from the coincidence of all normalized capacities on bounded convex domains). See
[Ost14] and [GHR22] for more information about these conjectures and their consequences
in convex geometry.
    Evidence for a positive answer to the strong Viterbo conjecture is given by the fact
that for many normalized symplectic capacities c it has been shown that

                                     c(C) = sys(C)                                     (3)

whenever C is a bounded convex domain with smooth boundary. Identity (3) has been
proven in [Vit89] for the Ekeland–Hofer capacity from [EH89], for the Hofer–Zehnder capac-
ity in [HZ90], and in [AK22, Iri22] for the capacity from symplectic homology introduced
in [FHW94], which by Hermann’s work [Her04] coincides also with Viterbo’s generating
functions capacity from [Vit92].
    Thanks to (1), proving the equivalence of all normalized capacities on bounded convex
domains amounts to prove that c = c for these domains, for instance by showing that both
these capacities satisfy (3). For an arbitrary bounded convex domain C, it is not known
whether (3) holds for c or for c.

                                            3
   In this paper, we restrict our analysis to smooth convex domains that are C k -close to
the ball B. More precisely, we observe that the domains in A are precisely the sets of the
form
                          Af := {rz | z ∈ S 2n−1 , 0 ≤ r < f (z)},
where f is an arbitrary smooth positive function on the unit sphere S 2n−1 of Cn , and
define the C k -distance between domains in A as the C k -distance between the corresponding
smooth functions on S 2n−1 . With the notation above, B = A1 .
    Our first main result is the following theorem.

Theorem 1. There exists a C 2 -neighborhood B of B in A such that for every A in B the
following facts hold:

 (i) There is a symplectomorphism φ : Cn → Cn mapping A into the cylinder whose
     systole coincides with the systole of A. In particular, c(A) = sys(A).

 (ii) There is a symplectomorphism φ : Cn → Cn mapping the ball whose systole coincides
      with the systole of A into A. In particular, c(A) = sys(A).

Consequently, all normalized symplectic capacities coincide on B.

    Statement (i) is an improvement of Proposition 4 from [AB23], where the analogous
result is proven for domains that are C 3 -close to the ball. The proof of (i) (as that of
[AB23, Proposition 4]) is elementary. It uses generating functions together with ideas from
[Edt22a] for constructing symplectic embeddings starting with suitable Hamiltonian dif-
feomorpisms. In [Edt22a], one of us showed that in dimension four the equality c = sys
holds also under non-perturbative assumptions: indeed, this equality holds whenever the
boundary of the smooth bounded convex domain C ⊂ C2 admits a closed characteristic of
minimal action which is unknotted as a closed curve on ∂C ∼  = S 3 and has self-linking num-
ber -1. The latter condition holds for domains that are C 2 -close to B, but it may actually
hold for any A in A which is convex, see the open question concluding the introduction in
[HWZ98].
    Statement (ii) is more subtle. In dimension four, [Edt22a] proves the equality c = sys
on a C 3 -neighborhood of the ball using global surfaces of section for the characteristic
foliation of ∂A. The novelty here is that we shall prove this in all dimensions, while at the
same time improving the closeness assumption from C 3 to C 2 . Note that domains that are
sufficiently C 2 -close to the ball are convex, whereas any C 1 -neighborhood of B contains
non-convex domains. This is consistent with the following statement which implies that
the C 2 -closeness assumption in Theorem 1 (ii) is optimal.

Proposition 1. There exists a smooth family of domains {Aλ }λ∈(0,λ0 ] in A which C 1 -
converges to the ball B for λ → 0, and such that

                        c(Aλ ) < sys(Aλ ) ≤ c(Aλ )     ∀λ ∈ (0, λ0 ].                    (4)



                                             4
    Here, the second inequality holds for any domain in A, just because there exist nor-
malized capacities which are spectral. So the new statement is the first strict inequality.
    In the proof of Theorem 1 (ii), we exploit the S 1 -symmetry of the ball. Recall that if the
domain A ∈ A is convex and invariant under the S 1 -action on Cn given by multiplication by
unitary complex numbers, then the equivalence of all normalized symplectic capacities for
A follows from a short argument of Ostrover: by the S 1 -invariance, the largest ball centered
at the origin and contained in A touches ∂A along a closed characteristic, proving that
c(A) ≥ sys(A). By convexity, A is contained in the cylinder over the disk spanned by this
closed characteristic, proving that c(A) ≤ sys(A), and the equivalence of all normalized
symplectic capacities for A follows from (1). Under a stronger symmetry assumption,
namely the invariance under the n-torus action given by multiplication of each coordinate
in Cn by unitary complex numbers, the equivalence of all normalized symplectic capacities
holds also by replacing convexity by a weaker monotonicity property (see [GHR22] for the
4-dimensional case and [CGH23] for the general case).
    We will prove Theorem 1 (ii) by showing that any A ∈ A which is C 2 -close to B
is symplectomorphic to a domain A′ which, albeit not being S 1 -invariant, is such that
the largest ball centered at the origin and contained in A′ touches ∂A′ along a closed
characteristic. This fact will be deduced from a general theorem on “quasi-invariant”
contact forms, which is described further down in this introduction. The same theorem
implies that the smallest ball centered at the origin and containing A′ touches ∂A′ along
a closed characteristic. The latter fact has a consequence for higher symplectic capacities,
which we now describe.

Higher symplectic capacities. In [EH90], Ekeland and Hofer used the S 1 -invariance
of the Hamiltonian action functional in order to define a sequence of symplectic capacities,
only the first of which is normalized (it is the already mentioned Ekeland–Hofer capac-
ity from [EH89]). In [GH18], Gutt and Hutchings introduced a sequence of symplectic
capacities using S 1 -equivariant symplectic homology, the first of which is the already men-
tioned normalized symplectic capacity from [FHW94]. Conjecturally, these two sequences
of symplectic capacities coincide.
    Let bck be either the k-th Ekeland–Hofer capacity or Gutt–Hutchings capacity. The
considerations below hold for both of them, as these symplectic capacities do coincide on
ellipsoids. Indeed, any ellipsoid in Cn is linearly symplectic equivalent to an ellipsoid of
the form                                n                        Pn π              o
                  E(a1 , . . . , an ) := (z1 , . . . , zn ) ∈ Cn        |z
                                                                  j=1 aj j | 2
                                                                               < 1  ,
where the aj are positive numbers, and

      b
      ck (E(a1 , . . . , an )) = k-th element in the sequence obtained by ordering the list
                                     
                                 haj h∈N,j=1,...,n in increasing order, allowing repetitions,

as proven in [EH90, Proposition 4] and [GH18, Example 1.8]. Building on this, we shall
say that a symplectic capacity c is k-normalized if it satisfies the following condition:

                                                5
   • (k-th normalization) c(E) = b
                                 ck (E) for every ellipsoid E.

   Normalized symplectic capacities, as defined at the beginning of this introduction, are
1-normalized. The k-th Ekeland–Hofer and Gutt–Hutchings symplectic capacities are k-
normalized. The functions
 ck (A) := sup{bck (E) | E ellipsoid, ∃φ : Cn → Cn symplectomorphisms with φ(E) ⊂ A},
               ck (E) | E ellipsoid, ∃φ : Cn → Cn symplectomorphisms with φ(A) ⊂ E},
 ck (A) := inf{b

are k-normalized capacities, and any k-normalized capacity c satisfies

                                            ck ≤ c ≤ ck .

The coincidence of all k-normalized capacities for the bounded open set A ⊂ Cn is equiv-
alent to the equality ck (A) = ck (A) and hence to the fact that A contains the symplectic
image of an ellipsoid E and can be symplectically embedded into an ellipsoid E ′ such that
               ck (E ′ ) − b
the difference b           ck (E) is arbitrarily small. Our next result concerns the case k = n,
where 2n is the dimension of the symplectic vector space we are considering.

Theorem 2. There exists a C 2 -neighborhood B of B in A such that for any A in B and
any n-normalized symplectic capacity c the following facts hold:

 (i) There exists an ellipsoid E with c(E) = c(A) and a symplectomorphism φ : Cn → Cn
     such that φ(E) ⊂ A.

 (ii) There exists a symplectomorphism ψ : Cn → Cn mapping A into the ball B ′ such that
      c(B ′ ) = c(A).

In particular, all n-normalized symplectic capacities coincide on B.

    It is natural to ask whether all k-normalized capacities coincide on bounded convex
domains of Cn . As we shall see, the answer is negative for most pairs (n, k), but besides
the pairs (n, 1) (corresponding to the strong Viterbo conjecture) there are other pairs for
which we do not know the answer. The next result shows that all 2-normalized capacities
of polydiscs in C2 coincide and that this is not true anymore for k ≥ 3.

Proposition 2. Let P (a, b) be the following four-dimensional polydisk:

                     P (a, b) := {(z1 , z2 ) ∈ C2 | πa |z1 |2 < 1,   π
                                                                       |z |2
                                                                     b 2
                                                                               < 1},

where a, b > 0. Then:

 (i) All 2-normalized symplectic capacities coincide on P (a, b): if c is such a capacity,
     then
                                  c(P (a, b)) = 2 min{a, b}.



                                                  6
 (ii) For every k ∈ N we have

                           ck (P (1, 1)) = b
                                           ck (E(1, 2)) ≤ k = ck (P (1, 1)),

     and the inequality is strict if and only if k ≥ 3.

   More generally, Gutt and Ramos have very recently proved that all 2-normalized ca-
pacities coincide on convex toric domains in C2 , see [GR23]. They also showed that when
k ≥ max{n, 3} the following strict inequality holds

                            ck (P (1, . . . , 1)) < b
                                                    ck (P (1, . . . , 1)) = k,

where
                     P (1, . . . , 1) := {z ∈ Cn | π|zj |2 < 1 ∀j = 1, . . . , n}
is the equilateral polydisc in Cn . Therefore, for these pairs (n, k) the k-normalized capaci-
ties do not coincide on bounded convex domains. In particular, the open C 2 -neighborhood
B of Theorem 2 does not contain all (smooth bounded) convex domains.
    These results suggest that the case k = n = 2 may be somehow special. As far as we
know, it is indeed possible that all 2-normalized capacities coincide on bounded convex
domains of C2 . Notice that this would imply the following inequality

                                         c2 (A)2 ≤ 4 vol(A)
                                         b                                                 (5)

since the ellipsoid E(1, 2) maximizes bc2 among all ellipsoids with the same volume. Some
evidence for the validity of (5) comes from a recent paper of Baracco, Bernardi, Lange and
Mazzucchelli: in [BBLM23], the authors studied the Ekeland–Hofer capacities on smooth
starshaped domains in C2 and for each k identified the local maximizers of the ratio

                                                ck (A)2
                                                b
                                                vol(A)

with respect to the C 3 -topology on A. In particular, their analysis shows that the inequality
(5) holds for every A in a C 3 -neighborhood of E(1, 2), with equality if and only if A is
symplectomorphic to a rescaled copy of the ellipsoid E(1, 2).
   The proofs of Theorems 1 and 2 are based on a general result about contact forms
which are close to Zoll ones, which might have independent interest and is described in the
next part of this introduction.

Quasi invariant contact forms. Let M be a (2n − 1)-dimensional closed manifold,
n ≥ 1, and α a contact form on M, that is, a smooth 1-form such that α ∧ dαn−1 is
nowhere vanishing. The corresponding Reeb vector field Rα is defined by the identities

                                    ıRα dα = 0,          ıRα α = 1.


                                                    7
The flow of Rα is called the Reeb flow of α. The set of periods of the closed orbits of
the Reeb flow of α is closed in R and we denote by sys(α) > 0 its minimum (setting
sys(α) := +∞ if Rα has no closed orbits, a possibility which never occurs if the Weinstein
conjecture holds true). If A belongs to the set A defined above, then the standard primitive
                                            n
                                         1X
                                 λ0 :=         (xj dyj − yj dxj )                            (6)
                                         2 j=1

of ω0 restricts to a contact form on ∂A, the Reeb vector field of λ0 |∂A spans the characteristic
distribution of ∂A, and we have the identity

                                     sys(A) = sys(λ0 |∂A ),

which justifies the choice of a common notation.
   The contact form α is said to be Zoll if all the orbits of its Reeb flow are closed and
have the same minimal period, that is, if the Reeb flow of α defines a free S 1 -action on M.
We shall deduce Theorems 1 and 2 from the following theorem.

Theorem 3. Let α0 be a Zoll contact form on the closed manifold M. Then for every
ǫ > 0 there is a δ > 0 such that, for any contact form α on M with kα − α0 kC 2 < δ, there
exists a diffeomorphism ϕ : M → M with the property that

                                         ϕ∗ α = T eh α0 ,                                    (7)

where:

(a) T is a smooth positive function on M invariant under the free S 1 -action defined by the
    Reeb flow of α0 ;

(b) h is a smooth real function on M such that h and dh vanish on the critical set of T ;

(c) min T eh = min T and max T eh = max T ;
     M           M           M            M

(d) any closed orbit of Rα is either long, meaning that its minimal period is larger than 1ǫ , or
    short, meaning that its minimal period belongs to the interval (sys(α0 ) − ǫ, sys(α0 ) + ǫ);

(e) the short closed orbits of Rα are precisely the images via ϕ of those orbits γ of the
    free S 1 -action defined by the Reeb flow of α0 consisting of critical points of T , and the
    minimal period of such an orbit is sys(α0 )T (γ).

Moreover, the map α 7→ (ϕ, T, h) is smooth and maps α0 to (id, 1, 0).

     The smoothness mentioned at the end of the statement is meant in the diffeological
sense: if {αt }t∈Rk is a smooth family of contact forms having C 2 -distance less than δ from
α0 , then the diffeomorphism ϕt and the functions Tt and ht which are given by the above
theorem depend smoothly on t ∈ Rk . This immediately implies that if ker α = ker α0 then

                                                8
the diffeomorphism ϕ belongs to the identity component in the contactomorphism group
of (M, ker α0 ).
    By statement (e), the S 1 -invariant function T in the above theorem gives us a finite di-
mensional variational principle for detecting the short closed Reeb orbits of α and, together
with (c), implies the identity

                          sys(α) = sys(α0 ) min T = sys(α0 ) min T eh .
                                               M                M

An immediate consequence of Theorem 3 is then the inequality

             sys(α)n                 n     minM T n enh              sys(α0 )n
                          = sys(α0 )                          ≤                     ,         (8)
                                          T n enh α0 ∧ dα0n−1   vol(M, α0 ∧ dα0n−1)
                                       ´
        vol(M, α ∧ dαn−1)               M

which gives us the following corollary.

Corollary 1. Zoll contact forms on the closed manifold M are C 2 -local maximizers of
the systolic ratio
                                            sys(α)n
                           ρsys (α) :=                    ,
                                       vol(M, α ∧ dαn−1 )
and if α is C 2 -close to the Zoll contact form α0 and satisfies ρsys (α) = ρsys (α0 ), then there
is a diffeomorphism ϕ : M → M such that ϕ∗ α = c α0 for some constant c > 0.

    The latter assertion follows from the fact that if the equality holds in (8) then the
function T is constant and hence by Theorem 3 (b) and (7) the diffeomorphism ϕ has the
required property. Under a stronger closeness assumption, this systolic inequality is proven
in [AB23] using arguments which will also be used in the proof of Theorem 3.
    A modification of the proof of Proposition 1 shows that the C 2 -closeness assumption is
sharp in Corollary 1.

Proposition 3. For every Zoll contact form α0 there exists a smooth family of contact
forms {αλ }λ∈(0,λ0 ] which C 1 -converges to α0 for λ → 0, and satisfies

                             ρsys (αλ ) > ρsys (α0 )   ∀λ ∈ (0, λ0 ].

    In this paper, our interest in Theorem 3 is that it allows us to construct symplectic
embedding of balls and into balls, thus proving Theorems 1 and 2.
    Contact forms of the form T eh α0 with α0 Zoll and T , h satisfying the conditions (a)-(e)
of Theorem 3 might be called “quasi-invariant”, the “quasi” referring to the presence of
the non-S 1 -invariant function eh . Note the similarity with the notion of quasi-autonomous
Hamiltonian: a compactly supported time-dependent Hamiltonian H = H(t, x) on a sym-
plectic manifold (M, ω) is said to be quasi-autonomous if the sets of maximizers and mini-
mizers of the functions H(t, ·) do not depend on t. As shown in [BP94] for (Cn , ω0 ) and in
[LM95] for closed symplectic manifolds, any compactly supported Hamiltonian diffeomor-
phism which is C 1 -close to the identity is generated by a quasi-autonomous Hamiltonian.

                                                   9
The proof of this fact is based on the Weinstein tubular neighborhood theorem and on a
version of the Hamilton–Jacobi equation.
    Theorem 3 seems much more subtle. Our proof is based on a normal form for contact
forms which are closed to Zoll ones (proved by two of us in [AB23]), and uses averaging
techniques and Moser’s homotopy argument.
    Paths in the Hamiltonian group which are generated by quasi-autonomous Hamiltonians
are minimizing geodesics in the Hofer metric, see again [BP94] and [LM95]. Results of the
same flavour hold also for a Banach–Mazur pseudo-metric on the space of contact forms
on a given contact manifold. In the next subsection, we discuss these results.

A contact Banach–Mazur pseudo-metric. Let ξ be a co-oriented contact structure
on the closed manifold M. We denote by F (M, ξ) the set of contact forms α defining ξ,
meaning that ker α = ξ and α is positive on tangent vectors which are positively transverse
to ξ. We denote by Cont0 (M, ξ) the identity component of the contactomorphism group of
(M, ξ), i.e. the set of all diffeomorphisms of M which are smoothly isotopic to the identity
by a path of diffeomorphisms whose differential maps ξ to itself. The following function
         d(α, β) := inf{max f − min f | ∃ϕ ∈ Cont0 (M, ξ) such that ϕ∗ β = ef α}                (9)
is readily seen to be a pseudo-metric on F (M, ξ). The group G := R × Cont0 (M, ξ) acts
on F (M, ξ) by
                                   (s, ϕ) · α 7→ es ϕ∗ α,
and the pseudo-metric d is clearly invariant under this action, meaning that
  d(es α, et β) = d(α, β),    ∀s, t ∈ R,      d(ϕ∗ α, ψ ∗ β) = d(α, β),     ∀ϕ, ψ ∈ Cont0 (M, ξ),
for every α, β ∈ F (M, ξ). In particular, the pseudo-distance of any two elements of the
same G-orbit is zero. More precisely, d measures the minimal distance between the G-
orbits of two contact forms in F (M, ξ) in the C 0 -topology and can be therefore considered
as a contact analogue of the Banach–Mazur pseudo-metric on the set of convex bodies in
Rn (there, the action is given by the group of affine transformations).
Remark 1. Banach–Mazur pseudo-metrics on contact forms were introduced by Rosen
and Zhang in [RZ21], where they considered the slightly different pseudo-metric
            d′ (α, β) := inf{max |f | | ∃ϕ ∈ Cont0 (M, ξ) such that ϕ∗ β = ef α}.
This pseudo-metric is invariant only under the action of Cont0 (M, ξ) and satisfies
         d′ (es α, et α) = |t − s|,    d′ (es α, et β) ≤ |t − s| + d′ (α, β),    ∀s, t ∈ R.
Here, we prefer to work with the pseudo-metric d which is invariant also under rescalings,
but all the results below hold for d′ as well. A different pseudo-metric on F (M, ξ) is
studied in Melistas’ PhD thesis [Mel21]. Banach–Mazur pseudo-metrics on contact forms
are strictly related to symplectically invariant Banach–Mazur pseudo-metrics on domains,
whose definition was suggested by Ostrover and Polterovich and which are studied in
[PRSZ20, SZ21, Ush22].

                                                 10
     By invariance, d descends to the quotient of F (M, ξ) by G. This quotient pseudo-metric
need not be a genuine metric either. Indeed, by construction d(α, β) vanishes if and only
if β belongs to the C 0 -closure of the G-orbit of α. Therefore, the element G · α in the orbit
space F (M, ξ)/G has positive pseudo-distance from all the other elements if and only if
G · α is C 0 -closed in F (M, ξ). The next examples show that some G-orbits are indeed
C 0 -closed, but other ones are not even C ∞ -closed.
     In both examples, we consider the standard tight contact structure ξst on S 3 which
is defined by the restriction of the one-form λ0 to S 3 ⊂ C2 . For every pair of positive
numbers a, b we denote by εa,b ∈ F (S 3 , ξst ) the contact form which is given by the pull-
back of the restriction of λ0 to the boundary of the ellipsoid E(a, b) by the radial projection
S 3 → ∂E(a, b).

Example 1. If ab is rational then the G-orbit of εa,b is C 0 -closed in F (S 3 , ξst). Here is
a proof based on Hutchings’ embedded contact homology (ECH), for which we refer to
[Hut14]. Let (sj , ϕj ) be a sequence in G such that esj ϕ∗j εa,b C 0 -converges to some contact
form α in F (S 3, ξst ). Then the total volume of this sequence of contact forms converges
to the total volume of α. This implies that the sequence sj converges to a real number s,
and hence ϕ∗j εa,b converges to the contact form β := e−s α. Denote by σk : F (S 3 , ξst ) → R
the ECH-spectral invariant which is associated with the generator of degree 2k of the
embedded contact homology of (S 3 , ξst ). Since the ECH-spectral invariants are invariant
under contactomorphisms and C 0 -continuous, we have σk (β) = σk (εa,b ) for every k. As
shown by Cristofaro-Gardiner and Mazzucchelli in [CGM20, Lemma 3.1], any contact form
β in F (S 3 , ξst) having the same ECH-spectral invariants of εa,b with ab rational is equivalent
to εa,b : there exists ψ ∈ Cont0 (S 3 , ξst) such that β = ψ ∗ εa,b (see also [MR23]). Therefore,
α = es ψ ∗ εa,b belongs to the G-orbit of εa,b , which is then C 0 -closed.

Example 2. If ab is irrational and not Diophantine then the G-orbit of εa,b is not C ∞ -
closed in F (S 3 , ξst ). Indeed, by the Anosov–Katok conjugation method from [AK70b] and
[Kat73] one can construct a sequence (ϕj ) in Cont0 (S 3 , ξst ) such that (ϕ∗j εa,b ) C ∞ -converges
to a contact form α ∈ F (S 3 , ξst ) whose Reeb flow has a dense orbit. See Theorem A.1 and
Remark A.2 in the appendix below. The contact form α cannot belong to the G-orbit of
εa,b because the Reeb flow of the latter contact form does not have dense orbits.
     Note however that the domain in C2 which corresponds to α in the correspondence
between domains in A and elements of F (S 3 , ξst) is symplectomorphic to the open ellipsoid
E(a, b), see Theorem A.3 in the appendix below. This is a case of “invisible symplectic
boundaries”, the first examples of which were found by Eliashberg and Hofer in [EH96].
     If ab is Diophantine we do not know whether the G-orbit of εa,b is closed in some C k -
topology. Nor we know whether there are domains in A which are symplectomorphic to
the open ellipsoid E(a, b) but such that the restriction of λ0 to ∂A is not conjugate to
εa,b . These questions are related to an open question of Hermann about area-preserving
diffeomorphisms of the annulus (see [Her98a, Question 3.2]), whose reformulation for flows
is the following: Is a Reeb flow on (S 3 , ξst ) with only two closed orbits having periods with
Diophantine ratio smoothly conjugated to the Reeb flow of an irrational ellipsoid?


                                                 11
    It is interesting to investigate the situation in which the infimum defining d(α, β) is
achieved, as this fact implies the existence of common invariant probability measures and
of minimizing geodesics connecting α and β in F (M, ξ). In this context, a continuous curve
γ : [0, 1] → F (M, ξ) is said to be a minimizing geodesic if its length
                          nX
                           k                                                                 o
       Length(γ) := sup             d(γ(tj−1), γ(tj )) k ∈ N, 0 = t0 < t1 < · · · < tk = 1
                              j=1

coincides with d(γ(0), γ(1)). The precise result is as follows.
Theorem 4. Let α, β ∈ F (M, ξ) and assume that
                                       d(α, β) = max f − min f
                                                  M         M

for some f ∈ C ∞ (M) and ϕ ∈ Cont0 (M, ξ) such that ϕ∗ β = ef α. Then the following facts
hold:
 (i) There are probability measures which are supported in the level sets f −1 (min f ) and
     f −1 (max f ) and are invariant with respect to the Reeb flows of both α and ϕ∗ β.
 (ii) For every path of contactomorphisms {ψt }t∈[0,1] ⊂ Cont0 (M, ξ) such that ψ0 = id and
      ψ1 = ϕ−1 , the path
                                 γ(t) := ψt∗ (etf α), t ∈ [0, 1],
     is a minimizing geodesic with γ(0) = α and γ(1) = β.
    In particular, if the contact form α in the above proposition is Zoll, then (i) implies that
the Reeb flow of β has a closed orbit which is the image by ϕ of a closed orbit of the Reeb
flow of α and whose β-action is emin f times its α-action. As an immediate consequence,
we obtain the following statement: If α ∈ F (M, ξ) is a Zoll contact form then any β in
F (M, ξ) such that the infimum in the definition of d(α, β) is achieved satisfies the contact
systolic inequality ρsys (β) ≤ ρsys (α).
    The fact that any co-oriented contact structure ξ admits defining contact forms with
arbitrarily large systolic ratio (see [Sağ21]) shows that the infimum in the definition of
d(α, β) may not be achieved. Actually, Example 3 shows that d(α, β) may not be achieved
for contact forms β which are C 1 -close to a Zoll contact form α. It is however achieved
when the contact form β is C 2 -close enough to the Zoll contact form α. Indeed, our last
result states the following,
Theorem 5. Let α0 be a Zoll contact form defining a co-oriented contact structure ξ on
the closed manifold M. If the contact form α ∈ F (M, ξ) is C 2 -close enough to α0 then
there exist f ∈ C ∞ (M) and ϕ ∈ Cont0 (M, ξ) such that
                                                                          Tmax (α)
               ϕ∗ α = ef α0     and d(α0 , α) = max f − min f = log                ,
                                                       M        M         Tmin (α)
where Tmax (α) and Tmin (α) = sys(α) are the maximum and minimum periods of the short
closed orbits of α, in the sense of Theorem 3 (d). In particular, there are minimizing
geodesics connecting α0 to α.

                                                  12
    This result will be deduced from Theorem 3 using a sequence of “elementary” spectral
invariants for elements of F (M, ξ), whose definition mimics analogous spectral invariants
for domains and Hamiltonian diffeomorphisms which were introduced by McDuff and Siegel
in [MS23], Hutchings in [Hut22a, Hut22b] and one of us in [Edt22b].

Organization of the paper. In Section 1, we show how suitable compactly supported
Hamiltonian diffeomorphisms of the unit ball in Cn−1 can be lifted as characteristic flows
on the boundary of certain domains in Cn . We use this construction in Section 2 to prove
Propositions 1 and 3, and in Section 3 to prove part (i) of Theorem 1. In Section 4, we
recall and complement the main result from [AB23], which is then used in Section 5 to
prove Theorem 3. Building on this theorem, we prove statement (ii) of Theorem 1 in
Section 6. The proof of Theorem 2 is similar to the proof of Theorem 1 and is sketched in
Section 7. Proposition 2 is proved in Section 8 and Theorem 4 is proved in Section 9. In
Section 10, we introduce the spectral invariants which are mentioned above. In Section 11,
we prove Theorem 5. Appendix A contains a discussion of the results which are mentioned
in Example 2.

Acknowledgments. We are grateful to Jean Gutt and Vinicius Ramos for sharing with
us the preliminary version of their work [GR23] on k-normalized capacities.
    A.A. is partially supported by the DFG under the Collaborative Research Center
SFB/TRR 191 - 281071066 (Symplectic Structures in Geometry, Algebra and Dynamics).
G.B. is partially supported by the DFG under Germany’s Excellence Strategy EXC2181/1
- 390900948 (the Heidelberg STRUCTURES Excellence Cluster). A.A. and G.B. grate-
fully acknowledge support from the Simons Center for Geometry and Physics, Stony Brook
University at which some of the research for this paper was performed during the program
Mathematical Billiards: at the Crossroads of Dynamics, Geometry, Analysis, and Mathe-
matical Physics.


1    From Hamiltonian diffeomorphisms to domains
In this section, which is inspired by analogous 4-dimensional arguments from [Edt22a], we
show how suitable compactly supported Hamiltonian diffeomorphisms of the unit ball in
Cn−1 can be lifted as characteristic flows on the boundary of certain domains in Cn . These
arguments will be used here to prove Propositions 1 and 3 in Section 2 and to prove part
(i) of Theorems 1 and 2 in Sections 3 and 7, respectively.
    Setting T := R/Z, we consider the domain

                    Ω := {(s, t, w) ∈ R × T × Cn−1 | s > π(|w|2 − 1)}

and the smooth map
                                                        q                  
                           n                     2πit          s        2
                 Φ:Ω→C ,          Φ(s, t, w) := e         1+   π
                                                                   − |w| , w ,        (1.1)

                                            13
which is a diffeomorphism onto C∗ × Cn−1 . Recall that λ0 denotes the standard primitive
(6) of the standard symplectic form ω0 of Cn . Denoting by λb0 and ω
                                                                   b0 the analogous forms
     n−1
on C , we have
                                                       b0 ,
                                  Φ∗ λ0 = (π + s) dt + λ                             (1.2)
where (s, t) denote the coordinates in R × T. By differentiating (1.2), we obtain that Φ is
a symplectomorphism from (Ω, ds ∧ dt + ω   b0 ) to (C∗ × Cn−1 , ω0 ). Since
                                                  q
                                  |Φ(s, t, w)| = 1 + πs ,                             (1.3)

this symplectomorphism maps each domain {(s, t, w) ∈ Ω | s < s0 } to a ball centered at
0 minus the linear subspace {z1 = 0}. In particular, the hypersurface {(s, t, w) ∈ Ω | s =
0} = {0} × T × B,b where Bb denotes the open unit ball in Cn−1 , is mapped onto the unit
sphere ∂B minus the linear subspace {z1 = 0}.
    Given a smooth compactly supported time-periodic Hamiltonian H : T × B     b → R such
that
                      H(t, w) > −π(1 − |w|2)                     b
                                                  ∀(t, w) ∈ T × B,                   (1.4)
we consider the subset of Cn given by
                                                     
                                       b s < H(t, w)} ∪ {z ∈ Cn | z1 = 0, |z| < 1},
        D(H) := Φ {(s, t, w) ∈ Ω | w ∈ B,

which is readily seen to be an open neighborhood of 0 diffeomorphic to a ball. The boundary
of D(H) is smooth and is given by the closure of the image of the graph of H by Φ:

                ∂D(H) = Φ(Γ(H)) = Φ(Γ(H)) ∪ {z ∈ Cn | z1 = 0, |z| = 1},

where
                                                    b | s = H(t, w)}.
                       Γ(H) := {(s, t, w) ∈ R × T × B
The domain D(H) coincides with B near the subspace {z1 = 0} and its boundary is trans-
verse to the vector field ∇|z1 | away from {z1 = 0}. Conversely, any smooth domain with
these properties has the form D(H) for a suitable compactly supported smooth function
H on T × B b satisfying (1.4). Moreover, D(H) is C k -close to B when H is C k -small and
supported uniformly away from the boundary of T × B.  b Denote by

                                      b → B,
                                φtH : B   b        φ0H = id,

the smooth path of compactly supported Hamiltonian diffeomorphisms of (B, b ω
                                                                            b0 ) which is
obtained by integrating the time-periodic Hamiltonian vector field XH given by

                           ıXHt ω
                                b0 = dHt ,     Ht (w) := H(t, w).

We recall that the Calabi invariant of ϕ := φ1H is the real number
                                          ˆ
                               Cal(ϕ) :=       H dt ∧ ωb0n−1.
                                               b
                                             T×B


                                              14
The notation is justified by the fact that the above integral does not depend on the choice
of the compactly supported Hamiltonian H defining ϕ. The Calabi invariant defines a real-
valued homomorphism on the group of compactly supported Hamiltonian diffeomorphisms
    b ω
of (B, b0 ). The action of a fixed point w of ϕ = φ1H is given by
                                   ˆ             ˆ 1
                         Aϕ (w) :=          b
                                            λ0 +     H(t, φtH (w)) dt.
                                        t
                                    t7→φH (w)
                                                     0
                                      t∈[0,1]

This quantity does not depend on the choice of the compactly supported Hamiltonian H
defining ϕ either. See [MS17, Chapter 9 and 10] for the proofs of the above mentioned
facts about the Calabi invariant and the action. The next result relates the properties of
the Hamiltonian diffeomorphism determined by H to those of the domain D(H).

Proposition 1.1. Let H : T × B     b → R be a smooth compactly supported Hamiltonian
                               1
satisfying (1.4) and set ϕ := φH . Then:
                   πn      1                          1
 (i) vol(D(H)) =      +          Cal(ϕ) = vol(B) +          Cal(ϕ).
                   n!   (n − 1)!                   (n − 1)!
 (ii) There is a one-to-one correspondence between the periodic points w of ϕ and the
      closed characteristics γ on the boundary of D(H) other than those which foliate the
      submanifold ∂D(H) ∩ {z1 = 0} = ∂B ∩ {z1 = 0}. The corresponding actions are
      related by                      ˆ
                                            λ0 = kπ + Aϕk (w),
                                        γ

     where k ∈ N denotes the minimal period of w.
(iii) Let {H λ }λ∈[0,1] be a smooth family of compactly supported Hamiltonians on T × Bb
                                             1
      satisfying (1.4) and such that ϕ := φH λ does not depend on λ. Then there exists a
      compactly supported symplectomorphism ψ : Cn → Cn such that ψ(D(H 0)) = D(H 1).
Proof. (i) The map
                          b → ∂D(H),
                     ψ :T×B                     ψ(t, w) := Φ(H(t, w), t, w),

is a diffeomorphism onto ∂D(H) \ {z1 = 0} and hence
                           ˆ         ˆ                ˆ
            n! vol(D(H)) =       n
                               ω0 =               n−1
                                           λ0 ∧ ω 0 =                  ψ ∗ (λ0 ∧ ω0n−1 ).
                             D(H)           ∂D(H)                  b
                                                                 T×B

From (1.2) we deduce
                                     b0 ,
               ψ ∗ λ0 = (π + H) dt + λ          ψ ∗ ω0 = dψ ∗ λ0 = dH ∧ dt + ω
                                                                             b0 ,

and hence

            ψ ∗ (λ0 ∧ ω0n−1) = (π + H)dt ∧ ω                        b0 ∧ ω
                                           b0n−1 + (n − 1)dH ∧ dt ∧ λ    b0n−2 .

                                                15
                b0 ∧ ω
Since dH ∧ dt ∧ λ    b0n−2 differs from H dt ∧ ω
                                               b0n−1 by an exact form, we obtain
                           ˆ                   ˆ
                                        n−1
        n! vol(D(H)) = π         dt ∧ ω
                                      b0 + n                b0n−1 = π n + n Cal(ϕ),
                                                     H dt ∧ ω
                                  b
                                T×B                          b
                                                           T×B

which implies (i).

(ii) The hypersurface Γ(H) is the zero level set of the autonomous Hamiltonian
                                      e t, w) := H(t, w) − s
                                      H(s,                                                           (1.5)

on the symplectic manifold (R × T × B,  b ds ∧ dt + ω b ), whose Hamiltonian vector field is
easily seen to be
                       XHe (s, t, w) = ∂t H(t, w) ∂s + ∂t + XHt (w).                   (1.6)
The closed characteristics of Γ(H) are the closed orbits of the flow of XHe on the energy
level {He = 0}. Every orbit of this flow intersects the hypersurface {t = 0} transversally
infinitely many times, and the orbit of (s, 0, w) is easily seen to be given by
                                                                     
             φτHe (s, 0, w) = s + H(τ, φτH (w)) − H(0, w), τ, φτH (w) ,    τ ∈ R.    (1.7)

In particular, the orbit of a point (s, 0, w) in Γ(H) is the curve
                                                           
                        γ (τ ) = H(τ, φτXH (w)), τ, φτH (w) ,
                        b                                       τ ∈ R.

This orbit is closed if and only if w is a k-periodic point of φ1X , for some k ∈ N, and in this
case              ˆ                 ˆ
                         ∗  ∗
                       γ (Φ λ0 ) =
                       b                                   b0 ) = πk + Aφk (w).
                                         γb∗ ((π + s) dt + λ               H
                     [0,k]             [0,k]

Statement (ii) follows by taking γ := Φ(b
                                        γ ).

(iii) For each (λ, t) ∈ [0, 1] × T, consider the Hamiltonian diffeomorphism
                                   b → B,
                             ψλt : B   b             ψλt := φtH λ ◦ (φtH 0 )−1 .

By assumption, we have
                                 ψλ0 = ψλ1 = idBb ,          ∀λ ∈ [0, 1].                            (1.8)
For each λ ∈ [0, 1], the loop of Hamiltonian diffeomorphisms t 7→ ψλt , t ∈ T, based at idBb
is generated by the Hamiltonian
            b → R,
   Gλ : T × B                Gλ (t, w) = H λ #H 0 (t, w) = H λ (t, w) − H 0 (t, (ψλt )−1 (w)).       (1.9)

Notice that Gλ is indeed 1-periodic in t since the same is true for ψλt . Moreover, G0 = 0.
We define the diffeomorphism
                   b → R × T × B,
     ψ̃λ : R × T × B           b               ψ̃λ (s, t, w) = (s + Gλ (t, ψλt (w)), t, ψλt (w)).   (1.10)

                                                      16
We have ψ̃0 = id and, using the formula for Gλ given in (1.9), that H    e λ ◦ ψ̃λ = H
                                                                                     e 0 , where
He λ is induced by H λ as in (1.5). In particular, ψ̃λ maps Γ(H 0 ) to Γ(H λ ).
    We now make the claim that the path λ 7→ ψ̃λ , λ ∈ [0, 1], is generated by a smooth path
of Hamiltonians F λ : R × T × B b → R which are supported in a set of the form R × T × K,
where K is a compact subset of B.  b If the claim is true, we can consider the multiplication
  λ
E := χ F of F by a cut-off function χλ supported in an arbitrarily small neighborhood
          λ λ     λ

of Γ(H λ ). Since F λ has support in R × T × K, the function E λ has compact support. Since
Φ is a symplectomorphism, the function E λ ◦ Φ−1 yields a compactly supported function of
the whole Cn . The time-one map ψ : Cn → Cn of the Hamiltonian flow of {E λ ◦ Φ−1 }λ∈[0,1]
yields the desired symplectomorphism.
    We are left with proving the claim. As a first step, consider the compactly supported
Hamiltonian function Ftλ : B b → R, which, for every t ∈ R, generates the path of compactly
supported Hamiltonian diffeomorphisms λ 7→ ψλt , λ ∈ [0, 1]. By [Ban97, Prop. 3.1.5], we
have
                                  ∂t Ftλ − ∂λ Gλt = {Ftλ , Gλt },                          (1.11)
where {Ftλ , Gλt } = dGλt [XFtλ ] = −dFtλ [XGλt ] = −{Gλt , Ftλ } is the Poisson bracket.
   We now make the subclaim that

                                      F1λ = F0λ ,        ∀λ ∈ [0, 1].                           (1.12)

If the subclaim is true, then the function
                                     b → R,
                         Fλ: R × T × B                    F λ (s, t, w) = Ftλ (w)

is well-defined and smooth, by the smoothness of t 7→ ψλt on T. This function has the
desired property. Indeed, by (1.10) and (1.11), we get
                                                          
              (∂λ ψ̃λ ) ◦ (ψ̃λ )−1 = ∂λ Gλt + {Ftλ , Gλt } ∂s + XFtλ = ∂t Ftλ ∂s + XFtλ

and
                                                  b ) = ∂t Ftλ dt + dFtλ = dF λ .
                       ı∂t Ftλ ∂s +X λ (ds ∧ dt + ω
                                 Ft


We are now left with proving the subclaim (1.12). Since Gλ has compact support and
                         b we have Aφ1 (w) = 0. Differentiating this equality with respect
φ1Gλ (w) = w for all w ∈ B,
                                     Gλ
to λ and setting
                                  b
                         γλ : T → B,    γλ (t) := φtGλ (w), ∀t ∈ T,
using the definition of the action we get
                     ˆ 1                                                     ˆ 1
                                                     λ
 0 = ∂λ Aφ1 λ (w) =        ω
                           b (∂λ γλ (t), γ̇λ(t)) + dGt (γλ(t))[∂λ γλ (t)] dt +     (∂λ Gλt )(γλ (t))dt
           G
                      0                                                         0
                     ˆ 1
                  =      (∂λ Gλt )(γλ(t))dt,                                                     (1.13)
                        0



                                                    17
where we used that γλ is a Hamiltonian trajectory for Gλ . On the other hand, by (1.11),
we get
      ˆ 1                     ˆ 1                                 ˆ 1                
               λ                       λ     λ
                                                                      d      λ
          (∂λ Gt )(γλ(t))dt =     ∂t Ft + dFt [γ̇λ ] (γλ (t))dt =           Ft (γλ (t) dt
       0                       0                                   0 dt
                                                                = F1λ (γλ (1)) − F0λ (γλ (0))
                                                                     = F1λ (w) − F0λ (w),

                                                                                           b
which, in combination with (1.13), yields 0 = F1λ (w) − F0λ (w) for all λ ∈ [0, 1] and w ∈ B,
as stated in the subclaim (1.12).
                                                                br is contained in Ω and
Remark 1.2. Let ǫ > 0 and r ∈ (0, 1) be such that (−ǫ, ǫ) × T × B
set
                             V := Φ((−ǫ, ǫ) × T × B br ).

The above proof shows that if H λ − H 0 is supported in T × B   br and |H λ| < ǫ for every
λ ∈ [0, 1], then the symplectomorphism ψ in (iii) can be chosen to be supported in V using
suitable cut-off functions χλ .


2    Proof of Proposition 1 and Proposition 3
In this section we show how the first two statements of Proposition 1.1 can be used to
prove Proposition 1 from the Introduction. The proof of Proposition 3 goes along similar
lines and is sketched in Remark 2.1 at the end of this section.
    We consider a compactly supported smooth Hamiltonian H : T × B    b → R such that
all the fixed points of φH have non-negative action while the Calabi invariant of φ1H is
                          1

negative. Following [ABHS18], a Hamiltonian with these properties can be constructed in
the following way. Consider an autonomous radial Hamiltonian on B b of the form

                                     F (w) := f (|w|2),

where f : [0, 1) → R is a smooth compactly supported monotonically decreasing function
such that f ′ (r 2 ) = − π2 for every r ∈ [0, 12 ]. The identity
                                                     ′ (|w|2 )
                                   φ1F (w) = e−2if               w

                                                                      b contained in the
shows that φ1F (w) = −w if |w| ≤ 21 . Next, consider an open ball U ⊂ B
               1                                        1
ball of radius 2 centered at the origin and such that φF (U) = −U does not intersect U.
Let G be a smooth function on B  b with support in U and such that
                                       ˆ
                                  1
                            Cal(φG ) =    Gω b n−1 < −Cal(φ1F ).                   (2.1)
                                         b
                                         B




                                             18
The Hamiltonian diffeomorphism φ1F ◦ φ1G is generated by the compactly supported Hamil-
tonian
                  H(t, w) = (F #G)(t, w) = F (w) + G(t, (φtF )−1 (w)).
The fact that φ1F displaces the support of G from itself implies that the only fixed points
of φ1H are fixed points of φ1F , and if w is such a fixed point we have

                      Aφ1H (w) = Aφ1F (w) = f (|w|2) − |w|2f ′ (|w|2 ) ≥ 0.

Finally, (2.1) implies that the Calabi invariant of φ1H is negative. This shows that H has
the required properties. The k-periodic points of φ1H may have negative action, but by the
definition of action we have a bound of the form

                                  AφkH (w) ≤ ck         ∀k ∈ N,                           (2.2)

for every k-periodic point w of φ1H , for a suitable number c.
                                                                         b The
    We now see H as a smooth Hamiltonian on T × Cn−1 with support in T × B.
family of rescaled Hamiltonians
                                                     
                             H λ (t, w) := λ2 H t, wλ , λ > 0,

C 1 -converges to the zero function for λ → 0. Therefore, there exists λ0 > 0 such that H λ
satisfies (1.4) and Aλ := D(H λ ) belongs to A if λ ∈ (0, λ0 ], and Aλ C 1 -converges to B for
λ → 0. The identity                                        
                                   XH λ (t, w) = λXH t, wλ
implies that the conformally symplectic diffeomorphism w 7→ λ · w conjugates the Hamil-
tonian dynamics of H and H λ , meaning that
                                                       
                                 φtH λ (w) = λ · φtH wλ .

               b is a fixed point of φ1 λ if and only if w is a fixed point of φ1 , and in this
Therefore, w ∈ B                      H                  λ                      H
case we have                                          
                               Aφ1 λ (w) = λ2 Aφ1H wλ ≥ 0.
                                         H

If γ is the closed characteristic on ∂Aλ corresponding to a fixed point of φ1H λ , statement
(ii) in Proposition 1.1 gives us the bound
                                  ˆ
                                    λ0 = π + Aφ1 λ (w) ≥ π.
                                                    H
                                     γ


If γ is the closed characteristic on ∂Aλ corresponding to a periodic point of φ1H λ of minimal
period k ≥ 2, then using also (2.2) we obtain
                                                         
                 ˆ
                    λ0 = kπ + Aφk λ (w) = kπ + λ2 AφkH wλ ≥ kπ − λ2 ck ≥ π,
                                 H
                 γ


                                               19
                      π
provided that λ2 ≤ 2c   . Together with the fact that the characteristics of ∂Aλ in {z1 = 0}
are closed with action π, up to reducing the size of λ0 we deduce that sys(Aλ ) = π for
every λ ∈ (0, λ0 ]. On the other hand,

                                  Cal(φ1H λ ) = λ2n Cal(φ1H ) < 0,

so by statement (i) in Proposition 1.1 we obtain
                                            πn
                                 vol(Aλ ) <           ∀λ ∈ (0, λ0],
                                            n!
and hence
                  sys(Aλ )n = π n > n! vol(Aλ ) ≥ c(Aλ )n            ∀λ ∈ (0, λ0 ],
proving the strict inequality

                                c(Aλ ) < sys(Aλ )      ∀λ ∈ (0, λ0],

which is stated in Proposition 1.
Remark 2.1. The proof of Proposition 3 is similar and we just sketch it. Assume without
loss of generality that the orbits of the Zoll contact form α0 have minimal period 1. Then
we can find an embedding ϕ : T × B    br → M such that

                                                      b0 .
                                         ϕ∗ α0 = dt + λ
      br denotes the open ball of radius r in Cn−1 . We modify α0 inside the image of ϕ
Here, B
and obtain a smooth family of contact forms {αλ }λ∈(0,λ0 ] such that

                                                            b0 ,
                                    ϕ∗ αλ = (1 + H λ ) dt + λ

where {H λ} is the family of rescaled Hamiltonians constructed above and λ0 > 0 is so
small that H λ is supported in T × Bbr for all λ ∈ (0, λ0 ]. The contact forms αλ converge to
α0 in the C 1 -topology for λ → 0. By the properties of H λ , we have sys(αλ ) = 1 = sys(α0 )
but the volume of (M, αλ ∧ dαλn−1 ) is strictly smaller than that of (M, α0 ∧ dα0n−1).


3    Proof of Theorem 1 (i)
The proof of statement (i) of Theorem 1 uses the construction from Section 1 together
with the following result.
Proposition 3.1. Let m ∈ N and denote by Br the open ball of radius r centered at 0 in
Cm and by Brc its complement. For every r > 0 and ǫ > 0 there exists δ > 0 such that if
H is a compactly supported Hamiltonian on T × Cm satisfying kHkC 2 < δ and such that 0
is a fixed point of φ1H with Aφ1H (0) = 0, then there exists a smooth family of Hamiltonians
{H λ }λ∈[0,1] ⊂ C ∞ (T × Cm ) such that:

                                                 20
 (i) H 0 = H;
 (ii) H λ = H on T × Brc for every λ ∈ [0, 1];
(iii) φ1H λ = φ1H for every λ ∈ [0, 1];
(iv) kH λ kC 0 < ǫ for every λ ∈ [0, 1];
 (v) |H 1 (t, z)| ≤ ǫ|z|2 for every (t, z) ∈ T × Cm .
    The proof of this proposition is given at the end of this section. Here, we show how
Propositions 1.1 and 3.1 imply statement (i) of Theorem 1.
    Our aim is to prove that if A ∈ A is C 2 -close enough to the unit ball B, then there
exists a symplectomorphism ψ : Cn → Cn mapping A into the cylinder whose systole
coincides with the systole of A. Up to rescaling, we may assume that sys(A) = π, so
that ψ is required to map A into the standard cylinder Z = {z ∈ Cn | |z1 | < 1}. A
closed characteristic γ on ∂A achieving sys(A) is C 0 -close to some closed characteristic on
∂B. Thus, up to applying a unitary transformation of Cn and a small translation we may
assume that γ passes through the point (1, 0, . . . , 0) ∈ Cn and is C 0 -close to the curve
                            γ 0 : T → Cn ,     t 7→ (e2πit , 0, . . . , 0).
Let Φ : Ω → C∗ × Cn−1 be the symplectomorphism introduced in (1.1). Noting that
Φ(0, t, 0) = γ0 (t) and that the set
                                                  
                                                         b2
                                    U := − π2 , π2 × T × B                 (3.1)
                                                               3


is contained in Ω, we consider the open neighborhood V := Φ(U) of γ0 . Here, B br denotes
                                                     n−1
the open ball of radius r centered at the origin of C . Being close to γ0 , the curve γ is
contained in V . Moreover, the form of Φ implies the identities
                B ∩ V = Φ(U ∩ {s < 0}),             Z ∩ V = Φ(U ∩ {s < π|w|2 }).        (3.2)
Since A is C 2 -close to B, the hypersurface ∂A ∩ V is the image via Φ of the graph of a
C 2 -small smooth function on T × Bb 2 . We can extend this function to a C 2 -small smooth
                                     3

function H on T × Cn−1 having support in T × B   b and satisfying (1.4), thus obtaining

                                      A ∩ V = D(H) ∩ V,                                 (3.3)
where D(H) denotes the domain which we introduced in the previous section. As it passes
through (1, 0, . . . , 0) = Φ(0, 0, 0), the closed characteristic γ0 on ∂A ∩ V = ∂D(H) ∩ V
corresponds via Φ to the fixed point 0 of φ1H and
                                                 ˆ
                                      Aφ1H (0) = λ0 − π = 0,
                                                γ

as shown in Proposition 1.1 (ii). Assuming that A is C 2 -close enough to B - and hence H
is C 2 -small enough - we apply Proposition 3.1 with m = n − 1 to get a smooth family of
Hamiltonian functions H λ : T × Cn−1 → R, λ ∈ [0, 1], such that:

                                               21
 (i) H 0 = H;
                     b c1 for every λ ∈ [0, 1];
 (ii) H λ = H on T × B
                             3


(iii) φ1H λ = φ1H for every λ ∈ [0, 1];
                   π
(iv) kH λ kC 0 <   2
                       for every λ ∈ [0, 1];
 (v) H 1 (t, w) ≤ π|w|2 for every (t, w) ∈ T × Cn−1 .
By properties (i) and (iii), Proposition 1.1 (iii) implies that there exists a symplectomor-
phism ψ : Cn → Cn such that ψ(D(H)) = D(H 1). Thanks to properties (ii), (iv) and
Remark 1.2, we can assume that ψ is supported in V and hence (3.3) implies
                                  ψ(A) = (A \ V ) ∪ (D(H 1 ) ∩ V ).
Since the closure of B \ V is contained in Z, C 0 -closeness of A to B implies that A \ V is
contained in Z. By (v) and the second identity in (3.2), D(H 1 ) ∩ V is also contained in Z.
We conclude that ψ(A) is contained in Z, hence proving statement (i) in Theorem 1.

     There remains to prove Proposition 3.1. The proof of this proposition makes use of
some properties of generating functions which we now recall, referring to [BP94] for more
details. Every compactly supported symplectomorphism ϕ : Cm → Cm which is sufficiently
C 1 -close to the identity is represented by a unique C 2 -small, compactly supported, smooth
generating function S : Cm → R which is characterized by the equation
                                                     
                             i(z − ϕ(z)) = ∇S z+ϕ(z)
                                                  2
                                                            ∀z ∈ Cm .                    (3.4)
Conversely, every C 2 -small, compactly supported, smooth function S defines by the above
identity a compactly supported symplectomorphism ϕ : Cm → Cm which is C 1 -close to
the identity. If {ϕt }t∈[0,1] is a smooth isotopy starting at the identity and consisting of
compactly supported symplectic diffeomorphisms of Cm which are C 1 -close to the identity,
then ϕt = φtH , where the smooth family of compactly supported Hamiltonians {Ht }t∈[0,1]
is related to the smooth family {St }t∈[0,1] of compactly supported generating functions for
{ϕt }t∈[0,1] by the Hamilton–Jacobi equation
                                                     
                     ∂t St (z) = H t, z + 2i ∇St (z)   ∀(t, z) ∈ [0, 1] × Cm .         (3.5)
Let ϕ : Cm → Cm be a compactly supported symplectomorphism which is C 1 -close to the
identity and denote by S : Cm → R the corresponding generating function. Let z be a
fixed point of ϕ. By (3.4), z is a critical point of S. The path of generating functions
{tS}t∈[0,1] produces a compactly supported symplectic isotopy from the identity to ϕ such
that z is a fixed point of all maps in this isotopy. If H denotes the compactly supported
Hamiltonian on [0, 1] × Cm associated with this isotopy, (3.5) implies that H(t, z) = S(z)
for every t ∈ [0, 1] and hence
                                       Aϕ (z) = S(z).                                 (3.6)
After these preliminaries, we can prove Proposition 3.1.

                                                 22
Proof of Proposition 3.1. In this proof, whenever we say that some function is C k -small,
we understand “provided that H is C 2 -small”.
    We first prove this proposition under the additional assumption that Ht is identically
zero for every t in a neighborhood of 0 in T.
    Since H is C 2 -small, each symplectomorphism φtH is C 1 -close to the identity and hence
generated by a unique C 2 -small generating function St0 . It follows from the Hamilton–
Jacobi equation (3.5) that St0 is also C 2 -small when viewed as a function on [0, 1] × Cm . By
our additional assumption on H, φtH = id for t close to 0 and φtH = φ1H for t close enough
to 1. Therefore, St0 = 0 for t close to 0 and St0 = S10 for t close to 1.
    Fix a smooth monotonically increasing function η : [0, 1] → [0, 1] such that η(t) = 0 for
t close to 0 and η(t) = 1 for t close to 1. By modifying St0 , we can find a smooth family of
functions St1 such that:
(a) St1 = η(t)S10 inside the ball B 3r , for every t ∈ [0, 1];

(b) St1 = St0 outside the ball B 2 r , for every t ∈ [0, 1];
                                    3


(c) St1 = 0 for t close to 0 and St1 = S10 for t close to 1;

(d) St1 is C 2 small when viewed as a function on [0, 1] × Cm .
For every λ ∈ [0, 1], we define

                                        Stλ := (1 − λ)St0 + λSt1 .

By (d), this function is C 2 -small when viewed as a function on [0, 1]2 × Cm . Moreover,
Stλ = 0 for t sufficiently close to 0 and Stλ = S10 for t sufficiently close to 1. For every λ ∈
[0, 1], let H λ be the unique compactly supported Hamiltonian generating the Hamiltonian
isotopy associated with the family of generating functions {Stλ }t∈[0,1] . It follows from the
Hamilton–Jacobi equation that H λ is C 1 -small, and in particular we can ensure that (iv)
holds. Clearly, H 0 = H and by (b) Htλ agrees with Ht outside Br , for every t ∈ [0, 1], so
(i) and (ii) hold. By the second assertion in (c), we have φ1H λ = φ1H for all λ ∈ [0, 1], which
proves (iii). Since Stλ is constant in t for t close to 0 and for t close to 1, the Hamiltonian
Htλ vanishes for t near 0 and 1. Therefore, it descends to a 1-periodic Hamiltonian. It
remains to check (v). By (a), the Hamilton–Jacobi equation (3.5) implies that
                                                
                          H 1 t, z + 2i ∇St1 (z) = ∂t St1 (z) = η ′ (t)S10 (z)              (3.7)

for every z ∈ B 3r . Since 0 is a fixed point of φ1H of zero action, the function S10 and its
derivative both vanish at 0, see (3.6). Therefore, we get

                              |S10 (z)| ≤ 21 kS10 kC 2 |z|2 ,   ∀z ∈ Cm .

The fact that S10 is C 2 -small implies that the diffeomorphism

                       θt : Cm → Cm ,       θt (z) := z + 2i ∇St1 (z), ∀z ∈ Cm ,

                                                     23
is C 1 -close to the identity and fixes the origin. Thus, the same is true for the inverse
diffeomorphism, and we get

                                    |θt−1 (w)| ≤ 2|w|,         ∀w ∈ Cm .

Using (3.7), we obtain for all (t, w) ∈ T × Cm

      |H 1 (t, w)| = |η ′(t)S10 (θt−1 (w))| ≤ kηkC 1 12 kS10 kC 2 |θt−1 (w)|2 ≤ kηkC 1 21 kS10 kC 2 4|w|2,

which implies the bound (v) since S10 is C 2 -small.

   We now show how the general case can be deduced from the special one considered
above. Using again the function η whose properties are described above, we introduce the
Hamiltonian
                   b z) := η ′ (t)H(η(t), z),
                   H(t,                         ∀(t, z) ∈ [0, 1] × Cm ,
which vanishes for t close to 0 and for t close to 1, and in particular can be seen as 1-
periodic in time, is C 2 -small and satisfies φ1Hb = φ1H . We claim that there exists a family of
smooth Hamiltonians K λ : T × Cm → R, λ ∈ [0, 1], such that:

(a’) K 0 = H;
           b on T × B r ;
(b’) K 1 = H          2


(c’) K λ = H on T × Brc for every λ ∈ [0, 1];

(d’) φ1K λ = φ1H for every λ ∈ [0, 1];

(e’) kK λ kC 0 < ǫ for every λ ∈ [0, 1].

Once this is proven, the conclusion of Proposition 3.1 in the general case follows. Indeed,
by applying the above special case to the Hamiltonian H, b we obtain a smooth family of
Hamiltonians {H b }λ∈[0,1] on T × C which satisfies:
                 λ                 m


       b 0 = H;
  (i’) H     b

       bλ = H
 (ii’) H    b on T × B cr for every λ ∈ [0, 1];
                              4


(iii’) φ1Hb λ = φ1Hb for every λ ∈ [0, 1];

       b λ kC 0 < ǫ for every λ ∈ [0, 1];
(iv’) kH
       b 1 (t, z)| ≤ ǫ|z|2 for every (t, z) ∈ T × Cm .
 (v’) |H




                                                      24
For λ ∈ [0, 1], we define                   (
                                                 b λ on T × B r ,
                                                 H
                                   Lλ :=                       2

                                                 K 1 on T × B cr .
                                                                  4

This yields a smooth family of Hamiltonians {L }λ∈[0,1] on T × Cm . One easily verifies that
                                                          λ

K 1 = L0 and that the smooth concatenation
                                   η(2λ)
                               λ     K        for λ ∈ [0, 12 ],
                            H :=
                                     Lη(2λ−1) for λ ∈ [ 12 , 1],

satisfies the desired properties (i)-(v).

    There remains to construct a family of smooth Hamiltonians K λ : T × Cm → R
satisfying (a’)-(e’). For λ ∈ [0, 1], we define η λ (t) := (1 − λ)t + λη(t). Note that η λ
descends to a smooth map from T to itself. Therefore, we may define a smooth family of
Hamiltonians H  e λ : T × Cm → R, λ ∈ [0, 1], by the formula

                                 e λ (t, z) := (η λ )′ (t)H(η λ(t), z).
                                 H

Note that He 0 = H and He1 = H
                             b and that φ1 = φ1 for all λ. Choose a family of smooth
                                            Heλ   H
functions F λ : T × Cm → R, λ ∈ [0, 1], such that

                         eλ
                    Fλ = H       on T × B 7 r ,           Fλ = H         on T × B c8 r .     (3.8)
                                                10                                     10



We can choose F 0 to be equal to H everywhere because H    e 0 = H. Since H and H
                                                                                e λ are C 2 -
        λ                2
small, F can be made C -small as well. Therefore, the compactly supported symplectic
diffeomorphism
                                   ψ λ := (φ1F λ )−1 ◦ φ1H
is C 1 -close to the identity and hence has a C 2 -small compactly supported generating func-
tion S λ : Cm → R. By choosing H to be C 2 -small enough, we can ensure that
                                              r                             r
                               kXH kC 0 <    10
                                                ,         kXF λ kC 0 <     10
                                                                              ,              (3.9)

and, together with (3.8) and the fact that the time-1-map of the Hamiltonian H         e λ is equal
to φ1H , we deduce that ψ λ is the identity on B 6 r and on B c9 r . The identity (3.4) guarantees
                                                     10               10
         λ
that ∇S vanishes on these two sets and hence

                            S λ = cλ   on B 6 r ,         S λ = 0 on B c9 r ,               (3.10)
                                                10                                10


for a suitable constant cλ ∈ R which we will later show to be zero. Now let ψtλ be the
symplectomorphism which is generated by η(t)S λ . Then ψtλ depends smoothly on t, ψtλ = id
for t close to 0 and ψtλ = ψ λ for t close to 1.



                                                     25
    Let Gλ : [0, 1] × Cm → R be the compactly supported Hamiltonian associated with the
symplectic isotopy {ψtλ }t∈[0,1] . Then Gλt = 0 for t close to either 0 or 1, and hence we can
see Gλ as a smooth function on T × Cm . The Hamilton–Jacobi equation (3.5) reads
                                                                     
                             η ′ (t)S λ (z) = Gλt z + 2i η(t)∇S λ (z) .

It implies that Gλ is C 1 -small. Moreover, thanks to (3.10) it implies:

                         Gλ (t, z) = cλ η ′ (t)        ∀(t, z) ∈ T × B 6 r ,           (3.11)
                                                                          10

                             Gλ (t, z) = 0        ∀(t, z) ∈ T × B c9 r .               (3.12)
                                                                     10


We set                                                                            
                K λ (t, z) := (F λ #Gλ )(t, z) = F λ (t, z) + Gλ t, (φtF λ )−1 (z) .
This function is smooth on T × Cm , compactly supported and satisfies

                             φ1K λ = φ1F λ ◦ φ1Gλ = φ1F λ ◦ ψ λ = φ1H ,

proving (d’). By (3.8), (3.9) and (3.12), K λ satisfies (c’). By (3.8), (3.9) and (3.11), we
obtain
                                e λ (t, z) + cλ η ′ (t)
                   K λ (t, z) = H                       ∀(t, z) ∈ T × B 2r .

The Hamiltonians K λ and H   e λ induce the same time-one map, so the independence of the
action of the fixed point 0 from the choice of the Hamiltonian implies that the number cλ
in the above identity is zero, as claimed above. This shows that K 1 satisfies (b’). Property
(e’) follows from the fact that F λ and Gλ are both C 0 -small. Since F 0 = H, we have
G0 = 0 and thus K 0 = H, showing (a’). This concludes the proof of Proposition 3.1.


4    Averaging and a preliminary normal form
We denote by Ωk (M), k ≥ 0, the space of smooth k-forms on the manifold M. We assume
that M is closed, odd dimensional and endowed with a Zoll contact form α0 ∈ Ω1 (M). We
denote by Rα0 the Reeb vector field of α0 and by θt its flow. By the Zoll property, the
flow θt defines a free, smooth S 1 -action on M. Here, we identify S 1 with R/T0 Z, where
T0 = sys(α0 ) is the minimal period of the Reeb orbits of α0 . Whenever we talk about
S 1 -invariant or S 1 -equivariant objects on M, we refer to this S 1 -action.
     On the space Ωk (M) we have the averaging operator mapping each β ∈ Ωk (M) to the
  1
S -invariant k-form                            ˆ T0
                                             1
                                       β :=         θt∗ β dt.
                                             T0 0
Averaging commutes with differentiation:

                                  dβ = dβ          ∀β ∈ Ωk (M).


                                                  26
Moreover, if Y is an S 1 -invariant vector field on M (for instance Y = Rα0 ), we have

                                ıY β = ıY β     ∀β ∈ Ωk (M).

We shall also need to average linear endomorphisms F : T ∗ M → T ∗ M lifting the identity.
The averaged endomorphism F : T ∗ M → T ∗ M is defined by
                                           ˆ T0
                                         1
                                   F :=         θt∗ F dt,
                                        T0 0
where the pull-back of F by a diffeomorphism θ : M → M is given by
                                                      
         (θ∗ F )(x)[p] := dθ(x)∗ F (θ(x))[p ◦ dθ(x)−1 ] , ∀x ∈ M, ∀p ∈ Tx∗ M.

If F is as above and β ∈ Ω1 (M) is S 1 -invariant, then the 1-form F [β] satisfies

                                        F [β] = F [β].

We fix an arbitrary S 1 -invariant Riemannian metric on M. The C k -norms of tensors on
M and the covariant derivatives we shall occasionally use in our proofs are induced by this
metric.
   Our proof of Theorem 3 relies on the following normal form which is proven in [AB23,
Theorem 2].

Theorem 4.1. Let α0 be a Zoll contact form on a closed manifold M with orbits having
minimal period T0 . There is δ0 > 0 such that if α is a contact form on M satisfying
kα − α0 kC 2 < δ0 , then there exists a diffeomorphism u : M → M such that

                                    u∗ α = Sα0 + η + df,                                 (4.1)

where:

 (i) S is a smooth positive function on M that is invariant under the Reeb flow of α0 ;

 (ii) f is a smooth function on M with average zero along each orbit of Rα0 ;

(iii) η is a smooth one-form on M satisfying ιRα0 η = 0;

(iv) ιRα0 dη = F [dS] for a smooth endomorphism F : T ∗ M → T ∗ M lifting the identity;

 (v) ιRα0 df = ιZ dS for a smooth vector field Z on M taking values in the contact distri-
     bution ker α0 and having average zero along each orbit of Rα0 ;

(vi) dS = −B[V ], where V is a smooth S 1 -invariant section of ker α0 and B : ker α0 →
     (ker α0 )∗ is a smooth isomorphism lifting the identity.



                                              27
Moreover, for every integer k ≥ 0, there is a modulus of continuity ωk such that
          
      max distC k+1 (u, id), kS − 1kC k+1 , kf kC k+1 , kηkC k , kdηkC k , kF kC k , kZkC k ,
                                                                                                (4.2)
                                        kV kC k+1 , kB − B0 kC k ≤ ωk (kα − α0 kC k+2 ),

where B0 : ker α0 → (ker α0 )∗ is the isomorphism given by the non-degenerate bilinear form
dα0 |ker α0 ×ker α0 . Finally, the map α 7→ (u, S, η, f ) is smooth and maps α0 to (id, 1, 0, 0).

    Here, by modulus of continuity we mean a monotonically increasing continuous function
ω : [0, +∞) → [0, +∞) such that ω(0) = 0.

Remark 4.2. Actually, statement (vi) and the corresponding bounds for V and B in (4.2)
are not present in the formulation of [AB23, Theorem 2], but they are easily recoverable
from its proof (see in particular equation (2.25) together with (2.4) and (2.24) therein).
    The smoothness which is mentioned at the end of the theorem is not explicitly stated
in [AB23, Theorem 2] either, but can be deduced from its proof. This amounts to checking
that in [AB23, Theorem B.1] the vector fields U, V and the function h depend smoothly
on the vector field X, and take the values U = 0, V = 0, h = 1 when X = X0 . The triplet
(U, V, h) is determined by X by means of a functional equation of the form ΦX (U, V, h) = 0,
where the dependence of Φ on X is affine and Φ0 (0, 0, 1) = 0. A standard argument
involving the parametric inverse mapping theorem and [AB23, Lemma B.4] gives us the
required smooth dependence on X of the solution (U, V, h) of this equation.

    We will also need the following additional fact.

Proposition 4.3. In the setting of Theorem 4.1, the following additional properties hold:
for every ǫ > 0 there is a positive number δ ≤ δ0 such that if kα − α0 kC 2 < δ then:

(vii) any closed orbit of Rα is either long, meaning that its minimal period is larger than 1ǫ ,
      or short, meaning that its minimal period is contained in the interval (T0 − ǫ, T0 + ǫ);

(viii) the short closed orbits of Rα are precisely the images via u of those orbits γ of the
       free S 1 -action θt consisting of critical points of S, and the minimal period of such an
       orbit is T0 S(γ).

    The above proposition sharpens [AB23, Proposition 1 and Remark 3.1], in which the
fact that all short closed orbits of Rα are given by the critical points of S is proven assuming
that α is C 3 -close to α0 . The proof of Proposition 4.3 uses the following lemma.

Lemma 4.4. There exists ν > 0 such that the following is true. Let V be an autonomous
smooth vector field on Rk with kV kC 1 < ν and A : [0, 1] × Rk → Hom(Rk , Rk ) a smooth
map into the space of linear endomorphisms of Rk with kAkC 0 < ν. Let us define the
time-dependent vector field Wt := (id +At )V and let (χt )t∈[0,1] be its flow. Then the fixed
points of χ1 are precisely the zeros of the vector field V .



                                                  28
Proof. Every zero of V is also a zero of Wt for all t and hence a fixed point of χ1 . Let
p ∈ Rk be a point such that V (p) 6= 0. We need to show that χ1 does not fix p if ν is small
enough. We claim that there is R > 0 such that for all (t, x) ∈ [0, 1] × BR (p), we have

                    (a) |Wt (x)| < R,         (b) |Wt (x) − V (p)| < |V (p)|,

where BR (p) denotes the open ball of radius R centered at p. Let us show how to use the
claim to prove the lemma. Call γ : [0, 1] → Rk the trajectory of Wt starting at p, that is,
γ(t) = χt (p) for all t ∈ [0, 1]. We need to show that γ is not closed. Let s ∈ [0, 1] be such
that γ([0, s]) is contained in BR (p). By (a),
                                            ˆ s             ˆ s
              |γ(s) − p| = |γ(s) − γ(0)| ≤      |γ̇(t)|dt =     |Wt (γ(t))|dt < Rs.
                                               0              0

Thus, γ(s) ∈ BsR (p) and we deduce that γ([0, 1]) is contained in BR (p). Denoting by h·, ·i
the euclidean inner product in Rk , for all t ∈ [0, 1] we have by (b)

   d
      hγ(t), V (p)i = hγ̇(t), V (p)i = hWt (γ(t)), V (p)i ≥ |V (p)|2 − |Wt (γ(t) − V (p)||V (p)|
   dt
                                                          > |V (p)|2 − |V (p)||V (p)|
                                                          = 0.

Therefore, t 7→ hγ(t), V (p)i is a strictly increasing function and hence γ is not closed.
   We are left to prove the claim. Set r := |V (p)| > 0. Let ν > 0 and write R = ar for
some positive number a > 0 to be determined. We have

                      |V (x) − V (p)| ≤ RkV kC 1 < aνr,           ∀x ∈ BR (p).

Thus,
               |V (x)| ≤ |V (x) − V (p)| + |V (p)| < (1 + aν)r,         ∀x ∈ BR (p).
For all (t, x) ∈ [0, 1] × BR (p), it follows that

                           |Wt (x)| < (1 + ν)(1 + aν)r,
                   |Wt (x) − V (p)| < |At (x)V (x)| + aνr < (1 + a + aν)νr.

To achieve (a) and (b), we need to show that for every sufficiently small ν > 0, there exists
a > 0 such that
                      (1 + ν)(1 + aν) < a and (1 + a + aν)ν < 1.
If ν > 0 is sufficiently small, these two inequalities are equivalent to
                                     1+ν             1−ν
                                              < a <
                                   1 − ν − ν2       ν(1 + ν)

and clearly possess a solution a > 0.

                                                   29
Proof of Proposition 4.3. If the contact form α is C 2 -close to α0 then the Reeb vector
field Rα is C 1 -close to Rα0 . Statement (vii) is then a consequence of a general fact about
C 1 -perturbations of vector fields inducing a free S 1 -action, see [Ban86, Corollary 1].
     By differentiating (4.1) and contracting along Rα0 , we obtain the identity

                  ıRα0 u∗ α = ıRα0 (dS ∧ α0 + S dα0 + dη) = −dS + F [dS],

which shows that the Reeb vector field of u∗ α is parallel to Rα0 on the critical set of S.
Therefore, every orbit γ of θt consisting of critical points of the S 1 -invariant function S is
a closed orbit of Ru∗ α of minimal period
                            ˆ        ˆ
                               ∗
                              u α = (Sα0 + η + df ) = T0 S(γ).
                             γ         γ

By (4.2), this number is close to T0 when kα − α0 kC 2 is small, so u(γ) is a short closed
orbit of Rα if kα − α0 kC 2 < δ with δ = δ(ǫ) small enough. There remains to show that,
up to reducing the number δ if necessary, every short closed orbit of Rα is obtained in this
way or, equivalently, that every short closed orbit of u∗ Rα is an orbit of the S 1 -action θt
consisting of critical points of S.
   The diffeomorphism u is given by a version of a theorem of Bottkol [Bot80] which is
proven in [AB23, Theorem 2.1]. By this result, u satisfies

                                    h u∗Rα = Rα0 − Q[V ],                                 (4.3)

where h is a smooth S 1 -invariant function on M, V is a smooth S 1 -invariant vector field
which is orthogonal to Rα0 with respect to the chosen S 1 -invariant metric on M, and Q is
an endomorphism of T M lifting the identity. Moreover

                  max{kh − 1kC 1 , kV kC 1 , kQ − idkC 0 } ≤ ω(kα − α0 kC 2 ),            (4.4)

for a suitable modulus of continuity ω, and the zeros of the S 1 -invariant vector field V are
precisely the critical points of the S 1 -invariant function S, see [AB23, Equation (2.25)].
    Therefore, we have to show that if kα − α0 kC 2 is small enough, then the short closed
orbits of the vector field u∗ Rα given by (4.3), where h, Q and V satisfy the above conditions,
are orbits of θt which are contained in the set of zeroes of V .
    The free S 1 -action θt defines the smooth S 1 -bundle
                                                        π
                                  S 1 = R/T0 Z → M −→ B.

Let {Bj }j∈{1,...,k} be a covering of B consisting of open embedded balls and consider a
second covering {Bj′ }j∈{1,...,k} with Bj′ ⊂ Bj for every j. By (4.4), the vector field u∗ Rα is
C 0 -close to Rα0 when kα − α0 kC 2 is small, so we can assume that any short closed orbit of
u∗ Rα which meets π −1 (Bj′ ) is fully contained in π −1 (Bj ). This allows us to fix j and work
in π −1 (Bj ).


                                              30
    By a suitable bundle trivialization, we can identify π −1 (Bj ) with the product S 1 × Bj
in such a way that, denoting by t the variable in S 1 , we have
              Rα0 = ∂t        and       V (t, x) = V (x) ∈ Tx Bj       ∀(t, x) ∈ S 1 × Bj .
From (4.3), we obtain that on π −1 (Bj ) ∼
                                         = S 1 × Bj the vector field u∗ Rα satisfies
                                         g u∗ Rα = ∂t + Q′ [V ],
where the smooth function g : S 1 × Bj → R is C 0 -close to the constant function 1 and Q′ is
an endomorphism of T Bj which lifts the identity and is C 0 -close to the identity. Therefore,
up to a time reparametrization which is determined by the function g, the short closed
orbits of u∗ Rα which are contained in π −1 (Bj ) correspond to the T0 -periodic orbits of the
non-autonomous T0 -periodic vector field Q′ (t, x)[V (x)] which are contained in Bj . The
desired conclusion that these T0 -periodic orbits are just the zeros of V when kα − α0 kC 2
is sufficiently small now follows from Lemma 4.4. This concludes the proof of statement
(viii).


5     Proof of Theorem 3
In this section, we prove Theorem 3. Let α be a contact form on M which is C 2 -close to
the Zoll contact form α0 . By Theorem 4.1, there exist a diffeomorphism u ∈ Diff(M), a
positive S 1 -invariant function S ∈ C ∞ (M), a 1-form η ∈ Ω1 (M) and a function f ∈ C ∞ (M)
satisfying the properties listed in Theorem 4.1. In particular,
                                         u∗ α = Sα0 + η + df.
   We claim that Sα0 + η is a contact form strictly contactomorphic to Sα0 + η + df , i.e.,
that there exists a diffeomorphism v : M → M such that
                                    v ∗ (Sα0 + η + df ) = Sα0 + η.                                      (5.1)
To see this, let us define the family of 1-forms
                                βt := Sα0 + η + t df,           t ∈ [0, 1].
Note that since α is C 2 -close to α0 , it follows from (4.2) that S − 1 and f are C 1 -small
and that η and dη are C 0 -small. This fact implies that βt is C 0 -close to α0 and that dβt
is C 0 -close to dα0 . Therefore, βt is a contact form for all t ∈ [0, 1]. Hence there exists a
unique time-dependent vector field Bt on M which is parallel to the Reeb vector field Rβt
and satisfies ιBt βt + f = 0. Let vt denote the flow generated by Bt . We compute:
       ∂t (vt∗ βt ) = vt∗ (LBt βt + ∂t βt ) = vt∗ (dιBt βt + ιBt dβt + df ) = vt∗ d(ιBt βt + f ) = 0.
Therefore, we have vt∗ βt = β0 for all t ∈ [0, 1] and hence the diffeomorphism v := v1 satisfies
identity (5.1).


                                                    31
   Next, let us define the family of 1-forms

                           γt := Sα0 + (1 − t)η + tη,            t ∈ [0, 1],

where η is the average of η with respect to the S 1 -action. Again, since S −1 is C 1 -small and
η and dη are C 0 -small, γt is a contact form for all t ∈ [0, 1]. We apply Moser’s homotopy
argument: Let Ct be the unique time-dependent vector field on M satisfying
                                 (
                                   Ct ∈ ker γt ,
                                   (ιCt dγt + ∂t γt )|ker γt = 0.

Let wt be the flow generated by Ct . Then

                                           wt∗ γt = egt γ0

where gt is the unique solution of
                                     (
                                      ∂t gt = wt∗(ιRγt ∂t γt )
                                                                                              (5.2)
                                      g0 = 0.

Set g := g1 and w := w1 . Then we have

                                   w ∗ (Sα0 + η) = eg (Sα0 + η).

We proceed with the following claim:
Claim 5.1. The function g satisfies the pointwise bound

                                   |g| ≤ σ(kα − α0 kC 2 ) · |dS|2,                            (5.3)

for some modulus of continuity σ. In particular, g and dg vanish on the critical set of S.
Proof. In order to prove (5.3), we begin by showing that
                                             1
                                    |η − η| ≤ T0 kF kC 0 · |dS|.                              (5.4)
                                             2
It follows from the properties of η and F listed in Theorem 4.1 that

                                LRα0 η = dιRα0 η + ιRα0 dη = F [dS].

Together with the S 1 -invariance of S, we deduce the pointwise estimate
                          ˆ t              ˆ t
             ∗                    ∗
           |θt η − η| =       ∂τ θτ η dτ =     θτ∗ (F [dS]) dτ ≤ |t| · kF kC 0 · |dS|,
                            0                   0

from which the desired estimate (5.4) follows:
                  ˆ T0                    ˆ T0
                1            ∗         1                                1
     |η − η| =         (η − θt η) dt ≤         |t| · kF kC 0 · |dS| dt ≤ T0 kF kC 0 · |dS|.
               T0 0                    T0 0                             2

                                                    32
   Recall that the vector field Ct is characterized by Ct ∈ ker γt and
                                    (ιCt dγt + η − η)|ker γt = 0.

Since γt is C 0 -close to α0 and dγt is C 0 -close to dα0 , this implies together with (5.4) that
                                        |Ct | ≤ b kF kC 0 · |dS|                                 (5.5)
for some constant b > 0 which can be chosen uniform among all α which are sufficiently
C 2 -close to α0 .
     Let us split Rγt = Xt + Yt into a vector field Xt parallel to Rα0 and a vector field
Yt taking values in the contact distribution ker α0 . We can write Xt = at · Rα0 where
at ∈ C ∞ (M) is C 0 -close to the constant function 1. We compute
                      ιXt dγt = ιXt (dS ∧ α0 + Sdα0 + (1 − t)dη + tdη)
                              = at (−dS + (1 − t)F [dS] + tF [dS]).
Here we use that ιRα0 dS = 0 because S is S 1 -invariant. Moreover, we use the identity
ιRα0 dη = F [dS] from Theorem 4.1. Combining the above computation with the identity
0 = ιRγt dγt = ιXt +Yt dγt , we obtain

                         ιYt dγt = −at (−dS + (1 − t)F [dS] + tF [dS]).
Since ker γt is C 0 -close to ker α0 and dγt is C 0 -close to dα0 , this implies the pointwise
estimate
                                         |Yt | ≤ b|dS|
for some constant b > 0 independent of the contact form α. In particular, we obtain, in
combination with (5.4), that
          |ιRγt ∂t γt | = |ιXt +Yt (η − η)| = |ιYt (η − η)| ≤ |Yt ||η − η| ≤ bkF kC 0 · |dS|2,
for some constant b independent of α. Using (5.2), this yields the estimate
                         ˆ 1                                  ˆ 1
                   |g| ≤     |ιRγt ∂t γt | ◦ wt dt ≤ bkF kC 0     |dS(wt)|2 dt                   (5.6)
                             0                                     0

In order to bound the integrand in the above expression, we argue as follows. By assertion
(vi) in Theorem 4.1, we have
                                      dS = −B[V ],
where, provided α is C k+2 -close to α0 , V is a C k+1 small, S 1 -invariant section of ker α0 ,
and B : ker α0 → (ker α0 )∗ is an isomorphism which is C k -close to the isomorphism B0
induced by the non-degenerate bilinear form dα0 on ker α0 . Since we are assuming that α
is C 2 -close to α0 , V is C 1 -small and B is C 0 -close to B0 . In particular, both B and B −1
are uniformly bounded and we obtain the pointwise bounds
                                        1
                                          |V | ≤ |dS| ≤ c|V |,                                   (5.7)
                                        c
                                                  33
for a suitable number c ≥ 1. Therefore, from (5.6) we deduce the bound
                                              ˆ 1
                                     2
                             |g| ≤ bc kF kC 0     |V (wt )|2 dt.
                                                                    0

Next we notice that
        d
           |V (wt )|2 = 2hV, ∇Ct V i ◦ wt ≤ 2kV kC 1 |V (wt )| · |Ct (wt )|
        dt
                      ≤ 2bkV kC 1 kF kC 0 |V (wt )| · |dS(wt)| ≤ 2bckV kC 1 kF kC 0 |V (wt )|2 ,

where we have used estimate (5.5) and the second inequality in (5.7). It follows from
Gronwall’s inequality that, for all t ∈ [0, 1],

                                           |V (wt )|2 ≤ e2bckV kC 1 kF kC 0 t · |V |2 .

Thus
                         ˆ       1
            2
    |g| ≤ bc kF kC 0 ·               e2bckV kC 1 kF kC 0 t dt · |V |2 ≤ bc2 kF kC 0 · e2bckV kC 1 kF kC 0 · |V |2 .    (5.8)
                             0

From (4.2) it follows that
                                        
                                     max kV kC 1 , kF kC 0 ≤ ω0 (kα − α0 kC 2 ).

Inequality (5.8) and the first estimate in (5.7) therefore imply a pointwise bound of the
form
                                |g| ≤ σ(kα − α0 kC 2 ) · |dS|2,
for some modulus of continuity σ. This concludes the proof of Claim 5.1.
   Our next task is to prove the following claim.
Claim 5.2. If kα − α0 kC 2 is small enough, then

                                     min Seg = min S,               max Seg = max S.                                   (5.9)
                                      M              M                  M            M

Proof. We claim that if kα − α0 kC 2 is small enough, then S satisfies the pointwise bound
                                                                     1
                                                 S ≥ min S +           4
                                                                         |dS|2 ,                                      (5.10)
                                                                    4c
where c is the number appearing in (5.7).
    As already recalled, the vector field V is C 1 -small when kα − α0 kC 2 is small. Therefore,
when kα − α0 kC 2 is sufficiently small the following fact holds: for any p ∈ M with r :=
|V (p)| > 0, setting R := 3rc we have
                                              r
                                                ≤ |V | ≤ 2r             on BR (p),                                    (5.11)
                                              2
                                                               34
where BR (p) ⊂ M denotes the ball of radius R centered at p.
   The bound (5.10) trivially holds at critical points of S. Let p ∈ M be a point such
that dS(p) 6= 0. By (5.7) the number r = |V (p)| is positive and (5.11) holds with R = 3rc.
Now let γ : R → M be the solution of the Cauchy problem

                             γ ′ (t) = −∇S(γ(t)),            γ(0) = p.

If γ([0, s]) is contained in BR (p) for some s ∈ [0, 1], then by (5.7) and (5.11) we have
                                                ˆ s               ˆ s
                                                       ′
             dist(γ(s), p) = dist(γ(s), γ(0)) ≤     |γ (t)| dt =      |dS(γ(t))| dt
                                                 0                 0
                                                 ˆ s
                                              ≤c      |V (γ(t))| dt ≤ 2rcs ≤ 2rc < R.
                                                   0

The above inequality implies that γ([0, 1]) ⊂ BR (p). Therefore, using again (5.7) and
(5.11), we obtain the chain of inequalities
                                  ˆ 1
                                                             1 1
                                                               ˆ
                                                2
        min S ≤ S(γ(1)) = S(p) −      |dS(γ(t))| dt ≤ S(p) − 2    |V (γ(t))|2 dt
                                    0                        c  0
                                    1 2             1                  1
                        ≤ S(p) − 2 r = S(p) − 2 |V (p)| ≤ S(p) − 4 |dS(p)|2,
                                                           2
                                   4c             4c                  4c
proving (5.10).
    We can now verify the first identity in (5.9). Let q ∈ M be a point at which S achieves
its minimum. Then dS(q) = 0 and hence g(q) = 0 thanks to Claim 5.1. From this, we
obtain the inequality
                           min Seg ≤ S(q)eg(q) = S(q) = min S.
                             M                                      M

In order to prove the opposite inequality, we consider the pointwise bound
                                               1                     
           g
        Se ≥ S(1 + g) ≥ S − 2|g| ≥ min S +         4
                                                     − 2σ(kα − α0 kC ) |dS|2 .
                                                                    2                   (5.12)
                                      M         4c
Here, the first inequality follows from the convexity of the exponential function, the second
one from the fact that we can assume that S ≤ 2, as by (4.2) S is C 0 -close to the constant
function 1 when kα − α0 kC 2 is small, and the third inequality follows from (5.3) and (5.10).
   If kα − α0 kC 2 is small enough, then the term in brackets at the end of (5.12) is non-
negative and hence
                                       min Seg ≥ min S.
                                       M               M

This proves the first identity in (5.9). The proof of the second one is analogous.
   Let us define the family of one-forms

                                 δt := Sα0 + tη,           t ∈ [0, 1].


                                              35
Since S − 1 is C 1 -small and η and dη are C 0 -small when α is C 2 -close to α0 , see (4.2), this
is a contact form for all t ∈ [0, 1]. Let Dt denote the time-dependent vector field on M
characterized by                 (
                                   Dt ∈ ker δt ,
                                   (ιDt dδt + ∂t δt )|ker δt = 0.
We observe that since both S and η are S 1 -invariant, the vector field Dt is S 1 -invariant
as well. Let ψt be the flow generated by Dt . Since Dt is S 1 -invariant, this flow is S 1 -
equivariant. Moreover, we have
                                     ψt∗ δt = edt δ0 ,
where dt ∈ C ∞ (M) is the unique solution of
                                 (
                                   ∂t dt = ψt∗ (ιRδt ∂t δt )
                                   d0 = 0.

We claim that ψ1∗ δ1 = (S ◦ ψ1 )α0 . Indeed, we compute

                      ιRα0 ψ1∗ δ1 = ψ1∗ (ι(ψ1 )∗ Rα0 δ1 ) = ψ1∗ (ιRα0 δ1 ) = S ◦ ψ1 .

On the other hand, we have

                                   ιRα0 ψ1∗ δ1 = ιRα0 ed1 δ0 = Sed1 .

Therefore, we have S ◦ ψ1 = Sed1 . This implies that

                              ψ1∗ δ1 = ed1 δ0 = Sed1 α0 = (S ◦ ψ1 )α0 .

Let us abbreviate ψ := ψ1 . Then the above computations show:

                                     ψ ∗ (Sα0 + η) = (S ◦ ψ)α0 .

   Let us define the diffeomorphism ϕ of M by ϕ := u ◦ v ◦ w ◦ ψ. Then

     ϕ∗ α = ψ ∗ w ∗ v ∗ u∗ α = ψ ∗ w ∗v ∗ (Sα0 + η + df ) = ψ ∗ w ∗(Sα0 + η) = ψ ∗ (eg (Sα0 + η))
                            
          = (eg S) ◦ ψ α0 .

Therefore, setting
                                    T := S ◦ ψ,          h := g ◦ ψ,
we obtain that the diffeomorphism ϕ satisfies the desired identity

                                            ϕ∗ α = T eh α0 .

   We now check that properties (a)-(e) from Theorem 3 hold. Since S is S 1 -invariant
and ψ is S 1 -equivariant, the function T is S 1 -invariant, proving (a). By Claim 5.1, both
g and dg vanish on the critical set of S. Therefore, the same is true for the pull-backs

                                                    36
h and T under the diffeomorphism ψ, proving property (b). Property (c) follows from
Claim 5.2. Property (d) follows from property (vii) in Proposition 4.3. By property (viii)
from the same proposition, the short closed orbits of Ru∗ α are precisely the orbits γ of the
free S 1 -action θt that consist of critical points of S, and the minimal period of such an
orbit is T0 S(γ) = T0 T (ψ −1 (γ)). Since u∗ α = Sα0 + η + df and v ∗ u∗ α = Sα0 + η induce
the same Reeb flow up to a time reparametrization which preserves the periods of closed
orbits, the same is true for the short closed Reeb orbits of v ∗ u∗ α. By estimate (5.5), the
diffeomorphism w fixes the critical set of S, so the same fact continues to hold for the
short closed Reeb orbits of w ∗v ∗ u∗ α. By applying the S 1 -equivariant diffeomorphism ψ,
we obtain that the short closed Reeb orbits of ϕ∗ α are the orbits of θt forming the image
by ψ −1 of the critical set of S. The latter set is precisely the critical set of T and property
(e) follows.
    The last claim in Theorem 3 is that the map α 7→ (ϕ, T, h) we have just constructed
is smooth and maps α0 to (id, 1, 0). This follows from the corresponding statement about
the map α 7→ (u, S, η, f ) in Theorem 4.1 and from the fact that the tuple (v, w, g, ψ) we
constructed above depends smoothly on the defining data (S, η, f ) and takes the value
(id, id, 0, id) for (S, η, f ) = (1, 0, 0). This concludes the proof of Theorem 3.


6     Proof of Theorem 1 (ii)
In this section, we deduce Theorem 1 (ii) from Theorem 3. Recall that A denotes the set
of domains in Cn of the form

                            Af = {rz | z ∈ S 2n−1 , 0 ≤ r < f (z)},

where f is a positive smooth function on S 2n−1 , and that the C k -distance on A is induced
by the C k -distance on the space of positive smooth functions on S 2n−1 . For every A in A,
we denote by
                                         αA := λ0 |∂A
the restriction of the standard primitive of ω0 to the boundary of A. In the case of the
unit ball A1 = B, we find the standard contact form

                                     α0 := αB = λ0 |S 2n−1

on S 2n−1 , which is Zoll with all orbits of period π. Given a domain A = Af in A, the radial
projection
                               ρA : S 2n−1 → ∂A,       z 7→ f (z)z,
satisfies
                                         ρ∗A αA = f 2 α0 .                                 (6.1)
This identity shows that A ∈ A is C k -close to B if and only if ρ∗A αA is C k -close to α0 .



                                                37
   We now proceed with the proof of statement (ii) in Theorem 1. Given a smooth positive
function f on S 2n−1 , we denote by

                                        {ft := 1 + t(f − 1)}t∈[0,1]

the smooth homotopy connecting the constant function 1 to f and we consider the corre-
sponding smooth path of domains {Aft }t∈[0,1] in A, connecting B to Af . We assume that
the domain A := Af is sufficiently C 2 -close to B, so that

                      kρ∗Af αAft − α0 kC 2 = k(ft2 − 1)α0 kC 2 < δ              ∀t ∈ [0, 1],
                           t


where δ is the positive number appearing in Theorem 3. Note that {ρ∗Af αAft }t∈[0,1] is a
                                                                           t
smooth path of contact forms on S 2n−1 connecting α0 to ρ∗A αA . By Theorem 3, there exists
a smooth family of diffeomorphism {ϕt : S 2n−1 → S 2n−1 }t∈[0,1] such that

                                        ϕ∗t (ρ∗Aft αAft ) = Tt eht α0 ,                            (6.2)

where {Tt }t∈[0,1] and {ht }t∈[0,1] are smooth families of functions on S 2n−1 satisfying prop-
erties (a)-(e) of Theorem 3 together with ϕ0 = id, T0 = 1 and h0 = 0. By (e) we have

          sys(A) = sys(αA ) = sys(ρ∗A αA ) = sys(ρ∗A1 αA1 ) = sys(α0 ) min
                                                                       2n−1
                                                                            T1 = π min
                                                                                   2n−1
                                                                                        T1 .
                                                                                 S             S

Therefore, Theorem 1 (ii) will be proven if we can find a symplectomorphism φ : Cn → Cn
mapping the open ball B ′ of radius
                                            q
                                      R := min 2n−1
                                                    T1
                                                        S

centered at the origin into A. By property (c) of Theorem 3, we have
                                 q                      p    h1
                             R = min      T  eh1 = min    T
                                           1               1e 2
                                     2n−1  S        n−1            S

and hence the domain
                                               A′ := A√              h1
                                                              T1 e    2

contains B ′ . Hence, it is enough to find a symplectomorphism φ : Cn → Cn such that
φ(A′ ) = A. Consider the smooth path of domains

                                          {A′t := A√          ht   }t∈[0,1] ,
                                                       Tt e   2


which satisfies A′0 = B and A′1 = A′ . By (6.2) and (6.1), the diffeomorphisms

                               ψt : ∂A′t → ∂Aft ,       ψt := ρAft ◦ ϕt ◦ ρ−1
                                                                           A′ ,      t


satisfy

             ψt∗ (λ0 |∂Aft ) = (ρ−1  ∗  ∗ ∗                 −1 ∗      ht
                                 A′ ) (ϕt (ρAft αAft )) = (ρA′ ) (Tt e α0 ) = αAt = λ0 |∂At .
                                                                                ′         ′
                                   t                                      t


                                                      38
The fact that the diffeomorphism ψt preserves the restriction of λ0 implies that its positively
1-homogeneous extension

            ψ̃t : Cn \ {0} → Cn \ {0},          ψ̃t (rz) := rψt (z) ∀z ∈ ∂A′t , ∀r > 0,

preserves λ0 . By construction, ψ̃t depends smoothly on t and satisfies

                            ψ̃0 = id,        ψ̃1 (A′ \ {0}) = A \ {0}.

Let Xt be the vector field generating ψ̃t . Since ψ̃t preserves λ0 , we have

                                0 = LXt λ0 = ıXt ω0 + dıXt λ0 ,

and hence Xt is the Hamiltonian vector field of the Hamiltonian Ht := −λ0 (Xt ) on Cn \
{0}. Multiplying Ht by a smooth function on Cn which is supported in Cn \ {0} and
equals 1 outside of a sufficiently small neighborhood of 0, we obtain a new time-dependent
Hamiltonian on Cn whose time-one map is a global symplectomorphism φ : Cn → Cn which
coincides with ψ̃1 on Cn \ A′ , and hence satisfies φ(A′ ) = ψ̃1 (A′ ) = A. This concludes the
proof of Theorem 1 (ii).


7     Proof of Theorem 2
Let A ∈ A be C 2 -close to B. Using the notation of Section 6, we deduce that the contact
form ρ∗A αA on S 2n−1 is C 2 -close to the Zoll contact form α0 and hence there exists a
diffeomorphism ϕ : S 2n−1 → S 2n−1 such that

                                        ϕ∗ (ρ∗A αA ) = T eh α0 ,                          (7.1)

where the smooth functions T : S 2n−1 → R and h : S 2n−1 → R satisfy all the requirements
of Theorem 3. In particular, ∂A has a closed characteristic γ whose action equals the
number
                                  Tmax (A) := π max
                                                2n−1
                                                     T.
                                                       S

Up to rescaling, we can assume that

                                  Tmax (A) = Tmax (B) = π,                                (7.2)

and we shall prove the following facts:

 (i’) There exists a symplectomorphism φ : Cn → Cn mapping the ellipsoid E :=
      E(π, π2 , . . . , π2 ) into A.

 (ii’) There exists a symplectomorphism ψ : Cn → Cn mapping A into B.



                                                  39
From the monotonicity property of the capacity c we then have

                                            c(E) ≤ c(A) ≤ c(B).

Since c is n-normalized, its values at E and B coincide with the corresponding values of
the n-th Ekeland-Hofer capacity, which assigns to both E and B the value π. Therefore,
c(E) = c(B) and both inequalities in the above expression are equalities. This proves
claims (i) and (ii) from Theorem 2. There remains to prove (i’) and (ii’).

   The proof of (i’) is similar to the proof of Theorem 1 (i) from Section 3, and here we use
the notation which we introduced there. Up to the application of a unitary isomorphism,
we may assume that the closed characteristic γ, which by (7.2) has action π, passes through
the point (1, 0, . . . , 0) ∈ Cn and is contained in the open set V = Φ(U), see (1.1) and (3.1).
By the explicit form (1.1) of the symplectomorphism Φ, we have the identities
                                                                               
               B ∩ V = Φ(U ∩ {s < 0}),            E ∩ V = Φ U ∩ {s < −π|w|2} .             (7.3)

Since A is C 2 -close to B, we can find a C 2 -small smooth function H : T × Cn−1 → R with
support in T × B  b satisfying (1.4) and such that

                                            A ∩ V = D(H) ∩ V.                             (7.4)

The closed characteristic γ of action π corresponds to the fixed point 0 of φ1H , which has
action 0, see Proposition 1.1 (ii). By Proposition 3.1, there exists a smooth family of
Hamiltonians {H λ }λ∈[0,1] on T × Cn−1 which satisfies:

 (i) H 0 = H;
                     b c1 for every λ ∈ [0, 1];
 (ii) H λ = H on T × B
                             3


(iii) φ1H λ = φ1H for every λ ∈ [0, 1];
                π
(iv) |H λ | <   2
                    for every λ ∈ [0, 1];

 (v) H 1 (t, w) ≥ −π|w|2 for every (t, w) ∈ T × Cn−1 .

By (i) and (iii), Proposition 1.1 (iii) gives us a symplectomorphism φe : Cn → Cn such that
φ̃(D(H)) = D(H 1), which by (ii), (iv) and Remark 1.2 is supported in V and by (7.4)
satisfies
                               φ̃(A) = (A \ V ) ∪ (D(H 1) ∩ V ).
Since the closure of E \ V is contained in B, C 0 -closeness of A to B implies that E \ V
is contained in A \ V . By (v) and the second identity in (7.3), E ∩ V is contained in
D(H 1 ) ∩ V , so the above identity implies that E is contained in φ̃(A). We conclude that
the symplectomorphism φ := φ̃−1 satisfies φ(E) ⊂ A, proving (i’).


                                                    40
   The proof of (ii’) is similar to the proof of Theorem 1 from Section 6. Arguing as in
that proof, we can apply Theorem 3 to a 1-parameter family of contact forms joining α0
and ρ∗A αA and obtain a symplectomorphism ψ : Cn → Cn mapping A onto the domain

                                          A′ := A√     h    ,
                                                     Te 2

see (7.1). By (7.2) and the fact that the maximum of T eh coincides with the maximum of
T (see statement (c) in Theorem 3), we get that
                                              √ h
                                     max
                                       2n−1
                                            (  T e 2 ) = 1.
                                      S

Hence, A′ ⊂ A1 = B, proving (ii’).


8    Proof of Proposition 2
We now prove Proposition 2. Up to exchanging the coordinates and rescaling, we may
assume that a = 1 ≤ b, so that

                        b
                        ck (P (1, b)) = k min{1, b} = k,          ∀k ∈ N,

see [EH90, Proposition 5] and [GH18, Example 1.7].
    Next we recall that the ellipsoid E(1, 2) symplectically embeds into P (1, 1) (see [FM15,
Theorem 1.3] and [Sch18, Lemma 8.2]). The existence of this symplectic embedding is
quite remarkable, because E(1, 2) and P (1, 1) have the same volume and are not symplec-
tomorphic, as b c3 (E(1, 2)) = 2 and b
                                     c3 (P (1, 1)) = 3.
    By the “extension after retraction principle” from [Sch02, Proposition 1.3], for every
θ < 1 there is a global symplectomorphism of C2 mapping E(θ, 2θ) into P (1, 1), and hence
also into P (1, b). Therefore,

                          ck (P (1, b)) ≥ b
                                          ck (E(θ, 2θ))         ∀k ∈ N,

and by taking the supremum over all θ < 1 we obtain

                           ck (P (1, b)) ≥ b
                                           ck (E(1, 2))         ∀k ∈ N.                 (8.1)

For every α > 1 we choose β large enough, so that P (1, b) ⊂ E(α, β). Then for every
k ∈ N we have
                          ck (P (1, b)) ≤ b
                                          ck (E(α, β)) ≤ kα,
and by taking the infimum over all α > 1 we obtain

                                ck (P (1, b)) ≤ k           ∀k ∈ N.                     (8.2)

On the other hand, k = b
                       ck (P (1, b)) ≤ ck (P (1, b)) and we conclude that

                               ck (P (1, b)) = k,           ∀k ∈ N.

                                              41
Since bc2 (E(1, 2)) = 2 and every 2-normalized capacity c satisfies c2 ≤ c ≤ c2 , the inequali-
ties (8.1) and (8.2) for k = 2 imply that

                                c(P (1, b)) = 2 = 2 min{1, b},

proving statement (i) of Proposition 2.
    In order to prove statement (ii), we use Hutchings’ ECH-capacities cECH    k    for four-
dimensional domains from [Hut11]. By [Hut11, Proposition 1.2], the k-th ECH-capacity
of the ellipsoid E(a, b) is the (k + 1)-th smallest element in the list (ha + jb)h,j≥0, again
allowing repetitions. In particular, cECH
                                      k   is not a k-normalized capacity in the sense of this
paper.
    Let E(α, β) be an ellipsoid which symplectically embeds into P (1, 1). As observed in
[Hut11, Remark 1.8], the ECH-capacities P (1, 1) and E(1, 2) coincide, and hence

                cECH
                 k   (E(α, β)) ≤ cECH
                                  k   (P (1, 1)) = cECH
                                                    k   (E(1, 2))     ∀k ∈ N.

As proven by McDuff in [McD11], the ECH-capacities form a complete list of obstructions
for the problem of symplectically embedding an ellipsoid into another, and hence the above
inequalities imply that E(α, β) symplectically embeds into E(1, 2). Therefore,

                           b
                           ck (E(α, β)) ≤ b
                                          ck (E(1, 2))     ∀k ∈ N,

and by taking the supremum over the space of all ellipsoids E(α, β) which embed into
P (1, 1) by a globally defined symplectomorphism of Cn we obtain the inequalities

                            ck (P (1, 1)) ≤ b
                                            ck (E(1, 2))   ∀k ∈ N.

Together with (8.1), we deduce the identities

                            ck (P (1, 1)) = b
                                            ck (E(1, 2))   ∀k ∈ N.

Moreover, bck (E(1, 2)) ≤ k with equality if and only if k equals 1 or 2. This proves statement
(ii) of Proposition 2.


9     Proof of Theorem 4
The proof of statement (i) in Theorem 4 makes use of the following well known alterna-
tive (see [Sul76, Theorem II.26] for a proof based on the Hahn–Banach theorem and the
appendix of [LS94] for a more elementary proof).

Theorem 9.1. Let X be a smooth vector field on the closed manifold M and let K be a
compact subset of M. Then there exists either a probability measure which is supported
in K and is invariant under the flow of X or a smooth real function h on M such that
dh[X] > 0 on K.


                                               42
    Let ξ be a co-oriented contact structure on the closed manifold M. A smooth vector
field on M is said to be a contact vector field if its flow consists of contactomorphisms of
(M, ξ). Contact vector fields are in one-to-one correspondence with smooth real functions
on M: A smooth real function H on M (the contact Hamiltonian) and a defining contact
form α for ξ define the contact vector field X by the identities
                                          
                          ıX dα = ıRα dH α − dH,          ıX α = H,                    (9.1)

see [Gei08, Section 2.3]. We shall make use of the following formula for the conformal
factor of the flow of contactomorphisms in terms of the generating contact Hamiltonian.

Lemma 9.2. Let X be the contact vector field induced by the contact Hamiltonian function
H ∈ C ∞ (M) together with the contact form α ∈ F (M, ξ), and denote by φtX its flow. Then
                                        ˆ t                  
                            t ∗
                         (φX ) α = exp       (ıRα dH) ◦ φsX ds α,
                                               0

for every t ∈ R.

Proof. We define ht by the identity (φtX )∗ α = eht α. Then h0 = 0 and differentiation in t
gives us
                                d t ∗                                            
                   ∂t ht eht α =  (φX ) α = (φtX )∗ LX α = (φtX )∗ ıX dα + dıX α
                               dt               
                             = (φtX )∗ (ıRα dH)α = (ıRα dH) ◦ φtX eht α,

where we have used (9.1). Then
                              ˆ t            ˆ t
                         ht =     ∂s hs ds =     (ıRα dH) ◦ φsX ds,
                                    0               0

as claimed.
   We can now prove Theorem 4.
Proof of Theorem 4. We are assuming that the contact forms α, β ∈ F (M, ξ) satisfy

                           d(α, β) = max f − min f      with ϕ∗ β = ef α,
                                        M      M

for some ϕ ∈ Cont0 (M, ξ).
(i) If f is constant, then Rϕ∗ β is a constant multiple of Rα and hence the probability
measure on M which is determined by the volume form α ∧ dαn−1 is invariant for the flows
of both these Reeb vector fields. Therefore, we can assume that the compact sets

                      Kmin := f −1 (min f )     and      Kmax := f −1 (max f )
                                     M                                  M



                                                   43
are disjoint. Since df vanishes on Kmin and Kmax , the identity

                                d(ϕ∗ β) = d(ef α) = ef (df ∧ α + dα)

implies that

                                    Rϕ∗ β = e− min f Rα   on Kmin ,
                                    Rϕ∗ β = e− max f Rα   on Kmax .
Therefore, any invariant measure for the Reeb flow of ϕ∗ β which is supported in either
Kmin or Kmax is invariant also for the Reeb flow of α. We shall show the existence of a
Rϕ∗ β -invariant probability measure supported in Kmin , the case of Kmax being analogous.
    We argue by contradiction and assume that the flow of Rϕ∗ β admits no invariant prob-
ability measure which is supported in Kmin . Then Theorem 9.1 implies the existence of a
smooth real function H on M such that
                                        
                                 dH Rϕ∗ β > 0 on Kmin .

By multiplication by a smooth function having the value 1 near Kmin and 0 near Kmax ,
we can assume that H = 0 on a neighborhood V of Kmax . Let X be the contact vector
field which is induced by the contact Hamiltonian H and the contact form ϕ∗ β = ef α. By
Lemma 9.2, we have
                                    (φtX )∗ (ϕ∗ β) = egt α,
where                           t                                    
                            ˆ
                gt := f +           ıRϕ∗ β dH ◦ φsX ds = f + t dH Rϕ∗ β + O(t2 )
                            0
for t → 0. We can find an open neighborhood U of Kmin and a positive number δ such
that
                           
                    dH Rϕ∗ β ≥ δ                   on U,
                    f ≥ min f + δ                  on M \ U,
                    f ≤ max f − δ                  on M \ V.

On U, we then have
                                       gt ≥ min f + δt + O(t2 ),
while on M \ U we have
                            gt ≥ min f + δ − ct + O(t2 ),
                                                    
where c is the supremum norm of the function dH Rϕ∗ β . We conclude that

                                      gt ≥ min f + δ2 t on M,                        (9.2)

for t ≥ 0 small enough. Since H vanishes on V , so does X and hence gt = f on V for every
t ∈ R. On M \ V we have
                               gt ≤ max f − δ + ct + O(t2 ).

                                                   44
Therefore,
                                              gt ≤ max f        on M,
for t ≥ 0 small enough. Together with (9.2), this implies that for t > 0 sufficiently small
the contactomorphism
                              ψ := ϕ ◦ φtX ∈ Cont0 (M, ξ)
satisfies
                                                     ψ ∗ β = egt α
with
                                    max gt − min gt < max f − min f,
contradicting the assumption that d(α, β) = max f − min f .
(ii) From the definition of the pseudo-metric d we obtain

                d(eg1 α, eg2 α) = d(eg1 α, eg2 −g1 eg1 α) ≤ max(g2 − g1 ) − min(g2 − g1 ),                             (9.3)
                                                                    M                       M

for every g1 and g2 in C ∞ (M). Given the path {ψt }t∈[0,1] in Cont0 (M, ξ) such that ψ0 = id
and ψ1 = ϕ−1 , we consider the curve

                                      γ(t) := ψt∗ (etf α),           t ∈ [0, 1],

which satisfies γ(0) = α and γ(1) = β. For any subdivision 0 = t0 < t1 < · · · < tk = 1
we have, by the invariance of the psedo-metric d under the action of Cont0 (M, ξ) and by
(9.3),
                k
                X                             k
                                              X                                                 k
                                                                                                X
  d(α, β) ≤           d(γ(tj−1 ), γ(tj )) =         d(ψt∗j−1 (etj−1 f α), ψt∗j (etj f α))   =         d(etj−1 f α, etj f α)
                j=1                           j=1                                               j=1
                k
                X
            ≤         max(tj f − tj−1 f ) − min(tj f − tj−1 f ) = max f − min f = d(α, β).
                       M                        M                            M              M
                j=1

The above inequalities are then equalities and hence γ has length d(α, β) and is therefore
a minimizing geodesic.


10      Elementary spectral invariants
The proof of Theorem 5 in Section 11 is based on a sequence of spectral invariants for
contact forms on a closed contact manifold, whose definition and properties are discussed
in the present section. These spectral invariants are inspired by the symplectic capacities
defined by McDuff–Siegel in [MS23], by those introduced by Hutchings in [Hut22a, Hut22b]
and by the spectral invariants for Hamiltonian diffeomorphisms defined by one of us in
[Edt22b]. They are more elementary than, say, the spectral invariants which are induced by
contact homology, in the sense that their definition and the proof of their properties require

                                                           45
only compactness results for pseudoholomorhpic curves. Their computation involves some
knowledge of Gromov–Witten invariants, see Proposition 10.3 below.
    Before discussing them, we recall some basic facts and fix notation concerning the
symplectization of a contact manifold. Let ξ be a co-oriented contact structure on the
(2n − 1)-dimensional manifold M. A cotangent vector x ∈ Tp∗ M is said to define ξ(p) if
ker x = ξ(p) and x is positive on vectors in Tp M which are positively transverse to ξ(p).
We denote by
                           f := {x ∈ T ∗ M | x defines ξ(π(x))}
                           M
the symplectization of M, where π : T ∗ M → M denotes the footpoint projection. The
                f is a 2n-dimensional submanifold of T ∗ M. It carries a Liouville 1-form
symplectization M
e and a symplectic form ω
λ                             e given by the restriction of the standard Liouville 1-form
                         e = dλ
                                        f carries a canonical R-action given by
and symplectic form of T ∗ M. Moreover, M
                                 f→M
                               R×M f,               (a, η) 7→ ea η.
                                    f generating this action. The choice of a contact form
We denote by ∂a the vector field on M
α ∈ F (M, ξ) induces the diffeomorphism
                                        f,
                           ϕα : R × M → M           (a, p) 7→ ea α(p),
such that ϕ∗α ωe = d(ea α) where a is the variable in the first factor of R × M. Under
                      f with R × M defined by ϕα , the canonical action of R is given by
the identification of M
translations in the first factor (this justifies the choice of denoting the generating vector
field by ∂a ) and the contact-type hypersurface
                                                        f
                                 Mα := {α(p) | p ∈ M} ⊂ M
corresponds to the set {a = 0}. The contact form α and the Reeb vector field Rα can be
seen as a translation invariant 1-form and vector field on R×M, and by the diffeomorphism
ϕα induce an R-invariant 1-form and an R-invariant vector field on Mf:

                                α
                                e := ϕα ∗ α,     eα := ϕα ∗ Rα .
                                                 R
                    eα is tangent to the hypersurface Mα and all its images by the canonical
   The vector field R
R-action, and spans the characteristic distribution of these hypersurfaces.
   Given contact forms α, β ∈ F (M, ξ) with α < β, we denote by M   fβ ⊂ Mf the symplectic
                                                                      α
cobordism which is given by the region in M f bounded by Mα and Mβ , that is,

fβ := {x ∈ M
M          f | α(π(x)) < x < β(π(x)) on vectors which are positively transverse to ξ(π(x))}.
 α

The diffeomorphism ϕβ exhibits the region M    f+ ⊂ M  f above the hypersurface Mβ as a
                                                 β
positive cylindrical end attached to the symplectic cobordism M  fβ . Similarly, we can regard
                                                                    α
the region Mf− ⊂ M  f below the hypersurface Mα as a negative cylindrical end of M      fβ via
              α                                                                           α
the diffeomorphism ϕα .
   Let us recall the following terminology from [BEH+ 03]. An almost complex structure J
   f is called adjusted to a contact form α ∈ F (M, ξ) if it satisfies the following properties:
on M

                                               46
                                                    f.
  1. J is invariant under the canonical R-action on M
           eα .
  2. J∂a = R

  3. The codimension-two distribution ξα := ϕα ∗ ((0) × ξ) is invariant under J. Moreover,
     the restriction of J to ξα is compatible with de
                                                    α.
                                                               f. Let u : (Σ, j) →
   Suppose that J is an α-adjusted almost complex structure on M
 f, J) be a J-holomorphic map whose domain is a punctured Riemann surface. We define
(M
the action Aα (u) to be              ˆ
                                      Aα (u) :=       u∗ de
                                                          α ∈ [0, +∞].
                                                  Σ
Note that since u is J-holomorphic and J is adjusted to α, the 2-form u∗ de
                                                                          α is everywhere
non-negative on Σ. Thus Aα (u) is indeed well-defined an non-negative. Moreover, we
define
                                           
                        ˆ
          Eα (u) := sup u∗ ρ ◦ aα daα ∧ αe ∈ [0, +∞],       with aα := a ◦ ϕ−1
                                                                             α ,
                      ρ     Σ

where the supremum is taken over all compactly supported functions ρ : R → R≥0 whose
integral is equal to 1. As before, the 2-form u∗ (ρ ◦ aα )daα ∧ α
                                                                e is everywhere non-negative
because J is α-adjusted and u is J-holomorphic. We say that u is a finite-energy curve
if both Aα (u) and Eα (u) are finite. If u is a finite-energy curve and α is non-degenerate,
then u is positively or negatively asymptotic to trivial cylinders over periodic Reeb orbits
of α at every puncture of Σ which is not a removable singularity of u. We define the
positive action A+ α (u) to be the sum of the actions (with respect to α) of the positive
asymptotic Reeb orbits of u. Similarly, we define the negative action A−   α (u) to be the sum
of the actions of the negative asymptotic Reeb orbits of u. By Stokes’ theorem, we have
Aα (u) = A+           −
            α (u) − Aα (u).
    Let α < β be contact forms and assume that J is a compatible almost complex structure
on Mf whose restriction to M f+ is β-adjusted and whose restriction to M    fα− is α-adjusted.
                               β
We define the action
                       ˆ                ˆ                ˆ
                                 ∗ e               ∗
               β
             Aα (u) :=          u dβ +            uω e+           u∗ de
                                                                      α ∈ [0, +∞].
                               f+ )
                          u−1 (M                   fαβ )
                                              u−1 (M                      fα− )
                                                                     u−1 (M
                                 β


All three summands in this expression are well-defined and non-negative. For the terms
                               f+ and M
involving the cylindrical ends M        fα− this follows from the fact that J is adjusted
                                 β
and for the cobordism M fβ this uses that J is compatible. In addition, we define
                          α

                                                                             
               ˆ                                  ˆ
   β
 Eα (u) := sup           ∗  +            e
                        u ρ ◦ aβ daβ ∧ β + sup             u∗ ρ− ◦ aα daα ∧ α
                                                                            e ∈ [0, +∞].
           ρ+        f+ )
                u−1 (M                                     ρ−        fα− )
                                                                u−1 (M
                       β


Here the suprema are taken over all compactly supported, non-negative functions ρ± on
R≥0 and R≤0 , respectively, with integral equal to 1. Again, a curve u is said to have

                                                      47
finite energy if both Aβα (u) and Eαβ (u) are finite. If u has finite energy and α and β are
non-degenerate, then u is positively asymptotic to periodic Reeb orbits of β or negatively
asymptotic to periodic Reeb orbits of α at punctures which are not removable singularites.
The positive action A+                                              −
                             β (u) and the negative action Aα (u) are defined as before and we
have Aβα (u) = A+                −
                      β (u) − Aα (u) by Stokes’ theorem.
    Given a contact form α ∈ F (M, ξ), define J (α) to be the set all compatible almost
complex structures J on the symplectization M             f with the property that there exist real
numbers s− < s+ such that J is adjusted to α on both M                f±± . Here the numbers s± are
                                                                       es α
allowed to depend on J. For a given J ∈ J (α), there are many choices of s± making it
                                                        +                         +
adjusted to α on M      f±± . While the action Aes− α and the energy E es− α themselves depend
                          es α                       es α                       es α
on the choice of s± , the property of being a finite-energy J-holomorphic curve does not.
    Suppose now that α ∈ F (M, ξ) is non-degenerate, that is, all the periodic orbits of the
Reeb vector field Rα are non-degenerate. Given an almost complex structure J ∈ J (α) and
k points x1 , . . . , xk ∈ M f, we define MJ (x1 , . . . , xk ) to be the set of all pseudoholomorphic
curves u : (Σ, j) → (M,    f J) with the following properties:

   1. The domain (Σ, j) is a finite union of Riemann spheres with finitely many punctures.

   2. The map u is non-constant on every connected component of Σ.

   3. The points x1 , . . . , xk are contained in the image of u.

   4. u has finite energy.

   5. u has at least one negative asymptotic orbit.
By Stokes’ theorem and exactness of M f, any curve u ∈ MJ (x1 , . . . , xk ) has at least one
positive asymptotic orbit. We recall that the positive action A+ α (u) is the sum of the
α-actions of these orbits. Given a non-degenerate α ∈ F (M, ξ), we define

                       ck (α) :=      sup           inf          A+
                                                                  α (u) ∈ (0, +∞].             (10.1)
                                                 J
                                     J∈J (α) u∈M (x1 ,...,xk )
                                               f
                                   x1 ,...,xk ∈M

    The proof of the basic properties of the functions ck which are listed in the next theorem
is similar to the analogous proofs in [MS23, Hut22a, Hut22b, Edt22b].
Theorem 10.1. The functions given by (10.1) have unique extensions ck : F (M, ξ) →
[0, +∞], k ≥ 0, satisfying the following properties:
   1. (Scaling) ck (r · α) = r · ck (α) for all positive real numbers r.

   2. (Increasing) ck (α) ≤ cℓ (α) for k ≤ ℓ.

   3. (Sublinearity) ck+ℓ (α) ≤ ck (α) + cℓ (α).

   4. (Invariance) ck (ϕ∗ α) = ck (α) for all ϕ ∈ Cont(M, ξ).

                                                    48
  5. (Monotonicity): ck (α) ≤ ck (β) whenever α ≤ β.

  6. (Lipschitz): If ck is finite, then ck is Lipschitz for the C 0 -norm on F (M, ξ).

  7. (Spectrality): If ck (α) is finite, then it is a finite sum of actions of periodic Reeb
     orbits of α.

  8. (Packing): Suppose that α < β and assume that the ball B 2n (A) of symplectic width
                                          fβ . Then
     A admits a symplectic embedding into Mα

                                            ck+1(β) ≥ ck (α) + A.

Remark 10.2. It follows from the monotonicity and scaling properties that, for fixed
(M, ξ) and k, the spectral invariant ck is either finite on all of F (M, ξ) or identically equal
to +∞. If ck is finite, then the Weinstein conjecture holds for (M, ξ).

Proof. It is enough to check the above properties for non-degenerate contact forms. Then,
the Lipschitz property allows us to extend ck continuously to all, possibly degenerate,
contact forms and, by continuity, all properties of ck continue to hold in the degenerate
case.
    Let fr be the diffeomorphism of R × M given by fr (a, p) = (ra, p). The pull-back by
fr yields a bijection
                                   fr∗ : J (rα) → J (α)
                                                      f, a bijection
and, for every J ∈ J (rα) and every x1 , . . . , xk ∈ M
                                                       ∗
                    fr∗ : MJ (x1 , . . . , xk ) → Mfr J (fr−1 (x1 ), . . . , fr−1(xk )).

For every u ∈ MJ (x1 , . . . , xk ), the positive action of u with respect to rα is related to the
positive action of fr∗ (u) with respect to α by A+               +   ∗
                                                     rα (u) = rAα (fr u). This implies the scaling
property ck (rα) = rck (α).
   It is clear from the definition that ck is increasing in k since adding point constraints
can only make the set of relevant curves smaller.
   Sublinearity is also clear because the union of two curves with k and ℓ point constraints,
respectively, yields a curve with k + ℓ point constraints.
   Invariance follows because a contactomorphism ϕ ∈ Cont(M, ξ) lifts to a symplecto-
morphism of the symplectization which induces bijections between the relevant sets of
auxiliary data and curves.
   The Lipschitz property follows from scaling and monotonicity.
   Spectrality follows from the fact that the action spectrum of α is a closed subset of R.
   The proofs of the monotonicity and packing property are more substantial and rely on
a neck-stretching argument and the SFT compactness theorem. Compared to analogous
arguments given in [Hut22a, Edt22b], the proofs are slightly simplified because we have an
apriori genus bound and can apply the SFT compactness theorem directly. As these two
proofs are very similar, we just sketch the proof of the packing property.

                                                    49
    We fix α < β and a symplectically embedded ball B = B 2n (A) in M   fβ . Moreover, we
                                                                           α
                                                                         f
fix an almost complex structure J ∈ J (α) and k points x1 , . . . , xk ∈ M . Let ǫ > 0 be
arbitrarily small. Our goal is to find a J-holomorphic curve u ∈ MJ (x1 , . . . , xk ) satisfying

                                  A+
                                   α (u) + A ≤ ck+1 (β) + ǫ.

Since the constant ǫ, the almost complex structure J, and the point constraints are arbi-
trary, this implies the desired packing inequality.
     Note that if we translate the almost complex structure J and the points xi using the
canonical R-action on M   f, the new moduli space MJ (x1 , . . . , xk ) is obtained from the
old one by applying the same translation. The positive action A+      α (u) is not affected by
translation. After translating J and the points xi sufficiently far downwards, we can
therefore assume w.l.o.g. that J is adjusted to α on an open neighbourhood of M       f+ and
                                                                                        α
that all points xi lie below the hypersurface Mα .
     Fix a real number S such that J is adjusted to α on an open neighbourhood of M  f−S and
                                                                                       e α
such that all xi lie above the hypersurface MeS α . Let T ∈ R be such that the hypersurface
MeT β lies below the hypersurface MeS α . Let Jα+ denote the α-adjusted almost complex
structure whose restriction to M fα+ agrees with J and let Jα− denote the α-adjusted almost
complex structure whose restriction to M  fe−s α agrees with J.
     Let Jαβ be a compatible almost complex structure which agrees with Jα+ on a neigh-
bourhood of M   fα− and which is equal to a β-adjusted almost complex structure J + on M   f+ .
                                                                                    β       β
Moreover, we assume that the restriction of Jαβ to the ball B agrees with the standard
integrable almost complex structure on Euclidean space.
     Let Jβα be a compatible almost complex structure agreeing with Jα− on a neighbourhood
of Mf+S and whose restriction to M  f−T agrees with a β-adjusted almost complex structure
      e α                             e β
Jβ− . Let us fix small tubular neighbourhoods nb(Mα ) and nb(MeS α ) and sequences of
diffeomorphisms

                    ϕ+   feν α                      ϕ−   feν α
                     ν : Me−ν α → nb(Mα ),           ν : Me−ν α → nb(MeS α )

with the following properties:

   1. ϕ±
       ν maps translates of the hypersurface Mα to translates of Mα .

   2. π ◦ ϕ±               f
           ν = π where π : M → M is the canonical projection.

   3. dϕ±                              feν α
        ν ∂a = ∂a near the boundary of Me−ν α .

Let us now define a sequence of almost complex structures J ν ∈ J (β) as follows:
   1. J ν agrees with Jαβ above nb(Mα ).

   2. J ν agrees with (ϕ+     +
                        ν )∗ Jα on nb(Mα ).

   3. J ν agrees with J between nb(Mα ) and nb(MeS α ).

                                               50
  4. J ν agrees with (ϕ−     −
                       ν )∗ Jα on nb(MeS α ).

  5. J ν agrees with Jβα below nb(MeS α ).
Let xk+1 denote the center of B. By the definition of ck+1 (β) given in (10.1), there exists
                                             ν
a sequence of J ν -holomorphic curves uν ∈ MJ (x1 , . . . , xk+1) such that
                                      A+   ν
                                       β (u ) ≤ ck+1 (β) + ǫ.

The SFT compactness theorem implies that, after possibly passing to a subsequence, the
sequence of curves uν converges to a pseudo-holomorphic building u = (u1 , . . . , uN ). Each
ui consists of finitely many pseudoholomorphic nodal punctured finite-energy spheres. In
the following we will ignore nodal points and ghost bubbles and regard each ui as a finite
union of non-constant pseudoholomorphic punctured finite-energy spheres. There exist
indices 1 ≤ r < s < t ≤ N such that
  1. ui is Jβ+ -holomorphic and positively and negatively asymptotic to closed orbits of Rβ
     for all 1 ≤ i < r.
  2. ur is Jαβ -holomorphic and passes through xk+1 . It is positively asymptotic to closed
     orbits of Rβ and negatively asymptotic to closed orbits of Rα .
  3. ui is Jα+ -holomorphic and positively and negatively asymptotic to closed orbits of Rα
     for all r < i < s.
  4. us is J-holomorphic and passes through x1 , . . . , xk . It is positively and negatively
     asymptotic to closed orbits of Rα .
  5. ui is Jα− -holomorphic and positively and negatively asymptotic to closed orbits of Rα
     for all s < i < t.
  6. ut is Jβα -holomorphic. It is positively asymptotic to closed orbits of Rα and negatively
     asymptotic to closed orbits of Rβ .
  7. ui is Jβ− -holomorphic and positively and negatively asymptotic to closed orbits of Rβ
     for all t < i ≤ N.
Note that since every uν has at least one negative end, each ui must have at least one
negative end. Therefore, us ∈ MJ (x1 , . . . , xk ). We need to estimate the positive action
A+α (us ). For every i, the negative asymptotic orbits of ui match the positive asymptotic
orbits of ui+1 . This implies that
                 r−1
                 X                              s−1
                                                X
                       Aβ (ui ) + Aβα (ur ) +           Aα (ui ) + A+          +
                                                                    α (us ) = Aβ (u1 ).
                 i=1                            i=r+1

Since A+    ν                                                     ν
        β (u ) ≤ ck+1 (β) + ε for all ν and the positive ends of u are converging to the
positive ends of u1 , we have
                                  A+ β (u1 ) ≤ ck+1 (β) + ε.


                                                   51
The actions Aβ (ui ) are non-negative for 1 ≤ i < r. Similarly, the actions Aα (ui ) are
non-negative for r < i < s. Since ur passes through the center of the ball B equipped with
the standard complex structure, Aβα (ur ) ≥ A by the monotonicity lemma. Combining the
above estimates yields
                                A+α (us ) + A ≤ ck+1 (β) + ε

as desired.
   In the next result, we determine the value of the first two spectral invariants for Zoll
contact forms.
Proposition 10.3. Let α0 ∈ F (M, ξ) be a Zoll contact form with minimal period sys(α0 ).
Then we have
                             c0 (α0 ) = c1 (α0 ) = sys(α0 ).
Proof. Let α be a C ∞ -small non-degenerate perturbation of α0 . Fix an almost complex
structure J ∈ J (α) and a point x ∈ M. f Our goal is to show that there exists a J-
                         J
holomorphic curve u ∈ M (x) with positive action bounded from above by

                                    A+       3
                                     α (u) ≤ 2 sys(α0 ).

Since J and x are arbitrary, this implies that c1 (α) ≤ 32 sys(α0 ). It then follows from
continuity of c1 that c1 (α0 ) ≤ 23 sys(α0 ). The increasing and the spectrality properties
imply that we must have c0 (α0 ) = c1 (α0 ) = sys(α0 ).
    We construct the pseudoholomorphic curve u by neck-stretching pseudoholomorphic
spheres in a ruled symplectic manifold (X, ω) which contains Mα as a contact type hy-
persurface. The symplectic manifold (X, ω) is constructed by performing a so-called sym-
                                                                   f with R × M by means
plectic cut, see [Ler95], as follows. Identify the symplectization M
                                                         s
of the diffeomorphism ϕα0 , so that ω   e is given by d(e α0 ). Fix a constant S > 0. The
boundary components of the compact region [−S, S] × M in the symplectization are fo-
liated by closed characteristic leaves. The symplectic manifold (X, ω) is obtained from
[−S, S] × M by collapsing each of these leaves to a point. Let B denote the quotient of
M by the free S 1 -action induced by the Zoll Reeb flow. The space X naturally fibers over
B. The fibers are symplectic 2-spheres of the form ([−S, S] × γ)/ ∼ where γ is a closed
Reeb orbit of α0 and ∼ collapses the circles {±S} × γ to points. Under the quotient,
the two boundary components of [−S, S] × M yield smooth symplectic divisors B± of X
both diffeomorphic to B. Clearly, X contains Mα as a contact type hypersurface. The
function (s, p) 7→ − sys(α0 )es on R × M induces a smooth Hamiltonian H on X which
generates a Hamiltonian S 1 -action. The two divisors B± are fixed point sets of this action
and the restriction of the action to the complement X \ (B+ ∪ B− ) is free. In particular,
the assumptions in [McD09, Proposition 4.3] are satisfied. Let F ∈ H2 (X; Z) denote the
homology class of a fiber. Then [McD09, Proposition 4.3] implies that the Gromov–Witten
invariant hptiXF does not vanish. We conclude that, for every ω-compatible almost complex
structure J on X and for every point p ∈ X, there exists a nodal Je-holomorphic sphere
           e
passing through p and representing the homology class F .

                                            52
    Let us now choose a constant S ′ large enough such that the almost complex structure
J ∈ J (α) fixed at the beginning of the proof is adjusted to α on neighbourhoods of the sets
f±±S ′ . Moreover, we assume that the point x strictly lies between the two hypersurfaces
M e    α
Me±S ′ α . We set the constant S appearing in the construction of X to be S := S ′ + ln(3/2).
Since α is C ∞ -close to α0 , the hypersurfaces Me±S ′ α lie between the hypersurfaces Me±S α0
and do not intersect them.
    We define a sequence of compatible almost complex structures J ν on X as follows.
Let Jα± denote the α-adjusted almost complex structure whose restriction to M    f±±S ′ agrees
                                                                                   e
with J. Let X± ⊂ X denote the connected component of X \ Me±S ′ α containing B± .
Let Xb± denote the symplectic completion of X± obtained by attaching the cylindrical end
f∓±S ′ . Let J± be a compatible almost complex structure on X
M                                                                   b± agreeing with J ± in a
  e    α                                                                                α
neighbourhood of the cylindrical end M  f∓±S ′ .
                                           e  α
    Let us fix small tubular neighbourhoods nb(Me±S ′ α ) and sequences of diffeomorphisms

                                 ϕ±   feν α
                                  ν : Me−ν α → nb(Me±S ′ α )

with the following properties:
  1. ϕ±
      ν maps translates of the hypersurface Mα to translates of Mα .

  2. π ◦ ϕ±               f
          ν = π where π : M → M is the canonical projection.

  3. dϕ±                              feν α
       ν ∂a = ∂a near the boundary of Me−ν α .

Let us now define the compatible almost complex structure J ν on X as follows:
  1. J ν agrees with J+ above nb(MeS ′ α ).

  2. J ν agrees with (ϕ+     +
                       ν )∗ Jα on nb(MeS ′ α ).

  3. J ν agrees with J between nb(MeS ′ α ) and nb(Me−S ′ α ).

  4. J ν agrees with (ϕ−     −
                       ν )∗ Jα on nb(Me−S ′ α ).

  5. J ν agrees with J− below nb(Me−S ′ α ).
By the above discussion, we may choose, for every ν, a J ν -holomorphic sphere uν represent-
ing the homology class of the fiber F and passing through the point x ∈ X. By the SFT
compactness theorem, this sequence converges, after possibly passing to a subsequence, to
a pseudoholomorphic building u = (u1, . . . , uN ). Every ui consists of finitely many nodal
punctured finite-energy pseudoholomorphic spheres. As before, we ignore nodes and ghost
bubbles and regard each ui as a finite union of punctured finite energy spheres.
    The homological intersection number F · B± is equal to 1. This implies that every uν
intersects both B± . Therefore, u1 is a J+ -holomorphic curve in X   b+ and its intersection
number with B+ is equal to 1. Similarly, uN is a J− -holomorphic curve in X    b− intersecting
B− with intersection number 1. Since the negative asymptotic limit orbits of ui match the

                                               53
positive asymptotic limit orbits of ui+1 , we conclude that every ui for 1 < i < N must have
both positive and negative asymptotic orbits. Let 1 < r < N be the index such that ur is
a J-holomorphic curve in M  f passing through x ∈ M     f. Since ur has a negative asymptotic
orbit, we have ur ∈ M (x). It remains to estimate A+
                        J
                                                           α (u).
    The complement of B+ in X   b+ is exact. Indeed, restricting the natural Liouville 1-form
e on M
λ     f to the interior of M
                           f−S yields a primitive λ of the symplectic form ω+ on X    b+ \ B+ .
                             e α0
Clearly, the primitive λ does not extend over B+ . In fact, if Σ is a compact oriented
surface, possibly with boundary, and v : Σ → X       b+ is a smooth map sending ∂Σ into the
complement of B+ , then
                           ˆ           ˆ
                                ∗
                               v ω+ =       v ∗ λ + eS sys(α0 )v · B+ .
                            Σ               ∂Σ

In order to make sense of the homological intersection numer v · B+ , we regard v as a
2-chain with fixed boundary in the complement of B+ . Applying this identity to u1 yields
           ˆ                         ˆ
                       −1      ∗
                     (ϕα ◦ u1 ) dα +        u∗1 ω+ = −A−      (u ) + eS sys(α0 ).
                                                        eS ′ α 1
            u−1 f−                            u−1
             1 (M S ′ )
                  e   α                        1 (X+ )



Since u1 is pseudoholomorphic, the left-hand side of this identity is non-negative. We
deduce that
                           ′
                         eS A−          −               S
                             α (u1 ) = AeS ′ α (u1 ) ≤ e sys(α0 ).                (10.2)
                                                             3
It follows from our choice S = S ′ + ln(3/2) that A−
                                                   α (u1 ) ≤ 2 sys(α0 ).
                                                f for 1 < i < r. We have
    The curve ui is a Jα+ -holomorphic curve in M
                                r−1
                                X
                                      Aα (ui ) + A+          −
                                                  α (ur ) = Aα (u1 ).
                                i=2

Since the actions Aα (ui ) are non-negative for 1 < i < r, we deduce that
                                                         3
                                A+          −
                                 α (ur ) ≤ Aα (u1 ) ≤      sys(α0 ).
                                                         2
This concludes the proof of the proposition.
   Next, we compute the first two spectral invariants for contact forms which are C 2 -small,
 1
S -invariant perturbations of a Zoll one.

Proposition 10.4. Let α0 ∈ F (M, ξ) be a Zoll contact form with minimal period sys(α0 ).
If the smooth function S : M → R is C 2 -close to the constant function 1 and S 1 -invariant,
then
                  c0 (Sα0 ) = sys(α0 ) min S,   c1 (Sα0 ) = sys(α0 ) max S.
                                          M                             M




                                                   54
Proof. We abbreviate max S := maxM S and min S := minM S. Our proof is based on the
following claims.
Claim 1. If S is C 2 -close to 1 and the positive number S > max S is close to max S,
then there is a symplectic embedding of the 2n-dimensional ball of symplectic width
                                           fSα0 .
sys(α0 )(max S − min S) into the cobordism M Sα0

Claim 2. If S is C 2 -close to 1 and the positive number S < min S is close to min S,
then there is a symplectic embedding of the 2n-dimensional ball of symplectic width
                                           fSα0 .
sys(α0 )(max S − min S) into the cobordism MSα0

    Before proving these claims, we show how they allow us to determine c0 (Sα0 ) and
                                                       fSα0 , the scaling property and
c1 (Sα0 ). By Claim 1, the packing property applied to MSα0
Proposition 10.3, we have

          c0 (Sα0 ) + sys(α0 )(max S − min S) ≤ c1 (Sα0 ) = S c1 (α0 ) = S sys(α0 ),

and hence, letting S tend to max S, we get

                                  c0 (Sα0 ) ≤ sys(α0 ) min S.

Since the right-hand side is the minimum of the action spectrum of Sα0 , the spectrality
property implies that the above inequality is an equality. Similarly, the packing property
           fSα0 , the scaling property and Proposition 10.3 imply
applied to M Sα0

   c1 (Sα0 ) ≥ c0 (Sα0 ) + sys(α0 )(max S − min S) = S c0 (α0 ) + sys(α0 )(max S − min S).

Letting S tend to min S, we get c1 (Sα0 ) ≥ sys(α0 ) max S. By monotonicity, we have

                        c1 (Sα0 ) ≤ c1 ((max S) α0 ) = sys(α0 ) max S,

and the desired formula for c1 (Sα0 ) follows.
   There remains to prove the above claims. We prove Claim 1, the proof of Claim 2
being analogous. Up to rescaling, we may assume that sys(α0 ) = 1. We identify the
                f with R × M via ϕα0 , so that ω
symplectization M                              e = d(ea α0 ) and
                         Sα0
                        MSα0
                             = {(a, q) ∈ R × M | S(q) < ea < S}.

Denote by p : M → Y the T-bundle, T := R/Z, whose base is the quotient of M by the free
action of the Reeb flow of α0 . Then Y is a (2n − 2)-dimensional manifold with a symplectic
form ω satisfying p∗ ω = dα0 . By means of a Darboux trivializing chart, we can identify a
neighborhood (Uy , ω) of any y in Y with a ball B  bρ of radius ρ > 0 in (Cn−1 , ω
                                                                                 e0 ), where ρ
                                                  −1
is independent of y. Moreover, we can identify p (Uy ) with T × B    bρ in such a way that α0
corresponds to λb0 + dt. Here, t denotes the variable in T and λ b0 is the standard primitive
                                          n−1
of the standard symplectic form ω b0 on C     (we are using the notation from Section 1).

                                              55
    Correspondingly, we obtain an identification of the open subset Uey := R × p−1 (Uy )
of Mf with R × T × B bρ , with respect to which the 1-form ea α0 reads ea (dt + b
                                                                                λ0 ). The
diffeomorphism
                                                       1
                                                                      bρ = U
        Ψy : {(s, t, w) ∈ (0, +∞) × T × Cn−1 | |w| < s 2 ρ} → R × T × B    ey ⊂ M
                                                                                f
                                       1
given by Ψy (s, t, w) := (log s, t, s− 2 w) satisfies

                                                          b0 )) = s dt + λ
                             Ψ∗y (ea α0 ) = Ψ∗y (ea (dt + λ              b0 ,

and hence
                                 Ψ∗y ω
                                     e = Ψ∗y d(ea α0 ) = ds ∧ dt + ω
                                                                   e0 .
Now we choose y ∈ Y so that the T-invariant function S achieves its minimum at the circle
p−1 (y). In the above identifications, we then have S(t, 0) = min S for every t ∈ T,

                   fSα0 ∩ U
                   M      ey = {(a, t, u) ∈ R × T × B
                                                    bρ | S(t, u) < ea < S},
                    Sα0

and hence it is enough to symplectically embed a 2n-ball of symplectic width max S −min S
into the open subset
                                                                  1                     1
V := Ψ−1 fSα0 e                                     n−1
                                                        | S(t, s− 2 w) < s < S, |w| < s 2 ρ}
      y (MSα0 ∩ Uy ) = {(s, t, w) ∈ (0, +∞) × T × C

of ((0, +∞) × T × Cn−1 , ds ∧ dt + ω
                                   b0 ). Since S(t, 0) = min S, we have

                     S(t, u) ≤ min S + ck∇2 SkC 0 |u|2                         bρ ,
                                                                 ∀(t, u) ∈ T × B
                                                                                      1
for a suitable constant c > 0. By this bound, the inequality s > S(t, s− 2 w) appearing in
the representation of V is implied by

                                                        2        |w|2
                                    s > min S + ck∇ Sk      C0        ,
                                                                  s
which is equivalent to
                                                 s                    !
                               1                        4ck∇2 SkC 0 2
                            s > min S       1+       1+            |w| .
                               2                         (min S)2
        √              r
Since       1+r ≤1+    2
                           for every r ≥ 0, the above inequality is implied by
                                                   c
                                 s > min S +           k∇2 SkC 0 |w|2,
                                                 min S
and we conclude that V contains the set

                  V ′ := {(s, t, w) ∈ (0, +∞) × T × Cn−1 | f (|w|2) < s < S},

                                                   56
where                                                                         
                                                 ck∇2 SkC 0 r
                            f (r) := max min S +           r, 2                    .
                                                   min S     ρ
If S is C 2 -close to 1 and S is close to max S, then both S − min S and k∇2 SkC 0 are small
and we can guarantee that
                                                                     
                         f (r) ≤ min S + πr     ∀r ∈ 0, π1 (S − min S) .

This implies that V ′ contains the set
                   
            V ′′ := (s, t, w) ∈ (0, +∞) × T × Cn−1 | min S + π|w|2 < s < S .

Let Φ : Ω → C∗ × Cn−1 be the symplectomorphism defined in (1.1). By (1.3), the sym-
plectomorphism                                              
                         (s, t, w) 7→ Φ s − min S − π, t, w
maps V ′′ onto B(ρ) \ {z1 = 0}, where B(ρ) is the ball of Cn of symplectic width ρ :=
S − min S and {z1 = 0} is a complex hyperplane. By Traynor’s embedding construction
(see Construction 3.2 in [Tra95]), for every ρ′ ∈ (0, ρ), there is a symplectic embedding
Ψ : B(ρ′ ) → B(ρ) \ {z1 = 0}. In the simple case we are considering here, the embedding is
constructed as follows. First, let ψ : D(ρ′ ) → D(ρ) \ {0} be an area preserving embedding
such that
                        π|ψ(ζ)|2 ≤ π|ζ|2 + (ρ − ρ′ ),    ∀ ζ ∈ D(ρ′ ).
Then,
                Ψ(ζ1 , . . . , ζn ) := (ψ(ζ1 ), ζ2 , . . . , ζn ),   (ζ1 , . . . , ζn ) ∈ B(ρ′ )
is the required symplectic embedding. Choosing ρ′ = max S − min S < S − min S = ρ, we
deduce that V ⊃ V ′′ contains a symplectic ball of width max S − min S.


11      Proof of Theorem 5
Let α ∈ F (M, ξ) be C 2 -close to the Zoll contact form α0 ∈ F (M, ξ). By Theorem 3, there
is ϕ ∈ Cont0 (M, ξ) such that
                                        ϕ∗ α = T eh α0 ,
where the functions T and h satisfy the conditions (a)-(e) listed in that theorem. We
denote by Tmin (α) = sys(α) and Tmax (α) the minimum and the maximum of the periods
of the short closed orbits of Rα , which by (e) satisfy

                  Tmin (α) = sys(α0 ) min T,                 Tmax (α) = sys(α0 ) max T.            (11.1)

Then ϕ∗ α = ef α0 with
                                               f := h + log T,



                                                        57
and by property (c) of Theorem 3 we have

                                                              Tmax (α)
                                max f − min f = log                    .
                                 M           M                Tmin(α)
This shows that
                                                        Tmax (α)
                                     d(α0 , α) ≤ log             .
                                                        Tmin (α)
In order to conclude our proof, we need to prove the reverse inequality
                                                        Tmax (α)
                                     d(α0 , α) ≥ log             .                   (11.2)
                                                        Tmin (α)
We shall deduce this from the following claim.
Claim. c0 (α) = Tmin (α) and c1 (α) = Tmax (α).
    Postponing its proof, we show how this claim implies (11.2). Let ψ ∈ Cont0 (M, ξ) and
let g be the smooth positive function such that

                                            ψ ∗ α = eg α0 .

Denoting by
                     a := eminM g = min eg ,           b := emaxM g = max eg ,
                                        M                                  M

we have
                                       aα0 ≤ ψ ∗ α ≤ bα0 .
Using Proposition 10.3, the scaling, monotonicity and invariance properties of the spectral
invariants, and the above claim, we compute

                    a sys(α0 ) = c0 (a α0 ) ≤ c0 (ψ ∗ α) = c0 (α) = Tmin (α),

and
                    b sys(α0 ) = c1 (b α0 ) ≥ c1 (ψ ∗ α) = c1 (α) = Tmax (α).
Therefore
                                                       b       Tmax (α)
                          max g − min g = log            ≥ log          .
                            M          M               a       Tmin (α)
Since the contactomorphism ψ ∈ Cont0 (M, ξ) was arbitrary, the above inequality implies
(11.2).
   There remains to prove the claim. In the notation from the proof of Theorem 3, we
have
                                  v ∗ u∗ α = Sα0 + η.
By invariance, this implies that

                                      ck (α) = ck (Sα0 + η),

                                                  58
where ck is viewed as a function on F (M, ker(Sα0 +η)) on the right hand side. For t ∈ [0, 1],
we define
                                       γt := Sα0 + tη.
Since S is C 1 -close to 1 and η and dη are both C 0 -small, γt is a contact form for all t.
Moreover, γt and dγt are C 0 -close to α0 and dα0 , respectively.
Claim. The short closed Reeb orbits of γt and their periods are independent of t ∈ [0, 1].
Proof. Let us split the Reeb vector field Rγt of γt into a component Xt parallel to Rα0
and a component Yt tangent to ξ = ker α0 . Since γt and dγt are C 0 -close to α0 and dα0 ,
the component Xt is C 0 -close to Rα0 and the component Yt is C 0 -small. Let us write
Xt = at Rα0 for some function at which is C 0 -close to 1. Then a direct computation shows
that
                               ιXt dγt = at (−dS + tF [dS])
and therefore
                                  ιYt dγt = at (dS − tF [dS]).
Recall that dS = −B[V ] where B : ker α0 → (ker α0 )∗ is C 0 -close to the isomorphism
induced by dα0 and V is a section of ker α0 which is C 1 -small. Let Bt : ker α0 → (ker α0 )∗
denote the isomorphism induced by dγt , which is also C 0 -close to the isomorphism induced
by dα0 . We have
                        Yt = at Bt−1 (B[V ] − tF [B[V ]]) = at Qt [V ]
where Qt is a suitable isomorphism of ker α0 which is C 0 -close to the identity. We may
therefore write Rγt = at (Rα0 + Qt [V ]). The argument in the proof of Proposition 4.3 (ii)
shows that the short closed Reeb orbits of γt are precisely the zero set of V or equivalently
the critical set of S. The action of such an orbit γ is simply given by S(γ) sys(α0 ) and in
particular independent of t.
   The above claim shows that the short action spectrum of γt is independent of t. The
function t 7→ ck (γt ) for k ∈ {0, 1} is continuous and takes values in this short action
spectrum. Since the action spectrum has measure zero, this implies that t 7→ ck (γt ) is
constant for k ∈ {0, 1}. In particular, we obtain

                          c0 (α) = c0 (Sα0 ) and c1 (α) = c1 (Sα0 ).

Next, we define βt := (1 − t)Sα0 + tα0 . Since S is C 1 -close to 1, this is a contact form for
all t ∈ [0, 1].
Claim. The short closed Reeb orbits of βt are independent of t ∈ (0, 1]. If γ is such a short
closed Reeb orbit, then its action with respect to βt is given by ((1 − t)S(γ) + t) sys(α0 ).
Proof. The proof of this claim is similar to the proof of the claim above.




                                              59
    It follows from continuity of the spectral invariants, the above claim and the fact that
action spectra have measure zero that

                      ck (βt ) = (1 − t)ck (Sα0 ) + t sys(α0 ) for k ∈ {0, 1}.

If t > 0 is sufficiently small, then (1 − t)S + t is C 2 -small. Therefore, we may apply
Proposition 10.4 to the contact form βt for t > 0 small. We conclude that

      c0 (βt ) = min((1 − t)S + t) sys(α0 ) and c1 (βt ) = max((1 − t)S + t) sys(α0 ).
                 M                                                                M

In combination with the above expression for ck (βt ), this implies

     c0 (Sα0 ) = min S sys(α0 ) = Tmin (α) and c1 (Sα0 ) = max S sys(α0 ) = Tmax (α).
                  M                                                                    M

This concludes the proof of the claim.


A      Appendix: Construction of exotic ellipsoids
The statements. Denote by α0 = λ0 |S 3 the standard contact form on the 3-sphere, see
(6), by ξst = ker α0 the induced co-oriented contact structure, and by F (S 3 , ξst) the space
of contact forms on S 3 which define ξst . This space can be identified with the space of
positive smooth functions on S 3 and is endowed with the C ∞ -topology. If (a, b) is a pair
of positive real numbers, we denote by E(a, b) the open ellipsoid
                                 n                 π        π          o
                       E(a, b) := (z1 , z2 ) ∈ C2 | |z1 |2 + |z2 |2 < 1 ,
                                                   a        b
and by εa,b ∈ F (S 3, ξst ) the contact form which is obtained by pulling the restriction of λ0
to ∂E(a, b) back to S 3 by the radial projection, i.e.
                             π              π        −1
                ǫa,b (z) =        |z1 |2 +     |z2 |2     α0 (z),           ∀z = (z1 , z2 ) ∈ S 3 .
                              a              b
The Reeb flow of εa,b has the form
                                                          2π            2π        
                                    φtεa,b (z1 , z2 ) = e a ti z1 , e    b
                                                                           ti
                                                                                z2 .

The circles Γ1 := {z2 = 0} ⊂ S 3 and Γ2 := {z1 = 0} ⊂ S 3 are closed Reeb orbits of εa,b of
period a and b, respectively. The Reeb flow of εa,b preserves the 2-tori
                                                                     
                    {|z1 | = cos β, |z2 | = sin β} ⊂ S 3 , ∀β ∈ 0, π2 ,

on which the flow is periodic if and only if the number ab is rational.
   The first aim of this Appendix is to present a self-contained proof of the following result.


                                                       60
Theorem A.1. Let (a0 , b0 ) be a pair of positive numbers. For every neighborhood U of
(a0 , b0 ) in (0, +∞)2 and every neighborhood U of εa0 ,b0 in F (S 3 , ξst) there exist (a, b) ∈ U,
a contact form α in U and a sequence (ϕj ) in Cont0 (S 3 , ξst ), each having support away
from Γ1 ∪ Γ2 , such that:

  (i) (ϕ∗j εa,b ) converges to α in the C ∞ -topology;

 (ii) the only closed Reeb orbits of α are Γ1 and Γ2 , with periods a and b, respectively;

(iii) the Reeb flow of α has a dense orbit.

    By (iii), the Reeb flow of α is not conjugate to the one of εa,b . Together with (i), this
implies that the orbit of εa,b by the action of Cont0 (S 3 , ξst ) is not C ∞ -closed in F (S 3, ξst ),
see Example 2 in the Introduction. The above result can be deduced from Theorem A in
[Kat73], where Katok used the conjugacy method which he had introduced together with
Anosov in [AK70a] to construct ergodic Hamiltonian flows arbitrarily close to suitably
integrable flows. Therefore, (iii) can be replaced by the stronger condition:

(iii’) the Reeb flow of α is ergodic with respect to the invariant measure induced by the
       volume form α ∧ dα.

Specializing the argument to the perturbation of εa0 ,b0 and showing just the existence of
a dense orbit, instead of ergodicity, allows us to give here a short self-contained proof,
still based on the Anosov-Katok conjugacy method. See also [CS16] for a reformulation of
Katok’s result in the Reeb setting.

Remark A.2. In the proof of Theorem A.1, the pair (a, b) is obtained as a limit of pairs
of positive numbers (aj , bj ) with rational ratio which is required to converge extremely fast
and we do not control which kind of limiting pairs (a, b) we can produce. It is actually
possible to strengthen the above result by prescribing the pair (a, b): For any pair of
positive numbers (a, b) whose ratio is irrational and not Diophantine, there exist a contact
form α ∈ F (S 3 , ξst ) and a sequence (ϕj ) in Cont0 (S 3 , ξst ) which satisfy (i), (ii) and (iii)
(or even (iii’)) with respect to (a, b). This follows from a corresponding result for area-
preserving diffeomorphisms of the 2-disk which is proven in [FS05] and by a construction
from [AGZ22], which allows to lift such diffeomorphisms to Reeb flows on (S 3 , ξst ).
    The condition on (a, b) is optimal. Indeed, if the ratio ab is rational, then any contact
form α which satisfies (i) (even just with C 0 -convergence) is smoothly conjugate to εa,b ,
as discussed in Example 1 in the Introduction, so cannot satisfy (iii). If the ratio ab is
irrrational and Diophantine and the contact form α satisfies (i), then the first-return map
to a local Poincaré section transverse to Γ1 of the Reeb flow of α is an area-preserving
diffeomorphism with an elliptic and Diophantine fixed point. In his “last geometric theo-
rem”, Herman proved that arbitrarily near to such a fixed point there are smooth invariant
circles, see [FK09, Theorem 4], and hence the Reeb flow of α has invariant tori. Since these
tori disconnect S 3 , α cannot satisfy (iii).


                                                  61
   Writing the contact form α of Theorem A.1 as α = f 2 α0 for a suitable smooth positive
function on S 3 , we consider the starshaped domain A ⊂ C2 which is given by

                                 A := {rz | z ∈ S 3 , 0 ≤ r < f (z)}.

The pullback by the radial projection S 3 → ∂A of λ0 |∂A to S 3 is then α. We can comple-
ment Theorem A.1 with the following result.

Theorem A.3. The domains A and E(a, b) are symplectomorphic but the contact forms
which are given by the restrictions of λ0 to their boundaries are not conjugate.

    The first examples of pairs of starshaped domains in Cn , n > 1, having the above
properties were constructed by Eliashberg and Hofer in [EH96]. The construction we
present here is somehow simpler, but builds on some results which are known to hold only
in dimension four.

Proof of theorem A.1. By slightly perturbing (a0 , b0 ), we may assume that the ratio ab00
is rational. Starting from the pair (a0 , b0 ) and from ϕ0 = id, we shall inductively construct
                               a
sequences (aj , bj ) in U with bjj ∈ Q and (ϕj ) in Cont0 (S 3 , ξst ) such that (aj , bj ) converges
to some (a, b) in U and the sequence of contact forms

                                               αj := ϕ∗j εaj ,bj

converges to some contact form α in U. Assuming that (aj , bj ) and ϕj have been chosen,
we choose ϕj+1 of the form
                                   ϕj+1 := ψj ◦ ϕj ,
for some ψj in Cont0 (S 3 , ξst ) satisfying ψj∗ εaj ,bj = εaj ,bj whose properties are discussed
below. With these choices, we have for every (a′ , b′ ) ∈ (0, +∞)2

                    ϕ∗j+1 εa′ ,b′ − ϕ∗j εaj ,bj = ϕ∗j+1εa′ ,b′ − (ψj−1 ◦ ϕj+1 )∗ εaj ,bj
                                                                                               (A.1)
                    = ϕ∗j+1 εa′ ,b′ − ϕ∗j+1 ((ψj−1 )∗ εaj ,bj ) = ϕ∗j+1(εa′ ,b′ − εaj ,bj ).

Once ψj , and hence ϕj+1, has be chosen, the above identity shows that by choosing
(aj+1 , bj+1 ) close to (aj , bj ) we can make the C k -norm of

                                αj+1 − αj = ϕ∗j+1 (εaj+1 ,bj+1 − εaj ,bj )

arbitrarily small. Therefore, we can ensure the convergence of (aj , bj ) to some (a, b) ∈ U
and of (αj ) to some α ∈ U in the C ∞ -topology. Moreover, (A.1) shows that by making the
convergence of (aj , bj ) fast enough and by a diagonal argument we can also ensure that the
sequence (ϕ∗j εa,b ) converges to α, as claimed in (i).
   Next, we remark that any open condition which is satisfied by αj can be transmitted
to α by making the convergence of (aj , bj ) sufficiently fast. More precisely: if Vj is some
open neighborhood of αj in F (S 3 , ξst), we can fix a closed neighborhood Wj ⊂ Vj of αj and

                                                      62
make all the subsequent choices of (ai , bi ) in such a way that for every i > j the contact
form αi belongs to Wj . By doing this, we can ensure that also the limiting contact form α
belongs to Vj .
     We now discuss the choice of ψj . By choosing each ψj to be supported away from the
link Γ := Γ1 ∪ Γ2 , we can ensure that α agrees up to every order with εa,b on Γ, and hence
Γ1 and Γ2 are closed Reeb orbits of α of period a and b, respectively.
             a      p
     Write bjj = qjj , where pj and qj are coprime natural numbers. By choosing (aj+1 , bj+1 )
close to (aj , bj ), but not equal to it, we have that the sequences (pj ) and qj ) diverge.
     We claim that if the convergence of (aj , bj ) is sufficiently fast then Γ1 and Γ2 are the
only closed Reeb orbits of α. Let Vj be a sequence of open neighborhoods of Γ whose
intersection is Γ. Since the Reeb orbits of αj other than Γ1 and Γ2 are all closed with
period pj bj = qj aj , the Reeb flow of αj has no closed orbits of period τ ≤ τj := pj bj − 1
which pass through S 3 \ Vj . The set of contact forms with the latter property is C 1 -open,
so by the remark above we can guarantee that the Reeb flow of α has no closed orbits of
period τ ≤ τj which pass through S 3 \ Vj , for every j ∈ N. Since the sequence (Vj ) shrinks
to Γ and (τj ) diverges, Γ1 and Γ2 are the only two closed Reeb orbits of α, as claimed in
(ii).
     Fix some infinitesimal sequence (ǫj ) of positive numbers. We claim that by a suitable
choice of the defining data we can ensure that for every j ≥ 1 the Reeb flow of αj has an
orbit which is ǫj -dense. Here, a subset of S 3 is said to be ǫ-dense if it meets any open ball
of radius ǫ, with respect to some fixed metric d on S 3 .
     Indeed, once this is achieved we can argue as follows. Having an ǫj -dense orbit is a
C -open condition in the space of smooth vector fields, and hence a C 1 -open condition
   0

in F (S 3 , ξst). Therefore, the above property of each αj can be transmitted to α, whose
Reeb flow then has an ǫj -dense orbit for every j. When non-empty, the set Oj of ǫj -dense
orbits of a flow is open and ǫj -dense. Using that ǫj → 0, a standard argument, which is
analogous to the proof of Baire’s theorem, shows that ∩k∈N Ojk is non-empty for a suitable
subsequence (jk ). Every element of this intersection has a dense orbit and, therefore, α
satisfies (iii).
     There remains to prove the following claim, in which ǫ is an arbitrary positive number:
Once ϕj and (aj , bj ) have been chosen, by a suitable choice of the diffeomorphism ψj , which
we recall is required to preserve the contact form εaj ,bj and to be supported away from Γ,
and by choosing (aj+1, bj+1 ) sufficiently close to (aj , bj ), we can make sure that the Reeb
flow of
                                      αj+1 = ϕ∗j (ψj∗ εaj+1 ,bj+1 )
has an orbit which is ǫ-dense. The proof of this fact uses the following lemma.

Lemma A.4. Let (a, b) be a pair of positive numbers with rational ratio. Let (γ1 , . . . , γk )
and (γ1′ , . . . , γk′ ) be k-tuples of distinct Reeb orbits of εa,b in S 3 \ Γ. Then there exists
ψ ∈ Cont0 (S 3 , ξst) which preserves the contact form εa,b , is supported away from Γ and
satisfies ψ(γj ) = γj′ for every j = 1, . . . , k.

   Postponing the proof of this lemma, we show how it implies the above claim. Note that

                                               63
the Reeb flow of αj+1 has the form

                                  φtαj+1 = ϕ−1   −1   t
                                            j ◦ ψj ◦ φεa                 ◦ ψj ◦ ϕj ,                 (A.2)
                                                             j+1 ,bj+1


where φtεa               denotes the Reeb flow of εaj+1 ,bj+1 . Let δ > 0 be such that
             j+1 ,bj+1


                              d(z, z ′ ) < δ   ⇒         d ϕ−1      −1 ′
                                                            j (z), ϕj (z )) < ǫ,                     (A.3)

and let A ⊂ S 3 \ Γ be a finite set which is 2δ -dense in S 3 . Fix some invariant torus T in
S 3 \ Γ, such as for instance
                                
                            T := (z1 , z2 ) ∈ S 3 | |z1 |2 = |z2 |2 = 21 .

Since this torus consists of infinitely many Reeb orbits of εaj ,bj , by the above lemma there
exists ψj ∈ Cont0 (S 3 , ξst) which preserves εaj ,bj and is supported away from Γ such that
ψj−1 (T ) contains the finite set A, and is therefore 2δ -dense in S 3 .
    Now let σ > 0 be such that

                             d(z, z ′ ) < σ    ⇒         d ψj−1 (z), ψj−1 (z ′ )) < 2δ .             (A.4)
                                                                                   ′
Let (a′ , b′ ) be a pair of positive numbers with rational ratio ab′ = pq , where p and q are
coprime natural numbers. Each Reeb orbit of εa′ ,b′ in T is σ-dense, provided that p and q
are large enough. Therefore, by choosing (aj+1 , bj+1 ) with the already required properties
and sufficiently close to (aj , bj ), we obtain that each Reeb orbit of εaj+1 ,bj+1 is σ-dense in
T.
   Let z be a point in ϕ−1       −1
                           j ◦ψj (T ) and set w := ψj ◦ϕj (z) ∈ T . Then the orbit of w by the
Reeb flow of εaj+1 ,bj+1 is σ-dense in T . By (A.4), the image of this orbit by ψj−1 is 2δ -dense
in ψj−1 (T ) and hence δ-dense in S 3 . Together with (A.2) and (A.3), we conclude that the
orbit of z by the Reeb flow of αj+1 is ǫ-dense. This concludes the proof of Theorem A.1.
Proof of Lemma A.4. Since ab is rational, the Reeb flow of εa,b defines a free S 1 -action on
S 3 \ Γ. The orbit space of this action is a smooth surface Σ which is endowed with a
symplectic form ω satisfying π ∗ ω = dεa,b , where

                                               π : S3 \ Γ → Σ

denotes the quotient projection. Let (p1 , . . . , pk ) and (p′1 , . . . , p′k ) be the k-tuples of distinct
points in Σ such that

                            γj = π −1 (pj ),    γj′ = π −1 (p′j )        ∀j = 1, . . . , k.

It is easy to show that the group of compactly supported Hamiltonian diffeomorphisms of
(Σ, ω) acts k-transitively: There exists a smooth function H : [0, 1] × Σ → R with compact
support such that the time-1 map of the induced Hamiltonian flow maps pj to p′j , for every
j = 1, . . . , k (see e.g. [Ban97, p. 109]). Lift H to a smooth function K : [0, 1] × S 3 → R

                                                       64
supported away from Γ and let X be the contact vector field on S 3 which is induced by
the Hamiltonian K and the contact form εa,b , i.e., the unique vector field satisfying

                       ıX dεa,b = ıRεa,b dK)εa,b − dK,          ıX εa,b = K,

where Rεa,b denote the Reeb vector field of εa,b . The flow of X consists of contactomor-
phisms of (S 3 , ξst ) preserving εa,b which are supported away from Γ and whose time-1 map
ψ maps γj to γj′ , for every j = 1, . . . , k.

Proof of theorem A.3. By composition with the radial projections, a diffeomorphism
from ∂E(a, b) to ∂A which intertwines the restrictions of λ0 would induce a smooth con-
jugacy between εa,b and α on S 3 , which cannot exist because the Reeb flow of α has a
dense orbit while the one of εa,b does not. So we only have to prove that E(a, b) and A are
symplectomorphic.
   Set for simplicity E := E(a, b). Write ϕ∗j εa,b = fj2 α0 and set

                             Aj := {rz | z ∈ S 3 , 0 ≤ r < fj (z)}.

By composing the contactomorphism ϕ−1         3    3
                                        j : S → S , which isisotopic to the identity, with
the radial projections ∂E → S 3 and S 3 → ∂Aj , we obtain a diffeomorphism from ∂E to
∂Aj which intertwines the restrictions of λ0 . The positively 1-homogeneous extension of
this map is a symplectomorphism of C2 \ {0} isotopic to the identity via such maps, and
mapping rE \ {0} to rAj \ {0}, for every r > 0. Repeating the argument at the end of the
proof of Theorem 1 (ii) in Section 6, this map can be smoothed near the origin producing
a global symplectomorphism φj : C2 → C2 such that

                                  φj (rE) = rAj        ∀r ≥ 21 .

The C 0 -convergence of (fj ) to f implies that, up to passing to a subsequence, we can find
a strictly increasing sequence (rj ) in the interval ( 21 , 1) which converges to 1 and satisfies
                                                      [
                             rj Aj ⊂ rj+1 Aj+1,             rj Aj = A.                     (A.5)
                                                      j≥0


Starting with ψ0 := φ0 and arguing inductively, we wish to construct symplectomorphisms
ψj : C2 → C2 such that for every j ≥ 0:

                                  ψj+1 = ψj         on rj E,                               (A.6)
                                 ψj+1 = φj+1        on C2 \ rj+1E.                         (A.7)

Once ψ0 , . . . , ψj have been constructed, the construction of ψj+1 goes as follows. The
symplectomorphism θ := φ−1                                                  −1
                                 j+1 ◦ ψj maps rj E to the domain φj+1 (rj Aj ), whose closure
satisfies
                       φ−1              −1              −1
                        j+1 (rj Aj ) = φj+1 (rj Aj ) ⊂ φj+1 (rj+1 Aj+1 ) = rj+1 E.


                                               65
Since the space of symplectic embeddings between ellipsoids is connected, see [McD09], we
can find a symplectic isotopy θt : C2 → C2 such that θ0 = id, θ1 = θ and

                             θt (rj E) ⊂ rj+1 E     ∀t ∈ [0, 1].

Let Ht : C2 → R be a smooth path of Hamiltonians generating θtS  . Let χ : C2 → R be a
smooth function with support in rj+1 E and taking the value 1 on t∈[0,1] θt (rj E). Denote
by η : C2 → C2 the time-1 map of the Hamiltonian isotopy induced by the non-autonomous
Hamiltonian χH. Then

                 η = θ1 = φ−1
                           j+1 ◦ ψj   on rj E,     η = id on C2 \ rj+1 E,

and hence ψj+1 := φj+1 ◦ η satisfies (A.6) and (A.7).
   Note that by (A.7) we have

                           ψj (rE) = rAj    ∀r ≥ rj ,     ∀j ≥ 0.                   (A.8)

By (A.6), the sequence ψj (z) stabilizes for every z ∈ E, and hence the map

                           ψ : E → C2 ,      ψ(z) := lim ψj (z)
                                                        j→∞


is a well defined symplectic embedding, which by (A.5) and (A.8) has image A. The maps
ψ is then the required symplectomorphism from E(a, b) to A.


References
[AB23]     A. Abbondandolo and G. Benedetti, On the local systolic optimality of Zoll
           contact forms, Geom. Funct. Anal. 33 (2023), 299–363.

[ABHS18] A. Abbondandolo, B. Bramham, U. L. Hryniewicz, and P. A. S. Salomão,
         Sharp systolic inequalities for Reeb flows on the three-sphere, Invent. Math.
         211 (2018), 687–778.

[AK22]     A. Abbondandolo and J. Kang, Symplectic homology of convex domains and
           Clarke’s duality, Duke Math. J. 171 (2022), 739–830.

[AGZ22]    P. Albers, H. Geiges, and K. Zehmisch, Pseudorotations of the 2-disk and Reeb
           flows on the 3-sphere, Ergodic Theory Dynam. Systems 42 (2022), 402–436.

[AK70a]    D. V. Anosov and A. B. Katok, New examples in smooth ergodic theory. Ergodic
           diffeomorphisms, Trudy Moskov. Mat. Obšč. 23 (1970), 3–36.

[AK70b]    D. V. Anosov and A. B. Katok, New examples of ergodic diffeomorphisms of
           smooth manifolds, Uspehi Mat. Nauk 25 (1970), 173–174.


                                             66
[Ban86]    V. Bangert, On the length of closed geodesics on almost round spheres, Math.
           Z. 191 (1986), 549–558.
[Ban97]    A. Banyaga, The structure of classical diffeomorphism groups, Kluwer Aca-
           demic Publishers Group, Dordrecht, 1997.
[BBLM23] L. Baracco, O. Bernardi, C. Lange, and M. Mazzucchelli, On the local maxi-
         mizers of higher capacity ratios, arXiv:2303.13348 [math.SG], 2023.
[BP94]     M. Bialy and L. Polterovich, Geodesics of Hofer’s metric on the group of Hamil-
           tonian diffeomorphisms, Duke Math. J. 76 (1994), 273–292.
[Bot80]    M. Bottkol, Bifurcation of periodic orbits on manifolds and Hamiltonian sys-
           tems, J. Differential Equations 37 (1980), 12–22.
[BEH+ 03] F. Bourgeois, Y. Eliashberg, H. Hofer, K. Wysocki, and E. Zehnder, Compact-
          ness results in Symplectic Field Theory, Geom. Topol. 7 (2003), 799–888.
[CS16]     R. Casals and O. Spáčil, Chern-Weil theory and the group of strict contacto-
           morphisms, J. Topol. Anal. 8 (2016), 59–87.
[CHLS07] K. Cieliebak, H. Hofer, J. Latschev, and F. Schlenk, Quantitative symplectic
         geometry, Dynamics, Ergodic Theory and Geometry, Mathematical Sciences
         Research Institute Publications, no. 54, Cambridge University Press, 2007,
         pp. 1–44.
[CGH23]    D. Cristofaro-Gardiner and R. Hind, On the agreement of symplectic capacities
           in higher dimension, arXiv:2307.12125 [math.SG], 2023.
[CGM20]    D. Cristofaro-Gardiner and M. Mazzucchelli, The action spectrum characterizes
           closed contact 3-manifolds all of whose Reeb orbits are closed, Comment. Math.
           Helv. 95 (2020), 461–481.
[Edt22a]   O. Edtmair, Disk-like surfaces of section and symplectic capacities,
           arXiv:2206.07847 [math.SG], 2022.
[Edt22b]   O. Edtmair, An elementary alternative to PFH spectral invariants,
           arXiv:2207.12553 [math.SG], 2022.
[EH89]     I. Ekeland and H. Hofer, Symplectic topology and Hamiltonian dynamics, Math.
           Z. 200 (1989), 355–378.
[EH90]     I. Ekeland and H. Hofer, Symplectic topology and Hamiltonian dynamics II,
           Math. Z. 203 (1990), 553–567.
[EH96]     Y. Eliashberg and H. Hofer, Unseen symplectic boundaries, Manifolds and ge-
           ometry (Pisa, 1993), Sympos. Math., XXXVI, Cambridge Univ. Press, Cam-
           bridge, 1996, pp. 178–189.

                                           67
[FK09]     B. Fayad and R. Krikorian, Herman’s last geometric theorem, Ann. Sci. Éc.
           Norm. Supér. (4) 42 (2009), 193–219.

[FS05]     B. Fayad and M. Saprykina, Weak mixing disc and annulus diffeomorphisms
           with arbitrary Liouville rotation number on the boundary, Ann. Sci. École Norm.
           Sup. (4) 38 (2005), 339–364.

[FHW94]    A. Floer, H. Hofer, and K. Wysocki, Applications of symplectic homology I,
           Math. Z. 217 (1994), 577–606.

[FM15]     D. Frenkel and D. Müller, Symplectic embeddings of 4-dim ellipsoides into cubes,
           J. Symplectic Geom. 13 (2015), 765–847.

[Gei08]    H. Geiges, An introduction to contact topology, Cambridge Studies in Advanced
           Mathematics, vol. 109, Cambridge University Press, Cambridge, 2008.

[Gro85]    M. Gromov, Pseudo holomorphic curves in symplectic manifolds, Invent. Math.
           82 (1985), 307–347.

[GH18]     J. Gutt and M. Hutchings, Symplectic capacities from positive S 1 -equivariant
           symplectic homology, Algebr. Geom. Topol 18 (2018), 3537–3600.

[GHR22]    J. Gutt, M. Hutchings, and V. G. B. Ramos, Examples around the strong
           Viterbo conjecture, J. Fixed Point Theory appl. 24 (2022), Paper No. 41, 22.

[GR23]     J. Gutt and V. G. B. Ramos, Characterizing symplectic capacities on ellipsoids,
           arXiv:2312.06476 [math.SG], 2023.

[Ham02]    U. Hamenstädt, Examples for nonequivalence of symplectic capacities,
           arXiv:math/0209052 [math.SG], 2002.

[Her98a]   M. Herman, Some open problems in dynamical systems, Proceedings of the
           International Congress of Mathematicians, Vol. II (Berlin, 1998), 1998, pp. 797–
           808.

[Her98b]   D. Hermann, Non-equivalence of symplectic capacities for open sets with re-
           stricted contact type boundary, Prépublication d’Orsay numéro 32, 1998.

[Her04]    D. Hermann, Inner and outer Hamiltonian capacities, Bull. Soc. Math. France
           132 (2004), 509–541.

[Hof90]    H. Hofer, On the topological properties of symplectic maps, Proc. Roy. Soc.
           Edinburgh Sect. A 115 (1990), 25–38.

[HWZ98]    H. Hofer, K. Wysocki, and E. Zehnder, The dynamics on three-dimensional
           strictly convex energy surfaces, Ann. of Math. 148 (1998), 197–289.



                                            68
[HZ90]     H. Hofer and E. Zehnder, A new capacity for symplectic manifolds, Analysis,
           et cetera, Academic Press, Boston, MA, 1990, pp. 405–427.

[Hut11]    M. Hutchings, Quantitative embedded contact homology, J. Differential Geom.
           88 (2011), 231–266.

[Hut14]    M. Hutchings, Lecture notes on embedded contact homology, Contact and sym-
           plectic topology, János Bolyai Math. Soc., Budapest, 2014, pp. 389–484.

[Hut22a]   M. Hutchings, An elementary alternative to ECH capacities, Proc. Natl. Acad.
           Sci. U.S.A. (35) 119 (2022), e2203090119.

[Hut22b]   M. Hutchings, Elementary spectral invariants and quantitative closing lemmas
           for contact three-manifolds, arXiv:2208.01767 [math.SG], 2022.

[Iri22]    K. Irie, Symplectic homology of fiberwise convex sets and homology of loop
           spaces, J. Symplectic Geom. 20 (2022), 417–470.

[Kat73]    A. Katok, Ergodic perturbations of degenerate integrable Hamiltonian systems,
           Izv. Akad. Nauk SSSR Ser. Mat. 37 (1973), 539–576.

[LM95]     F. Lalonde and D. McDuff, Hofer’s L∞ -geometry: energy and stabiliy of Hamil-
           tonian flows, part II, Invent. Math. 122 (1995), 35–69.

[LS94]     F. Laudenbach and J.-C. Sikorav, Hamiltonian disjunction and limits of La-
           grangian submanifolds, Internat. Math. Res. Notices (1994), no. 4, 161 ff., ap-
           prox. 8 pp.

[Ler95]    E. Lerman, Symplectic cuts, Math. Res. Lett. 2 (1995), 247–258.

[MR23]     M. Mazzucchelli and M. Radeschi, On the structure of Besse convex contact
           spheres, Trans. Amer. Math. Soc. 376 (2023), 2125–2153.

[McD09]    D. McDuff, Symplectic embeddings of 4-dimensional ellipsoids, J. Topol. 2
           (2009), 1–22.

[McD11]    D. McDuff, The Hofer conjecture on embedding symplectic ellipsoids, J. Differ-
           ential Geom. 88 (2011), 519–532.

[MS17]     D. McDuff and D. Salamon, Introduction to symplectic topology, third ed., Ox-
           ford Mathematical Monographs, The Clarendon Press Oxford University Press,
           New York, 2017.

[MS23]     D. McDuff and K. Siegel, Symplectic capacities, unperturbed curves, and convex
           toric domains, Geom. Topol., to appear.

[Mel21]    T. Melistas, The large scale geometry of overtwisted contact forms, Ph.D. thesis,
           University of Georgia, 2021.

                                            69
[Ost14]    Y. Ostrover, When symplectic topology meets Banach space geometry, Proceed-
           ings of the International Congress of Mathematicians—Seoul 2014. Vol. II,
           Kyung Moon Sa, Seoul, 2014, pp. 959–981.

[PRSZ20] L. Polterovich, D. Rosen, K. Samvelyan, and J. Zhang, Topological persistence
         in geometry and analysis, University Lecture Series, vol. 74, American Mathe-
         matical Society, Providence, RI, 2020.

[RZ21]     D. Rosen and J. Zhang, Relative growth rate and contact Banach-Mazur dis-
           tance, Geom. Dedicata 215 (2021), 1–30.

[Sağ21]   M. Sağlam, Contact forms with large systolic ratio in arbitrary dimensions,
           Ann. Scuola Norm. Sup. Pisa Cl. Sci. (4) 22 (2021), 1265–1308.

[Sch02]    F. Schlenk, An extension theorem in symplectic geometry, Manuscripta Math.
           109 (2002), 329–348.

[Sch18]    F. Schlenk, Symplectic embedding problems, old and new, Bull. Amer. Math.
           Soc. (N.S.) 55 (2018), 139–182.

[Sul76]    D. Sullivan, Cycles for the dynamical study of foliated manifolds and complex
           manifolds, Invent. Math. 36 (1976), 225–255.

[SZ21]     V. Stojisavljević and J. Zhang, Persistence modules, symplectic Banach-Mazur
           distance and Riemannian metrics, Internat. J. Math. 32 (2021), Paper No.
           2150040, 76.

[Tra95]    L. Traynor, Symplectic packing constructions, J. Differential Geom. 42 (1995),
           411–429.

[Ush22]    M. Usher, Symplectic Banach-Mazur distances between subsets of Cn , J. Topol.
           Anal. 14 (2022), 231–286.

[Vit87]    C. Viterbo, A proof of the Weinstein conjecture in R2n , Ann. Inst. H. Poincaré
           Anal. Non Linéaire 4 (1987), 337–356.

[Vit89]    C. Viterbo, Capacité symplectiques et applications, Astérisque 177-178 (1989),
           no. 714, Séminaire Bourbaki 41éme année, 345–362.

[Vit92]    C. Viterbo, Symplectic topology as the geometry of generating functions, Math.
           Ann. 292 (1992), 685–710.

[Vit00]    C. Viterbo, Metric and isoperimetric problems in symplectic geometry, J. Amer.
           Math. Soc. 13 (2000), 411–431.




                                           70
