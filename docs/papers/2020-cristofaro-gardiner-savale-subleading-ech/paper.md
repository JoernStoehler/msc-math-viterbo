---
source: arXiv:1811.00485
fetched: 2025-10-20
---
# Sub-leading asymptotics of ECH capacities

                                          SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES

                                                    DAN CRISTOFARO-GARDINER AND NIKHIL SAVALE
arXiv:1811.00485v1 [math.SG] 1 Nov 2018




                                                 Abstract. In previous work [11], the first author and collabora-
                                                 tors showed that the leading asymptotics of the embedded contact
                                                 homology (ECH) spectrum recovers the contact volume. Our main
                                                 theorem here is a new bound on the sub-leading asymptotics.




                                                                      1. Introduction
                                          1.1. The main theorem. Let Y be a closed, oriented three-manifold.
                                          A contact form on Y is a one-form λ satisfying
                                                                          λ ^ dλ ą 0.
                                          A contact form determines the Reeb vector field, R, defined by
                                                                  λpRq “ 1,     dλpR, ¨q “ 0,
                                          and the contact structure ξ :“ Kerpλq. Closed orbits of R are called
                                          Reeb orbits.
                                             If pY, λq is a closed three-manifold equipped with a nondegenerate
                                          contact form and Γ P H1 pY q, then the embedded contact homology
                                          ECHpY, λ, Γq is defined. This is the homology of a chain complex freely
                                          generated over Z2 by certain sets of Reeb orbits in the homology class
                                          Γ, relative to a differential that counts certain J-holomorphic curves
                                          in R ˆ Y. (ECH can also be defined over Z, but for the applications in
                                          this paper we will not need this.) It is known that the homology only
                                          depends on ξ, and so we sometimes denote it ECHpY, λ, ξq. Any Reeb
                                          orbit γ has a symplectic action
                                                                                 ż
                                                                         Apγq “ λ
                                                                                    γ

                                          and this induces a filtration on ECHpY, λq; we can use this filtration
                                          to define a number cσ pY, λq for every nonzero class in ECH, called
                                          the spectral invariant associated to σ; the spectral invariants are C 0
                                          continuous and so can be extended to degenerate contact forms as well.
                                            2000 Mathematics Subject Classification. 53D35, 57R57, 57R58 .
                                            D. C-G. is partially supported by NSF grant 1711976.
                                            N. S. is partially supported by the DFG funded project CRC/TRR 191.
                                                                                1
             SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                       2

We will review the definition of ECH and of the spectral invariants in
§2.1.
  When the class c1 pξq`2P.D.pΓq P H 2 pY ; Zq is torsion, then ECHpY, ξ, Γq
has a relative Z grading, which we can refine to a canonical absolute
grading grQ by rationals [18], and which we will review in §2.3. It is
known that for large gradings the group is eventually 2-periodic and
non-vanishing:
                ECH˚ pY, ξ, Γq “ ECH˚`2 pY, ξ, Γq ‰ 0, ˚ " 0.
The main theorem of [11] states that in this case, the asymptotics of
the spectral invariants recover the contact volume
                                     ż
                         volpY, λq “    λ ^ dλ.
                                         Y
Specifically:
Theorem 1. [11, Thm. 1.3] Let pY, λq be a closed, connected oriented
three-manifold with a contact form, and let Γ P H1 pY q be such that
c1 pξq ` 2P.D.pΓq is torsion. Then if tσj u is any sequence of nonzero
classes in ECHpY, ξ, Γq with definite gradings tending to positive in-
finity,
                            cσj pY, λq2
(1.1)                     lim           “ volpY, λq.
                         jÑ8 grQ pσj q

   The formula (1.1) has had various implications for dynamics. For
example, it was a crucial ingredient in recent work [10] of the first au-
thor and collaborators showing that many Reeb vector fields on closed
three-manifolds have either two or infinitely many distinct closed or-
bits, and it was used in [9] to show that every Reeb vector field on
a closed three-manifold has at least two distinct closed orbits. It has
also been used to prove C 8 closing lemmas for Reeb flows on closed
three-manifolds and Hamiltonian flows on closed surfaces [1, 17].
   By (1.1), we can write
                            b
(1.2)           cσj pY, λq “ volpY, λq ¨ grQ pσj q ` dpσj q,

where dpσj q is opgrQ pσj q1{2 q as grQ pσj q tends to positive infinity. It is
then natural to ask:
Question 2. What can we say about the asymptotics of dpσj q as grQ pσj q
tends to positive infinity?
  Previously, W. Sun has shown that dpσj q is OpgrQ pσj qq125{252 [27,
Thm. 2.8]. Here we show:
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                     3

Theorem 3. Let pY, λq be a closed, connected oriented three-manifold
with contact form λ, and let Γ P H1 pY q be such that c1 pξq ` 2P.D.pΓq is
torsion. Let tσj u be any sequence of nonzero classes in ECHpY, λ, Γq
with definite gradings tending to positive infinity. Define dpσj q by (1.2).
Then dpσj q is OpgrQ pσj qq2{5 as grQ pσj q Ñ `8.

   We do not know whether or not the OpgrQ pσj qq2{5 asymptotics here
are optimal — in other words, we do not know whether there is some
contact form on a three-manifold realizing these asymptotics. We will
show in §4.3 that there exist contact forms with Op1q asymptotics for
the dpσj q. In Remark 13, we clarify where the exponent 25 comes from in
our proof, and why the methods in the current paper can not improve
on it.
   Another topic which we do not address here, except in a very specific
example, see §4.3, but which is of potential future interest, is whether
the asymptotics of the dpσj q carry interesting geometric information.
In this regard, a similar question in the context of the spectral flow
of a one-parameter family of Dirac operators was recently answered in
[22]. This is particularly relevant in the context of the argument we
give here, as our argument also involves estimating spectral flow, see
Remark 13.

1.2. A dynamical zeta function and a Weyl law. We now mention
two corollaries of Theorem 3.
  Given Γ P H1 pY q, define a set of nonnegative real numbers, called
the ECH spectrum for pY, λ, Γq
             ΣpY,λ,Γq :“ Y˚ ΣpY,λ,Γq,˚
            ΣpY,λ,Γq,˚ :“ tcσ pλq |0 ‰ σ P ECH˚ pY, ξ, Γ; Z2qu .
(To emphasize, in the set ΣpY,λ,Γq,˚ we are fixing the grading ˚.) Then,
define the Weyl counting function for ΣpY,λ,Γq
                                                      (
(1.3)           NpY,λ,Γq pRq :“ # c P ΣpY,λ,Γq |c ď R .
We now have the following:
Corollary 4. If c1 pξq ` 2P.D.pΓq is torsion, then the Weyl counting
function (1.3) for the ECH spectrum satisfies the asymptotics
                          „ d         
                             2 ´1             `     ˘
(1.4)             N pRq “               R2 ` O R9{5
                           vol pY, λq
where d “ dim ECH˚ pY, ξ, Γ; Z2q ` dim ECH˚`1 pY, ξ, Γ; Z2q, ˚ " 0.
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                      4

  As another corollary, one may obtain information on the correspond-
ing dynamical zeta function. To this end, first note that the ECH zeta
function
                                         ÿ
(1.5)             ζECH ps; Y, λ, Γq :“           c´s
                                         c‰0PΣpY,λ,Γq

converges for Re psq ą 2 by (1.1) and defines a holomorphic function
of s in this region whenever c1 pξq ` 2P.D.pΓq is torsion, by (1.4).
   In view of for example [13, 14], one can ask if ζECH has a mero-
morphic continuation to C, and, if so, whether it contains interesting
geometric information. The Weyl law (1.4) then shows:
Corollary 5. The zeta function (1.5) continues meromorphically to
the region Re psq ą 35 . The only pole in this region is ”at s “ 2ı which is
                                                           2d ´1
further simple with residue Ress“2 ζECH ps; Y, λ, Γq “    volpY,λq
                                                                     .

   In §4.3, we give an example of a contact form for which ζECH has
a meromorphic extension to all of C with two poles at s “ 1, 2. The
meromorphy and location of the poles of (1.5) would be interesting to
figure out in general.

1.3. Idea of the proof and comparison with previous works.
The method of the proof uses previous work by C. Taubes relating
embedded contact homology to Monopole Floer homology. By us-
ing Taubes’s results, we can estimate spectral invariants associated
to nonzero ECH classes by estimating the energy of certain solutions
of the deformed three-dimensional Seiberg-Witten equations. This is
also the basic idea behind the proofs of Theorem 1 and the result of
Sun mentioned above, and it was inspired by a similar idea in Taubes’s
proof of the three-dimensional Weinstein conjecture [24].
   The essential place where our proof differs from these arguments
involves a particular estimate, namely a key “spectral flow” bound
for families of Dirac operators that appears in all of these proofs. This
estimate bounds the difference between the grading of a Seiberg-Witten
solution, and the “Chern-Simons” functional, which we review in §2.2,
and is important in all of the works mentioned above. We prove a
stronger bound of this kind than any previous bound, see Proposition 6
and the discussion about the eta invariant below, and this is the key
point which allows us to prove OpgrQ pσj qq2{5 asymptotics. Spectral
flow bounds for families of Dirac operators were also considered in
[20, 21, 26]. The main difference here is that in those works the bounds
were proved on reducible solutions where the connections needed to
           SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                   5

define the relevant Dirac operators were explicitly given. Here we must
consider irreducible solutions, and so we rely on a priori estimates.
   We have chosen to phrase this spectral flow bound in terms of a
bound on the eta invariants of a family of operators. By the Atiyah-
Patodi-Singer index theorem, the bound we need on the spectral flow
is equivalent to a bound on the eta invariant, and we make the rela-
tionship between these two quantities precise in the appendix.
   The paper is organized as follows. In §2, we review what we need
to know about embedded contact homology, Monopole Floer cohomol-
ogy and Taubes’s isomorphism. §3 reviews the eta invariant, reviews
the necessary estimates on irreducible solutions to the Seiberg-Witten
equations, and proves the key Proposition 6. We then give the proof
of Theorem 3 in §4 — while our argument in this section is novel, one
could instead argue here as in [27], but we give our own argument here
since it might be of independent interest, see Remark 15. The end
of the paper reviews the sub-leading asymptotics and the dynamical
zeta function in the case of ellipsoids, and an appendix rephrases the
grading in Seiberg-Witten in terms of the eta invariant rather than in
terms of spectral flow.

1.4. Acknowledgments. The first author thanks M. Hutchings, D.
McDuff, and W. Sun for very helpful discussions.

                       2. Floer homologies
 We begin by reviewing the facts that we will need about ECH and
Monopole Floer homology.

2.1. Embedded contact homology. We first summarize what we
will need to know about ECH. For more details and for definitions of
the terms that we have not defined here, see [16].
   Let pY, λq be a closed oriented three-manifold with a nondegenerate
contact form. Fix a homology class Γ P H1 pY q. As stated in the intro-
duction, the embedded contact homology ECHpY, λ, Γq is the homol-
ogy of a chain complex ECCpY, λ, Γq. To elaborate, the chain complex
ECC is freely generated over Z2 by orbit sets α “ tpαj , mj qu where the
αj ’s are distinct embeddedřReeb orbits while each mj P N; we further
have the constraints that mj αj “ Γ P H1 pY q and mj “ 1 if αj is
hyperbolic. To define the chain complex differential B, we consider the
symplectization pRt ˆ Y, d pet λqq, and choose an almost complex struc-
ture J that is R-invariant, rotates the contact hyperplane ξ :“ kerλ
positively with respect to dλ, and satisfies JBt “ R. The differential
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                      6

on ECC pY, λ, Γq is now defined via
                          ÿ
                     Bα “     7loooooooomoooooooon
                                rM1 pα, βq {Rs β.
                              β
                                        “:xBα,βy

Here M1 pα, βq denotes the moduli space of J-holomorphic curves C
of ECH index I pCq “ 1 in the symplectization, modulo translation in
the R-direction, and modulo equivalence as currents, with the set of
positive ends given by α and the set of negative ends given by β. If
J is generic, then the differential squares to zero B 2 “ 0 and defines
the ECH group ECH pY, λ; Γq . We will not review the definition of the
ECH index here, see [16] for more details, but the key point is that
the condition IpCq “ 1 forces C to be (mostly) embedded and rigid
modulo translation.
   As stated in the introduction, the homology ECH pY, λ; Γq does not
depend on the choice of generic J, and only depends on the associated
contact structure ξ; we therefore denote it ECH pY, ξ; Γq. (In fact, the
homology only depends on the spinc structure determined by ξ, but we
will not need that.) This follows from a canonical isomorphism between
ECH and Monopole Floer homology [25], which we will soon review.
The ECH index I induces a relative Z{dZ grading on ECH pY, ξ; Γq ,
where d is the divisibility of c1 pξq ` 2P.D. pΓq P H 2 pY ; Zq mod torsion.
In particular, it is relatively Z-graded when this second homology class
is torsion
   Recall now the action of a Reeb orbit from the introduction. This
induces an action on orbit sets α “ tpαj , mj qu by
                                          ˜ż     ¸
                                   ÿN
                          A pαq :“     mj       λ .
                                  j“1          αj

The differential decreases action, and so we can define ECC L pY, λ, Γq
to be the homology of the sub-complex generated by orbit sets of action
strictly less than L. The homology of this sub-complex ECH L pY, λ, Γq
is again independent of J but now depends on λ; there is an inclusion
induced map ECH L pY, λ, Γq Ñ ECH pY, ξ, Γq . Using this filtration,
we can define the spectral invariant associated to a nonzero class σ in
ECH
                                                                  ˘
  cσ pY, λq :“ inf L | σ P image pECH L pY, λ, Γq Ñ ECH pY, ξ, Γq u.
  As stated in the introduction, the spectral invariants are known to
be C 0 continuous in the contact form, and so extend to degenerate
contact forms as well by taking a limit over nondegenerate forms, see
[15].
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                       7

2.2. Monopole Floer homology. We now briefly review what we
need to know about Monopole Floer homology, referring to [18] for
additional details and definitions.
   Recall that a spin c structure on an oriented Riemannian three-manifold
Y is a pair pS, cq consisting of a rank 2 complex Hermitian vector bundle
and a Clifford multiplication endomorphism c : T ˚ Y bC Ñ End pSq sat-
isfying c pe1 q2 “ ´1 and c pe1 q c pe2 q c pe3 q “ 1 for any oriented orthonor-
mal frame pe1 , e2 e3 q of Ty Y . Let su pSq denote the bundle of traceless,
skew-adjoint endomorphisms of S with inner product 12 tr pA˚ Bq. Clif-
ford multiplication c maps T ˚ Y isometrically onto su pSq. Spinc struc-
tures exist on any three-manifold, and the set of spinc structures is an
affine space over H 2 pY ; Zq. A spin c connection A on S is a connection
such that c is parallel. Given two spin-c connections A1 , A2 on S, their
difference is of the form A1 ´ A2 “ a b 1S for some a P Ω1 pY, iRq .
If we denote by At1 , At2 the induced connections on det pSq “ Λ2 S, we
then have At1 ´ At2 “ 2a. Hence prescribing a spinc connection on
S is the same as prescribing a unitary connection on det pSq. We let
A pY, sq denote the space of all spinc connections on S. Given a spinc
connection A, we denote by ∇A the associated covariant derivative.
We then define the spin c Dirac operator DA : C 8 pSq Ñ C 8 pSq via
DA Ψ “ c ˝ ∇A Ψ.
   Given a spinc structure s “ pS, cq on Y , monopole Floer homology
assigns three groups denoted by HM        z pY, sq , HM ~ pY, sq and HM pY, sq.
These are defined via infinite dimensional Morse theory on the configu-
ration space C pY, sq “ A pY, sq ˆ C 8 pSq using the Chern-Simons-Dirac
functional L, defined as
                         ż                                        ż
                       1     ` t         t
                                           ˘ `                ˘ 1
(2.1) L pA, Ψq “ ´             A ´ A0 ^ FAt ` FAt0 `                xDA Ψ, Ψy dy
                       8 Y
                    loooooooooooooooooooomoooooooooooooooooooon 2 Y
                                 “:CSpAq


using a fixed base spin-c connection A0 (we pick one with At0 flat in
the case of torsion spin-c structures) and a metric g T Y .
   The gauge group G pY q “ Map pY, S 1 q acts on the configuration space
C pY, sq by u. pA, Ψq “ pA ´ u´1 du b I, uΨq . The gauge group action
is free on the irreducible part C ˚ pY, sq “ tpA, Ψq P C pY, sq |Ψ ‰ 0u Ă
C pY, sq and not free along the reducibles. The blow up of the configu-
ration space along the reducibles

                 C σ pY, sq “ tpA, s, Φq | }Φ}L2 “ 1, s ě 0u

then has a free G pY q action u ¨ pA, s, Φq “ pA ´ u´1 du b I, s, uΦq .
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                    8

   To define the Monopole Floer homology groups one needs to perturb
the Chern-Simons-Dirac functional (2.1). First given
                                                   ş    a one form µ P
  1                                              1
Ω pY ; iRq, one defines the functional eµ pAq :“ 2 Y µ ^ FAt whose gra-
dient is calculated to be ˚dµ. To achieve non-degeneracy and transver-
sality of configurations one uses the perturbed Chern-Simons-Dirac
functional
(2.2)                Lµ pA, Ψq “ L pA, Ψq ´ eµ pAq
where µ is a suitable finite linear combination of eigenvectors of ˚d with
non-zero eigenvalue. Next let
                   T “ tA P A pY, sq |FAt “ 0u {G pY q
be the space of At flat spin-c connections up to gauge equivalence.
We choose a Morse function f : T Ñ          `“ R  to define
                                                      ‰˘    the functional
f : C pY, sq Ñ R, f pA0 ` a, s, Ψq :“ f A0 ` a , where ah denotes
     σ                                         t    h

the harmonic
          ´     part of a P¯ Ω1 pY, iRq. The gradient may be calculated
p∇fqσA “ p∇f qpAt qh , 0, 0 .
   The Monopole Floer homology groups are now defined using solu-
tions pA, s, Φq P C σ pY, sq to the three-dimensional Seiberg-Witten equa-
tions
              1
                 ˚ FAt ` s2 c´1 pΦΦ˚ q0 ` p∇f qppAq ` ˚dµ “ 0
              2
                                              sΛ pA, s, Φq “ 0
(2.3)                              DA Φ ´ Λ pA, s, Φq Φ “ 0

where Λ pA, s, Φq “ xDA Φ, ΦyL2 and pΦΦ˚ q0 :“ Φ b Φ˚ ´ 12 |Φ|2 defines
a traceless, Hermitian endormophism of S. We denote by C the set of
solutions to the above equations.
  We first subdivide the solutions as follows:
   Co “ tpA, s, Φq P C|s ‰ 0u {G pY q ,
   Cs “ tpA, 0, Φq P C|Λ pA, 0, Φq ą 0u {G pY q
        "
                  1
      “ pA, Φq | FAt ` dµ “ 0, rAs is a critical point of f,
                  2
           Φ is a (positive-)normalized eigenvector of DA u {G pY q
   Cu “ tpA, 0, Φq |Λ pA, 0, Φq ă 0u {G pY q
        "
                  1
      “ pA, Φq | FAt ` dµ “ 0, rAs is a critical point of f,
                  2
           Φ is a (negative-)normalized eigenvector of DA u {G pY q .
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                        9

Next, we consider the free Z2 modules generated by the three sets above
               C o “ Z2 rCo s , C s “ Z2 rCs s , C u “ Z2 rCu s .
The chain groups for the three versions of Floer homology mentioned
above are defined by
               Č “ C o ‘ C s , Ĉ “ C o ‘ C u , C̄ “ C s ‘ C u .
These chain groups Č, Ĉ, C̄ can be endowed with differentials B̌, B̂, B̄
with square zero; we do not give the precise details here, but the idea
is to count Fredholm index one solutions of the four-dimensional equa-
tions, see [18, Thm. 22.1.4] for the details. The homologies of these
three complexes are by definition the three monopole Floer homology
groups
                    ~ pY, sq , HM
                    HM           z pY, sq , HM pY, sq .
They are independent of the choice of metric and perturbations µ, f.
   Each of the above Floer groups has a relative Z{dZ grading where
d is the divisibility of c1 pSq P H 2 pY ; Zq mod torsion. This is defined
using the extended Hessian
HppA,Ψq : C 8 pY ; iT ˚ Y ‘ R ‘ Sq Ñ C 8 pY ; iT ˚ Y ‘ R ‘ Sq ; pA, Ψq P C pY, sq
                               » ﬁ »                          ﬁ» ﬁ
                                 a      ˚da ` 2c´1 pψΨq0 ´ df      a
                       HppA,Ψq – f ﬂ “ – ´d˚ a ` iRe xψ, Ψy ﬂ – f ﬂ
                                ψ        c paq Ψ ` DA ψ ` f Ψ     ψ
                                       »                      ﬁ »   ﬁ
                                           ˚d    ´d c´1 p.Ψq0     a
(2.4)                                “ – ´d˚     0     x., Ψy ﬂ – f ﬂ .
                                        c p.q Ψ .Ψ      DA        ψ
The relative grading between two irreducible generators !ai “ pAi , )si , Φi q,
(si ‰ 0), i “ 1, 2, is now defined via gr pa1 , a2 q “ sf HppAt ,Ψt q
                                                                        0ďtď1
(mod d) for some path of configurations pAt , Ψt q starting at pA2 , s2 Φ2 q
and ending at pA1 , s1 Φ1 q, where sf denotes the spectral flow.
   In the case when the spin-c structure is torsion, the monopole Floer
groups are further equipped with an absolute Q-grading, refining this
relative grading. As we will review in the appendix, this is given via
(2.5)     #                                            `          ˘
                             1       1                          A
  Q
            2k ´´ η pD A q `
                           ¯ 4
                               ηY ´ 2π 2 CS pAq ;  a “  A,  0, Φk   P Cs ,
gr ras “          ppA,sΦ1 q ` 5 ηY ´ 1 2 CS pAq ; a “ pA, s, Φq P Co , s ‰ 0.
            ´η H               4       2π

where ΦAk above denotes the kth positive eigenvector of DA (see §A),
and ηY and ηDA denote the eta invariant of the corresponding operator,
which we will review in §3.
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                      10

2.3. ECH=HM. We now state the isomorphism between the ECH
and HM, proved in [25]. Given a contact manifold pY 3 , λq with dλ-
compatible almost complex structure J as before, we define a metric
g T Y via g T Y |ξ “ dλ p., J.q, |R| “ 1 and R and ξ are orthogonal. This
metric is adapted to the contact form in the sense ˚dλ “ 2λ, |λ| “ 1.
                                      ´1
Decompose ξ bC “ loomo    K on ‘ loKomoon into the i, ´i eigenspaces of J. The
                        ξ 1,0    ξ 0,1
contact structure now determines the canonical spin-c structure sξ via
S ξ “ C ‘ K ´1 with Clifford multiplication cξ given by
                            „     
                   ξ         i
                  c pRq “           ,
                               ´i
                            „              
                    ξ               ´iv1,0
                   c pvq “ 0,1               , v P ξ.
                             v ^
Furthermore, there„ is a unique spin-c connection Ac on S ξ with the
                      1
property that DAc         “ 0 and we call the induced connection Atc on
             ` ˘      0
K ´1 “ det S ξ the canonical connection. Tensor product with an
auxiliary Hermitian line bundle E via S E “ S ξ b E and cE “ cξ b 1
gives all other spin-c structures sE . Furthermore all spin-c connections
on S E arise as A “ Ac b 1 ` 1 b ∇A for some unitary connection ∇A
on E. The ECH/HM isomorphism is then
                     `       ˘
(2.6)          ~˚ ´Y, sE “ ECH˚ pY, ξ; P.D.c1 pEqq .
               HM
In the literature, this isomorphism is often stated with the left hand
side given by the cohomology group HMz˚ pY q instead; the point is that
z˚ pY q and HMp´Y
HM              ~         q are canonically isomorphic, see [18, S 22.5,
Prop. 28.3.4]. The isomorphism (2.6) allows us to define a Q-grading
on ECH, by declaring that (2.6) preserves this Q-grading.
   We now state the main ideas involved
                                  `     ˘in the isomorphism (we restrict
attention to the case when c1 det sE is torsion, which is the case
which is relevant here, and we sometimes state estimates that, while
true, are stronger than those originally proved by Taubes). To this
                  `        ˘
             ~ ´Y, sE . We use the perturbed Chern-Simons-Dirac
end, let σ P HM
functional (2.2) and its gradient flow (2.3) with µ “ irλ, r P r0, 8q,
in defining monopole Floer homology. (One also adds a small term
η to µ to achieve transversality, see for example [11], but to simplify
the notation we will for now suppress this term.) Giving a family of
                                           `       ˘
(isomorphic) monopole Floer groups HM ~ ´Y, sE , the class σ is hence
representable by a formal sum of solutions to (2.3) corresponding to
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                    11

µ “ irλ. Denote by Cˇr the µ “ irλ version of the complex Č and
note that its reducible generators are all of the form a “ pA, 0, Φk q
where A “ A0 ´ irλ, At0 is flat and Φk is the kth positive eigenvector
of DA . An important estimate η pDA0 ´irλ ş q “ O prq now gives that    the
                                      r2
grading of this generator grQ ras “ 4π  2   λ ^ dλ `  Oprq  ą  gr Q
                                                                    rσs  by
(2.5) for r " 0. Hence for r " 0 the class σ is represented by a formal
sum of irreducible solutions to (2.3) with µ “ irλ, and by a max-min
argument, we may choose a family pAr , Ψr q :“ pAr , sr Φr q satisfying
                        grQ rσs “ grQ rpAr , Ψr qs .
Following a priori estimates on solutions to the Seiberg-Witten
                                                              ´       equa-
                                                                       ¯
tions, one then proves another important estimate η HpAr ,Ψr q “p
   `    ˘                                                       `   ˘
O r 3{2 uniformly in the class σ. This gives CS pAr q “ O r 3{2 which
in turn by a differential relation (see §4) leads to eλ pAr q “ O p1q. The
final step in the proof shows that for any sequence of solutions pAr , Ψr q
to Seiberg-Witten equations with e¨   λ pAr q bounded,   ˛ the E-component
                         „ `
                          Ψr       8˝
Ψ`r of the spinor Ψr “ Ψ´ P C               E ‘ K ´1 E ‚ satisfies the weak
                                        Y ; looooomooooon
                            r
                                               “S E
                   ´1
convergence  pΨ`
               r q   p0q á tpαj , mj qu to some ECH orbit set. This last
                                                        `     ˘
                                                   ~ Y, sE in ECH
orbit set is what corresponds to the image of σ P HM
under the isomorphism (2.6). Furthermore, crucially for our purposes,
one has
                                        eλ pAr q
(2.7)                    cσ pλq “ lim            ,
                                   rÑ8     2π
see [11, Prop. 2.6]. (The proof in [11, Prop. 2.6] is given in the case
where λ is nondegenerate, but it holds for all λ by continuity.)

                 3. Estimating the eta invariant
  Let D be a generalized Dirac operator acting on sections of a Clifford
bundle E over a closed, oriented Riemannian manifold Y . Then the
sum
                                   ÿ sgnpλq
(3.1)                   ηpD, sq :“
                                   λ‰0
                                        |λ|s

is a convergent analytic function of a complex variable s, as long as
Repsq is sufficiently large; here, the sum is over the nonzero eigenvalues
of D. Moreover, the function ηpD, sq has an analytic continuation to a
meromorphic function on C of s, which we also denote by ηpD, sq, and
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                 12

which is holomorphic near 0. We now define
                            ηpDq :“ ηpD, 0q.
We should think of this as a formal signature of D, which we call the
eta invariant of Atiyah-Patodi-Singer [2].
  We will be primarily concerned with the case where D “ DAr ,
namely D is the spin-c Dirac operator for a connection Ar solving
(2.3). Another case of interest to us is where D is the odd signature
operator on C 8 pY ; iT ˚ Y ‘ Rq sending
                       pa, f q ÞÑ p˚da ´ df, d˚aq ,
in which case we denote the corresponding η invariant by ηY .
   Now consider the Seiberg-Witten equations (2.3) corresponding to
µ “ irλ, for a torsion spinc structure as above, and note that an irre-
ducible solution (after rescaling the spinor) corresponds to a solution
pAr , Ψr q to the Seiberg-Witten equations on C pY, sq given via
                    1
                      c p˚FAt q ` r pΨΨ˚ q0 ` c pirλq “ 0
                    2
(3.2)                                           DA Ψ “ 0.,
A further small perturbation is needed to obtain transversality of so-
lutions see [11, S 2.1]. We ignore these perturbation as they make no
difference to the overall argument.
   We can now state the primary result of this section:
                                                ´            ¯   ´ ¯
Proposition 6. Any solution to (3.2) satisfies η H ppAr ,Ψr q “ O r 23 .

  The purpose of the rest of the section will be to prove this.
3.1. Known estimates. We first collect some known estimates on
solutions to the equations (3.2).
Lemma 7. For some constants cq , q “ 0, 1, 2, . . ., we have
                                `            ˘
(3.3)             |∇q FAt | ď cq 1 ` r 1`q{2 .
Proof. We first note that we have the estimates:
                        ˇ `ˇ
                        ˇΨ ˇ ď 1 ` c0
                          r
                                   ˆr                 ˙
                        ˇ ´ ˇ c0 ˇˇ       ˇ ` ˇ2 ˇˇ 1
                        ˇΨ ˇ ď            ˇ    ˇ
                                     ˇ 1 ´ Ψr ˇ `
                          r
                                 r                  r
                 ˇ` A ˘q ` ˇ       `         ˘
                 ˇ ∇     Ψr ˇ ď cq 1 ` r q{2
                 ˇ` A ˘q ´ ˇ       `             ˘
                 ˇ ∇     Ψr ˇ ď cq 1 ` r pq´1q{2
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                  13

The first two of these estimates are proved in [24, Lem. 2.2]. The third
and fourth are proved in [24, Lem. 2.3].
  The lemma now follows by combining the above estimates with the
equation (3.2).                                                       
  In (4), we will also need:
Lemma 8. One has the bound
(3.4)                 |CS pAr q| ď c0 r 2{3 eλ pAr q4{3
where the constant c0 only depends on the metric contact manifold.
Proof. This is proven in [24, eq. 4.9], see also [11, Lem. 2.7].       

3.2. The η invariant of families of Dirac operators. In this sec-
tion, we prove the key Proposition 6. The main point that we need is
the following fact concerning the η invariant:
Proposition 9. Let Ar be a solution to (3.2). Then η pDAr q is Opr 3{2q
as r Ñ 8.
   Before giving the proof, we first explain our strategy.
   The first point is that we have the following integral formula for the
η invariant:
                                  ż8
                               1                  2
(3.5)              ηpDAr q “ ?        tr pDAr e´tDAr qdt
                               πt 0
where the right hand side is a convergent integral. This is proved in
[7, S 2], by Mellin transform it is equivalent to the fact that the eta
function ηpDAr , sq in (3.1) is holomorphic for Repsq ą ´2.
   We therefore have to estimate the integral in (3.5). To do this, we
will need the following estimates:
Lemma 10. There exists a constant c0 independent of r such that for
all r ě 1, t ą 0:
                  ˇ ´           ¯ˇ
                  ˇ           2  ˇ
                  ˇtr DAr e´tDAr ˇ ď c0 r 2 ec0 rt ,   and
                      ˇ ´       ¯ˇ
                      ˇ       2  ˇ
                      ˇtr e´tDAr ˇ ď c0 t´3{2 ec0 rt .

  Once we have proved Lemma 10, Proposition 9 will follow from a
short calculation, which we will give at the end of this section.
  The proof of Lemma 10 will require two auxiliary lemmas, see Lemma 11
and Lemma 12 below, and some facts about the heat equation associ-
ated to a Dirac operator that we will now first recall. Let D be a Dirac
             SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                                 14

operator on a Clifford bundle V over a closed manifold Y . The heat
equation associated to D is the equation
                                Bs
                                   ` D2s “ 0
                                Bt
                                                          2
for sections s, and nonnegative time t; the operator e´tD is the solution
operator for this equation. The heat equation has an associated heat
kernel Ht px, yq which is a (time-dependent) section of the bundle V bV
over Y ˆ Y whose fiber over a point px, yq is Vx b Vy˚ ; it is smooth for
t ą 0. For any smooth section s of V and t ą 0, the heat kernel satisfies
                                  ż
                     ´tD 2
(3.6)               e      spxq “    Ht px, yqspyqvolpyq.
                                       Y
Also,
                           „      
                        B       2
(3.7)                       ` Dx Ht px, yq “ 0,
                       Bt
where Dx denotes the Dirac operator applied in the x variables.
  Moreover,
                              ż
                     ´tD 2
(3.8)            trpe      q“      trpHt py, yqqvolpyq.
                                       Y
Hence, we can prove Lemma 10 by bounding |Ht | along the diagonal.
                    2
The operator De´tD has a kernel Lt px, yq as well, and the analogous
results hold.
  A final fact we will need is Duhamel’s principle: this says that the
inhomogeneous heat equation
                             Bs̃
                                 ` D 2 s̃t “ st
                             Bt
has a unique solution tending to 0 with t, given by
                             żt
                                         1  2
(3.9)               s̃t pxq “ pe´pt´t qD st1 qpxqdt1 ,
                                   0
as long as st is a smooth section of S, continuous in t.
  Now let D be DAr , and V the spinor bundle for the spinc structure
S, and let Htr and Lrt be defined as above, but with D “ DAr . Let
ρpx, yq the Riemannian distance function. Define an auxiliary function
                                                     ρpx,yq2
                        ht px, yq :“ p4πtq´3{2 e´       4t     .
                            3
  In the case of Y “ R , with ρ the standard Euclidean distance, the
function ht px, yq is precisely the ordinary heat kernel. In our case, the
kernel Htr px, yq has an asymptotic expansion as t Ñ 0,
(3.10)   Htr px, yq „ ht px, yq pbr0 px, yq ` br1 px, yqt ` br2 px, yqt2 ` . . . `q,
                  SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                                             15

that is studied in detail in [6, Ch. 2]; here, the bri px, yq are defined on
all of Y ˆ Y . The following lemma summarizes what we need to know
about the results from [6, Ch. 2]:
Lemma 11. There exists for all i “ 0, 1, 2, . . . sections bri px, yq such
that:
      ‚ The bri are supported in any neighborhood of the diagonal.
      ‚ The asymptotic expansion (3.10) may be formally differentiated
        to obtain asymptotic expansions for the derivative. In particu-
        lar, there is an asymptotic expansion
(3.11)     Lrt px, yq „ ht px, yq pb̃r0 px, yq ` b̃r1 px, yqt ` b̃r2 px, yqt2 ` . . . `q
         where
                      b̃rn px, yq “ pDAr ` cpρdρ{2tqqbrn px, yq.
     ‚ For any n, t ą 0,
                                                    ÿ
                                                    n
                         Lrt px, yq   ´ ht px, yq            b̃ri px, yqti
                                                    i“0

         is Opti´1{2 q, in the C 0 - norm on the product, as t Ñ 0.
     ‚
                         ˜                                                     ¸
                                                       ÿ
                                                       n
(3.12)     pBt ` DA2 r q Lrt px, yq ´ ht px, yq                b̃ri px, yqti       “ ´DA2 r b̃rn tn .
                                                       i“0

Proof. The lemma summarizes those parts of the proof of [6, Thm.
2.30] that we will soon need; the arguments in [6, Thm. 2.30] provide
the proof. The idea behind the first bullet point is that ht px, yq is on
the order of t8 away from the diagonal. The reason for the i ´ 1{2
exponent in the third bullet point is that ht has a t´3{2 term. For the
fourth bullet point, the point is that the coefficients bri are constructed
so as to satisfy (3.7) when formally differentiating (3.10) and equating
powers of t; this gives a recursion which is relevant for our purposes
because it implies that when we truncate the expansion at a finite n,
the inhomogeneous equation (3.12) is satisfied.                          
  In view of the first bullet point of the above lemma, we only have to
understand the coefficients bri in a neighborhood of the diagonal. To
facilitate this, let igT Y denote the injectivity
                                         ´     ¯ radius of the Riemannian
                                                    ig T Y
metric g, and given y P Y , let By                    2
                                                                 denote a geodesic ball of
         ig T Y
radius     2
                  centered at y, and let y denote a choice of co-ordinates
             SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                                  16
                              ´ ´ i ¯¯
                                     TY
on this ball. Define Gky Ă C 8 By g 2      to be the subspace of (r-
                                                        ´    |α|
                                                                 ¯
                                                α          k` 2
dependent) functions f satisfying the estimate By f “ O r          as
                3                    1
r Ñ 8, @α´ P ´Ni0 , and
                     ¯¯ for each j P 2 N0 , further define the subspace
                  TY
Wyj Ă C 8 By g 2        via

                                     ÿ
                                     N
                                                                                    |α|
(3.13) f P   Wyj   ðñ f “                 fi ,   with each fi P y α Gky , k ď j `       .
                                    i“1
                                                                                     2
Finally, ´given¯y P Y , we choose a convenient frame for Syξ and E
           i TY
over By g 2 , which we will call a synchronous frame; specifically,
choose an orthonormal basis for each of Syξ , Ey , and parallel transport
along geodesics with Ac , Ar to obtain local orthonormal trivializations
                                                              i TY
ts1 , s2 u , teu. Now, if b is any section of S b Sy over By p g 2 q ˆ tyu write
                             ÿ
                             2
(3.14)         bp¨, yq “              y
                                     fb,kl p.q psk b eq p.q psl b eq˚ pyq .
                            k,l“1

Lemma 12. There is a constant c0 independent of r such that for any
t ą 0, r ě 1, we have
(3.15)                     |Htr px, yq| ď c0 h2t px, yq ec0 rt .
                                                                ´i ¯
                                                                   TY
Further, for any y P Y , the restriction of the terms brj to By g 2 ˆtyu
have the property that their corresponding functions fbyrj ,kl in (3.14) are
all in Wyj .
Proof. The first bullet point is similar to [20, Prop. 3.1].
  To prove the second bullet point, we use the fact that the terms brj
in the heat kernel expression (3.10) are known to satisfy a recursion,
as alluded to above, and explained in the proof of [6, Thm. 2.30].
Specifically, fix y P Y , choose geodesic coordinates y around y, mapping
0 to y, and choose a synchronous frame as in (3.14). Then, in these
coordinates, we have
                               ÿ
                               2
(3.16)        br0 px, yq   “         g ´1{4 pxqpsi b eqpxqpsi b eq˚ pyq,
                               i“1

where g “ det pgjk q. Moreover, if use these coordinates to identify
sections of S b Sy˚ with a vector of functions, then we have
                            ż1
        r              1
(3.17) bj px, yq “ ´ 1{4       ρj´1 g 1{4 pρxq DA2 r brj´1 pρx, yq dρ, j ě 1
                    g pxq 0
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                                           17

where on the right hand side of this equation, we mean that we are
integrating this vector of functions component by component.
   Now recall the Bochner-Lichnerowicz-Weitzenbock formula for the
Dirac operator:
                                     κ 1
(3.18)               DA2 r “ ∇˚Ar ∇Ar `
                                      ´ cp˚FAr q,
                                     4 2
where κ denotes the scalar curvature; we will want to combine this
with (3.17). In coordinates, we have
(3.19)              ∇Ar “ pB1 ` Γ1 , B2 ` Γ2 , B3 ` Γ3 q,
where each Γi is the ith Christoffel symbol for Ar . We also have
                             ÿ
                             3                                ÿ
                                                              3
(3.20)           ∇˚Ar   “´               jk
                                   ´g pBk ` Γk q `                g jk Γijk ,
                             j,k                          i,j,k

where the Γij,k are the Christoffel symbols of the Riemannian metric.
Since we have Ar “ 1bA`Ac b1, where Ac is the canonical connection
on S ξ , we can decompose each Christoffel symbol
(3.21)                              Γi “ ci ` ai ,
where the ci are Christoffel symbols for Ac and the ai are Christoffel
symbols for A.
  The ci are independent of r. To understand the aj , first write the
defining equations for the curvature
                              Fkj “ Bk aj ´ Bj ak .
                                                                                ř3    k
Now write the coordinate x “ px1 , x2 , x3 q, and consider                       k“1 x pBk aj   ´
Bj ak q. Reintroducing the radial coordinate ρ, we have
                                   ÿ
                                   3
                                                      Baj
                             ρ           xk Bk aj “       .
                                   k“1
                                                      Bρ
On the other hand, since the frame e is parallel, we have ∇x1 Bx1 `x2 Bx2 `x3 Bx3 e “
                 ř
∇ρBρ e “ 0, hence 3k“1 xk ak “ 0. Thus, we have
                              ÿ3 ż1
(3.22)              aj pxq “        dρρxk Fkj pρxq .
                                    k“1 0

In particular, it follows from the a priori estimate (3.23) and (3.22)
that each
                                                 1
(3.23)                                   aj P Wy2 .
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                              18

                                                 maxtj,ku
  Now note that we have Wyj ` Wyk Ă Wy                      , Wyj ¨ Wyk Ă Wyj`k , and
            j` 1
By Wyj Ă Wy 2 . Hence, by (3.18), (3.19), (3.20), and (3.23), we have
that the square of the Dirac operator has the schematic form
                            ÿ
(3.24)              DA2 r “   ´g jk Bj Bk ` Pj Bj ` Q
                                    j,k

                   1
where Pj P W 2 and Q P Wy1 . The Lemma now follows by induction,
using (3.16) and (3.17).                                      

  We now give the promised:

Proof of Lemma 10. The second bullet point follows by combining (3.8)
and (3.15).
   To prove the first bullet point, our strategy will be to bound the
pointwise size of the kernel Lrt py, yq and appeal to the version of (3.8)
for Lrt .
   To do this, consider the asymptotic expansion (3.11). By a theorem
                                                                  ` ˘
of Bismut-Freed ([7, Thm. 2.4]), for any y P Y , trLrt py, yq is O t1{2 as
t Ñ 0. So we have trLrt py, yq “ tr Rtr py, yq for the remainder
                       Rtr px, yq :“Lrt ´ DAr rht pb0 ` tb1 qs .
By (3.12), Rtr satisfies the inhomogeneous heat equation
                                  "            ˆ     ˙       *
       `        2
                  ˘ r                  3         ρdρ    2
         Bt ` DAr Rt px, yq “ ht t ´DAr b1 ` c         DAr b1 ,
                                                  2t
and by the third bullet point of Lemma 11, Rtr Ñ 0 as t Ñ 0. We can
then apply Duhamel’s principle (3.9) to write
(3.25)         żt                      "                 ˆ      ˙         *
     r                   2
                  ´pt´sqDA                    3            ρdρ       2
    Rt px, yq “ e          r h px, yqs
                              s          ´DA b1 ` c                DA b1 ds.
                0                                           2s
                                       looooooooooooooooomooooooooooooooooon
                                                            “:Ks

We can then apply the key property of the heat kernel (3.6) to write
                    żtż
          r               r
        Rt px, yq “     Ht´s px, zqhs pz, yqsKs pz, yqvolpzqds,
                            0   Y

and we can apply the second bullet point of Lemma 11 to conclude
that
                    żtż
     r
   |Rt py, yq| ď c0     ec0 rpt´sq hs pz, yqh2pt´sq pz, yqsKs pz, yqvolpzqds.
                        0   Y
               SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                          19

By the first bullet      ¯ of Lemma 11, we can assume that Ks pz, yq is
                 ´ i point
                    gT Y
supported in By       2
                           ˆ tyu. Thus, we just have to bound
            żtż
                            c rpt´sq
(3.26)             ˆi
                       TY
                          ˙e 0       hs py, 0qh2pt´sq py, 0qsKspy, 0qdyds,
                            g
               0   By           2


where y are geodesic coordinates centered at y. To do this, choose a
synchronous frame for the spinor bundle, as we have been doing above.
Then, following (3.21), (3.22), in these coordinates the Dirac operator
is seen to have the form
                                      DAr “ w jk Bj ` K,
                                                            1
for r´independent w jk and K P Wy2 , in the geodesic coordinates and
orthonormal frame introduced before. Combining this with the second
                                                             5
bullet point of Lemma 12 gives that the term Ks P Wy2 . So, (3.26) is
dominated by a finite sum of integrals of the form
   żt ż
                           c rt                         α k    5 |α|
      ds     ˆi     ˙ dy se 0 h2pt´sq py, 0q hs py, 0q y r , kď `    .
    0     By   gT Y
                2
                                                               2  2
         ´i ¯
             TY
   On By g 2 , we have
                                                                1
                                y I ht py, 0q ď c1 t 2 |I| h2t py, 0q ,
for some constant c1 . Hence, we can bound the above integral by
                     żt ż
                                          |α|
             k c0 rt                    1` 2
(3.27)   c1 r e        ds  ˆi
                              TY
                                 ˙ dy s       h2pt´sq py, 0q h2s py, 0q dy.
                                           g
                            0        By        2


We also have
                    ż
                            ht px, yqht1 py, zqdy ď c2 h4pt`t1 q px, zq
                        Y
as proved in [20, Sec. A]. So, we can bound (3.27) by
            żt                                 żt
                1` |α|                               |α|                      1`|α|
    k c0 rt
c3 r e                                k
               s 2 h8t p0, 0qds “ c3 r ec 0 rt
                                                  s1` 2 p4tq´3{2 ds ď c3 r k t 2 ec0 rt .
           0                                                        0
                            1`|α|
We know that k ´         ď 2. So, putting all of the above together,
                              2
(3.26) is dominated by a finite sum of terms of the form
                                                1`|α|       1`|α|
                                          r2r     2     t     2     ec0 rt
which proves the result.                                                          
  We can finally give:
           SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                     20
                                                                         ş8      2
Proof of Proposition 9. Define Epxq :“ signpxqerfcp|x|q “ signpxq¨ ?2π   |x|
                                                                               e´s ds ă
   2
e´x . This is a rapidly decaying function, so the function EpDAr q is
defined, and its trace is a convergent sum
                                       ÿ
                         trpEpDAr qq “    Epλq,
                                        λ

where λ is an eigenvalue of DAr . The eta invariant in unchanged under
positive rescaling
                                    ˆ        ˙
                                       1
                       η pDAr q “ η ? DAr .
                                        r
Now use (3.5) to rewrite the right hand side of the above equation as
         ˇż 1          „                       ˆ        ˙ˇ
         ˇ                                                ˇ
         ˇ dt ?1 tr ?1 DAr e´ rt DA2 r ` tr E ?1 DAr ˇ .
         ˇ       πt       r                         r     ˇ
           0

The absolute value of the first summand in the above expression is
bounded from above by a constant multiple of r 3{2 , by the first bullet
point in Lemma 10. The absolute value of the second summand in
                                                    1 2
the same expression is bounded from above by tr e´ r DAr , which by the
second bullet point in Lemma 10 is bounded by a constant multiple of
r 3{2 as well.                                                        

Proof of Proposition 6. An application of the Atiyah-Patodi-Singer in-
dex theorem as in §A gives
       1 ´p          ¯ 1              1       !
                                                p
                                                       )
         η HpAr ,Ψr q “ η pDAr q ` ηY ` sf HpAr ,ǫΨr q         .
       2                 2            2                  0ďεď1
                                                    ` ˘
The spectral flow term`above ˘ is estimated to be O r 3{2 as in [24, S
5.4] while η pDAr q “ O r 3{2 by Proposition 9.                     

  We also note that the constant in Proposition 6 above is only a
function of pY, λ, Jq and independent of the class σ “ rpAr , Ψr qs P
     `       ˘
~ ´Y, sE defined by the Seiberg-Witten solution.
HM

Remark 13. The reason that we can not improve upon     gr˘Q asymptotics
                                                    ` 3{2
is because we do not know how to strengthen the O r        spectral flow
estimate on the irreducible solutions of Propositions 6 or 9. A better
Oprq estimate does however exist [20, 22] for reducible solutions for
which one understands the connection precisely in the limit r Ñ 8.
However, the a priori estimates (3.23) are not strong enough to carry
out the same for irreducibles.
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                    21

                   4. Asymptotics of capacities
4.1. The main theorem. In this section we now prove our main the-
orem Theorem 3 on ECH capacities.

Proof of Theorem 3. Let 0 ‰ σj P ECH pY, λ, Γq, j “ 0, 1, 2, . . ., be a
sequence of non-vanishing classes with definite gradings grQ pσj q tending
to positive infinity. As in §2.3, we use the perturbed Chern-Simons-
Dirac functional (2.2) Lµ and its gradient flow (2.3) with µ “ irλ, r P
r0, 8q, in defining monopole Floer homology. Hence for each r P r1, 8q,
the class σj may be represented by a formal sum of solutions to (2.3)
with µ “ irλ. As noted in §2.3, this solution is eventually irreducible.
Without loss of generality we may assume

                             grQ pσj q “ q ` j,

where q is a fixed rational number and j P 2N.
   We now estimate r1 pjq, the infimum of the values of r such that each
solution raj sr to (2.3) representing σj is irreducible. For this note that
a reducible solution is of the form a “ pA, 0, Φk q where A “ A0 ´ irλ,
At0 flat and Φk the kth positive eigenvector of DA . The ˇgradingˇof such
                                                           ˇ       ˇ
a reducible is given by (2.5). The important estimate ˇηDA1 `rλ ˇ ď c0 r,
([21, Thm. 1.2]) now shows grQ rpA, 0, Φk qs ą grQ pσj q “ q ` j for
                           "                               *
                               r2
         r ą r̄1 pjq :“ sup r| 2 vol pY, λq ă c0 r ` q ` j .
                              4π
Hence r1 pjq ă r̄1 pjq. Furthermore
                          „            1{2
                                 j
(4.1)        r̄1 pjq “ 2π                   ` O p1q as j Ñ 8
                            vol pY, λq
from the above definition. A max-min argument, as also mentioned
in §2.3 then gives @j P 2N a piecewise-smooth family of irreducible
solutions rasr “ pAr , Ψr q , r ą r1 pjq, of fixed grading grQ ras “ q ` j
such that Lµ is continuous, see (for example) [11, S 2.6].
   By (3.4), we have

(4.2)                  |CS pAq| ď c0 r 2{3 eλ pAq4{3 .

In addition, by combining (2.5) and Proposition 9, we have
                   ˇ                        ˇ
                   ˇ 1                      ˇ
                   ˇ                        ˇ       3{2
(4.3)              ˇ 2π 2 CS pAr q ´ pq ` jqˇ ď c0 r ,
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                               22

with the constant c0 ą 0 being independent of the grading j. We also
have the differential relation
                                         deλ   dCS
                                     r       “
                                         dr     dr
between the two functionals, away from the discrete set of points where
derivatives are şundefined, see [11, Lem. 2.5]. Now define F prq “
1 2              r
 r vol pY, λq ` r1 eλ pAs q ds. This is a continuous function, and v is
2 1
continuous as well, so we may integrate the above equation to conclude
that

                               CS prq “ rF 1 ´ F

valid for all r away from the above discrete set; here, we have used [27,
Property 2.3.(i)], together with the computation in [11, Lem. 2.3] in
the computation of the terms at r1 .
  On account of (4.3), F is then a super/subsolution to the ODEs

                  ´c2 r 3{2 ď rF 1 ´ F ´ pq ` jq ď c2 r 3{2

for r ě r1 . This gives
                           „                                      
      1 2                    q`j q`j               1{2        1{2
        r vol pY, λq ` r           ´       ´ 2c2 r ` 2c2 r1         ďF
      2 1                      r1      r
                                                                     k
                           „                                      
      1 2                    q`j q`j                          1{2
        r1 vol pY, λq ` r          ´       ` 2c2 r 1{2 ´ 2c2 r1     ěF
      2                        r1      r
                    1 2                q`j                      1{2
                       r1 vol pY, λq `      ´ 3c2 r 1{2 ` 2c2 r1 ďF 1
                    2r                  r1
                                                                     k
                    1 2                q`j                      1{2
(4.4)                  r1 vol pY, λq `      ` 3c2 r 1{2 ´ 2c2 r1 ěF 1 .
                    2r                  r1
  Next the estimate (4.2) in terms of F is
                                   4{3                                  4{3
(4.5)          ´ c1 r 2{3 pF 1 q         ď rF 1 ´ F ď c1 r 2{3 pF 1 q         .
                                                         1
We let ρ0 be the smallest positive root of               3
                                                             ´ rρ ` ρ2 ` ρ3 ` ρ4 s “ 0
and define
                                                                 (
                     r̄2 pjq “ sup r|c1 r ´2{3 F 1{3 ě ρ0
                 SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                          23
                                                                        ` 2c ˘
which is finite on account of (4.4). Further with c3 “ 1 ` 3                 1
                                                                                    `
 ` ˘2                                                                       3
3 2c31 define
                 #                                                ˆ         ˙3
                1                 q`j                    1{2           3
r̃2 pjq “ sup r| r12 vol pY, λq `     ` 3c2 r 1{2 ´ 2c2 r1 ě                        r
                2r                 r1                                 4c1
                                                                                    ˆ          ˙3
                      1 2                q`j q`j                    1{2                   1
                 or      r1 vol pY, λq `     ´   ` 2c2 r 1{2 ´ 2c2 r1 ě                             r
                      2r                  r1   r                                         9c3
(4.6)
                                                                                         *
                      1 2                q`j q`j                    1{2
                 or      r1 vol pY, λq `     ´   ` 2c2 r 1{2 ´ 2c2 r1 ě r
                      2r                  r1   r
                                                                  ` ˘
and set r2 pjq :“ max tr̄2 pjq , r̃2 pjqu. We note that r2 pjq “ O j 1{2 .
We now have the following lemma.

Lemma 14. For r ą r2 pjq we have
             ˆ ˙1{3                    ˆ ˙1{3
              F       2c1 2{3      1{3  F       2c1 2{3
(4.7)               ´    F ď pF 1 q ď         `    F .
              r       3r                r       3r

Proof. By definition,

                                 r ą r2 pjq ě r̄2 pjq
                           ùñ ρ :“ c1 r ´2{3 F 1{3 ă ρ0
                                                     1
                           ùñ ρ ` ρ2 ` ρ3 ` ρ4 ă
                                                     3
as well as

         r ą r2 pjq ě r̃2 pjq
(4.8)
                                                              ˆ         ˙3
          1  1              q`j                    1{2             3
    ùñ F ď r12 vol pY, λq `     ` 3c2 r 1{2 ´ 2c2 r1 ă                          r
          2r                 r1                                   4c1

by (4.4).
  For y “ pF 1 q1{3 equations (4.5) become the pair of quartic inequalities

(4.9)                      0 ď c1 r ´1{3 y 4 ´ y 3 ` r ´1 F
(4.10)                     0 ď c1 r ´1{3 y 4 ` y 3 ´ r ´1 F
                       SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                   24
                       ` F ˘1{3
With y0˘ “               r
                                  ˘   2c1 2{3
                                      3r
                                         F      we calculate
                ` ˘4 ` ˘3
      c1 r ´1{3 y0` ´ y0` ` r ´1 F
                          „                        
            ´5{3      4{3      4    64 2 32 3 16 4
     “´r         c1 F      1 ´ ρ ´ ρ ´ ρ ´ ρ ă 0 and
                               3    27    27   81
                `    ˘ 4    `  ˘ 3
      c1 r ´1{3 y0´ ` y0´ ´ r ´1 F
                      „                           
        ´5{3      4{3         4    64 2 32 3 16 4
     “r      c1 F       ´1 ´ ρ ` ρ ´ ρ ` ρ ă 0.
                              3    27    27   81

Since the minimum of the quartic (4.9) is attained at ymin “ pF 1 q1{3 “
    r , this gives pF 1 q1{3 “ y ď y0` or pF 1 q1{3 “ y ě ymin “ 4c31 r 1{3 .
 3 1{3
4c1
The second possibility being disallowed on account of (4.8), gives the
desired upper bound of (4.7). Similarly, the minimum of quartic (4.10)
is attained at the negative ymin “ ´ 4c31 r 1{3 . Hence y0´ ď y “ pF 1 q1{3
which is the lower bound in (4.7).                                        

  Next we cube (4.7) and use (4.6) to obtain

                                  F     F 4{3       F    F 4{3
(4.11)                              ´ c3 5{3 ď F 1 ď ` c3 5{3
                                  r     r           r    r
for r ě r2 pjq. This gives
                   „                                „               
             1{3           r 1{3           1{3   1{3       r 1{3
         r                               ďF ďr                         ,   where
                       ´c3 ` c´ 0r
                                   1{3                 c3 ` c`0r
                                                                 1{3
                                                           ”      ı1{3
                                                r3 ¯ c3 F rpr33 q
                                                 1{3

(4.12)                                     c˘
                                            0 “
                                                      F pr3 q1{3

for r ě r3 pjq ě r2 pjq. The last equation with (4.11) gives

      F pr3 q                 1         1         1             F pr3 q
ˆ       ”         ı1{3 ˙3 “ ` ´ ˘3 ď lo
                                     Fomo  on ď ` ` ˘ 3 “ ˆ
                                          prq                     ”         ı1{3 ˙3
  1{3     F pr3 q            c0                  c0         1{3     F pr3 q
 r3 ` c3 r 3                         “eλ pAr q             r3 ´ c3 r 3

and hence
        «         ˆ       ˙1{3 ﬀ                      «         ˆ       ˙1{3 ﬀ
F pr3 q       4c3 F pr3 q                     F pr3 q       4c3 F pr3 q
          1 ´ 1{3                ď eλ pAr q ď           1 ` 1{3
  r3         r3     r3                          r3         r3     r3
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                     25

from (4.6) for r " 0. However, (4.4) for r “ r3 when substituted into
the last equation above becomes
„                                                  „                                     
  q`j q`j               1{2                            1 2              q`j           1{2
        ´        ´ 2c2 r3 p1 ´ Rq ď eλ pAr q ď           r vol pY, λq `      ` 2c2 r3 p1 ` Rq
   r1        r3                                       2r3 1              r1
                                                „                                    1{3
                                            4c3 1 2                  q`j         1{2
                              with R :“ 1{3           r vol pY, λq `      ` 2c2 r3         .
                                            r3    2r3 1               r1
                                               ` ˘
Setting r3 “ j 4{5 (satisfying r3 ě r2 “ O j 1{2 for j " 0) and using
(2.7), (4.1) gives
                               1            1     ` ˘
(4.13)             cσj pλq “ j 2 vol pY, λq 2 ` O j 2{5 ,
as j Ñ 8, which is our main result Theorem 3.                             
Remark 15. One could replace the arguments in this subsection with
the arguments in Sun’s paper [27], if desired — the key reason why
we have a stronger bound than Sun is because of our stronger bound
on the Chern-Simons functional, and not because of anything we do in
this subsection. We have chosen to include our argument here, which
we developed independently of the arguments in [27], for completeness,
and because it might be of independent interest, although we emphasize
that we do use the result of Sun establishing [27, Property 2.3.(i)].
   On the other hand, the arguments in [11] are not quite strong enough
for Theorem 3, even with the improved bound in Proposition 9.
4.2. Proofs of Corollaries. Here we prove the two corollaries Corol-
lary 4 and Corollary 5, both following immediately from the capacity
formula Theorem 3.

Proof of Corollary 4. The Z2 vector space ECHpY, ξ, Γ; Z2q is known
to be two-periodic, and nontrivial, in sufficiently high grading, see for
example [16]. Thus, for ˚ " 0 sufficiently large there exists a finite set
of classes tσ1 , . . . , σ2d ´1 u Ă ECH˚ pY, ξ, Γ; Z2q Y ECH˚`1 pY, ξ, Γ; Z2q
such that
                               (
 0, U j σ1 , . . . , U j σ2d ´1 “ ECH˚`2j pY, ξ, Γ; Z2qYECH˚`1`2j pY, ξ, Γ; Z2q , @j ě 0.
Thus the ECH spectrum modulo a finite set is given by
                    !                                    )
                 8
               Yj“0 cU j σ1 pλq , . . . , cU j σ2d ´1 pλq .
                                             1            1    ` ˘
The corollary now follows as cU j σl pλq “ j 2 vol pY, λq 2 ` O j 2{5 , 1 ď
l ď 2d ´ 1, by Theorem 3.                                                 
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                    26

Proof of Corollary 5. As in the previous corollary, the ECH zeta func-
tion is given, modulo a finite and holomorphic in s P C, sum by
                ÿ8 ”                                        ı
                               ´s                        ´s
                    cU j σ1 pλq ` . . . ` cU j σ2d ´1 pλq     .
                j“0
       R
With ζ psq denoting the Riemann zeta function, we may using Theo-
rem 3 compare
        ˇ                                        ˇ
        ˇ                                        ˇ
        ˇ                                        ˇ
        ˇ                                        ˇ    ˜            ¸
        ˇ8                                       ˇ
        ˇÿ                         s ÿ
                                      8
                                               sˇ
                                                        ÿ8
                                                               1
        ˇ cU j σ pλq ´ vol pY, λq 2
                    ´s           ´
                                         j ´ 2 ˇˇ “ O
        ˇ       1
                                                            j 3s{5
        ˇj“0                         j“0
                                     loomoon     ˇ      j“0
        ˇ                                        ˇ
        ˇ                                        ˇ
        ˇ                            “ζ R p 2s q ˇ

whence the difference is holomorphic for Re psq ą 53 . The corollary
now follows `on˘knowing s “ 2 to be the only pole of the Riemann zeta
function ζ R 2s with residue 1.                                    
4.3. The ellipsoid example. We close by presenting an example with
Op1q asymptotics, and where the corresponding ζECH function extends
meromorphically to all of C.
  Consider the symplectic ellipsoid
                         " 2                *
                          |z1 |   |z2 |2
              Epa, bq :“        `        ď 1 Ă C2 “ R4 .
                            a       b
The symplectic form on R4 has a standard primitive
                                1ÿ
                                    2
                       λstd   “       pxi dyi ´ yi dxi q.
                                2 i“1
This restricts to BEpa, bq as a contact form, and the ECH spectrum of
pBEpa, bq, λq is known. Specifically, let Npa, bq be the sequence whose
j th element (indexed starting at j “ 0) is the pj ` 1qst smallest element
in the matrix
                         pma ` nbqpm,nqPZě0 ˆZě0 .
Then, the ECH spectrum SBEpa,bq is precisely the values in Npa, bq.
Moreover, the homology
                              ECH˚ pBEpa, bqq
has a canonical Z-grading, such that the empty set of Reeb orbits has
grading 0, and it is known to have one generator σj in each grading
2j, see [16]. The spectral invariant associated to σj is precisely the j th
element in the sequence Npa, bq.
             SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                          27

  With this understood, we now have:
Proposition 16. Let σj be any sequence of classes in ECHpBEpa, bqq
with grading tending to infinity. Then, the dpσj q are Op1q. In fact, if
a{b is irrational, then
                                   dpσj q   a`b
                               lim        “     .
                               jÑ8   j       2
Proof. Assume that a{b is irrational. If t “ cσj , then by the above
description, the grading of σj is precisely twice the number of terms in
Npa, bq that have value less than t. With this understood, the example
follows from [12, Lem. 2.1].
   When a{b is rational, a similar argument still works to show Op1q
asymptotics. Namely, if t “ cσj , then by above, the grading of σj is
precisely twice the number of terms in Npa, bq that have value?less than
t, up to an error no larger than some constant multiple of j. Now
apply [5, Thm. 2.10].                                                  
Proposition 17. The ECH ζ function ζECH for ECHpBEpa, bqq has
a meromorphic continuation to all of C. It has exactly two poles, at
s “ 1 and s “ 2, with residues
                                                         ˆ       ˙
                           1                            1 1 1
Ress“2 ζECH ps; Y, λ, Γq “ , Ress“1 ζECH ps; Y, λ, Γq “     `      .
                          ab                            2 a b
Proof. The ECH zeta function in this example
                    ÿ
ζECH ps; Y, λ, Γq “   pma ` nbq´s
                       m,nPN
                     1“ B                                                            ‰
(4.14)             “     ζ ps, a|a, bq ` ζ B ps, b|a, bq ´ a´s ζ R psq ´ b´s ζ R psq
                     2
is given in terms of the classical zeta functions of Riemann and Barnes
[4, 19]
                               ÿ
           ζ B ps, w|a, bq :“        pw ` ma ` nbq´s , w P Rą0 .
                               m,nPN0

Thus ζECH ps; Y, λ, Γq (4.14) is known to possess a meromorphic con-
tinuation to the entire complex plane in this example. Its only two
poles are at s “ 1, 2 with residues
                                            1
                 Ress“2 ζECH ps; Y, λ, Γq “
                                            abˆ     ˙
                                            1 1 1
                 Ress“1 ζECH ps; Y, λ, Γq “     `
                                            2 a b
             SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                    28

respectively; while its values at the non-positive integers are also known
[23, Cor. 2.4]. In particular its value at zero is
                                              ˆ        ˙
                                       1   1 b a
                  ζECH p0; Y, λ, Γq “ `            `     .
                                       4 12 a b
                                                                          

        Appendix A. The Q-grading and the η invariant
   In this appendix we give a formula for the absolute grading grQ on
monopole Floer groups of torsion spin-c structures (from [18, S 28.3])
in terms of a relevant eta invariant. First to recall the definition of
grQ paq , a “ pA, s, Φq, choose a four manifold X which bounds Y and
form the manifold with cylindrical end Z “ X Y pY ˆ r0, 8qt q. Choose
a metric g T Z on Z which is of product type    TZ
                                           ` TgZ Z“
                                                      TX      2
                                                   ˘ g ` dt on the
cylindrical end. Choose a spin-c structure S , c over Z which is of
the form
          S T Z “ S`T Z ‘ S´T Z ; S`T Z “ S´T Z “ S T Y ,
                                  „                
                                       0     cY pαq
                           Z
                          c pαq “                     , α P T ˚Y
                                    cY pαq      0
                                  „        
                          Z         0 ´I
                         c pdtq “
                                    I 0
over the cylindrical end and a spin-c connection on S T Z of the form
B “ dt ^ Bt ` A ‘ A on the cylindrical end. One may now form the
Fredholm elliptic operator
                        `                ˘       `                      ˘
   d˚ ` d` ` DB  `
                   : L21 Z; Λ1T Z ‘ S`T Z Ñ L2 Z; Λ0T Z ‘ Λ2,`TZ ‘ S TZ
                                                                     ´    .
                            “`           ˘‰
The absolute grading of A, 0, ΦA       0    P Cs , with At flat and ΦA 0 the
first positive eigenvector of DA , is given in terms of this operator. The
precise formula [18, Defn. 28.3.1] simplifies to
        “`         ˘‰          ` `˘ 1 @ ` `˘        ` ` ˘D 1
  grQ     A, 0, ΦA
                 0    “ ´2 ind  D B `   c 1 S , c 1  S    ´ σ pZq ,
                                      4                    4
with σ pZq denoting the signature of Z, S ` the bundle S`T Z from above,
and ind denoting the complex index, namely the difference in complex
dimensions, compare [8, S 3.4].
  The APS index theorem for spin-c Dirac operators now gives
                                                ż
              `     1 @ ` `˘      ` ` ˘D     1          ηpDA q
        indpDB q “     c1 S , c1 S        ´       p1 `          .
                    8                        24 X           2
            SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                    29

The APS signature theorem for the manifold X with boundary also
gives                        ż
                          1               1          1
                       ´          p1 “ ´ σ pZq ´ ηY ,
                          24 X            8          8
where ηY is the eta invariant of the odd signature operator on C 8 pY ; T ˚Y ‘ Rq
sending
                         pa, f q ÞÑ p˚da ´ df, ´d˚aq .
Combining the above we have
                         “`           ˘‰                1
                    grQ A, 0, ΦA    0    “ ´ηpDA q ` ηY .
                                                        4
                                     “`        ˘‰
A reducible generator rak s “ A, 0, Φk       A
                                                  P C however has 12 FAt “
                                                        s

´dµ and ΦA  k the kth eigenvector of DA . Hence,
          “`          ˘‰                        1
      grQ A, 0, ΦA  k    “ 2k ` p´η pDA q ` ηY q ´ 2 sftDAs u0ďsď1 ,
                                                4
where DAs is a family of Dirac operators, associated to a family of
connections starting at the flat connection and ending at one satisfying
1
  F t “ ´dµ. Hence, by interpreting this spectral flow as an index
2 A
through another application of Atiyah-Patodi-Singer [3, p. 95], and
applying [2, eq. 4.3] to compute this index, we get
                 “`           ˘‰                      1      1
(A.1)        grQ A, 0, ΦA   k     “ 2k ´ η pDA q ` ηY ´ 2 CSpAq
                                                      4     2π
as the absolute grading of a reducible generator.
   The absolute grading of an irreducible generator ra1 s “ pA1 , s, Φ1 q,
s ‰ 0, is then given by
                                             !            )
                grQ ra1 s “ grQ ra0 s ´ 2 sf H  ppAε ,Ψεq
                                                     0ďεď1

in terms of spectral flow of the Hessians (2.4) for a path pAε , Ψε q P
A pY, sq ˆ C 8 pSq, ε P r0, 1s starting at ra0 s “ rpA0 , 0qs and ending at
pA1 , sΦ1 q. As above, we can interpret this spectral flow as an index;
this time, to compute the relevant index, we need to apply ([2, Thm.
3.10]), which gives that the above is equal to
                           ´           ¯ 5         ż
                Q  1
              gr ra s “ ´η H  ppA,sΦ1 q ` ηY ´ 2              ρ0 .
                                          4         Y ˆr0,1sε

Here ρ0 is the usual Atiyah-Singer integrand, namely the local index
density defined as the
                    ´ constant
                             ¯    term in the small time expansion of the
                       ´tD 2
local supertrace str e         with
                        „              „    
                              ´1            1 p
(A.2)             D“               Bε `        HpAε ,Ψε q ,
                          1               1
               SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                                    30

and where pAǫ , Ψǫ q is the chosen path of configurations. To compute
the index density we choose a path of the form
                         #
                            pA ` 2ε pA1 ´ Aq , 0q ; 0 ď ε ď 21 ,
            pAε , Ψε q “
                            pA1 , p2ε ´ 1q Ψq ;                   1
                                                                  2
                                                                    ď ε ď 1.
                  “ 1‰
On the interval 0, 2 , the integral of the local density is given by the
usual local index theorem: as above, we have
                         ż
                                                        1
                      ´2                 ρ0 “ ´ 2 CS pAq .
                          Y ˆr0, 21 s                 2π
                                      ε
                                                                 “ ‰
  On the other hand, for the calculation on Y ˆ 21 , 1 ε , we have ρ0 “ 0.
                                                               „         
                           2           2         p 2            ´1
To see this, first note D “ ´Bε ` H                pAε ,Ψε q `             2MΨ gives
                                                                       1
      ´        ¯      ´            p2                                   p2
                                                                                            ¯
         ´tD 2          ´tr´Bε2 `H                ´2MΨ s       ´tr´Bε2 `H            `2MΨ s
   str e         “ tr e              pA  ε ,Ψ ε q          ´e             pAε ,Ψ ε q          .

Duhamel’s principle then gives that the coefficients in the small time
heat kernel expansion of the difference above are of the form
                             »         ﬁ
                                0 0 ˚
                             –0 0 ˚ ﬂ
                                ˚ ˚ 0
with respect to the decomposition iT ˚ Y ‘ R ‘ S.
  Hence we have in summary:
         #                  1        1
                                                      `        A
                                                                 ˘
  Q
           2k ´´ η pD A q¯` 4
                              η Y ´ 2π 2 CS pAq ; a “   A, 0, Φk   P Cs ,
gr ras “         ppA,sΦq ` 5 ηY ´ 1 2 CS pAq ; a “ pA, s, Φq P Co , s ‰ 0.
           ´η H               4       2π


                                      References
 [1] M. Asaoka and K. Irie, A C 8 closing lemma for Hamiltonian diffeomor-
     phisms of closed surfaces, Geom. Funct. Anal., 26 (2016), pp. 1245–1254. 1.1
 [2] M. F. Atiyah, V. K. Patodi, and I. M. Singer, Spectral asymmetry
     and Riemannian geometry. I, Math. Proc. Cambridge Philos. Soc., 77 (1975),
     pp. 43–69. 3, A, A
 [3]      , Spectral asymmetry and Riemannian geometry. III, Math. Proc. Cam-
     bridge Philos. Soc., 79 (1976), pp. 71–99. A
 [4] E. W. Barnes, The theory of the double gamma function., Philos. Trans. R.
     Soc. Lond., Ser. A, Contain. Pap. Math. Phys. Character, 196 (1901), pp. 265–
     387. 4.3
 [5] M. Beck and S. Robins, Computing the continuous discretely, Undergradu-
     ate Texts in Mathematics, Springer, New York, second ed., 2015. Integer-point
     enumeration in polyhedra, With illustrations by David Austin. 4.3
              SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                             31

 [6] N. Berline, E. Getzler, and M. Vergne, Heat kernels and Dirac op-
     erators, Grundlehren Text Editions, Springer-Verlag, Berlin, 2004. Corrected
     reprint of the 1992 original. 3.2, 3.2, 3.2
 [7] J.-M. Bismut and D. S. Freed, The analysis of elliptic families. II. Dirac
     operators, eta invariants, and the holonomy theorem, Comm. Math. Phys., 107
     (1986), pp. 103–163. 3.2, 3.2
 [8] D. Cristofaro-Gardiner, The absolute gradings on embedded contact ho-
     mology and Seiberg-Witten Floer cohomology, Algebr. Geom. Topol., 13 (2013),
     pp. 2239–2260. A
 [9] D. Cristofaro-Gardiner and M. Hutchings, From one Reeb orbit to two,
     J. Differential Geom., 102 (2016), pp. 25–36. 1.1
[10] D. Cristofaro-Gardiner, M. Hutchings, and D. Pomerleano, Torsion
     contact forms in three-dimensions have two or infinitely many Reeb orbits,
     Preprint, available online at arxiv:1701.02262 (2017). 1.1
[11] D. Cristofaro-Gardiner, M. Hutchings, and V. G. B. Ramos, The
     asymptotics of ECH capacities, Invent. Math., 199 (2015), pp. 187–214.
     (document), 1.1, 1, 2.3, 2.3, 3, 3.1, 4.1, 4.1, 15
[12] D. Cristofaro-Gardiner, T. Li, and R. Stanley, Irrational triangles
     with polynomial Ehrhart functions, (2018). 4.3
[13] S. Dyatlov and M. Zworski, Dynamical zeta functions for Anosov flows
     via microlocal analysis, Ann. Sci. Éc. Norm. Supér. (4), 49 (2016), pp. 543–577.
     1.2
[14] P. Giulietti, C. Liverani, and M. Pollicott, Anosov flows and dynam-
     ical zeta functions, Ann. of Math. (2), 178 (2013), pp. 687–773. 1.2
[15] M. Hutchings, Quantitative embedded contact homology, J. Differential
     Geom., 88 (2011), pp. 231–266. 2.1
[16]       , Lecture notes on embedded contact homology, in Contact and symplec-
     tic topology, vol. 26 of Bolyai Soc. Math. Stud., János Bolyai Math. Soc.,
     Budapest, 2014, pp. 389–484. 2.1, 4.2, 4.3
[17] K. Irie, Dense existence of periodic Reeb orbits and ECH spectral invariants,
     J. Mod. Dyn., 9 (2015), pp. 357–363. 1.1
[18] P. Kronheimer and T. Mrowka, Monopoles and three-manifolds, vol. 10
     of New Mathematical Monographs, Cambridge University Press, Cambridge,
     2007. 1.1, 2.2, 2.2, 2.3, A
[19] S. N. M. Ruijsenaars, On Barnes’ multiple zeta and gamma functions, Adv.
     Math., 156 (2000), pp. 107–132. 4.3
[20] N. Savale, Asymptotics of the Eta Invariant, Comm. Math. Phys., 332 (2014),
     pp. 847–884. 1.3, 3.2, 3.2, 13
[21] N. Savale, Koszul complexes, Birkhoff normal form and the magnetic Dirac
     operator, Anal. PDE, 10 (2017), pp. 1793–1844. 1.3, 4.1
[22]       , A Gutzwiller type trace formula for the magnetic Dirac operator, Geom.
     Funct. Anal., 28 (2018), pp. 1420–1486. 1.1, 13
[23] M. Spreafico, On the Barnes double zeta and Gamma functions, J. Number
     Theory, 129 (2009), pp. 2035–2063. 4.3
[24] C. H. Taubes, The Seiberg-Witten equations and the Weinstein conjecture,
     Geom. Topol., 11 (2007), pp. 2117–2202. 1.3, 3.1, 3.1, 3.2
[25]       , Embedded contact homology and Seiberg-Witten Floer cohomology I,
     Geom. Topol., 14 (2010), pp. 2497–2581. 2.1, 2.3
             SUB-LEADING ASYMPTOTICS OF ECH CAPACITIES                      32

[26] C.-J. Tsai, Asymptotic spectral flow for Dirac operations of disjoint Dehn
     twists, Asian J. Math., 18 (2014), pp. 633–685. 1.3
[27] S. Weifeng, An estimate on energy of min-max Seiberg-Witten Floer gener-
     ators , 2018. arXiv:1801.02301. 1.1, 1.3, 4.1, 15

  Department of Mathematics, University of California Santa Cruz,
CA 95064, United States
  E-mail address: dcristof@ucsc.edu

  Universität zu Köln, Mathematisches Institut, Weyertal 86-90, 50931
Köln, Germany
  E-mail address: nsavale@math.uni-koeln.de
