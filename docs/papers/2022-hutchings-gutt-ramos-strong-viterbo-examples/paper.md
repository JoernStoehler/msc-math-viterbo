---
source: arXiv:2003.10854
fetched: 2025-10-20
---
# Examples around the strong Viterbo conjecture

                                                       Examples around the strong Viterbo conjecture
                                                           Jean Gutt∗, Michael Hutchings† & Vinicius G. B. Ramos‡

                                                                                    October 6, 2020
arXiv:2003.10854v3 [math.SG] 3 Oct 2020




                                                                                        Abstract
                                                       A strong version of a conjecture of Viterbo asserts that all normalized symplectic
                                                  capacities agree on convex domains. We review known results showing that certain spe-
                                                  cific normalized symplectic capacities agree on convex domains. We also review why all
                                                  normalized symplectic capacities agree on S 1 -invariant convex domains. We introduce
                                                  a new class of examples called “monotone toric domains”, which are not necessarily con-
                                                  vex, and which include all dynamically convex toric domains in four dimensions. We
                                                  prove that for monotone toric domains in four dimensions, all normalized symplectic
                                                  capacities agree. For monotone toric domains in arbitrary dimension, we prove that the
                                                  Gromov width agrees with the first equivariant capacity. We also study a family of ex-
                                                  amples of non-monotone toric domains and determine when the conclusion of the strong
                                                  Viterbo conjecture holds for these examples. Along the way we compute the cylindrical
                                                  capacity of a large class of “weakly convex toric domains” in four dimensions.


                                          1       Introduction
                                          If X and X 0 are domains1 in R2n = Cn , a symplectic embedding from X to X 0 is a smooth
                                          embedding ϕ : X ,→ X 0 such that ϕ? ω = ω, where ω denotes the standard symplectic form
                                          on R2n . If there exists a symplectic embedding from X to X 0 , we write X ,→ X 0 .
                                                                                                                      s
                                              An important problem in symplectic geometry is to determine when symplectic embed-
                                          dings exist, and more generally to classify the symplectic embeddings between two given
                                          domains. Modern work on this topic began with the Gromov nonsqueezing theorem [11],
                                          which asserts that the ball

                                                                         B 2n (r) = z ∈ Cn π|z|2 ≤ r
                                                                                   


                                          symplectically embeds into the cylinder

                                                                      Z 2n (R) = z ∈ Cn π|z1 |2 ≤ R
                                                                                 

                                              ∗
                                               Université Toulouse III - Paul Sabatier, 118 route de Narbonne, 31062 Toulouse Cedex 9, France
                                          & Institut National Universitaire Champollion, Place de Verdun, 81012 Albi, France
                                             †
                                               University of California, Berkeley, partially supported by NSF grant DMS-2005437
                                             ‡
                                               Instituto de Matemática Pura e Aplicada, Estrada Dona Castorina, 110, Rio de Janeiro - RJ -
                                          Brasil, 22460-320, partially supported by grants from the Serrapilheira Institute, FAPERJ and CNPq
                                             1
                                               In this paper, a “domain” is the closure of an open set. One can of course also consider domains in
                                          other symplectic manifolds, but we will not do so here.


                                                                                             1
if and only if r ≤ R. Many questions about symplectic embeddings remain open, even for
simple examples such as ellipsoids and polydisks.
    If there exists a symplectic embedding X ,→ X 0 , then we have the volume constraint
                                                           s
Vol(X) ≤ Vol(X 0 ). To obtain more nontrivial obstructions to the existence of symplectic
embeddings, one often uses various symplectic capacities. Definitions of the latter term
vary; here we define a symplectic capacity to be a function c which assigns to each domain
in R2n , possibly in some restricted class, a number c(X) ∈ [0, ∞], satisfying the following
axioms:

 (Monotonicity) If X and X 0 are domains in R2n , and if there exists a symplectic embedding
    X ,→ X 0 , then c(X) ≤ c(X 0 ).
             s

 (Conformality) If r is a positive real number then c(rX) = r2 c(X).

We say that a symplectic capacity c is normalized if it is defined at least for convex domains
and satisfies
                               c B 2n (1) = c Z 2n (1) = 1.
                                                      

      The first example of a normalized symplectic capacity is the Gromov width defined by
                                                            
                                                  2n
                              cGr (X) = sup r B (r) ,→ X .
                                                                    s


This trivially satisfies all of the axioms except for the normalization requirement cGr (Z 2n (1)),
which holds by Gromov non-squeezing. A similar example is the cylindrical capacity defined
by                                                              
                                   cZ (X) = inf R X ,→ Z 2n (R) .
                                                               s

    Additional examples of normalized symplectic capacities are the Hofer-Zehnder capacity
cHZ defined in [16] and the Viterbo capacity cSH defined in [31]. There are also useful families
of symplectic capacities parametrized by a positive integer k, including the Ekeland-Hofer
capacities cEH
            k defined in [8, 9] using calculus of variations; the “equivariant capacities” ck
                                                                                             CH

defined in [12] using positive equivariant symplectic homology; and in the four-dimensional
case, the ECH capacities cECH k    defined in [17] using embedded contact homology. For
each of these families, the k = 1 capacities cEH       CH
                                                  1 , c1 , and c1
                                                                  ECH are normalized. Some

additional symplectic capacities defined using rational symplectic field theory were recently
introduced in [27, 28]. For more about symplectic capacities in general we refer to [6, 25]
and the references therein.
    The goal of this paper is to discuss some results and examples related to the following
conjecture, which apparently has been folkore since the 1990s.

Conjecture 1.1 (strong Viterbo conjecture). If X is a convex domain in R2n , then all
normalized symplectic capacities of X are equal.

      Viterbo conjectured the following statement2 in [32]:
  2
      Viterbo also conjectured that equality holds in (1.1) only if int(X) is symplectomorphic to an open ball.



                                                       2
Conjecture 1.2 (Viterbo conjecture). If X is a convex domain in R2n and if c is a
normalized symplectic capacity, then

                                         c(X) ≤ (n! Vol(X))1/n .                                        (1.1)

   The inequality (1.1) is true when c is the Gromov width cGr , by the volume constraint,
because Vol(B 2n (r)) = rn /n!. Thus Conjecture 1.1 implies Conjecture 1.2. The Viterbo
conjecture recently gained more attention as it was shown in [4] that it implies the Mahler
conjecture3 in convex geometry.

Lemma 1.3. If X is a domain in R2n , then cGr (X) ≤ cZ (X), with equality if and only if
all normalized symplectic capacities of X agree (when they are defined for X).

Proof. It follows from the definitions that if c is a normalized symplectic capacity defined
for X, then cGr (X) ≤ c(X) ≤ cZ (X).

   Thus the strong Viterbo conjecture is equivalent to the statement that every convex
domain X satisfies cGr (X) = cZ (X). We now discuss some examples where it is known
that cGr = cZ . Hermann [13] showed that all T n -invariant convex domains have to satisfy
cGr = cZ . This generalizes to S 1 -invariant convex domains by the following elementary
argument:

Proposition 1.4 (Y. Ostrover, private communication). Let X be a compact convex do-
main in Cn which is invariant under the S 1 action by eiθ · z = (eiθ z1 , . . . , eiθ zn ). Then
cGr (X) = cZ (X).

Proof. By compactness, there exists z0 ∈ ∂X minimizing the distance to the origin. Let
r > 0 denote this minimal distance. Then the ball (|z| ≤ r) is contained in X, so by
definition cGr (X) ≥ πr2 .
     By applying an element of U (n), we may assume without loss of generality that z0 =
(r, 0, . . . , 0). By a continuity argument, we can assume without loss of generality that ∂X
is a smooth hypersurface in R2n . By the distance minimizing property, the tangent plane
to ∂X at z0 is given by (z · (1, 0, . . . , 0) = r) where · denotes the real inner product. By
convexity, X is contained in the half-space (z · (1, 0, . . . , 0) ≤ r). By the S 1 symmetry, X
is also contained in the half-space (z · (eiθ , 0, . . . , 0) ≤ r) for each θ ∈ R/2πZ. Thus X is
contained in the intersection of all these half-spaces, which is the cylinder |z1 | ≤ r. Then
cZ (X) ≤ πr2 by definition.

Remark 1.5. A similar argument shows that if k ≥ 3 is an integer and if X ⊂ Cn is a
convex domain invariant under the Z/k action by j · z = (e2πij/k z1 , . . . , e2πij/k zn ), then

                                          cZ (X)   k
                                                  ≤ tan(π/k).
                                          cGr (X)  π
   3
       The Mahler conjecture [22] states that for any n-dimensional normed space V , we have
                                                                  4n
                                          Vol(BV ) Vol(BV ∗ ) ≥      ,
                                                                  n!
where BV denotes the unit ball of V , and BV ∗ denotes the unit ball of the dual space V ∗ . For some examples
of Conjectures 1.1 and 1.2 related to the Mahler conjecture, see [26].


                                                      3
   The role of the convexity hypothesis in Conjecture 1.1 is somewhat mysterious. We now
explore to what extent non-convex domains can satisfy cGr = cZ .
   To describe some examples, if Ω is a domain in Rn≥0 , define the toric domain

                             XΩ = z ∈ Cn π(|z1 |2 , . . . , |zn |2 ) ∈ Ω .
                                 


The factors of π ensure that
                                          Vol(XΩ ) = Vol(Ω).                                        (1.2)
Let ∂+ Ω denote the set of µ ∈ ∂Ω such that µj > 0 for all j = 1, . . . , n.

Definition 1.6. A monotone toric domain is a compact toric domain XΩ with smooth
boundary such that if µ ∈ ∂+ Ω and if v an outward normal vector at µ, then vj ≥ 0 for all
j = 1, . . . , n. See Figure 1c.
   A strictly monotone toric domain is a compact toric domain XΩ with smooth boundary
such that if µ ∈ ∂+ Ω and if v is a nonzero outward normal vector at µ, then vj > 0 for all
j = 1, . . . , n.

       One of our main results is the following:

Theorem 1.7. (proved in §4) If XΩ is a monotone toric domain in R4 , then cGr(X) =
cZ (X).

    Note that monotone toric domains do not have to be convex; see §2 for details on
when toric domains are convex. (Toric domains that are convex are already covered by
Proposition 1.4.)
    To clarify the hypothesis in Theorem 1.7, let X be a compact domain in R2n with
smooth boundary, and suppose that X is “star-shaped”, meaning that the radial vector
field on R2n is transverse to ∂X. Then there is a well-defined Reeb vector field R on ∂X.
We say that X is dynamically convex if, in addition to the above hypotheses, every Reeb
orbit γ has Conley-Zehnder index CZ(γ) ≥ n+1 if nondegenerate, or in general has minimal
Conley-Zehnder index4 at least n + 1. It was shown by Hofer-Wysocki-Zehnder [14] that if
X is strictly convex, then X is dynamically convex. However the Viterbo conjecture implies
that not every dynamically convex domain is symplectomorphic to a convex domain; see
Remark 1.9 below.

Proposition 1.8. (proved in §2) Let XΩ be a compact star-shaped toric domain in R4 with
smooth boundary. Then XΩ is dynamically convex if and only if XΩ is a strictly monotone
toric domain.

       Thus Theorem 1.7 implies that all dynamically convex toric domains in R4 have cGr =
cZ .
   If X is a star-shaped domain with smooth boundary, let Amin (X) denote the minimal
period of a Reeb orbit on ∂X.
   4
     If γ is degenerate then there is an interval of possible Conley-Zehnder indices of nondegenerate Reeb
orbits near γ after a perturbation, and for dynamical convexity we require the minimum number in this
interval to be at least n + 1. In the 4-dimensional case (n = 2), this means that the dynamical rotation
number of the linearized Reeb flow around γ, which we denote by ρ(γ) ∈ R, is greater than 1.


                                                    4
Remark 1.9. Without the toric hypothesis, not all dynamically convex domains in R4 have
cGr = cZ . In particular, it is shown in [1] that for ε > 0 small, there exists a dynamically
convex domain X in R4 such that Amin (X)2 /(2 vol(X)) ≥ 2−ε. One has cCH   1 (X) ≥ Amin (X)
                                 2
by [12, Thm. 1.1], and cGr (X) ≤ 2 vol(X) by the volume constraint. Thus
                                          cZ (X)   √
                                                  ≥ 2 − ε.
                                          cGr (X)
Remark 1.10. It is also not true that all star-shaped toric domains have cGr = cZ . Coun-
terexamples have been known for a long time, see e.g. [13], and in §5 we discuss a new
family of counterexamples.
    For monotone toric domains in higher dimensions, we do not know how to prove that
all normalized symplectic capacities agree, but we can at least prove the following:
Theorem 1.11. (proved in §3) If XΩ is a monotone toric domain in R2n , then
                                        cGr (XΩ ) = cCH
                                                     1 (XΩ ).                                      (1.3)
   Returning to convex domains, some normalized symplectic capacities are known to agree
(not the Gromov width or cylindrical capacity however), as we review in the following
theorem:
Theorem 1.12 (Ekeland, Hofer, Zehnder, Abbondandolo-Kang, Irie). If X is a convex
domain in R2n , then:
 (a) cEH                           CH
      1 (X) = cHZ (X) = cSH (X) = c1 (X).

 (b) If in addition ∂X is smooth5 , then all of the capacities in (a) agree with Amin (X).
Proof. Part (b) implies part (a) because every convex domain can be C 0 approximated by
one with smooth boundary; and the capacities in (a) are C 0 continuous functions of the
convex domain X, by monotonicity and conformality.
   Part (b) was shown for cHZ (X) by Hofer-Zehnder in [16], and for cSH (X) by Irie [20] and
Abbondandolo-Kang [2]. The agreement of these two capacities with cCH    1 (X) for convex
domains now follows from the combination of [12, Theorem 1.24] and [10, Lemma 3.2], as
explained by Irie in [20, Remark 2.15]. Finally, part (b) for cEH
                                                               1 (X) has been claimed and
understood for a long time, but since we could not find a complete proof in the literature
we give one here in §6.

Organization of the paper
In §2 we discuss different kinds of toric domains and when they are convex or dynamically
convex. In §3 we consider the first equivariant capacity and prove Theorem 1.11. In §4 we
use ECH capacities to prove Theorem 1.7. In §5 we consider a family of examples of non-
monotone toric domains and determine when they do or do not satisfy the conclusions of
Conjectures 1.1 and 1.2. Along the way we compute the cylindrical capacity of a large class
of “weakly convex toric domains” in four dimensions (Theorem 5.6). In §6 we review the
definition of the first Ekeland-Hofer capacity and complete the (re)proof of Theorem 1.12.
   5
    Without the smoothness assumption, it is shown in [3, Prop. 2.7] that cHZ (X) agrees with the minimum
action of a “generalized closed characteristic” on ∂X.


                                                   5
Acknowledgements
We thank A. Oancea, Y. Ostrover and M. Usher for useful discussions, J. Kang and E.
Shelukhin for bringing some parts of the literature to our attention and F. Schlenk for
detailed comments on an earlier version of this paper.


2     Toric domains
In this section we review some important classes of toric domains and discuss when they
are convex or dynamically convex.
    If Ω is a domain in Rn , define
                             b = µ ∈ Rn (|µ1 |, . . . , |µn |) ∈ Ω .
                                  
                            Ω

Definition 2.1. [12] A convex toric domain is a toric domain XΩ such that Ω
                                                                          b is compact
and convex. See Figure 1a.
    This terminology may be misleading because a “convex toric domain” is not the same
thing as a compact toric domain that is convex in R2n ; see Proposition 2.3 below.
Definition 2.2. [12] A concave toric domain is a toric domain XΩ such that Ω is compact
and Rn≥0 \ Ω is convex. See Figure 1b.
   We remark that if XΩ is a convex toric domain or concave toric domain and if XΩ has
smooth boundary, then it is a monotone toric domain.
Proposition 2.3. A toric domain XΩ is a convex subset of R2n if and only if the set
                                                            
                                 n         2            2
                                                          
                       Ω = µ ∈ R π |µ1 | , . . . , |µn | ∈ Ω
                       e                                                           (2.1)

is convex in Rn .
Proof. (⇒) The set Ω
                   e is just the intersection of the toric domain XΩ with the subspace
  n    n
R ⊂ C . If XΩ is convex, then its intersection with any linear subspace is also convex.
                             e is convex. Let z, z 0 ∈ XΩ and let t ∈ [0, 1]. We need to
    (⇐) Suppose that the set Ω
show that
                                  (1 − t)z + tz 0 ∈ XΩ .
That is, we need to show that

                              (1 − t)z1 + tz10 , . . . , (1 − t)zn + zn0
                                                                           
                                                                                ∈ Ω.
                                                                                  e                   (2.2)
                                                               e as are the 2n points (±|z 0 |, . . . , ±|z 0 |).
We know that the 2n points (±|z1 |, . . . , ±|zn |) are all in Ω,                         1                n
By the triangle inequality we have

                                 |(1 − t)zj + tzj0 | ≤ (1 − t)|zj | + t|zj0 |

for each j = 1, . . . , n. It follows that the point in (2.2) can be expressed as (1 − t) times a
convex combination of the points (±|z1 |, . . . , ±|zn |), plus t times a convex combination of
the points (±|z10 |, . . . , ±|zn0 |). Since Ω
                                             e is convex, it follows that (2.2) holds.

                                                      6
                  Ω
                                                           Ω

             (a) A convex toric domain                 (b) A concave toric domain




                                                                 Ω
                      Ω


           (c) A monotone toric domain               (d) A weakly convex toric domain

                      Figure 1: Examples of toric domains XΩ in R4


Example 2.4. If XΩ is a convex toric domain, then XΩ is a convex subset of R2n .

Proof. Similarly to the above argument, this boils down to showing that if w, w0 ∈ C and
0 ≤ t ≤ 1 then
                          |(1 − t)w + tw0 |2 ≤ (1 − t)|w|2 + t|w0 |2 .
The above inequality holds because the right hand side minus the left hand side equals
(t − t2 )|w − w0 |2 .

   However the converse is not true:

Example 2.5. Let p > 0, and let Ω be the positive quadrant of the Lp unit ball,
                                                      
                                            n         
                                                µpj ≤ 1 .
                                            X
                          Ω = µ ∈ Rn≥0
                                                      
                                               j=1

Then XΩ is a concave toric domain if and only if p ≤ 1, and a convex toric domain if and
only if p ≥ 1. By Proposition 2.3, the domain XΩ is convex in R2n if and only if p ≥ 1/2.

   We now work out when four-dimensional toric domains are dynamically convex.


                                           7
Proof of Proposition 1.8. As a preliminary remark, note that if a Reeb orbit has rotation
number ρ > 1, then so does every iterate of the Reeb orbit. Thus XΩ is dynamically convex
if and only if every simple Reeb orbit has rotation number ρ > 1.
    Since XΩ is star-shaped, Ω itself is also star-shaped. Since XΩ is compact with smooth
boundary, ∂+ Ω is a smooth arc from some point (0, b) with b > 0 to some point (a, 0) with
a > 0.
    We can find the simple Reeb orbits and their rotation numbers by the calculations in
[5, §3.2] and [12, §2.2]. The conclusion is the following. There are three types of simple
Reeb orbits on ∂XΩ :

 (i) There is a simple Reeb orbit corresponding to (a, 0), whose image is the circle in ∂XΩ
      with π|z1 |2 = a and z2 = 0.

 (ii) Likewise, there is a simple Reeb orbit corresponding to (0, b), whose image is the circle
       in ∂XΩ with z1 = 0 and π|z2 |2 = b.

 (iii) For each point µ ∈ ∂+ Ω where ∂+ Ω has rational slope, there is an S 1 family of simple
       Reeb orbits whose images sweep out the torus in ∂XΩ where π(|z1 |2 , |z2 |2 ) = µ.

Let s1 denote the slope of ∂+ Ω at (a, 0), and let s2 denote the slope of ∂+ Ω at (0, b). Then
the Reeb orbit in (i) has rotation number ρ = 1−s−1  1 , and the Reeb orbit in (ii) has rotation
number ρ = 1 − s2 . For a Reeb orbit in (iii), let ν = (ν1 , ν2 ) be the outward normal vector
to ∂+ Ω at µ, scaled so that ν1 , ν2 are relatively prime integers. Then each Reeb orbit in
this family has rotation number ρ = ν1 + ν2 .
     If XΩ is strictly monotone, then s1 , s2 < 0, and for each Reeb orbit of type (iii) we have
ν1 , ν2 ≥ 1. It follows that every simple Reeb orbit has rotation number ρ > 1.
     Conversely, suppose that every simple Reeb orbit has rotation number ρ > 1. Applying
this to the Reeb orbits (i) and (ii), we obtain that s1 , s2 < 0. Thus ∂+ Ω has negative slope
near its endpoints. The arc ∂+ Ω can never go horizontal or vertical in its interior, because
otherwise there would be a Reeb orbit of type (iii) with ν = (1, 0) or ν = (0, 1), so that
ρ = 1. Thus XΩ is strictly monotone.


3    The first equivariant capacity
We now prove Theorem 1.11. (Some related arguments appeared in [12, Lem. 1.19].) If
a1 , . . . , an > 0, define the “L-shaped domain”

                           L(a1 , . . . , an ) = µ ∈ Rn≥0 µj ≤ aj for some j .
                                                


Lemma 3.1. If a1 , . . . , an > 0, then
                                                           n
                                                         X
                                   cCH
                                    1  XL(a1 ,...,a n )  =   aj .
                                                            j=1

Proof. Observe that

                       Rn≥0 \ L(a1 , . . . , an ) = (a1 , ∞) × · · · × (an , ∞).

                                                  8
is convex. Thus XL(a1 ,...,an ) satisfies all the conditions in the definition of “concave toric
domain”, except that it is not compact.
    A formula for cCH
                   k of a concave toric domain is given in [12, Thm. 1.14]. The k = 1 case
of this formula asserts that if XΩ is a concave toric domain in R2n , then
                                                ( n               )
                                                 X
                            cCH
                              1 (XΩ ) = min          µi µ ∈ ∂+ Ω .                         (3.1)
                                                    i=1

By an exhaustion argument (see [12, Rmk. 1.3]), this result also applies to XL(a1 ,...,an ) . For
Ω = L(a1 , . . . , an ), the minimum in (3.1) is realized by µ = (a1 , . . . , an ).

Lemma 3.2. If XΩ is a monotone toric domain in R2n and if µ ∈ ∂+ Ω, then Ω ⊂
L(µ1 , . . . , µn ).

Proof. By an approximation argument we can assume without loss of generality that XΩ
is strictly monotone. Then ∂+ Ω is the graph of a positive function f over an open set
        n−1
U ⊂ R≥0      with ∂j f < 0 for j = 1, . . . , n − 1. It follows that if (µ01 , . . . , µ0n−1 ) ∈ U and
µj > µj for all j = 1, . . . , n − 1, then f (µ01 , . . . , µ0n−1 ) < f (µ1 , . . . , µn−1 ). Consequently
  0

Ω does not contain any point µ0 with µ0j > µj for all j = 1, . . . , n. This means that
Ω ⊂ L(µ1 , . . . , µn ). Figure 2 illustrates this inclusion for n = 2.

Proof of Theorem 1.11. For a > 0, consider the simplex
                                                         
                                                n
                                                 X        
                          ∆n (a) = µ ∈ Rn≥0         µi ≤ a .
                                                         
                                                          j=1


Observe that the toric domain X∆n (a) is the ball B 2n (a). Now let a > 0 be the largest real
number such that ∆n (a) ⊂ Ω; see Figure 2.
   We have B 2n (a) ⊂ XΩ , so by definition a ≤ cGr (XΩ ). Since cCH         1   is a normalized
symplectic capacity, cGrP(XΩ ) ≤ cCH
                                   1   (XΩ ).  By    the    maximality property of a, there exists
a point µ ∈ ∂+ Ω with nj=1 µj = a. By an approximation argument we can assume that
µ ∈ ∂+ Ω. By Lemma 3.2, XΩ ⊂ XL(µ1 ,...,µn ) . By the monotonicity of cCH    1   and Lemma 3.1,
we then have
                                                                n
                                                              X
                        cCH
                         1  (X Ω ) ≤ cCH
                                      1   X   L(µ1 ,...,µn )  =   µj = a.
                                                                 j=1

Combining the above inequalities gives cGr (XΩ ) = cCH
                                                    1 (XΩ ) = a.




4    ECH capacities
We now recall some facts about ECH capacities which we will use to prove Theorem 1.7.

Definition 4.1. A weakly convex toric domain in R4 is a compact toric domain XΩ ⊂ R4
such that Ω is convex, and ∂+ Ω is an arc with one endpoint on the positive µ1 axis and one
endpoint on the positive µ2 axis. See Figure 1d.


                                                    9
                           µ2
                                                   Ω      L(µ1 , µ2 )
                                                ∆2 (a)
                                            µ1
               Figure 2: The inclusions ∆n (a) ⊂ Ω ⊂ L(µ1 , . . . , µn ) for n = 2

Theorem 4.2 (Cristofaro-Gardiner [7]). In R4 , let XΩ be a concave toric domain, and let
XΩ0 be a weakly convex toric domain. Then there exists a symplectic embedding int(XΩ ) ,→
                                                                                                s
XΩ0 if and only if cECH
                    k   (XΩ ) ≤ cECH
                                 k   (XΩ0 ) for all k ≥ 0.
    To make use of this theorem, we need some formulas to compute the ECH capacities
cECH
 k   .  To start, consider a 4-dimensional concave toric domain XΩ . Associated to XΩ is a
“weight sequence” W (XΩ ), which is a finite or countable multiset of positive real numbers
defined in [5], see also [23], as follows. Let r be the largest positive real number such that
the triangle ∆2 (r) ⊂ Ω. We can write Ω \ ∆2 (r) = Ω   e1 t Ω
                                                            e 2 , where Ω
                                                                        e 1 does not intersect the
µ2 -axis and Ω2 does not intersect the µ1 -axis. It is possible that Ω
              e                                                          e 1 and/or Ωe 2 is empty.
After translating
                the         of Ω1 or Ω2 by (−r, 0) and (0, −r) and multiplying them by
                      closures
                                     e     e
                1 1           1 0
the matrices           and          , respectively, we obtain two new domains Ω1 and Ω2 in
                0 1           1 1
R2≥0 such that XΩ1 and XΩ2 are concave toric domains. We then inductively define
                             W (XΩ ) = (r) ∪ W (XΩ1 ) ∪ W (XΩ2 ),                           (4.1)
where ‘∪’ denotes the union of multisets, and the term W (XΩi ) is omitted if Ωi is empty.
    Let us call two subsets of R2 “affine equivalent” if one can be obtained from the other
by the composition of a translation and an element of GL(2, Z). If W (XΩ ) = (a1 , a2 , . . .),
then the domain Ω is canonically decomposed into triangles, which are affine equivalent to
the triangles ∆2 (a1 ), ∆2 (a2 ), . . . and which meet only along their edges; the first of these
triangles is ∆2 (r). See [19, §3.1] for more details. We now recall the “Traynor trick”:
Proposition 4.3. [29] If T ⊂ R2≥0 is a triangle affine equivalent to ∆2 (a), then there is a
symplectic embedding int(B 4 (a)) ,→ Xint(T ) .
                                    s
As a result, there is a symplectic embedding
                                    a
                                      int(B 4 (ai )) ⊂ XΩ .
                                            i

Consequently, by the monotonicity property of ECH capacities, we have
                                               !
                                a
                          ECH
                         ck        int(B (ai )) ≤ cECH
                                         4
                                                   k   (XΩ ).                               (4.2)
                                        i


                                                         10
Theorem 4.4 ([5]). If XΩ is a four-dimensional concave toric domain with weight expan-
sion W (XΩ ) = (a1 , a2 , . . .), then equality holds in (4.2).
       To make this more explicit, we know from [17] that6
                                          !
                          a                          X
                     ECH            4
                    ck        int(B (ai )) = sup         cECH       4
                                                           ki (int(B (ai )))                         (4.3)
                              i                        k1 +···=k   i

and
                              cECH
                               k   (int(B 4 (a))) = cECH
                                                     k   (B 4 (a)) = da,                             (4.4)
where d is the unique nonnegative integer such that

                                          d2 + d ≤ 2k ≤ d2 + 3d.

       To state the next lemma, given a1 , a2 > 0, define the polydisk
                                                                      
                                            2       2            2
                        P (a1 , a2 ) = z ∈ C π|z1 | ≤ a1 , π|z2 | ≤ a2 .

This is a convex toric domain XΩ0 where Ω0 is a rectangle of side lengths a1 and a2 .
Lemma 4.5. Let XΩ be a four-dimensional concave toric domain. Let (a, 0) and (0, b) be
the points where ∂+ Ω intersects the axes. Let µ be a point on ∂+ Ω minimizing µ1 + µ2 , and
write r = µ1 + µ2 . Then there exists a symplectic embedding

                                      int(XΩ ) ,→ P (r, max(b, a − r)).
                                               s

Proof. One might hope for a direct construction using some version of “symplectic folding”
[24], but we will instead use the above ECH machinery. By Theorem 4.2, it is enough to
show that
                           cECH
                            k   (XΩ ) ≤ cECH
                                         k   (P (r, max(b, a − r))                    (4.5)
for each nonnegative integer k.
    Consider the weight expansion W (XΩ ) = (a1 , a2 , . . .) where a1 = r. The decomposition
of Ω into triangles corresponding to the weight expansion consists of the triangle ∆2 (r),
plus some additional triangles in the triangle with corners (0, r), (µ1 , µ2 ), (0, b), plus some
additional triangles in the triangle with corners (µ1 , µ2 ), (r, 0), (a, 0); see Figure 3a. The
latter triangle is affine equivalent to the triangle with corners (µ1 , µ2 ), (r, 0), (r, a − r); see
Figure 3b. This allows us to pack triangles affine equivalent to ∆2 (a1 ), ∆2 (a2 ), . . . into the
rectangle with horizontal side length r and vertical side length max(b, a − r). Thus by the
Traynor trick, we have a symplectic embedding
                             a
                                 int(B(ai )) ,→ P (r, max(b, a − r)).
                                                   s
                                  i

Then Theorem 4.4 and the monotonicity of ECH capacities imply (4.5).

   6
    For the sequence of numbers ai coming from a weight expansion, or for any finite sequence, the supremum
in (4.3) is achieved, so we can write ‘max’ instead of ‘sup’.


                                                       11
                                                               a−r
       b                                                         b


       r


      µ2
                µ1             a                                                r
           (a) Weights of XΩ                                    (b) Ball packing into a polydisk

               Figure 3: Embedding a concave toric domain into a polydisk


Proof of Theorem 1.7. Let r be the largest positive real number such that ∆2 (r) ⊂ Ω. We
have B 4 (r) ⊂ XΩ , so r ≤ cGr (XΩ ), and we just need to show that cZ (XΩ ) ≤ r.
    Let µ be a point on ∂+ Ω such that µ1 + µ2 = r. By an approximation argument, we can
assume that XΩ is strictly monotone, so that the tangent line to ∂+ Ω at µ is not horizontal
or vertical. Then we can find a, b > r such that Ω is contained in the quadrilateral with
vertices (0, 0), (a, 0), (µ1 , µ2 ), and (0, b). It then follows from Lemma 4.5 that there exists
a symplectic embedding int(XΩ ) ,→ P (r, R) for some R > 0. Since P (r, R) ⊂ Z 4 (r), it
                                       s
follows that cZ (XΩ ) ≤ r.


5    A family of non-monotone toric examples
We now study a family of examples of non-monotone toric domains, and we determine when
they satisfy the conclusions of Conjecture 1.1 or Conjecture 1.2.
    For 0 < a < 1/2, let Ωa be the convex polygon with corners (0, 0), (1 − 2a, 0), (1 − a, a),
(a, 1 − a) and (0, 1 − 2a), and write Xa = XΩa ; see Figure 4a. Then Xa is a weakly convex
(but not monotone) toric domain.

Proposition 5.1. Let 0 < a < 1/2. Then the Gromov width and cylindrical capacity of
Xa are given by

                                   cGr (Xa ) = min(1 − a, 2 − 4a),                                 (5.1)
                                   cZ (Xa ) = 1 − a.                                               (5.2)

Corollary 5.2. Let 0 < a < 1/2 and let Xa be as above. Then:

 (a) The conclusion of Conjecture 1.1 holds for Xa , i.e. all normalized symplectic capacities
      defined for Xa agree, if and only if a ≤ 1/3.

 (b) The conclusion of Conjecture 1.2 holds
                                         p for Xa , i.e. every normalized symplectic capacity
      c defined for Xa satisfies c(Xa ) ≤ 2 Vol(Xa ), if and only if a ≤ 2/5.


                                                 12
Proof of Corollary 5.2. (a) By Lemma 1.3, we need to check that cGr (Xa ) = cZ (Xa ) if and
only if a ≤ 1/3. This follows directly from (5.1) and (5.2).
    (b) Since cZ is the largest normalized symplectic capacity, the conclusion of Conjecture
1.2 holds for Xa if and only if
                                             p
                                   cZ (Xa ) ≤ 2 Vol(Xa ).                              (5.3)
By equation (1.2), we have
                                                  1 − 4a2
                                         Vol(XΩa ) =      .
                                                     2
It follows from this and (5.2) that (5.3) holds if and only if a ≤ 2/5.

Remark 5.3. To recap, the conclusion of Conjecture 1.1 holds if and only if the ratio
cZ /cGr = 1, and the conclusion of Conjecture 1.2 holds if and only if the ratio cnZ /(n! Vol) ≤
1. The above calculations show that both of these ratios for Xa go to infinity as a → 1/2.




                                                                                         µ1 = µ2
                                                              M2




            Ωa
                                                                           Ω

                  1 − 2a       1                                                     M1

           (a) The domain Ωa                             (b) A domain to which Theorem 5.6 applies

                                     Figure 4: Some domains

   To prove Proposition 5.1, we will use the following formula for the ECH capacities of
a weakly convex toric domain XΩ . Let r be the smallest positive real number such that
Ω ⊂ ∆2 (r). Then ∆2 (r) \ Ω = Ω  e1 t Ω
                                      e 2 where Ωe 1 does not intersect the µ2 -axis, and Ω
                                                                                          e 2 does
not intersect the µ1 -axis. It is possible that Ω1 and/or Ω2 is empty. As in the discussion
                                                  e           e
preceding (4.1), the closures of Ωe 1 and Ωe 2 are affine equivalent to domains Ω1 and Ω2 such
that XΩ1 and XΩ2 are concave toric domains. Denote the union (as multisets) of their
weight sequences by
                                W (XΩ1 ) ∪ W (XΩ2 ) = (a1 , . . .).
We then have:
Theorem 5.4 (Choi–Cristofaro-Gardiner [7]). If XΩ is a four-dimensional weakly convex
toric domain as above, then
                               (                                  !)
                                                      a
                  ECH            ECH   4        ECH         4
                                            
                 ck (XΩ ) = inf ck+l B (r) − cl           B (ai )    .           (5.4)
                                   l≥0
                                                                i


                                                 13
    We need one more lemma, which follows from [21, Cor. 4.2]:
Lemma 5.5. Let µ1 , µ2 ≥ a > 0. Let Ω be the “diamond” in R2≥0 given by the convex hull
of the points (µ1 ± a, µ2 ) and (µ1 , µ2 ± a). Then there is a symplectic embedding
                                            int(B 4 (2a)) ,→ XΩ .
                                                             s

Proof of Proposition 5.1. To prove (5.1), we first describe the ECH capacities of Xa . In
the formula (5.4) for Xa , we have r = 1, while the weight expansions of Ω1 and Ω2 are
both (a, a); the corresponding triangles are shown in Figure 5(b). Thus by Theorem 5.4
and equation (4.3), we have
                                                             4
                                (                                            )
                                                             X
             cECH                  cECH               4
                                                               cECH B 4 (a) .
                                                                          
              k   (Xa ) = inf       k+l1 +l2 +l3 +l4 B (1) −    li                  (5.5)
                            l1 ,...,l4 ≥0
                                                                         i=1

We also note from (4.4) that
                  cECH
                   1   (B 4 (r)) = cECH
                                    2   (B 4 (r)) = r,                cECH
                                                                       5   (B 4 (r)) = 2r.
    Taking k = 1 and (l1 , . . . , l4 ) = (1, 0, 0, 0) in equation (5.5), we get
                                            cECH
                                             1   (XΩa ) ≤ 1 − a.                               (5.6)
Taking k = 1 and (l1 , . . . , l4 ) = (1, 1, 1, 1) in equation (5.5), we get
                                            cECH
                                             1   (XΩa ) ≤ 2 − 4a.                              (5.7)
By (5.6) and (5.7) and the fact that cECH
                                      1    is a normalized symplectic capacity, we conclude
that
                              cGr (XΩa ) ≤ min(1 − a, 2 − 4a).                         (5.8)
    To prove the reverse inequality to (5.8), suppose first that 0 < a ≤ 1/3. It is enough to
prove that there exists a symplectic embedding int(B 4 (1 − a)) ,→ XΩa . By Theorem 4.2, it
                                                                  s
is enough to show that
                              cECH
                               k    (B 4 (1 − a)) ≤ cECH
                                                     k   (XΩa )
for all nonnegative integers k. By equation (5.5), the above inequality is equivalent to
                                            4
                                            X
                 cECH
                  k   (B 4 (1   − a)) +           cECH
                                                   li  (B 4 (a)) ≤ cECH                4
                                                                    k+l1 +l2 +l3 +l4 (B (1))   (5.9)
                                            i=1

for all nonnegative integers k, l1 , . . . , l4 ≥ 0. To prove (5.9), by the monotonicity of ECH
capacities and the disjoint union formula (4.3), it suffices to find a symplectic embedding
                                                             !
                                                   a
                           int B 4 (1 − a) t          B 4 (a) ,→ B 4 (1).
                                                                     s
                                                        4

This embedding exists by the Traynor trick (Proposition 4.3) using the triangles shown in
Figure 5(a).
   Finally, when 1/3 ≤ a < 1/2, it is enough to show that there exists a symplectic
embedding int(B 4 (2 − 4a)) ,→ XΩa . This exists by Lemma 5.5 using the diamond shown
                                   s
in Figure 5(b).
    This completes the proof of (5.1). Equation (5.2) follows from Theorem 5.6 below.

                                                        14
       1                                                              1



                                                                   1−a


   a

                                                                                      1−a          1
              (a) 0 < a ≤ 1/3                                               (b) 1/3 ≤ a < 1/2

                                        Figure 5: Ball packings


Theorem 5.6. Let XΩ ⊂ R4 be a weakly convex toric domain, see Definition 4.1. For
j = 1, 2, let
                           Mj = max{µj | µ ∈ Ω}.
Assume that there exists (M1 , µ2 ) ∈ ∂+ Ω with µ2 ≤ M1 , and that there exists (µ1 , M2 ) ∈
∂+ Ω with µ1 ≤ M2 . Then
                                  cZ (XΩ ) = min(M1 , M2 ).

    That is, under the hypotheses of the theorem, see Figure 4b, an optimal symplectic
embedding of XΩ into a cylinder is given by the inclusion of XΩ into (π|z1 |2 ≤ M1 ) or
(π|z2 |2 ≤ M2 ).

Proof. From the above inclusions we have cZ (XΩ ) ≤ min(M1 , M2 ). To prove the reverse
inequality, suppose that there exists a symplectic embedding

                                             XΩ ,→ Z 4 (R).                                            (5.10)
                                                   s

We need to show that R ≥ min(M1 , M2 ). To do so, we will use ideas7 from [18].
   Let ε > 0 be small. Let (A, 0) and (0, B) denote the endpoints of ∂+ Ω. By an approxi-
mation argument, we can assume that ∂+ Ω is smooth, and that ∂+ Ω has positive slope less
than ε near (A, 0) and slope greater than ε−1 near (0, B). As in the proof of Proposition 1.8,
there are then three types of Reeb orbits on ∂XΩ :

 (i) There is a simple Reeb orbit whose image is the circle with π|z1 |2 = A and z2 = 0. This
      Reeb orbit has symplectic action (period) equal to A, and rotation number 1 − ε−1 .
    7
      The main theorem in [18] gives a general obstruction to a symplectic embedding of one four-dimensional
convex toric domain into another, which sometimes goes beyond the obstruction coming from ECH capac-
ities. This theorem can be generalized to weakly convex toric domains; but rather than carry out the full
generalization, we will just explain the simple case of this that we need.




                                                       15
 (ii) There is a simple Reeb orbit whose image is the circle with z1 = 0 and π|z2 |2 = B.
      This Reeb orbit has symplectic action B and rotation number 1 − ε−1 .

 (iii) For each point µ ∈ ∂+ Ω where ∂+ Ω has rational slope, there is an S 1 family of simple
       Reeb orbits in the torus where π(|z1 |2 , |z2 |2 ) = µ. If ν = (ν1 , ν2 ) is the outward
       normal vector to ∂+ Ω at µ, scaled so that ν1 , ν2 are relatively prime integers, then
       these Reeb orbits have rotation number ν1 + ν2 and symplectic action µ · ν. See [12,
       §2.2].

   We claim now that:

 (*) Every Reeb orbit on ∂XΩ with positive rotation number has symplectic action at least
      min(M1 , M2 ).

To prove this claim, we only need to check the type (iii) simple Reeb orbits where ν1 +ν2 ≥ 1.
For such an orbit we must have ν1 ≥ 1 or ν2 ≥ 1. Suppose first that ν1 ≥ 1. By the
hypotheses of the theorem there exists µ02 such that (M1 , µ02 ) ∈ ∂+ Ω and M1 ≥ µ02 . Since
Ω is convex and ν is an outward normal at µ, the symplectic action

          µ · ν ≥ (M1 , µ02 ) · ν = M1 + (ν1 − 1)(M1 − µ02 ) + (ν1 + ν2 − 1)µ02 ≥ M1 .

Likewise, if ν2 ≥ 1, then the symplectic action µ · ν ≥ M2 .
   As in [18, §5.3], starting from the symplectic embedding (5.10), by replacing XΩ with
an appropriate subset and replacing Z 4 (R) with an appropriate superset, we obtain a
symplectic embedding X 0 ,→ int(Z 0 ), where:
                             s

   • Z 0 is an ellipsoid whose boundary has one simple Reeb orbit γ+ with symplectic action
     A(γ+ ) = R + ε and Conley-Zehnder index CZ(γ+ ) = 3, another simple Reeb orbit
     with very large symplectic action, and no other simple Reeb orbits.

   • X 0 is a (non-toric) star-shaped domain with smooth boundary, all of whose Reeb
     orbits are nondegenerate. Every Reeb orbit on ∂X 0 with rotation number greater
     than or equal to 1 has action at least min(M1 , M2 ) − ε.

    The symplectic embedding gives rise to a strong symplectic cobordism W whose positive
boundary is ∂Z 0 and whose negative boundary is ∂X 0 . The argument in [18, §6] shows that
for a generic “cobordism-admissible” almost complex structure J on the “completion” of
W , there exists an embedded J-holomorphic curve u with one positive end asymptotic to
the Reeb orbit γ+ in ∂Z 0 , negative ends asymptotic to some Reeb orbits γ1 , . . . , γm in ∂X 0 ,
and Fredholm index ind(u) = 0. The Fredholm index is computed by the formula
                                                        m
                                                        X
                        ind(u) = 2g + [CZ(γ+ ) − 1] −          [CZ(γi ) − 1]               (5.11)
                                                         i=1

where g denotes the genus of u. Furthermore, since J-holomorphic curves decrease sym-
plectic action, we have
                                          m
                                          X
                                 A(γ+ ) ≥     A(γi ).                           (5.12)
                                                i=1


                                               16
   We claim now that at least one of the Reeb orbits γi has action at least min(M1 , M2 )−ε.
Then the inequality (5.12) gives

                                      R + ε ≥ min(M1 , M2 ) − ε,

and since ε > 0 was arbitrarily small, we are done.
    To prove the above claim, suppose to the contrary that all of the Reeb orbits γi have
action less than min(M1 , M2 ) − ε. Then all of the Reeb orbits γi have rotation number
ρ(γi ) < 1, which means that they all have Conley-Zehnder index CZ(γi ) ≤ 1. It now follows
from (5.11) that ind(u) ≥ 2, which is a contradiction8 .


6     The first Ekeland-Hofer capacity
The goal of this section is to (re)prove the following theorem. This is well-known in the
community and is attributed to Ekeland, Hofer and Zehnder [9, 15]. It was first mentioned
by Viterbo in [30, Proposition 3.10].

Theorem 6.1 (Ekeland-Hofer-Zehnder). Let W ⊂ R2n be a compact convex domain with
smooth boundary. Then
                              cEH
                               1 (W ) = Amin (W ).

    We start by recalling the definition of the first Ekeland-Hofer capacity cP       EH . Let E =
                                                                                      1
H 1/2 (S 1 , R2n ). That is, if x ∈ L2 (S 1 , R2n ) is written as a Fourier series x = k∈Z e2πikt xk
where xk ∈ R2n , then                               X
                                    x ∈ E ⇐⇒            |k||xk |2 < ∞.
                                                    k∈Z

Recall that there is an orthogonal splitting E = E + ⊕ E 0 ⊕ E − and orthogonal projections
P ◦ : E → E ◦ where ◦ = +, 0, −. The symplectic action of x ∈ E is defined to be
                                         1
                                           kP + xk2H 1/2 − kP − xk2H 1/2 .
                                                                        
                               A(x) =
                                         2
                                                                                 R
It follows from a simple calculation that if x is smooth, then A(x) =             x λ0 ,   where λ0 denotes
the standard Liouville form on R2n .
    Let H denote the set of H ∈ C ∞ (R2n ) such that

    • H|U ≡ 0 for some U ⊂ R2n open,

    • H(z) = c|z|2 for z >> 0 where c 6∈ {π, 2π, 3π, . . . }.

For H ∈ H, the action functional AH : H 1/2 (S 1 , R2n ) → R is defined by
                                                          Z   1
                                   AH (x) = A(x) −                H(x(t))dt.                           (6.1)
                                                          0
   8
     One way to think about the information that we are getting out of (5.11), as well as the general sym-
plectic embedding obstruction in [18], is that we are making essential use of the fact that every holomorphic
curve has nonnegative genus.



                                                     17
Note that the natural action of S 1 on itself induces an S 1 -action on E. Let Γ be the set of
homeomorphisms h : E → E such that h can be written as

                      h(x) = eγ+ (x) P + x + P 0 x + eγ− (x) P − x + K(x),

where γ+ , γ− : E → R are continuous, S 1 -invariant and map bounded sets to bounded
sets, and K : E → E is continuous, S 1 -equivariant and maps bounded sets to precompact
sets. Let S + denote the unit sphere in E + with respect to the H 1/2 norm. The first
Ekeland-Hofer capacity is defined in [9] by

                        cEH
                         1 (W ) = inf{cH,1 | H ∈ H, W ⊂ supp H},

where

        cH,1 = inf{sup AH (ξ) | ξ ⊂ E is S 1 -invariant, and ∀h ∈ Γ : h(ξ) ∩ S + 6= ∅}.

Proof of Theorem 6.1. Since W is star-shaped, there is a unique differentiable function
r : R2n → R which is C ∞ in R2n \ {0} satisfying r(cz) = c2 r(z) for c ≥ 0 such that

                                  W = {z ∈ R2n | r(z) ≤ 1},
                                ∂W = {z ∈ R2n | r(z) = 1}.

Let α = Amin (W ) and fix ε > 0. Let f ∈ C≥0∞ (R) be a convex function such that f (r) = 0

for r ≤ 1 and f (r) = Cr − (α + ε) for r ≥ 2 for some constant C > α. In particular,

                              f (r) ≥ Cr − (α + ε),          for all r.                   (6.2)

We now choose a convex function H ∈ C ∞ (R2n ) such that

                  H(z) = f (r(z)),       if r(z) ≤ 2,
                  H(z) ≥ f (r(z)),       for all z ∈ R2n ,                                (6.3)
                  H(z) = c |z|2 ,        if z >> 0 for some c ∈ R>0 \ πZ.

Let x0 ∈ E be an action-minimizing Reeb orbit on ∂W , reparametrized as a map x0 :
R/Z = S 1 → R2n of speed α, so that A(x0 ) = α and r(x0 ) ≡ 1 and ẋ0 = αJ∇r(x0 ). From
a simple calculation we deduce that x0 is a critical point of the functional Ψ : E → R
defined by                                   Z         1
                               Ψ(x) = A(x) − α             r(x(t)) dt.                    (6.4)
                                                   0

Observe that Ψ(cx) = c2 Ψ(x) for c ≥ 0. So sx0 is a critical point of Ψ for all s ≥ 0. Let
ξ = [0, ∞) · P + x0 ⊕ E 0 ⊕ E − .
   We now claim that Ψ(x) ≤ 0 for all x ∈ ξ. To prove this, let ξs = sP + x0 ⊕ E 0 ⊕ E − .
Observe that Ψ|ξs is a concave function. Since sx0 is a critical point of Ψ|ξs it follows that
max Ψ(ξs ) = Ψ(sx0 ) = s2 Ψ(x0 ) = 0.
   From (6.1), (6.2), (6.3) and (6.4) we obtain
                                                   Z 1
                   AH (x) ≤ Ψ(x) + α + ε + (C − α)     r(x(t)) dt ≤ α + ε.
                                                            0


                                              18
Note that ξ is S 1 -invariant. Moreover it is proven in [8] that h(ξ) ∩ S + 6= ∅ for all h ∈ Γ.
So cH,1 ≤ α + ε. Hence cEH 1 (W ) ≤ α + ε for all ε > 0. Therefore

                                        cEH
                                         1 (W ) ≤ α.

    To prove the reverse inequality, recall from [9, Prop. 2] that cEH
                                                                    1 (W ) is the symplectic
action of some Reeb orbit on ∂W . Thus

                                        cEH
                                         1 (W ) ≥ α.




References
 [1] A. Abbondandolo, B. Bramham, U. Hryniewicz, and P. Salamão, Systolic ratio, index
     of closed orbits and convexity for tight contact forms on the three-sphere, Compos.
     Math. 154 (2018), 2643–2680.

 [2] A. Abbondandolo and J. Kang, Symplectic homology of convex domains and Clarke’s
     duality, arXiv:1907.07779.

 [3] S. Artstein-Avidan and Y. Ostrover, Bounds for Minkowski billiard trajectories in
     convex bodies, IMRN 2014, 165–193.

 [4] S. Artstein-Avidan, R. Karasev, and Y. Ostrover, From symplectic measurements to
     the Mahler conjecture, Duke Math. J. 163 (2014), 2003–2022.

 [5] K. Choi, D. Cristofaro-Gardiner, D. Frenkel, M. Hutchings, and V.G.B. Ramos, Sym-
     plectic embeddings into four-dimensional concave toric domains, J. Topol. 7 (2014),
     1054–1076.

 [6] K. Cieliebak, H. Hofer, J. Latschev, and F. Schlenk, Quantitative symplectic geometry,
     in Dynamics, ergodic theory, and geometry, Math. Sci. Res. Inst. Publ. bf 54 (2007),
     1–44.

 [7] D. Cristofaro-Gardiner, Symplectic embeddings from concave toric domains into convex
     ones, J. Diff. Geom. 112 (2019), 199–232.

 [8] I. Ekeland and H. Hofer, Symplectic topology and Hamiltonian dynamics, Math. Z.
     200 (1989), 355–378.

 [9] I. Ekeland and H. Hofer, Symplectic topology and Hamiltonian dynamics II, Math. Z.
     203 (1990), 553-567.

[10] V. Ginzburg and J. Shon, On the filtered symplectic homology of prequantization bun-
     dles, Int. J. Math. 29 (2018), 1850071, 35pp.

[11] M. Gromov, Pseudoholomorphic curves in symplectic manifolds, Invent. Math. 82
     (1985), 307–347.

                                              19
[12] J. Gutt and M. Hutchings, Symplectic capacities from positive S 1 -equivariant symplec-
     tic homology, Algebr. Geom. Topol. 18 (2018), 3537–3600.

[13] D.     Hermann,     Non-equivalence     of   symplectic    capacities     for
     open    sets   with   restricted   contact  type     boundary,      preprint,
     www.math.u-psud.fr/~biblio/pub/1998/abs/ppo1998_32.html.

[14] H. Hofer, K. Wysocki, and E. Zehnder, The dynamics on three-dimensional strictly
     convex energy surfaces, Ann. Math. 148 (1998), 197–289.

[15] H. Hofer and E. Zehnder, Periodic solutions on hypersurfaces and a result by C.
     Viterbo, Invent. Math. 90 (1987), 1–9.

[16] H. Hofer and E. Zehnder, A new capacity for symplectic manifolds, Analysis, et cetera,
     405–427, Academic Press, Boston MA (1990).

[17] M. Hutchings, Quantitative embedded contact homology, J. Diff. Geom. 88 (2011),
     321–266.

[18] M. Hutchings, Beyond ECH capacities, Geom. Topol. 20 (2016), 1085–1126.

[19] M. Hutchings, ECH capacities and the Ruelle invariant, arXiv:1910.08260.

[20] K. Irie, Symplectic homology of fiberwise convex sets and homology of loop spaces,
     arXiv:1907.09749.

[21] J. Latschev, D. McDuff, and F. Schlenk, The Gromov width of 4-dimensional tori ,
     Geom. Topol. 17 (2013), 2813–1853.

[22] K. Mahler, Ein Übertragungsprinzip für konvexe Körper , Cas. Mat. Fys. 68 (1939),
     93–102.

[23] V.G.B. Ramos, Symplectic embeddings and the Lagrangian bidisk , Duke Math. J. 166
     (2017), 1703–1738.

[24] F. Schlenk, On symplectic folding, arXiv:math/9903086.

[25] F. Schlenk, Symplectic embedding problems, old and new , Bull. Amer. Math. Soc. 55
     (2017), 139–182.

[26] Kun Shi and Guangcun Lu, Some cases of the Viterbo conjecture and the Mahler one,
     arXiv:2008.04000.

[27] K. Siegel, Higher symplectic capacities, arXiv:1902.01490.

[28] K. Siegel, Computing higher symplectic capacities I , arXiv:1911.06466.

[29] L. Traynor, Symplectic packing constructions, J. Diff. Geom. 42 (1995), 411–429.

[30] C. Viterbo, Capacités symplectiques et applications (d’après Ekeland-Hofer, Gromov),
     Séminaire Bourbaki, volume 1988/89, Astérisque No. 177-178 (1989), 345–362.


                                            20
[31] C. Viterbo, Functors and computations in Floer homology with applications I Geom.
     Funct. Anal. 9 (1999), 985–1033.

[32] C. Viterbo, Metric and isoperimetric problems in symplectic geometry, J. Amer. Math.
     Soc. 13 (2000), 411–431.




                                           21
