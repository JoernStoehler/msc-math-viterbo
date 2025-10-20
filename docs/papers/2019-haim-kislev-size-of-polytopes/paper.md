---
source: arXiv:1712.03494
fetched: 2025-10-20
---
# On the symplectic size of convex polytopes

                                               On the symplectic size of convex polytopes
                                                                        Pazit Haim-Kislev

                                                                          January 7, 2019
arXiv:1712.03494v3 [math.SG] 4 Jan 2019




                                                                               Abstract
                                                    In this paper we introduce a combinatorial formula for the Ekeland-
                                                Hofer-Zehnder capacity of a convex polytope in R2n . One application of
                                                this formula is a certain subadditivity property of this capacity.



                                          1    Introduction and main results
                                          Symplectic capacities are well studied invariants in symplectic geometry which,
                                          roughly speaking, measure the “symplectic size” of sets (see for example [5] and
                                          [13]). The first appearance of a symplectic capacity in the literature (although
                                          not under this name) was in Gromov [8] where the theory of pseudo-holomorphic
                                          curves was developed and implemented. The concept of a symplectic capacity
                                          was later formalized by Ekeland and Hofer in [7], where they also gave additional
                                          examples using Hamiltonian dynamics. Since then, many other examples of
                                          symplectic capacities were constructed. They are divided, roughly speaking,
                                          into ones which are related to obstructions for symplectic embeddings, ones
                                          which are defined using pseudo-holomorphic curves, and ones related to the
                                          existence and behaviour of periodic orbits in Hamiltonian dynamics.
                                              Two well known examples of symplectic capacities are the Ekeland-Hofer
                                          capacity defined in [7] and the Hofer-Zehnder capacity defined in [9]. These
                                          two capacities are known to coincide on the class of convex bodies in R2n ([19],
                                          Proposition 3.10 and [9], Proposition 4). Moreover, in this case they are equal
                                          to the minimal action of a closed characteristic on the boundary of the body.
                                          In what follows we refer to this quantity as the Ekeland-Hofer-Zehnder capacity
                                          (abbreviate by EHZ capacity). See Section 2 below, for the definition of a closed
                                          characteristic, and for a generalization of this definition to polytopes in R2n . We
                                          remark that even on the special class of convex bodies in R2n , there are very
                                          few methods to explicitly calculate symplectic capacities, specifically the EHZ
                                          capacity, and it is in general not an easy problem to find closed characteristics
                                          and in particular the minimal ones (cf. [16]). The goal of this paper is to give a
                                          combinatorial formula for the EHZ capacity for convex polytopes, and discuss
                                          some of its applications.
                                             To state our result we introduce some notations. We work in R2n with the
                                          standard symplectic structure ω. Let K ⊂ R2n be a convex polytope with a
                                          non-empty interior. Denote the number of (2n − 1)-dimensional facets of K
                                          by FK , and the facets by {Fi }F
                                                                         i=1 . Let hK (y) := supx∈K hx, yi be the support
                                                                          K




                                                                                   1
function of K. Denote by ni the unit outer normal to Fi , and hi = hK (ni ) the
“oriented height” of Fi . Finally, let SFK be the symmetric group on FK letters.
Our main result is the following.
Theorem 1.1. For every convex polytope K ⊂ R2n
                                                                    −1
             1                    X
  cEHZ (K) =        max                 βσ(i) βσ(j) ω(nσ(i) , nσ(j) )    ,                            (1)
             2 σ∈SFK ,(βi )∈M (K)
                                           1≤j<i≤FK

where                     (                                                             )
                                                   FK
                                                   X                  FK
                                                                      X
               M (K) =      (βi )FK
                                 i=1   : βi ≥ 0,         βi hi = 1,         βi n i = 0 .
                                                   i=1                i=1


   Note that the maximum runs over SFK , which is a finite set of permutations,
and over M (K), which is a finite dimensional convex polytope. Hence the
combinatorial nature of the formula. Moreover, this formula allows us (up to
computational power) to calculate the capacity of every convex polytope using a
computer. We also note that from continuity of the EHZ capacity, some possible
applications of Theorem 1.1 about properties of the EHZ capacity on polytopes
are automatically extended to all convex bodies (cf. Theorem 1.8 below).
    For a centrally symmetric convex polytope K (i.e., when K = −K), the
above formula can be slightly simplified. In this case one can write the normals
to the (2n−1)-dimensional facets of K as {n1 , . . . , nF0 , −n1 , . . . , −nF0 } , where
                                                                      K                     K
        FK
F0K =    2 .

Corollary 1.2. For a centrally symmetric convex polytope K ⊂ R2n ,

                                                                                           −1
               1                                  X
    cEHZ (K) =          max                                 βσ(i) βσ(j) ω(nσ(i) , nσ(j) )        ,
               4 σ∈SF0K ,(βi )∈M 0 (K)
                                             1≤j<i≤F0K

where                                                       
                                              F0K
                                        0
                                        FK
                                              X             1
                         M 0 (K) = (βi )i=1 :     |βi |hi =    .
                                  
                                              i=1
                                                            2

Remark 1.3. We note that Formula (1) in Theorem 1.1 can be written as

                                                     FK
                                                                               !2
                               1                     X
                    cEHZ (K) =             min           βi hK (ni )                ,
                               2 (βi ,ni )Fi=1
                                            K ∈M (K)
                                                2    i=1

where
               (                                                                                      )
                   (βi , ni )FK
                                : β ≥ 0, (ni )FK
                                                 are different outer normals to K
 M2 (K) =          PFK i=1 i              P i=1                                                           .
                     i=1 βi ni = 0,          1≤j<i≤FK βi βj ω(ni , nj ) = 1


   In this form of the formula for cEHZ (K), instead of the permutation σ ∈ SFK
that appeared in (1), we minimize over different orders of the normals, by going
over different sequences (ni )F
                              i=1 . (We refer to Section 4 for the details.)
                                K




                                                   2
Remark 1.4. As shown in [1], using Clarke’s dual action principle (see [6]),
it is possible to express the EHZ capacity of any convex body K ⊂ R2n (not
necessarily a polytope) as
                                         "     Z 1            #−1
                                       1
                            cEHZ (K) =     sup     h−J ż, zi     ,
                                       2 z∈Ee 0

where                                                  Z   1                     
                Ee =       z ∈ W 1,2 ([0, 1], R2n ) :           żdt = 0, ż ∈ K ◦ ,
                                                        0
  ◦             2n
K = {y ∈ R : hx, yi ≤ 1, for every x ∈ K} is the polar body of K, and J
is the standard complex structure in R2n . When discretizing this formula, one
gets a formula which is similar to the one we get in Theorem 1.1. However, in
this discrete version, as opposed to Theorem 1.1, one needs to maximize over
an infinite dimensional space of piecewise affine loops. The essence of Theorem
1.1, as will be described later, is that on the boundary of a convex polytope
there exists a minimizer with a very specific description, and this enables us to
maximize, roughly speaking, over a much smaller space.

      We turn now to describe the main ingredient in the proof of Theorem 1.1.
    Let K ⊂ R2n be a convex polytope, and let γ : [0, 1] → ∂K be a closed
characteristic (for the definition see Section 2). From the definition, if γ(t) ∈
int(Fi ), then γ̇(t) must be a positive multiple of Jni (except maybe for t in a
subset of [0, 1] of measure zero). Similarly, if γ(t) belongs to the intersection
of more than one facet, then γ̇(t) is a non-negative linear combination of Jni
for i in the participating facets. A priori, γ(t) could return to each facet and
each intersection of facets many times. For the purpose of finding the minimal
action on the boundary of a convex polytope, we may ignore these options by
the following.
Theorem 1.5. For every convex polytope K ⊂ R2n , there exists a closed char-
acteristic γ : [0, 1] → ∂K with minimal action such that γ̇ is piecewise constant
and is composed of a finite sequence of vectors, i.e. there exists a sequence of
vectors (w1 , . . . , wm ), and a sequence (0 = τ0 < . . . < τm−1 < τm = 1) so
that γ̇(t) = wi for τi−1 < t < τi . Moreover, for each j ∈ {1, . . . , m} there
exists i ∈ {1, . . . , FK } so that wj = Cj Jni , for some Cj > 0, and for each
i ∈ {1, . . . , FK }, the set {t : ∃C > 0, γ̇(t) = CJni } is connected, i.e. for every i
there is at most one j ∈ {1, . . . , m} with wj = Cj Jni . Hence there are at most
FK points of discontinuity in γ̇, and γ visits the interior of each facet at most
once.

    Theorem 1.1 follows from the combination of the existence of a simple closed
characteristic as described in Theorem 1.5, and Clarke’s dual action principle
(see Section 2 for the details).

Remark 1.6. There are examples for polytopes with action minimizing closed
characteristics which do not satisfy the properties of the closed characteristics
one gets from Theorem 1.5. One example, which can be easily generalized to any
convex polytope with an action minimizing closed characteristic passing through


                                                 3
a Lagrangian face, is the standard simplex in R4 where for example on the face
{x1 = 0} ∩ {x2 = 0} one is free to choose a non-trivial convex combination of
e3 and e4 as the velocity of an action minimizing closed characteristic, one can
also choose it to be equal to e3 for some time, and then to e4 , and then e3 again
so that the set {t : ∃C > 0, γ̇(t) = CJni } is not connected. See [15] for a full
description of the dynamics of action minimizing closed characteristics on the
standard simplex.

    As an application of Theorem 1.1 we solve a special case of the subadditivity
conjecture for capacities. This conjecture, raised in [3], which is related with
a classical problem from convex geometry known as Bang’s problem, can be
stated as follows:

Conjecture 1.7. If a convex body K ⊂ R2n is covered by a finite set of convex
bodies {Ki } then                   X
                         cEHZ (K) ≤   cEHZ (Ki ).
                                         i


   In Section 8 of [3], the motivation of this conjecture and its relation with
Bang’s problem is explained together with some examples. It is known that
when cutting the euclidean ball B ⊂ R2n with some hyperplane into K1 and K2 ,
one has cEHZ (B) = cEHZ (K1 ) + cEHZ (K2 ). The fact that cEHZ (B) ≥ cEHZ (K1 ) +
cEHZ (K2 ) was first proved in [20] using an argument involving pseudo-holomorphic
curves, and in [3] it is shown that cEHZ (B) ≤ cEHZ (K1 ) + cEHZ (K2 ). As a conse-
quence of Theorem 1.1 above, we are able to prove subadditivity for hyperplane
cuts of arbitrary convex domains.

Theorem 1.8. Let K ⊂ R2n be a convex body. Let n ∈ S 2n−1 , c ∈ R, and
H − = {x : hx, ni ≤ c}, H + = {x : hx, ni ≥ c}. Then for K1 = K ∩ H + and
K2 = K ∩ H − , we have

                       cEHZ (K) ≤ cEHZ (K1 ) + cEHZ (K2 ).

    The structure of the paper is the following. In Section 2 we recall some
relevant definitions. In Section 3 we prove Theorem 1.5, Theorem 1.1 and
Corollary 1.2, and in Section 4 we use Theorem 1.1 to prove Theorem 1.8.
    Acknowledgement: This paper is a part of the author’s thesis, being car-
ried out under the supervision of Professor Shiri Artstein-Avidan and Professor
Yaron Ostrover at Tel-Aviv university. I also wish to thank Roman Karasev
and Julian Chaidez for helpful comments and remarks. I am grateful to the
anonymous referee for a thorough review and very helpful comments and sug-
gestions. The work was supported by the European Research Council (ERC)
under the European Union Horizon 2020 research and innovation programme
[Grant number 637386], and by ISF grant number 667/18.




                                        4
2      Preliminaries

2.1     The EHZ capacity
Let R2n be equipped with the standard symplectic structure ω. A normalized
symplectic capacity on R2n is a map c from subsets U ⊂ R2n to [0, ∞] with the
following properties.

    1. If U ⊆ V , c(U ) ≤ c(V ),
    2. c(φ(U )) = c(U ) for any symplectomorphism φ : R2n → R2n ,
    3. c(αU ) = α2 c(U ) for α > 0,
    4. c(B 2n (r)) = c(B 2 (r) × R2n−2 ) = πr2 .

For a discussion on symplectic capacities and their properties see e.g. [5], [13]
and [10].
    As mentioned in the introduction, two important examples for symplectic
capacities are the Ekeland-Hofer capacity (see [7]) and the Hofer-Zehnder ca-
pacity (see [9]). On the class of convex bodies in R2n (i.e., compact convex sets
with non-empty interior), they coincide and we call the resulting function, the
EHZ capacity. Moreover, for a smooth convex body, the EHZ capacity equals
the minimal action of a closed characteristic on the boundary of the body. Since
the focus of this paper is the EHZ capacity, we omit the general definitions of
the Hofer-Zehnder and Ekeland-Hofer capacities, and define the EHZ capacity
directly.
    We start with the definition of a closed characteristic. Recall that the re-
striction of the standard symplectic form to the boundary of a smooth domain
∂Σ, defines a 1-dimensional subbundle ker(ω|∂Σ). A closed characteristic γ
on ∂Σ is an embedded circle in ∂Σ, whose velocity belongs to ker(ω|∂Σ), i.e.
ω(γ̇, v) = 0, ∀v ∈ T ∂Σ. This holds if and only if γ̇(t) is parallel to Jn, where
n is the outer normal to ∂Σ in the point γ(t), and J is the standard complex
structure.
    From the dynamical point of view, a closed characteristic is any reparametriza-
tion of a periodic solution to the Hamiltonian equation γ̇(t) = J∇H(γ(t)), for
a smooth Hamiltonian function H : R2n → R with H|∂Σ = c, and H|Σ ≤ c
for some c ∈ R a regular value of H. We call these periodic solutions closed
Hamiltonian trajectories.
    We recall that the action of a closed loop γ : [0, T ] → R2n is defined by
                                          Z   T
                                      1
                            A(γ) :=               hJγ(t), γ̇(t)idt,
                                      2   0

and it equals the symplectic area of a disc enclosed by γ.
    The EHZ capacity of a smooth convex body K ⊂ R2n is

            cEHZ (K) = min{A(γ) : γ is a closed characteristic on ∂K}.



                                                  5
It is known that the minimum is always attained (see [7], [10]). One can extend
this definition by continuity to non-smooth convex domains with non-empty
interior. We elaborate in the next section (see e.g. [11], [12]).


2.2    The case of convex polytopes
The explicit definition of the EHZ capacity above was given only for smooth
bodies, and extended by continuity to all convex domains with non-empty in-
terior. It turns out that also in the case of a non-smooth body, the capacity
is given by the minimal action of a closed characteristic on the boundary of K
(see [4]), however one then needs to discuss generalized closed characteristics.
Let us state this precisely here for the case of convex polytopes.
    Let K ⊂ R2n be a convex polytope. For the following discussion suppose that
the origin 0 belongs to K. Recall that we denote the (2n − 1)-dimensional facets
of K by {Fi }F                                                  FK
              i=1 , and their outward unit normals by {ni }i=1 . Let x ∈ ∂K. We
                K


define the outward normal cone of K at x to be NK (x) := R+ conv{ni : x ∈ Fi }
(for the definition of the outward normal cone for a general convex body see
[12]). Recall that W 1,2 ([0, 1], R2n ) is the Hilbert space of absolutely continuous
functions whose derivatives are square integrable. We equip this space with the
natural Sobolev norm:
                                 Z    1                       12
                                                  2         2
                     kzk1,2 :=             kz(t)k + kż(t)k dt     .
                                   0

Definition 2.1. A closed characteristic on ∂K is a closed loop γ ∈ W 1,2 ([0, 1], R2n )
which satisfies Im(γ) ⊂ ∂K, and γ̇(t) ∈ JNK (γ(t)) for almost every t ∈ [0, 1].

    We remark that the condition Im(γ) ⊂ ∂K can be weakened to γ(0) ∈ ∂K,
since the assumption on γ̇ and the fact that γ is a closed loop already imply
that γ(t) ∈ ∂K for each t (see [12]).
   Definition 2.1 also has a Hamiltonian dynamics interpretation. Let H be
a Hamiltonian function for which K is a sub-level set, and ∂K is a level set.
Just like in the smooth case, (generalized) closed Hamiltonian trajectories of
the Hamiltonian H on ∂K, are reparametrizations of closed characteristics on
∂K, and upto a reparametrization, every closed characteristic is a closed Hamil-
tonian trajectory, only instead of γ̇(t) = J∇H(γ(t)), the Hamiltonian equation
becomes an inclusion

                      γ̇(t) ∈ J∂H(γ(t)) almost everywhere,

where ∂H is the subdifferential of H (see e.g. [17]). We remark that if H is
smooth at the point x, then ∂H(x) = {∇H(x)}, and hence if H is smooth
the two Hamiltonian equations coincide. For simplicity, we shall work with a
specific Hamiltonian function. Denote the gauge function of K by
                                                      x
                             gK (x) = inf{λ :           ∈ K},
                                                      λ
                                        2                 2
and consider the Hamiltonian function gK  . Note that gK    |∂K = 1. For each
                         2
1 ≤ i ≤ FK let pi = J∇(gK )(x), for a point x ∈ int(Fi ). It is easily seen that


                                              6
                        2
the subdifferential of gK at the point x ∈ ∂K is equal to
                                       2
                               conv{∇(gK )|int(Fi ) : x ∈ Fi },

which implies
                                  2
                               J∂gK (x) = conv{pi : x ∈ Fi }.

    To conclude, for a convex polytope K ⊂ R2n , the EHZ capacity is the mini-
mal action over all periodic solutions γ ∈ W 1,2 ([0, T ], ∂K), to the Hamiltonian
inclusion:
                γ̇(t) ∈ conv{pi : γ(t) ∈ Fi } almost everywhere.


2.3      Clarke’s dual action principle
Let K ⊂ R2n be a convex body (not necessarily smooth). Recall that the
support function of K is hK (x) = sup{hy, xi; y ∈ K}. Note that hK is the
gauge function of K ◦ and that 4−1 gK
                                    2
                                      is the Legendre transform of h2K (see e.g.
[4]).
    Following Clarke (see [6]), we look for a dual variational principle where solu-
tions would correspond to closed characteristics (cf. [10, Section 1.5]). Consider
the problem                        Z            1
                                   min              h2K (−J ż(t))dt,
                                   z∈E      0
where
                                           Z   1                     Z     1                         
      E=       z ∈ W 1,2 ([0, 1], R2n ) :           ż(t)dt = 0,                h−J ż(t), z(t)idt = 1 .
                                            0                           0

Define                                              Z    1
                                            1
                               IK (z) =                      h2K (−J ż(t))dt.
                                            4        0
Let
         E † = z ∈ E : ∃α ∈ R2n such that 8IK (z)z + α ∈ ∂h2K (−J ż) .
              

This is the set of weak critical points of the functional IK (see [4]). The following
lemma is an adjustment of the dual action principle to the non-smooth case,
and it appears e.g., as Lemma 5.1 in [4].
Lemma 2.2. Let K ⊂ R2n be a convex polytope. There is a correspondence
between the set of closed characteristics γ on ∂K, and the set of elements z ∈
E † . Under this correspondence, there exist λ ∈ R+ , and b ∈ R2n so that z =
λγ + b and moreover A(γ) = 2IK (z). In particular, any minimizer z ∈ E of
IK (z) belongs to E † and therefore has a corresponding closed characteristic with
minimal action.


3     Action minimizing orbits on polytopes
We start with the proof of Theorem 1.5. Let us first describe the idea of the
proof. We start from a closed characteristic with minimal action, and consider


                                                         7
                                                                                       cpj4


                               P                                                        cpj3
                           c       aji pji
                                                              cpj1        cpj2




      T1                                     T2       T1                                                         T2
                I = (T1 , T2 )                                aj1 |I|        aj2 |I|      aj3 |I|      aj4 |I|
                           P
               ż(t) = c       aji pji                     ż(t) = cpj1   ż(t) = cpj2 ż(t) = cpj3 ż(t) = cpj4



Figure 1: Description of the first change to the loop z: We break a convex
combination and move in each velocity separately


its corresponding element z ∈ E † (see Lemma 2.2). We then approximate it
with a certain sequence of piecewise affine loops. By piecewise     Pmaffine we mean
that the velocity of the loop z can be written as ż(t) =             j=1 1Ij (t)wj for
almost every t ∈ [0, 1], where (Ij )m  j=1 is a partition of [0, 1] into intervals (see
Definition 3.2 below) and (wj )m    j=1 is a finite sequence of vectors which we call
the velocities of z. Our goal is to construct from each piecewise affine loop in
the approximating sequence a new simple loop in the sense of the requirements
of Theorem 1.5, i.e. that the sequence (wj )m    j=1 is composed of positive multiples
of Jni , where ni is some outer normal vector to a 2n − 1-dimensional facet
of K, and that for each i = 1, . . . FK there is at most one j so that wj is a
positive multiple of Jni . The limit of these simple loops gives us the desired
minimizer of IK and by invoking Lemma 2.2 again we get the desired closed
characteristic. In order to construct a simple loop from each piecewise affine
loop z, we make two changes to it which are described in Lemma 3.4 and Lemma
3.5 below. Recall that the velocities wj are positive linear combinations of Jni ,
i = 1, . .P
          . , FK , and additionally, maybe after a reparametrization, one can write
             l            Pl
wj = c i=1 aji pji , i=1 aji = 1, where c > 0 is a constant independent of i,
and pi , i = 1, . . . , FK are the vectors described in Section 2.2 above. The first
change, roughly speaking, takes a time segment I of the loop z where the velocity
is a convex combination of {c · pji }li=1 and changes it to a sequence of l segments
where in each segment the velocity is c·pji , and the time of each segment is aji |I|
(see Figure 1). In addition, we show that one can choose the order of {pji }li=1
                                      R1
to make sure that the value of 0 h−J ż, zidt does not decrease. The second
change changes the order of the velocities and, roughly speaking, moves all the
time segments where the velocities are proportional to a certain Jni to become
adjacent to one another (see Figure 2). This change thus ensures that the set
{t : ż(t) is a positive multiple of Jni } is connected for every i = 1, . . . , FK , i.e.
that for each i, there is at most one j so that wj is a positive multiple of Jni . In
                                                                         R1
addition, one can do this change while ensuring that the value of 0 h−J ż, zidt
                                                                    R1
does not decrease. Finally, after dividing the simple loop by 0 h−J ż, zidt, one
gets an element in E whose value under IK does not increase, and hence it is
still a minimizer. This loop, by virtue of Lemma 2.2, gives the required simple
closed characteristic.
   We begin by describing the piecewise affine approximation.



                                                  8
                                cpi

                                                                              cpi
               cpi




             ż(t) = cpi         ż(t) = cpi                             ż(t) = cpi
        T1                 T2   T3         T4                      T1             T2 + T4 − T3          T4


Figure 2: Description of the second change to the loop z: We bring segments of
the loop where it moves in the same velocity together


Lemma 3.1. Fix a set of vectors v1 , . . . , vk ∈ R2n . Suppose z ∈ W 1,2 ([0, 1], R2n )
satisfies that for almost every t ∈ [0, 1], one has ż(t) ∈ conv{v1 , . . . , vk }. Then
for every ε > 0, there exists a piecewise affine function ζ with kz − ζk1,2 < ε,
and so that ζ̇ is composed of vectors from the set conv{v1 , . . . , vk }, and ζ(0) =
z(0), ζ(1) = z(1).

Proof. Let ε > 0. Using [18], there exists a partition 0 = t1 < t2 < . . . < tm = 1
of [0, 1] so that the piecewise affine function ζ defined by the requirements that
for each i = 1, . . . , m−1, the restriction ζ|(ti ,ti+1 ) is affine, and ζ(ti ) = z(ti ), sat-
isfies that kz − ζk1,2 < ε. We are left with showing that ζ̇(t) ∈ conv{v1 , . . . , vk }.
Note that for t ∈ (ti , ti+1 ),
                                                                    R ti+1
                                        z(ti+1 ) − z(ti )               ti
                                                                               ż(t)dt
                                ζ̇(t) =                   =                                .
                                           ti+1 − ti
                                                ti+1 − ti
                                                         
                                                     (N )
It is a standard fact that there exists a sequence ξj                                                          so that
                                                                                           N ∈N,j∈{1,...,N }
 (N )
ξj −ti
ti+1 −ti      ∈ [ j−1  j
                   N , N ], and

                                     R ti+1                       N     (N )
                                      ti
                                                ż(t)dt           X ż(ξj )
                                                          = lim                        .
                                      ti+1 − ti            N →∞
                                                                  j=1
                                                                             N

                                PN ż(ξ(N ) )
Note that for each N , one has j=1 Nj         ∈ conv(Im(ż)) ⊂ conv{v1 , . . . , vk }.
This observation together with the fact that conv{v1 , . . . , vk } is closed, gives
ζ̇(t) ∈ conv{v1 , . . . , vk }.


Definition 3.2. We call a finite sequence of disjoint open intervals (Ii )mi=1 a
partition of [0, 1], if there exists an increasing sequence of numbers 0 = τ0 ≤
τ1 ≤ . . . ≤ τm = 1, with Ii = (τi−1 , τi ).

    The following proposition will be helpful later.
Proposition    3.3. Let z ∈ W 1,2 ([0, 1], R2n ) be a closed loop such that ż(t) =
  i=1 1Ii (t)wi almost everywhere, where (Ii = (τi−1 , τi ))i=1 is a partition of
Pm                                                            m




                                                            9
[0, 1], and w1 , . . . , wm ∈ R2n . Then
                          Z     1                                 m X
                                                                  X i−1
                                    h−J ż, zidt =                               |Ij ||Ii |ω(wi , wj ).
                            0                                     i=1 j=1


Proof.
 Z 1                Z 1                Z t
     h−J ż, zidt =     h−J ż, z(0) +     ż(s)dsidt
  0                    0                                          0
                      Z     1                m                            m
                                                                       Z tX
                                                     1Il (t)wl ,                        1
                                             X
                  =             h−J                                               Il (s)wl dsidt
                       0                     l=1                           0 l=1
                      m Z                           m                        Z τi−1 Xm                              Z    t
                                                            1Il (t)wl ,                           1Il (s)wl ds +
                      X                             X
                  =                      h−J                                                                                   wi dsidt
                      i=1           Ii              l=1                          0          l=1                         τi−1
                      m Z                               i−1 Z X
                                                              m
                                                                                 1Il (s)wl ds + (t − τi−1 )wi idt
                      X                                 X
                  =                      h−Jwi ,
                      i=1           Ii                  j=1           Ij l=1

                      m Z
                      X                                 i−1 Z
                                                        X                                    m X
                                                                                             X i−1
                  =                      h−Jwi ,                           wj dsidt =                     |Ii ||Ij |ω(wi , wj ).
                      i=1           Ii                  j=1           Ij                      i=1 j=1




Lemma 3.4. Fix a set of vectors v1 , . . . , vk ∈ R2n . Let z ∈ W 1,2 ([0, 1], R2n ) be
a piecewise affine loop, where ż(t) ∈ conv{v1 , . . . , vk } for almost every t ∈ [0, 1],
then there exists another piecewise affine loop z 0 ∈ W 1,2 ([0, 1], R2n ) so that
ż 0 (t) ∈ {v1 , . . . , vk } for almost every t, and
                                    Z        1                               Z     1
                                                        0     0
                                                 h−J ż , z idt ≥                      h−J ż, zidt.
                                         0                                     0


Proof. The idea of the proof is to replace any convex combination of {vi }ki=1
in
Pmthe velocity of z by moving in each velocity vi separately. Write                     ż(t) =
   j=1 Ij1  (t)w  j , where  for each  j, wj ∈ conv{v 1 , . . . , v k }, and (I )
                                                                                  m
                                                                               j j=1 is  a par-
                                           Pl
tition of [0, 1]. Suppose that wi = j=1 aij vij , where aij > 0, ij ∈ {1, . . . , k},
                                              Pl
and l ∈ N dependent on i. Note that j=1 aij = 1. Consider the partition of
Ii to disjoint subintervals Iij ⊂ Ii for every j = 1, . . . , l where the length of Iij
is |Iij | = aij |Ii |. Define the following loop

                              i−1                                 l                               m
                                         1Ij (t)wj +                       1Iij (t)vij +                  1Ij (t)wj .
                              X                                   X                               X
               ż 0 (t) =                                                                                                            (2)
                              j=1                                 j=1                             j=i+1

We shall specify the order of the subintervals Iij ’s and the velocities vij ’s ap-
                                                     R1           R1
pearing in (2) later. It follows immediately that 0 ż 0 (t)dt = 0 ż(t)dt = 0.
Next we show that, if the order of the vectors vij is properly chosen, then
                                    Z        1                               Z     1
                                                 h−J ż 0 , z 0 idt ≥                  h−J ż, zidt.
                                         0                                     0


                                                                       10
Indeed, by Proposition 3.3,
 Z 1                      X                         l X
                                                    X
     h−J ż 0 , z 0 idt =   |Ir ||Is |ω(ws , wr ) +     |Ir ||Ii |aij ω(vij , wr )
  0                      r<s                                       j=1 r<i
                        r,s6=i
                        l X
                        X                                                        X
                    +                  |Ir ||Ii |aij ω(wr , vij ) +                        |Ii |2 air ais ω(vis , vir )
                        j=1 r>i                                                1≤r<s≤l
                        X                                          X
                    =             |Ir ||Is |ω(ws , wr ) +                    |Ir ||Ii |ω(wi , wr )
                         r<s                                       r<i
                        r,s6=i
                        X                                          X
                    +            |Ir ||Ii |ω(wr , wi ) +                        |Ii |2 air ais ω(vis , vir )
                        r>i                                       1≤r<s≤l
                        Z    1                            X
                    =            h−J ż, zidt +                        |Ii |2 air ais ω(vis , vir ).
                         0                              1≤r<s≤l



Finally, we wish to prove that
                            X
                                                    air ais ω(vis , vir ) ≥ 0.                                            (3)
                                       1≤r<s≤l

Note that we are free to select the order of vi1 , vi2 , . . . , vil . If we reverse the
order of the velocities we get that the sum in (3) changes sign. Therefore, by
rearranging the vij ’s in (2) one can choose the order so that inequality (3) would
hold. By applying this argument to all intervals Ii one gets the thesis.
Lemma 3.5. Fix a finite sequence of pairwise distinct vectors (vP          1 , . . . , vk ). Let
                                                                              i=1 1Ii (t)wi ,
                                                                              m
z ∈ W 1,2 ([0, 1], R2n ) be a piecewise affine loop so that ż(t) =
                         m
where (Ii = (τi−1 , τi ))i=1 is a partition of [0, 1], and for each i, wi ∈ {v1 , . . . , vk }.
Then there exists another piecewise affine loop z 0 so that ż 0 (t) ∈ {v1 , . . . , vk } for
almost every t, and {t : ż 0 (t) = vj } is connected for every j = 1, . . . , k. In
addition,                  Z           1        Z                      1
                                           h−J ż 0 , z 0 idt ≥            h−J ż, zidt.
                                   0                               0

Proof. Assume that for some r < s one has wr = ws , consider a rearrangement
of the intervals Ii where we erase the interval Is and increase the length of the
interval Ir by |Is | = τs − τs−1 , more precisely,
                    
                    
                                   (τi−1 , τi ),              i<r
                            (τi−1 , τi + τs − τs−1 ),          i=r
                    
                    
                    
              Ii0 = (τi−1 + τs − τs−1 , τi + τs − τs−1 ),     r<i<s
                                        ∅,                     i=s
                    
                    
                    
                    
                                    (τi−1 , τi ),              i>s
                    

Now define z by ż (t) = i=1 1Ii0 (t)wi . We will show that the action of this
               0      0
                             Pm
loop z 0 or the analogous loop z 00 which is defined by erasing Ir and increasing
the length of Is by |Ir | is not smaller than the action of z. First note that
                                    Z 1           m
                                                  X
                              0=        żdt =      |Ii |wi ,
                                                    0             i=1


                                                           11
while                                        Z      1               m
                                                                    X
                                                        ż 0 dt =         |Ii0 |wi .
                                                0                   i=1
Since wr = ws the two sums are only different in the order of summation and
thus equal. Next, we claim that
                       Z 1                  Z 1
                                 0 0
                           h−J ż , z idt ≥     h−J ż, zidt.            (4)
                                        0                             0

By Proposition 3.3,
                            Z     1                         m X
                                                            X
                                      h−J ż, zidt =                      |Ij ||Ii |ω(wi , wj ).                    (5)
                              0                             i=1 j<i
                              R1
Consider the change in 0 h−J ż, zidt after removing Is and adding |Is | to the
length of Ir . Since wr = ws , the coefficient of ω(wr , wi ) does not change for
i < r or i > s. For r < i < s instead of the term |Is ||Ii |ω(ws , wi ) in (5) we add
|Is ||Ii |ω(wi , ws ) to the term |Ir ||Ii |ω(wi , wr ), so the action difference is
            Z    1                          Z    1                            s−1
                                                                              X
                     h−J ż 0 , z 0 idt −            h−J ż, zidt =                    2|Is ||Ii |ω(wi , ws ).
             0                               0                              i=r+1

Note that if one erases Ir and increases the length of Is by |Ir | instead, the
action difference becomes
                                            s−1
                                            X
                                                        2|Ir ||Ii |ω(wr , wi ),
                                            i=r+1

which has an opposite sign, and hence either z 0 or z 00 satisfies (4). Finally, we
continue to join different disjoint intervals Ir ,Is whenever wr = ws = vi by
induction, until {t : ż 0 (t) = vi } is connected for every i = 1, . . . , k.
Proposition 3.6. Let K ⊂ R2n be a convex polytope so that the origin 0 belongs
to K. Let {ni }F
               i=1 be the normal vectors to the 2n − 1-dimensional facets of K,
                 K


and let pi = J∂gK  2
                     |Fi = h2i Jni . Recall that hi := hK (ni ). Let c > 0 be a
constant and let z ∈ E be a loop that satisfies that for almost every t, there is a
non-empty face of K, Fj1 ∩ . . . ∩ Fjl 6= ∅, with ż(t) ∈ c · conv{pj1 , . . . , pjl }. Then
                                                         IK (z) = c2 .
                                                                                             Pl
Proof. Fix t0 ∈ [0, 1] and assume that ż(t0 ) = c ·                                            i=1   ai pji for ai ≥ 0,
Pl
  i=1 ai = 1. By the definition of hK one has

                                   l                           l                       l
                                  X   ai                      X   ai                  X   ai
hK (−J ż(t0 )) = hK (2c                   nji ) = sup hx, 2c          nji i = 2c sup        hx, nji i.
                                  i=1
                                      h ji         x∈K        i=1
                                                                  h ji            x∈K    h
                                                                                      i=1 ji

On the other hand supx∈K hx, nji i = hji , and it is attained for every x ∈ Fji .
Hence for any choice of y ∈ Fj1 ∩ . . . ∩ Fjl ,
                           l                   l                  l
                          X   ai              X   ai             X
                      sup         hx, nji i =        hy, nji i =     ai = 1.
                      x∈K i=1 hji                h
                                              i=1 ji             i=1


                                                               12
Hence
                                     hK (−J ż(t)) = 2c,
for almost every t, and
                                          Z      1
                                      1
                        IK (z) =                     h2K (−J ż(t))dt = c2 .
                                      4      0



Proof of Theorem 1.5. Since the existence of a closed characteristic with
the desired properties is independent on translations, we assume without loss of
generality that the origin 0 belongs to K (see also Remark 3.9). Assume that
γ : [0, 1] → ∂K is a closed characteristic with minimal action such that γ̇(t) ∈
         2
dJ∂gK      (γ(t)) for almost every t, where d > 0 is a constant independent of t (recall
that every closed characteristic equals upto a reparametrization to a solution
                                                2
to the Hamiltonian inclusion γ̇(t) ∈ J∂gK         (γ(t)) almost everywhere, and one
can reparametrize by some constant d to get γ(0) = γ(1)). From Lemma 2.2 it
follows that there is z ∈ E † such that A(γ) = 2IK (z), and z = λγ + b, with some
constants λ ∈ R+ , b ∈ R2n . Note that ż(t) = λγ̇(t) ∈ λd · conv{p1 , . . . , pFK },
and denote c = λd. Moreover, z satisfies the conditions of Proposition 3.6 and
hence IK (z) = c2 . From Lemma 3.1 for every N ∈ N one can find a piecewise
affine loop ζN such that kz − ζN k1,2 ≤ N1 and ζ̇N (t) ∈ c · conv{p1 , . . . , pFK } for
almost every t. By applying first Lemma 3.4 with vi = cpi , i = 1, . . . , FK to ζN ,
and then take the result and apply to it Lemma 3.5 again with vi = cpi , i =
1, . . . , FK , one gets a piecewise affine loop zN which can be written as
                                                     mN

                                                           1IiN (t)viN ,
                                                     X
                                  żN (t) =
                                                     i=1

where viN = c · pj for some j ∈ {1, . . . , FK } and for every j there is at most one
such i. Moreover one has
                        s                         s
                         Z 1                        Z 1
                AN :=        h−J żN , zN idt ≥         h−J ζ̇N , ζN idt.
                            0                                       0

              0       zN                                           viN         0
Hence denote zN =     AN   ∈ E, and write wiN =        for the velocities of zN
                                                                   AN             , and
            c                                N →∞           R 1                  N →∞
write cN = AN . The fact that ζN −−−−→ z implies that 0 h−J ζ̇N , ζN idt −−−−→
1. Hence limN →∞ AN ≥ 1, and limN →∞ cN ≤ c. Moreover, from Proposition
                                                                 0
3.6 and from the minimality of IK (z), one has c2N = IK (zN          ) ≥ IK (z) = c2 ,
                                                                   0
and hence limN →∞ cN = c and consequently limN →∞ IK (zN ) = IK (z), and
                                   0
limN →∞ AN = 1. (Note that zN          satisfies the conditions of Proposition 3.6
because each single pi trivially satisfies that the face Fi of K is non-empty.)
   Consider the space E 1 of piecewise affine curves z 0 , whose velocities are in
the set C · {p1 , . . . , pFK } for some C > 0 and each pi appears at most once. Let
us define a map Φ : E 1 → SFK × RFK , z 0 7→ (σ, (|I1 |, . . . , |IFK |)), where

                                             FK
                                                      1Ii (t)C · pσ(i) .
                                             X
                                ż 0 (t) =
                                             i=1



                                                      13
A point in the image (σ, (t1 , . . . , tFK )) ∈ Im(Φ) satisfies ti ≥ 0 for each i,
      PFK
and i=1      ti = 1, which implies that Im(Φ) belongs to a compact set in the
                                    0                                                      0
usual topology. Note that zN              ∈ E 1 with C = cN . Suppose that Φ(zN               ) =
    N   N           N
(σ , (t1 , . . . , tF )), then after passing to a subsequence, one can assume that
                     K
                                                                          ∞            ∞
σ N = σ is constant, and (tN                 N
                                1 , . . . , tFK ) converges to a vector (t1 , . . . , tFK ). Let
  0                                                            ∞       ∞
z∞ be the piecewise affine curve identified with (σ, (t1 , . . . , tFK )), and with C =
                                      0        0
limN →∞ cN = c. Note that kzN              − z∞  k1,2 → 0. Indeed, let T N ⊂ [0, 1] be the
                          0       c 0                      N →∞               R          0
set of times where żN (t) = cN ż∞ (t). Since cN −−−−→ c, one has T N kżN                  (t) −
  0          N →∞                                                 0            0
ż∞ (t)k2 dt −−−−→ 0. Note that for each t ∈ [0, 1] such that żN    (t) and ż∞   (t) are
             0      0     2
defined, kżN (t)−ż∞ (t)k is bounded, since both belong to a finite set of velocities
                                              N →∞                                0
and cN is bounded. Hence since |T N | −−−−→ 1, one has [0,1]\T N kżN
                                                                    R
                                                                                     (t) −
  0           N →∞                         1 0
    (t)k2 dt −−−−→ 0. Moreover, since 0 żN
                                         R
ż∞                                             (t)dt = 0 for each N , one gets that
R1 0                            0
    ż
  0 ∞
       (t)dt   = 0 and hence  z ∞ is a closed  loop.   Similarly, one can check that
  0                                                 0
z∞    ∈ E, and finally by Proposition 3.6, IK (z∞     ) = c2 = IK (z). Since z was
                                           0
chosen to be a minimizer, we get that z∞     is also a minimizer, and therefore it is
a weak critical point of IK , i.e. z∞ ∈ E † . Finally by invoking Lemma 2.2, one
                                     0

gets a piecewise affine closed characteristic γ 0 where γ̇ 0 (t) ∈ d · {p1 , . . . , pFK }
outside a finite subset of [0, 1], and the set {t : γ̇ 0 (t) = dpi } is connected for
every i, i.e. every velocity pi appears at most once.


    We are now in a position to prove Theorem 1.1.
Proof of Theorem 1.1.              Let K be a convex polytope. From Lemma 2.2 it
follows that
                                   cEHZ (K) = min2IK (z).                                     (6)
                                                    z∈E
Theorem 1.5 implies that there exists z ∈ E which minimizes IK and is of the
form
                                    FK
                                       1Ii cpσ(i) .
                                    X
                            ż(t) =
                                              i=1
for some σ ∈ SFK , and c > 0. Therefore, when calculating the minimum in
(6), one can restrict to loops of this form in E. Let us rewrite the conditions
                                                  R1
for z to be in E in this case. The condition 0 ż(t)dt = 0 is equivalent to
PFK
   i=1 Ti pσ(i) = 0, where we denote Ti = |Ii |. By means of Proposition 3.3 the
            R1
condition 0 h−J ż(t), z(t)idt = 1 can be written as
                 Z 1                         X
             1=      h−J ż(t), z(t)idt = c2        Ti Tj ω(pσ(i) , pσ(j) ).
                    0                               1≤j<i≤FK

Finally by Proposition 3.6,
                                          IK (z) = c2 .
Overall we get that
                           cEHZ (K) = 2               min                 c2 ,
                                             (Ti )∈M T (K) s.t.
where                     (                ∀σ∈Sk ,c2 AK (σ,(Ti ))≤1                  )
                                                XFK             FK
                                                                X
                T
             M (K) =         (Ti )FK
                                  i=1   : Ti ≥ 0,         Ti = 1,         Ti pσ(i) = 0 ,
                                                    i=1             i=1


                                               14
and                                              X
                   AK (σ, (Ti )F
                               i=1 ) =
                                 K
                                                           Ti Tj ω(pσ(i) , pσ(j) ).
                                           1≤j<i≤FK

This can be written as
                                                                                     −1
                                                   X                                 
         cEHZ (K) = 2            max                          Ti Tj ω(pσ(i) , pσ(j) )     .
                                 σ∈S  FK
                                                                                      
                                  FK             1≤j<i≤FK
                             (Ti )i=1 ∈M T (K)

             2                                    Ti
Since pi =   hi Jni ,   we can set βσ(i) =       hσ(i)   and get the required formula.

Remark 3.7. By plugging the simple closed characteristic from Theorem 1.5
in the formula for cEHZ from Remark 1.4, one gets a similar proof for Theorem
1.1.
Remark 3.8. From the proof of Theorem 1.1 we see that if one considers
loops z ∈ E with ż piecewise constant, and whose velocities are of the form dpi ,
without the restriction that each pi appears at most once, one still gets an upper
bound for cEHZ (K). More precisely each selection of a sequence of unit outer
normals to facets of K (ni )m                                   m
                            i=1 and a sequence of numbers (βi )i=1 that satisfy

                                  m
                                  X                             m
                                                                X
                        βi ≥ 0,         βi hK (ni ) = 1,              βi ni = 0,
                                  i=1                           i=1

gives an upper bound of the form
                                                                         −1
                             1              X
                  cEHZ (K) ≤                             βi βj ω(ni , nj )     .
                             2
                                          1≤j<i≤m

This fact will be useful for us in the proof of Theorem 1.8.
Remark 3.9. Note that formula (1) for cEHZ in Theorem 1.1 is invariant under
translations and is 2-homogeneous. Indeed, if we take K         e = K + x0 we get
the same normals and the oriented heights change to hi = hi + hx0 , ni i. For
                                                              e
(βi )F
                                     P e         P               P
     i=1 ∈ M (K), one can check that     βi hi =     βi hi +hx0 , βi ni i = 1. Hence
      K


(βi )F
     i=1 ∈ M (K) so we get the same value for
      K       e
                           X
                                 βσ(i) βσ(j) ω(nσ(i) , nσ(j) ).
                            1≤j<i≤FK


Hence cEHZ (K) = cEHZ (K).
                       e

   On the other hand, consider K   e = λK for some λ > 0, then it has the same
                                                 hi = λhi . For (βi )F
normals as K, and the oriented heights change to e                   i=1 ∈ M (K),
                                                                      K

          βi              FK
take βi = λ , to get (βi )i=1 ∈ M (K). We get that
     e                e             e
      X                                          1          X
               βeσ(i) βeσ(j) ω(nσ(i) , nσ(j) ) = 2                    βσ(i) βσ(j) ω(nσ(i) , nσ(j) ).
                                                λ
   1≤j<i≤FK                                              1≤j<i≤FK

             e = λ2 c (K).
Hence, cEHZ (K)      EHZ




                                                  15
Remark 3.10. Formula (1) is invariant under multiplication by a symplec-
tic matrix A ∈ Sp(2n). Indeed, take K       e = AK. The new normals are n         ei =
  (At )−1 ni                                                   hi
 k(At )−1 ni k , and the new  oriented heights are  h
                                                    e i = k(At )−1 ni k . One can  take
             β
βei = k(A )
               i
            t −1
              ni k   and get that c (K) = c (K).
                                             EHZ
                                                   e
                                                                EHZ


Remark 3.11. The number of permutations in SFK grows exponentially in FK
and thus can be a huge number. For computational goals, it is worth noting
that this set can be reduced. Consider a directed graph G, with vertex set {j}
corresponding to facets of K, {Fj }, and where there exists an edge ij if there
exists a point x ∈ Fi , and a constant c > 0 so that x + cpi ∈ Fj . Denote by A
the set of all cycles on G. An element I ∈ A is a sequence (I(1), . . . , I(l)), where
there are edges I(i)I(i + 1) for i = 1, . . . , l − 1 and there is an edge I(l)I(1). We
get that
                                                                                  −1
                     1                      X
       cEHZ (K) =           max                      βI(i) βI(j) ω(nI(i) , nI(j) )     ,
                     2 I∈A,(βi )∈MI (K)
                                                       1≤j<i≤|I|
where
                                                                                                     
                                                    |I|                        |I|                   
                               |I|
                                                     X                          X
         MI (K) =        (βi )i=1 : βi ≥ 0,                βI(i) hI(i) = 1,           βI(i) nI(i)   =0 .
                                                                                                     
                                                     i=1                        i=1


Proof of Corollary 1.2. Let K ⊂ R2n be a convex polytope that satisfies
K = −K. Let n1 , . . . , nF0 , −n1 , . . . , −nF0 be the normals to the (2n − 1)-
                                         K                       K
dimensional facets of K. Recall that pi = J∂gK       2
                                                       |Fi = h2i Jni . By Theorem 1.5,
there exists a closed characteristic γ on the boundary of K whose velocities are
                                                                F0K
piecewise constant, and are a positive multiple of {±pi }i=1        , so that for each i,
the velocity which is a positive multiple of pi (and the one which is a positive
multiple of −pi ) appears at most once. Consider a reparametrization of γ such
that γ̇(t) ∈ d{±pi } almost everywhere, for some d > 0 independent of i. From
Lemma 2.2 there exists a corresponding element z ∈ E † , such that z = λγ + b
and z is a minimizer of IK . The velocities of z are positive multiples of the
velocities of γ and hence have the same properties. The idea of the proof is to
change z to z 0 so that z 0 would also be a minimizer whose velocities have the
same properties, and which satisfies z 0 (t + 21 ) = −z 0 (t). The next argument (see
[2]) was communicated to us by R. Karasev, we include it here for completeness.
                                                    R1
    Translate z so that z(0) = −z( 12 ). Since 0 h−J ż(t), z(t)idt = 1, we either
      R1                             R1
have 02 h−J ż(t), z(t)idt ≥ 12 , or 1 h−J ż(t), z(t)idt ≥ 21 . Assume without loss
                                      2
of generality that the first inequality holds, i.e.
                              Z 12
                                                         1
                                   h−J ż(t), z(t)idt ≥ .
                               0                         2
Define
                                                                      t ∈ [0, 21 ]
                                             
                                     0             z(t),
                                 z =
                                                 −z(t − 12 ),         t ∈ [ 12 , 1]
                          F0                           F0
Since ż(t) ∈ c{±pi }i=1
                      K
                         = c{± h2i Jni }i=1
                                         K
                                            , where c = λd, one has h2K (−J ż(t)) =
4c . Note that since K = −K one has hK (x) = hK (−x) for all x ∈ R2n , hence
  2



                                                           16
one gets
                                                                                                   1
                      Z          1                                                         Z
                1                                                         1                        2
       IK (z) =                      h2K (−J ż(t))dt                =c =    2
                                                                                                       h2K (−J ż(t))dt = IK (z 0 ).
                4            0                                            2                 0

Moreover,
                                                                     1                             1
                                 Z       1                     Z     2
                                                                                           Z       2
                                              0
                                             ż (t)dt =                  ż(t)dt −                      ż(t)dt = 0,
                                     0                           0                             0
and                                                                                   1
                  Z      1                                                       Z    2
                                             0             0
                             h−J ż (t), z (t)idt = 2                                     h−J ż(t), z(t)idt ≥ 1.
                     0                                                            0
                                                    R1
Hence one can divide z 0 by a constant to get 0 h−J ż 0 (t), z 0 (t)idt = 1 and
IK (z 0 ) ≤ IK (z). Since z was chosen to be a minimizer, the constant must be 1,
and IK (z 0 ) = IK (z). Hence z 0 is a minimizer that satisfies z 0 = −z 0 .
    After plugging z 0 in Formula (1) for cEHZ (K) from Theorem 1.1 we get a
maximum, hence there exists an order of the normals that gives maximum in
(1) which has the following form.

           a(1)nσ(1) , . . . , a(F0K )nσ(F0K ) , −a(1)nσ(1) , . . . , −a(F0K )nσ(F0K ) ,

where a(i) = ±1, and σ ∈ SF0K . Recall that here the number of facets is 2F0K .
In addition, since βi = Thii (see the proof of Theorem 1.1), from the symmetry
of z 0 the oriented heights hi and the times Ti in the first half are equal to the
oriented heights and the times in the second half, and hence the “betas” in the
first half are equal to the “betas” in the second half. Let us consider the sum
we try to maximize in (1) :
      P
        1≤i<j≤F0      βσ(i) βσ(j) (ω(a(i)nσ(i) , a(j)nσ(j) ) + ω(−a(i)nσ(i) , −a(j)nσ(j) ))
                 K
                                                 0                       0
                                             FK                      FK
                                             X                       X
                                     +                βσ(i) a(i)             βσ(j) a(j)ω(nσ(i) , −nσ(j) )
                                             i=1                     j=1
       P
  =2       1≤i<j≤F0K                 βσ(i) βσ(j) ω(a(i)nσ(i) , a(j)nσ(j) )
                                                      0                                        0
                                                  FK                                       FK
                                                  X                                        X
                                     +ω(                   βσ(i) a(i)nσ(i) , −                         βσ(i) a(i)nσ(i) )
                                                     i=1                                   i=1
       P
  =2       1≤i<j≤F0K                 βσ(i) βσ(j) ω(a(i)nσ(i) , a(j)nσ(j) ).

We get that the sum we try to maximize in (1) is equal to twice the sum
over the normals in the first half. In addition, in M (K) we can remove the
           P2F0
constraint i=1K βi ni = 0 because we get it automatically (since the second
half of the normals are minus the first half and the “betas” are equal). The
           P2F0                        PF0K
constraint i=1K βi hi = 1 becomes i=1         βi hi = 21 and instead of considering
the constraints βi ≥ 0 for each i, we can remove the signs a(i) from the normals,
and allow for negative “betas” as well. In conclusion, we get that the only
                              PF0K
constraint for the “betas” is i=1    |βi |hi = 12 and this gives us the formula we
need and thus proves Corollary 1.2.



                                                                             17
4     Subadditivity for hyperplane cuts
In the proof of Theorem 1.8, we use the formula for the capacity that was proved
in Theorem 1.1, in its equivalent formulation which was given in Remark 1.3,
namely,
                                                  FK
                                                                 !2
                            1                     X
                 cEHZ (K) =             min           βi hK (ni ) ,          (7)
                            2 (βi ,ni )Fi=1
                                         K ∈M (K)
                                             2    i=1

where
              (                                                                    )
                  (βi , ni )FK
                               : β ≥ 0, (ni )FK
                                                are different outer normals to K
 M2 (K) =         PFK i=1 i              P i=1                                         .
                          β
                    i=1 i i  n =  0,        1≤j<i≤FK βi βj ω(ni , nj ) = 1


    To see that this is indeed equivalent to the form given in Theorem 1.1, note
that
                                                                       −1
             1                       X
 cEHZ (K) =            max                 βσ(i) βσ(j) ω(nσ(i) , nσ(j) )
             2 σ∈SFK ,(βi )∈M (K)
                                  1≤j<i≤FK
                                                                                    −1
                                         P
             1                                        β    β
                                           1≤j<i≤FK σ(i) σ(j)     ω(n  σ(i) , n    )
                                                                               σ(j) 
          =              max                                      2
             2 σ∈SF ,βi ≥0,PFi=1
                                                    P                               
                                K β n =0                 F K
                                                         i=1 βi hi
                     K             i i


                                                  −1
              1                           1
          =           max
                                                       
                                                    2 
              2 (βi ,ni )∈M2 (K) PFK
                               
                                     i=1 βi hK (ni )

                                 FK
                                               !2
              1                  X
          =           min            βi hK (ni )   .
              2 (βi ,ni )∈M2 (K) i=1

    Before providing the full proof of Theorem 1.8, let us briefly describe the
main idea. Suppose we cut a convex polytope K by a hyperplane H into K1
and K2 . Our strategy is to take minimizers in M2 (K1 ) and in M2 (K2 ), and
construct from them a sequence of normals and coefficients on K that gives an
upper bound for cEHZ (K) which is less than or equal to cEHZ (K1 ) + cEHZ (K2 ).
By Theorem 1.5, we know that one can take the minimizers so that the normal
to the shared facet K1 ∩ H = K2 ∩ H appears at most once. This enables us to
choose coefficients so that this normal in both minimizers cancels out and we
are left with a minimizer in M2 (K).
Proof of Theorem 1.8.             From the continuity of the EHZ capacity (see
e.g. [14], Exercise 12.7) it is enough to prove the statement for polytopes. Let
K ⊂ R2n be a convex polytope.
    Suppose we cut K by a hyperplane into K1 and K2 . Without loss of gen-
erality, choose the origin to be on the hyperplane that divides K into K1
                          FK1             FK2
and K2 . Choose (βi , ni )i=1 , (αi , wi )i=1 to be minimizers in Equation (7) for
cEHZ (K1 ) and cEHZ (K2 ) respectively. In addition, denote by n the normal to
the hyperplane splitting K into K1 and K2 where we choose the positive di-
rection to go into K1 . Note that for each outer normal ni 6= n of K1 , one


                                            18
has hK1 (ni ) = hK (ni ), and for each outer normal wi 6= n of K2 , one has
hK2 (wi ) = hK (wi ). In addition, one has hK1 (n) = hK2 (n) = 0. Assume with-
out loss of generality that n1 = −n and wFK = n (this can be assumed because
                                                 2
one can always take cyclic permutations of the sequences to get new sequences
that satisfy the constraints and give the same result). By means of Theorem 1.5,
each normal vector appears at most once, and hence for each i 6= 1, ni 6= n, and
for each i 6= FK2 , wi 6= n. First note that if β1 = 0 or αFK = 0 we are done.
                                                                        2
Indeed, suppose that β1 = 0. All the normals ni for i ≥ 2 are normals also to
                                              FK1
facets of K. Hence β1 = 0 implies (βi , ni )i=2    ∈ M2 (K) (after adding the rest of
the normals with coefficients zero), and this gives cEHZ (K) ≤ cEHZ (K1 ). From
now on assume β1 6= 0 and similarly αFK 6= 0. Next, consider the following
                                               2
sequence of coefficients

           FK2 +FK1 −2
                          
                            β1             β1        αFK                 α FK         
                                                         2                    2
      (δi )i=1         :=      α1 , . . . , αFK −1 ,       β2 , . . . ,         β FK ,
                             c              c   2      c                    c       1


                                  K1 K2  F    +F       −2
and the sequence of normals (ui )i=1      := (w1 , . . . , wFK −1 , n2 , . . . , nFK ),
           q                                                  2                     1

where c := β12 + αF2 . Note that here we allow for repetitions of the normals
                         K2

and we may have FK1 + FK2 − 2 > FK . However, from Remark 3.8 we know
                                             m
that if one considers any sequence
                              Pm (δi , ui )i=1 withP δi ≥ 0, ui a normal to K,
that satisfies the constraints i=1 δi ui = 0 and 1≤j<i≤m δi δj ω(ui , uj ) = 1 for
any m ∈ N, the value
                                   m
                                                  !2
                               1 X
                                       δi hK (ui )
                               2 i=1
                                                                               K1 K2                      F   +F   −2
still gives an upper bound for cEHZ (K). Hence we wish to show that (δi , ui )i=1
                                                      PFK2             PFK1
satisfies the constraints for K. First note that since i=1 αi wi = 0, i=1 βi ni =
0 one has
                 FK1 +FK2 −2
                     X                β1             αFK
                             δi ui = − · αFK n +         2
                                                           · β1 n = 0.
                     i=1
                                       c      2        c
Next, note that
                  X                                  β12         X
                               δi δj ω(ui , uj ) =                               αi αj ω(wi , wj )+
                                                     c2
       1≤j<i≤FK1 +FK2 −2                                   1≤j<i≤FK2 −1

                                                             K1 FK2 −1
       αF2           X                               β1 αFK F
                                                            X    X
             K2                                              2
                              βi βj ω(ni , nj ) +                                    βi αj ω(ni , wj ).
         c2                                            c2        i=2       j=1
                  2≤j<i≤FK1
                          PFK2 −1                                                     PFK2 −1
Since −αFK wFK =              i=1     αi wi , one has ω(αFK wFK ,                        i=1     αi wi ) = 0,
           2   2                                                       2         2
and therefore
              X                                            X
                            αi αj ω(wi , wj ) =                    αi αj ω(wi , wj ) = 1.
             1≤j<i≤FK2 −1                            1≤j<i≤FK2

Similarly,                          X
                                             βi βj ω(ni , nj ) = 1.
                                2≤j<i≤FK1



                                                19
Finally, note that
  FK1 FK2 −1                                   FK1              FK2 −1
  X X                                          X                 X
                βi αj ω(ni , wj ) = ω(               βi n i ,             αi wi ) = −β1 αFK ω(n, n) = 0.
                                                                                                   2
  i=2     j=1                                  i=2               j=1

Overall we get that

                               X                                   β12 αF2
                                                                           K
                                                δi δj ω(ui , uj ) = 2 + 2 2 = 1.
                                                                    c    c
                    1≤j<i≤FK1 +FK2 −2

                 K1 F     +FK2 −2
Hence (δi , ui )i=1                  indeed satisfies the constraints, and therefore cEHZ (K)
                                     P                           2
                                 1       FK1 +FK2 −2
is less than or equal to         2       i=1          δ i hK (ui )   . It remains to show that
                        2                   2                    2
    FK +FK −2                  FK2                    FK1
 1  1X2                    1   X                 1   X
              δi hK (ui ) ≤      αi hK2 (wi ) +       βi hK1 (ni ) .
 2     i=1
                            2 i=1                 2 i=1

A straightforward computation gives
                                      2                                                    2
        FK1 +FK2 −2                                                     FK2 −1
             X                                       1  2               X
                         δi hK (ui )         =         β1                     αi hK2 (wi )
             i=1
                                                     c2                   i=1

                                                                     FK2 −1                    FK1
                                                                        X                      X
                                               +     2β1 αFK                    αi hK2 (wi )           βi hK1 (ni )
                                                                 2
                                                                        i=1                    i=2
                                                                               2 
                                                                  FK1
                                                                  X
                                               +     αF2            βi hK1 (ni )  .
                                                                                  
                                                           K2
                                                                    i=2


Since
             FK2 −1                    FK1
              X                        X
2β1 αFK                 αi hK2 (wi )         βi hK1 (ni )
         2
              i=1                      i=2
                                                                                 2                                 2
                                                     FK2 −1                                        FK1
                                                        X                                          X
                                       ≤ αF2                    αi hK2 (wi ) + β12                    βi hK1 (ni ) ,
                                               K2
                                                        i=1                                        i=2




                                                           20
one has
                                   2
               FK +FK −2
           1  1X2
cEHZ (K) ≤               δi hK (ui )
           2      i=1
                                                2                                2 
                               FK2 −1                               FK1
            1                   X                                   X
         ≤ 2 (β12 + αF2 )           αi hK2 (wi ) + (β12 + αF2 )      βi hK1 (ni ) 
                                                                                        
           2c           K2
                                 i=1
                                                                K2
                                                                     i=2
                                  2                    2
                FK2 −1                   FK1
            1 X                      1   X
          =            αi hK2 (wi ) +       βi hK1 (ni )
            2    i=1
                                      2   i=2
                              2                     2
                   FK2                   FK1
              1   X               1       X
          =         αi hK2 (wi ) +     βi hK1 (ni ) = cEHZ (K1 ) + cEHZ (K2 ),
              2 i=1                2 i=1



where the second to last equality is due to the fact that

                              hK1 (n) = hK2 (n) = 0.




References
 [1] A. Abbondandolo and P. Majer, A non-squeezing theorem for convex
     symplectic images of the Hilbert ball, Calculus of Variations and Partial
     Differential Equations, 54 (2015), pp. 1469–1506.
 [2] A. Akopyan and R. Karasev, Estimating symplectic capacities from
     lengths of closed curves on the unit spheres, arXiv:1801.00242, (2017).
 [3] A. Akopyan, R. Karasev, and F. Petrov, Bang’s problem and sym-
     plectic invariants, arXiv:1404.0871, (2014).
 [4] S. Artstein-Avidan and Y. Ostrover, Bounds for Minkowski bil-
     liard trajectories in convex bodies, Int. Math. Res. Not. IMRN, (2014),
     p. 165–193.
 [5] K. Cieliebak, H. Hofer, J. Latschev, and F. Schlenk, Quantitative
     symplectic geometry, in Dynamics, ergodic theory, and geometry, vol. 54
     of Math. Sci. Res. Inst. Publ., Cambridge Univ. Press, Cambridge, 2007,
     pp. 1–44.
 [6] F. H. Clarke, A classical variational principle for periodic Hamiltonian
     trajectories, Proc. Amer. Math. Soc., 76 (1979), pp. 186–188.

 [7] I. Ekeland and H. Hofer, Symplectic topology and Hamiltonian dynam-
     ics, Math. Z., 200 (1989), p. 355–378.



                                       21
 [8] M. Gromov, Pseudoholomorphic curves in symplectic manifolds, Invent.
     Math., 82 (1985), pp. 307–347.

 [9] H. Hofer and E. Zehnder, A new capacity for symplectic manifolds, in
     Analysis, et cetera, Academic Press, Boston, MA, 1990, p. 405–427.
[10] H. Hofer and E. Zehnder, Symplectic Invariants and Hamiltonian Dy-
     namics, Birkhauser Advanced Texts, Birkhauser Verlag, 1994.
[11] A. Künzle, Une capacité symplectique pour les ensembles convexes et
     quelques applications, Ph.D. thesis, Université Paris IX Dauphine, (1990).
[12] A. Künzle, Singular Hamiltonian systems and symplectic capacities, Sin-
     gularities and differential equations, Banach Center Publ., 33, Polish Acad.
     Sci., Warsaw, (1996), pp. 171–187.

[13] D. McDuff, Symplectic topology today, AMS Joint Mathematics Meeting,
     (2014).
[14] D. McDuff and D. Salamon, Introduction to Symplectic Topology, 2nd
     edition, Oxford University Press, Oxford, England, 1998.
[15] Y. Nir, On closed characteristics and billiards in convex bodies, Master’s
     thesis, Tel Aviv University, (2013).
[16] Y. Ostrover, When symplectic topology meets Banach space geometry,
     proceedings of the ICM, 2 (2014), pp. 959–981.
[17] R. Schneider, Convex Bodies: the Brunn-Minkowski theory, Encyclope-
     dia of Mathematics and its Applications, 44. Cambridge University Press,
     1993.
[18] J. Van Schaftingen, Approximation in Sobolev spaces by piecewise affine
     interpolation, Journal of Mathematical Analysis and Applications, 420
     (2014), pp. 40–47.

[19] C. Viterbo, Capacités symplectiques et applications, Séminaire Bourbaki,
     Vol. 1988/89, Astérisque 177–178 (1989), no. 714, 345–362.
[20] K. Zehmisch, The codisc radius capacity, Electron. Res. Announc. Math.
     Sci., 20 (2013), pp. 77–96.




School of Mathematical Sciences
Tel Aviv University
Tel Aviv 6997801, Israel
pazithaim@mail.tau.ac.il




                                       22
