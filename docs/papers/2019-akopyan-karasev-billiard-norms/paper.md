---
source: arXiv:1912.12685
fetched: 2025-10-20
---
# When different norms lead to same billiard trajectories?

                                                      WHEN DIFFERENT NORMS LEAD TO SAME BILLIARD
                                                                    TRAJECTORIES?

                                                                        ARSENIY AKOPYAN AND ROMAN KARASEV

                                                  Abstract. In this paper, extending the works of Milena Radnović and Serge Tabachnikov, we
                                                  establish conditions for two different non-symmetric norms to define the same billiard reflection
                                                  law.
arXiv:1912.12685v1 [math.MG] 29 Dec 2019




                                              Milena Radnović in [6] and independently Serge Tabchnikov in [7, Section 2] made the
                                           following remarkable observation:
                                           Theorem 1. Let k · kξ be a not necessarily symmetric norm in the plane, having an ellipse with
                                           focus at the origin o as the unit circle. Then k · kξ defines the same law of reflection as in the
                                           Euclidean metric: The angle of reflection equals to the angle of incidence.
                                              In the first paper it was also notices that it leads to the fact that billiard trajectories in the
                                           plane with norm, defined by an ellipse as the unit circle, are the same as in the Euclidean plane
                                           after a suitably chosen affine transform.
                                              Another consequence of their theorem is that the Euclidean and normed ellipses with foci
                                           o and f coincide, where f is the second focus of ξ. Indeed, if x is a point in the plane, then
                                           by Theorem 1 the differential of k−  → + k−
                                                                                oxk
                                                                                         →
                                                                                        xf kξ is proportional to the differential of the same
                                                                                    ξ
                                           expression for the Euclidean norm for every x. Hence the value k−    → + k−
                                                                                                                oxk
                                                                                                                          →
                                                                                                                          xf kξ does not change
                                                                                                                    ξ
                                           when x moves along the ellipse with foci f and o, along the zero direction of both differentials.
                                              It is interesting, that the latter statement may be deciphered to the following elementary
                                           geometric formulation, for which we do not know any short synthetic proof essentially different
                                           from the one stated in the above paragraph:
                                           Corollary 2. Let ξ1 and ξ2 bet two confocal ellipses with foci at f1 and f2 . For each point x
                                           on ξ1 , denote by y1 and y2 the points of intersections of ξ2 with rays f1 x and f2 x respectively.
                                           Then for any point x on ξ1 :
                                                                                |xf1 |   |xf2 |
                                                                                       +        = const.
                                                                               |y1 f1 | |y2f2 |
                                                                                                                             ℓ1 −|f1 f2 |       ℓ1 +|f1 f2 |
                                              It can be shown, that the constant in the corollary above equals:              ℓ2 −|f1 f2 |
                                                                                                                                            +   ℓ2 +|f1 f2 |
                                                                                                                                                             ,   where
                                           ℓ1 and ℓ2 are the major axes of the ellipses ξ1 and ξ2 .
                                                                                             x
                                                                                                   y2
                                                                                        y1



                                                                                       f1                    f2
                                                                                                        ξ2
                                                                                                                  ξ1

                                                                                  Fig. 1.
                                             Now we extend the Radnović–Tabachnikov theorem to normed spaces in higher dimension:
                                             2010 Mathematics Subject Classification. 53B40, 53D99, 70H05.
                                             Key words and phrases. Billiards, Finsler geometry, Hamiltonian systems.
                                                                                                 1
2                             ARSENIY AKOPYAN AND ROMAN KARASEV

Theorem 3. Let K be a smooth convex body in Rn containing the origin, and T be its con-
vex image under a projective transform, which maps each line passing through origin to itself
preserving its orientation at the origin. Then the billiard reflection law in the space with norm
k · kK is the same as in the space with norm k · kT .
Remark 1. It is known (see [3, Lemma 4.6]) that such kind of projective transforms send spheres
with center at origin to ellipsoids of rotation with one of the foci at the origin. Therefore this
theorem directly generalizes Theorem 1 to higher dimension.
Remark 2. The law of reflection is not well-defined for convex bodies K, which are not strictly
convex. In this case we may follow the conventions in [1] and define billiard trajectories, for
which the reflection direction is not uniquely defined.




                                                      T

                                                          K
                       Fig. 2. Convex body K and its projective image T .

Proof. We use ideas from [5, 4, 2], suggesting to work with billiard trajectory in the Banach
space U = Rn with norm k · kK in terms of momenta in the dual space U ∗ with norm k · kK ◦ ,
whose unit ball is the polar body K ◦ . From the smoothness of K, to each unit velocity u ∈ ∂K
there corresponds a conjugate unit momentum u∗ ∈ ∂K ◦ , such that u∗ (u) = 1 and ku∗ kK ◦ = 1.
The equation u∗ (x) = 1 defines the hyperplane tangent to ∂K at u, while the equation u(y) = 1
defines the hyperplane tangent to ∂K ◦ at u∗ .
   Let q1 q2 q3 be a part of a billiard trajectory in U, where q2 is the point where the trajectory
hits a hypersurface S and reflects. Then the sum k−      q→        −→
                                                          1 xkK + kxq3 kK as a function of x ∈ S has
a critical value at q2 . The criticality in terms of first derivatives means:
(∗)                                         u∗2 − u∗1 = λn∗ ,
where u∗1 and u∗2 are momenta corresponding to unit vectors in directions −      q− →       −−→
                                                                                   1 q2 and q2 q3 , and n
                                                                                                         ∗

is the normal covector to S at q2 .
   It is crucial that equation (∗) is preserved under a positive similiarity of the body K ◦ , possibly
with different factor λ. Indeed, let T ◦ = tK ◦ +v ∗ , where t > 0 and v ∗ ∈ U ∗ . It is easy to see that
the momentum, corresponding to velocity u with respect to the body T ◦ , equals u∗T = tu∗ + v ∗ ,
because u is a linear function on U ∗ and the points, where its maximum is obtained on K ◦ and
T ◦ are moved one to another by the homothety. Hence the difference of the new momenta at
a reflection point equals to t(u∗2 − u∗1 ), which is still parallel to n∗ .
   We obtain that the reflection laws for two norms k · kK and k · kT coincide if K ◦ and T ◦
are positive homothets of each other. A positive homothety is a projective transform which
maps any point at infinity to itself. Therefore in the dual space (our original U) a positive
homothety corresponds to the map, which preserves its polar images as a sets, that is planes
passing through the origin. It is easy to see, that this is the projective transform described in
the statement of the theorem.
   In simpler words, K is given by the system of linear inequalities of the form
                                        u∗ (x) ≤ 1,   ∀u∗ ∈ K ◦ .
                 WHEN DIFFERENT NORMS LEAD TO SAME BILLIARD TRAJECTORIES?                                          3

Hence the equations of T must be (assuming working not far from the origin, where v(x) < 1)
                                                         
                       ∗       ∗            ∗       tx
                     tu (x) + v (x) ≤ 1 ⇔ u                 ≤ 1, ∀u∗ ∈ K ◦ .
                                                 1 − v(x)
It remains to note that
                                              tx
                                     x 7→           , t>0
                                          1 − v(x)
is the general form of projective maps that preserve lines thorough the origin and keep their
orientations at the origin.
                                                                                           
                                            Acknowledgments
   The authors thank Alexey Balitskiy, Milena Radnović, and Serge Tabachnikov for useful
discussions.
   AA was supported by European Research Council (ERC) under the European Union’s Hori-
zon 2020 research and innovation programme (grant agreement No 78818 Alpha). RK was
supported by the Federal professorship program grant 1.456.2016/1.4 and the Russian Founda-
tion for Basic Research grants 18-01-00036 and 19-01-00169.
                                                  References
[1] A. Akopyan and A. Balitskiy. Billiards in convex bodies with acute angles. Israel Journal of Mathematics,
    216(2):833–845, 2016.
[2] A. Akopyan, A. Balitskiy, R. Karasev, and A. Sharipova. Elementary approach to closed billiard trajectories
    in asymmetric normed spaces. Proc. Amer. Math. Soc., 144(10):4501–4513, 2016.
[3] A. V. Akopyan and A. I. Bobenko. Incircular nets and confocal conics. Transactions of the American Math-
    ematical Society, 370(4):2825–2854, nov 2018.
[4] S. Artstein-Avidan and Y. Ostrover. A Brunn–Minkowski inequality for symplectic capacities of convex
    domains. IMRN: International Mathematics Research Notices, 2008, 2008.
[5] E. Gutkin and S. Tabachnikov. Billiards in Finsler and Minkowski geometries. J. Geom. Phys., 40(3-4):277–
    301, 2002.
[6] M. Radnović. A note on billiard systems in Finsler plane with elliptic indicatrices. Publ. Inst. Math. (Beograd)
    (N.S.), 74(88):97–101, 2003.
[7] S. Tabachnikov. Remarks on magnetic flows and magnetic billiards, Finsler metrics and a magnetic analog of
    Hilbert’s fourth problem. In Modern dynamical systems and applications, pages 233–250. Cambridge Univ.
    Press, Cambridge, 2004.

  Arseniy Akopyan, Institute of Science and Technology Austria (IST Austria), Am Campus 1,
3400 Klosterneuburg, Austria
  E-mail address: akopjan@gmail.com

  Roman Karasev, Moscow Institute of Physics and Technology, Institutskiy per. 9, Dol-
goprudny, Russia 141700 and Institute for Information Transmission Problems RAS, Bolshoy
Karetny per. 19, Moscow, Russia 127994
  E-mail address: r n karasev@mail.ru
  URL: http://www.rkarasev.ru/en/
