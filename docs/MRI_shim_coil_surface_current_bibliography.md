# MRI shim-coil, general coil-design, and surface-current bibliography

Last updated: 2026-07-14

## Scope

This is a curated engineering bibliography rather than a claim of literal completeness. It emphasizes papers that are directly useful for designing MRI/NMR B0 shim coils, gradient coils, localized multi-coil arrays, and arbitrary-surface windings using target-field, inverse-boundary-element, and stream-function methods. RF-coil papers are included where they establish foundational coil physics, array design, or optimal surface-current formulations. Papers concerned only with pulse-sequence shimming, image reconstruction, or passive magnet shimming are mostly excluded.

## 1. Best entry points and reviews

1. Turner R. **Gradient coil design: a review of methods.** *Magnetic Resonance Imaging.* 1993;11(7):903-920. [doi:10.1016/0730-725X(93)90209-V](https://doi.org/10.1016/0730-725X(93)90209-V). — Classical review of discrete-wire and continuous-current methods, target fields, inductance/power optimization, and active shielding.
2. Juchem C, de Graaf RA. **B0 magnetic field homogeneity and shimming for in vivo magnetic resonance spectroscopy.** *Analytical Biochemistry.* 2017;529:17-29. [doi:10.1016/j.ab.2016.06.003](https://doi.org/10.1016/j.ab.2016.06.003). — Compact modern review of SH, dynamic, and multi-coil shimming.
3. Juchem C, Cudalbu C, de Graaf RA, et al. **B0 shimming for in vivo magnetic resonance spectroscopy: Experts' consensus recommendations.** *NMR in Biomedicine.* 2021;34:e4350. [doi:10.1002/nbm.4350](https://doi.org/10.1002/nbm.4350). — Practical consensus on field mapping, shim optimization, and reporting.
4. Stockmann JP, Wald LL. **In vivo B0 field shimming methods for MRI at 7 T.** *NeuroImage.* 2018;168:71-87. [doi:10.1016/j.neuroimage.2017.06.013](https://doi.org/10.1016/j.neuroimage.2017.06.013). — High-field brain shimming, including dynamic SH and local arrays.
5. While PT. **A comparative simulation study of the performance of gradient coils designed using conventional, minimum-inductance and minimum-power methods.** *Concepts in Magnetic Resonance Part B.* 2012. — Useful engineering comparison of regularization objectives and coil metrics.
6. Mäkinen AJ, Zetter R, Iivanainen J, et al. **Magnetic-field modeling with surface currents: Physical and computational principles of bfieldtools.** *Journal of Applied Physics.* 2020;128:063906. [doi:10.1063/5.0016090](https://doi.org/10.1063/5.0016090). — Modern triangular-mesh stream-function formulation, operators, energy, inductance, and topology.
7. Zetter R, Mäkinen AJ, Iivanainen J, et al. **Magnetic-field modeling with surface currents: Implementation and usage of bfieldtools.** *Journal of Open Source Software.* 2020;5:2363. [doi:10.21105/joss.02363](https://doi.org/10.21105/joss.02363). — Software-oriented companion with coil-design examples.
8. Amrein P, Jia F, Zaitsev M, Littin S. **CoilGen: Open-source MR coil layout generator.** *Magnetic Resonance in Medicine.* 2022;88:1465-1479. [doi:10.1002/mrm.29294](https://doi.org/10.1002/mrm.29294). — Automated post-processing from a surface-current solution to connected, non-overlapping manufacturable wire paths.

## 2. Classical shim coils and spherical-harmonic foundations

9. Roméo F, Hoult DI. **Magnet field profiling: Analysis and correcting coil design.** *Magnetic Resonance in Medicine.* 1984;1(1):44-65. [doi:10.1002/mrm.1910010107](https://doi.org/10.1002/mrm.1910010107). — Foundational spherical-harmonic analysis and construction of zonal/tesseral correction coils.
10. Hoult DI, Lee D. **Shimming a superconducting nuclear-magnetic-resonance imaging magnet with steel.** *Review of Scientific Instruments.* 1985;56:131-135. — Important classical passive-shim counterpart to active coils.
11. Schenck JF, Jolesz FA, Roemer PB, et al. **Superconducting open-configuration MR imaging system for image-guided therapy.** *Radiology.* 1995;195:805-814. — System-level context for open geometry and constrained correction/gradient coils.
12. Koch KM, McIntyre S, Nixon TW, Rothman DL, de Graaf RA. **Dynamic shim updating on the human brain.** *Journal of Magnetic Resonance.* 2006;180:286-296. [doi:10.1016/j.jmr.2006.03.007](https://doi.org/10.1016/j.jmr.2006.03.007). — Dynamic higher-order SH shimming and hardware demands.
13. Morrell G, Spielman D. **Dynamic shimming for multi-slice magnetic resonance imaging.** *Magnetic Resonance in Medicine.* 1997;38:477-483. — Seminal slice-wise dynamic-shim implementation.
14. Gruetter R, Boesch C. **Fast, noniterative shimming of spatially localized signals: In vivo analysis of the magnetic field along axes.** *Journal of Magnetic Resonance.* 1992;96:323-334. — FASTMAP; important bridge between field characterization and coil-current fitting.

## 3. Target-field, inverse-problem, and stream-function foundations

15. Turner R. **A target field approach to optimal coil design.** *Journal of Physics D: Applied Physics.* 1986;19:L147-L151. [doi:10.1088/0022-3727/19/8/001](https://doi.org/10.1088/0022-3727/19/8/001). — Foundational inversion of cylindrical surface currents from a prescribed interior target field.
16. Pissanetzky S. **Minimum energy MRI gradient coils of general geometry.** *Measurement Science and Technology.* 1992;3:667-673. [doi:10.1088/0957-0233/3/7/005](https://doi.org/10.1088/0957-0233/3/7/005). — Minimum magnetic-energy/inductance solution on general surfaces.
17. Peeren GN. **Stream function approach for determining optimal surface currents.** *Journal of Computational Physics.* 2003;191(1):305-321. [doi:10.1016/S0021-9991(03)00320-6](https://doi.org/10.1016/S0021-9991(03)00320-6). — General mathematical foundation for divergence-free sheet currents represented directly by a scalar stream function.
18. Brideson MA, Forbes LK, Crozier S. **Determining complicated winding patterns for shim coils using stream functions and the target-field method.** *Concepts in Magnetic Resonance.* 2002;14(1):9-18. [doi:10.1002/cmr.10000](https://doi.org/10.1002/cmr.10000). — Directly relevant to high-order shim winding topology and contour extraction.
19. Lemdiasov RA, Ludwig R. **A stream function method for gradient coil design.** *Concepts in Magnetic Resonance Part B.* 2005;26B:67-80. [doi:10.1002/cmr.b.20040](https://doi.org/10.1002/cmr.b.20040). — Piecewise-linear stream functions on general surfaces with Biot-Savart target fitting.
20. Poole M, Bowtell R. **Novel gradient coils designed using a boundary element method.** *Concepts in Magnetic Resonance Part B.* 2007;31B:162-175. [doi:10.1002/cmr.b.20091](https://doi.org/10.1002/cmr.b.20091). — Arbitrary triangular surfaces, asymmetric/shielded geometries, and practical BEM coil examples.
21. Poole M, Weiss P, Sánchez López H, Ng M, Crozier S. **Minimax current density coil design.** *Journal of Physics D: Applied Physics.* 2010;43:095001. [doi:10.1088/0022-3727/43/9/095001](https://doi.org/10.1088/0022-3727/43/9/095001). — L-infinity current-density control for wire spacing, efficiency, and localized heating.
22. Sánchez López H, Poole M, Crozier S. **Quasi-static multi-domain inverse boundary element method for MRI coil design with minimum induced electric field.** *Engineering Analysis with Boundary Elements.* 2011;35:238-245. [doi:10.1016/j.enganabound.2010.08.006](https://doi.org/10.1016/j.enganabound.2010.08.006). — Extends inverse coil design to induced-E-field objectives.
23. While PT, Forbes LK, Crozier S. **An inverse method for designing gradient coils with minimum maximum current density.** *Concepts in Magnetic Resonance Part B.* — Related minimax/BEM development; useful for current-density and spacing constraints.
24. Jia F, Littin S, Amrein P, et al. **Design of a shim coil array matched to the human brain anatomy.** *Magnetic Resonance in Medicine.* 2020;83:1447-1461. [doi:10.1002/mrm.28016](https://doi.org/10.1002/mrm.28016). — Anatomy- and population-matched shim-array design using piecewise-linear stream functions, field-map training data, and power constraints.
25. de Vos B, Fuchs P, O'Reilly T, Webb A, Remis R. **Gradient coil design and realization for a Halbach-based MRI system.** *IEEE Transactions on Magnetics.* 2020;56:1-8. [doi:10.1109/TMAG.2020.3000906](https://doi.org/10.1109/TMAG.2020.3000906). — Target-field and stream-function design for transverse-field low-field MRI.
26. Hidalgo-Tobon SS. **Theory of gradient coil design methods for magnetic resonance imaging.** *Concepts in Magnetic Resonance Part A.* 2010;36A:223-242. [doi:10.1002/cmr.a.20163](https://doi.org/10.1002/cmr.a.20163). — Pedagogical synthesis of analytical, target-field, and numerical approaches.

## 4. Gradient-coil engineering closely related to shim-coil design

27. Mansfield P, Chapman B. **Active magnetic screening of gradient coils in NMR imaging.** *Journal of Magnetic Resonance.* 1986;66:573-576. — Seminal active-shielding concept.
28. Turner R, Bowley RM. **Passive screening of switched magnetic field gradients.** *Journal of Physics E: Scientific Instruments.* 1986;19:876-879. — Shielding and eddy-current context.
29. Roemer PB, Hickey JS. **Self-shielded gradient coils for nuclear magnetic resonance imaging.** *Magnetic Resonance in Medicine.* 1988;6:1-15. — Classical actively shielded gradient construction.
30. Bowtell R, Mansfield P. **Gradient coil design using active magnetic screening.** *Magnetic Resonance in Medicine.* 1991;17:15-27. — Analytical treatment of shielded cylindrical gradients.
31. Chronik BA, Rutt BK. **A comparison between force-balanced and torque-balanced asymmetric gradient coils.** *Magnetic Resonance in Medicine.* 1998;39:766-774. — Mechanical balance criteria relevant to asymmetric shim/gradient designs.
32. Tang F, Lopez HS, Freschi F, et al. **Skin and proximity effects in the conductors of split gradient coils for a hybrid MRI-LINAC system.** *Medical Physics.* — Conductor-loss effects beyond filamentary Biot-Savart models.
33. Davids M, Guérin B, Schad LR, Wald LL. **Optimization of MRI gradient coils with explicit peripheral nerve stimulation constraints.** *IEEE Transactions on Medical Imaging.* 2021;40:129-142. [doi:10.1109/TMI.2020.3023329](https://doi.org/10.1109/TMI.2020.3023329). — BEM-SF optimization with field, shielding, torque, wire-spacing, and PNS constraints.
34. Davids M, Guerin B, Vom Endt A, Schad LR, Wald LL. **Prediction of peripheral nerve stimulation thresholds of MRI gradient coils using coupled electromagnetic and neurodynamic simulations.** *Magnetic Resonance in Medicine.* 2019;81:686-701. [doi:10.1002/mrm.27405](https://doi.org/10.1002/mrm.27405). — Forward model underlying PNS-aware coil optimization.
35. Zhou Y, Liu Z, Zhang Q, et al. **A novel method for highly linear gradient coil design based on stream function.** *Medical Physics.* 2024;51:7180-7190. [doi:10.1002/mp.17301](https://doi.org/10.1002/mp.17301). — Recent parameterized stream-function/particle-swarm method.

## 5. Localized, multi-coil, and integrated RF/B0 shim arrays

36. Juchem C, Nixon TW, McIntyre S, Boer VO, Rothman DL, de Graaf RA. **Dynamic multi-coil shimming of the human brain at 7 Tesla.** *Journal of Magnetic Resonance.* 2011;212:280-288. [doi:10.1016/j.jmr.2011.07.005](https://doi.org/10.1016/j.jmr.2011.07.005). — Landmark human multi-coil shim demonstration.
37. Juchem C, Brown PB, Nixon TW, McIntyre S, Rothman DL, de Graaf RA. **Multi-coil shimming of the mouse brain.** *Magnetic Resonance in Medicine.* 2011;66:893-900. — Establishes localized basis coils as an alternative to SH modes.
38. Juchem C, Herman P, Sanganahalli BG, et al. **DYNAmic Multi-coIl TEchnique (DYNAMITE) shimming of the rat brain at 11.7 T.** *NMR in Biomedicine.* 2014;27:897-906. [doi:10.1002/nbm.3133](https://doi.org/10.1002/nbm.3133).
39. Juchem C, Rudrapatna US, Nixon TW, McIntyre S, de Graaf RA. **Dynamic Multi-Coil Technique (DYNAMITE) shimming for echo-planar imaging of the human brain at 7 Tesla.** *NeuroImage.* 2015;105:462-472. [doi:10.1016/j.neuroimage.2014.11.011](https://doi.org/10.1016/j.neuroimage.2014.11.011).
40. Aghaeifar A, Zhou J, Heule R, et al. **Dynamic B0 shimming of the human brain at 9.4 T with a 16-channel multi-coil shim setup.** *Magnetic Resonance in Medicine.* 2018;80:1714-1725. [doi:10.1002/mrm.27110](https://doi.org/10.1002/mrm.27110).
41. Stockmann JP, Witzel T, Keil B, et al. **A 32-channel combined RF and B0 shim array for 3T brain imaging.** *Magnetic Resonance in Medicine.* 2016;75:441-451. [doi:10.1002/mrm.25587](https://doi.org/10.1002/mrm.25587). — AC/DC receive-loop array; localized shimming without a separate shim former.
42. Stockmann JP, Wald LL. **A 31-channel integrated AC/DC B0 shim and radiofrequency receive array coil for improved 7T MRI.** *Magnetic Resonance in Medicine.* 2022;87:1625-1640. [doi:10.1002/mrm.29022](https://doi.org/10.1002/mrm.29022).
43. Truong TK, Darnell D, Song AW. **Integrated RF/shim coil array for parallel reception and localized B0 shimming in the human brain.** *NeuroImage.* 2014;103:235-240. [doi:10.1016/j.neuroimage.2014.09.052](https://doi.org/10.1016/j.neuroimage.2014.09.052). — iPRES architecture.
44. Overson DK, Darnell D, Robb F, Song AW, Truong TK. **Flexible multi-purpose integrated RF/shim coil array for MRI and localized B0 shimming.** *Magnetic Resonance in Medicine.* 2024;91:842-849. [doi:10.1002/mrm.29891](https://doi.org/10.1002/mrm.29891).
45. Juchem C, Theilenberg S, Kumaragamage C, et al. **Dynamic multicoil technique (DYNAMITE) MRI on human brain.** *Magnetic Resonance in Medicine.* 2020;84:2953-2963. [doi:10.1002/mrm.28323](https://doi.org/10.1002/mrm.28323). — Demonstrates that the same localized coil basis can provide spatial encoding as well as shimming.

## 6. RF-coil foundations and optimal surface-current formulations

46. Hoult DI, Richards RE. **The signal-to-noise ratio of the nuclear magnetic resonance experiment.** *Journal of Magnetic Resonance.* 1976;24:71-85. [doi:10.1016/0022-2364(76)90233-X](https://doi.org/10.1016/0022-2364(76)90233-X). — Fundamental RF-coil SNR and reciprocity analysis.
47. Hoult DI. **The NMR receiver: A description and analysis of design.** *Progress in Nuclear Magnetic Resonance Spectroscopy.* 1978;12:41-77. [doi:10.1016/0079-6565(78)80002-8](https://doi.org/10.1016/0079-6565(78)80002-8).
48. Ackerman JJH, Grove TH, Wong GG, Gadian DG, Radda GK. **Mapping of metabolites in whole animals by 31P NMR using surface coils.** *Nature.* 1980;283:167-170. [doi:10.1038/283167a0](https://doi.org/10.1038/283167a0). — Early MRI/MRS surface-coil application.
49. Hayes CE, Edelstein WA, Schenck JF, Mueller OM, Eash M. **An efficient, highly homogeneous radiofrequency coil for whole-body NMR imaging at 1.5 T.** *Journal of Magnetic Resonance.* 1985;63:622-628. [doi:10.1016/0022-2364(85)90229-1](https://doi.org/10.1016/0022-2364(85)90229-1). — Birdcage coil.
50. Roemer PB, Edelstein WA, Hayes CE, Souza SP, Mueller OM. **The NMR phased array.** *Magnetic Resonance in Medicine.* 1990;16:192-225. [doi:10.1002/mrm.1910160203](https://doi.org/10.1002/mrm.1910160203). — Foundational receive-array theory and optimal combination.
51. Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. **SENSE: Sensitivity encoding for fast MRI.** *Magnetic Resonance in Medicine.* 1999;42:952-962. [doi:10.1002/(SICI)1522-2594(199911)42:5%3C952::AID-MRM16%3E3.0.CO;2-S](https://doi.org/10.1002/(SICI)1522-2594(199911)42:5%3C952::AID-MRM16%3E3.0.CO;2-S). — Makes array geometry/noise covariance a direct design objective.
52. Wiesinger F, Boesiger P, Pruessmann KP. **Electrodynamics and ultimate SNR in parallel MR imaging.** *Magnetic Resonance in Medicine.* 2004;52:376-390. [doi:10.1002/mrm.20183](https://doi.org/10.1002/mrm.20183). — Ultimate intrinsic SNR via complete current/field bases.
53. Lattanzi R, Sodickson DK. **Ideal current patterns yielding optimal signal-to-noise ratio and specific absorption rate in magnetic resonance imaging: Computational methods and physical insights.** *Magnetic Resonance in Medicine.* 2012;68:286-304. [doi:10.1002/mrm.23198](https://doi.org/10.1002/mrm.23198). — Optimal current modes on enclosing surfaces.
54. Sodickson DK, Lattanzi R, Vaidya M, et al. **The optimality principle for MR signal excitation and reception: New physical insights into ideal radiofrequency coil design.** *Magnetic Resonance in Medicine.* 2018. [doi:10.1002/mrm.27515](https://doi.org/10.1002/mrm.27515). — Reciprocity-based prediction of point-optimal surface-current patterns.
55. Vaidya MV, Collins CM, Sodickson DK, et al. **Dependence of B1+ and B1− field patterns of surface coils on the electrical properties of the sample and the MR operating frequency.** *Concepts in Magnetic Resonance Part B.* — Useful for judging when quasi-static stream-function design ceases to represent RF behavior.
56. Zhu Y. **Parallel excitation with an array of transmit coils.** *Magnetic Resonance in Medicine.* 2004;51:775-784. [doi:10.1002/mrm.20011](https://doi.org/10.1002/mrm.20011). — Foundational parallel-transmit/B1-shimming formulation.

## 7. Additional recent or specialized papers worth tracking

57. Huang Q, Huang L, Zhang H, et al. **Design method of magnetic resonance biplanar gradient coil based on target field method and stream function.** *Frontiers in Physics.* 2025;13:1579043. [doi:10.3389/fphy.2025.1579043](https://doi.org/10.3389/fphy.2025.1579043).
58. Niu J, et al. **High-order spherical-harmonic shim-coil design using stream-function optimization.** 2022. — Locate by title/author before formal citation; relevant recent application of mesh-based SF optimization to high-order SH coils.
59. Tamada D, Kose K. **A new planar single-channel shim coil using multiple circular currents for magnetic resonance imaging.** *Applied Physics Express.* 2012. — Planar/local correction using circular-current superposition and stream-function concepts.
60. Su S, Zhang B, Henning A. **Development of an automated workflow for stream-function based B0 shim coil design and experimental integration with RF coils at 7T.** *ISMRM 2025*, abstract 3474. [doi:10.58530/2025/3474](https://doi.org/10.58530/2025/3474). — Conference work; useful for current implementation practice but not yet a full journal paper.

## 8. Recommended reading order for CoilGen-related work

1. Turner (1986) for the target-field inverse problem.
2. Roméo and Hoult (1984) plus Brideson et al. (2002) for SH targets and complicated shim winding topology.
3. Peeren (2003) for the rigorous stream-function formulation and boundary/topology constraints.
4. Lemdiasov and Ludwig (2005), then Poole and Bowtell (2007), for discretization on surfaces and arbitrary-geometry BEM design.
5. Poole et al. (2010) for current-density regularization and wire-spacing implications.
6. Mäkinen et al. (2020) for modern triangular-mesh operators, inductance, energy, and multiply connected surfaces.
7. Jia et al. (2020) for data-driven anatomy-specific shim design with power constraints.
8. Amrein et al. (2022) for contour discretization, loop interconnection, and manufacturable track generation.
9. Davids et al. (2021) for explicit engineering and physiological constraints in BEM-SF optimization.

## 9. Reproducible search strings for extending the list

- `("shim coil" OR "B0 shimming") AND (MRI OR NMR) AND (design OR optimization)`
- `("stream function" OR "surface current") AND (MRI OR "magnetic resonance") AND coil`
- `("target field method" OR "inverse boundary element") AND (gradient OR shim) AND coil`
- `("multi-coil" OR "local shim array" OR DYNAMITE OR iPRES OR "AC/DC array") AND MRI`
- `("spherical harmonic" AND "shim coil") AND (design OR construction)`
- `("minimum inductance" OR "minimum power" OR "minimum current density" OR minimax) AND "gradient coil"`
- `("ideal current patterns" OR "ultimate intrinsic SNR") AND MRI AND coil`

Use citation chaining from Turner (1993), Juchem and de Graaf (2017), Stockmann and Wald (2018), Mäkinen et al. (2020), and the 2024 magnetic-field-control review to capture older terminology and recent papers that keyword searches miss.
