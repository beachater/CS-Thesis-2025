2025-09-08 15:35

Link or File:

Status:

Tags: 

# Improved clonal selection algorithm based  on the directional update strategy

> [!PDF|yellow] [[thesis2025/files/Improved clonal selection algorithm based on the directional update strategy.pdf#page=1&selection=39,25,41,29&color=yellow|p.19312]]
> > To solve this problem, our improved algorithm introduces a crowding degree factor in the antibody updating stage to determine whether there is crowding between antibodies. 



> [!PDF|yellow] [[thesis2025/files/Improved clonal selection algorithm based on the directional update strategy.pdf#page=1&selection=35,76,37,47&color=yellow|p.19312]]
> > The clonal selection algorithm is traditionally updated through a random complement of antibodies, which is a blind and uncertain process.

> [!PDF|yellow] [[thesis2025/files/Improved clonal selection algorithm based on the directional update strategy.pdf#page=1&selection=42,69,44,10&color=yellow|p.19312]]
> > to update in the direction of the global optimal solution and ensures stable convergence with fewer iterations

> [!PDF|note] [[thesis2025/files/Improved clonal selection algorithm based on the directional update strategy.pdf#page=3&selection=34,81,36,38&color=note|p.19314]]
> >  In this paper, we refer to the phenomenon of antibody clustering caused by random updates of the algorithm as “crowding.

> [!PDF|note] [[thesis2025/files/Improved clonal selection algorithm based on the directional update strategy.pdf#page=3&selection=39,17,45,10&color=note|p.19314]]
> > The basic idea of the algorithm is to set a minimum distance threshold to determine whether the complementary antibodies are “crowded” and to eliminate antibodies that are “crowded” and have poor affinity with the optimal antibodies in the current iteration so that the complementary antibodies are updated along a direction that is closer to the global optimal solution, increasing the chance of obtaining the global optimal solution and ensuring a stable convergence of the algorithm.

> [!PDF|note] [[thesis2025/files/Improved clonal selection algorithm based on the directional update strategy.pdf#page=6&selection=29,61,31,82&color=note|p.19317]]
> > By directing the randomly added antibodies to be updated in a direction close to the global optimum, the algorithm converges steadily and quickly, and the accuracy of the search is higher

> [!PDF|red] [[thesis2025/files/Improved clonal selection algorithm based on the directional update strategy.pdf#page=7&selection=14,7,17,79&color=red|p.19318]]
> >  The integration of the directed update strategy into the update operation of the clonal selection algorithm regulates the balance between randomness and determinism, thus avoiding the problems of low accuracy in finding the best solution and slow convergence rate due to complete randomness in the local search phase [18]

> [!PDF|important] [[thesis2025/files/Improved clonal selection algorithm based on the directional update strategy.pdf#page=10&selection=45,1,46,56&color=important|p.19321]]
> > n this section, 10 functions from the CEC test functions for optimization algorithms are selected to test the performance of the algorithms. 

> [!PDF|important] [[thesis2025/files/Improved clonal selection algorithm based on the directional update strategy.pdf#page=10&selection=47,72,49,77&color=important|p.19321]]
> > his paper (DUSCSA) with the classical genetic algorithm (GA) and clonal selection algorithm (CSA) to verify whether the improvements proposed in this paper are effective

> [!PDF|important] [[thesis2025/files/Improved clonal selection algorithm based on the directional update strategy.pdf#page=11&selection=26,0,52,25&color=important|p.19322]]
> > The test functions selected for this paper are shown in Table 4. Among them, f1 ∼ f6 are single-peaked test functions, which have the common feature of having one global minimum, but are otherwise complex functions with multiple local minima and are suitable for testing the convergence speed and convergence stability of the detection algorithms. f7 ∼ f10 are multipeaked test functions, which have the common feature of having multiple global minima and multiple local minima, so they are suitable for testing the performance of the detection algorithms on high-dimensional complex functions.

> [!PDF|important] [[thesis2025/files/Improved clonal selection algorithm based on the directional update strategy.pdf#page=14&selection=5,0,12,80&color=important|p.19325]]
> > For the single-peak test functions, the DUSCSA has more stable and reliable convergence and better accuracy than the GA and CSA. For the multipeak test functions, the optimization accuracy of all three algorithms decreases as the dimensionality increases, but the DUSCSA’s optimization accuracy is less affected by the dimensionality and maintains good convergence stability. The experimental results show that the update strategy proposed in this paper is effective, and the algorithm has good robustness and ensures stable convergence. The comparative graphs of the convergence curves of the six tested functions clearly

> [!PDF|important] [[thesis2025/files/Improved clonal selection algorithm based on the directional update strategy.pdf#page=17&selection=0,0,11,15&color=important|p.19328]]
> > 19328 C. Yang et al.1 3 show that the DUSCSA is more accurate than the CSA under the same number of iterations; the DUSCSA has fewer iterations than the CSA under the same affinity and therefore converges faster. The comparison of the results reveals that the overall performance of the DUSCSA is 1% higher than that of the CSA and 2.2% higher than that of the GA. In summary, the update strategy proposed in this paper is an effective way to improve the algorithm’s search accuracy and convergence stability

> [!PDF|yellow] [[thesis2025/files/Improved clonal selection algorithm based on the directional update strategy.pdf#page=18&selection=15,22,18,61&color=yellow|p.19329]]
> >  If the distance is less than the threshold and the affinity is worse than that of the optimal antibody in the current iteration, then the antibody is readded to increase the probability of finding the global optimal solution and improve the convergence speed and stability of the algorithm.

> [!PDF|yellow] [[thesis2025/files/Improved clonal selection algorithm based on the directional update strategy.pdf#page=18&selection=18,62,20,79&color=yellow|p.19329]]
> > If the randomly added antibodies are located close to the global optimum or have high affinity, the probability of obtaining the global optimum is higher, and convergence is faste

PROBLEM:
 For the multipeak test functions, the optimization accuracy of all three algorithms decreases as the dimensionality increases, but the DUSCSA’s optimization accuracy is less affected by the dimensionality and maintains good convergence stability.