2025-08-28 11:54

# Addition of Synaptic Pruning Proposal

#### What is Synaptic Pruning?
https://www.youtube.com/watch?v=rxPT78F_ZVE



| Section                            | Summary                                                                                                                                     | Significance                                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **3.2 Parameter Sensitivity**      | The new parameter δ affects exploration-exploitation balance; δ = 3 yielded best results. Performance drops with extreme values.            | Shows that IICO is sensitive to parameter tuning—its performance depends heavily on careful calibration.        |
| **3.3 Improvement Analysis**       | IICO outperforms ICO-OBL and ICO-APS in convergence speed and accuracy, especially in unimodal and some multimodal functions.               | Confirms synergy between OBL and adaptive strategy, but improvements are less pronounced in complex landscapes. |
| **3.4 Qualitative Analysis**       | Visual comparisons show IICO converges faster and clusters near optima better than ICO. However, differences are minimal in some functions. | Offers intuitive evidence of improvement, but also reveals that gains are function-dependent and not universal. |
| **3.5 Intensification Capability** | IICO achieves optimal values with zero variance on unimodal functions, but so do several other algorithms including ICO.                    | Highlights IICO’s stability, yet shows limited numerical advantage over simpler methods in unimodal cases.      |
| **3.6 Diversification Capability** | IICO performs well on multimodal functions like F10, F13–F18, but shows no improvement over ICO in F11, F12, and F15.                       | Demonstrates enhanced exploration, but also exposes inconsistency across multimodal landscapes.                 |
| **3.7 Acceleration Convergence**   | IICO converges faster than most algorithms in unimodal and some multimodal functions, but underperforms in F18 and fixed-dimension cases.   | Validates speed advantage, but reveals that fast convergence doesn’t always translate to better accuracy.       |
| **3.8 Scalability Experiments**    | IICO maintains performance in unimodal functions across dimensions, but becomes unstable in several multimodal cases.                       | Indicates good scalability in simple problems, but limited robustness in high-dimensional complex landscapes.   |
| **3.9 Wilcoxon Rank Sum Test**     | Statistically significant differences found in most cases, but little difference between IICO and ICO in unimodal functions.                | Confirms improvements are statistically valid, but also shows that gains are marginal in simpler scenarios.     |

so basically ang problem currently

FCSA (as pointed ni via)
- Low- optimization precision sa high dimensional test functions (suggests a limitation or scalability issue in the algorithm's design.)
- basically lacks real time optimal solution (forget the outdated experience)

IICO
- under performs sa fixed dimension in terms of acceleration convergance
- doesnt do well in multimodal cases in terms of scalability
- doesnt do well in CEC function "deceptive, rotated, and constrained nature" - characteristics of CEC functions accroding ni chatgpt; leading to instability and poor generalization.

Synaptic pruning is ==the natural process in which the brain eliminates weak and unused synaptic connections to increase the efficiency of neural pathways==, a principle known as "use it or lose it". This essential neurodevelopmental process starts in infancy, is most active during childhood and adolescence, and continues at a reduced pace into adulthood. By streamlining neural networks, synaptic pruning supports healthy brain development, adaptability, and the maturation of complex cognitive functions, though disruptions in this process are linked to conditions like [autism spectrum disorder](https://www.google.com/search?sca_esv=79222c262c147aa4&sxsrf=AE3TifOyXNh5E7b4DuGM4bWVCVub2tU7aw%3A1756353225555&q=autism+spectrum+disorder&sa=X&sqi=2&ved=2ahUKEwjP2e_QzayPAxX3lK8BHa2-DyAQxccNegQIJxAB&mstk=AUtExfBz6a7PN1hApiKVT92hdFwxmpgc5gGA3V9OFPzqcNXc8tDCnfFOuh7b-6ns4mHt2DeUN7tum_NaYpqNvRYsrQkIHH-a1Yuxd8UjsyT5hoWRlo-cxIBJfq4-Oxhwfr-o8xxYGjsMzDt46QPAWO9kO6PryfRVVbr_KXephzBVPIDGIGLPfUdrwBy-035stOgadWsGKlwCVdbxw1hGsN3dmf6_T59saZq56ZQ2QaAakq-hHhXAZNsHRNgtEZKten-4uKi5HvawhT_UYyIA9CKTS7Rg&csui=3) (under-pruning) and [schizophrenia](https://www.google.com/search?sca_esv=79222c262c147aa4&sxsrf=AE3TifOyXNh5E7b4DuGM4bWVCVub2tU7aw%3A1756353225555&q=schizophrenia&sa=X&sqi=2&ved=2ahUKEwjP2e_QzayPAxX3lK8BHa2-DyAQxccNegQIJxAC&mstk=AUtExfBz6a7PN1hApiKVT92hdFwxmpgc5gGA3V9OFPzqcNXc8tDCnfFOuh7b-6ns4mHt2DeUN7tum_NaYpqNvRYsrQkIHH-a1Yuxd8UjsyT5hoWRlo-cxIBJfq4-Oxhwfr-o8xxYGjsMzDt46QPAWO9kO6PryfRVVbr_KXephzBVPIDGIGLPfUdrwBy-035stOgadWsGKlwCVdbxw1hGsN3dmf6_T59saZq56ZQ2QaAakq-hHhXAZNsHRNgtEZKten-4uKi5HvawhT_UYyIA9CKTS7Rg&csui=3) (over-pruning).

##### How it may help the current problems

**Synaptic pruning** is a brain-inspired process where weak or redundant connections are trimmed so that stronger, more useful ones can thrive.

**In optimization (like FCSA/IICO):**

- It means regularly removing **inactive or duplicate solutions** and replacing them with fresh or diverse ones.
    
- This keeps the search **focused on promising directions**, avoids wasting effort on dead ends, and maintains **diversity** across the population.
    

**General benefit for your problem:**  
Synaptic pruning improves **precision in high dimensions**, speeds up **convergence in simple cases**, and helps scale to **multimodal landscapes** by ensuring the algorithm doesn’t get stuck or overcrowded in one area.

- **High-dimensions:** SP acts as a **filter**, sharpening focus on useful directions.
- **Multimodal:** SP acts as a **diversity keeper**, spreading search across many optima.
### **High Dimensions**

- In high-dimensional problems, many candidate solutions contribute little or duplicate each other.
    
- **Synaptic pruning cuts away these low-activity or redundant solutions**, freeing up search capacity.
    
- This lets the algorithm **focus its effort on dimensions/directions that actually matter**, improving precision and reducing wasted evaluations.
    

### **Multimodal Problems**

- In multimodal landscapes, populations often collapse around one local optimum (crowding).
    
- **Synaptic pruning protects diversity** by pruning duplicates and maintaining a spread of elites across different basins.
    
- This ensures the algorithm can **explore multiple peaks/valleys simultaneously**, instead of getting trapped in just one.


| Algorithm | Weakness                                                       | Does SP Help? | Why                                                                                                                                                                  |
| --------- | -------------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FCSA**  | Low optimization precision in high-dimensional test functions  | ✅ Yes         | SP removes redundant/low-activity solutions, focusing search power on meaningful directions → improves precision in high-dim spaces.                                 |
| **FCSA**  | Lacks real-time optimal solution (forgets outdated experience) | ⚠️ Partial    | SP refreshes population by pruning stagnant solutions, indirectly aiding responsiveness. But true “real-time” tracking still depends on FCSA’s forgetting mechanism. |
| **IICO**  | Underperforms in fixed dimensions (slow convergence)           | ✅ Yes         | SP trims redundant clusters, accelerates convergence by forcing introduction of new and diverse candidates.                                                          |
| **IICO**  | Poor scalability in multimodal cases                           | ✅ Yes         | SP preserves niche diversity by pruning duplicates and protecting diverse elites, allowing exploration of multiple optima.                                           |
| **IICO**  | Weak on deceptive functions (CEC)                              | ⚠️ Partial    | SP prevents over-commitment to deceptive valleys by cutting overspecialized groups, but cannot fully escape deception without adaptive mutation control.             |
| **IICO**  | Weak on rotated functions (CEC)                                | ❌ No          | SP does not address rotation; this requires IICO’s rotation-aware parameter adaptation.                                                                              |
| **IICO**  | Weak on constrained functions (CEC)                            | ⚠️ Partial    | SP maintains diverse feasible solutions by pruning redundant ones, but constraint handling requires specialized operators.                                           |
### Algorithm Proposal (With IICO as the main, since its more advance iteration/extension of CSA)

<img width="1724" height="3262" alt="image" src="https://github.com/user-attachments/assets/06574868-5991-4b09-b11d-a8bc7d747c58" />


