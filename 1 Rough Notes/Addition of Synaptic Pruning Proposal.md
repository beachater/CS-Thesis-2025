2025-08-28 11:54

# Addition of Synaptic Pruning Proposal

#### What is Synaptic Pruning?

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

> ```
> [IICO]  Input f, dimension d, population size N, elite size m, immigrant rate r_fresh
[IICO]  Initialize population P = {x_i}^N from prior
[IICO]  Initialize memory M as empty
[IICO]  Initialize activity traces a_i = 0 for all x_i in P and for future memory items
[IICO]  Initialize stagnation counter S = 0
[IICO]  Set p_min, beta, epsilon_imp, alpha, q_max, B_max, b0, gamma
[IICO]  Set diversity guard kappa for elite protection

[IICO]  while not termination do
[IICO]      Evaluate f(x_i) for all x_i in P
[IICO]      Find elites E by top m with farthest point spread
[IICO]      Update best value f_star and compare to previous to update S

[IICO]      Compute elite covariance C from E
[IICO]      Eigendecompose C = U Lambda U^T

[IICO]      for each x_i in P do
[IICO]          Clone x_i according to cloning rule
[IICO]          In eigen basis, compute plasticity weights w_t using improvement signals
[IICO]          Sample mutation mask M_k with probabilities from normalized w_t with p_min floor
[IICO]          Mutate clone along eigen components selected by mask with IICO adaptive step rules
[IICO]      end for

[IICO]      Evaluate offspring and select next P using IICO replacement policy

[IICO]      For all survivors update improvement Δ_i = max{0, f_star_prev − f(x_i)}
[IICO]      Update activity traces a_i = beta a_i + (1 − beta) phi(Δ_i)

[FCSA]      Compute adaptive forgetting threshold theta_t from stagnation S
[FCSA]      For each memory item y in M if its activity a_y < theta_t then delete y
[FCSA]      Refill M to target size by inserting diverse elites or immigrants

[SP]        Compute redundancy R_i for each x_i in P using kernel similarity
[SP]        Compute pruning score Π_i = α (1 − a_i) + (1 − α) R_i
[SP]        Determine pruning budget B_t = min{B_max, b0 + γ S}
[SP]        Protect kappa farthest elites from pruning
[SP]        Remove the top B_t items by Π_i excluding protected elites
[SP]        For each removed slot
[SP]            With probability r_fresh insert immigrant from broad prior
[SP]            Otherwise insert child of a diverse elite far from current population
[SP]        end for

[IICO]      Optionally update memory M with a few best diverse items from P
[IICO]      Ensure population size and memory size are restored

[IICO]  end while

> ```

