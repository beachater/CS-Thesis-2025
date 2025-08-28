
- [ ] Usage of dynamic 
- [ ] Synaptic Pruning
- [ ] Apoptosis
- [ ] 




Regulatory T cells (Tregs) 


- Introduce **biologically inspired diversity regulators** (e.g., immune feedback or Treg suppression).
    
- Develop **landscape-aware mutation control** that adapts to ruggedness and rotation.
    
- Embed **constraint-handling logic** into the selection or mutation phase.

addittional notes:

ICO (the origin sa IICO) kai indirect extenstion of CSA (not directly kai wala jud na cite properly ang CSA sa paper, nabutang siya sa reference pero ang in-text citation lahi ang pasabot), according gpt murag thematic borrowing daw, same principles ang gina use pero mas more advance ang each step ( not suprising kai 2002 pa ang original CSA then kani 2021) duda nako wala na consider sa mga reviewers ang CSA sad ani since na publish man jud siya as novel

FCSA weakness and gap:

sakto and certain na jud ang na mention na problem, ako lang gi simple:

- Low- optimization precision sa high dimensional test functions (suggests a limitation or scalability issue in the algorithm's design.)
- basically lacks real time optimal solution (forget the outdated experience)

IICO weakness and gap:

> 
> > "Table 10 indicates the results of all algorithms in fixed-dimension multimodal functions. Similar to the previous analysis, the results of IICO are overall better than ICO. Overall, IICO, EO and AOA have the best performance."
> 
> sakto japun pero ako lang clarify nga unsa pasabot sa "Not good in fixed-dimension multimodel functions " kai sa paper kai significanlty better siya sa ICO and apil sad ang IICO sa mga best performance

pero ako emphasize nga mentioned sa paper nga na fall short siya CEC test functions (which is basically kato mentioned ni via na more chance na ma fall sa local optimum) kai ang CEC test functions kai ga mimic na sa real world (like naa daw deceptive local optima) this suggest na

"IICO lacks the robustness and landscape-awareness needed to effectively navigate the deceptive, rotated, and constraint-heavy nature of CEC test function" summary ni gpt sa gap sa IICO regarding sa CEC, **so basically it doesnt generalize well


we are planning to hybrid a forgetting based CSA and the IICO, from ive gathered so far the IICO lacks the robustness and generalizbility for CEC

| Section                            | Summary                                                                                                                                                                                     | Significance                                                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **3.1 Experiment Setup**           | Benchmarks include 27 functions (unimodal, multimodal, fixed-dimension), 8 CEC2014 functions, and 3 engineering problems. IICO is compared against 10 algorithms using 30 Monte Carlo runs. | Establishes a rigorous and diverse testing framework to validate IICO’s performance across problem types.          |
| **3.2 Parameter Sensitivity**      | Analyzed the impact of the new parameter δ\delta controlling exploration-exploitation balance. Best performance observed at δ=3\delta = 3.                                                  | Shows that fine-tuning this parameter is crucial for maintaining diversity and avoiding premature convergence.     |
| **3.3 Improvement Analysis**       | Compared IICO with ICO-OBL and ICO-APS. IICO outperforms both in convergence speed and accuracy.                                                                                            | Demonstrates that combining opposition-based learning and adaptive parameter strategy yields synergistic benefits. |
| **3.4 Qualitative Analysis**       | Visual comparisons show IICO converges faster and clusters near optima more effectively than ICO.                                                                                           | Provides intuitive evidence of IICO’s superior search behavior and population dynamics.                            |
| **3.5 Intensification Capability** | On unimodal functions, IICO achieves optimal values with zero variance, matching or slightly outperforming ICO.                                                                             | Confirms IICO’s strong exploitation ability and stability in simple landscapes.                                    |
| **3.6 Diversification Capability** | On multimodal functions, IICO shows better global search and solution accuracy than ICO, especially in F10, F13–F18.                                                                        | Validates that OBL enhances exploration, helping IICO escape local optima.                                         |
| **3.7 Acceleration Convergence**   | IICO consistently reaches optima faster than other algorithms across most functions.                                                                                                        | Highlights IICO’s efficiency and reduced stagnation due to adaptive acceleration.                                  |
| **3.8 Scalability Experiments**    | IICO maintains performance across increasing dimensions in unimodal functions; less stable in multimodal cases.                                                                             | Indicates strong scalability in simple problems, but reveals limitations in complex high-dimensional landscapes.   |
| **3.9 Wilcoxon Rank Sum Test**     | Statistically significant differences (p < 0.05) between IICO and most other algorithms, especially in multimodal and fixed-dimension cases.                                                | Confirms that IICO’s improvements are not just anecdotal—they’re statistically validate                            |


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