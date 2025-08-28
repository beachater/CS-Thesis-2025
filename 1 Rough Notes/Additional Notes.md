2025-08-28 07:30

Link or File:

Status:

Tags: [[IICO.pdf]] [[Improved_Clonal_Selection_Algorithm_Based_on_Biolo.pdf]]

# Additional Notes

From what i have understand so far: 

IICO solvea these problems 

1. Stagnation sa optimal value 

2. Getting trapped in local optima Their solution: Adaptive parameter: designed to prevent the stagnation of optimal value (like dili na ga improve after many iterations) 
	1. 2 Quasi mechanism: which solves problem in local optima (when algo fails to find the best solution) 
		1. Opposition based - solves population diversitu kato opposite direction iya isearch 
		2. Reflection - more refined local search improvind convergence speed and accuracy

FCSA: Problems tackled: 

Black hole phenomenon - inactive solutions nga nag accumulate making it harder for potential good solutions to be introduced and hinders algorithm to finding the global optima 
			 
Solution: Forgetting mechanism so here sa paper murag gi state na nga able to improve na the population diversity, convergence, and speed 

Ang FCSA nga black hole phenomena ang stagnation iya gina solve is only in the stagnation of accumulation of inactive solutions 

Ang IICO nga stagnation is more general, like stagnation due to numerous factors like stagnation kay dili diverse ang popu, or stagnation kay stuck siya sa local optima Same ra silag goal tho finding global optima, and global optima ia found by enhance diversity, convergence stability, and etc which mao to ila gipang tests
		
			 

##### 

Via Nicole

mao to first idea is have iico then forgetting mechanism but then si iico wala naman guy problem niya nga accumulation of inactive solutions kay na handle na ni sa iico nga algorithm, ila algo naka cater na to always update and keep only the best (elitism)

##### 

Via Nicole

IICO weaknesses: 
- Not good in fixed-dimension multimodel functions 
- Unimodal test functions is same ra siyag result sa ICO so it means wala ra kaayo changes sa original algo 
- Si adaptive parameter is prevent siyag stagnation however gina reduce niya ang time of exploration it means instead of like searching more daritso na siya mag fine tuning saiya napili nga smaller search space and since wala siya nag explore more, naa chance na fall sa local optimum

##### 

Via Nicole

FCSA weakness 
- So si FCSA naa uban test functions not good kaayo ang performance, naa sad uban nga best siya. 
- Si FCSA as stated sa limitations naa japun siya problem with the fine tuning sa finding sa global optima or the exact precise optimal value. 
- Low- optimization precision kumbaga in high dimensional test functions based sa experiment. Gibutang diri sa future studies na like having it improve to cater problems where optimal solution changes over time


# Additional Notes

> [!PDF|yellow] [[Improved_Clonal_Selection_Algorithm_Based_on_Biolo.pdf#page=9&selection=51,0,54,23&color=yellow|Improved_Clonal_Selection_Algorithm_Based_on_Biolo, p.9]]
> > when using this algorithm in the initial experimental environment which is the same as the CSA and GA optimization under the condition of higher precision, convergence is stable and reliable.

> [!PDF|red] [[Improved_Clonal_Selection_Algorithm_Based_on_Biolo.pdf#page=9&selection=61,0,64,41&color=red|Improved_Clonal_Selection_Algorithm_Based_on_Biolo, p.9]]
> > Overall, the experimental results show that FCSA has higher optimization accuracy and stability than CSA, and FCSA has higher optimization accuracy and convergence stability than GA in most test functions.

> [!PDF|red] [[Improved_Clonal_Selection_Algorithm_Based_on_Biolo.pdf#page=9&selection=126,0,129,9&color=red|Improved_Clonal_Selection_Algorithm_Based_on_Biolo, p.9]]
> > However, from the experimental performance in high-dimensional test function, FCSA still has the problem of low optimization precision


addittional notes:

ICO (the origin sa IICO) kai indirect extenstion of CSA (not directly kai wala jud na cite properly ang CSA sa paper, nabutang siya sa reference pero ang in-text citation lahi ang pasabot), according gpt murag thematic borrowing daw, same principles ang gina use pero mas more advance ang each step ( not suprising kai 2002 pa ang original CSA then kani 2021) duda nako wala na consider sa mga reviewers ang CSA sad ani since na publish man jud siya as novel



FCSA weakness and gap:

sakto and certain na jud ang na mention na problem, ako lang gi simple:
- Low- optimization precision sa high dimensional test functions (suggests a limitation or scalability issue in the algorithm's design.)
- basically lacks real time optimal solution (forget the outdated experience)

IICO weakness and gap:


> [!PDF|red] [[IICO.pdf#page=26&selection=7,0,9,26&color=red|IICO, p.26]]
> > Table 10 indicates the results of all algorithms in fixed-dimension multimodal functions. Similar to the previous analysis, the results of IICO are overall better than ICO. Overall, IICO, EO and AOA have the best performance.
> 
> sakto japun pero ako lang clarify nga unsa pasabot sa "Not good in fixed-dimension multimodel functions " kai sa paper kai significanlty better siya sa ICO and apil sad ang IICO sa mga best performance


pero ako emphasize nga mentioned sa paper nga na fall short siya CEC test functions (which is basically kato mentioned ni via na more chance na ma fall sa local optimum) kai ang CEC test functions kai ga mimic na sa real world (like naa daw deceptive local optima) this suggest na 

"IICO lacks the robustness and landscape-awareness needed to effectively navigate the deceptive, rotated, and constraint-heavy nature of CEC test function" summary ni gpt sa gap sa IICO regarding sa CEC, **so basically it doesnt generalize well

	