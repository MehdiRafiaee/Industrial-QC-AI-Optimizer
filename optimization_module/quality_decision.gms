* ========================================================
* Industrial Quality Control Decision Model
* Objective: Minimize total cost of handling defective items
* Author: Mehdi Rafiee
* ========================================================

Sets
    i   Products /p1*p10/
    d   Decision options /repair, recycle, discard/;

Parameters
    defect_prob(i)      Probability of defect (from ML model)
    cost_repair(i)      Cost to repair product i
    cost_recycle(i)     Cost to recycle product i
    cost_discard(i)     Cost to discard product i
    revenue_loss(i)     Revenue loss if discarded;

* Sample data (in real use, this comes from ML output)
defect_prob(i) = uniform(0.3, 0.9);
cost_repair(i) = uniform(5, 15);
cost_recycle(i) = uniform(2, 8);
cost_discard(i) = 0;
revenue_loss(i) = uniform(20, 40);

Variables
    z               Total cost
    x(i,d)          Binary: 1 if decision d is chosen for product i;

Binary Variable x;
Equations
    obj             Objective function
    one_decision(i) Exactly one decision per product;

obj..
    z =e= sum((i,d),
        x(i,'repair') * (cost_repair(i)) +
        x(i,'recycle') * (cost_recycle(i)) +
        x(i,'discard') * (revenue_loss(i))
    );

one_decision(i)..
    sum(d, x(i,d)) =e= 1;

Model qc_decision /all/;
Solve qc_decision using MIP minimizing z;

Display x.l, z.l;