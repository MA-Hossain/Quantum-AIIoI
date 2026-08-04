# Paper Claim Validation Roadmap

## Objective

Validate the defensible claims of the SENTINEL paper with a faithful implementation, fair baselines, small-instance correctness checks, actual Qiskit experiments, and a final 100-user scalability study.

Increasing the number of users alone does not prove correctness, quantum advantage, or an approximation guarantee. The 100-user experiment must therefore be performed only after the formulation and implementation have been validated on small instances.

## Step 1: List Every Manuscript Claim

- [ ] Extract the exact claims from the abstract, contributions, theory, results, and conclusion.
- [ ] Classify each claim as mathematical, implementation, performance, robustness, quantum, or scalability.
- [ ] Define the evidence required for every claim.
- [ ] Mark claims that cannot be established by simulation alone.

Expected output: a claim-to-evidence table.

## Step 2: Align the Code With the Paper

- [ ] Confirm that the implemented objective is identical to the manuscript objective.
- [ ] Implement update-generation-rate decisions.
- [ ] Implement UE-to-BS/SAT association decisions.
- [ ] Implement discrete bandwidth-allocation levels.
- [ ] Implement discrete computing-allocation levels.
- [ ] Implement freshness-budget constraints.
- [ ] Implement multiple disruption scenarios.
- [ ] Verify that shared bandwidth and computing constraints couple the UE decisions correctly.
- [ ] Ensure that reported worst-case and average AoII use the same definitions as the paper.

Expected output: one documented optimization model shared by the manuscript, QUBO builder, solvers, and metric code.

## Step 3: Validate the AoII Renewal Decomposition

- [ ] Generate Markov source trajectories with known transition matrices.
- [ ] Calculate AoII directly from the simulated trajectories.
- [ ] Calculate AoII using the renewal expression.
- [ ] Compare both estimates over many Monte Carlo runs.
- [ ] Report mean error and confidence intervals.
- [ ] Repeat for slow, volatile, asymmetric, and persistent source models.

Success criterion: the renewal calculation agrees with direct simulation within statistical error.

## Step 4: Validate the Original Problem and QUBO

- [ ] Construct very small instances that can be exhaustively enumerated.
- [ ] Solve the original constrained optimization problem exactly.
- [ ] Solve the corresponding QUBO exactly.
- [ ] Confirm that both produce the same feasible decisions and objective value.
- [ ] Check constraint violations and penalty sensitivity.
- [ ] Test several random seeds and problem configurations.

Success criterion: the QUBO recovers the original optimum consistently without relying on repair or local search.

## Step 5: Establish Fair Classical Baselines

- [ ] Random feasible allocation.
- [ ] Best-rate greedy allocation.
- [ ] AoII-threshold policy with a documented threshold-selection method.
- [ ] Greedy-AoII allocation.
- [ ] Greedy-AoII plus local search.
- [ ] Centralized simulated annealing on the actual QUBO.
- [ ] Exact optimization for small cases.
- [ ] PPO only if training, reward, testing, and hyperparameters are reproducible.
- [ ] Apply identical feasibility repair and post-processing rules to comparable solvers.
- [ ] Report raw and local-search-refined results separately.

Success criterion: comparisons isolate the contribution of each solver rather than the contribution of post-processing.

## Step 6: Validate the Distributed ADMM-Style Decomposition

- [ ] Partition the complete QUBO into meaningful satellite-ground domains.
- [ ] Compare the decomposed result with the centralized exact or SA result.
- [ ] Record primal residual, dual residual, objective value, feasibility, and iteration count.
- [ ] Test multiple penalty values, initializations, domain counts, and random seeds.
- [ ] Report convergence empirically without claiming guaranteed global convergence for binary nonconvex problems.
- [ ] Measure the reduction in variables and subproblem size per domain.

Success criterion: decomposition produces feasible solutions with a documented objective gap and materially smaller subproblems.

## Step 7: Run Actual Qiskit QAOA Experiments

- [ ] Add `exact`, `sa`, and `qaoa` as interchangeable sub-QUBO solvers.
- [ ] Use Qiskit QAOA on small domain subproblems, preferably about 8 to 20 qubits.
- [ ] Test QAOA depths `p = 1, 2, 3` where computationally feasible.
- [ ] Report qubit count, circuit depth, gate count, shots, optimizer evaluations, and runtime.
- [ ] Compare QAOA with exact and SA on identical sub-QUBOs.
- [ ] Report feasibility rate, objective gap, and final network-level AoII.
- [ ] Avoid claiming quantum advantage unless the evidence genuinely establishes it.

Success criterion: QAOA demonstrably participates in the actual SENTINEL workflow and its solution quality is quantified against classical solvers.

## Step 8: Design Robustness Experiments

- [ ] Use nominal, handover, shadowing, correlated outage, congestion, and mixed disruption scenarios.
- [ ] Vary disruption severity systematically.
- [ ] Vary the number of scenarios.
- [ ] Use heterogeneous source transition and distortion models.
- [ ] Vary freshness budgets and resource scarcity.
- [ ] Test both known and unseen disruption scenarios.
- [ ] Report worst-scenario AoII, average AoII, freshness violations, feasibility, and runtime.

Success criterion: SENTINEL's robustness benefit appears consistently in the conditions for which it was designed, not only in one selected instance.

## Step 9: Perform Statistical Evaluation

- [ ] Use enough independent seeds for each configuration.
- [ ] Report means, standard deviations, and 95% confidence intervals.
- [ ] Use paired statistical comparisons on identical instances.
- [ ] Report wins, ties, and losses against strong baselines.
- [ ] Include negative or neutral results where appropriate.
- [ ] Save all configurations, seeds, raw results, and plotting scripts.

Success criterion: conclusions are reproducible and supported by uncertainty estimates rather than isolated runs.

## Step 10: Reconcile Results With Manuscript Claims

- [ ] Retain only claims supported by the completed experiments or valid proofs.
- [ ] Distinguish formulation novelty from solver superiority.
- [ ] Do not describe empirical convergence as a mathematical guarantee.
- [ ] Do not claim finite-depth QAOA guarantees without a correct proof.
- [ ] Do not claim quantum advantage from simulator-only experiments.
- [ ] Update the abstract, contributions, result discussion, and conclusion consistently.

Expected output: a final claim-to-evidence table with every retained claim supported by a figure, table, experiment, or proof.

## Step 11: Final 100-User Experiment

- [ ] Run the validated full model with 100 UEs.
- [ ] Use multiple seeds and disruption levels.
- [ ] Run scalable methods such as greedy, greedy plus local search, and classical distributed decomposition.
- [ ] Use Qiskit only for small decomposed sub-QUBOs, not for the complete 100-user QUBO.
- [ ] Record total variable count, variables/qubits per domain, runtime, memory, iterations, feasibility, and AoII.
- [ ] Compare centralized and distributed methods where centralized execution remains feasible.
- [ ] Present this experiment as evidence of classical/distributed scalability, not as proof of quantum advantage.

Success criterion: the 100-user study is reproducible, remains feasible, and demonstrates the scaling effect of domain decomposition without weakening solution quality unacceptably.

## Final Deliverables

- [ ] Claim-to-evidence table.
- [ ] Correct and documented full optimization implementation.
- [ ] Renewal-validation results.
- [ ] Exact-versus-QUBO correctness results.
- [ ] Fair classical baseline results.
- [ ] ADMM decomposition ablation.
- [ ] Qiskit QAOA comparison.
- [ ] Robustness and statistical evaluation.
- [ ] Final 100-user scalability results.
- [ ] Revised manuscript figures, tables, and claims.
