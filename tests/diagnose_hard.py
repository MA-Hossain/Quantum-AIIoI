"""Test on capacity-scarce instance where greedy should fail."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run import run_single

# 10 UEs, 4 nodes, cap=3 each → total cap=12, barely enough.
# Some UEs MUST use satellite links.
for n_ue in [5, 10, 15]:
    res = run_single(
        num_ues=n_ue, seed=42,
        capacity_per_node=3,  # Tight: 4*3=12 total
        sa_reads=200, sa_sweeps=1000,
        admm_max_iter=50, admm_restarts=3,
    )

    print(f"\n{'='*60}")
    print(f"{n_ue} UEs, {res['num_nodes']} nodes, {res['num_vars']} vars, "
          f"cap=3, step={res['level_step']:.4f}")

    for name in ["random", "greedy", "greedy_aoii", "sa", "admm"]:
        s = res["solvers"][name]
        bs = sum(1 for m in s["association"].values() if m < 2)
        sat = n_ue - bs
        e_str = f"  E={s.get('energy', 0):8.1f}" if "energy" in s else ""
        print(f"  {name:12s}: worst={s['worst_aoii']:8.4f}  avg={s['avg_aoii']:7.4f}  "
              f"BS/SAT={bs}/{sat}  deliv={s['delivery_ratio']:.2f}  "
              f"cap_ok={s['capacity_ok']}  t={s['timing_s']:.3f}s{e_str}")
