"""Quick diagnostic: verify ADMM matches SA after fixes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run import run_single

# 5 UEs — easy instance where all UEs should go to BS
res = run_single(num_ues=5, seed=42, sa_reads=100, sa_sweeps=500,
                 admm_max_iter=30, admm_restarts=3)

print(f"Instance: {res['num_ues']} UEs, {res['num_nodes']} nodes, "
      f"{res['num_vars']} vars, level_step={res['level_step']:.4f}")
print(f"Capacity: {res['capacity']}")

for name in ["random", "greedy", "greedy_aoii", "sa", "admm"]:
    s = res["solvers"][name]
    print(f"\n{name:12s}: worst={s['worst_aoii']:7.4f}  avg={s['avg_aoii']:7.4f}  "
          f"delivery={s['delivery_ratio']:.2f}  fresh={s['fresh_ratio']:.2f}  "
          f"cap_ok={s['capacity_ok']}  t={s['timing_s']:.3f}s")
    print(f"{'':12s}  assoc={s['association']}")
    if "energy" in s:
        print(f"{'':12s}  energy={s['energy']:.1f}")

# Key check: ADMM worst_aoii should be <= SA worst_aoii
sa_w = res["solvers"]["sa"]["worst_aoii"]
admm_w = res["solvers"]["admm"]["worst_aoii"]
print(f"\n{'='*50}")
print(f"SA   worst_aoii = {sa_w:.4f}")
print(f"ADMM worst_aoii = {admm_w:.4f}")
print(f"ADMM <= SA: {admm_w <= sa_w + 0.01}")
