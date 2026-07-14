# Desk-eval ONLY: re-aggregate existing round61 per-dataset RF-blend results
# under various gates. gate-D fires the RF blend on a subset; because round61
# computed base & blend independently per dataset, a gate is an exact post-hoc
# subset selection -> no re-fit needed, numbers are exact. Reads round61 CSV,
# writes nothing outside this dir, never touches submissions/.
import csv, os
r61 = os.path.join(os.path.dirname(__file__), "..", "round61_rf_blend", "results.csv")
rows = list(csv.DictReader(open(r61)))
for r in rows:
    r["n_train"] = int(r["n_train"]); r["n_object_cols"] = int(r["n_object_cols"])
    r["dpub"] = float(r["delta_public"]); r["dpriv"] = float(r["delta_private"])

def report(name, fire):
    fired = [r for r in rows if fire(r)]
    n = len(rows)
    sp = sum(r["dpub"] for r in fired); spr = sum(r["dpriv"] for r in fired)
    # regressions among FIRED datasets only (non-fired contribute exactly 0)
    reg_pub = [r["dataset"] for r in fired if r["dpub"] < -1e-9]
    reg_priv = [r["dataset"] for r in fired if r["dpriv"] < -1e-9]
    ds = ",".join(r["dataset"] for r in fired)
    print(f"--- {name} ---")
    print(f"  fired={len(fired)} [{ds}]")
    print(f"  over-16 mean dPublic ={sp/n:+.6f}   mean dPrivate={spr/n:+.6f}")
    if fired:
        print(f"  fired-subset mean dPublic ={sp/len(fired):+.6f}   dPrivate={spr/len(fired):+.6f}")
    print(f"  regressions among fired: Public={reg_pub or 'NONE'}  Private={reg_priv or 'NONE'}")
    clean = (not reg_pub and not reg_priv and sp>0 and spr>0)
    print(f"  CLEAN-WIN (zero reg both splits, net+ both): {'YES' if clean else 'NO'}")
    print()

report("gate-D  (n_train<2000 AND obj>0)", lambda r: r["n_train"]<2000 and r["n_object_cols"]>0)
report("gate-D' (n_train<5000 AND obj>0)", lambda r: r["n_train"]<5000 and r["n_object_cols"]>0)
report("gate-D''(n_train<4000 AND obj>0)", lambda r: r["n_train"]<4000 and r["n_object_cols"]>0)
report("gate-C  (obj>0, all sizes) [reference]", lambda r: r["n_object_cols"]>0)

# separation diagnostic: sort obj>0 datasets by n_train, show both-split sign
print("=== obj>0 datasets by n_train (both-split clean-win = W) ===")
objrows = sorted([r for r in rows if r["n_object_cols"]>0], key=lambda r: r["n_train"])
for r in objrows:
    w = "W" if (r["dpub"]>1e-9 and r["dpriv"]>1e-9) else ("L" if (r["dpub"]<-1e-9 or r["dpriv"]<-1e-9) else "T")
    print(f"  {r['dataset']:9s} n={r['n_train']:6d} obj={r['n_object_cols']:2d}  dPub={r['dpub']:+.5f} dPriv={r['dpriv']:+.5f}  [{w}]")
