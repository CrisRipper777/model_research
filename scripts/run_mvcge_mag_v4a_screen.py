#!/usr/bin/env python3
"""Run V4A label-free preflight, one-epoch smoke, and frozen 36-run campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

BRANCH="exp/mvcge_mag_v4a_raw_anchored_residual_screen"
PARENT_SHA="a9853de789fd52297b717e79c4fdb13ede738e0b"
DATASETS=("Movies","Grocery","ele-fashion")
SEEDS=(42,43,44)
VARIANTS=("R0_raw","R1_global_residual","R2_expert_residual","R3_adaptive_residual")
OUTPUT_ROOT=ROOT/"outputs"/"mvcge_mag_v4a_raw_anchored_residual_screen"
RESEARCH_ROOT=ROOT/"research"/"mvcge_mag_v4a_raw_anchored_residual_screen"
DATA_ROOT=RESEARCH_ROOT/"data"
FREEZE_MESSAGE="Freeze MvCGE-MAG V4A raw-anchored adaptive role residual screen"
FORMAL_ARTIFACTS={
    "REPORT.md","data/campaign_manifest.json","data/run_rows.json","data/summary.csv",
    "data/paired_comparisons.csv","data/role_partition_diagnostics.csv",
    "data/decomposition_diagnostics.csv","data/role_contrast_diagnostics.csv",
    "data/action_space_novelty.csv","data/residual_gate_diagnostics.csv",
    "data/residual_profile_diagnostics.csv","data/routing_diagnostics.csv",
    "data/strength_diagnostics.csv","data/expert_profiles.csv","data/expert_similarity.csv",
    "data/historical_r0_regression.csv",
}

def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def git(*args): return subprocess.run(["git",*args],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
def write_json(path:Path,value:Any):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8"); tmp.replace(path)
def read_json(path): return json.loads(path.read_text(encoding="utf-8"))
def _dirty_paths(): return {line[3:].split(" -> ")[-1] for line in git("status","--porcelain","--untracked-files=all").splitlines()}
def _freeze_sha(): return git("log","--format=%H","-1",f"--grep=^{FREEZE_MESSAGE}$") or None

def provenance(mode:str,resume:bool=False):
    branch,head,dirty=git("branch","--show-current"),git("rev-parse","HEAD"),_dirty_paths()
    if branch!=BRANCH: raise RuntimeError(f"expected {BRANCH}, found {branch}")
    if git("merge-base",PARENT_SHA,"HEAD")!=PARENT_SHA: raise RuntimeError("branch ancestry does not include exact V3C parent")
    freeze=_freeze_sha()
    if mode=="campaign":
        if freeze is None or head!=freeze: raise RuntimeError(f"campaign requires HEAD at freeze SHA; head={head}, freeze={freeze}")
        if dirty and not resume: raise RuntimeError(f"campaign requires a clean worktree, found {sorted(dirty)}")
        if dirty and resume:
            allowed={f"research/mvcge_mag_v4a_raw_anchored_residual_screen/{p}" for p in FORMAL_ARTIFACTS}
            unexpected=dirty-allowed
            if unexpected: raise RuntimeError(f"resume has non-artifact modifications: {sorted(unexpected)}")
        pre=read_json(DATA_ROOT/"preflight_summary.json"); smoke=read_json(DATA_ROOT/"smoke_summary.json")
        if not pre.get("passed") or not smoke.get("audit_passed"): raise RuntimeError("campaign requires passing preflight and 4/4 smoke")
    return {"branch":branch,"parent_commit_sha":PARENT_SHA,"head_at_start":head,
            "freeze_commit_sha":freeze,"worktree_clean_at_start":not dirty,"started_at_utc":now()}

def environment(device:str):
    import torch
    return {"python":sys.version,"platform":sys.platform,"torch":torch.__version__,
        "cuda_available":torch.cuda.is_available(),"cuda_device_count":torch.cuda.device_count(),"device":device,
        "device_name":torch.cuda.get_device_name(torch.device(device)) if "cuda" in device else "CPU",
        "protocol":"unified_full_graph_nc_v1"}

def run_preflight(device:str):
    from scripts.analyze_mvcge_mag_v4a_screen import preflight,write_csv
    prov=provenance("preflight")
    rows,summary=preflight(device)
    write_csv(DATA_ROOT/"preflight_decomposition_diagnostics.csv",rows)
    write_csv(DATA_ROOT/"role_partition_preflight.csv",summary.pop("role_partition_rows"))
    summary["provenance"]=prov
    write_json(DATA_ROOT/"preflight_summary.json",summary)
    write_json(DATA_ROOT/"environment.json",environment(device))
    print(f"[V4A preflight] passed={summary['passed']} checks={summary['checks']}",flush=True)
    if not summary["passed"]: raise SystemExit(2)

def validate_run(metrics_path:Path,checkpoint:Path):
    if not metrics_path.is_file() or not checkpoint.is_file(): raise FileNotFoundError(f"missing run files {metrics_path} / {checkpoint}")
    payload=read_json(metrics_path)
    if len(payload.get("runs",[]))!=1: raise RuntimeError(f"expected one run in {metrics_path}")
    run=payload["runs"][0]; metrics=dict(run.get("metrics",{}))
    if any(str(k).lower().startswith("test") for k in metrics): raise RuntimeError(f"Test metrics found in {metrics_path}")
    for key in ("val_acc","val_macro_f1"):
        if key not in metrics or not math.isfinite(float(metrics[key])): raise FloatingPointError(f"invalid {key} in {metrics_path}")
    if int(run.get("metadata",{}).get("best_epoch",0))<1: raise RuntimeError("missing Validation Accuracy-selected epoch")
    return {"metrics":metrics,"metadata":dict(run["metadata"])}

def run_one(dataset,seed,variant,*,device,mode,epochs=None,resume=False):
    run_dir=OUTPUT_ROOT/mode/"runs"/dataset/f"seed_{seed}"/variant
    metrics_path=run_dir/"run_metrics.json"; checkpoint=run_dir/"best.pt"
    if resume and metrics_path.is_file() and checkpoint.is_file():
        return {"status":"reused","mode":mode,"dataset":dataset,"seed":seed,"variant":variant,**validate_run(metrics_path,checkpoint),
                "metrics_path":str(metrics_path.relative_to(ROOT)),"checkpoint_path":str(checkpoint.relative_to(ROOT))}
    run_dir.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-m","src.main",f"dataset={dataset}","task=nc","model=mvcge_mag_v4a",
        f"model.variant={variant}",f"seed={seed}","num_runs=1",f"device={device}","task.evaluate_test=false",
        f"task.run_metrics_path={metrics_path}",f"task.save_ckpt_path={checkpoint}",f"hydra.run.dir={run_dir/'hydra'}"]
    if epochs is not None: command += [f"task.epochs={epochs}","task.early_stop_min_epoch=1","task.patience=2","task.early_stop_min_delta=0.0"]
    log_path=run_dir/"training.log"
    print(f"[V4A {mode}] {dataset} seed={seed} {variant}",flush=True)
    with log_path.open("w",encoding="utf-8") as log:
        proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,env=os.environ.copy())
        assert proc.stdout is not None
        for line in proc.stdout: print(line,end="",flush=True); log.write(line)
        code=proc.wait()
    if code:
        raise RuntimeError(f"training exit={code}; log tail:\n{log_path.read_text(encoding='utf-8',errors='replace')[-6000:]}")
    return {"status":"completed","mode":mode,"dataset":dataset,"seed":seed,"variant":variant,**validate_run(metrics_path,checkpoint),
            "metrics_path":str(metrics_path.relative_to(ROOT)),"checkpoint_path":str(checkpoint.relative_to(ROOT))}

def run_smoke(device:str):
    from scripts.analyze_mvcge_mag_v4a_screen import audit_checkpoint,load_features_and_edges
    prov=provenance("smoke")
    pre=read_json(DATA_ROOT/"preflight_summary.json")
    if not pre.get("passed"): raise RuntimeError("smoke requires passing label-free preflight")
    results=[]; failures=[]
    for variant in VARIANTS:
        try:
            row=run_one("Movies",42,variant,device=device,mode="smoke",epochs=1,resume=False)
            audit=audit_checkpoint(row,device)
            if not audit["finite"] or not audit["role_partition_valid"] or not audit["decomposition_valid"] or not audit["top2_verified"]:
                raise RuntimeError("selected smoke checkpoint failed finite/identity/static top-2 audits")
            row["audit"]={k:v for k,v in audit.items() if not k.endswith("_rows")}
            row["diagnostics"]={"residual_gate_rows":audit["residual_gate_rows"],"residual_profile_rows":audit["residual_profile_rows"],
                                "routing_rows":audit["routing_rows"],"strength_rows":audit["strength_rows"]}
            results.append(row)
        except Exception as exc:
            msg=repr(exc); lower=msg.lower()
            failures.append({"dataset":"Movies","seed":42,"variant":variant,"error":msg,
                "oom":("out of memory" in lower or "cuda oom" in lower),"nonfinite":any(t in lower for t in ("nan","inf","non-finite"))})
        write_json(DATA_ROOT/"smoke_summary.json",{"protocol":"unified_full_graph_nc_v1","task_evaluate_test":False,
            "dataset":"Movies","seed":42,"epochs":1,"device":device,"expected_runs":4,"completed_runs":len(results),
            "checkpoint_count":sum((ROOT/r["checkpoint_path"]).is_file() for r in results),"failures":failures,"rows":results,
            "provenance":prov,"audit_passed":False})
        print(f"[V4A smoke] progress={len(results)}/4 failures={len(failures)}",flush=True)
    checks={"four_runs_complete":len(results)==4,"all_checkpoints_finite":len(results)==4 and all(r["audit"]["finite"] for r in results),
        "no_test_metrics":len(results)==4 and all(not r["audit"]["test_metrics_present"] for r in results),
        "task_evaluate_test_false":True,"identities_valid":len(results)==4 and all(r["audit"]["decomposition_valid"] and r["audit"]["role_partition_valid"] for r in results),
        "static_top2_verified":len(results)==4 and all(r["audit"]["top2_verified"] and r["audit"]["static_routing_verified"] for r in results),
        "no_oom_or_nonfinite_failures":not any(f["oom"] or f["nonfinite"] for f in failures)}
    summary={"protocol":"unified_full_graph_nc_v1","task_evaluate_test":False,"dataset":"Movies","seed":42,"epochs":1,
        "device":device,"expected_runs":4,"completed_runs":len(results),"checkpoint_count":sum((ROOT/r["checkpoint_path"]).is_file() for r in results),
        "failures":failures,"rows":results,"checks":checks,"audit_passed":len(results)==4 and not failures and all(checks.values()),"provenance":prov}
    write_json(DATA_ROOT/"smoke_summary.json",summary); write_json(DATA_ROOT/"environment.json",environment(device))
    print(f"[V4A smoke] {len(results)}/4 complete; audit_passed={summary['audit_passed']}",flush=True)
    if not summary["audit_passed"]: raise SystemExit(1)

def run_campaign(device:str,resume:bool):
    prov=provenance("campaign",resume)
    rows_path=DATA_ROOT/"run_rows.json"; manifest_path=DATA_ROOT/"campaign_manifest.json"
    rows=read_json(rows_path) if resume and rows_path.is_file() else []
    failures_path=OUTPUT_ROOT/"formal"/"failures.json"
    failures=read_json(failures_path) if resume and failures_path.is_file() else []
    completed={(r["dataset"],int(r["seed"]),r["variant"]) for r in rows if r.get("status") in {"completed","reused"}}
    for dataset in DATASETS:
        for seed in SEEDS:
            for variant in VARIANTS:
                key=(dataset,seed,variant)
                if key in completed: continue
                try:
                    row=run_one(dataset,seed,variant,device=device,mode="formal",resume=resume)
                    rows.append(row); completed.add(key)
                    for failure in failures:
                        if (failure.get("dataset"),int(failure.get("seed",-1)),failure.get("variant"))==key:
                            failure["resolved"]=True; failure["resolved_at_utc"]=now()
                except Exception as exc:
                    msg=repr(exc); lower=msg.lower()
                    failures.append({"dataset":dataset,"seed":seed,"variant":variant,"error":msg,
                        "oom":("out of memory" in lower or "cuda oom" in lower),
                        "nonfinite":any(t in lower for t in ("nan","inf","non-finite")),"resolved":False,"at_utc":now()})
                    print(f"[V4A failure] {failures[-1]}",flush=True)
                write_json(rows_path,rows)
                write_json(manifest_path,{"protocol":"unified_full_graph_nc_v1","task_evaluate_test":False,"expected_runs":36,
                    "completed_runs":len(completed),"device":device,"datasets":DATASETS,"seeds":SEEDS,"variants":VARIANTS,
                    "failures":failures,"provenance":prov,"updated_at_utc":now()})
                write_json(failures_path,failures)
                print(f"[V4A campaign] progress={len(completed)}/36",flush=True)
    unresolved=[f for f in failures if not f.get("resolved",False)]
    print(f"[V4A campaign] completed={len(completed)}/36 unresolved={len(unresolved)}",flush=True)
    if len(completed)!=36 or unresolved: raise SystemExit(1)
    subprocess.run([sys.executable,"scripts/analyze_mvcge_mag_v4a_screen.py","--device",device],cwd=ROOT,check=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--mode",choices=("preflight","smoke","campaign"),required=True)
    parser.add_argument("--device",default="cuda:0"); parser.add_argument("--resume",action="store_true"); args=parser.parse_args()
    if args.mode=="preflight":run_preflight(args.device)
    elif args.mode=="smoke":run_smoke(args.device)
    else:run_campaign(args.device,args.resume)

if __name__=="__main__": main()
