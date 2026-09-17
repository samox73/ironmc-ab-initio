# VSC5 setup

From the local ironmc checkout, upload the repository and dataset, install Nushell/Rust/fd/rg under the VSC storage directories, and prebuild the workspace:

```nu
nu vsc5/install-on-vsc5.nu
```

Then log in and dispatch jobs through the ironmc module:

```nu
ssh samuel_soap@vsc5.vsc.ac.at
nu -c 'use ($env.DATA)/ironmc/nu/ironmc; ironmc slurm submit { ironmc ladder run 20 --node } --devel --out-dir runs/ladder/n20'
```

Monitor and inspect jobs:

```nu
squeue --me
seff JOB_ID
```

Every submission needs `--out-dir runs/<experiment>` (required). Each gets its own job dir with `job.sh`, `job-<job-id>.out`, and (for non-run targets) `result-<job-id>.json` under `$DATA/ironmc/<out-dir>/jobs/<stamp>-<name>/`, with the run's cumulative dumps beside it in `<out-dir>/results/`. (`$DATA/ironmc/slurm/` is the read-only legacy tree from before `--out-dir` was mandatory.)
