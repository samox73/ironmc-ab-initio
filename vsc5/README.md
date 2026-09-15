# VSC5 setup

From the local ironmc checkout, upload the repository and dataset, install Nushell/Rust/fd/rg under the VSC storage directories, and prebuild the workspace:

```nu
nu vsc5/install-on-vsc5.nu
```

Then log in and dispatch jobs through the ironmc module:

```nu
ssh samuel_soap@vsc5.vsc.ac.at
nu -c 'use ($env.DATA)/ironmc/nu/ironmc; ironmc slurm submit { ironmc ladder run 20 --node } --devel'
```

Monitor and inspect jobs:

```nu
squeue --me
seff JOB_ID
```

Each submission gets its own job dir with `job.sh`, `job-<job-id>.out`, and (for non-run targets) `result-<job-id>.json`: `$DATA/ironmc/<out-dir>/jobs/<stamp>-<name>/` when submitting with `--out-dir runs/...` (the run's cumulative dumps sit beside it in `<out-dir>/results/`), else the legacy flat `$DATA/ironmc/slurm/<stamp>-<name>/`.
