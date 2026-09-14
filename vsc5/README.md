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

Batch scripts, logs, and result JSONs land in `$DATA/ironmc/slurm/`, named `<job-name>-<job-id>.{sh,out,json}`.
