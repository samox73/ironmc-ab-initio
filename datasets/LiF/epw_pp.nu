#!/usr/bin/env nu

const root = path self .
cd $root

if (["save", "out/save"] | any { |path| $path | path exists }) {
    error make { msg: "save output already exists; remove or rename it first" }
}

cp LiF.dyn* out/
cd out
"LiF" | ^python3 (which epw_pp.py | get path.0)
if $env.LAST_EXIT_CODE != 0 { exit $env.LAST_EXIT_CODE }
mv save ../save
