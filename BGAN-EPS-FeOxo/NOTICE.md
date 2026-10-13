# Attribution and provenance

The BGAN-EPS training, network architecture, internal-coordinate conversion, and
entropy routines derive from [rshin1209/bgan_eps](https://github.com/rshin1209/bgan_eps).
The original license is MIT, Copyright (c) 2022 Zhongyue John Yang. Its text is
preserved without alteration in `LICENSE`.

The Fe-oxo working code was supplied in this publication project. The copies in
`bgan_eps_feoxo/` and `bgan_eps_feoxo_tqacl/` were byte-identical for the four core
Python files before release packaging. This release distributes one core.

The exact original upstream Git commit was not recorded in the supplied local
snapshot. `provenance/source_manifest.json` records SHA-256 hashes of the local
upstream snapshot, modified working snapshot, and packaged core. It does not claim
that the local snapshot matches today's upstream branch.

`provenance/upstream_to_modified.patch` documents changes from that local original
snapshot to the modified research code. `provenance/packaging_changes.patch`
documents subsequent import, path, interface, and input-check changes for this
release. Entropy arithmetic and model/training mathematics were retained.

The independent validation and figure scripts were supplied with the Fe-oxo
project. They are described separately because they use system-specific atom maps
and input filenames. External libraries retain their own licenses and are
installed as dependencies; their source or binaries are not bundled here.
