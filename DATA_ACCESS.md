# Asset access

The release does not redistribute PartNet-Mobility URDFs or meshes. The official SAPIEN
site requires registration and acceptance of the PartNet-Mobility terms before access:
https://sapien.ucsd.edu/downloads and https://sapien.ucsd.edu/about.

After approval, download each ID in `manifests/object_manifest.csv` into a directory named
`partnet-mobility-dataset/<object_id>/`. Verify each `mobility.urdf` against the recorded
SHA-256 before a physics rerun. SAPIEN also documents token-based per-model download through
`sapien.asset.download_partnet_mobility(object_id)`.

The private phone video is not included. Only the derived four-frame manual 2D annotation
table is archived; it is not needed for any simulation result.
