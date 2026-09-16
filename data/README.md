# Local data layout

Mentor-supplied videos can live anywhere outside Git. The project tools create:

```text
data/working/    extracted JPGs and editable JSON annotations
data/generated/  leakage-safe YOLO dataset, manifest, and test truth
```

Both directories are ignored. To share labels within a student team, archive
them outside the repository or use an approved team storage location. Public
starter samples may be committed under `data/samples/`; private staff test
footage must never be committed.
