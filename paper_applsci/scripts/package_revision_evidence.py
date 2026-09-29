"""Package scripts, control-flow inputs, and numeric records, never model weights."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper_applsci"


def main():
    paths = set((ROOT / "lara_align").rglob("*.py"))
    paths.update((ROOT / "scripts").glob("*.py"))
    paths.update((PAPER / "scripts").glob("*.py"))
    paths.update((PAPER / "data").glob("revision_*_metrics.csv"))
    for name in ["reference_corpus.tar.gz", "artifacts.json", "evidence.json", "metrics.csv",
                 "revision_analysis.json", "reviewer_experiments.json",
                 "reviewer_experiment_records.json.gz", "reviewer_inputs.pkl.gz", "sepsis_source.json"]:
        paths.add(PAPER / "data" / name)
    paths.update([PAPER / "notes/reviewer-experiments.md", PAPER / "notes/revision-validation.md",
                  PAPER / "README.md", PAPER / "Makefile", ROOT / "README.md", ROOT / "pyproject.toml",
                  ROOT / "tests/test_revision_experiments.py"])
    paths.update(ROOT.glob("LICENSE*"))
    # A concrete file allowlist keeps checkpoint directories out of the artifact.
    assert all(p.is_file() and "runs" not in p.relative_to(ROOT).parts for p in paths)
    assert all(p.suffix not in {".pt", ".pth", ".safetensors", ".ckpt"} for p in paths)
    target = PAPER / "build/revision_evidence.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
    print(f"Packaged {len(paths)} files without checkpoint weights: {target}")


if __name__ == "__main__":
    main()
