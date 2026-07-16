import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
JAX_ROOTS = {
    SRC / "tools" / "convert_jax_to_pytorch.py",
    *(SRC / "tools" / "jax_conversion").glob("*.py"),
}


def imported_roots(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_jax_is_confined_to_checkpoint_conversion():
    offenders = []
    for path in SRC.rglob("*.py"):
        if path in JAX_ROOTS:
            continue
        forbidden = imported_roots(path) & {"jax", "flax", "orbax", "optax"}
        if forbidden:
            offenders.append((path.relative_to(ROOT), sorted(forbidden)))
    assert not offenders, offenders


def test_stage3_uses_canonical_loss_module():
    source = (SRC / "post_train.py").read_text(encoding="utf-8")
    assert "from models.losses import PostTrainingLoss" in source
    assert not (SRC / "models" / "loss.py").exists()


def test_nocfg_reconstruction_emits_one_batch():
    source = (SRC / "scripts" / "generate_eval_images.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "mode == 'our_rfid_nocfg'" in ast.unparse(node.test):
            appends = [
                child
                for statement in node.body
                for child in ast.walk(statement)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "append"
                and ast.unparse(child.func.value) == "local_output_images"
            ]
            assert len(appends) == 1
            return
    raise AssertionError("our_rfid_nocfg branch not found")


if __name__ == "__main__":
    test_jax_is_confined_to_checkpoint_conversion()
    test_stage3_uses_canonical_loss_module()
    test_nocfg_reconstruction_emits_one_batch()
    print("runtime contract ok")
