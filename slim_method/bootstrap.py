import importlib
import os
import sys
from pathlib import Path


SLIM_ROOT = Path(__file__).resolve().parents[1]
VENDORED_AGENT_SYSTEM = SLIM_ROOT / "agent_system"
VENDORED_GIGPO = SLIM_ROOT / "gigpo"
EXTERNAL_REPO_NAMES = tuple(name for name in os.environ.get("SLIM_EXTERNAL_REPO_NAMES", "").split(":") if name)


def _dedupe(paths):
    seen = set()
    ordered = []
    for path in paths:
        if not path:
            continue
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        ordered.append(norm)
    return ordered


def configure_imports():
    slim_root = str(SLIM_ROOT)
    repo_root = os.environ.get("REPO_ROOT")
    use_vendored_agent_system = VENDORED_AGENT_SYSTEM.is_dir()
    use_vendored_gigpo = VENDORED_GIGPO.is_dir()
    current_paths = [path for path in sys.path if path]
    retained = []
    for path in current_paths:
        abspath = os.path.abspath(path)
        if abspath == slim_root:
            continue
        if (use_vendored_agent_system or use_vendored_gigpo) and any(
            abspath.endswith(f"/{name}") or f"/{name}/" in abspath for name in EXTERNAL_REPO_NAMES
        ):
            continue
        retained.append(path)

    prioritized = [slim_root]
    if repo_root and not use_vendored_agent_system:
        prioritized.append(repo_root)
        alfworld_env = os.path.join(repo_root, "agent_system", "environments", "env_package", "alfworld")
        if os.path.isdir(alfworld_env):
            prioritized.append(alfworld_env)

    sys.path[:] = _dedupe([*prioritized, *retained])

    pythonpath_entries = [slim_root]
    if repo_root and not use_vendored_agent_system:
        pythonpath_entries.append(repo_root)
        alfworld_env = os.path.join(repo_root, "agent_system", "environments", "env_package", "alfworld")
        if os.path.isdir(alfworld_env):
            pythonpath_entries.append(alfworld_env)

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_entries.extend(existing_pythonpath.split(os.pathsep))

    os.environ["PYTHONPATH"] = os.pathsep.join(_dedupe(pythonpath_entries))
    return slim_root


def report_import_sources(prefix):
    try:
        verl = importlib.import_module("verl")
        print(f"{prefix} verl={Path(verl.__file__).resolve()}")
    except Exception as exc:
        print(f"{prefix} failed to import verl: {exc}")

    try:
        agent_system = importlib.import_module("agent_system")
        agent_file = getattr(agent_system, "__file__", None)
        if agent_file is not None:
            location = Path(agent_file).resolve()
        else:
            location = list(getattr(agent_system, "__path__", []))
        print(f"{prefix} agent_system={location}")
    except Exception as exc:
        print(f"{prefix} failed to import agent_system: {exc}")

    try:
        gigpo = importlib.import_module("gigpo")
        print(f"{prefix} gigpo={Path(gigpo.__file__).resolve()}")
    except Exception as exc:
        print(f"{prefix} failed to import gigpo: {exc}")
