#!/usr/bin/env sh
# Checks the built image before it becomes :v1.
#
# What used to be here: `docker run ... review --mr 1 || true`. The reviewer started,
# died on a missing .ai-review.yml, and `|| true` threw the result away. A check that
# cannot fail is not a check - build_image was green while the image it pushed failed
# on startup.
#
# Everything here is about the artifact, not about the sources. The sources are
# covered by unit tests, and those run on the same commit in the `test` job. What a
# unit test cannot see is exactly what breaks during assembly: whether pip really
# installed the package, whether package data came with it, whether gitleaks landed
# in the image, whether the entrypoint is wired.
#
# Usage:
#   image-selfcheck.sh <image>
#
# The container runtime can be replaced for testing:
#   DOCKER_CMD=./my-stub image-selfcheck.sh <image>
set -eu

image=${1:-}
docker=${DOCKER_CMD:-docker}

if [ -z "$image" ]; then
    echo "usage: image-selfcheck.sh <image>" >&2
    exit 2
fi

step() { printf '\n== %s\n' "$1"; }
fail() { printf 'image self-check: FAILED - %s\n' "$1" >&2; exit 1; }

step "gitleaks is in the image"
# gitleaks is pulled from GitHub with curl at build time. A broken download or a moved
# release does not fail `docker build` loudly, and after that the deterministic half of
# the review is silently gone.
"$docker" run --rm --entrypoint gitleaks "$image" version || fail "gitleaks does not run"

step "entrypoint and imports"
# --help brings up argparse, and with it the whole module: httpx, openai, every
# reviewer.* import. A broken import shows here instead of in somebody else's
# pipeline a day later.
"$docker" run --rm "$image" --help >/dev/null || fail "python -m reviewer --help did not run"

step "exit code contract: a technical failure is 2"
# /workspace is empty, so there is no .ai-review.yml and the review cannot happen. By
# contract that is exit code 2, not 1: 1 means "there are blocking findings". If the
# startup path ever starts returning 1, the consuming pipeline reads a missing config
# as findings - and a retry rule keyed on 2 stops firing where it should.
set +e
out=$("$docker" run --rm "$image" review --mr 1 2>&1)
code=$?
set -e
printf '%s\n' "$out"
[ "$code" -eq 2 ] || fail "expected exit code 2 (technical failure), got $code"
printf '%s\n' "$out" | grep -q 'AI review: итог' \
    || fail "no outcome line in the output - nothing tells a reader which class of failure this is"

step "installed package: package data and behaviour"
"$docker" run --rm -i --entrypoint python "$image" - <<'PY'
import pathlib
import tempfile

import reviewer
from reviewer.prompt_builder import SYSTEM_TEMPLATE
from reviewer.tools import RepoTools

# The installed package, specifically. If this imported a checkout instead, the check
# would be talking about the repository rather than the artifact - that is, about
# nothing.
installed = pathlib.Path(reviewer.__file__).resolve().parent
if "site-packages" not in str(installed):
    raise SystemExit(f"imported package is not the installed one: {installed}")

# reviewer/prompts/*.md reach the wheel through tool.setuptools.package-data. A mistake
# in that setting does not break the build: the package installs and the prompt is
# gone, and it only fails in a consuming repository on the first review.
if not SYSTEM_TEMPLATE.strip():
    raise SystemExit("prompts/system.md is empty or did not make it into the installed package")

with tempfile.TemporaryDirectory() as tmp:
    root = pathlib.Path(tmp)
    (root / "internal" / "service").mkdir(parents=True)
    (root / "internal" / "service" / "mail_test.go").write_text(
        "func TestSend(t *testing.T) {}\n", encoding="utf-8"
    )
    hits = RepoTools(root=root).grep("func Test", "*_test.go")

if len(hits) != 1:
    raise SystemExit(f"a bare mask did not reach a subdirectory: {hits}")

print(f"installed at {installed}, prompt {len(SYSTEM_TEMPLATE)} chars, grep found {len(hits)}")
PY

printf '\nimage self-check: image is fit to push\n'
