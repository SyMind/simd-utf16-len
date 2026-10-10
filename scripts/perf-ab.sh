#!/usr/bin/env bash
# Compare `utf16_len` in the working tree against a base ref.
#
# Usage: scripts/perf-ab.sh [base-ref] [--fail-above <percent>] [--json <path>] [--runs <n>] [--rounds <n>]
#
# The base ref defaults to `main`. Pass `HEAD` to measure the no-change spread.
#
# Each side gets its own binary, built by the same harness from the same
# directory, so identical code sits at identical addresses in both. With both
# sides in one binary, the copy that landed in the worse spot for the branch
# predictors measured 10 to 15% slower on some inputs with no code change.
set -euo pipefail

root=$(git rev-parse --show-toplevel)
base_ref=${1:-main}
if [ $# -gt 0 ]; then
  shift
fi

exe=simd-utf16-len-ab
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) exe=$exe.exe; align_log2=12 ;;
  *) align_log2=14 ;;
esac

# Start every function on a 16 KB boundary and every loop on a 64-byte one,
# so a change in one function moves nothing else and the two binaries differ
# only inside the changed functions. Microsoft's linker rejects sections
# aligned beyond 4 KB. An unchanged kernel loop measured 5 to 12% slower on
# Zen 3 after a change elsewhere in its function moved it off a 64-byte
# boundary.
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C llvm-args=-align-all-functions=$align_log2 -C llvm-args=-align-loops=64"
# One codegen unit per crate keeps functions in source order in both binaries.
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1

tree="$root/target/ab-tree"

# build_side <base|head> <tree-ish>: builds the harness against the crate in
# that tree, into target/ab-build-<side>. Both sides use the working tree's
# harness and inputs.
build_side() {
  rm -rf "$tree"
  mkdir -p "$tree/perf/ab"
  # Keep LF line endings, which Git on Windows would otherwise convert to CRLF.
  # `-m` dates the files now instead of at their commit, so cargo rebuilds a
  # base that is older than the previous build in target/ab-build-<side>.
  git -C "$root" -c core.autocrlf=false archive "$2" | tar -x -m -C "$tree"
  rm -rf "$tree/perf/ab"
  mkdir -p "$tree/perf/ab"
  cp "$root/perf/ab/Cargo.toml" "$tree/perf/ab/"
  cp -R "$root/perf/ab/src" "$tree/perf/ab/src"
  # And the working tree's inputs, so both sides measure the same list.
  cp "$root/benches/inputs.rs" "$tree/benches/inputs.rs"
  local features=""
  if [ "$1" = head ] && [ -n "${AB_HEAD_FEATURES:-}" ]; then
    # Crate features for the head side only, so an opt-in feature can be
    # measured against a base that doesn't have it.
    features="simd-utf16-len/$AB_HEAD_FEATURES"
  fi
  cargo build --release --quiet --manifest-path "$tree/perf/ab/Cargo.toml" --target-dir "$root/target/ab-build-$1" ${features:+--features "$features"}
}

# The working tree as a tree object, so head builds from the same path as base.
index="$root/target/ab-index"
mkdir -p "$root/target"
cp "$(git -C "$root" rev-parse --absolute-git-dir)/index" "$index"
GIT_INDEX_FILE=$index git -C "$root" add --all -- .
head_tree=$(GIT_INDEX_FILE=$index git -C "$root" write-tree)
rm -f "$index"

build_side base "$base_ref"
build_side head "$head_tree"

AB_BASE_LABEL=$(git -C "$root" rev-parse --short "$base_ref^{commit}")
AB_HEAD_LABEL=$(git -C "$root" rev-parse --short "${AB_HEAD_REF:-HEAD}^{commit}")
if ! git -C "$root" diff --quiet HEAD --; then
  AB_HEAD_LABEL="$AB_HEAD_LABEL with local changes"
fi
if [ -n "${AB_HEAD_FEATURES:-}" ]; then
  AB_HEAD_LABEL="$AB_HEAD_LABEL with $AB_HEAD_FEATURES"
fi
export AB_BASE_LABEL AB_HEAD_LABEL

"$root/target/ab-build-head/release/$exe" --base-exe "$root/target/ab-build-base/release/$exe" "$@"
