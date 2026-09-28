#!/bin/sh
# gpu-stress installer.
#
#   curl -fsSL https://raw.githubusercontent.com/HamzaGbada/gpu-stress-test/main/install.sh | sh
#
# By default this installs the zero-dependency lite pipeline, which gives you the
# `gpu-stress-lite` command and needs nothing but an NVIDIA driver. The PyTorch
# pipeline (`gpu-stress`) is an optional extra worth several GB of wheels:
#
#   curl -fsSL .../install.sh | sh -s -- --torch
#
# Note the `sh -s --`: options go to the shell running this script. Writing
#   GPU_STRESS_EXTRAS=torch curl ... | sh
# does NOT work, because the variable is set for `curl`, not for `sh`. To use the
# environment instead, put it on the shell: `curl ... | GPU_STRESS_EXTRAS=torch sh`.
#
# With uv, pipx or pip present the Python package is installed; otherwise a
# single-file zipapp is dropped into ~/.local/bin/gpu-stress-lite.
#
# Options (or the matching environment variable):
#   --extras LIST   GPU_STRESS_EXTRAS   comma list, e.g. torch,plot (default: none)
#   --torch                             shorthand for --extras torch,plot
#   --version REF   GPU_STRESS_VERSION  tag to install, or "main" (default: latest release)
#   --method M      GPU_STRESS_METHOD   auto | uv | pipx | pip | zipapp (default: auto)
#   --bin DIR       GPU_STRESS_BIN      install dir for the zipapp (default: ~/.local/bin)
#                   GPU_STRESS_REPO     owner/name of the GitHub repo
set -eu

REPO="${GPU_STRESS_REPO:-HamzaGbada/gpu-stress-test}"
VERSION="${GPU_STRESS_VERSION:-}"
METHOD="${GPU_STRESS_METHOD:-auto}"
EXTRAS="${GPU_STRESS_EXTRAS:-}"
BIN_DIR="${GPU_STRESS_BIN:-$HOME/.local/bin}"

say()  { printf '%s\n' "$*"; }
warn() { printf '!  %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

usage() {
    cat <<'USAGE'
gpu-stress installer

  curl -fsSL <url>/install.sh | sh                      # lite pipeline (no dependencies)
  curl -fsSL <url>/install.sh | sh -s -- --torch        # + the PyTorch pipeline

Options:
  --extras LIST    extras to install, e.g. torch,plot
  --torch          shorthand for --extras torch,plot
  --version REF    tag to install, or "main" (default: latest release)
  --method M       auto | uv | pipx | pip | zipapp (default: auto)
  --bin DIR        install directory for the zipapp (default: ~/.local/bin)
  -h, --help       show this help

Each option has a matching GPU_STRESS_* environment variable. When piping, set it
on the shell (`curl ... | GPU_STRESS_EXTRAS=torch sh`), not before curl.
USAGE
}

# Options are parsed after the environment, so a flag wins over a variable.
while [ $# -gt 0 ]; do
    case "$1" in
        --extras)   EXTRAS="${2:?--extras needs a value, e.g. --extras torch,plot}"; shift 2 ;;
        --extras=*) EXTRAS="${1#*=}"; shift ;;
        --torch)    EXTRAS="torch,plot"; shift ;;
        --version)   VERSION="${2:?--version needs a value, e.g. --version v0.3.2}"; shift 2 ;;
        --version=*) VERSION="${1#*=}"; shift ;;
        --method)   METHOD="${2:?--method needs a value}"; shift 2 ;;
        --method=*) METHOD="${1#*=}"; shift ;;
        --bin)      BIN_DIR="${2:?--bin needs a value}"; shift 2 ;;
        --bin=*)    BIN_DIR="${1#*=}"; shift ;;
        -h|--help)  usage; exit 0 ;;
        *)          usage >&2; die "unknown option: $1" ;;
    esac
done

# ---------------------------------------------------------------- environment
case "$(uname -s)" in
    Linux)  ;;
    *)      die "only Linux is supported (NVIDIA's CUDA driver library is Linux/Windows only)" ;;
esac

if ! have nvidia-smi && ! ls /usr/lib/libcuda.so.1 /usr/lib64/libcuda.so.1 \
        /usr/lib/x86_64-linux-gnu/libcuda.so.1 >/dev/null 2>&1; then
    warn "no NVIDIA driver found (no nvidia-smi, no libcuda.so.1) - installing anyway,"
    warn "but the tool needs a driver to run."
fi

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if have "$c" && "$c" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
        PY="$c"; break
    fi
done
[ -n "$PY" ] || die "python 3.10+ is required"

# Resolve the source reference for pip-style installs.
REF="${VERSION:-}"
if [ -z "$REF" ]; then
    if have curl; then
        REF=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null \
              | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1)
    elif have wget; then
        REF=$(wget -qO- "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null \
              | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n1)
    fi
    [ -n "$REF" ] || REF="main"
fi
# A bare VCS URL for the default install; a PEP 508 direct reference when extras
# are requested, which is the only form that carries them: gpu-stress[torch] @ git+...
if [ -n "$EXTRAS" ]; then
    SPEC="gpu-stress[$EXTRAS] @ git+https://github.com/$REPO@$REF"
else
    SPEC="git+https://github.com/$REPO@$REF"
fi

# ------------------------------------------------------------------- download
fetch() {  # fetch URL DEST
    if have curl; then curl -fsSL "$1" -o "$2"
    elif have wget; then wget -qO "$2" "$1"
    else die "need curl or wget to download"
    fi
}

install_zipapp() {
    if [ "${VERSION:-}" = "main" ]; then
        die "the zipapp is published per release; use GPU_STRESS_VERSION=<tag> or another method"
    fi
    if [ -n "$EXTRAS" ]; then
        warn "the zipapp cannot carry extras ($EXTRAS) - it is the lite pipeline only."
        warn "For the PyTorch pipeline, install with uv, pipx or pip instead."
    fi
    url="https://github.com/$REPO/releases/latest/download/gpu-stress.pyz"
    [ "$REF" = "main" ] || url="https://github.com/$REPO/releases/download/$REF/gpu-stress.pyz"
    mkdir -p "$BIN_DIR"
    tmp="$(mktemp)"
    say "downloading $url"
    fetch "$url" "$tmp" || die "download failed - is there a published release yet?"
    "$PY" -c 'import zipfile,sys; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)' "$tmp" \
        || { rm -f "$tmp"; die "downloaded file is not a valid zipapp"; }
    mv "$tmp" "$BIN_DIR/gpu-stress-lite"
    chmod 755 "$BIN_DIR/gpu-stress-lite"
    INSTALLED="$BIN_DIR/gpu-stress-lite"
}

# -------------------------------------------------------------------- install
INSTALLED=""
case "$METHOD" in
    auto)
        if   have uv;   then say "installing with uv from $REF";   uv tool install --force "$SPEC" && INSTALLED="gpu-stress-lite"
        elif have pipx; then say "installing with pipx from $REF"; pipx install --force "$SPEC" && INSTALLED="gpu-stress-lite"
        elif "$PY" -m pip --version >/dev/null 2>&1; then
            say "installing with pip --user from $REF"
            "$PY" -m pip install --user --upgrade "$SPEC" && INSTALLED="gpu-stress-lite"
        else
            say "no uv/pipx/pip found - falling back to the standalone zipapp"
            install_zipapp
        fi
        ;;
    uv)     have uv   || die "uv not found";   uv tool install --force "$SPEC"; INSTALLED="gpu-stress-lite" ;;
    pipx)   have pipx || die "pipx not found"; pipx install --force "$SPEC";    INSTALLED="gpu-stress-lite" ;;
    pip)    "$PY" -m pip install --user --upgrade "$SPEC"; INSTALLED="gpu-stress-lite" ;;
    zipapp) install_zipapp ;;
    *)      die "unknown GPU_STRESS_METHOD: $METHOD" ;;
esac

say ""
say "installed: $INSTALLED"
# Note: no bare `a && b` here - under `set -e` a false test would abort the
# script before the usage hints below ever print.
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        if [ -f "$BIN_DIR/gpu-stress-lite" ]; then
            warn "$BIN_DIR is not on your PATH; add it or run $BIN_DIR/gpu-stress-lite"
        fi
        ;;
esac
say ""
say "  gpu-stress-lite --list          # show the steps"
say "  gpu-stress-lite --burn 60       # full run, 60s per sustained load"
case "$EXTRAS" in
    *torch*)
        say "  gpu-stress --epochs 3           # PyTorch pipeline (installed)"
        ;;
    *)
        say ""
        say "The PyTorch pipeline (gpu-stress) is not installed: it pulls several GB of"
        say "wheels, so it is an opt-in extra. To add it, re-run with --torch:"
        say ""
        say "  curl -fsSL https://raw.githubusercontent.com/$REPO/main/install.sh | sh -s -- --torch"
        ;;
esac
