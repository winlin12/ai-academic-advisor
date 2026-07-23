#!/usr/bin/env bash
# pull_moe_models.sh — pulls the MoE test bracket for Winston's two machines
#
# Usage:
#   ./pull_moe_models.sh server   # 2060 Super box (8GB VRAM + 32GB RAM envelope)
#   ./pull_moe_models.sh big      # 5070 Ti box (16GB VRAM + 64GB DDR5)
#   ./pull_moe_models.sh all      # everything (only sensible on the 5070 Ti box)

set -euo pipefail

SERVER_MODELS=(
  "qwen3.6:35b-a3b"    # current quality leader (~21GB Q4)
  "gpt-oss:20b"        # lighter MoE, reasoning-trained (~14GB MXFP4)
  "qwen3-coder:30b"    # coder-specialized MoE, SQL task alignment (~19GB Q4)
  "glm-4.7-flash"
  "laguna-xs-2.1"
  "gemma4:26b"
  "nemotron-cascade-2"
)

BIG_MODELS=(
  "qwen3.5:122b-a10b"  # next quality tier, needs ~64GB RAM
  "qwen3-next:80b-a3b" # fits in gpu, but needs a lot of ram
  "llama4:scout"       # may not fit in 64 gb ram
  "gpt-oss:120b"       # OpenAI large MoE, ~5.1B active
)

usage() {
  grep '^#   ' "$0" | sed 's/^#   //'
  exit 1
}

check_free_space() {
  # Ollama stores models under ~/.ollama by default; warn if low on space
  local needed_gb=$1
  local ollama_dir="${OLLAMA_MODELS:-$HOME/.ollama}"
  local avail_gb
  avail_gb=$(df -BG --output=avail "$ollama_dir" 2>/dev/null | tail -1 | tr -dc '0-9' || echo "")
  if [[ -n "$avail_gb" && "$avail_gb" -lt "$needed_gb" ]]; then
    echo "WARNING: ~${avail_gb}GB free at $ollama_dir, this set needs roughly ${needed_gb}GB."
    read -rp "Continue anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
  fi
}

pull_models() {
  local -n models=$1
  local failed=()
  for m in "${models[@]}"; do
    echo ""
    echo "==> Pulling $m"
    if ! ollama pull "$m"; then
      echo "!!  Failed to pull $m (tag may have changed — check ollama.com/library)"
      failed+=("$m")
    fi
  done
  echo ""
  echo "=== Done. Installed models: ==="
  ollama list
  if [[ ${#failed[@]} -gt 0 ]]; then
    echo ""
    echo "Failed pulls (verify tags on ollama.com/library):"
    printf '  %s\n' "${failed[@]}"
  fi
}

command -v ollama >/dev/null || { echo "ollama not found in PATH"; exit 1; }

case "${1:-}" in
  server)
    check_free_space 60
    pull_models SERVER_MODELS
    ;;
  big)
    check_free_space 160
    pull_models BIG_MODELS
    ;;
  all)
    check_free_space 220
    pull_models SERVER_MODELS
    pull_models BIG_MODELS
    ;;
  *)
    usage
    ;;
esac