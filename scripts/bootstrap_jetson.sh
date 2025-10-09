#!/usr/bin/env bash
set -euo pipefail

# Idempotent setup for KataGo inside NVIDIA PyTorch Jetson container
# - Ensures deps
# - Builds/repairs /katago-bin/katago linking to container libs (libzip.so.5)
# - Ensures model and default cfg
# - Nano-safe defaults unless cfg already exists

KATAGO_BIN_DIR="/katago-bin"
KATAGO_EXE="$KATAGO_BIN_DIR/katago"
KATAGO_CFG="$KATAGO_BIN_DIR/default_gtp.cfg"
KATAGO_MODEL_BIG="$KATAGO_BIN_DIR/kata1-b28c512nbt-s11084575488-d5365903487.bin.gz"
KATAGO_MODEL_SMALL="$KATAGO_BIN_DIR/kata1-b10c128.bin.gz"

ensure_deps() {
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git cmake g++ zlib1g-dev libzip-dev libboost-all-dev curl ca-certificates >/dev/null
}

build_katago() {
  rm -rf /tmp/KataGo
  git clone --depth=1 https://github.com/lightvector/KataGo.git /tmp/KataGo >/dev/null 2>&1
  cd /tmp/KataGo/cpp
  rm -rf build && mkdir -p build && cd build
  cmake .. -DUSE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release >/dev/null
  make -j"$(nproc)" >/dev/null
  install -m 0755 ./katago "$KATAGO_EXE"
}

has_libzip_mismatch() {
  if [ ! -x "$KATAGO_EXE" ]; then return 0; fi
  if ldd "$KATAGO_EXE" | grep -q 'libzip.so.4 .* not found'; then return 0; fi
  return 1
}

ensure_model_and_cfg() {
  mkdir -p "$KATAGO_BIN_DIR"
  # Prefer large model; fall back to small if download fails
  if [ ! -f "$KATAGO_MODEL_BIG" ]; then
    curl -fsSL -o "$KATAGO_MODEL_BIG" \
      "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-s11084575488-d5365903487.bin.gz" \
      || true
  fi
  if [ ! -f "$KATAGO_MODEL_BIG" ] && [ ! -f "$KATAGO_MODEL_SMALL" ]; then
    curl -fsSL -o "$KATAGO_MODEL_SMALL" \
      "https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b10c128-s4387285504-d1220457265.bin.gz"
  fi

  # Create cfg if missing (nano-safe defaults)
  if [ ! -f "$KATAGO_CFG" ]; then
    # Try to copy example from source if present
    if [ -f /tmp/KataGo/cpp/configs/gtp_example.cfg ]; then
      cp /tmp/KataGo/cpp/configs/gtp_example.cfg "$KATAGO_CFG"
    else
      # Minimal cfg
      cat >"$KATAGO_CFG" <<EOF
logAllGTPCommunication = true
rules = tromp-taylor
allowResignation = true
resignThreshold = -0.90
resignConsecTurns = 3
numSearchThreads = 2
cudaUseFP16 = false
cudaGraphMaxBatchSize = 0
cudaGraphWorkspaceMB = 0
EOF
    fi
  fi

  # Point modelFile to whichever exists
  MODEL_PATH="$KATAGO_MODEL_BIG"
  if [ ! -f "$MODEL_PATH" ] && [ -f "$KATAGO_MODEL_SMALL" ]; then
    MODEL_PATH="$KATAGO_MODEL_SMALL"
  fi
  if grep -q '^modelFile' "$KATAGO_CFG"; then
    sed -i "s|^modelFile *=.*|modelFile = $MODEL_PATH|" "$KATAGO_CFG"
  else
    printf "\nmodelFile = %s\n" "$MODEL_PATH" >> "$KATAGO_CFG"
  fi

  # Remove playout cap if present; prefer visits
  sed -i '/^maxPlayouts[[:space:]]*=/d' "$KATAGO_CFG" || true
}

sanity_or_fix_symlink() {
  set +e
  echo name | "$KATAGO_EXE" gtp -model "$(grep -m1 '^modelFile' "$KATAGO_CFG" | awk -F'= ' '{print $2}')" -config "$KATAGO_CFG" >/dev/null 2>&1
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    # As last resort, provide libzip.so.4 symlink if binary still requests it
    if ldd "$KATAGO_EXE" | grep -q 'libzip.so.4 .* not found' && [ -f /usr/lib/aarch64-linux-gnu/libzip.so.5 ]; then
      ln -sf /usr/lib/aarch64-linux-gnu/libzip.so.5 /usr/lib/aarch64-linux-gnu/libzip.so.4 || true
    fi
  fi
}

main() {
  ensure_deps
  mkdir -p "$KATAGO_BIN_DIR"

  if [ ! -x "$KATAGO_EXE" ] || has_libzip_mismatch; then
    build_katago
  fi

  ensure_model_and_cfg
  sanity_or_fix_symlink

  # Final sanity
  echo name | "$KATAGO_EXE" gtp -model "$(grep -m1 '^modelFile' "$KATAGO_CFG" | awk -F'= ' '{print $2}')" -config "$KATAGO_CFG" || true
  echo "[bootstrap] KataGo ready: $KATAGO_EXE, cfg=$KATAGO_CFG"
}

main "$@"


