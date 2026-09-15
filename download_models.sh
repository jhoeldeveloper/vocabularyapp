#!/bin/bash
# Downloads the Kokoro-82M ONNX model + voices bundle into models/onnx/.
#
#   - model_q8f16.onnx : quantized ONNX export from
#       onnx-community/Kokoro-82M-v1.0-ONNX  (default, best CPU/size trade-off)
#   - voices-v1.0.bin  : bundled per-voice style vectors from the kokoro-onnx
#       project release (the HF repo only ships per-voice .bin files, which
#       kokoro-onnx cannot load directly).
#
# Default is fp32 (model.onnx): on the i5-6200U (no VNNI) it is both faster and
# higher quality than the int8-quantized exports. Set MODEL_VARIANT=q8f16 (or
# fp16) to grab a quantized model instead for a low-RAM A/B comparison.
#
# Honors HF_TOKEN from the environment for authenticated HuggingFace downloads
# (avoids anonymous rate limits). Leave it unset to download anonymously.

set -e

APP_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$APP_DIR"

MODELS_DIR="$APP_DIR/models/onnx"
mkdir -p "$MODELS_DIR"

VARIANT="${MODEL_VARIANT:-fp32}"
case "$VARIANT" in
  q8f16) MODEL_FILE="model_q8f16.onnx" ;;
  fp16)  MODEL_FILE="model_fp16.onnx" ;;
  fp32)  MODEL_FILE="model.onnx" ;;
  *)     echo "Unknown MODEL_VARIANT=$VARIANT (use q8f16|fp16|fp32)" >&2; exit 1 ;;
esac

MODEL_URL="https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/${MODEL_FILE}"
VOICES_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin"

# Build optional auth header for HuggingFace (GitHub release is public).
AUTH_HEADER=()
if [ -n "$HF_TOKEN" ]; then
  AUTH_HEADER=(-H "Authorization: Bearer $HF_TOKEN")
  echo "Using HF_TOKEN for authenticated download."
fi

download() {
  local url="$1" local_path="$2"
  if [ -s "$local_path" ]; then
    echo "✓ already present: $local_path (skipping)"
    return
  fi
  echo "↓ downloading $url"
  curl -L "${AUTH_HEADER[@]}" -o "$local_path" "$url"
  if [ ! -s "$local_path" ]; then
    echo "✗ failed to download $url" >&2
    exit 1
  fi
  echo "✓ saved $local_path"
}

download "$MODEL_URL"  "$MODELS_DIR/$MODEL_FILE"
download "$VOICES_URL" "$MODELS_DIR/voices-v1.0.bin"

# If the user asked for a non-default model, point KOKORO_MODEL at it via .env
# (so the app picks it up without further flags). Only write if not already set.
if [ "$VARIANT" != "q8f16" ] && ! grep -q "KOKORO_MODEL" "$APP_DIR/.env" 2>/dev/null; then
  echo "KOKORO_MODEL=$MODELS_DIR/$MODEL_FILE" >> "$APP_DIR/.env"
  echo "→ wrote KOKORO_MODEL to .env (override; delete the line to use q8f16)"
fi

echo "Done. Models are in $MODELS_DIR"
