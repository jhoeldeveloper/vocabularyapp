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

# Load environment variables from .env
if [ -f "$APP_DIR/.env" ]; then
  set -a
  source "$APP_DIR/.env"
  set +a
fi

# Alternative-engine sources (used by the `alt` mode below).
INFLECT_BASE="https://huggingface.co/owensong/Inflect-Micro-v2-ONNX/resolve/main"
POCKET_GGUF_BASE="https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/PocketTTS-GGUF/english"
POCKET_VOICES="alba anna azelma bill_boerst caro_davy charles cosette eponine estelle eve fantine george giovanni jane javert jean juergen lola marius mary michael paul peter_yearsley rafael stuart_bell vera"

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

# ---------------------------------------------------------------------------
# Alternative engines:  ./download_models.sh alt
#
# Kokoro (above) stays the default and the most expressive. The other two are
# here for speed and for voice choice; both land in models/alt/ and are picked
# in the UI (Voice picker) or with TTS_ENGINE in .env.
#
#   inflect  37.7 MB ONNX  ~2x faster than Kokoro on the i5-6200U (RTF 0.28 vs
#                          0.85), one flat voice. Apache-2.0.
#   pocket   128 MB GGUF   26 English voices + cloning, but RTF 0.88 here --
#                          the smallest download and the slowest of the three.
#                          Needs the audio.cpp CPU binary, fetched here.
#
# Nothing here is needed to run the app: a missing engine shows up in the
# picker with the reason instead of breaking startup.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "alt" ]; then
  ALT_DIR="$APP_DIR/models/alt"
  mkdir -p "$ALT_DIR/inflect/onnx" "$ALT_DIR/pocket/embeddings"

  download "$INFLECT_BASE/onnx/duration.onnx" "$ALT_DIR/inflect/onnx/duration.onnx"
  download "$INFLECT_BASE/onnx/decode.onnx"   "$ALT_DIR/inflect/onnx/decode.onnx"

  AUDIO_CPP_VER="${AUDIO_CPP_VER:-v0.9.0}"
  if [ ! -x "$ALT_DIR/audiocpp_cli" ]; then
    echo "↓ downloading audio.cpp $AUDIO_CPP_VER (CPU, linux x64)"
    curl -L -o /tmp/audio.cpp.tar.gz \
      "https://github.com/0xShug0/audio.cpp/releases/download/${AUDIO_CPP_VER}/audio-${AUDIO_CPP_VER}-bin-ubuntu-x64-cpu.tar.gz"
    tar xzf /tmp/audio.cpp.tar.gz -C "$ALT_DIR" audiocpp_cli model_specs
    chmod +x "$ALT_DIR/audiocpp_cli"
    rm -f /tmp/audio.cpp.tar.gz
  else
    echo "✓ already present: audiocpp_cli (skipping)"
  fi

  download "$POCKET_GGUF_BASE/pocket-tts-english-q8_0.gguf" \
           "$ALT_DIR/pocket/pocket-tts-english-q8_0.gguf"
  for voice in $POCKET_VOICES; do
    download "$POCKET_GGUF_BASE/embeddings/${voice}.safetensors" \
             "$ALT_DIR/pocket/embeddings/${voice}.safetensors"
  done

  echo "Done. Alternative engines are in $ALT_DIR"
  exit 0
fi

download "$MODEL_URL"  "$MODELS_DIR/$MODEL_FILE"
download "$VOICES_URL" "$MODELS_DIR/voices-v1.0.bin"

# Point KOKORO_MODEL at a non-default variant. fp32 needs no line at all -- the
# app already defaults to models/onnx/model.onnx -- so nothing is written for it.
# The value is written RELATIVE to the project: the app resolves relative model
# paths against the project root (tts/base.py:resolve_path), not against the
# working directory, so the checkout stays movable and an absolute path here
# would quietly stop being true the day the folder is renamed or moved.
if [ "$VARIANT" != "fp32" ] && ! grep -q "^KOKORO_MODEL=" "$APP_DIR/.env" 2>/dev/null; then
  echo "KOKORO_MODEL=models/onnx/$MODEL_FILE" >> "$APP_DIR/.env"
  echo "→ wrote KOKORO_MODEL=models/onnx/$MODEL_FILE to .env (delete the line for fp32)"
fi

echo "Done. Models are in $MODELS_DIR"
