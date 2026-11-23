#!/usr/bin/env bash
# macOS (Apple Silicon / Metal) yardimci uretim betigi.
# Windows betiklerine ve sabit kalite ayarlarina (6 adim, cfg 1.0, 640x384 alt siniri, parcasiz VAE, tohum 42) dokunmaz.
#
# Iki mod:
#   TAM_METAL=0  Guvenli: --offload-to-cpu --backend te=cpu, chunk_budget=0.03
#   TAM_METAL=1  Tum agirliklar MTL0 GPU'da (--auto-fit on --flash-attn on), chunk_budget=0
# TAM_METAL verilmezse bellege gore secilir: 48 GiB ve ustu -> 1, altinda -> 0.
#
# Kullanim:
#   scripts/mac_uret.sh <istem.txt> [kare=56] [genislik=640] [yukseklik=384] [ilk_kare.png]
#   TAM_METAL=1 scripts/mac_uret.sh <istem.txt> 209
#   EXTRA_ARGS='--disable-segmented-compute' scripts/mac_uret.sh <istem.txt> 56
#
# Ortam degiskenleri (hepsi istege bagli):
#   VT_ROOT, VT_ENGINE_BIN, VT_MODEL_DIR   yollar (varsayilan: betigin ust klasoru, engine_mac/bin, model)
#   TAM_METAL        0 veya 1 (varsayilan: bellege gore otomatik)
#   H3_CHUNK_BUDGET  dikkat bellek butcesi (TAM_METAL=0 icin 0.03, TAM_METAL=1 icin 0)
#   SEED=42  FPS=24  THREADS=8 (yalniz TAM_METAL=0)
#   EXTRA_ARGS       sd-cli'ye eklenecek ek bayraklar (deney icin; kaliteyi degistirebilir, sorumluluk sende)
#
# macOS'un varsayilan bash'i 3.2'dir; bu betik onunla uyumlu yazildi.

set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "Kullanim: $0 <istem.txt> [kare=56] [genislik=640] [yukseklik=384] [ilk_kare.png]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ENGINE_BIN="${VT_ENGINE_BIN:-$ROOT/engine_mac/bin}"
MODEL_DIR="${VT_MODEL_DIR:-$ROOT/model}"
SD_CLI="$ENGINE_BIN/sd-cli"
TIME_BIN="${VT_TIME_BIN:-/usr/bin/time}"

PROMPT_FILE="$1"
FRAMES="${2:-56}"
WIDTH="${3:-640}"
HEIGHT="${4:-384}"
INIT_IMG="${5:-}"

SEED="${SEED:-42}"
FPS="${FPS:-24}"
THREADS="${THREADS:-8}"

hata() { echo "HATA: $*" >&2; exit 1; }

# --- Girdi denetimleri (hepsi uzun bir kosu baslamadan once) ---
for v in FRAMES WIDTH HEIGHT; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || hata "$v sayi olmali: '${!v}'"
done

# H3 kare izgarasi: 17k+5 (56, 73, ... 192, 209), Windows uret.ps1 ile ayni kural.
if (( FRAMES < 22 || (FRAMES - 5) % 17 != 0 )); then
  hata "Kare sayisi 17k+5 izgarasinda olmali (ornek: 56, 73, 90, 107, 124, 141, 158, 175, 192, 209). Verilen: $FRAMES"
fi
if (( WIDTH % 32 != 0 || HEIGHT % 32 != 0 )); then
  hata "Genislik ($WIDTH) ve yukseklik ($HEIGHT) 32'nin kati olmalidir."
fi
if (( WIDTH * HEIGHT < 640 * 384 )); then
  hata "Cozunurluk alani 640x384 altina dusurulemez."
fi

[[ -x "$SD_CLI" ]] || hata "sd-cli bulunamadi veya calistirilabilir degil: $SD_CLI"

if [[ ! -f "$PROMPT_FILE" ]]; then
  if [[ -f "$ROOT/$PROMPT_FILE" ]]; then
    PROMPT_FILE="$ROOT/$PROMPT_FILE"
  else
    hata "Istem dosyasi bulunamadi: $PROMPT_FILE"
  fi
fi
if [[ -n "$INIT_IMG" && ! -f "$INIT_IMG" ]]; then
  hata "Baslangic gorseli bulunamadi: $INIT_IMG"
fi

DIT="$MODEL_DIR/minimax_h3_fl2va_hyperflow_flash_14b-Q4_K_M.gguf"
LLM="$MODEL_DIR/qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
VAE="$MODEL_DIR/minimax_h3_sol_vae_fp16.safetensors"
AVAE="$MODEL_DIR/minimax_h3_audio_vae_fp32.safetensors"
for f in "$DIT" "$LLM" "$VAE" "$AVAE"; do
  [[ -f "$f" ]] || hata "Model dosyasi yok: $f"
done

# --- Mod secimi ---
MEM_GIB=0
if MEM_BYTES="$(sysctl -n hw.memsize 2>/dev/null)" && [[ "$MEM_BYTES" =~ ^[0-9]+$ ]]; then
  MEM_GIB=$(( MEM_BYTES / 1073741824 ))
fi
if [[ -z "${TAM_METAL:-}" ]]; then
  if (( MEM_GIB >= 48 )); then TAM_METAL=1; else TAM_METAL=0; fi
  MOD_NOTU="otomatik (${MEM_GIB} GiB bellek)"
else
  MOD_NOTU="elle"
fi
[[ "$TAM_METAL" == "0" || "$TAM_METAL" == "1" ]] || hata "TAM_METAL 0 veya 1 olmali: '$TAM_METAL'"

if [[ -z "${H3_CHUNK_BUDGET:-}" ]]; then
  if [[ "$TAM_METAL" == "1" ]]; then H3_CHUNK_BUDGET="0"; else H3_CHUNK_BUDGET="0.03"; fi
fi

# TAM_METAL=1'de metin kodlayici (17 GB) GPU'da kalir; cok buyuk piksel x kare hesap tamponuna yetmeyebilir.
# Olculen: 1344x768x56 gecti, 1344x768x209 blocks.0'da bellek yetmedi (48 GB M5 Pro).
if [[ "$TAM_METAL" == "1" ]] && (( WIDTH * HEIGHT * FRAMES > 100000000 )); then
  echo "UYARI: ${WIDTH}x${HEIGHT}x${FRAMES} TAM_METAL=1'de bellege sigmayabilir (1344x768x209 olculen basarisiz ornek)." >&2
  echo "       'failed during weight preparation' gorursen TAM_METAL=0 ile dene." >&2
fi

# --- Cihaz dogrulamasi ---
echo "=== Cihaz Dogrulamasi (sd-cli --list-devices) ==="
DEVICES="$("$SD_CLI" --list-devices 2>&1)"
echo "$DEVICES"
if ! echo "$DEVICES" | grep -qiE "MTL0|metal"; then
  hata "Metal GPU (MTL0) gorunmuyor; CPU'ya dusmemek icin durduruldu. (Kisitli kabuklarda Metal gizlenir; Terminal.app'ten calistir.)"
fi

STEM="$(basename "$PROMPT_FILE" .txt)"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/${STAMP}_${STEM}_${FRAMES}k_${WIDTH}x${HEIGHT}"
LOG_FILE="$RUN_DIR/sd_cli.log"
OUT_VIDEO="$RUN_DIR/video.avi"
mkdir -p "$RUN_DIR"
rm -f "$OUT_VIDEO"

CMD=(
  "$SD_CLI" -M vid_gen
  --diffusion-model "$DIT"
  --llm "$LLM"
  --vae "$VAE"
  --audio-vae "$AVAE"
  --prompt-file "$PROMPT_FILE"
  -W "$WIDTH" -H "$HEIGHT" --video-frames "$FRAMES" --fps "$FPS"
  --steps 6 --sampling-method euler --sigmas hyperflow-h3-flash-6 --cfg-scale 1.0
  --rng cpu --seed "$SEED"
  --model-args "minimax_h3_attn_chunk_budget=$H3_CHUNK_BUDGET"
  --mmap
  -o "$OUT_VIDEO"
)

if [[ "$TAM_METAL" == "1" ]]; then
  CMD+=(--flash-attn on --auto-fit on)
else
  CMD+=(-t "$THREADS" --flash-attn off,diffusion=on --offload-to-cpu --backend te=cpu)
fi
if [[ -n "$INIT_IMG" ]]; then
  CMD+=(--init-img "$INIT_IMG")
fi

EXTRA=()
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA <<< "$EXTRA_ARGS"
  echo "UYARI: EXTRA_ARGS kullaniliyor (${EXTRA_ARGS}); kalite kapisi ve baytlik karsilastirma gecersiz sayilmali." >&2
fi

echo "Mod: TAM_METAL=$TAM_METAL [$MOD_NOTU] | ${WIDTH}x${HEIGHT} | Kare: $FRAMES | FPS: $FPS | Seed: $SEED | ChunkBudget: $H3_CHUNK_BUDGET"
echo "Log: $LOG_FILE"

# set -e kullanilmiyor: sd-cli hata verirse log'u gosterip anlamli cikmak istiyoruz.
"$TIME_BIN" -l "${CMD[@]}" ${EXTRA[@]+"${EXTRA[@]}"} > "$LOG_FILE" 2>&1
RC=$?

if (( RC != 0 )); then
  echo "HATA: sd-cli cikis kodu $RC. Log'un son satirlari:" >&2
  tail -n 15 "$LOG_FILE" | tr '\r' '\n' >&2
  exit "$RC"
fi

# --- Cikti dogrulamasi ---
SORUN=0
if grep -qiE "retrying with temporal tiling" "$LOG_FILE"; then
  echo "GECERSIZ CIKTI: Logda 'retrying with temporal tiling' goruldu (VAE parcali calisti)." >&2; SORUN=1
fi
if grep -qE "ALLOC_FAILED|failed during weight preparation|^\[E\]" "$LOG_FILE"; then
  echo "GECERSIZ CIKTI: Logda bellek/hata satiri var:" >&2
  grep -aE "ALLOC_FAILED|failed during weight preparation|^\[E\]" "$LOG_FILE" | head -n 5 >&2; SORUN=1
fi
if [[ ! -s "$OUT_VIDEO" ]]; then
  echo "GECERSIZ CIKTI: Video dosyasi yok veya bos: $OUT_VIDEO" >&2; SORUN=1
fi

if [[ -s "$OUT_VIDEO" ]]; then
  if command -v ffprobe >/dev/null 2>&1; then
    NKARE="$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$OUT_VIDEO" 2>/dev/null | head -n 1)"
    SES="$(ffprobe -v error -select_streams a -show_entries stream=codec_type -of csv=p=0 "$OUT_VIDEO" 2>/dev/null | head -n 1)"
    if [[ "$NKARE" != "$FRAMES" ]]; then
      echo "GECERSIZ CIKTI: Kare sayisi $NKARE, beklenen $FRAMES." >&2; SORUN=1
    fi
    if [[ -z "$SES" ]]; then
      echo "GECERSIZ CIKTI: Videoda ses akisi yok." >&2; SORUN=1
    fi
  else
    echo "BILGI: ffprobe yok; kare sayisi ve ses kontrolu atlandi (brew install ffmpeg)."
  fi
fi

(( SORUN == 0 )) || exit 1

echo "=== TAMAM ==="
echo "Video: $OUT_VIDEO"
grep -aiE "get_learned_condition completed|sampling completed|decode_first_stage completed|audio VAE decode completed|generate_video completed" "$LOG_FILE" | sed -E 's/ --- .*//' || true
SON_ADIM="$(tr '\r' '\n' < "$LOG_FILE" | grep -aoE '[0-9.]+s/it' | tail -n 1)"
[[ -n "$SON_ADIM" ]] && echo "Son adim suresi: $SON_ADIM"
grep -aE " real |maximum resident set size|peak memory footprint" "$LOG_FILE" || true
