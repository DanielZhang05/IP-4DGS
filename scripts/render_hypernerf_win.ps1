# ===================== exp setup =====================
$env:centers_num = "3"
$env:ONLY_EVAL   = "t"

$clip_feat_dim  = 3
$video_feat_dim = 6
$dataset_name   = "americano"

# (可选) 避免中文/emoji 输出乱码
# chcp 65001 | Out-Null

# ===================== time-agnostic language field =====================
$env:language_feature_hiddendim = "$clip_feat_dim"

$buildDir = "submodules\4d-langsplat-rasterization\build"

$env:use_discrete_lang_f = "f"

foreach ($level in 1,2,3) {
  foreach ($mode in "lang","rgb") {
    python render.py `
      -s "data/hypernerf/$dataset_name" `
      --model_path "output/hypernerf/$dataset_name/${dataset_name}_$level" `
      --skip_train `
      --skip_test `
      --configs "arguments/hypernerf/default.py" `
      --mode $mode `
      --no_dlang 1 `
      --load_stage "fine-lang"
  }
}

# ===================== time-sensitive language field =====================
$env:language_feature_hiddendim = "$video_feat_dim"

if (Test-Path $buildDir) { Remove-Item $buildDir -Recurse -Force }

python -m pip install --no-cache-dir --no-build-isolation -e "submodules\4d-langsplat-rasterization"

$env:use_discrete_lang_f = "t"

foreach ($level in 0) {
  foreach ($mode in "lang","rgb") {
    python render.py `
      -s "data/hypernerf/$dataset_name" `
      --model_path "output/hypernerf/$dataset_name/${dataset_name}_$level" `
      --skip_train `
      --skip_test `
      --configs "arguments/hypernerf/default.py" `
      --mode $mode `
      --no_dlang 0 `
      --load_stage "fine-lang-discrete"
  }
}