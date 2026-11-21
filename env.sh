# source this: every cache and virtualenv stays on the external disk
export AESBENCH_ROOT=/Volumes/Media/traning
export HF_HOME=$AESBENCH_ROOT/cache/hf TORCH_HOME=$AESBENCH_ROOT/cache/torch UV_CACHE_DIR=$AESBENCH_ROOT/cache/uv
export UV_PROJECT_ENVIRONMENT=/Volumes/Media/aesbench/.venv PYTORCH_ENABLE_MPS_FALLBACK=1
