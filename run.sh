#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- Active config: RankMixer NS tokenizer + sparse MoE token fusion ----
python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --rank_mixer_mode sparse_moe \
    --moe_num_experts 4 \
    --moe_top_k 2 \
    --d_model 72 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --num_workers 8 \
    "$@"

# ---- Alternative config: GroupNSTokenizer driven by ns_groups.json ----
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# sparse_moe does not require d_model % T == 0. If switching back to
# --rank_mixer_mode full with calendar-time NS tokens enabled, num_ns=14
# and T = num_queries*4 + 14, so choose d_model divisible by T.
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --rank_mixer_mode sparse_moe \
#     --d_model 72 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
