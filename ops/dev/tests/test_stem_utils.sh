echo "PYTHONPATH: $PYTHONPATH"
export DEBUG=True

export OPENAI_BASE_URL="http://pool0-02778:8000/v1"
export OPENAI_API_KEY="dummy_key"
export OPENAI_MODEL="/lustre/fs1/portfolios/nvr/projects/nvr_lpr_agentic/users/jianh/data/models/Qwen3-30B-A3B-Instruct-2507-FP8"
export OPENAI_TEMPERATURE="0"
export OPENAI_TOP_P="1.0"
export OPENAI_TOP_K="-1"
export OPENAI_MAX_TOKENS="1024"
export OPENAI_MAX_MODEL_LEN="9216"

for i in {0..0}; do
    for j in {0..64}; do
        echo "Running test $i $j"
        python scripts/tests/test_stem_utils.py --instance_id $j &> ./test_stem_utils_log/test_stem_utils_${i}_${j}.log &
    done
    wait
done

# Wait for all background tasks to complete
wait

echo "All tests completed"
