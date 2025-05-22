# greedy decoding
# bash run.sh run.exp.llama.eagle.eagle_sd_8b run-grid-search 1 4,8,16,32,48,64 6 --max-samples 20
# sleep 10
# bash run.sh run.exp.llama.classic.classic_sd_1b_8b run-grid-search 1 4,8,16,32,48,64 6 --max-samples 20
# sleep 10
# bash run.sh run.exp.llama.subspec.subspec_sd_8gb run-grid-search 0.2 4,8,16,32,48,64 6 --max-samples 20
# sleep 10

# Test Tdraft under temperature sampling
# bash run.sh run.exp.qwen.subspec.subspec_sd_8gb run-grid-search 0.6,0.8,1.0,1.2,1.4 48 6 --max-samples 20
# bash run.sh run.exp.qwen.subspec.subspec_sd_24gb run-benchmark --benchmarks cnn-dm,mt-bench,human-eval,gsm8k,alpaca --max-samples 20
# sleep 10

#NEW with temp sampling
# bash run.sh run.exp.qwen.classic.classic_sd_1pt5b_7b run-grid-search 1 4,8,16,32,48,64 6 --max-samples 20
# sleep 10
# bash run.sh run.exp.qwen.subspec.subspec_sd_8gb run-grid-search 0.6 4,8,16,32,48,64 6 --max-samples 20
# sleep 10
# bash run.sh run.exp.qwen.eagle.eagle_sd_7b run-grid-search 0.6,1 10 6 --max-samples 20
# sleep 10
# bash run.sh run.offload.naive run-benchmark --benchmarks cnn-dm,mt-bench,human-eval,gsm8k,alpaca --max-samples 5
# sleep 10

# ablation studies
# bash run.sh run.share_layer_sd run-benchmark --benchmarks mt-bench,gsm8k --max-samples 20
# bash run.sh run.share_layer_sd run-benchmark --benchmarks mt-bench,gsm8k --max-samples 20
# bash run.sh run.share_layer_sd run-benchmark --benchmarks mt-bench,gsm8k --max-samples 20
# bash run.sh run.share_layer_sd run-benchmark --benchmarks mt-bench,gsm8k --max-samples 20
# bash run.sh run.share_layer_sd run-benchmark --benchmarks mt-bench,gsm8k --max-samples 20


# bash run.sh run.offload.naive run-benchmark --benchmarks gsm8k --max-samples 20
# bash run.sh run.exp.qwen.classic.classic_sd_1pt5b_7b run-benchmark --benchmarks gsm8k --max-samples 20
# bash run.sh run.exp.qwen.eagle.eagle_sd_7b run-benchmark --benchmarks gsm8k --max-samples 20
# sleep 10
# bash run.sh run.exp.qwen.classic.classic_sd_1pt5b_7b run-benchmark --benchmarks gsm8k --max-samples 20
# sleep 10
# bash run.sh run.exp.qwen.subspec.subspec_sd_8gb run-benchmark --benchmarks cnn-dm,mt-bench,human-eval,gsm8k,alpaca --max-samples 20
# sleep 10

# N offload
# bash run.sh run.exp.llama.eagle.eagle_sd_8b run-benchmark --benchmarks mt-bench --max-samples 20
# sleep 10
# bash run.sh run.exp.llama.classic.classic_sd_1b_8b run-benchmark --benchmarks mt-bench --max-samples 20
# sleep 10
# bash run.sh run.exp.llama.subspec.subspec_sd_8gb run-benchmark --benchmarks mt-bench --max-samples 20
# sleep 10
# bash run.sh run.offload.naive run-benchmark --benchmarks mt-bench --max-samples 5
# sleep 10


# bash run.sh run.exp.qwen.subspec.subspec_sd_24gb run-benchmark --benchmarks cnn-dm,mt-bench,human-eval,gsm8k,alpaca --max-samples 20
# sleep 10
# bash run.sh run.exp.qwen.subspec.subspec_sd_12gb run-benchmark --benchmarks cnn-dm,mt-bench,human-eval,gsm8k,alpaca --max-samples 20
# sleep 10
# bash run.sh run.exp.llama.subspec.subspec_sd_8gb run-benchmark --benchmarks cnn-dm,mt-bench,human-eval,gsm8k,alpaca --max-samples 20
# sleep 10
# bash run.sh run.exp.qwen.subspec.subspec_sd_8gb run-benchmark --benchmarks mt-bench,gsm8k --max-samples 20
# sleep 10


# bash run.sh run.exp.qwen.subspec.subspec_sd_24gb run-benchmark --benchmarks cnn-dm,mt-bench,human-eval,gsm8k,alpaca --max-samples 20
# sleep 10
# bash run.sh run.exp.qwen.classic.classic_sd_1pt5b_32b run-benchmark --benchmarks cnn-dm,mt-bench,human-eval,gsm8k,alpaca --max-samples 20
# sleep 10
# bash run.sh run.exp.qwen.classic.classic_sd_7b_32b run-benchmark --benchmarks cnn-dm,mt-bench,human-eval,gsm8k,alpaca --max-samples 20
# sleep 10


# Remeber to test no offload subspec after lunch!

# pure offload (n prefetch)
# bash run.sh run.exp.qwen.subspec.subspec_sd_8gb run-benchmark --benchmarks mt-bench --max-samples 20
# sleep 10

# bash run.sh run.offload.naive run-benchmark --benchmarks mt-bench --max-samples 5
# sleep 10

# bash run.sh run.exp_offload.subspec_sd_qwen_14b run-benchmark --benchmarks aime --max-samples 20
bash run.sh run.exp_offload.subspec_sd_qwen_7b run-benchmark --benchmarks mt-bench --max-samples 20
# bash run.sh run.exp_offload.reasoning.subspec_sd_qwq_32b run-benchmark --benchmarks aime,gpqa,math-500,livecodebench --max-samples 20
# bash run.sh run.exp_offload.subspec_sd_llama_8b run-benchmark --benchmarks aime,gpqa,math-500,livecodebench --max-samples 20

# bash run.sh run.exp_offload.reasoning.vanilla_dqwen_14b run-benchmark --benchmarks aime,gpqa,math-500,livecodebench --max-samples 5
# bash run.sh run.exp_offload.reasoning.vanilla_dqwen_7b run-benchmark --benchmarks aime,gpqa,math-500,livecodebench --max-samples 5
# bash run.sh run.exp_offload.reasoning.vanilla_dllama_8b run-benchmark --benchmarks aime,gpqa,math-500,livecodebench --max-samples 5
# bash run.sh run.exp_offload.reasoning.vanilla_dqwen_32b run-benchmark --benchmarks aime,gpqa,math-500,livecodebench --max-samples 5
# bash run.sh run.exp_offload.reasoning.vanilla_qwq_32b run-benchmark --benchmarks aime,gpqa,math-500,livecodebench --max-samples 5