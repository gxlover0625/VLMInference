CUDA_VISIBLE_DEVICES=0 python3 ../utils/batch_infer.py \
--model_name "InternVL2-8B" \
--model_path "/root/autodl-tmp/llm/OpenGVLab/InternVL2-8B" \
--batch_size 8 \
--input_file "/root/autodl-tmp/pub/VLMInference/test.jsonl" \
--output_file "/root/autodl-tmp/codes/test/otuput.jsonl"