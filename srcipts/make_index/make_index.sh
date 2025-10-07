export GITHUB_TOKEN=""

python -m swebench.inference.make_datasets.build_retrieval_mt \
    --dataset_name_or_path "" \
    --output_dir "" \
    --splits test \
    --leave_indexes False \
    --document_encoding_style file_name_and_ast_parsed \
    --num_workers 1