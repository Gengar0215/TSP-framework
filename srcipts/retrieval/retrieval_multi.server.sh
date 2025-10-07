export CUDA_VISIBLE_DEVICES=""


multi_index_dir=
retriever=

python retrieval_multi_server.py \
    --multi_index_dir $multi_index_dir \
    --topk 3 \
    --retriever_model $retriever \
    --faiss_gpu \
    --num_gpus 2