deepspeed --num_gpus 2 --bind_cores_to_rank deepspeed_train.py --deepspeed $@