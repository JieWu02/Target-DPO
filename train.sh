#!/bin/bash
ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
--num_processes 8 \
--num_machines 1 \
--machine_rank 0 \
--main_process_ip 127.0.0.1 \
--main_process_port 29501 \
scripts/run_target_dpo.py recipes/code_dpo/config.yaml