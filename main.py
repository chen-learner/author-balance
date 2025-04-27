import torch.distributed
import datetime
from model.deepseek_v2.modeling_deepseek import (
    DeepseekV2ForCausalLM,
    DeepseekV2Config,
    DeepseekV2Model,
)
from model.deepseek_v2.configuration_deepseek import mpu
from model.deepseek_v2.tokenization_deepseek_fast import DeepseekTokenizerFast
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import io
import sys
import re
import json
import pandas as pd
from datetime import timedelta
import os
import threading
import time
from dataclasses import dataclass
import torch.distributed as dist
from typing import Dict, List, Optional, Union, Callable

DEVICE_PLATFORM = os.environ.get("DEVICE_PLATFORM", "gpu")

    # import moxing as mox

if DEVICE_PLATFORM == "gpu":
    try:
        import flash_attn
    except ImportError:
        flash_attn = None


def initialize_dist():
    skip_msg = ""
    if not torch.distributed.is_initialized():
        """Initialize torch.distributed and execute the user function."""
        init_method = "tcp://"
        master_ip = os.getenv("MASTER_ADDR", "localhost")
        master_port = os.getenv("MASTER_PORT", "6000")
        init_method += master_ip + ":" + master_port
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=mpu.world_size,
            rank=mpu.global_rank,
            timeout=timedelta(minutes=10),
        )

        if torch.cuda.is_available():
            torch.cuda.set_device(mpu.local_rank)


def load_model(rank):
    time1 = time.time()
    for i in model.model.layers[1:]:
        # i.mlp.experts.cuda(rank)
        for param in i.mlp.experts.parameters():
            param.data.cuda()
    time2 = time.time()


def create_shared_cpu_model(model):
    for param in model.parameters():
        param.data = param.data.pin_memory()
        # param.data.share_memory_()
    return model


def parse_slurm_tasks_per_node(s):
    res = []
    for part in s.split(","):
        m = re.match(r"^([0-9]+)(\(x([0-9]+)\))?$", part)
        if m:
            tasks = int(m.group(1))
            repetitions = m.group(3)
            if repetitions is None:
                repetitions = 1
            else:
                repetitions = int(repetitions)
            if repetitions > 1000:
                raise RuntimeError("task list repetitions too large")
            for i in range(repetitions):
                res.append(tasks)
        else:
            raise RuntimeError("bad task list syntax")

    pivot = res[0]

    assert len(res) > 0, "Unexpected empty node configurations"
    assert all(elem == pivot for elem in res), "Non-symmetric gpus are not supported"

    return pivot


# torch.cuda.set_device("cuda:3")
if os.environ.get("RANK", None) is not None:
    mpu.global_rank = int(os.environ["RANK"])
    mpu.world_size = int(os.environ["WORLD_SIZE"])
    mpu.local_rank = int(os.environ["LOCAL_RANK"])
    mpu.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
elif os.environ.get("SLURM_JOBID", None) is not None:
    mpu.global_rank = int(os.environ["SLURM_PROCID"])
    mpu.world_size = int(os.environ["SLURM_NTASKS"])
    mpu.local_world_size = parse_slurm_tasks_per_node(
        os.environ["SLURM_TASKS_PER_NODE"]
    )
    mpu.local_rank = mpu.global_rank % mpu.local_world_size
else:
    assert False, "Invalid distributed arguments"

initialize_dist()

# torch.set_default_tensor_type("torch.cuda.FloatTensor")
model_name = "deepseek_v2"
torch.set_default_dtype(torch.bfloat16)
config = json.load(open(f"config/{model_name}/config.json", "r"))
# Initialize configuration
config = DeepseekV2Config(**config)
tokenizer = AutoTokenizer.from_pretrained(f"config/{model_name}")
# Initialize model with the configuration
model = DeepseekV2ForCausalLM(config)
model.generation_config = GenerationConfig.from_pretrained(f"config/{model_name}")
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.eval()

torch.random.seed = 1234
if torch.distributed.get_rank() == 0:
    print(model)

input_tensor = torch.randint(0, 10086, [1, 1024])
config.balance = True
torch.cuda.empty_cache()
a = torch.tensor([1, 2, 3]).cuda()
torch.distributed.barrier()

n_run_times = 2
prof_start = 9
prof_end = 2
token_dist = config.token_dist
time_ = 0
for run_id in range(n_run_times):
    for i in range(3, 4):
        input_tensor = torch.randint(0, 10086, [pow(2, i), 2048])
        print(f"The shape of input: {input_tensor.shape}")
        for j in range(token_dist+1):
            config.token_dist = j
            if run_id >= prof_start and run_id < prof_end:
                prof_dir = f"./profile/test_{datetime.datetime.now().strftime('%Y-%m-%d-%H')}/{config.balance}_{datetime.datetime.now().strftime('%M')}/{i}"
                dist.broadcast_object_list([prof_dir], 0)
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    with_stack=False,
                    record_shapes=True,
                    profile_memory=True,
                    schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, skip_first=0),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        prof_dir,
                        worker_name=f"rank_{torch.distributed.get_rank()}"
                    ),
                ) as prof:
                    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=4)
                    prof.step()
            else:
                time1 = time.time()
                outputs = model.generate(input_tensor.to(model.device), max_new_tokens=19)
                time2 = time.time()
                if torch.distributed.get_rank() == 0:
                    print(f"End to end time {input_tensor.size()} - {j} - {(time2-time1)*1000}")
                
            # Reload experts from CPU, so that we can do the next prefill.
            model.reload_moe_expert_group()
            torch.cuda.empty_cache()



