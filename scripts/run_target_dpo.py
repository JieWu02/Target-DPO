import logging
import random
import sys
import io
import os
import json
import torch
import wandb
import transformers
import time
from datetime import datetime
from transformers import AutoModelForCausalLM, set_seed

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    apply_chat_template,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from peft import PeftConfig, PeftModel
from customer_trainer import CustomDPOTrainer
from trl import DPOTrainer
from datasets import Dataset, DatasetDict

# 设置日志记录器
logger = logging.getLogger(__name__)

# 配置 Weights & Biases
wandb.login(key="your_wandb_key")  # 替换为你的W&B API密钥
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_ENTITY"] = "your_entity"  # 替换为你的实体名称
os.environ["WANDB_PROJECT"] = 'your_project'  # 替换为你的项目名称
# 设置运行名称，包含时间戳
os.environ["WANDB_RUN_NAME"] = 'target-dpo' + str(
    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M'))


def _make_w_io_base(f, mode: str):
    """创建或打开写入模式的文件对象"""
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    """创建或打开读取模式的文件对象"""
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """从JSON文件加载数据到字典"""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def main():
    # 解析命令行参数
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    model_args.trust_remote_code = True
    model_args.use_flash_attention_2 = True

    # 设置基础日志配置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 设置日志级别
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 记录训练参数
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # 检查是否存在检查点
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # 设置随机种子
    set_seed(training_args.seed)

    # 加载数据集
    is_mask = True
    DATA_PATH = "data/dpo_59k.json"
    list_data_dict = jload(DATA_PATH)
    raw_datasets = Dataset.from_list(list_data_dict)

    logger.info(f"Training set length: {len(list_data_dict)}")

    # 设置分词器参数并加载
    data_args.truncation_side = "left"  # 从左侧截断以确保不丢失最后一轮的标签
    # For base
    tokenizer = get_tokenizer(model_args, data_args, auto_set_chat_template=False)

    # 划分训练集和验证集
    train_val_split = raw_datasets.train_test_split(test_size=0.05)
    raw_datasets = DatasetDict({
        "train": train_val_split["train"],
        "test": train_val_split["test"]
    })

    # 记录训练集样本示例
    for index in random.sample(range(len(raw_datasets["train"])), 1):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    # 设置模型参数
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    # 加载模型
    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision) is True:
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )

    # 设置参考模型
    ref_model = model

    if model_args.use_peft is True:
        ref_model = None
    
    if is_mask:
        print('Using our MaskDPO')
        trainer = CustomDPOTrainer(
            model,
            ref_model,
            args=training_args,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"],
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args)
        )
    else:
        print('Using Vanilla DPO from TRL')
        trainer = DPOTrainer(
            model,
            ref_model,
            args=training_args,
            train_dataset=raw_datasets["train"],
            eval_dataset=raw_datasets["test"],
            processing_class=tokenizer,
            peft_config=get_peft_config(model_args)
        )

    # 开始训练
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # 在主进程上保存其他内容
    if trainer.accelerator.is_main_process:
        # 恢复k,v缓存以加速推理
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # 评估
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()