from typing import Dict, Optional, Union, Tuple, Any
import torch
import torch.nn.functional as F
from trl import DPOTrainer
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.utils.data import Dataset
from dataclasses import dataclass
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import os
import torch.nn as nn
from diff_lib import get_line_masks, get_augmented_line_masks, get_token_level_masks, get_prefix_suffix_masks


class CustomDPOTrainer(DPOTrainer):
    """
    CustomDPOTrainer 继承自 DPOTrainer，添加了基于代码差异的掩码机制。
    """

    def __init__(
            self,
            model: Union[PreTrainedModel, torch.nn.Module],
            ref_model: Optional[Union[PreTrainedModel, torch.nn.Module]] = None,
            args=None,
            data_collator=None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            model_init=None,
            compute_metrics=None,
            callbacks=None,
            optimizers=(None, None),
            preprocess_logits_for_metrics=None,
            peft_config=None,
            **kwargs,
    ):

        self.processing_class = processing_class
        self.current_batch = None

        super().__init__(
            model=model,
            ref_model=ref_model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
            **kwargs,
        )

    def concatenated_forward(self, model: nn.Module, batch: dict[str, Union[list, torch.LongTensor]]):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        num_examples = batch["prompt_input_ids"].shape[0]

        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_kwargs = {}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True

        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        # prompt_input_ids: Concatenated prompt input IDs of shape `(2 * batch_size, prompt_length)
        # prompt_attention_mask: Concatenated prompt attention masks of shape `(2 * batch_size, prompt_length)

        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        # completion_input_ids: Concatenated chosen and rejected completion input IDs of shape `(2 * batch_size, max_completion_length)
        # completion_attention_mask: Concatenated chosen and rejected attention masks of shape `(2 * batch_size, max_completion_length).

        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else: # Decoder-Only
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )

            # Flush left to reduce the memory usage
            # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
            #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
            for i in range(attention_mask.size(0)):
                first_one_idx = torch.nonzero(attention_mask[i])[0].item()
                input_ids[i] = torch.roll(input_ids[i], shifts=-first_one_idx)
                attention_mask[i] = torch.roll(attention_mask[i], shifts=-first_one_idx)
                loss_mask[i] = torch.roll(loss_mask[i], shifts=-first_one_idx)

            # Get the first column idx that is all zeros and remove every column after that
            empty_cols = torch.sum(attention_mask, dim=0) == 0
            first_empty_col = torch.nonzero(empty_cols)[0].item() if empty_cols.any() else attention_mask.size(1)
            input_ids = input_ids[:, :first_empty_col]
            attention_mask = attention_mask[:, :first_empty_col]
            loss_mask = loss_mask[:, :first_empty_col]

            # Truncate right
            if self.args.max_length is not None:
                input_ids = input_ids[:, : self.args.max_length]
                attention_mask = attention_mask[:, : self.args.max_length]
                loss_mask = loss_mask[:, : self.args.max_length]

            # input_ids: (2 *bs, prompt+completion)
            # attention_mask: (2 *bs, prompt+completion)

            # Get diff mask: (2 *bs, prompt+completion)
            bs, seq_length = input_ids.size(0) // 2, input_ids.size(1)
            diff_masks = torch.zeros_like(input_ids, dtype=torch.bool)
            start_ids = self.processing_class('```python', add_special_tokens=False)['input_ids']
            # single get mask
            for i in range(bs):
                chosen_input_ids = input_ids[i]  # 第 i 个 chosen
                rejected_input_ids = input_ids[i + bs]  # 第 i 个 rejected
            
                # 寻找 ```python 子序列在 chosen 和 rejected 中的起始位置
                def find_start_index(sequence, start_ids):
                    for j in range(len(sequence) - len(start_ids) + 1):
                        if sequence[j:j + len(start_ids)].tolist() == start_ids:
                            return j + len(start_ids)  # 从子序列结束后一个 token 开始
                    return -1 
            
                chosen_start_index = find_start_index(chosen_input_ids, start_ids) -2
                rejected_start_index = find_start_index(rejected_input_ids, start_ids) -2
            
                if chosen_start_index == -1 or rejected_start_index == -1:
                    raise ValueError("`start_ids` not found in input sequences.")
                # 提取代码部分的 token ids
                chosen_code_ids = chosen_input_ids[chosen_start_index:]
                rejected_code_ids = rejected_input_ids[rejected_start_index:]
            
                # 获取差异掩码
                chosen_diff_mask, rejected_diff_mask = get_line_masks(
                    chosen_ids=chosen_code_ids,
                    rejected_ids=rejected_code_ids,
                    tokenizer=self.processing_class
                )
            
                # 完整的chosen + rejected中的错误部分
                diff_masks[i, chosen_start_index:chosen_start_index + len(chosen_diff_mask)] = True
                diff_masks[i + bs, rejected_start_index:rejected_start_index + len(rejected_diff_mask)] = rejected_diff_mask

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)

            # Offset the logits by one to align with the labels
            logits = outputs.logits[:, :-1, :] #  [2*bs, seq_len-1, vocab_size]
            labels = input_ids[:, 1:].clone() #  [2*bs, seq_len-1]

            loss_mask = loss_mask[:, 1:].bool() # [2*bs, seq_len-1]
            diff_masks = diff_masks[:, 1:].bool() # [2*bs, seq_len-1]
            # mask dpo
            final_mask = loss_mask & diff_masks

            assert loss_mask.shape == diff_masks.shape, "Mismatch in mask shapes!"

            # if self.use_num_logits_to_keep:
            #     # Align labels with logits
            #     # logits:    -,  -, [x2, x3, x4, x5, x6]
            #     #                     ^ --------- ^       after logits[:, :-1, :]
            #     # labels:   [y0, y1, y2, y3, y4, y5, y6]
            #     #                         ^ --------- ^   with num_logits_to_keep=4, [:, -4:]
            #     # loss_mask: [0,  0,  0,  1,  1,  1,  1]
            #     labels = labels[:, -num_logits_to_keep:]
            #     final_mask = final_mask[:, -num_logits_to_keep:]

        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_logps[~final_mask] = 0

        all_logps = per_token_logps.sum(-1)  # 只累加差异位置的对数概率

        output = {}

        if self.args.rpo_alpha is not None:
            # Only use the chosen logits for the RPO loss
            chosen_logits = logits[:num_examples]
            chosen_labels = labels[:num_examples]

            # Compute the log probabilities of the labels (mean)
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )

        output["chosen_logps"] = all_logps[:num_examples]  # 取前半部分作为 chosen
        output["rejected_logps"] = all_logps[num_examples:]  # 取后半部分作为 rejected
        output["mean_chosen_logits"] = logits[:num_examples][final_mask[:num_examples]].mean()
        output["mean_rejected_logits"] = logits[num_examples:][final_mask[num_examples:]].mean()

        if self.aux_loss_enabled:
            output["aux_loss"] = outputs.aux_loss

        return output


    def get_batch_loss_metrics(
            self,
            model: Union[PreTrainedModel, torch.nn.Module],
            batch: Dict[str, Union[torch.Tensor, Any]],
            train_eval: str = "train",
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算一个批次的损失和指标
        """
        self.current_batch = batch  # 保存当前批次
        metrics = {}

        model.train()
        model_output = self.concatenated_forward(model, batch)
        nll_loss = model_output["nll_loss"]

        # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        dpo_losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            model_output["chosen_logps"], model_output["rejected_logps"], ref_chosen_logps, ref_rejected_logps
        )

        if self.args.rpo_alpha is not None:
            losses = dpo_losses + self.args.rpo_alpha * model_output["nll_loss"]  # RPO loss

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean()
        metrics[f"{prefix}logps/chosen"] = model_output["chosen_logps"].detach().mean().cpu()
        metrics[f"{prefix}logps/rejected"] = model_output["rejected_logps"].detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = model_output["mean_chosen_logits"].detach().cpu()
        metrics[f"{prefix}logits/rejected"] = model_output["mean_rejected_logits"].detach().cpu()
        metrics[f"{prefix}dpo_loss"] = dpo_losses.detach().mean().cpu()
        metrics[f"{prefix}nll_loss"] = nll_loss.detach().mean().cpu()
        metrics[f"{prefix}total_loss"] = losses.detach().mean().cpu()

        return losses.mean(), metrics