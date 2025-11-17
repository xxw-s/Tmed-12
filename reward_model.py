#from typing import List, Dict, Any
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
import numpy as np

from transformers import AutoTokenizer, pipeline


class PipelineDataset(torch.utils.data.Dataset):
    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TPORewardModel():
    def __init__(self, reward_model: str = "sfairXC/FsfairX-LLaMA3-RM-v0.1"):
        self.tokenizer = AutoTokenizer.from_pretrained(reward_model)
        self.pipe = pipeline(
                "sentiment-analysis",
                model=reward_model,
                device_map="auto",
                tokenizer=self.tokenizer,
                model_kwargs={"torch_dtype": torch.bfloat16})

    #def compute_reward_scores(self, messages: List, pipe_kwargs: Dict[str, Any]) -> List[float]:
        #test_texts = [_.replace(self.tokenizer.bos_token, "") for _ in self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)]
        #test_dataset = PipelineDataset(test_texts)
    def compute_reward_scores(self, prompts: List[str], pipe_kwargs: Dict[str, Any]) -> List[float]:
        test_dataset = PipelineDataset(prompts)

        rewards = [output[0]["score"] for output in self.pipe(test_dataset, **pipe_kwargs)]
        
        return rewards
#++
    def _coerce_messages(self, qa_pair: Union[Tuple[str, str], Dict[str, Any]]) -> List[Dict[str, str]]:
            if isinstance(qa_pair, dict):
                if "messages" in qa_pair:
                    messages = qa_pair["messages"]
                    if not isinstance(messages, list):
                        raise TypeError("'messages' must be provided as a list of role/content dictionaries.")
                    formatted_messages: List[Dict[str, str]] = []
                    for message in messages:
                        if not isinstance(message, dict) or "role" not in message or "content" not in message:
                            raise ValueError("Each message must include 'role' and 'content' keys.")
                        formatted_messages.append({"role": message["role"], "content": message["content"]})
                    return formatted_messages

                prompt = qa_pair.get("prompt") or qa_pair.get("query") or qa_pair.get("question")
                answer = qa_pair.get("answer") or qa_pair.get("response") or qa_pair.get("generation")
                if prompt is None or answer is None:
                    raise ValueError("Reward model pairs must include both a prompt and an answer.")

                system_prompt = qa_pair.get("system_prompt")
                context = qa_pair.get("context")

                conversation: List[Dict[str, str]] = []
                if isinstance(system_prompt, str) and system_prompt.strip():
                    conversation.append({"role": "system", "content": system_prompt.strip()})

                if isinstance(context, list):
                    for message in context:
                        if isinstance(message, dict) and "role" in message and "content" in message:
                            conversation.append({"role": message["role"], "content": message["content"]})
                elif isinstance(context, str) and context.strip():
                    conversation.append({"role": "user", "content": context.strip()})

                conversation.append({"role": "user", "content": prompt})
                conversation.append({"role": "assistant", "content": answer})
                return conversation

            prompt, answer = qa_pair
            return [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]


    #def perform_rm(self, qa_pairs):
    def perform_rm(self, qa_pairs: Sequence[Union[Tuple[str, str], Dict[str, Any]]]) -> List[float]:
        
        pipe_kwargs = {
            # "return_all_scores": True,
            "top_k": None,
            "function_to_apply": "none",
            "batch_size": 1, 
        }
        compiled_prompts: List[str] = []
        for qa_pair in qa_pairs:
            conversation = self._coerce_messages(qa_pair)
            text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            if self.tokenizer.bos_token:
                text = text.replace(self.tokenizer.bos_token, "")
            compiled_prompts.append(text)
        rewards = self.compute_reward_scores(compiled_prompts, pipe_kwargs)

        return rewards

    def get_contrastive_samples(self, scores, qa_pairs):
        """
        Get contrastive samples based on the best and worst scores from a reward model.

        Args:
            qa_pairs (list of dict): List of question-answer pairs, each a dictionary containing 'question' and 'answer'.
            rm_pipe (Callable): Reward model pipeline that returns scores for QA pairs.
            rm_tokenizer (Callable): Tokenizer for preprocessing QA pairs before scoring.

        Returns:
            dict: A dictionary containing the best and worst QA pairs.
        """

        def truncate_text(text, tokenizer, max_length=2048):
            """
            Truncate a text to a maximum token length using a given tokenizer.
            """
            token_ids = tokenizer.encode(text, truncation=False)
            if len(token_ids) <= max_length:
                return text
            truncated_token_ids = token_ids[:max_length]
            truncated_text = tokenizer.decode(truncated_token_ids, skip_special_tokens=True)

            return truncated_text

        
        # Get the indices of the best and worst scores
        best_index = np.argmax(scores)
        worst_index = np.argmin(scores)
        delta = max(scores) - min(scores)
        # Extract the best and worst QA pairs

        def _extract_answer(entry):
            if isinstance(entry, dict):
                return entry.get("answer") or entry.get("response") or entry.get("generation")
            return entry[1]

        best_answer = truncate_text(_extract_answer(qa_pairs[best_index]), self.tokenizer)
        worst_answer = truncate_text(_extract_answer(qa_pairs[worst_index]), self.tokenizer)
        
        return {
            "best": best_answer,
            "worst": worst_answer
        }, delta

