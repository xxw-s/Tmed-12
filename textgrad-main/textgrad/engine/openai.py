try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    raise ImportError("If you'd like to use OpenAI models, please install the openai package by running `pip install openai`, and add 'OPENAI_API_KEY' to your environment variables.")
# -------------------------- 新增：导入Tokenizer --------------------------
from transformers import AutoTokenizer
# -------------------------------------------------------------------------
import os
import json
import base64
from uu import Error
import platformdirs
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union

from .base import EngineLM, CachedEngine
from .engine_utils import get_image_type_from_bytes

# Default base URL for OLLAMA
OLLAMA_BASE_URL = 'http://localhost:11434/v1'

# Check if the user set the OLLAMA_BASE_URL environment variable
if os.getenv("OLLAMA_BASE_URL"):
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

class ChatOpenAI(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str="gpt-3.5-turbo-0613",
        system_prompt: str=DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool=False,
        base_url: str=None,
        **kwargs):
        """
        :param model_string:
        :param system_prompt:
        :param base_url: Used to support Ollama
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_openai_{model_string}.db")

        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        self.base_url = base_url
        
        if not base_url:
            if os.getenv("OPENAI_API_KEY") is None:
                raise ValueError("Please set the OPENAI_API_KEY environment variable if you'd like to use OpenAI models.")
            
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif base_url and base_url == OLLAMA_BASE_URL:
            self.client = OpenAI(
                base_url=base_url,
                api_key="ollama"
            )
        else:
            raise ValueError("Invalid base URL provided. Please use the default OLLAMA base URL or None.")

        self.model_string = model_string
        self.is_multimodal = is_multimodal

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt: str=None, **kwargs):
        if isinstance(content, str):
            return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)
        
        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if (has_multimodal_input) and (not self.is_multimodal):
                raise NotImplementedError("Multimodal generation is only supported for Claude-3 and beyond.")
            
            return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)

    def _generate_from_single_prompt(
        self, prompt: str, system_prompt: str=None, temperature=0, max_tokens=2000, top_p=0.99
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        if cache_or_none is not None:
            return cache_or_none

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response = response.choices[0].message.content
        self._save_cache(sys_prompt_arg + prompt, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        """Helper function to format a list of strings and bytes into a list of dictionaries to pass as messages to the API.
        """
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                # For now, bytes are assumed to be images
                image_type = get_image_type_from_bytes(item)
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}"
                    }
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        cache_key = sys_prompt_arg + json.dumps(formatted_content)
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": formatted_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response_text = response.choices[0].message.content
        self._save_cache(cache_key, response_text)
        return response_text


class AzureChatOpenAI(ChatOpenAI):
    def __init__(
        self,
        model_string="gpt-35-turbo",
        system_prompt=ChatOpenAI.DEFAULT_SYSTEM_PROMPT,
        **kwargs):
        """
        Initializes an interface for interacting with Azure's OpenAI models.

        This class extends the ChatOpenAI class to use Azure's OpenAI API instead of OpenAI's API. It sets up the necessary client with the appropriate API version, API key, and endpoint from environment variables.

        :param model_string: The model identifier for Azure OpenAI. Defaults to 'gpt-3.5-turbo'.
        :param system_prompt: The default system prompt to use when generating responses. Defaults to ChatOpenAI's default system prompt.
        :param kwargs: Additional keyword arguments to pass to the ChatOpenAI constructor.

        Environment variables:
        - AZURE_OPENAI_API_KEY: The API key for authenticating with Azure OpenAI.
        - AZURE_OPENAI_API_BASE: The base URL for the Azure OpenAI API.
        - AZURE_OPENAI_API_VERSION: The API version to use. Defaults to '2023-07-01-preview' if not set.

        Raises:
            ValueError: If the AZURE_OPENAI_API_KEY environment variable is not set.
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_azure_{model_string}.db")  # Changed cache path to differentiate from OpenAI cache

        super().__init__(cache_path=cache_path, system_prompt=system_prompt, **kwargs)

        self.system_prompt = system_prompt
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-07-01-preview")
        if os.getenv("AZURE_OPENAI_API_KEY") is None:
            raise ValueError("Please set the AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_BASE, and AZURE_OPENAI_API_VERSION environment variables if you'd like to use Azure OpenAI models.")
        
        self.client = AzureOpenAI(
            api_version=api_version,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            azure_deployment=model_string,
        )
        self.model_string = model_string


class VllmServer(EngineLM, CachedEngine):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."
    def __init__(
        self,
        model_string: str = "gpt-3.5-turbo-0613",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        base_url: str = None,
        api_key: str = None,
        max_tokens: int = 8192,
        # -------------------------- 新增：Tokenizer 和模型最大长度参数 --------------------------
        tokenizer_path: str = "/home/jx-vmlab/user/WJ/Med-1.0/Llava_med-1.0",  # 你的llava模型路径（含Tokenizer）
        model_max_length: int = 2048,  # llava-1.5-7b默认4096，可根据模型修改
        # -------------------------------------------------------------------------
        **kwargs,
    ):
        """
        :param model_string:
        :param system_prompt:
        :param base_url: Used to support Ollama
        """
        root = platformdirs.user_cache_dir("textgrad")
        cache_path = os.path.join(root, f"cache_openai_{model_string}.db")

        super().__init__(cache_path=cache_path)

        self.system_prompt = system_prompt
        self.base_url = base_url

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        self.model_string = model_string.replace("server-", "")
        self.max_tokens = max_tokens # max_tokens for the general vllm server for all forward passes, including loss(), grad() and step(); will override forward params by min() function
        self.is_multimodal = is_multimodal

        # -------------------------- 新增：初始化Tokenizer --------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model_max_length = model_max_length  # 模型最大上下文长度

    # -------------------------------------------------------------------------

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt: str=None, **kwargs):
        if isinstance(content, str):
            return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)

        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if (has_multimodal_input) and (not self.is_multimodal):
                raise NotImplementedError("Multimodal generation is only supported for Claude-3 and beyond.")

            return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)
    def _generate_from_single_prompt(
        self, prompt: str, system_prompt: str=None, temperature=0.7, min_tokens=16, max_tokens=4096, top_p=0.99, n=1, seed=None
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        # -------------------------- 新增：动态计算输入token数和max_tokens，并处理超限情况 --------------------------
        # 1. 分别计算system prompt和user prompt的token数
        sys_prompt_tokens = len(self.tokenizer.encode(sys_prompt_arg, add_special_tokens=False)) if sys_prompt_arg else 0
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        # 2. 预留空间：至少保留100 tokens用于生成，加上一些安全余量（20 tokens用于特殊token和格式）
        reserved_tokens = 120  # 100 (生成) + 20 (安全余量)
        max_input_tokens = self.model_max_length - reserved_tokens

        # 3. 如果输入超过限制，需要截断prompt（保留system prompt）
        total_input_tokens = sys_prompt_tokens + prompt_tokens
        if total_input_tokens > max_input_tokens:
            # 计算prompt最多允许的token数
            max_prompt_tokens = max_input_tokens - sys_prompt_tokens
            if max_prompt_tokens <= 0:
                # 如果system prompt本身就超过了限制，需要截断system prompt（极端情况）
                print(f"⚠️ 警告：System prompt过长（{sys_prompt_tokens} tokens），将截断")
                sys_prompt_tokens_list = self.tokenizer.encode(sys_prompt_arg, add_special_tokens=False)
                sys_prompt_tokens_list = sys_prompt_tokens_list[:max_input_tokens - 50]  # 保留50给prompt
                sys_prompt_arg = self.tokenizer.decode(sys_prompt_tokens_list, skip_special_tokens=True)
                sys_prompt_tokens = len(sys_prompt_tokens_list)
                max_prompt_tokens = 50

            # 截断prompt：从后往前截断（保留前面的重要内容）
            prompt_tokens_list = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(prompt_tokens_list) > max_prompt_tokens:
                print(f"⚠️ 警告：输入文本过长（{total_input_tokens} tokens），将截断prompt从{len(prompt_tokens_list)}到{max_prompt_tokens} tokens")
                prompt_tokens_list = prompt_tokens_list[:max_prompt_tokens]
                prompt = self.tokenizer.decode(prompt_tokens_list, skip_special_tokens=True)
                prompt_tokens = len(prompt_tokens_list)

        # 4. 重新计算总输入token数（截断后）
        input_token_count = sys_prompt_tokens + prompt_tokens

        # 5. 动态计算允许的max_tokens：模型最大长度 - 输入token数 - 安全余量
        dynamic_max_tokens = self.model_max_length - input_token_count - 20
        # 6. 兜底：避免dynamic_max_tokens过小（至少10，保证能生成内容），且不超过原max_tokens参数
        dynamic_max_tokens = max(dynamic_max_tokens, 10)
        dynamic_max_tokens = min(dynamic_max_tokens, self.max_tokens, max_tokens)

        # 7. 警告：若输入token数接近模型上限，提示可能生成不完整
        if input_token_count > self.model_max_length - 100:
            print(f"⚠️ 警告：输入文本较长（{input_token_count} tokens），生成内容可能不完整（模型上限{self.model_max_length} tokens）")
        # -------------------------------------------------------------------------

        # cache_or_none = self._check_cache(sys_prompt_arg + prompt)
        # if cache_or_none is not None:
        #     return cache_or_none
        max_tokens = min(max_tokens, self.max_tokens)
        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            frequency_penalty=0,
            presence_penalty=0,
            temperature=temperature,
#-------------------------用动态计算的结果-----------------
            max_tokens=dynamic_max_tokens,
            #max_tokens=max_tokens,
            top_p=top_p,
            n=n,
            stop=['<|end_of_text|>', '<|eot_id|>', '<|im_end|>'],
            seed=seed,
        )
        if n > 1:
            response = [response.choices[i].message.content for i in range(n)]
        else:
            response = response.choices[0].message.content
            # self._save_cache(sys_prompt_arg + prompt, response)
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: Union[str, dict, List[Union[str, bytes, dict]]]) -> List[dict]:
        """简化的内容格式化方法，正确处理各种类型输入。"""
        # 基础类型处理
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        elif isinstance(content, bytes):
            try:
                return [{"type": "text", "text": content.decode('utf-8', errors='ignore')}]
            except:
                return [{"type": "text", "text": ""}]
        
        # 多模态字典输入处理
        elif isinstance(content, dict):
            formatted = []
            # 处理文本部分
            if "text" in content and isinstance(content["text"], (str, bytes)):
                text = content["text"]
                if isinstance(text, bytes):
                    try:
                        text = text.decode('utf-8', errors='ignore')
                    except:
                        text = ""
                formatted.append({"type": "text", "text": text})
            # 处理图像部分（对Llava_med特殊处理）
            if "image" in content and isinstance(content["image"], str):
                if "Llava_med" in self.model_string:
                    self.current_image = content["image"]
                else:
                    formatted.append({"type": "image_url", "image_url": {"url": content["image"]}})
            return formatted
        
        # 列表处理 - 只提取文本内容，避免重复计算
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, bytes):
                    try:
                        text_parts.append(item.decode('utf-8', errors='ignore'))
                    except:
                        pass
                elif isinstance(item, dict):
                    # 提取字典中的文本内容
                    if item.get("type") == "text" and isinstance(item.get("text"), (str, bytes)):
                        text = item.get("text")
                        if isinstance(text, bytes):
                            try:
                                text = text.decode('utf-8', errors='ignore')
                            except:
                                text = ""
                        text_parts.append(text)
                    # 处理图像URL
                    elif item.get("type") == "image_url" and "Llava_med" in self.model_string:
                        img_url = item.get("image_url", {}).get("url", "")
                        if img_url:
                            self.current_image = img_url
            # 合并所有文本部分
            combined_text = " ".join(text_parts).strip()
            return [{"type": "text", "text": combined_text}] if combined_text else [{"type": "text", "text": ""}]
        
        # 默认返回
        return [{"type": "text", "text": ""}]

    def _generate_from_multiple_input(
    self, content: Union[str, dict, List], system_prompt=None, temperature=0, max_tokens=2048, top_p=0.99, n=1, seed=None):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)
        
        # 改进的token计算 - 避免过大的token数量
        total_text = " ".join([item.get("text", "") for item in formatted_content if item.get("type") == "text"])
        
        # 限制输入文本长度，避免token溢出
        if len(total_text) > 2000:  # 限制输入长度
            total_text = total_text[:2000]
        
        try:
            text_tokens = len(self.tokenizer.encode(total_text, add_special_tokens=False)) if total_text else 0
            sys_tokens = len(self.tokenizer.encode(sys_prompt_arg, add_special_tokens=False)) if sys_prompt_arg else 0
        except Exception as e:
            print(f"⚠️ Token计算错误: {str(e)}")
            text_tokens = 0
            sys_tokens = 0
        
        # 安全的max_tokens计算
        available_tokens = max(100, min(1000, self.model_max_length - (sys_tokens + text_tokens) - 100))
        final_max_tokens = min(available_tokens, max_tokens, self.max_tokens)
        
        # 构建缓存键
        cache_key = sys_prompt_arg + json.dumps(formatted_content)
        cache_or_none = self._check_cache(cache_key)
        if cache_or_none is not None:
            return cache_or_none
        
        try:
            print(f"调试信息：准备调用API，model={self.model_string}, temperature={temperature}, max_tokens={final_max_tokens}, n={n}")
            
            # 为Llava_med模型构建正确的多模态输入
            messages = []
            if sys_prompt_arg:
                messages.append({"role": "system", "content": sys_prompt_arg})
            
            # Llava_med模型的特殊处理
            if "Llava_med" in self.model_string and hasattr(self, "current_image") and self.current_image:
                # 合并所有文本内容
                user_content = []
                # 添加文本部分
                if total_text:
                    user_content.append({"type": "text", "text": total_text})
                # 添加图像部分
                user_content.append({"type": "image_url", "image_url": {"url": self.current_image}})
                
                # 构建单个用户消息包含文本和图像
                messages.append({"role": "user", "content": user_content})
                # 清除图像引用
                self.current_image = None
            else:
                # 非多模态或其他模型
                messages.append({"role": "user", "content": total_text})
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.model_string,
                messages=messages,
                temperature=temperature,
                max_tokens=final_max_tokens,
                top_p=top_p,
                n=n,
                seed=seed,
                stop=["<|end_of_text|", "<|eot_id|", "<|im_end|>"],
            )
            
            # 修复的响应处理逻辑
            if hasattr(response, "choices") and response.choices:
                if n > 1:
                    # 多响应处理
                    results = []
                    for choice in response.choices[:n]:
                        if hasattr(choice, "message") and hasattr(choice.message, "content"):
                            choice_content = choice.message.content
                            if isinstance(choice_content, str):
                                choice_content = choice_content.strip()
                                # 增强前缀清理
                                for prefix in ["Assistant:", "INDEX-1<SEP>", "Human:"]:
                                    if choice_content.startswith(prefix):
                                        choice_content = choice_content[len(prefix):].strip()
                                if choice_content and choice_content != "无法生成有效响应":
                                    results.append(choice_content)
                    return results if results else []
                else:
                    # 修复单响应处理 - 正确获取content变量
                    if len(response.choices) > 0:
                        first_choice = response.choices[0]
                        if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
                            choice_content = first_choice.message.content
                            if isinstance(choice_content, str):
                                choice_content = choice_content.strip()
                                # 增强前缀清理
                                for prefix in ["Assistant:", "INDEX-1<SEP>", "Human:"]:
                                    if choice_content.startswith(prefix):
                                        choice_content = choice_content[len(prefix):].strip()
                                # 过滤无效响应
                                if choice_content and choice_content != "无法生成有效响应":
                                    return choice_content
                
            # 默认返回空列表或空字符串
            return [] if n > 1 else ""
            
        except Exception as e:
            print(f"⚠️ 警告：API调用出错: {type(e).__name__}: {str(e)}")
            return [] if n > 1 else ""
