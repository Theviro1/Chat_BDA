from typing import Optional, List, Mapping, Any
import torch
import yaml
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun


class CustomLLM(LLM):
    model_path: str = None
    device: torch.device = None
    tokenizer: PreTrainedTokenizer = None
    model: PreTrainedModel = None

    def __init__(self, config_dir: str = '/home/hjl/Chat_BDA/config/model/model_config.yaml', **kwargs: Any):
        # 读取配置文件加载模型
        super().__init__(**kwargs)
        with open(config_dir, 'r') as f:
            config = yaml.safe_load(f)
        # 设置参数
        self.model_path = config['model_path']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 可使用的硬件设备
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)  # 指定模型的分词器并信任
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)  # 指定模型自身并信任
        # 初始化
        self.model.to(self.device).eval().bfloat16()
            

    # 处理模型的输出
    def parse_output(self, output:str):
        return output.replace('\n', '').replace('\t', '')

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # 将prompt经过tokenizer进行处理，input_ids设置为long保证对于任何模型都可以使用
        inputs = self.tokenizer.apply_chat_template([{'role':'user', 'content':prompt}], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors='pt').to(self.device)
        # 输入模型
        outputs = self.model.generate(**inputs)[:, inputs['input_ids'].shape[1]:]
        # 解码并输出
        return self.parse_output(self.tokenizer.decode(outputs[0], skip_special_tokens=True))


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {}

    @property
    def _llm_type(self) -> str:
        return 'custom_model'
