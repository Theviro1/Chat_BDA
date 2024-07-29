from typing import Optional, List, Mapping, Any
import torch
import yaml
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun


class CustomModel(LLM):
    config_dir: str = None
    config: dict = None
    device: torch.device = None
    tokenizer: PreTrainedTokenizer = None
    model: PreTrainedModel = None

    def __init__(self, config_dir: str = 'config/model_config.yaml', **kwargs: Any):
        # 读取配置文件
        super().__init__(**kwargs)
        with open(config_dir, 'r') as f:
            config = yaml.safe_load(f)
        model_path = config['model_path']
        # 设置参数
        self.config_dir = config_dir
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 可使用的硬件设备
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  # 指定模型的分词器并信任
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True)  # 指定模型自身并信任
        # 初始化
        self.model.eval()

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # 将prompt经过tokenizer进行处理，input_ids设置为long保证对于任何模型都可以使用
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs['input_ids'] = inputs['input_ids'].to(torch.long)
        # 输入模型
        outputs = self.model(**inputs)
        # 解码并输出
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            'config_dir': self.config_dir,
            'config': self.config,
            'device': self.device,
            'tokenizer': self.tokenizer,
            'model': self.model
        }

    @property
    def _llm_type(self) -> str:
        return 'custom_model'
