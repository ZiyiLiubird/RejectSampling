import torch
from dataclasses import dataclass
import torch.nn as nn
from torch import distributed as dist

from typing import Any, ClassVar, Literal
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.generic import ModelOutput
from transformers import LlamaModel,PretrainedConfig, AutoTokenizer, LlamaPreTrainedModel, PreTrainedModel
from transformers.utils.doc import add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.models.llama.modeling_llama import _CONFIG_FOR_DOC, LLAMA_INPUTS_DOCSTRING

from models.normalizer import Normalizer, NormalizeFunction

@dataclass
class ScoreModelOutput(ModelOutput):
    """
    Output of the score model.

    Args:
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim, sequence_length)`):
            Prediction scores of the score model.
        end_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, score_dim)`):
            Prediction scores of the end of the sequence.
    """
    def __init__(self, scores, end_scores):
        self.scores = scores  # size = (B, L, D)
        self.end_scores = end_scores  # size = (B, D)


class ScoreModelMixin:
    """Base class for score models."""

    score_head: nn.Linear
    normalizer: Normalizer
    do_normalize: bool = False
    normalize_function: NormalizeFunction = 'affine'
    _initialized: bool = False

    def init_score_head(self, config: PretrainedConfig, hidden_size: int, **kwargs: Any) -> None:
        """Initialize the score head."""
        if self._initialized:
            return

        config.score_dim = kwargs.pop('score_dim', getattr(config, 'score_dim', 1))
        config.bias = kwargs.pop('bias', getattr(config, 'bias', False))

        config.score_type = kwargs.pop('score_type', getattr(config, 'score_type', 'reward'))
        if config.score_type == 'reward':
            self.normalize_function = 'affine'
        elif config.score_type == 'cost':
            self.normalize_function = 'scale'
        elif config.score_type == 'critic':
            self.normalize_function = 'identity'
        else:
            raise ValueError(
                f"Invalid score type: {config.score_type}. Expected one of 'reward', 'cost', or 'critic'.",
            )

        config.do_normalize = kwargs.pop(
            'do_normalize',
            getattr(config, 'do_normalize', False),
        )
        self.do_normalize = config.do_normalize

        config.normalizer_type = kwargs.pop(
            'normalizer_type',
            getattr(config, 'normalizer_type', None),
        )
        if config.normalizer_type not in {'RunningMeanStd', 'ExponentialMovingAverage', None}:
            raise ValueError(
                f'Invalid norm type: {config.normalizer_type}.'
                "Expected one of 'RunningMeadStd', 'ExponentialMovingAverage', or None.",
            )
        if config.normalizer_type == 'ExponentialMovingAverage':
            config.momentum = kwargs.pop('momentum', getattr(config, 'momentum', None))
        momentum = getattr(config, 'momentum', None)

        self.score_head = nn.Linear(hidden_size, config.score_dim, bias=config.bias)
        self.normalizer = Normalizer.instantiate(
            normalizer_type=config.normalizer_type,
            normalize_function=self.normalize_function,
            shape=(config.score_dim,),
            momentum=momentum,
        )

        mean = getattr(config, 'mean', None)
        var = getattr(config, 'var', None)
        self.normalizer.set_mean_var(mean, var)

        self._initialized = True

    def get_score(
        self,
        hidden_state: torch.Tensor,  # size = (B, L, E)
        attention_mask: torch.BoolTensor,  # size = (B, L)
        return_dict= None,
    ) -> ScoreModelOutput:
        """Forward pass of the score model."""
        scores = self.score_head(hidden_state)  # size = (B, L, D)

        end_score = []
        for i in range(hidden_state.size(0)):
            end_index = attention_mask[i].nonzero()[-1].item()
            end_score.append(scores[i, end_index])  # size = (D,)
        end_score = torch.stack(end_score, dim=0)  # size = (B, D)

        if self.training:
            if dist.is_initialized():
                gathered_end_score_list = [
                    torch.zeros_like(end_score) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_end_score_list, end_score)
                gathered_end_score = torch.cat(gathered_end_score_list, dim=0)
                self.normalizer.update(gathered_end_score)
            else:
                self.normalizer.update(end_score)
            self.config.mean = self.normalizer.mean.tolist()
            self.config.var = self.normalizer.var.tolist()

        if self.do_normalize:
            scores = self.normalizer.normalize(scores)
            end_score = self.normalizer.normalize(end_score)

        if not return_dict:
            return scores, end_score

        return ScoreModelOutput(
            scores=scores,  # size = (B, L, D)
            end_scores=end_score,  # size = (B, D)
        )

    def set_normalize(self, mode: bool = True) -> None:
        if self.do_normalize == mode:
            return

        self.do_normalize = self.config.do_normalize = mode


class LlamaModelForScore(ScoreModelMixin, LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing: ClassVar[list[str]] = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig, **kwargs: Any) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)

        config.architectures = [self.__class__.__name__]
        self.init_score_head(config, hidden_size=config.hidden_size, **kwargs)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> None:
        return None

    def set_decoder(self, decoder: PreTrainedModel) -> None:
        self.model = decoder

    def get_decoder(self) -> PreTrainedModel:
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ScoreModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(  # pylint: disable=too-many-arguments
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor = None,
        past_key_values: list[torch.FloatTensor] = None,
        inputs_embeds: torch.FloatTensor = None,
        use_cache: bool = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:

        Returns:

        Examples:

        ```python
        >>> from safe_rlhf.models.llama.modeling_llama import LlamaModelForScore
        >>> from transformers import LlamaTokenizer

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        # got score
        >>> outputs = model(**inputs)
        >>> scores = outputs.scores
        >>> scores
        tensor([[[0.0000]]])
        ```
        """
        assert attention_mask is not None
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # size = (B, L, E)
        return self.get_score(
            hidden_states,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )


@torch.no_grad()
def get_single_reward_from_model(
    model: LlamaModelForScore,
    tokenizer: AutoTokenizer,
    answer: str,
    messages: list[dict] = [],
    message_str: str = None,
    max_length: int = 4096,
    device: torch.device = torch.device("cuda")
):
    """Get reward from model.

    Args:
        model (ScoreModelMixin): init from LlamaModelForScore
        tokenizer (AutoTokenizer): init from AutoTokenizer
        messages (list[str]): list of messages
        answer (str): generator output
        max_length (int, optional): Defaults to 2048.
    """
    def tokenize(text):
        return tokenizer(text, return_tensors="pt", padding=True, truncation=True)['input_ids']
    
    if message_str is not None:
        prompt = message_str
    else:
        prompt = "[Start of Init]: "
        for mess in messages:
            if mess.role == 'system':
                prompt += mess.content + "[End of Init]"
            elif mess.role == 'user':
                prompt += '[User]: ' + mess.content
            else:
                prompt += '[Assistant]: ' + mess.content
    
    reward_input = 'BEGINNING OF CONVERSATION: USER: {} ASSISTANT:{}'
    reward_input = reward_input.format(prompt, answer)
    input_ids = tokenize(reward_input + tokenizer.eos_token)
    pad_len = max_length - input_ids.size(1)
    input_ids = torch.cat([input_ids, torch.zeros((1, pad_len), dtype=torch.long)], dim=1)
    attention_mask = (input_ids != 0).bool()
    outputs = model(input_ids.to(device), attention_mask.to(device))
    end_scores = outputs.end_scores.squeeze(dim=-1)
    return end_scores
