import torch

from minivla.llm_backbone import QwenBackbone
from minivla.action_tokenizer import ActionTokenizer


llm = QwenBackbone(
    model_name="Qwen/Qwen2.5-0.5B"
)

action_q01 = [
    -0.760714,
    -0.656250,
    -0.937500,
    -0.108214,
    -0.204643,
    -0.186429,
]

action_q99 = [
     0.937500,
     0.873214,
     0.934821,
     0.105000,
     0.175714,
     0.143571,
]


action_tokenizer = ActionTokenizer(
    tokenizer=llm.tokenizer,
    num_bins=256,
    action_q01=action_q01,
    action_q99=action_q99,
)

action = torch.tensor(
    [
        [
            0.06428571,
            0.0,
            0.08303571,
            0.0,
            0.05892857,
            0.0,
            -1.0,
        ]
    ],
    dtype=torch.float32,
)


token_ids = (
    action_tokenizer.encode(
        action
    )
)

tokens = (
    llm.tokenizer
    .convert_ids_to_tokens(
        token_ids[0].tolist()
    )
)

decoded_action = (
    action_tokenizer.decode(
        token_ids
    )
)


print(
    "Original action:"
)

print(
    action
)


print(
    "\nAction tokens:"
)

print(
    tokens
)


print(
    "\nDecoded action:"
)

print(
    decoded_action
)


print(
    "\nAbsolute error:"
)

print(
    (
        decoded_action
        - action
    ).abs()
)