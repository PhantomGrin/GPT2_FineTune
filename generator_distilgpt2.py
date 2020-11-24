#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Text generation using GPT2/DistilGPT2
"""

import logging
import torch
from jsonmerge import merge
from transformers import GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop
SEED = 42

ARGS = {
    'trigger_token':":",
    'length': 20,
    'stop_token': "",
    'temperature': 1.0,
    'k': 0,
    'p': 0.9,
    'num_return_sequences': 3,
    'repetition_penalty': 1.0
}

MODEL_PATH = './model/Model_DistilGPT2'

def set_seed(args):
    global SEED
    torch.manual_seed(SEED)
    if args['n_gpu'] > 0:
        torch.cuda.manual_seed_all(SEED)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = MAX_LENGTH  # avoid infinite loop
    return length


def load_model():
    ARGS['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARGS['n_gpu'] = torch.cuda.device_count()

    logger.info(
        "device: %s, n_gpu: %s",
        ARGS['device'],
        ARGS['n_gpu']
    )

    set_seed(ARGS)

    # Initialize the model and tokenizer
    try:
        model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = tokenizer_class.from_pretrained(MODEL_PATH)
        model = model_class.from_pretrained(MODEL_PATH)
        model.to(ARGS['device'])
        logger.info("Model loading completed")
    except KeyError:
        raise KeyError("There is an error loading model GPT2")
 
    return model, tokenizer


def service(input_json, model, tokenizer):
    args = merge(ARGS, input_json)
    print(args)

    args['length'] = adjust_length_to_model(args['length'], max_sequence_length=model.config.max_position_embeddings)
    logger.info(args)

    data_provider = {}

    for prompt in args['prompt']:
        prompt_text = prompt + args['trigger_token']
        encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=True, return_tensors="pt")
        encoded_prompt = encoded_prompt.to(args['device'])

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=args['length'] + len(encoded_prompt[0]),
            temperature=args['temperature'],
            top_k=args['k'],
            top_p=args['p'],
            repetition_penalty=args['repetition_penalty'],
            do_sample=True,
            num_return_sequences=args['num_return_sequences'],
        )

        # Remove the batch dimension when returning multiple sequences
        if len(output_sequences.shape) > 2:
            output_sequences.squeeze_()

        generated_sequences = []

        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            # print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
            generated_sequence = generated_sequence.tolist()

            # Decode text
            text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

            # Remove all text after the stop token
            text = text[: text.find(args['stop_token']) if args['stop_token'] else None]

            # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
            total_sequence = (
                text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
            )

            generated_sequences.append(total_sequence)
            # print(total_sequence)
        
        generated_sequences = [item.strip() for item in generated_sequences]
        data_provider[prompt] = generated_sequences
    
    return data_provider