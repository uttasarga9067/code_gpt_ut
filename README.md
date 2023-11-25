# code_gpt_ut

---
license: apache-2.0
datasets:
- iamtarun/python_code_instructions_18k_alpaca
language:
- en
library_name: peft
pipeline_tag: text2text-generation
tags:
- code
---


Here's a brief description of my project.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

# Introduction

colab_code_generator_FT_code_gen_UT, an instruction-following large language model trained on the Google Colab Pro with T4 GPU and fine-tuned on 'Salesforce/codegen-350M-mono' that is licensed for commercial use. Code Generator_UT is trained on ~19k instructions/response fine-tuning records from 'iamtarun/python_code_instructions_18k_alpaca'.

# Getting Started


## Installation
Loading the fine-tuned Code Generator
```
from peft import AutoPeftModelForCausalLM>
test_model_UT = AutoPeftModelForCausalLM.from_pretrained("01GangaPutraBheeshma/colab_code_generator_FT_code_gen_UT")
test_tokenizer_UT = AutoTokenizer.from_pretrained("01GangaPutraBheeshma/colab_code_generator_FT_code_gen_UT")
```

## Usage
For re-training this model, I would highly recommend using this format to provide input to the tokenizer.

```
def prompt_instruction_format(sample):
  return f"""### Instruction:
    Use the Task below and the Input given to write the Response, which is a programming code that can solve the following Task:

    ### Task:
    {sample['instruction']}

    ### Input:
    {sample['input']}

    ### Response:
    {sample['output']}

```

Then, we can leverage the above function to format our input prompts that can be pre-processed and used in the Model Training using Supervised Fine-Tuning or SFTTrainer Class.

```
trainer = SFTTrainer(
    model=model,
    train_dataset=code_dataset,
    peft_config=peft_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=prompt_instruction_format,
    args=trainingArgs,
)

```

This is a crucial step when we perform Reinforcement Learning with Human Feedback or RLHF for short. Here are the six reasons why its important:
1. Sample Efficiency
2. Task Adaptation
3. Transfer Learning
4. Human Guidance
5. Reducing Exploration Challenges
6. Addressing Distribution Shift


# Documentation

This model was fine-tuned using LoRA because I wanted the model's weights to be efficient in solving other types of Python problems(Ones that were not included in the training data). 
Setting lora_alpha to 16 suggests a relatively strong regularization. The specific value of this hyperparameter often requires experimentation and tuning to find the optimal balance between preventing overfitting and allowing the model to capture important patterns in the data.

The lora_dropout rate is 0.1, which dropped out 10% of the neurons randomly during training. This helped to prevent overfitting by introducing a level of randomness and redundancy in the network.
'r' in LoRa represents a rank that helps to decide the level of representation of the model in terms of a number of dimensions or features. This proved to be advantageous for tasks like fine-tuning, where we witness a reduction in the complexity of the model while preserving information is our paramount goal.

I am using bitsAndBytesConfig by loading the main model in 4 bits, as I wanted something to be trained quickly and be efficient rather than being super precise with its results. This tradeoff was needed due to the cluster that I am involved in working with.
There is a use of double quantization for the 4-bit representation. Quantization is a process of mapping a range of values to a smaller set of discrete values. "Double quantization" here implies an additional refinement or quantization step, possibly to improve the precision of the representation within the constraints of 4-bit storage.

The Datatype involved during the computational steps involving 4-bit representation is "float16". Using floating-point numbers allows for more precision in mathematical operations in comparision to integer representations.


### Lora Config
```
peft_config = LoraConfig(
      lora_alpha=16,
      lora_dropout=0.1,
      r=64,
      bias="none",
      task_type="CAUSAL_LM"
)
```

### BitsAndBytesConfig
```
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)
```




