from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from peft import  get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
import pandas as pd
from pathlib import Path
import os, sys
import logging
from peft import PeftModel
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_training_arguments(path, learning_rate=1e-5, epochs=10):
    training_args = TrainingArguments(
        output_dir=path, # Where the model predictions and checkpoints will be written
        use_cpu=True, # This is necessary for CPU clusters.
        auto_find_batch_size=True, # Find a suitable batch size that will fit into memory automatically
        learning_rate= learning_rate, # Higher learning rate than full Fine-Tuning
        num_train_epochs=epochs,
        remove_unused_columns=False        
    )
    return training_args


def create_trainer(model, training_args, train_dataset, eval_dataset):
    trainer = Trainer(
        model=model, # We pass in the PEFT version of the foundation model, bloomz-560M
        args=training_args, #The args for the training.
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False) # mlm=False indicates not to use masked language modeling
    )
    return trainer
    



def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, truncation=True)
    labels = tokenizer(
        targets, max_length=target_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

def convert_types(record):
    return {k: int(v) if isinstance(v, (np.int64, np.int32)) else v for k, v in record.items()}

def objective(var):
    NUM_VIRTUAL_TOKENS, NUM_EPOCHS = var[0], var[1]
    generation_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM, #This type indicates the model will generate text.
        prompt_tuning_init=PromptTuningInit.RANDOM,  #The added virtual tokens are initializad with random numbers
        num_virtual_tokens=NUM_VIRTUAL_TOKENS, #Number of virtual tokens to be added and trained.
        tokenizer_name_or_path=foundational_model #The pre-trained model.
    )
    peft_model = get_peft_model(foundational_model, generation_config)
    logger.info(f"trainable parameters = \n{peft_model.print_trainable_parameters()}")

    
    #train
    training_args_prompt = create_training_arguments(output_directory_prompt, 1e-5, NUM_EPOCHS)
    trainer = create_trainer(peft_model, training_args_prompt, train_dataset, eval_dataset)
    trainer.train()
    trainer.model.save_pretrained(output_directory_prompt)


    eval_results = trainer.evaluate()
    return -eval_results['eval_f1']  



if __name__ == "__main__":

    device="cuda"
    model_id = "mistralai/Mistral-7B-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    foundational_model = AutoModelForCausalLM.from_pretrained(model_id)
    
    script_dir = Path(os.getcwd()).resolve() 
    df_example=pd.read_csv(f"{script_dir}/df_prompt.csv")
    df_check=pd.read_csv(f"{script_dir}/df_recognize.csv")

    
    # создание папки 
    working_dir = "./"
    output_directory_prompt =  os.path.join(working_dir, "peft_outputs")
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    if not os.path.exists(output_directory_prompt):
        os.mkdir(output_directory_prompt)


    #data
    text_column = 'prompt'
    label_column = 'entity'
    dataset = load_dataset("csv", data_files=f"{script_dir}/df_perf.csv")
    dataset_eval = Dataset.from_pandas(df_check)
    target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in dataset_eval[label_column]])
    logger.info(f"target_max_length = {target_max_length}") 
    
    dataset_dict = DatasetDict({
        'train': dataset_eval,
        'eval': dataset['train']
    })

    
    logger.info(f"dataset = {dataset_dict}")
    processed_datasets = dataset_dict.map(
            preprocess_function,
            batched=True,
            num_proc=1,
            remove_columns=dataset_dict["train"].column_names,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )    
    
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["eval"]
    train_dataset = train_dataset.map(convert_types)
    eval_dataset = eval_dataset.map(convert_types)

    """check 1 epoch

    # training_args_prompt = create_training_arguments(output_directory_prompt, 1e-5, 1)
    # trainer = create_trainer(foundational_model, training_args_prompt, train_dataset, eval_dataset)
    # trainer.train()
    # trainer.model.save_pretrained(output_directory_prompt)


    # eval_results = trainer.evaluate()
    # logger.info(f"eval_results = {eval_results}")
    """   
 
    
   

    #scikit-optimize
    search_space = [
        Integer(5, 10), #25              
        Integer(2, 5)  #50                   
    ]

    # Run the optimization
    res = gp_minimize(objective, search_space, n_calls=10, random_state=42)

    # Best hyperparameters    
    logger.info("Best hyperparameters: ", res.x)
    logger.info(f"All info: {res}")