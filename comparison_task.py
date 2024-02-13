import json
from datasets import load_dataset, Dataset
import pandas as pd
import os
from collections import Counter

class ComparisonTask():
    current_task_id = 0

    def __init__(self, model_name: str, human_eval: int | str, llm_eval: int | str, writer1: str, writer2: str, prompt: str):
        ComparisonTask.current_task_id += 1
        self.task_id = f"t_{ComparisonTask.current_task_id}"
        self.worker_id = f"w_{model_name}"

        if human_eval == "win":
            self.human_label = 0
        elif human_eval == "lose":
            self.human_label = 1
        elif -2 <= human_eval <= 1:
            self.human_label = human_eval
        else:
            raise ValueError(f"Invalid human evaluation label: {human_eval}")
        
        if llm_eval == "win":
            self.llm_label = 0
        elif llm_eval == "lose":
            self.llm_label = 1
        elif -2 <= llm_eval <= 1:
            self.llm_label = llm_eval
        else:
            raise ValueError(f"Invalid LLM evaluation label: {llm_eval}")
        
        self.generator_1 = writer1
        self.generator_2 = writer2
        self.premise = prompt

class ComparisonTaskPandaLM(ComparisonTask):
    def __init__(self, model_name: str, human_eval: int, llm_eval: int, writer1: str, writer2: str, instruction: str, input: str, output_1: str, output_2: str):
        super().__init__(model_name, human_eval, llm_eval, writer1, writer2, prompt=None)

        self.instruction = instruction
        self.input = input
        self.output_1 = output_1
        self.output_2 = output_2


if __name__ == "__main__":
    hf_write_token = os.getenv('HUGGINGFACE_WRITE_TOKEN')
    dataset_gpt = load_dataset("llm-aes/pandalm-annotated")
    df_gpt = pd.DataFrame(dataset_gpt['train'])

    comparison_tasks = []
    for _, row in df_gpt.iterrows():
        instruction = row['instruction']
        input = row['input_x']
        output_1 = row['output_1']
        output_2 = row['output_2']
        model_name = row['annotator']
        llm_eval = row['preference'] - 1

        annotator1 = row['annotator1']
        annotator2 = row['annotator2']
        annotator3 = row['annotator3']
        counter = Counter([annotator1, annotator2, annotator3])
        human_eval, _ = counter.most_common(1)[0]
        human_eval -= 1

        writer1 = row['generator_1']
        writer2 = row['generator_2']

        comparison_task = ComparisonTaskPandaLM(model_name, human_eval, llm_eval, writer1, writer2, instruction, input, output_1, output_2)
        comparison_tasks.append(comparison_task)

    path = "jsons/gemini_pandalm.json"
    with open(path, 'r') as file:
        data_gemini = json.load(file)
    
    num_not_found = 0
    for item in data_gemini:
        instruction = item['instruction']
        input = item['input']
        output_1 = item['output_1']
        output_2 = item['output_2']

        model_name = item['annotator']
        llm_eval = item['preference'] - 1

        found_task = False
        for comparison_task in comparison_tasks:
            if (comparison_task.instruction == instruction) and (comparison_task.input == input) and (comparison_task.output_1 == output_1) and (comparison_task.output_2 == output_2):
                generator1 = comparison_task.generator_1
                generator2 = comparison_task.generator_2
                human_eval = comparison_task.human_label
                found_task = True
                break

        if not found_task:
            num_not_found += 1
            continue
        
        task = ComparisonTaskPandaLM(model_name, human_eval, llm_eval, generator1, generator2, instruction, input, output_1, output_2)
    
        comparison_tasks.append(task)
    
    print(f"Number of gemini tasks not found in gpt tasks: {num_not_found}")
    
    filtered_comparison_tasks = []
    for comparison_task in comparison_tasks:
        if (comparison_task.llm_label >= 0) and (comparison_task.human_label >= 0):
            filtered_comparison_tasks.append(comparison_task)
    
    data = [{
            'task_id': task.task_id,
            'worker_id': task.worker_id,
            'human_label': task.human_label,
            'llm_label': task.llm_label,
            'generator_1': task.generator_1,
            'generator_2': task.generator_2,
            'instruction': task.instruction,
            'input': task.input,
            'output_1': task.output_1,
            'output_2': task.output_2
        } for task in filtered_comparison_tasks]
    
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    # dataset.push_to_hub(repo_id='llm-aes/pandalm-annotated-full', token=hf_write_token)
    dataset.push_to_hub(repo_id='llm-aes/some_dataset', token=hf_write_token)
