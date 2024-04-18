import csv
import json
import pickle
import os
import pandas as pd
from datasets import Dataset
from comparison_task import ComparisonTaskLLMBar
from datasets import load_dataset


class DataChecker:
    def __init__(self, url, dataset_name):
        self.url = url
        self.data = self.dataset_to_list_of_dicts()
        self.dataset_name = dataset_name
        self.update_params()

    def update_params(self):
        if self.dataset_name == "meva":
            self.num_inputs = 200
            self.num_generators = 5
            self.num_evaluators = 6
            self.num_keys = 200 * 10    # 2000

            self.generators = ["gpt", "plan_write", "s2s", "gpt_kg", "fusion"]

            self.key1 = 'premise'
            self.key2 = 'generator_1'
            self.key3 = 'generator_2'
        elif self.dataset_name == "hanna":
            self.num_inputs = 96
            self.num_generators = 11
            self.num_evaluators = 6
            self.num_keys = 96 * 55     # 5280

            self.generators = ["Human", "BertGeneration", "CTRL", "GPT", "GPT-2 (tag)", "GPT-2", "RoBERTa", "XLNet", "Fusion", "HINT", "TD-VAE"]

            self.key1 = 'premise'
            self.key2 = 'generator_1'
            self.key3 = 'generator_2'
        elif self.dataset_name == "SummEval":
            self.num_inputs = 100
            self.num_generators = 16
            self.num_evaluators = 4
            self.num_keys = 100 * 120   # 12000

            self.generators = ["LEAD-3", "NEUSUM", "BanditSum", "RNES", "Point Generator", "Fast-abs-rl", "Bottom-Up", "Improve-abs", "Unified-ext-abs", "ROUGESal", "Multi-task", "Closed book decoder", "T5", "GPT-2", "BART", "Pegasus"]

            self.key1 = 'premise'
            self.key2 = 'generator_1'
            self.key3 = 'generator_2'

        elif self.dataset_name == "SummEval_gpt2":
            self.num_inputs = 100
            self.num_generators = 16
            self.num_evaluators = 6
            self.num_keys = 100 * 15   # 1500

            self.generators = ["LEAD-3", "NEUSUM", "BanditSum", "RNES", "Point Generator", "Fast-abs-rl", "Bottom-Up", "Improve-abs", "Unified-ext-abs", "ROUGESal", "Multi-task", "Closed book decoder", "T5", "GPT-2", "BART", "Pegasus"]

            self.key1 = 'premise'
            self.key2 = 'generator_1'
            self.key3 = 'generator_2'

        elif self.dataset_name == "llmbar":
            self.num_inputs = 418
            self.num_generators = 2
            self.num_evaluators = 15
            self.num_keys = 419
            self.generators = ['correct', 'incorrect']
            self.key1 = 'instruction'
            self.key2 = 'output_1'
            self.key3 = 'output_2'
        
        elif self.dataset_name in {"FairEval", "LLMEval^2", "MT-Bench"}:
            if self.dataset_name == "FairEval":
                self.num_inputs = 66
            elif self.dataset_name == "LLMEval^2":
                self.num_inputs = 200
            else:
                self.num_inputs = 75

            self.num_generators = 2
            self.num_evaluators = 8

            if self.dataset_name == "FairEval":
                self.num_keys = 66
            elif self.dataset_name == "LLMEval^2":
                self.num_keys = 200
            else:
                self.num_keys = 195
            
            self.generators = ['correct', 'incorrect']
            self.key1 = 'instruction'
            self.key2 = 'output_1'
            self.key3 = 'output_2'

        elif self.dataset_name == "PandaLM":
            self.num_inputs = 158
            self.num_generators = 5
            self.num_evaluators = 8
            self.evaluators = ['gpt4_2', 'chatgpt_1', 'chatgpt_2', 'chatgpt_3', 'gemini_1', 'gemini_2', 'gemini_3', 'gemini_4']
            self.num_keys = 894
            self.generators = ['bloom-7b', 'cerebras-gpt-6.7B', 'llama-7b', 'opt-7b', 'pythia-6.9b']
            self.key1 = 'instruction'
            self.key2 = 'input'
            self.key3 = 'generator_1'
            self.key4 = 'generator_2'

        else:
            raise ValueError(f"Dataset name {self.dataset_name} not recognized.")

    def dataset_to_list_of_dicts(self):
        """
        Converts a Hugging Face dataset into a list of dictionaries, where each dictionary represents a row.

        Args:
        url (str): The URL or path of the Hugging Face dataset.

        Returns:
        List[Dict]: A list of dictionaries, where each dictionary corresponds to a row in the dataset.
        """
        # Load the dataset
        dataset = load_dataset(self.url)

        # Assume 'train' split is available, modify as needed for other splits
        if 'train' in dataset:
            data_split = 'train'
        else:
            # Default to the first available split if 'train' is not present
            data_split = list(dataset.keys())[0]

        # Convert to a dictionary of lists
        data_dict = dataset[data_split].to_dict()  

        # Create a list of dictionaries, one for each row
        num_rows = dataset[data_split].num_rows
        rows_as_dicts = [{column: data_dict[column][i] for column in data_dict} for i in range(num_rows)]

        return rows_as_dicts
    
    def summarize(self):
        evaluator_set = set()
        task_set = set()
        generator1_set = set()
        generator2_set = set()
        key1_set = set()
        key_set = set()
        # This set is used to check alignment between the task and the keys
        key_task_set = set()
        for i, row in enumerate(self.data):
            evaluator_set.add(row['worker'])
            task_set.add(int(row['task'][2:]))
            generator1_set.add(row['generator_1'])
            generator2_set.add(row['generator_2'])
            key1_set.add(row[self.key1])

            if self.dataset_name != 'PandaLM':
                key_set.add((row[self.key1], row[self.key2], row[self.key3]))
                key_task_set.add((row['task'], row[self.key1], row[self.key2], row[self.key3]))
            else:
                key_set.add((row[self.key1], row[self.key2], row[self.key3], row[self.key4]))
                key_task_set.add((row['task'], row[self.key1], row[self.key2], row[self.key3], row[self.key4]))
        
        generator_set = generator1_set.union(generator2_set)
        task_list = list(task_set)
        
        assert len(key1_set) == self.num_inputs
        assert len(key_set) == len(task_set) == len(key_task_set) == self.num_keys
        assert len(evaluator_set) == self.num_evaluators

        if self.dataset_name == "SummEval_gpt2":
            assert len(generator_set) == 16
            assert len(generator1_set) == 1
            assert len(generator2_set) == 15
        else:
            assert len(generator_set) == len(generator1_set) + 1 == len(generator2_set) + 1 == self.num_generators

        print(f"Dataset name: {self.dataset_name}")
        print(f"Number of inputs: {self.num_inputs}")
        print(f"Number of generators: {self.num_generators}")
        print(generator_set)
        print()
        print(f"Number of evaluators: {self.num_evaluators}")
        print(evaluator_set)
        print()
        print(f"Number of tasks: {self.num_keys}")
        print(f"Min task: {min(task_list)}, Max task: {max(task_list)}")

    def find_repeated_instructions(self):
        key_task_set = set()
        for i, row in enumerate(self.data):
            key_task_set.add((row['task'], row[self.key1], row[self.key2], row[self.key3]))
        
        repeated_set1 = set()
        repeated_set2 = set()
        for key1 in key_task_set:
            for key2 in key_task_set:
                if key1[1] == key2[1]:
                    if (key1[2] != key2[2]) or (key1[3] != key2[3]):
                        repeated_set1.add(key1)
                        repeated_set2.add(key2)
        
        print("Repeated Set 1")
        for key in repeated_set1:
            print(key)
            print()
        print("=============================================")
        print("Repeated Set 2")
        for key in repeated_set2:
            print(key)
            print()

    def find_repeated_keys(self):
        repeat_counter = {}
        key_task_set = set()
        for i, row in enumerate(self.data):
            key = (row['task'], row[self.key1], row[self.key2], row[self.key3])
            if key in repeat_counter:
                repeat_counter[key] += 1
            else:
                repeat_counter[key] = 1
            key_task_set.add((row['task'], row[self.key1], row[self.key2], row[self.key3]))
        
        key_task_list = list(key_task_set)
        idx = 0
        for key in key_task_list:
            if repeat_counter[key] != 8:
                idx += 1
                print("=============================================")
                print(f"The {idx}th key is repeated {repeat_counter[key]} times")
                print("=============================================")
                print(f"task: {key[0]}")
                print(f"instruction: {key[1]}")
                print(f"output1: {key[2]}")
                print(f"output2: {key[3]}")
                print()
        print()
        print("=============================================")
        print(f"There are {idx} repeated keys in total")

    def remove_repeated_keys(self, url=None):
        key_set = set()
        new_data = []
        for i, row in enumerate(self.data):
            key = (row['task'], row[self.key1], row[self.key2], row[self.key3], row['worker'])
            if key not in key_set:
                key_set.add(key)
                new_data.append(row)

        self.data = new_data
        self.summarize()

        if url is None:
            return
        
        data = [{
            'task': row['task'],
            'worker': row['worker'],
            'human_label': row['human_label'],
            'llm_label': row['llm_label'],
            'generator_1': row['generator_1'],
            'generator_2': row['generator_2'],
            'instruction': row['instruction'],
            'output_1': row['output_1'],
            'output_2': row['output_2'],
            'sub_dataset': row['sub_dataset'],
            'swap_equal': row['swap_equal'],
        } for row in new_data]

        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        hf_write_token = os.getenv('HUGGINGFACE_WRITE_TOKEN')
        # dataset.push_to_hub(repo_id=self.url, token=hf_write_token)
        dataset.push_to_hub(repo_id=url, token=hf_write_token)

    def examine_PandaLM(self):
        key_set = set()
        key_evaluators_map = {}
        generator_task_counts = {}
        instruction_input_set = set()
        for i, row in enumerate(self.data):
            key = (row['task'], row['instruction'], row['input'], row['generator_1'], row['generator_2'])

            instruction_input_set.add((row['instruction'], row['input']))

            if key not in key_set:
                key_set.add(key)
                key_evaluators_map[key] = 1

                if (row['generator_1'], row['generator_2']) in generator_task_counts:
                    generator_task_counts[(row['generator_1'], row['generator_2'])] += 1
                else:
                    generator_task_counts[(row['generator_1'], row['generator_2'])] = 1

            else:
                key_evaluators_map[key] += 1

                
        for i, generator1 in enumerate(self.generators):
            for j, generator2 in enumerate(self.generators[i+1:]):
                print(f"Generator 1: {generator1}, Generator 2: {generator2}, Count: {generator_task_counts[(generator1, generator2)]}")

        print()
        print(f"Number of instruction-input pairs: {len(instruction_input_set)}")

        print()
        num_evaluators_hist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        for key in key_evaluators_map:
            num_evaluators_hist[key_evaluators_map[key]] += 1
        print(f"Number of evaluators per task histogram:\n{num_evaluators_hist}")
       
        
                
if __name__ == "__main__":
    # "FairEval", "LLMEval^2", "MT-Bench"
    url = 'llm-aes/pandalm-annotated-latest'
    # url = 'llm-aes/toy'
    dataset_name = 'PandaLM'
    data_checker = DataChecker(url, dataset_name)
    # data_checker.summarize()
    # data_checker.find_repeated_keys()
    # data_checker.remove_repeated_keys()
    data_checker.examine_PandaLM()
    

        
