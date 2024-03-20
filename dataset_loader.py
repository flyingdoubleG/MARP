import csv
import json
import pickle
import os
import pandas as pd
from datasets import Dataset
from comparison_task import ComparisonTaskLLMBar


class DatasetLoader:
    """
    Loads a story/essay evaluation dataset, presented as a csv file. The first line of the CSV file are the column names. Then each subsequent line is a row of data.
    """
    def __init__(self, dataset_name, filepath, writers, load_from_pickle=True):
        self.filepath = filepath
        self.dataset_name = dataset_name
        self.writers = writers
        self.load_from_pickle = load_from_pickle
        self.save_dir_name = "dataset"
        if dataset_name == "hanna":
            self.num_prompts = 96
            self.num_human_evaluators = 3
            self.num_categories = 6
            self.process_data = self.process_data_hanna
            self.pickle_path = os.path.join(self.save_dir_name, "hanna_data.pkl")
        elif dataset_name == "meva":
            self.num_prompts = 200
            self.num_human_evaluators = 5
            self.num_categories = 1
            self.process_data = self.process_data_meva
            self.pickle_path = os.path.join(self.save_dir_name, "meva_data.pkl")
        elif dataset_name == "SummEval":
            self.num_prompts = 100
            self.num_human_evaluators = 3
            self.num_categories = 4
            self.process_data = self.process_data_SummEval
            self.pickle_path = os.path.join(self.save_dir_name, "SummEval_data.pkl")
        elif dataset_name == "llmbar":
            self.num_prompts = 419
            self.num_human_evaluators = 1
            self.num_categories = 1
            self.process_data = self.process_data_llmbar
            self.pickle_path = os.path.join(self.save_dir_name, "llmbar_data.pkl")
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
    @staticmethod
    def load_csv(filepath):
        """
        Load a csv file. The first line of the CSV file are the column names.
        Then each subsequent line is a row of data.
        """
        with open(filepath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            row_num = 0
            for row in reader:
                if row_num == 0:
                    column_names = row
                    data = []
                else:
                    data.append(row)
                row_num += 1

        return column_names, data
    
    def process_data_prefix(self):
        if self.load_from_pickle and os.path.exists(self.pickle_path):
            with open(self.pickle_path, 'rb') as f:
                prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories = pickle.load(f)

            print(f"{self.dataset_name} data loaded from local successfully.\n")

            return prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories
        else:
            return None

    def process_data_suffix(self, prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories):
        os.makedirs(self.save_dir_name, exist_ok=True)
        with open(self.pickle_path, 'wb') as f:
            pickle.dump((prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories), f)
        print(f"{self.dataset_name} data saved to local successfully.\n")

        return prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories
    
    def process_data_llmbar(self):
        if self.load_from_pickle and os.path.exists(self.pickle_path):
            with open(self.pickle_path, 'rb') as f:
                data = pickle.load(f)
            df = pd.DataFrame(data)
            dataset = Dataset.from_pandas(df)
            return data, df, dataset

        sub_datasets = ["Natural", "Adversarial/Neighbor", "Adversarial/GPTInst", "Adversarial/GPTOut", "Adversarial/Manual"]
        evaluators = ["GPT-4", "PaLM2"]
        strategies = ["Vanilla_NoRules", "Vanilla", "CoT", "Swap", "Swap_CoT", "Metrics", "Reference", "Metrics_Reference"]
        assert len(strategies) == 8

        top_dir = "LLMBar"
        sub_dataset_counter = {}
        comparison_tasks = []
        # num_null_cases = 0
        # num_single_null_cases = 0
        for sub_dataset in sub_datasets:
            second_dir = os.path.join(top_dir, sub_dataset, "evaluators")
            sub_dataset_counter[sub_dataset] = 0
            for evaluator in evaluators:
                third_dir = os.path.join(second_dir, evaluator)
                for strategy in strategies:
                    if strategy == "CoT" and evaluator == "PaLM2":
                        continue
                    filepath = os.path.join(third_dir, strategy, "result.json")
                    if not os.path.exists(filepath):
                        raise ValueError(f"File does not exist: {filepath}")
                    
                    num_null_cases = 0
                    num_single_null_cases = 0
                    with open(filepath, 'r') as file:
                        entries = json.load(file)
                        num_entries = len(entries)
                        if sub_dataset_counter[sub_dataset] == 0:
                            sub_dataset_counter[sub_dataset] = num_entries
                        elif sub_dataset_counter[sub_dataset] != num_entries:
                            raise ValueError(f"Number of entries in {filepath} is not equal to the number of entries in the other files.")
                        num_agreements = 0
                        for entry in entries:
                            instruction = entry["input"]
                            output_1 = entry["output_1"]
                            output_2 = entry["output_2"]
                            # The human label is 0 if the first output is correct, and 1 if the second output is correct.
                            human_label = entry["label"] - 1
                            result_1 = entry["results"][-1]["swap = False"]["winner"]
                            result_2 = entry["results"][-1]["swap = True"]["winner"]

                            if result_1 is None or result_2 is None:
                                num_null_cases += 1
                                if result_1 is None and result_2 is not None:
                                    num_single_null_cases += 1
                                    llm_label = int(result_2) - 1
                                    swap_equal = False
                                elif result_1 is not None and result_2 is None:
                                    num_single_null_cases += 1
                                    llm_label = int(result_1) - 1
                                    swap_equal = False
                                else:
                                    llm_label = 0
                                    swap_equal = False
                            else:
                                result_1 = int(result_1)
                                result_2 = int(result_2)
                                llm_label = result_1 - 1
                                swap_equal = (result_1 == result_2)

                            assert human_label in {0, 1}
                            assert llm_label in {0, 1}
                            num_agreements += swap_equal

                            evaluator_name = evaluator + "@" + strategy
                            if human_label == 0:
                                comparison_task = ComparisonTaskLLMBar(evaluator_name, human_label, llm_label, "correct", "incorrect", 
                                instruction, output_1, output_2, swap_equal, sub_dataset)
                            elif human_label == 1:
                                comparison_task = ComparisonTaskLLMBar(evaluator_name, 1-human_label, 1-llm_label, "correct", "incorrect", 
                                instruction, output_2, output_1, swap_equal, sub_dataset)
                            else:
                                raise ValueError(f"Invalid human label: {human_label}")
                            comparison_tasks.append(comparison_task)
   
                        print(f"Swap agreement rate for {filepath}: {num_agreements/num_entries:.4f}")
                        if num_null_cases > 0:
                            print("=====================================================")
                            print(f"Number of null cases: {num_null_cases}")
                            print(f"Number of single null cases: {num_single_null_cases}")
                            print("=====================================================")

            print(f"Number of entries in {sub_dataset}: {sub_dataset_counter[sub_dataset]}\n")

        assert len(comparison_tasks) == 419 * (8 + 7) == 6285
        data = [{
            'task_id': task.task_id,
            'worker_id': task.worker_id,
            'human_label': task.human_label,
            'llm_label': task.llm_label,
            'generator_1': task.generator_1,
            'generator_2': task.generator_2,
            'instruction': task.instruction,
            'output_1': task.output_1,
            'output_2': task.output_2,
            'sub_dataset': task.sub_dataset,
            'swap_equal': task.swap_equal,
        } for task in comparison_tasks]

        os.makedirs(self.save_dir_name, exist_ok=True)
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"{self.dataset_name} data saved to local successfully.\n")

        df = pd.DataFrame(data)
        dataset = Dataset.from_pandas(df)
        return data, df, dataset


    def process_data_hanna(self):
        """
        Load the dataset. The first line of the CSV file are the column names.
        Then each subsequent line is a row of data.
        """
        prefix_result = self.process_data_prefix()
        if prefix_result is not None:
            return prefix_result
        
        column_names, data = DatasetLoader.load_csv(self.filepath)
        writers = self.writers

        idx2Prompt = {}
        prompt2Idx = {}
        prompt2Scores = {}

        # Helper dictionary to ensure each writer version of a story is evaluated {self.num_human_evaluators} times.
        prompt2AddCount = {}
        prompt2Stories = {}

        idx = 0
        for row in data:
            prompt = row[1].strip()
            if prompt not in prompt2Idx:
                prompt2Idx[prompt] = idx
                idx2Prompt[idx] = prompt
                idx += 1
                prompt2Scores[prompt] = {}
                prompt2AddCount[prompt] = {}
                prompt2Stories[prompt] = {}
                for writer in writers:
                    prompt2Scores[prompt][writer] = 0
                    prompt2AddCount[prompt][writer] = 0

            writer = row[4].strip()
            story = row[3].strip()

            relevance = int(row[5])
            coherence = int(row[6])
            empathy = int(row[7])
            surprise = int(row[8])
            engagement = int(row[9])
            complexity = int(row[10])
            score = relevance + coherence + empathy + surprise + engagement + complexity


            prompt2Scores[prompt][writer] += score
            prompt2AddCount[prompt][writer] += 1
            prompt2Stories[prompt][writer] = story
        
        for value in prompt2AddCount.values():
            for writer in writers:
                assert value[writer] == self.num_human_evaluators
        
        assert len(prompt2Idx) == len(idx2Prompt) == len(prompt2Scores) == len(prompt2AddCount) == self.num_prompts

        return self.process_data_suffix(prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories)
    
    def process_data_meva(self):
        prefix_result = self.process_data_prefix()
        if prefix_result is not None:
            return prefix_result

        writers = self.writers
        assert len(writers) == 5

        idx2Prompt = {}
        prompt2Idx = {}
        prompt2Scores = {}
        prompt2Stories = {}

        # Helper dictionary to ensure each writer only writes for a prompt once.
        prompt2AddCount = {}
        prompt2Stories = {}

        with open(self.filepath, 'r') as file:
            data = json.load(file)

        idx = 0
        for key in data:
            entry = data[key]
            prompt = entry["prompt"].strip()
            if prompt not in prompt2Idx:
                prompt2Idx[prompt] = idx
                idx2Prompt[idx] = prompt
                idx += 1
                prompt2Scores[prompt] = {}
                prompt2AddCount[prompt] = {}
                prompt2Stories[prompt] = {}
                for writer in writers:
                    prompt2Scores[prompt][writer] = 0
                    prompt2AddCount[prompt][writer] = 0
            else:
                raise ValueError(f"Prompt {prompt} already exists in the dataset.")

            gen = entry["gen"]
            for writer in writers:
                writer_data = gen[writer]
                writer_story = writer_data["text"]
                writer_score = sum(writer_data["score"])
                
                prompt2Scores[prompt][writer] += writer_score
                prompt2AddCount[prompt][writer] += 1
                prompt2Stories[prompt][writer] = writer_story.strip()
        
        assert idx == self.num_prompts == 200
        for value in prompt2AddCount.values():
            for writer in writers:
                assert value[writer] == 1
        
        assert len(prompt2Idx) == len(idx2Prompt) == len(prompt2Scores) == len(prompt2AddCount) == self.num_prompts

        return self.process_data_suffix(prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories)
    
    def process_data_SummEval(self):
        prefix_result = self.process_data_prefix()
        if prefix_result is not None:
            return prefix_result

        writers = self.writers
        writer_map = {"M0": "LEAD-3", "M1": "NEUSUM", "M2": "BanditSum", "M5": "RNES",
                      "M8": "Point Generator", "M9": "Fast-abs-rl", "M10": "Bottom-Up",
                      "M11": "Improve-abs", "M12": "Unified-ext-abs", "M13": "ROUGESal", "M14": "Multi-task", "M15": "Closed book decoder", "M17": "T5",
                      "M20": "GPT-2", "M22": "BART", "M23": "Pegasus"}
        assert len(writers) == 16

        idx2Prompt = {}
        prompt2Idx = {}
        prompt2Scores = {}
        prompt2Stories = {}

        # Helper dictionary to ensure each writer only writes for a prompt once.
        prompt2AddCount = {}
        prompt2Stories = {}

        article_ids = set()

        with open(self.filepath, 'r') as file:
            # idx records the largest prompt index so far
            idx = 0
            for line in file:
                entry = json.loads(line)
                
                article_id = entry["id"].strip()
                article_ids.add(article_id)

                prompt = entry["text"].strip()
                writer_story = entry["decoded"].strip()

                expert_annotations = entry["expert_annotations"]

                # Records the total scores for the current summary
                total_scores = {'coherence': 0, 'consistency': 0, 'fluency': 0, 'relevance': 0}

                num_experts = 0
                for annotation in expert_annotations:
                    num_experts += 1
                    total_scores['coherence'] += annotation['coherence']
                    total_scores['consistency'] += annotation['consistency']
                    total_scores['fluency'] += annotation['fluency']
                    total_scores['relevance'] += annotation['relevance']

                assert num_experts == self.num_human_evaluators
                writer_score = sum(total_scores.values())

                if prompt not in prompt2Idx:
                    prompt2Idx[prompt] = idx
                    idx2Prompt[idx] = prompt
                    idx += 1
                    prompt2Scores[prompt] = {}
                    prompt2AddCount[prompt] = {}
                    prompt2Stories[prompt] = {}
                    for writer in writers:
                        prompt2Scores[prompt][writer] = 0
                        prompt2AddCount[prompt][writer] = 0
                elif article_id not in article_ids:
                    raise ValueError(f"Article id: {article_id} exists in the dataset but article: {prompt} does not exist in the dataset. The model id: {entry['model_id']}.")
                
                # Make sure curr_writer is not the same as the "writer" in the
                # previous for loop.
                curr_writer = writer_map[entry["model_id"].strip()]
                    
                prompt2Scores[prompt][curr_writer] += writer_score
                prompt2AddCount[prompt][curr_writer] += 1
                prompt2Stories[prompt][curr_writer] = writer_story
        
        assert idx == self.num_prompts == 100
        for value in prompt2AddCount.values():
            for writer in writers:
                assert value[writer] == 1
        
        assert len(prompt2Idx) == len(idx2Prompt) == len(prompt2Scores) == len(prompt2AddCount) == self.num_prompts

        return self.process_data_suffix(prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories)

    @staticmethod
    def splitTrainTest(prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories, num_train):
        """
        Splits the data into train and test sets.
        """
        trainPrompt2Idx = {}
        trainIdx2Prompt = {}
        trainPrompt2Scores = {}
        trainPrompt2Stories = {}

        testPrompt2Idx = {}
        testIdx2Prompt = {}
        testPrompt2Scores = {}
        testPrompt2Stories = {}

        for prompt in prompt2Idx:
            idx = prompt2Idx[prompt]
            if idx < num_train:
                trainPrompt2Idx[prompt] = idx
                trainIdx2Prompt[idx] = prompt
                trainPrompt2Scores[prompt] = prompt2Scores[prompt]
                trainPrompt2Stories[prompt] = prompt2Stories[prompt]
            else:
                testPrompt2Idx[prompt] = idx
                testIdx2Prompt[idx] = prompt
                testPrompt2Scores[prompt] = prompt2Scores[prompt]
                testPrompt2Stories[prompt] = prompt2Stories[prompt]

        assert len(trainPrompt2Idx) == len(trainIdx2Prompt) == len(trainPrompt2Scores) == len(trainPrompt2Stories) == num_train
        assert len(testPrompt2Idx) == len(testIdx2Prompt) == len(testPrompt2Scores) == len(testPrompt2Stories) == len(prompt2Idx) - num_train

        return [trainPrompt2Idx, trainIdx2Prompt, trainPrompt2Scores, trainPrompt2Stories], [testPrompt2Idx, testIdx2Prompt, testPrompt2Scores, testPrompt2Stories]
    
class StoryAnnotation:
    def __init__(self, premise, generator, story, human_score):
        self.premise = premise
        self.generator = generator
        self.story = story
        self.human_score = human_score


def push_data_to_hub(prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories, writers, num_annotators, hub_url):

    story_annotations = []
    for prompt in prompt2Idx:
        for writer in writers:
            story = prompt2Stories[prompt][writer]
            score = prompt2Scores[prompt][writer] / num_annotators
            story_annotation = StoryAnnotation(prompt, writer, story, score)
            story_annotations.append(story_annotation)
    
    data = [{
        'input': story_annotation.premise,
        'generator': story_annotation.generator,
        'output': story_annotation.story,
        'human_score': story_annotation.human_score
    } for story_annotation in story_annotations]

    df = pd.DataFrame(data)

    try:
        if hub_url is not None:
            hf_write_token = os.getenv('HUGGINGFACE_WRITE_TOKEN')
            dataset = Dataset.from_pandas(df)
            dataset.push_to_hub(repo_id=hub_url, token=hf_write_token)
    except Exception as e:
        print(f"Error pushing to hub: {e}")
    
    return df

if __name__ == "__main__":
    loader = DatasetLoader("llmbar", None, None, load_from_pickle=False)
    data, df, dataset = loader.process_data()

    # hf_write_token = os.getenv('HUGGINGFACE_WRITE_TOKEN')
    # dataset.push_to_hub(repo_id='llm-aes/LLMBar_GPT4_PaLM2', token=hf_write_token)

    evaluator_set = set()
    for entry in data:
        evaluator_set.add(entry['worker_id'])
    for evaluator in evaluator_set:
        print(evaluator)