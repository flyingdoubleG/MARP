import csv
import json
import pickle
import os
import pandas as pd
from datasets import Dataset


class DatasetLoader:
    """
    Loads a story/essay evaluation dataset, presented as a csv file. The first line of the CSV file are the column names. Then each subsequent line is a row of data.
    """
    def __init__(self, dataset_name, filepath, writers, load_from_pickle=True):
        self.filepath = filepath
        self.dataset_name = dataset_name
        self.writers = writers
        self.load_from_pickle = load_from_pickle
        if dataset_name == "hanna":
            self.num_prompts = 96
            self.num_human_evaluators = 3
            self.num_categories = 6
            self.process_data = self.process_data_hanna
        elif dataset_name == "meva":
            self.num_prompts = 200
            self.num_human_evaluators = 5
            self.num_categories = 1
            self.process_data = self.process_data_meva
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

    def process_data_hanna(self):
        """
        Load the dataset. The first line of the CSV file are the column names.
        Then each subsequent line is a row of data.
        """
        pickle_path = "data/hanna_data.pkl"
        if self.load_from_pickle and os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories = pickle.load(f)

            print("hanna data loaded from local successfully.\n")

            return prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories

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

        directory = "data/"
        os.makedirs(directory, exist_ok=True)
        with open('data/hanna_data.pkl', 'wb') as f:
            pickle.dump((prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories), f)
        print("hanna data saved to local successfully.\n")

        return prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories
    
    def process_data_meva(self):
        pickle_path = "data/meva_data.pkl"
        if self.load_from_pickle and os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as f:
                prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories = pickle.load(f)

            print("meva data loaded from local successfully.\n")

            return prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories

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

        directory = "data/"
        os.makedirs(directory, exist_ok=True)
        with open('data/meva_data.pkl', 'wb') as f:
            pickle.dump((prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories), f)
        print("meva data saved to local successfully.\n")

        return prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories
        
    
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
        'premise': story_annotation.premise,
        'generator': story_annotation.generator,
        'story': story_annotation.story,
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
    writers = ["gpt", "plan_write", "s2s", "gpt_kg", "fusion"]
    path = "meva/mans_wp.json"
    loader = DatasetLoader("meva", path, writers)
    prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories = loader.process_data()
    print(len(prompt2Idx), len(idx2Prompt), len(prompt2Scores), len(prompt2Stories))

    # df = push_data_to_hub(prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories, writers, 5, "llm-aes/meva_original")

    # print(df)