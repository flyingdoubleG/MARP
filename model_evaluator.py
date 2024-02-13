import re
import numpy as np
import litellm
import time
import vertexai
import pandas as pd
from datasets import Dataset, load_dataset
import os

from comparison_task import ComparisonTask


from standard_prompts import *
from dataset_loader import DatasetLoader

litellm.vertex_project = "multi-agent-411823"
litellm.vertex_location = "us-central1"
# litellm.set_verbose=True

# from gemini import gemini_gen

# Helper function
def get_response(model, message, temperature=None, top_p=None):
    """
    Query the LLM model with a message and return the response.
    """
    response = litellm.completion(
        model=model,
        messages=[{"content": message, "role": "user"}],
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content


# Helper function
def extract_first_number(text):
        """
        Helper function to extract the first number encountered within a string.
        """
        match = re.search(r'\d+', text)
        if match:
            return int(match.group(0))
        else:
            return None


class ModelEvaluator():
    def __init__(self, model, dataset_name, filepath, num_prompts_eval=3, 
                 num_categories=1, bidir_eval=False, eval_rounds=1, verbose=False, query_mode="score only", temperature=None, top_p=None, special_mark=""):
        """
        num_prompts_eval: number of prompts to evaluate
        num_categories: number of scoring categories to evaluate

        bidir_eval: whether to evaluate both orders (directions) of presenting the two stories
        eval_rounds: number of rounds of evaluation for the same two stories

        num_rounds: number of rounds of evaluation for the same two stories (if bidi_eval is true, evaluations for both directions count together as one round).
        """
        self.model = model
        self.dataset_name = dataset_name
        self.filepath = filepath
        self.num_prompts_eval = num_prompts_eval
        self.num_categories = num_categories
        self.bidir_eval = bidir_eval
        self.eval_rounds = eval_rounds
        self.verbose = verbose
        self.query_mode = query_mode
        self.temperature = temperature
        self.top_p = top_p
        self.special_mark = special_mark

        if dataset_name == "hanna":
            self.evaluate_prefix = self.evaluate_prefix_hanna
            self.num_all_prompts = 96

            self.rate_double_essay_prompt_template = HANNA_RATE_DOUBLE_ESSAY_PROMPT_TEMPLATE
            
            self.analyze_rate_double_essay_prompt_template = HANNA_ANALYZE_RATE_DOUBLE_ESSAY_PROMPT_TEMPLATE
            
            self.rate_explain_double_essay_prompt_template = HANNA_RATE_EXPLAIN_DOUBLE_ESSAY_PROMPT_TEMPLATE

            self.rate_single_essay_prompt_template = HANNA_RATE_SINGLE_ESSAY_PROMPT_TEMPLATE

            self.analyze_rate_single_essay_prompt_template = HANNA_ANALYZE_RATE_SINGLE_ESSAY_PROMPT_TEMPLATE

            self.rate_explain_single_essay_prompt_template = HANNA_RATE_EXPLAIN_SINGLE_ESSAY_PROMPT_TEMPLATE

        elif dataset_name == "meva":
            self.evaluate_prefix = self.evaluate_prefix_meva
            self.num_all_prompts = 200

            self.rate_double_essay_prompt_template = MEVA_RATE_DOUBLE_ESSAY_PROMPT_TEMPLATE
            
            self.analyze_rate_double_essay_prompt_template = MEVA_ANALYZE_RATE_DOUBLE_ESSAY_PROMPT_TEMPLATE
            
            self.rate_explain_double_essay_prompt_template = MEVA_RATE_EXPLAIN_DOUBLE_ESSAY_PROMPT_TEMPLATE

            # self.rate_single_essay_prompt_template = MEVA_RATE_SINGLE_ESSAY_PROMPT_TEMPLATE

            # self.analyze_rate_single_essay_prompt_template = MEVA_ANALYZE_RATE_SINGLE_ESSAY_PROMPT_TEMPLATE

            # self.rate_explain_single_essay_prompt_template = MEVA_RATE_EXPLAIN_SINGLE_ESSAY_PROMPT_TEMPLATE
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
        
    def compute_model_eval_acc(self, scores1, scores2, llmScores1, llmScores2, writer1, writer2, prompts):
        """
        Computes the model evaluation accuracy given the human and LLM scores.
        """
        assert len(scores1) == len(scores2) == len(llmScores1) == len(llmScores2)

        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        llmScores1 = np.array(llmScores1)
        llmScores2 = np.array(llmScores2)

        scores1 = scores1.reshape([len(scores1), -1])
        scores2 = scores2.reshape([len(scores2), -1])
        llmScores1 = llmScores1.reshape([len(llmScores1), -1])
        llmScores2 = llmScores2.reshape([len(llmScores2), -1])

        assert len(scores1[0]) == len(scores2[0]) == len(llmScores1[0]) == len(llmScores2[0])

        scores1 = scores1.sum(axis=1)
        scores2 = scores2.sum(axis=1)
        llmScores1 = llmScores1.sum(axis=1)
        llmScores2 = llmScores2.sum(axis=1)

        acc = 0
        comparison_tasks = []

        evaluator_name = self.model + "-" + self.query_mode.replace(" ", "-") + self.special_mark
        for i in range(len(scores1)):
            if (scores1[i] > scores2[i]) and (llmScores1[i] > llmScores2[i]):
                acc += 1
                comparison_tasks.append(ComparisonTask(evaluator_name, "win", "win", writer1, writer2, prompts[i]))
            elif (scores1[i] <= scores2[i]) and (llmScores1[i] <= llmScores2[i]):
                acc += 1
                comparison_tasks.append(ComparisonTask(evaluator_name, "lose", "lose", writer1, writer2, prompts[i]))
            elif (scores1[i] > scores2[i]) and (llmScores1[i] <= llmScores2[i]):
                comparison_tasks.append(ComparisonTask(evaluator_name, "win", "lose", writer1, writer2, prompts[i]))
            else:
                comparison_tasks.append(ComparisonTask(evaluator_name, "lose", "win", writer1, writer2, prompts[i]))
        return acc / len(scores1), comparison_tasks
    
    def parse_double_scores_advanced(self, response):
        """
        Parse scores for two stories evaluated together.
        """
        try:
            # Splitting the response by newlines
            lines = response.strip().split('\n')
            
            # Placeholder for scores
            llm_attr_scores1, llm_attr_scores2 = [], []

            reached_ratings1 = False
            reached_ratings2 = False
            reached_explanations = False

            if self.query_mode  == "rate explain":
                for line in lines:
                    # This trio-if block is used to identify what is the nature of the current line
                    if not reached_ratings1:
                        if bool(re.search(r'##\s*Story\s*1\s*Ratings\s*##', line)):
                            reached_ratings1 = True
                            continue
                        else:
                            continue # We use continue here to emphasize the logic. The Story1 Ratings section has not been reached yet, so we can directly proceed to the next line.
                    elif not reached_ratings2:
                        if bool(re.search(r'##\s*Story\s*2\s*Ratings\s*##', line)):
                            reached_ratings2 = True
                            # If the current line is the start of the Story2 Ratings section, continue to the next line after recognizing it
                            continue
                        # Even if the current line is not the start of the Story2 Ratings section, we can still proceed for further processing.
                    elif not reached_explanations:
                        if bool(re.search(r'##\s*Explanations\s*##', line)):
                            reached_explanations = True
                            break
                    
                    assert reached_ratings1
                    if bool(re.search(r'\*[^*]+\*', line)):
                        score = extract_first_number(line)
                        if score is None:
                            continue
                        if reached_ratings2:
                            llm_attr_scores2.append(score)
                        elif reached_ratings1:
                            llm_attr_scores1.append(score)
            elif self.query_mode == "analyze rate":
                for line in lines:
                    # This trio-if block is used to identify what is the nature of the current line
                    if not reached_ratings1:
                        if bool(re.search(r'##\s*Story\s*1\s*Ratings\s*##', line)):
                            reached_ratings1 = True
                            continue
                        else:
                            continue # We use continue here to emphasize the logic. The Story1 Ratings section has not been reached yet, so we can directly proceed to the next line.
                    elif not reached_ratings2:
                        if bool(re.search(r'##\s*Story\s*2\s*Ratings\s*##', line)):
                            reached_ratings2 = True
                            # If the current line is the start of the Story2 Ratings section, continue to the next line after recognizing it
                            continue
                    
                    if bool(re.search(r'\*[^*]+\*', line)):
                        score = extract_first_number(line)
                        if score is None:
                            continue
                        if reached_ratings2:
                            llm_attr_scores2.append(score)
                        elif reached_ratings1:
                            llm_attr_scores1.append(score)
            else:
                raise InvalidParameterError(f"Invalid query mode: {self.query_mode}")

            
            if not (len(llm_attr_scores1) == len(llm_attr_scores2) == self.num_categories):
                raise ValueError(f"Incorrect number of scoring categories for the story.\nThe current response is:\n{response}")
            return llm_attr_scores1, llm_attr_scores2
        
        except ValueError as e:
            # Handling any potential errors
            raise ValueError(f"Error parsing scores: {e}")

    def parse_scores_advanced(self, response):
        """
        Parses single story scores from analyze-rate format.
        """
        try:
            # Splitting the response by newlines
            lines = response.strip().split('\n')
            
            # Placeholder for scores
            llm_score = []

            if self.query_mode == "analyze rate":
                reached_ratings = False
                for line in lines:
                    if not reached_ratings:
                        if bool(re.search(r'##\s*Ratings\s*##', line)):
                            reached_ratings = True
                            continue
                        else:
                            continue
                    
                    if bool(re.search(r'\*[^*]+\*', line)):
                        score = extract_first_number(line)
                        if score is None:
                            continue
                        else:
                            llm_score.append(score)
            
            elif self.query_mode == "rate explain":
                reached_ratings = False
                reached_explanations = False
                for line in lines:
                    if not reached_ratings:
                        if bool(re.search(r'##\s*Ratings\s*##', line)):
                            reached_ratings = True
                            continue
                        else:
                            continue
                    elif not reached_explanations:
                        if bool(re.search(r'##\s*Explanations\s*##', line)):
                            reached_explanations = True
                            break
                    
                    if bool(re.search(r'\*[^*]+\*', line)):
                        score = extract_first_number(line)
                        if score is None:
                            continue
                        else:
                            llm_score.append(score)

            else:
                raise InvalidParameterError(f"Invalid mode: {self.query_mode}")
            
            if len(llm_score) != self.num_categories:
                raise ValueError(f"Incorrect number of scoring categories for the story.\nThe current response is:\n{response}")
            return llm_score
        
        except ValueError as e:
            # Handling any potential errors
            raise ValueError(f"Error parsing scores: {e}")

    def parse_scores(self, response, double_story=True, keyword="Story"):
        """
        Parses the scores from the evaluator's response.
        The response should be in the following format (the categories are just examples):
        Story1
        Relevance: 3
        Coherence: 4
        Empathy: 5
        Surprise: 2
        Engagement: 4
        Complexity: 3
        Story2
        Relevance: 3
        Coherence: 4
        Empathy: 5
        Surprise: 2
        Engagement: 4
        Complexity: 3
        """
        try:
            # Splitting the response by newlines
            lines = response.strip().split('\n')
            
            # Placeholder for scores
            llmScore1, llmScore2 = [], []

            # Variable to keep track of which story's scores we are currently parsing
            current_story = 1
        
            for line in lines:
                if (keyword is not None) and (keyword in line):
                    story_num = extract_first_number(line)
                    if story_num == 1:
                        current_story = 1
                    elif story_num == 2:
                        current_story = 2
                    else:
                        raise ValueError(f"Invalid story number in response:\n{response}")
                else:
                    score = extract_first_number(line)
                    if score is None:
                        continue
                    if current_story == 1:
                        llmScore1.append(score)
                    else:
                        llmScore2.append(score)

            if double_story:
                if len(llmScore1) != self.num_categories or len(llmScore2) != self.num_categories:
                    raise ValueError(f"Incorrect number of scoring categories for one or both stories.\nThe current response is:\n{response}")

                return llmScore1, llmScore2
            else:
                if len(llmScore1) != self.num_categories:
                    raise ValueError(f"Incorrect number of scoring categories for the story.\nThe current response is:\n{response}")
                return llmScore1
        
        except Exception as e:
            # Handling any potential errors
            raise ValueError(f"Error parsing scores: {e}")

    def evaluate_stories(self, premises, stories1, stories2):
        """
        Using an LLM, evaluates two lists of stories generated by two different entities based on given premises.

        premises: list of premises
        stories1: list of articles generated by entity1
        stories2: list of articles generated by entity2
        scores1: list of scores for stories1. Each score is a list of integers for each category.
        scores2: list of scores for stories2. Each score is a list of integers for each category.

        return: list of LLM scores for stories1, list of LLM scores for stories2
        """
        assert len(premises) == len(stories1) == len(stories2)
        
        llmScores1, llmScores2 = [], []
        
        for i in range(len(stories1)):
            premise = premises[i]
            story1 = stories1[i]
            story2 = stories2[i]

            if self.query_mode == "score only":
                prompt = self.rate_double_essay_prompt_template.format(premise, story1, story2)
            elif self.query_mode == "analyze rate":
                prompt = self.analyze_rate_double_essay_prompt_template.format(premise, story1, story2)
            elif self.query_mode == "rate explain":
                prompt = self.rate_explain_double_essay_prompt_template.format(premise, story1, story2)
            else:
                raise ValueError(f"Invalid query mode: {self.query_mode}")

            repeat_query = True
            max_tries = 5
            while repeat_query and max_tries > 0:
                repeat_query = False

                # Try getting response from LLM
                try:
                    response = get_response(self.model, prompt, temperature=self.temperature, top_p=self.top_p)
                except Exception as e:
                    repeat_query = True
                    print(f"Error querying model. Error message: {e}\n Retrying...\n")
                    # time.sleep(90)
                    max_tries -= 1
                    if max_tries == 0: 
                        raise e
                    continue

                # Try parsing the LLM response
                try:
                    if self.query_mode == "score only":
                        llmScore1, llmScore2 = self.parse_scores(response, double_story=True, keyword="Story")
                    elif self.query_mode == "analyze rate":
                        llmScore1, llmScore2 = self.parse_double_scores_advanced(response)
                    elif self.query_mode == "rate explain":
                        llmScore1, llmScore2 = self.parse_double_scores_advanced(response)

                except Exception as e:
                    print(f"Error parsing scores. Error message: {e}\nRetrying...\n")
                    repeat_query = True
                    max_tries -= 1
                    if max_tries == 0:
                        llmScore1, llmScore2 = None, None

            if llmScore1 is None or llmScore2 is None:
                raise Exception(f"Error evaluating stories for premise {i+1}:\n\n{premise}")
            else:
                llmScores1.append(llmScore1)
                llmScores2.append(llmScore2)

        llmScores1 = np.array(llmScores1).sum(axis=1)
        llmScores2 = np.array(llmScores2).sum(axis=1)
        return llmScores1, llmScores2
    
    def evaluate_single_story(self, premise, story):
        if self.query_mode == "score only":
            prompt = self.rate_single_essay_prompt_template.format(premise, story)
        elif self.query_mode == "analyze rate":
            prompt = self.analyze_rate_single_essay_prompt_template.format(premise, story)
        elif self.query_mode == "rate explain":
            prompt = self.rate_explain_single_essay_prompt_template.format(premise, story)
        else:
            raise ValueError(f"Invalid query mode: {self.query_mode}")

        repeat_query = True
        max_tries = 5
        while repeat_query and max_tries > 0:
            repeat_query = False

            try:
                response = get_response(self.model, prompt, temperature=self.temperature, top_p=self.top_p)
            except Exception as e:
                repeat_query = True
                print(f"Error querying model. Error message: {e}\n Retrying...\n")
                # time.sleep(90)
                max_tries -= 1
                if max_tries == 0: 
                    raise e
                continue

            if self.verbose:
                print(f"\nResponse:\n{response}\n")

            try:
                if self.query_mode == "score only":
                    llmScore = self.parse_scores(response, double_story=False, keyword=None)
                else:
                    llmScore = self.parse_scores_advanced(response)
            except ValueError as e:
                repeat_query = True
                print(f"Error parsing scores. Error message: {e}\nRetrying...\n")
                max_tries -= 1
                if max_tries == 0: 
                    llmScore = None

        if llmScore is None:
            raise Exception(f"Error evaluating stories for premise:\n\n{premise}")
        else:
            if self.verbose:
                print(f"LLM score: {llmScore}\n")
            llmScore = sum(llmScore)
            return llmScore
            
    def evaluate_prefix_hanna(self):
        NUM_TRAIN = self.num_all_prompts
        assert self.num_prompts_eval <= NUM_TRAIN

        writers = ["Human", "BertGeneration", "CTRL", "GPT", "GPT-2 (tag)", "GPT-2", "RoBERTa", "XLNet", "Fusion", "HINT", "TD-VAE"]

        loader = DatasetLoader(self.dataset_name, self.filepath, writers)

        prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories = loader.process_data()

        train_set, test_set = loader.splitTrainTest(prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories, NUM_TRAIN)

        writers = ["BertGeneration", "CTRL", "GPT", "GPT-2 (tag)", "GPT-2", "RoBERTa", "XLNet", "Fusion", "HINT", "TD-VAE"]

        return writers, train_set, test_set
    
    def evaluate_prefix_meva(self):
        NUM_TRAIN = self.num_all_prompts
        assert self.num_prompts_eval <= NUM_TRAIN

        writers = ["gpt", "plan_write", "s2s", "gpt_kg", "fusion"]

        loader = DatasetLoader(self.dataset_name, self.filepath, writers)

        prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories = loader.process_data()

        train_set, test_set = loader.splitTrainTest(prompt2Idx, idx2Prompt, prompt2Scores, prompt2Stories, NUM_TRAIN)

        writers = ["gpt", "plan_write", "s2s", "gpt_kg", "fusion"]

        return writers, train_set, test_set

    def evaluate(self):
        writers, train_set, test_set = self.evaluate_prefix()
        trainPrompt2Idx, trainIdx2Prompt, trainPrompt2Scores, trainPrompt2Stories = train_set
        testPrompt2Idx, testIdx2Prompt, testPrompt2Scores, testPrompt2Stories = test_set

        comparison_tasks = []

        acc = 0
        acc_count = 0
        prompts = list(trainPrompt2Idx.keys())[:self.num_prompts_eval]
        
        for i in range(len(writers)):
            writer1 = writers[i]
            for j in range(i+1, len(writers)):
                writer2 = writers[j]
                premises = []
                scores1 = []
                scores2 = []
                stories1 = []
                stories2 = []

                # Initialize the premises, stories, and human scores into corresponding lists.
                for prompt in prompts:
                    premises.append(prompt)
                    stories1.append(trainPrompt2Stories[prompt][writer1])
                    stories2.append(trainPrompt2Stories[prompt][writer2])
                    scores1.append(trainPrompt2Scores[prompt][writer1])
                    scores2.append(trainPrompt2Scores[prompt][writer2])
                
                llmScores1 = np.zeros(self.num_prompts_eval)
                llmScores2 = np.zeros(self.num_prompts_eval)
                for _ in range(self.eval_rounds):
                    tmp_llmScores1, tmp_llmScores2 = self.evaluate_stories(premises, stories1, stories2)
                    llmScores1 += tmp_llmScores1
                    llmScores2 += tmp_llmScores2

                    # Bidirectional evaluation
                    if self.bidir_eval:
                        tmp_llmScores2, tmp_llmScores1 = self.evaluate_stories(premises, stories2, stories1)
                        llmScores2 += tmp_llmScores2
                        llmScores1 += tmp_llmScores1

                tmp_acc, tmp_comparison_tasks = self.compute_model_eval_acc(scores1, scores2, llmScores1, llmScores2, writer1, writer2, prompts)
                comparison_tasks.extend(tmp_comparison_tasks)
                acc += tmp_acc
                acc_count += 1
                print(f"Train Accuracy for {writer1} vs {writer2}: {tmp_acc}; cumulative accuracy: {acc / acc_count}")
                
        
        acc /= (len(writers) * (len(writers) - 1) / 2)
        print(f"\nOverall Train Accuracy: {acc}")
        assert acc_count == len(writers) * (len(writers) - 1) / 2
        return f"Overall Train Accuracy: {acc}", comparison_tasks

    @staticmethod
    def sort_model_by_score(overall_scores: dict) -> list:
        """
        overal_scores: a dictionary whose keys are model names (str) and values are the overall scores for each model.
        """
        # Sorting the dictionary by values (scores) in descending order
        sorted_scores = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)

        # sorted_scores is now a list of tuples sorted by the score
        return sorted_scores

    def evaluate_models(self):
        """
        Compare the overall story generation performance of different models (including possibly human) on a given number of prompts. The total human-evaluted scores for the stories generated by each model are averaged. The models are then ranked from the best to the worst based on the average scores.
        
        return: a list of (model, accuracy) tuples, sorted by accuracy from highest to lowest.
        """
        writers, train_set, test_set = self.evaluate_prefix()

        trainPrompt2Idx, trainIdx2Prompt, trainPrompt2Scores, trainPrompt2Stories = train_set
        testPrompt2Idx, testIdx2Prompt, testPrompt2Scores, testPrompt2Stories = test_set
        
        human_overall_scores = {writer: 0.0 for writer in writers}

        # Collect total human-evaluated scores over selected prompts for each writer model
        for writer in writers:
            for prompt in list(trainPrompt2Idx.keys())[:self.num_prompts_eval]:
                human_overall_scores[writer] += trainPrompt2Scores[prompt][writer]


        # Next, we will collect total scores evaluated by the scorer model over selected prompts for each writer model.
                
        # model_scores is a dictionary of dictionaries, where model_scores[prompt][writer] is the total model-evaluted score for the story generated by writer with respect to prompt.
        model_scores = {}
        for prompt in list(trainPrompt2Idx.keys())[:self.num_prompts_eval]:
            model_scores[prompt] = {writer: 0.0 for writer in writers}
        
        # Collect total scores evaluated by the scorer model over all selected prompts for each writer model.
        model_overall_scores = {writer: 0.0 for writer in writers}

        for prompt in list(trainPrompt2Idx.keys())[:self.num_prompts_eval]:
            for writer in writers:
                story = trainPrompt2Stories[prompt][writer]
                model_scores[prompt][writer] = self.evaluate_single_story(prompt, story)
                model_overall_scores[writer] += model_scores[prompt][writer]
                print(f"Model score for {writer} on prompt {trainPrompt2Idx[prompt]}: {model_scores[prompt][writer]}")

                # Hanlding the 60 quotas/min limit for gemini-pro
                # if self.model == "gemini-pro":
                #     time.sleep(0.0)

        # Compare writer models pair-wise to check whether the scorer model agrees with the human evaluation.
        cumu_acc = 0
        acc_count = 0
        for i in range(len(writers)):
            writer1 = writers[i]
            for j in range(i+1, len(writers)):
                writer2 = writers[j]

                acc = 0
                for prompt in list(trainPrompt2Idx.keys())[:self.num_prompts_eval]:
                    acc_count += 1

                    human_score1 = trainPrompt2Scores[prompt][writer1]
                    human_score2 = trainPrompt2Scores[prompt][writer2]
                    llm_score1 = model_scores[prompt][writer1]
                    llm_score2 = model_scores[prompt][writer2]

                    if (human_score1 > human_score2) and (llm_score1 > llm_score2):
                        acc += 1
                        cumu_acc += 1
                    elif (human_score1 <= human_score2) and (llm_score1 <= llm_score2):
                        acc += 1
                        cumu_acc += 1
                    
                print(f"Train Accuracy for {writer1} vs {writer2}: {acc / self.num_prompts_eval}; cumulative accuracy: {cumu_acc / acc_count}")
        
        assert acc_count == len(writers) * (len(writers) - 1) / 2 * self.num_prompts_eval
        print(f"\nOverall Train Accuracy: {cumu_acc / acc_count}\n\n")

        # Obtain the ranking of the writer models based on the human evaluation
        sorted_human_overall_scores = self.sort_model_by_score(human_overall_scores)
        sorted_model_overall_scores = self.sort_model_by_score(model_overall_scores)
        sorted_model_overall_scores = [(model, score*3) for model, score in 
                                       sorted_model_overall_scores]

        print("Writer model ranking based on human evaluation:")
        for i in range(len(sorted_human_overall_scores)):
            print(f"{i+1}. {sorted_human_overall_scores[i][0]}: {sorted_human_overall_scores[i][1]}")
        print("\n")

        print("Wrtier model ranking based on model evaluation:")
        for i in range(len(sorted_model_overall_scores)):
            print(f"{i+1}. {sorted_model_overall_scores[i][0]}: {sorted_model_overall_scores[i][1]}")

        return f"Overall Train Accuracy: {cumu_acc / acc_count}"
    
    def collect_data(self, hub_url: str = None):
        """
        Collect model evaluation data and human evaluation data for the selected number of prompts. Form these data into Panda dataframes.
        """
        acc_string, comparison_tasks = self.evaluate()

        data = [{
            'task_id': task.task_id,
            'worker_id': task.worker_id,
            'human_label': task.human_label,
            'llm_label': task.llm_label,
            'generator_1': task.generator_1,
            'generator_2': task.generator_2,
            'premise': task.premise
        } for task in comparison_tasks]

        df = pd.DataFrame(data)

        try:
            if hub_url is not None:
                hf_write_token = os.getenv('HUGGINGFACE_WRITE_TOKEN')
                dataset = Dataset.from_pandas(df)
                dataset.push_to_hub(repo_id=hub_url, token=hf_write_token)
        except Exception as e:
            print(f"Error pushing to hub: {e}")
        
        return df


class InvalidParameterError(Exception):
    """Exception raised for errors due to invalid parameter values."""
    
    def __init__(self, message="Invalid parameter value"):
        self.message = message
        super().__init__(self.message)