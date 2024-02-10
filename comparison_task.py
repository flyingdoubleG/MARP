class ComparisonTask():
    current_task_id = 0

    def __init__(self, model_name: str, human_eval: int, llm_eval: int, writer1: str, writer2: str, prompt: str):
        ComparisonTask.current_task_id += 1
        self.task_id = f"t_{ComparisonTask.current_task_id}"
        self.worker_id = f"w_{model_name}"

        if human_eval == "win":
            self.human_label = 0
        elif human_eval == "lose":
            self.human_label = 1
        else:
            raise ValueError(f"Invalid human evaluation label: {human_eval}")
        
        if llm_eval == "win":
            self.llm_label = 0
        elif llm_eval == "lose":
            self.llm_label = 1
        else:
            raise ValueError(f"Invalid LLM evaluation label: {llm_eval}")
        
        self.generator_1 = writer1
        self.generator_2 = writer2
        self.premise = prompt

    