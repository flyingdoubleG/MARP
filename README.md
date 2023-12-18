<h1 align="center">  <span style="color:blue">MARP</span> </h1>

<h3 align="center">
    <p>Multi-Agent Story Generation</p>
</h3>

MARP is a Story Generation Framework based on multi-agent collaboration and role playing. Agents include a Global Designer, a Scene Designer, a Controller, an Environment Manager, a Writer, and several Player agents (the characters).

## Getting Started

### Installation
There is no particular requirements on the operating system. MARP is compatible with Windows, macOS, Linux operating systems.

Package Requirements:

- Python >= 3.7
- OpenAI API key (We use gpt-4-1106-preview as default LLM backend)

Create conda environment:
```bash
conda create --name marp python=3.7
conda activate marp
```

Install dependencies:
```bash
pip install -r requirements.txt
```

To use gpt-4-1106-preview as LLM backend, set your OpenAI API key:

- Windows:
```bash
set OPENAI_API_KEY=sk-xxxx
```

- MacOS:
```bash
export OPENAI_API_KEY=sk-xxxx
```

### Run code

```bash
python story_generation.py
```
The generated conversations will be printed in terminal and the story will be saved as txt files in storys/ directory.

### Code structure 
```
│   .gitignore
│   LICENSE
│   README.md
│   requirements.txt
│   story_generation.py  # specify environment and agents for story generation
├───marp
│   │   agent.py    # define agent class
│   │   arena.py    # define how agents run in environment
│   │   config.py
│   │   message.py   # define message system
│   │   utils.py
│   │   __init__.py
│   ├───backends
│   │       bard.py
│   │       base.py
│   │       openai.py  # query GPT-4 api
│   │       __init__.py
│   │
│   └───environments
│           base.py
│           story_environment.py  # environment for story generation
│           __init__.py
│
├───storys   # save generated stories and logs
│   │   story_20231126_213925.txt
│   │   ...
│   └───logs
│           log_20231217_104735.txt
│           ...
└───story_samples   # generated story for evaluation
        ibrusia_gpt.txt
        ibrusia_marp.txt
        ...
```

## Contact
If you have any questions or suggestions, feel free to open an issue or submit a pull request. We will provide timely feedback.

## Credits
This code is built on [ChatArena](https://github.com/Farama-Foundation/chatarena), a wonderful open-source framework providing multi-agent language game environments.

