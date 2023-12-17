<h1 align="center">  <span style="color:blue">MARP</span> </h1>

<h3 align="center">
    <p>Multi-Agent Story Generation</p>
</h3>

MARP is a Story Generation Framework based on multi-agent collaboration and role-playing. Agents include Controller, Global designer, Scene designer, Character, Environment Manager, Writer.

## Getting Started

### Installation

Requirements:

- Python >= 3.7
- OpenAI API key (WE use gpt-4-1106-preview as default LLM backend)

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

The generated log and story will be stored in "storys/" directory.

## Contact
If you have any questions or suggestions, feel free to open an issue or submit a pull request. We will provide timely feedback.

## Credits
This code is built on [ChatArena](https://github.com/Farama-Foundation/chatarena), a wonderful open-source framework providing multi-agent language game environments.

