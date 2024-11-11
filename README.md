# 🤖 Formal Language Automata

A Python implementation of Deterministic Finite Automata (DFA) and Non-deterministic Finite Automata (NFA) with visualization capabilities.

## ✨ Features

- 🔄 DFA & NFA Implementation
- 📊 Automaton Visualization
- 🔍 Word Simulation
- 💾 JSON Import/Export
- 🔄 NFA to DFA Conversion
- ⚡ DFA Minimalization
- 🧪 Comprehensive Test Suite

## 🛠️ Installation

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

### Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

### Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

### Install dependencies
uv pip sync uv.lock

## 📦 Dependencies

The project uses uv.lock for dependency management. Key dependencies include:

- networkx
- pydot  
- pytest
- loguru
- ipython

## 🧪 Testing

The project uses pytest for testing. Tests cover:
- Basic Automaton Operations
- DFA & NFA Simulation
- JSON Import/Export
- Automaton Visualization
- Word Permutation Generation
- NFA to DFA Conversion
- DFA Minimalization

### Running Tests

#### Run all tests
pytest tests.py -v

#### Run specific test class
pytest tests.py::TestDFA -v

#### Run with coverage report
pytest tests.py --cov=Automatons

## 📝 Usage Example

from Automatons import DFA, NFA

# Create a DFA
dfa = DFA()
dfa._name = "example_dfa"
dfa.add_state("q0", starting=True)
dfa.add_state("q1")
dfa.add_state("q2", accepting=True)
dfa.add_transition("q0", "q1", "a")
dfa.add_transition("q1", "q2", "b")

# Test words
dfa.simulate(["ab", "abc", "ba"])

# Visualize
dfa.visualize()  # Creates PNG in result_images/

# Export to JSON
dfa.export_to_json("jsons/example_dfa.json")

## 📁 Project Structure

. <br>
├── Automatons.py - Main implementation <br>
├── tests.py - Test suite <br>
├── jsons/ - JSON automata definitions <br>
├── result_images/ - Visualization outputs <br>
└── permutations/ - Generated word permutations <br>

## 🔧 Required Directories

The following directories should exist:

mkdir jsons result_images permutations

## 🎯 Features in Detail

### DFA (Deterministic Finite Automaton)
- Complete transition function
- Word simulation
- Minimalization
- Visualization

### NFA (Non-deterministic Finite Automaton)
- Multiple transitions per symbol
- Subset construction (NFA to DFA conversion)
- Word simulation
- Visualization

## 📄 License

MIT License

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.