import networkx as nx
from IPython.display import display, Image
import pydot
import json
import os
from typing import Set, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
from collections import deque


class Automaton(ABC):
    def __init__(self, file: Optional[str] = None) -> None:
        """
        Initialize the Automaton with optional JSON file input.

        :param file: The path to a JSON file containing the Automaton definition.
        :type file: Optional[str]
        """
        self._Q: Set[str] = set()  # Set of states
        self._Sigma: Set[str] = set()  # Alphabet
        self._delta: Dict[Tuple[str, str], Set[str]] = {}  # Transition function
        self._q0: Optional[str] = None  # Initial state
        self._F: Set[str] = set()  # Set of final states
        self._G = nx.MultiDiGraph() # Graph representation of the Automaton
        self._test_words: List[str] = [] # Test words
        self._name: Optional[str] = None # Name of the Automaton
        self._file_name: Optional[str] = None # Name of the JSON file
        
        if file:
            self.import_from_json(file)
    
    @property
    def states(self) -> Set[str]:
        return self._Q
    
    @property
    def alphabet(self) -> Set[str]:
        return self._Sigma

    @property
    def start_state(self) -> Optional[str]:
        return self._q0

    @property
    def accept_states(self) -> Set[str]:
        return self._F

    @property
    def transitions(self) -> Dict[Tuple[str, str], Set[str]]:
        return self._delta

    def add_state(self, state: str, accepting: bool = False, starting: bool = False) -> None:
        """
        Add a state to the Automaton.

        :param state: The state to add.
        :type state: str
        :param accepting: Whether the state is an accepting state. Defaults to False.
        :type accepting: bool
        :param starting: Whether the state is the start state. Defaults to False.
        :type starting: bool
        """
        if state in self._Q:
            return

        self._Q.add(state)
        self._G.add_node(state, shape='circle')

        if accepting:
            self._F.add(state)
            self._G.nodes[state]['peripheries'] = '2'

        if starting:
            if self._q0 is not None:
                print(f"Warning: Automaton already has a start state '{self._q0}'. Ignoring start state flag for '{state}'.")
            else:
                self._q0 = state
                self._G.nodes[state]['style'] = 'filled'
                self._G.nodes[state]['fillcolor'] = 'lightgrey'
    
    def add_transition(self, from_state: str, to_state: str, symbol: str) -> None:
        """
        Add a transition from one state to another with a given symbol.

        :param from_state: The state from which the transition starts.
        :type from_state: str
        :param to_state: The state to which the transition goes.
        :type to_state: str
        :param symbol: The symbol that triggers the transition.
        :type symbol: str
        """
        if from_state not in self._Q:
            raise ValueError(f"From state '{from_state}' is not in the set of states.")
        if to_state not in self._Q:
            raise ValueError(f"To state '{to_state}' is not in the set of states.")
        if symbol not in self._Sigma:
            self._Sigma.add(symbol)

        if (from_state, symbol) not in self._delta:
            self._delta[(from_state, symbol)] = set()
        self._delta[(from_state, symbol)].add(to_state)
        self._G.add_edge(from_state, to_state, label=symbol)
    
    def simulate(self, words: Optional[List[str]] = None, verbose: bool = False) -> None:
        """
        Simulate the Automaton on the input words. If no words are provided, use the test words from the JSON.

        :param words: A list of words to simulate on the Automaton.
        :type words: Optional[List[str]]
        :param verbose: Whether to print detailed simulation steps.
        :type verbose: bool
        """
        if words is None:
            if not self._test_words:
                print("No test words provided in JSON or as input.")
                return
            words = self._test_words
        
        for word in words:
            result = self._simulate_word(word, verbose=verbose)
            print(f"Word '{word}' is {'accepted' if result else 'rejected'}.")
    
    @abstractmethod
    def _simulate_word(self, word: str, verbose: bool = False) -> bool:
        """
        Simulate the Automaton on a single word. This method should be implemented in child classes.

        :param word: The word to simulate.
        :type word: str
        :param verbose: Whether to print detailed simulation steps.
        :type verbose: bool
        :returns: Whether the word is accepted by the Automaton.
        :rtype: bool
        """
        raise NotImplementedError("_simulate_word method must be implemented in child classes.")

    def last_statement_of(self, word: str) -> Set[str]:
        """
        Simulate the word and return the set of possible last states.

        :param word: The word to simulate on the NFA.
        :type word: str
        :returns: The set of possible last states after simulating the word.
        :rtype: Set[str]
        """
        current_states = {self._q0}
        for symbol in word:
            if not current_states:
                print(f"Word '{word}' is rejected.")
                print(f"No valid states reached.")
                return set()
            next_states = set()
            for state in current_states:
                if (state, symbol) in self._delta:
                    next_states.update(self._delta[(state, symbol)])
            current_states = next_states
        
        if current_states & self._F:
            print(f"Word '{word}' is accepted by the NFA.")
        else:
            print(f"Word '{word}' is rejected.")
        print(f"Ended on states: {current_states}")
        return current_states
    
    def visualize(self) -> None:
        """
        Visualize the Automaton using graph representation.
        """
        G = nx.MultiDiGraph()
        
        for state in self._Q:
            if state == self._q0:
                G.add_node(state, shape='circle', style='bold')
            elif state in self._F:
                G.add_node(state, shape='doublecircle')
            else:
                G.add_node(state, shape='circle')

        for (from_state, symbol), to_states in self._delta.items():
            for to_state in to_states:
                G.add_edge(from_state, to_state, label=symbol)

        pydot_graph = nx.drawing.nx_pydot.to_pydot(G)

        pydot_graph.set_rankdir('LR')

        start_node = pydot.Node('start', shape='point')
        pydot_graph.add_node(start_node)
        edge = pydot.Edge('start', self._q0, arrowhead='normal')
        pydot_graph.add_edge(edge)

        image_data = pydot_graph.create_png()
        
        if image_data is None:
            raise RuntimeError("Failed to generate PNG image from Automaton graph.")
        
        display(Image(image_data))
        
        with open("result_images/" + self._name + ".png", "wb") as f:
            f.write(image_data)
    
    def import_from_json(self, filename: str) -> None:
        """
        Import the Automaton definition from a JSON file.

        :param filename: The path to the JSON file containing the Automaton definition.
        :type filename: str
        """
        with open(filename, 'r') as f:
            automaton_dict = json.load(f)
        self._file_name = os.path.splitext(os.path.basename(filename))[0]
        self._name = automaton_dict.get('name', None)
        self._Q = set(automaton_dict['states'])
        self._Sigma = set(automaton_dict['alphabet'])
        self._q0 = automaton_dict['start_state']
        self._F = set(automaton_dict['accept_states'])
        
        for state in self._Q:
            accepting = state in self._F
            starting = state == self._q0
            self.add_state(state, accepting=accepting, starting=starting)
        
        for transition in automaton_dict['transitions']:
            self.add_transition(transition['from'], transition['to'], transition['symbol'])
        
        self._test_words = automaton_dict.get('test_words', [])
        print(f"Automaton imported from {filename}")

    def export_to_json(self, filename: str) -> None:
        """
        Export the Automaton definition to a JSON file.

        :param filename: The path where to save the JSON file.
        :type filename: str
        """
        automaton_dict = {
            "name": self._name or "Unnamed Automaton",
            "states": list(self._Q),
            "alphabet": list(self._Sigma),
            "start_state": self._q0,
            "accept_states": list(self._F),
            "transitions": []
        }

        for (from_state, symbol), to_states in self._delta.items():
            for to_state in to_states:
                automaton_dict["transitions"].append({
                    "from": from_state,
                    "to": to_state,
                    "symbol": symbol
                })

        if self._test_words:
            automaton_dict["test_words"] = self._test_words

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as f:
            json.dump(automaton_dict, f, indent=2)
        
        print(f"Automaton exported to {filename}")

    def validate(self) -> None:
        """
        Validate the Automaton to ensure it follows basic Automaton properties.
        """
        errors: List[str] = []
        
        if not self._q0:
            errors.append("Automaton must have a start state.")
        elif self._q0 not in self._Q:
            errors.append(f"Start state '{self._q0}' is not in the set of states.")
        
        invalid_accept_states = self._F - self._Q
        if invalid_accept_states:
            errors.append(f"Accept states {invalid_accept_states} are not in the set of states.")
        
        for (from_state, symbol), to_states in self._delta.items():
            if from_state not in self._Q:
                errors.append(f"Transition from unknown state '{from_state}'.")
            if not to_states.issubset(self._Q):
                errors.append(f"Transition to unknown state(s) '{to_states - self._Q}' from state '{from_state}' on symbol '{symbol}'.")
            if symbol not in self._Sigma:
                errors.append(f"Transition with unknown symbol '{symbol}'.")

        if errors:
            raise ValueError("Automaton validation errors:\n" + "\n".join(errors))
        else:
            print("DFA validation complete. No errors found.")
    
    def permutations(self, length: int) -> List[str]:
        """
        Generate all words accepted by the Automaton up to a certain length and optionally save to a file.

        :param length: The maximum length of words to generate.
        :type length: int
        :param output_file: The path to save the generated words. If None, words are not saved to a file.
        :type output_file: str, optional
        :returns: A sorted list of words accepted by the Automaton.
        :rtype: List[str]
        """
        accepted_words: Set[str] = set()
        queue: List[Tuple[Set[str], str]] = [({self._q0}, "")]
        while queue:
            current_states, word = queue.pop(0)
            if any(state in self._F for state in current_states) and len(word) > 0:
                accepted_words.add(word)
            if len(word) >= length:
                continue
            for symbol in self._Sigma:
                next_states: Set[str] = set()
                for state in current_states:
                    if (state, symbol) in self._delta:
                        next_states.update(self._delta[(state, symbol)])
                if next_states:
                    queue.append((next_states, word + symbol))
        
        sorted_words = sorted(accepted_words)
        
        if os.path.exists("permutations") is False:
            os.makedirs("permutations")
        
        self._save_permutations(sorted_words, "permutations/" + self._name + ".json")
        
        return sorted_words

    def _save_permutations(self, words: List[str], output_file: str) -> None:
        """
        Save the generated words to a JSON file.

        :param words: List of generated words to save.
        :type words: List[str]
        :param output_file: The path to save the generated words.
        :type output_file: str
        """
        data = {
            "automaton_name": self._name or self._file_name or "Unnamed Automaton",
            "max_length": max(len(word) for word in words) if words else 0,
            "word_count": len(words),
            "words": words
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Permutations saved to {output_file}")
    
    def print_transition_matrix(self) -> None:
        """
        Print the transition matrix of the automaton.
        """
        states = sorted(self._Q)
        symbols = sorted(self._Sigma)

        state_width = max(len(state) for state in states)
        symbol_width = max(len(symbol) for symbol in symbols)
        cell_width = max(state_width, symbol_width, len("Present State"))

        print(f"{'Present State':<{cell_width}} | ", end="")
        for symbol in symbols:
            print(f"Next State for Input {symbol:<{cell_width}} | ", end="")
        print()

        print("-" * (cell_width + 2) + "+" + "-" * (len(symbols) * (cell_width + 24) - 1))

        for state in states:
            print(f"{state:<{cell_width}} | ", end="")
            for symbol in symbols:
                if (state, symbol) in self._delta:
                    next_states = self._delta[(state, symbol)]
                    cell_content = ",".join(sorted(next_states))
                else:
                    cell_content = "-"
                print(f"{cell_content:^{cell_width+20}} | ", end="")
            print()

        print(f"\nStart state: {self._q0}")
        print(f"Accept states: {', '.join(sorted(self._F))}")
    
    def compare_with(self, other: 'Automaton', comper_permutations_n:int = 4) -> None:
        """
        Compare this DFA with another DFA and display their differences.
        
        :param other: Another DFA to compare with
        :type other: DFA
        """
        print("\nDFA Comparison:")
        print("=" * 50)
        
        print("\nBasic Properties:")
        print(f"{'Property':<15} | {'Original':<20} | {'Minimal':<20}")
        print("-" * 60)
        print(f"{'States':<15} | {len(self):<20} | {len(other):<20}")
        print(f"{'Alphabet':<15} | {len(self._Sigma):<20} | {len(other._Sigma):<20}")
        print(f"{'Accept States':<15} | {len(self._F):<20} | {len(other._F):<20}")
        
        print("\nTransition Matrices Comparison:")
        print("Self DFA:")
        self.print_transition_matrix()
        print("\nOther DFA:")
        other.print_transition_matrix()
        
        print("\nEquivalence Test:")
        
        print("\nChecking:")
        print(f"{'Word':<10} | {'Original':<10} | {'Minimal':<10}")
        print("-" * 35)
        for word in self.permutations(comper_permutations_n):
            res = self._simulate_word(word=word)
            res_other = other._simulate_word(word=word)
            print(f"{word:<10} | {str(res):<10} | {str(res_other):<10}")
            if res != res_other:
                print(f"Word '{word}' is accepted by one DFA and rejected by the other.")
        print("The DFAs are equivalent.")
        
    @staticmethod
    def nfa_to_dfa(nfa: 'NFA') -> 'DFA':
        """
        Egy NFA átalakítása ekvivalens DFA-vá.

        :param nfa: Az átalakítandó NFA.
        :type nfa: NFA
        :return: Egy ekvivalens DFA.
        :rtype: DFA
        """
        start_state = frozenset([nfa.start_state])
        dfa_states = {start_state}
        unmarked_states = [start_state]
        dfa_delta = {}
        dfa_accept_states = set()

        while unmarked_states:
            current = unmarked_states.pop()
            for symbol in nfa.alphabet:
                next_state = set()
                for nfa_state in current:
                    if (nfa_state, symbol) in nfa.transitions:
                        next_state.update(nfa.transitions[(nfa_state, symbol)])
                if next_state:
                    next_state = frozenset(next_state)
                    if next_state not in dfa_states:
                        dfa_states.add(next_state)
                        unmarked_states.append(next_state)
                    dfa_delta[(current, symbol)] = next_state

        for state_set in dfa_states:
            if nfa.accept_states & state_set:
                dfa_accept_states.add(state_set)

        state_names = {state_set: f"q{index}" for index, state_set in enumerate(dfa_states)}

        dfa = DFA()
        dfa._Q = set(state_names.values())
        dfa._Sigma = nfa.alphabet
        dfa._q0 = state_names[start_state]
        dfa._F = {state_names[state_set] for state_set in dfa_accept_states}

        for (from_states, symbol), to_states in dfa_delta.items():
            from_state_name = state_names[from_states]
            to_state_name = state_names[to_states]
            dfa.add_state(from_state_name)
            dfa.add_state(to_state_name)
            dfa.add_transition(from_state_name, to_state_name, symbol)

        dfa._name = (nfa._name or "NFA") + "_converted_to_DFA"

        return dfa
    
    def __iter__(self):
        """
        Initialize iteration over the automaton's states.

        :returns: The iterator object itself.
        :rtype: Automaton
        """
        self._iterator_states = list(self._Q)
        self._iterator_index = 0
        return self

    def __next__(self) -> Tuple[str, Dict[str, Set[str]]]:
        """
        Return the next state and its transitions during iteration.

        :returns: A tuple containing the state and its transitions.
        :rtype: Tuple[str, Dict[str, Set[str]]]
        """
        if self._iterator_index >= len(self._iterator_states):
            raise StopIteration

        state = self._iterator_states[self._iterator_index]
        transitions = {symbol: self._delta.get((state, symbol), set()) for symbol in self._Sigma}
        self._iterator_index += 1
        return (state, transitions)
    
    def __eq__(self, other: 'Automaton') -> bool:
        """
        Check if two automatons are equivalent by comparing their minimal DFA representations.

        :param other: The other automaton to compare with.
        :type other: Automaton
        :returns: True if equivalent, False otherwise.
        :rtype: bool
        """
        if not isinstance(self, DFA):
            self_dfa = Automaton.nfa_to_dfa(self)
        else:
            self_dfa = self

        if not isinstance(other, DFA):
            other_dfa = Automaton.nfa_to_dfa(other)
        else:
            other_dfa = other

        dfa1 = self_dfa.minimalize()
        dfa2 = other_dfa.minimalize()

        queue = deque([(dfa1.start_state, dfa2.start_state)])
        visited = {(dfa1.start_state, dfa2.start_state)}

        while queue:
            state1, state2 = queue.popleft()
            
            if (state1 in dfa1.accept_states) != (state2 in dfa2.accept_states):
                return False

            for symbol in dfa1.alphabet:
                next_state1 = next(iter(dfa1.transitions.get((state1, symbol), set())), None)
                next_state2 = next(iter(dfa2.transitions.get((state2, symbol), set())), None)

                if (next_state1 is None) != (next_state2 is None):
                    return False

                if next_state1 is not None and next_state2 is not None:
                    if (next_state1, next_state2) not in visited:
                        visited.add((next_state1, next_state2))
                        queue.append((next_state1, next_state2))

        return True

    def __len__(self) -> int:
        """
        Return the number of states in the automaton.
        
        :returns: The number of states in the automaton.
        :rtype: int
        """
        return len(self._Q)

    def __getitem__(self, state: str) -> Dict[str, Set[str]]:
        """
        Get all transitions from a given state.
        
        :param state: The state to get transitions for.
        :type state: str
        :returns: A dictionary mapping input symbols to sets of target states.
        :rtype: Dict[str, Set[str]]
        :raises KeyError: If the state does not exist in the automaton
        """
        if state not in self._Q:
            raise KeyError(f"State '{state}' does not exist in the automaton")
        
        transitions = {}
        for symbol in self._Sigma:
            if (state, symbol) in self._delta:
                transitions[symbol] = self._delta[(state, symbol)]
            else:
                transitions[symbol] = set()
        return transitions

    def __contains__(self, item: str) -> bool:
        """
        Check if a state exists in the automaton.
        
        :param item: The state to check for.
        :type item: str
        :returns: True if the state exists in the automaton, False otherwise.
        :rtype: bool
        """
        return item in self._Q

class DFA(Automaton):
    def __init__(self, file: Optional[str] = None) -> None:
        super().__init__(file)
    
    def add_transition(self, from_state: str, to_state: str, symbol: str) -> None:
        """
        Add a transition to the DFA's transition function.

        :param from_state: The state from which the transition starts.
        :param to_state: The state to which the transition goes.
        :param symbol: The input symbol for this transition.
        """
        if (from_state, symbol) in self._delta:
            raise ValueError(f"DFA already has a transition from state '{from_state}' on symbol '{symbol}'.")
        super().add_transition(from_state, to_state, symbol)
        self._delta[(from_state, symbol)] = {to_state}

    def validate(self) -> None:
        """
        Validate the automaton to ensure it follows basic automaton properties.
        """
        super().validate()
        errors: List[str] = []

        for state in self._Q:
            for symbol in self._Sigma:
                transitions = self._delta.get((state, symbol), set())
                
                if not transitions:
                    errors.append(f"Missing transition for state '{state}' and symbol '{symbol}'.")
                elif len(transitions) > 1:
                    errors.append(f"Multiple transitions from state '{state}' on symbol '{symbol}'. DFA must have exactly one transition.")

        if errors:
            raise ValueError("DFA validation errors:\n" + "\n".join(errors))
        else:
            print("DFA validation complete. No errors found.")

    def _simulate_word(self, word: str, verbose: bool = False) -> bool:
        """
        Simulate the DFA on a single word.

        :param word: The word to simulate on the DFA.
        :param verbose: Whether to print detailed simulation steps.
        :returns: Whether the word is accepted by the DFA.
        """
        if not self:
            raise ValueError("DFA has no initial state defined.")
        
        current_state = self._q0
        if verbose:
            print(f"Starting simulation for word '{word}' from initial state '{current_state}'")
        
        for symbol in word:
            if symbol not in self._Sigma:
                if verbose:
                    print(f"Symbol '{symbol}' not in alphabet, rejecting word.")
                return False
            if (current_state, symbol) not in self._delta:
                if verbose:
                    print(f"No transition from state '{current_state}' on symbol '{symbol}', rejecting word.")
                return False
            
            next_state = next(iter(self._delta[(current_state, symbol)]))
            if verbose:
                print(f"Transition: {current_state} --{symbol}--> {next_state}")
            current_state = next_state
        
        is_accepted = current_state in self._F
        if verbose:
            print(f"Word '{word}' is {'accepted' if is_accepted else 'rejected'}, "
                  f"ended in {'accepting' if is_accepted else 'non-accepting'} state '{current_state}'")
        return is_accepted
    
    def minimalize(self) -> 'DFA':
        """
        Minimizes the DFA using Moore's algorithm and returns a new DFA instance.
        
        :return: A minimized DFA instance.
        :rtype: DFA
        """
        non_accept_states = self._Q - self._F
        P = [self._F, non_accept_states] if self._F and non_accept_states else [self._F or non_accept_states]

        while True:
            new_P = []
            for group in P:
                split = {}
                for state in group:
                    profile = []
                    for symbol in sorted(self._Sigma):
                        if (state, symbol) in self._delta:
                            target_state = next(iter(self._delta[(state, symbol)]))
                            target_group = next(i for i, g in enumerate(P) if target_state in g)
                            profile.append(target_group)
                        else:
                            profile.append(None)
                    profile = tuple(profile)
                    
                    if profile not in split:
                        split[profile] = set()
                    split[profile].add(state)
                new_P.extend(split.values())
            
            if len(new_P) == len(P):
                break
            P = new_P

        state_to_name = {}
        for i, part in enumerate(P):
            state_name = f"q{i}"
            for state in part:
                state_to_name[state] = state_name

        minimized_automaton = DFA()
        minimized_automaton._Q = {f"q{i}" for i in range(len(P))}
        minimized_automaton._Sigma = self._Sigma
        minimized_automaton._q0 = state_to_name[self._q0]
        minimized_automaton._F = {state_to_name[state] for state in self._F}

        for (state, symbol), target_states in self._delta.items():
            target_state = next(iter(target_states))
            minimized_source = state_to_name[state]
            minimized_target = state_to_name[target_state]
            minimized_automaton._delta[(minimized_source, symbol)] = {minimized_target}

        minimized_automaton._name = (str(self._name) + "_minimized") if self._name else "minimized_automaton"

        return minimized_automaton

class NFA(Automaton):
    def __init__(self, file: Optional[str] = None) -> None:
        super().__init__(file)

    def validate(self) -> None:
        """
        Validate the NFA to ensure it follows NFA properties.
        """
        super().validate()
        errors: List[str] = []

        if not self._q0:
            errors.append("NFA must have a start state.")

        for (from_state, symbol), to_states in self._delta.items():
            if from_state not in self._Q:
                errors.append(f"Transition from unknown state '{from_state}'.")
            if not to_states.issubset(self._Q):
                errors.append(f"Transition to unknown state(s) '{to_states - self._Q}' from state '{from_state}' on symbol '{symbol}'.")
            if symbol not in self._Sigma:
                errors.append(f"Transition with unknown symbol '{symbol}'.")

        if errors:
            raise ValueError("NFA validation errors:\n" + "\n".join(errors))
        else:
            print("NFA validation complete. No errors found.")

    def _simulate_word(self, word: str, verbose: bool = False) -> bool:
        current_states = {self._q0}
        if verbose:
            print(f"Starting simulation for word '{word}' from initial state(s) {current_states}")

        for symbol in word:
            if verbose:
                print(f"Processing symbol '{symbol}'")
            
            next_states = set()
            for state in current_states:
                if (state, symbol) in self._delta:
                    next_states.update(self._delta[(state, symbol)])
                    if verbose:
                        print(f"Transition: {state} --{symbol}--> {self._delta[(state, symbol)]}")

            if not next_states:
                if verbose:
                    print(f"No valid transitions for symbol '{symbol}', rejecting word.")
                return False

            current_states = next_states

        is_accepted = any(state in self._F for state in current_states)
        if verbose:
            print(f"Word '{word}' is {'accepted' if is_accepted else 'rejected'}, "
                  f"ended in state(s) {current_states}")
        return is_accepted
    
def this_is_not_a_test(self) -> None:
    print(f"\nTesting {self.__class__.__name__}:")
    print("=" * 50)
    print("Validating automaton...")
    self.validate()
    print(f"\nStates: {self._Q}")
    print(f"Alphabet: {self._Sigma}")
    print(f"Start state: {self._q0}")
    print(f"Accept states: {self._F}")
    print("\nTransition Matrix:")
    self.print_transition_matrix()
    print("\nVisualizing automaton...")
    self.visualize()
    print("\nSimulating test words:")
    self.simulate()
    permutations_length = 6
    print("\nGenerating permutations up to length {permutations_length}:")
    perms = self.permutations(permutations_length)
    print(f"Accepted words: {perms}")

if __name__ == "__main__":
    dfa = DFA(file="jsons/test_dfa.json")
    this_is_not_a_test(dfa)


