import pytest
from Automatons import Automaton, DFA, NFA
import os
import json

@pytest.fixture
def simple_dfa() -> DFA:
    """Fixture for a simple DFA that accepts strings ending in 'ab'"""
    dfa = DFA()
    dfa._name = "test_dfa"
    dfa.add_state("q0", starting=True)
    dfa.add_state("q1")
    dfa.add_state("q2", accepting=True)
    dfa.add_transition("q0", "q1", "a")
    dfa.add_transition("q0", "q0", "b")
    dfa.add_transition("q1", "q2", "b")
    dfa.add_transition("q1", "q1", "a")
    dfa.add_transition("q2", "q2", "a")
    dfa.add_transition("q2", "q1", "b")
    return dfa

@pytest.fixture
def simple_nfa() -> NFA:
    """Fixture for a simple NFA that accepts strings containing 'ab'"""
    nfa = NFA()
    nfa._name = "test_nfa"
    nfa.add_state("s0", starting=True)
    nfa.add_state("s1")
    nfa.add_state("s2", accepting=True)
    nfa.add_transition("s0", "s0", "a")
    nfa.add_transition("s0", "s1", "a")
    nfa.add_transition("s1", "s2", "b")
    return nfa

class TestAutomatonBase:
    """Base test class for common automaton functionality"""
    
    def test_add_state(self, simple_dfa):
        """Test adding states"""
        assert "q0" in simple_dfa.states
        assert simple_dfa.start_state == "q0"
        assert "q2" in simple_dfa.accept_states

    def test_add_invalid_transition(self, simple_dfa):
        """Test adding invalid transitions"""
        with pytest.raises(ValueError):
            simple_dfa.add_transition("nonexistent", "q1", "a")


    def test_permutations(self, simple_dfa):
        """Test word permutation generation"""
        perms = simple_dfa.permutations(6)
        
        expected_perms = ["aaabaa", "aabaaa"]
        for perm in expected_perms:
            assert perm in perms
        
        assert os.path.exists(f"permutations/{simple_dfa._name}.json")
        
        with open(f"permutations/{simple_dfa._name}.json", 'r') as f:
            data = json.load(f)
            assert data["automaton_name"] == simple_dfa._name
            assert data["max_length"] == 6
            for perm in perms:
                assert perm in data["words"]
    
    def test_visualize(self, simple_dfa, simple_nfa):
        """Test visualization"""
    
        simple_dfa.visualize()
        if exists := os.path.exists(f"result_images/{simple_dfa._name}.png"):   
            os.remove(f"result_images/{simple_dfa._name}.png")
            
        assert exists
        
        simple_nfa.visualize()

        if exists := os.path.exists(f"result_images/{simple_nfa._name}.png"):
            os.remove(f"result_images/{simple_nfa._name}.png")
        
        assert exists
            
class TestDFA:
    """Test DFA specific functionality"""
    
    def test_simulation(self, simple_dfa):
        """Test DFA word simulation"""
        assert simple_dfa._simulate_word("ab") == True
        assert simple_dfa._simulate_word("a") == False
        assert simple_dfa._simulate_word("") == False

    def test_minimalization(self, simple_dfa):
        """Test DFA minimalization"""
        dfa = DFA()
        
        dfa.add_state("q0", starting=True)
        dfa.add_state("q1")
        dfa.add_state("q2", accepting=True)
        dfa.add_state("q3", accepting=True)
        
        dfa.add_transition("q0", "q1", "a")
        dfa.add_transition("q0", "q0", "b")
        
        dfa.add_transition("q1", "q2", "b")
        dfa.add_transition("q1", "q1", "a")
        
        dfa.add_transition("q2", "q2", "a")
        dfa.add_transition("q2", "q1", "b")
        
        dfa.add_transition("q3", "q3", "a")
        dfa.add_transition("q3", "q1", "b")
        
        min_dfa = dfa.minimalize()
        
        assert len(min_dfa.states) < len(dfa.states)
        
        test_words = ["ab", "aba", "abab", "a", "b", ""]
        for word in test_words:
            assert dfa._simulate_word(word) == min_dfa._simulate_word(word)

    @pytest.mark.parametrize("word,expected", [
        ("ab", True),
        ("aba", True),
        ("abaaaaaaa", True),
        ("", False),
        ("abab", False),
        ("ababa", False),
        ("a", False),
        ("b", False)
    ])
    def test_word_acceptance(self, simple_dfa, word: str, expected: bool):
        """Test word acceptance with various inputs"""
        assert simple_dfa._simulate_word(word) == expected
        
    def test_validate_dfa(self, simple_dfa):
        """Test DFA validation"""
        simple_dfa.validate()
        
        invalid_dfa = DFA()
        invalid_dfa.add_state("q0")
        with pytest.raises(ValueError) as exc:
            invalid_dfa.validate()
        assert "must have a start state" in str(exc.value)
        
        invalid_dfa = DFA()
        invalid_dfa.add_state("q0", starting=True)
        invalid_dfa.add_state("q1")
        invalid_dfa._Sigma.add("a")
        with pytest.raises(ValueError) as exc:
            invalid_dfa.validate()
        assert "Missing transition" in str(exc.value)
        
        invalid_dfa = DFA()
        invalid_dfa.add_state("q0", starting=True)
        invalid_dfa.add_state("q1")
        invalid_dfa.add_state("q2")
        invalid_dfa._Sigma.add("a")
        invalid_dfa._delta[("q0", "a")] = {"q1", "q2"}
        with pytest.raises(ValueError) as exc:
            invalid_dfa.validate()
        assert "Multiple transitions" in str(exc.value)

class TestNFA:
    """Test NFA specific functionality"""
    
    def test_simulation(self, simple_nfa):
        """Test NFA word simulation"""
        assert simple_nfa._simulate_word("ab") == True
        assert simple_nfa._simulate_word("aab") == True
        assert simple_nfa._simulate_word("aa") == False

    def test_nfa_to_dfa_conversion(self, simple_nfa):
        """Test NFA to DFA conversion"""
        dfa = Automaton.nfa_to_dfa(simple_nfa)
        assert dfa._simulate_word("ab") == True
        assert dfa._simulate_word("aab") == True
        assert dfa._simulate_word("aa") == False

    def test_nfa_to_dfa_complex(self, simple_nfa):
        """Test NFA to DFA conversion with a more complex NFA"""
        nfa = NFA()
        nfa._name = "complex_nfa"
        nfa.add_state("q0", starting=True)
        nfa.add_state("q1")
        nfa.add_state("q2", accepting=True)
        nfa.add_transition("q0", "q0", "a")
        nfa.add_transition("q0", "q1", "a")
        nfa.add_transition("q1", "q2", "b")
        nfa.add_transition("q2", "q2", "a")
        
        dfa = Automaton.nfa_to_dfa(nfa)
        
        assert len(dfa.states) >= 3
        assert len(dfa.alphabet) == 2
        
        test_words = [
            ("ab", True),
            ("aab", True),
            ("aba", True),
            ("b", False),
            ("aa", False),
            ("", False)
        ]
        
        for word, expected in test_words:
            assert dfa._simulate_word(word) == expected, f"Failed for word: {word}"
            assert nfa._simulate_word(word) == expected, f"NFA failed for word: {word}"

    def test_nfa_to_dfa_epsilon(self):
        """Test NFA to DFA conversion with epsilon transitions"""
        nfa = NFA()
        nfa._name = "epsilon_nfa"
        nfa.add_state("q0", starting=True)
        nfa.add_state("q1")
        nfa.add_state("q2", accepting=True)
        nfa.add_transition("q0", "q1", "a")
        nfa.add_transition("q1", "q2", "b")
        nfa.add_transition("q1", "q2", "a")
        nfa.add_transition("q2", "q1", "b")
        nfa.add_transition("q2", "q1", "a")
        nfa.add_transition("q2", "q0", "a")
        nfa.add_transition("q2", "q0", "b")
        
        dfa = Automaton.nfa_to_dfa(nfa)
        
        for perm in dfa.permutations(6):
            assert dfa._simulate_word(perm) == nfa._simulate_word(perm)
    
    def test_validate_nfa(self, simple_nfa):
        """Test NFA validation"""
        simple_nfa.validate()
        
        invalid_nfa = NFA()
        invalid_nfa.add_state("q0")
        with pytest.raises(ValueError) as exc:
            invalid_nfa.validate()
        assert "must have a start state" in str(exc.value)
        
        invalid_nfa = NFA()
        invalid_nfa.add_state("q0", starting=True)
        invalid_nfa._delta[("q0", "a")] = {"nonexistent"}
        with pytest.raises(ValueError) as exc:
            invalid_nfa.validate()
        assert "unknown state" in str(exc.value)
        
        invalid_nfa = NFA()
        invalid_nfa.add_state("q0", starting=True)
        invalid_nfa.add_state("q1")
        invalid_nfa._delta[("q0", "?")] = {"q1"}
        with pytest.raises(ValueError) as exc:
            invalid_nfa.validate()
        assert "unknown symbol" in str(exc.value)
        

class TestJSONOperations:
    """Test JSON import/export operations"""
    
    def test_export_import(self, simple_dfa, tmp_path):
        """Test JSON export and import"""
        json_path = tmp_path / "test_export.json"
        simple_dfa.export_to_json(str(json_path))
        assert json_path.exists()

        imported_dfa = DFA(str(json_path))
        assert simple_dfa.states == imported_dfa.states
        assert simple_dfa.alphabet == imported_dfa.alphabet
        assert simple_dfa.transitions == imported_dfa.transitions

class TestComparison:
    """Test automaton comparison operations"""
    
    def test_equality(self, simple_dfa):
        """Test automaton equality"""
        dfa1 = simple_dfa
        dfa2 = simple_dfa.minimalize()
        assert dfa1 == dfa2

    def test_iteration(self, simple_dfa):
        """Test automaton iteration"""
        states = {state for state, _ in simple_dfa}
        assert states == simple_dfa.states

    def test_contains(self, simple_dfa):
        """Test state membership"""
        assert "q0" in simple_dfa
        assert "nonexistent" not in simple_dfa

    def test_getitem(self, simple_dfa):
        """Test transition access"""
        transitions = simple_dfa["q0"]
        assert transitions["a"] == {"q1"}

