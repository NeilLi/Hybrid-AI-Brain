import pytest
from src.memory.long_term_memory import LongTermMemory

def test_add_and_total_items():
    ltm = LongTermMemory()
    assert ltm.total_items == 0
    ltm.add_knowledge("Fact 1")
    ltm.add_knowledge("Fact 2")
    assert ltm.total_items == 2

def test_retrieve_relevant_knowledge_simple_match():
    ltm = LongTermMemory()
    ltm.add_knowledge("Argentina won the World Cup in 2022.")
    ltm.add_knowledge("GDP per capita of Argentina is high.")
    ltm.add_knowledge("Python is a programming language.")
    # Should retrieve the 2 Argentina facts
    result = ltm.retrieve_relevant_knowledge("Argentina")
    assert len(result) == 2
    # Should retrieve only the GDP fact
    result2 = ltm.retrieve_relevant_knowledge("gdp", top_k=1)
    assert result2 == ["GDP per capita of Argentina is high."]

def test_retrieve_with_empty():
    ltm = LongTermMemory()
    result = ltm.retrieve_relevant_knowledge("anything")
    assert result == []

def test_repr():
    ltm = LongTermMemory()
    assert "total_items=0" in repr(ltm)
    ltm.add_knowledge("A fact")
    assert "total_items=1" in repr(ltm)

