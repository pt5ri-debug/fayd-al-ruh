from datetime import timedelta
from core.episodic_memory import EpisodicMemory
from core.memory_decay import MemoryDecay


def test_memory_decays_with_time():
    memory = EpisodicMemory(); episode = memory.store("E1", "rain", 1.)
    assert MemoryDecay().retention(episode, .1, episode.created_at + timedelta(hours=10)) < 1.
