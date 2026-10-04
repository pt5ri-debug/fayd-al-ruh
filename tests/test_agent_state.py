from core.agent_state import AgentState
from core.observation import Observation


def test_agent_state_records_observation():
    state = AgentState(); state.record_observation(Observation("x", "test"))
    assert state.cycle_count == 1
