from core.agent_state import AgentState
from core.observation import Observation


def test_agent_contract_records_cycle():
    state = AgentState(); state.record_observation(Observation("state", "environment"))
    assert state.cycle_count == 1
