from research_agent.workflow.graph import route_after_approval, route_after_critic


def test_critic_retries_only_before_round_limit():
    assert route_after_critic({"sufficient": False, "round": 1}) == "retrieve"
    assert route_after_critic({"sufficient": False, "round": 2}) == "finalize"
    assert route_after_critic({"sufficient": True, "round": 1}) == "finalize"


def test_rejected_plan_does_not_retrieve():
    assert route_after_approval({"approval_status": "rejected"}) == "finalize"
    assert route_after_approval({"approval_status": "approved"}) == "retrieve"
