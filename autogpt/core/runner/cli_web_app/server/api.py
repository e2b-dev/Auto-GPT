from typing import Any
from agent_protocol import StepHandler, StepResult

from autogpt.core.agent import AgentSettings, SimpleAgent
from autogpt.core.runner.client_lib.logging import get_client_logger
from autogpt.core.runner.client_lib.parser import (
    parse_agent_name_and_goals,
    parse_agent_plan,
)

async def task_handler(task_input) -> StepHandler:
    task = task_input.__root__ if task_input else {}

    client_logger = get_client_logger()
    client_logger.debug("Getting agent settings")

    user_configuration = task.get("user_configuration", {})
    user_objective = task.get("user_objective")

    if not user_objective:
        raise ValueError("No user objective provided.")

    agent_workspace = (
        user_configuration.get("workspace", {}).get("configuration", {}).get("root", "")
    )

    if not agent_workspace:  # We don't have an agent yet.
        #################
        # Bootstrapping #
        #################
        # Collate the user's settings with the default system settings.
        agent_settings: AgentSettings = SimpleAgent.compile_settings(
            client_logger,
            user_configuration,
        )

        # Ask a language model to determine a name and goals for a suitable agent.
        name_and_goals = await SimpleAgent.determine_agent_name_and_goals(
            user_objective,
            agent_settings,
            client_logger,
        )
        print(parse_agent_name_and_goals(name_and_goals))
        # Finally, update the agent settings with the name and goals.
        agent_settings.update_agent_name_and_goals(name_and_goals)

        # Provision the agent.
        agent_workspace = SimpleAgent.provision_agent(agent_settings, client_logger)
        print("agent is provisioned")

    # launch agent interaction loop
    agent = SimpleAgent.from_workspace(
        agent_workspace,
        client_logger,
    )
    print("agent is loaded")

    plan = await agent.build_initial_plan()
    print(parse_agent_plan(plan))

    current_task: Any | None = None
    next_ability: dict[Any, Any] | None = None

    async def step_handler(step_input):
        # step = step_input.__root__ if step_input else {}
        nonlocal current_task, next_ability

        result: Any | None = None

        if current_task:
            result = await agent.execute_next_ability("y")

        next_determined = await agent.determine_next_ability(plan)

        if type(next_determined) is dict:
            return StepResult(output=next_determined, is_last=True)
        else:
            current_task, next_ability = next_determined
        
        return StepResult(output={
            "current_task": current_task,
            "next_ability": next_ability,
            "result": result,
        })

    return step_handler
