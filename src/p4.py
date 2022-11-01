import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import sys


class UserState(pomdp_py.State):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, UserState):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "UserState(%s)" % self.name

    def other(self):
        if self.name.startswith("CAR"):
            return UserState("BULK")
        else:
            return UserState("CARDIO")


class UserAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, UserAction):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "UserAction(%s)" % self.name


class UserObservation(pomdp_py.Observation):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, UserObservation):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "UserObservation(%s)" % self.name


# Observation model


class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if next_state.name == "BULK":
            if action.name == "BURPEE" or action.name == "PUSHUP":
                return 1.0
            else:
                return 0.05
        else:
            return 0.5

    def sample(self, next_state, action):
        if action.name == "BURPEE":
            thresh = 1.0 - self.noise
        else:
            thresh = 0.5

        if random.uniform(0, 1) < thresh:
            return UserObservation(next_state.name)
        else:
            return UserObservation(next_state.other().name)

    def get_all_observations(self):
        return [UserObservation(s) for s in {"Complete", "Skip"}]


# Transition Model


class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        if action.name.startswith("BUR"):
            return 0.5
        else:
            if next_state.name == state.name:
                return 1.0 - 1e-9
            else:
                return 1e-9

    def sample(self, state, action):
        if action.name.startswith("BUR"):
            return random.choice(self.get_all_states())
        else:
            return UserState(state.name)

    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)"""
        return [UserState(s) for s in {"CARDIO", "BULK"}]


# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        reward = 0.5
        if action.name == "BURPEE" or action.name == "SQUAT":
            if state.name == "CARDIO":
                return reward * 1.5
            else:
                return reward * 0.5
        elif action.name == "BURPEE" or action.name == "PUSHUP":
            if state.name == "BULK":
                return reward * 1.7
            else:
                return reward * 0.7
        else:
            return reward * 0.9

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)


# Policy Model


class PolicyModel(pomdp_py.RandomRollout):
    """This is an extremely dumb policy model; To keep consistent
    with the framework."""

    # A stay action can be added to test that POMDP solver is
    # able to differentiate information gathering actions.
    ACTIONS = {UserAction(s) for s in {"BURPEE", "PUSHUP", "SQUAT"}}

    def sample(self, state, **kwargs):
        return random.sample(self.get_all_actions(), 1)[0]

    def get_all_actions(self, **kwargs):
        return PolicyModel.ACTIONS


class UserProblem(pomdp_py.POMDP):
    """
    In fact, creating a TigerProblem class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, obs_noise, init_true_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(
            init_belief,
            PolicyModel(),
            TransitionModel(),
            ObservationModel(obs_noise),
            RewardModel(),
        )
        env = pomdp_py.Environment(init_true_state, TransitionModel(), RewardModel())
        super().__init__(agent, env, name="UserProblem")

    @staticmethod
    def create(state="CARDIO", belief=0.4, obs_noise=0.15):
        """
        Args:
            state (str): could be 'tiger-left' or 'tiger-right'; True state of the environment
            belief (float): Initial belief that the target is on the left; Between 0-1.
            obs_noise (float): Noise for the observation model (default 0.15)
        """
        init_true_state = UserState(state)
        init_belief = pomdp_py.Histogram(
            {UserState("CARDIO"): belief, UserState("BULK"): 1.0 - belief}
        )
        user_problem = UserProblem(
            obs_noise, init_true_state, init_belief  # observation noise
        )
        user_problem.agent.set_belief(init_belief, prior=True)
        return user_problem


def test_planner(user_problem, planner, nsteps=3, debug_tree=False):
    for i in range(nsteps):
        action = planner.plan(user_problem.agent)
        if debug_tree:
            from pomdp_py.utils import TreeDebugger

            dd = TreeDebugger(user_problem.agent.tree)
            import pdb

            pdb.set_trace()

        print("==== Step %d ====" % (i + 1))
        print("True state: %s" % user_problem.env.state)
        print("Belief: %s" % str(user_problem.agent.cur_belief))
        print("Action: %s" % str(action))
        print(
            "Reward: %s"
            % str(
                user_problem.env.reward_model.sample(
                    user_problem.env.state, action, None
                )
            )
        )

        # Let's create some simulated real observation; Update the belief
        # Creating true observation for sanity checking solver behavior.
        # In general, this observation should be sampled from agent's observation model.
        # real_observation = UserObservation(user_problem.env.state.name)
        real_observation = user_problem.env.provide_observation(
            user_problem.agent.observation_model, action
        )
        print(">> Observation: %s" % real_observation)
        user_problem.agent.update_history(action, real_observation)

        # If the planner is POMCP, planner.update also updates agent belief.
        planner.update(user_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims: %d" % planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        if isinstance(user_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(
                user_problem.agent.cur_belief,
                action,
                real_observation,
                user_problem.agent.observation_model,
                user_problem.agent.transition_model,
            )
            user_problem.agent.set_belief(new_belief)

        if action.name.startswith("BUR"):
            # Make it clearer to see what actions are taken until every time door is opened.
            print("\n")


def main():
    init_true_state = random.choice([UserState("BULK"), UserState("CARDIO")])
    init_belief = pomdp_py.Histogram({UserState("BULK"): 0.5, UserState("CARDIO"): 0.5})
    user_problem = UserProblem(0.15, init_true_state, init_belief)  # observation noise

    print("** Testing value iteration **")
    vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
    test_planner(user_problem, vi, nsteps=3)

    # Reset agent belief
    user_problem.agent.set_belief(init_belief, prior=True)

    print("\n** Testing POUCT **")
    pouct = pomdp_py.POUCT(
        max_depth=12,
        discount_factor=0.95,
        num_sims=4096,
        exploration_const=50,
        rollout_policy=user_problem.agent.policy_model,
        show_progress=True,
    )
    test_planner(user_problem, pouct, nsteps=10)
    TreeDebugger(user_problem.agent.tree).pp

    # Reset agent belief
    user_problem.agent.set_belief(init_belief, prior=True)
    user_problem.agent.tree = None

    print("** Testing POMCP **")
    user_problem.agent.set_belief(
        pomdp_py.Particles.from_histogram(init_belief, num_particles=100), prior=True
    )
    pomcp = pomdp_py.POMCP(
        max_depth=12,
        discount_factor=0.95,
        num_sims=1000,
        exploration_const=50,
        rollout_policy=user_problem.agent.policy_model,
        show_progress=True,
        pbar_update_interval=500,
    )
    test_planner(user_problem, pomcp, nsteps=10)
    TreeDebugger(user_problem.agent.tree).mbp


if __name__ == "__main__":
    main()
