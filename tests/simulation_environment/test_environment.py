import numpy as np
import torch

from rl_recsys.simulation_environment.environment import MusicGym


def test_environment_reset(user_sampler, doc_catalogue, content_rec):
    env = MusicGym(
        user_sampler=user_sampler,
        doc_catalogue=doc_catalogue,
        rec_model=content_rec,
        k=100,
    )
    env.reset()
    assert env.curr_user is not None and env.candidate_docs is not None


def test_environment_step(user_sampler, doc_catalogue, content_rec):
    env = MusicGym(
        user_sampler=user_sampler,
        doc_catalogue=doc_catalogue,
        rec_model=content_rec,
        k=100,
    )
    env.reset()
    # create a random torch tensor with 14 elements
    dummy_belief_state = torch.rand(14)
    slate = np.arange(5)
    selected_doc_feature, response, terminated, _, _ = env.step(
        slate=slate, belief_state=dummy_belief_state
    )
    assert selected_doc_feature is not None
    assert response is not None
    assert terminated is not None
