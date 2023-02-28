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
