from myosuite.utils import gym; register=gym.register
from myosuite.envs.env_variants import register_env_variant
import os


def register_env_with_variants(id, entry_point, max_episode_steps, kwargs):
    # register_env_with_variants base env
    register(
        id=id,
        entry_point=entry_point,
        max_episode_steps=max_episode_steps,
        kwargs=kwargs,
        disable_env_checker = True
    )
    # env_names = [spec for spec in gym.envs.registry]
    # for name in sorted(env_names):
    #     print(name)
    #register variants env with sarcopenia
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'sarcopenia'},
            variant_id=id[:3]+"Sarc"+id[3:],
            silent=True
        )
    #register variants with fatigue
    if id[:3] == "myo":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'fatigue'},
            variant_id=id[:3]+"Fati"+id[3:],
            silent=True
        )

    #register variants with tendon transfer
    if id[:7] == "myoHand":
        register_env_variant(
            env_id=id,
            variants={'muscle_condition':'reafferentation'},
            variant_id=id[:3]+"Reaf"+id[3:],
            silent=True
        )


