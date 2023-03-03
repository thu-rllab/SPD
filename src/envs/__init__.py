from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv

from .starcraft import StarCraft2Env
from .matrix_game import OneStepMatrixGame
from .stag_hunt import StagHunt

try:
    gfootball = True
    from .gfootball import GoogleFootballEnv, AcademyPassAndShootWithKeeper, Academy_3_vs_1_with_Keeper
except:
    gfootball = False

try:
    mpe = True
    from .mpe import SimpleTag
except:
    mpe = False

def env_fn(env, **kwargs) -> MultiAgentEnv:
    if kwargs.get('map_name') == "academy_pass_and_shoot_with_keeper":
        return AcademyPassAndShootWithKeeper(**kwargs)
    elif kwargs.get('map_name') == "academy_3_vs_1_with_keeper":
        return Academy_3_vs_1_with_Keeper(**kwargs)
    elif kwargs.get('map_name') == "simple_tag":
        return SimpleTag(**kwargs)
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)

if gfootball:
    REGISTRY["gfootball"] = partial(env_fn, env=GoogleFootballEnv)

if mpe:
    REGISTRY["mpe"] = partial(env_fn, env=None)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
