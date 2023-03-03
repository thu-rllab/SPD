# import buffer
# import disc
# import model
# import util
from .wurl import apwd
from .gwd import gwd 
from .diayn import diayn 


REGISTRY = {}
REGISTRY["wurl"] = apwd.assign_reward
REGISTRY["gwd"] = gwd.assign_reward
REGISTRY["diayn"] = diayn.assign_reward

