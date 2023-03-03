import torch as th
import numpy as np

from . import BasicMAC, NMAC


# This URL multi-agent contoller.
class URLMAC(NMAC):
    def __init__(self, scheme, groups, args):
        super(URLMAC, self).__init__(scheme, groups, args)

    def save_models(self, path, mode_id):
        th.save(self.agent.state_dict(), "{}/agent_{}.th".format(path, mode_id))

    def load_models(self, path, mode_id):
        self.agent.load_state_dict(th.load("{}/agent_{}.th".format(path, mode_id), map_location=lambda storage, loc: storage))
