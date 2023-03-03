from .run import run as default_run
from .on_off_run import run as on_off_run
from .dop_run import run as dop_run
from .per_run import run as per_run
from .url_run import run as url_run
from .url_load_run import run as url_load_run
from .aps_run import run as aps_run

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["on_off"] = on_off_run
REGISTRY["dop_run"] = dop_run
REGISTRY["per_run"] = per_run
REGISTRY["url_run"] = url_run
REGISTRY["url_load_run"] = url_load_run
REGISTRY["aps_run"] = aps_run
