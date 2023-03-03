REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .url_runner import URLRunner
REGISTRY["url"] = URLRunner

from .url_load_runner import URLLoadRunner
REGISTRY["url_load"] = URLLoadRunner

from .url_evaluator import URLEvaluator
REGISTRY["url_eval"] = URLEvaluator
