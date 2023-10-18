# from .runner import run_net
from .runner import test_net
# from .runner_pretrain import run_net as pretrain_run_net
from .runner_jepa_pretrain import jepa_run_net as pretrain_jepa_run_net
from .runner_finetune import run_net as finetune_run_net
from .runner_finetune import test_net as test_run_net