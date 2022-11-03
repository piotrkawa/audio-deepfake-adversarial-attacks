from typing import Dict, Optional, List

import neptune.new as neptune


def get_metric_logger(
    should_log_metrics: bool,
    config: Dict,
    api_token_path: str = "configs/tokens/neptune_api_token",
):
    if should_log_metrics:
        run_mode = "offline"
    else:
        run_mode = "debug"

    return _prepare_neptune_instance(
        config=config,
        run_mode=run_mode,
        api_token_path=api_token_path,
    )

def _prepare_neptune_instance(
    config: Dict,
    run_mode: str,
    api_token_path: str = "configs/tokens/neptune_api_token",
):
    # Preprocess config dict
    neptune_params = config.pop("logging")

    run_id = neptune_params.get("existing_experiment_id", None)
    if config == {}:
        config = None

    neptune_instance = create_neptune_instance(
        name=neptune_params.get("name", None),
        description=neptune_params.get("description", None),
        parameters=config,
        tags=neptune_params.get("tags", None),
        run_id=run_id,
        run_mode=run_mode,
        api_token_path=api_token_path,
    )

    return neptune_instance


def create_neptune_instance(
    name: str,
    description: str,
    parameters: Optional[Dict] = None,
    tags: Optional[List[str]] = None,
    run_id: Optional[str] = None,
    run_mode: str = "async",
    api_token_path: str = "configs/tokens/neptune_api_token",
):
    if parameters is None:
        parameters = {}
    if tags is None:
        tags = []
    neptune_instance = neptune.init(
        project="KPI/aad-plus-plus",
        api_token=load_api_token(api_token_path),
        name=name,
        description=description,
        tags=tags,
        source_files=[],
        mode=run_mode,
        run=run_id,
    )
    neptune_instance["parameters"] = parameters
    return neptune_instance


def load_api_token(
    api_token_path: str = "configs/tokens/neptune_api_token",
) -> str:
    with open(api_token_path, "r") as f:
        return f.readline().strip()
