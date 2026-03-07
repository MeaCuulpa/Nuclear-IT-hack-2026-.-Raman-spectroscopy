from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from typing import Any

import pandas as pd

from evaluation import evaluate_group_cv
from models import get_center_training_cfg, is_model_enabled_for_center, make_single_model_for_center
from utils import save_json

SUPPORTED_PBT_MODELS = ("extra_trees", "catboost", "pls_logreg")

DEFAULT_PBT_SEARCH_SPACES = {
    "extra_trees": {
        "n_estimators": [400, 600, 900, 1200, 1500],
        "min_samples_split": [2, 4, 6, 8],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", 0.5, 0.75],
        "class_weight": ["balanced", "balanced_subsample"],
    },
    "catboost": {
        "iterations": [250, 400, 600, 800],
        "depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
        "l2_leaf_reg": [3.0, 5.0, 7.0, 9.0],
        "subsample": [0.7, 0.8, 0.9, 1.0],
    },
    "pls_logreg": {
        "n_components": [2, 3, 5, 8, 10, 15, 20],
        "C": [0.1, 0.3, 0.7, 1.0, 2.0, 5.0, 10.0],
        "class_weight": ["balanced", None],
    },
}


def _get_nested(node, path: str, default=None):
    current = node
    for part in path.split('.'):
        if current is None or not hasattr(current, part):
            return default
        current = getattr(current, part)
    return current


def _node_to_dict(node: Any):
    if isinstance(node, list):
        return [_node_to_dict(item) for item in node]
    if hasattr(node, '__dict__'):
        return {key: _node_to_dict(value) for key, value in vars(node).items()}
    return node


def _normalize_search_space(raw_space: Any) -> dict[str, list]:
    if raw_space is None:
        return {}

    raw_dict = _node_to_dict(raw_space)
    normalized = {}
    for param_name, spec in raw_dict.items():
        if isinstance(spec, dict) and 'values' in spec:
            values = list(spec['values'])
        elif isinstance(spec, list):
            values = list(spec)
        else:
            values = [spec]
        if not values:
            continue
        normalized[str(param_name)] = values
    return normalized


def _get_pbt_cfg(config, center: str):
    center_cfg = get_center_training_cfg(config, center)
    return getattr(center_cfg, 'pbt', None) if center_cfg is not None else None


def _search_space_for_model(config, center: str, model_name: str) -> dict[str, list]:
    pbt_cfg = _get_pbt_cfg(config, center)
    custom_spaces = _get_nested(pbt_cfg, f'search_spaces.{model_name}', None)
    space = _normalize_search_space(custom_spaces)
    if not space:
        space = deepcopy(DEFAULT_PBT_SEARCH_SPACES.get(model_name, {}))
    return space


def _base_params_from_config(config, center: str, model_name: str) -> dict:
    center_cfg = get_center_training_cfg(config, center)
    model_cfg = getattr(center_cfg, model_name, None) if center_cfg is not None else None
    base = {}
    if model_cfg is not None:
        for key, value in vars(model_cfg).items():
            base[str(key)] = value
    return base


def _sample_from_values(values: list, rng: random.Random):
    return deepcopy(values[rng.randrange(len(values))])


def _mutate_value(current_value, values: list, rng: random.Random):
    if len(values) == 1:
        return deepcopy(values[0])
    try:
        idx = values.index(current_value)
        neighbor_indices = [candidate for candidate in (idx - 1, idx + 1) if 0 <= candidate < len(values)]
        if neighbor_indices and rng.random() < 0.7:
            return deepcopy(values[rng.choice(neighbor_indices)])
    except ValueError:
        pass

    candidates = [value for value in values if value != current_value]
    if not candidates:
        candidates = values
    return deepcopy(candidates[rng.randrange(len(candidates))])


def _sample_candidate(base_params: dict, search_space: dict[str, list], rng: random.Random) -> dict:
    candidate = deepcopy(base_params)
    for param_name, values in search_space.items():
        if param_name not in candidate:
            candidate[param_name] = _sample_from_values(values, rng)
    return candidate


def _mutate_candidate(parent: dict, search_space: dict[str, list], rng: random.Random, mutation_prob: float) -> dict:
    child = deepcopy(parent)
    for param_name, values in search_space.items():
        if rng.random() < mutation_prob or param_name not in child:
            child[param_name] = _mutate_value(child.get(param_name), values, rng)
    return child


def _candidate_key(model_name: str, params: dict) -> str:
    return f"{model_name}::" + json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)


def _evaluate_candidate(
    config,
    center: str,
    model_name: str,
    params: dict,
    x,
    y,
    groups,
    sample_ids,
    n_splits: int,
    cv_strategy: str,
    random_state: int,
) -> dict:
    selected_pls_n = params.get('n_components') if model_name == 'pls_logreg' else None
    model = make_single_model_for_center(
        config,
        center=center,
        model_name=model_name,
        selected_pls_n_components=selected_pls_n,
        overrides=params,
    )
    results, results_df, _ = evaluate_group_cv(
        x,
        y,
        groups,
        {model_name: model},
        n_splits=n_splits,
        dataset_name=f'pbt_{model_name}_center{center}',
        cv_strategy=cv_strategy,
        random_state=random_state,
        sample_ids=sample_ids,
        ensemble_config=None,
        verbose=False,
    )
    row = results[0]
    compact = {
        'acc_mean': float(row['acc_mean']),
        'bacc_mean': float(row['bacc_mean']),
        'macro_f1_mean': float(row['macro_f1_mean']),
        'oof_acc': float(row['oof_acc']),
        'oof_bacc': float(row['oof_bacc']),
        'oof_macro_f1': float(row['oof_macro_f1']),
    }
    return {
        'params': deepcopy(params),
        'score': float(row['oof_macro_f1']),
        'result': compact,
    }


def run_pbt_for_model(
    config,
    center: str,
    model_name: str,
    x,
    y,
    groups,
    sample_ids,
    center_output_dir,
    n_splits: int,
    cv_strategy: str,
    random_state: int,
) -> dict | None:
    pbt_cfg = _get_pbt_cfg(config, center)
    if pbt_cfg is None or not bool(getattr(pbt_cfg, 'enabled', False)):
        return None
    if not is_model_enabled_for_center(config, center, model_name, default=False):
        return None
    if model_name not in SUPPORTED_PBT_MODELS:
        return None

    requested_models = getattr(pbt_cfg, 'models', list(SUPPORTED_PBT_MODELS))
    requested_models = [str(name) for name in requested_models]
    if requested_models and model_name not in requested_models:
        return None

    search_space = _search_space_for_model(config, center, model_name)
    if not search_space:
        return None

    population_size = max(2, int(getattr(pbt_cfg, 'population_size', 6)))
    generations = max(1, int(getattr(pbt_cfg, 'generations', 3)))
    top_fraction = float(getattr(pbt_cfg, 'exploit_top_fraction', 0.5))
    mutation_prob = float(getattr(pbt_cfg, 'mutation_prob', 0.8))
    rng = random.Random(int(getattr(pbt_cfg, 'random_state', random_state)) + hash((center, model_name)) % 10000)

    base_params = _sample_candidate(_base_params_from_config(config, center, model_name), search_space, rng)
    population = [deepcopy(base_params)]
    while len(population) < population_size:
        candidate = _mutate_candidate(base_params, search_space, rng, mutation_prob=1.0)
        population.append(candidate)

    history = []
    cache = {}
    best_eval = None

    print(f"[PBT] center={center} | model={model_name} | population={population_size} | generations={generations}")
    for generation in range(1, generations + 1):
        evaluated = []
        for candidate_idx, params in enumerate(population, start=1):
            key = _candidate_key(model_name, params)
            if key not in cache:
                cache[key] = _evaluate_candidate(
                    config=config,
                    center=center,
                    model_name=model_name,
                    params=params,
                    x=x,
                    y=y,
                    groups=groups,
                    sample_ids=sample_ids,
                    n_splits=n_splits,
                    cv_strategy=cv_strategy,
                    random_state=random_state,
                )
            result = deepcopy(cache[key])
            result['generation'] = generation
            result['candidate_idx'] = candidate_idx
            evaluated.append(result)
            history.append({
                'generation': generation,
                'candidate_idx': candidate_idx,
                'model': model_name,
                'center': center,
                **result['params'],
                **result['result'],
            })

        evaluated.sort(key=lambda item: item['score'], reverse=True)
        generation_best = evaluated[0]
        print(
            f"[PBT] center={center} | model={model_name} | generation={generation} | "
            f"best_oof_macro_f1={generation_best['score']:.4f} | params={generation_best['params']}"
        )
        if best_eval is None or generation_best['score'] > best_eval['score']:
            best_eval = deepcopy(generation_best)

        elite_count = max(1, math.ceil(population_size * top_fraction))
        elites = [deepcopy(item['params']) for item in evaluated[:elite_count]]
        next_population = elites[:]

        while len(next_population) < population_size:
            if rng.random() < 0.15:
                child = _mutate_candidate(base_params, search_space, rng, mutation_prob=1.0)
            else:
                parent = deepcopy(elites[rng.randrange(len(elites))])
                child = _mutate_candidate(parent, search_space, rng, mutation_prob=mutation_prob)
            next_population.append(child)
        population = next_population

    history_df = pd.DataFrame(history).sort_values(['oof_macro_f1', 'generation'], ascending=[False, True]).reset_index(drop=True)
    csv_path = center_output_dir / f'pbt_{model_name}_center{center}.csv'
    json_path = center_output_dir / f'pbt_{model_name}_center{center}.json'
    history_df.to_csv(csv_path, index=False)

    payload = {
        'model': model_name,
        'center': center,
        'best_params': best_eval['params'] if best_eval is not None else {},
        'best_score': float(best_eval['score']) if best_eval is not None else None,
        'population_size': population_size,
        'generations': generations,
        'search_space': search_space,
        'history_csv': csv_path.name,
    }
    save_json(payload, json_path)
    return payload


def run_pbt_for_center(
    config,
    center: str,
    x,
    y,
    groups,
    sample_ids,
    center_output_dir,
    n_splits: int,
    cv_strategy: str,
    random_state: int,
) -> tuple[dict[str, dict], dict | None]:
    pbt_cfg = _get_pbt_cfg(config, center)
    if pbt_cfg is None or not bool(getattr(pbt_cfg, 'enabled', False)):
        return {}, None

    tuned_params = {}
    payload = {
        'enabled': True,
        'population_size': int(getattr(pbt_cfg, 'population_size', 6)),
        'generations': int(getattr(pbt_cfg, 'generations', 3)),
        'models': {},
    }

    for model_name in SUPPORTED_PBT_MODELS:
        result = run_pbt_for_model(
            config=config,
            center=center,
            model_name=model_name,
            x=x,
            y=y,
            groups=groups,
            sample_ids=sample_ids,
            center_output_dir=center_output_dir,
            n_splits=n_splits,
            cv_strategy=cv_strategy,
            random_state=random_state,
        )
        if result is None:
            continue
        tuned_params[model_name] = deepcopy(result['best_params'])
        payload['models'][model_name] = result

    if not payload['models']:
        return {}, None
    return tuned_params, payload
