#!/usr/bin/env python3
# ДЗ 3. Обучение ранжированию — шаблон решения.
#
# Обучение:  ./solution.py --train --model_file=model.bin PATH-TO-DATA
# Сабмишн:   ./solution.py --model_file=model.bin --submission_file=submission.csv PATH-TO-DATA

import argparse
import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool


SEED = 2026
MAX_ITERATIONS = 1200


def add_query_ranks(data):
    columns = [
        column
        for column in data.columns
        if column not in ("ObjectId", "QueryId", "Label")
        and not column.startswith("q_")
    ]
    ranks = data.groupby("QueryId", sort=False)[columns].rank(
        method="average", pct=True
    )
    ranks.columns = [f"{column}__qrank" for column in columns]
    return pd.concat([data, ranks], axis=1)


def feature_columns(data):
    return [
        column
        for column in data.columns
        if column not in ("ObjectId", "QueryId", "Label")
    ]


def make_pool(data, features):
    return Pool(
        data[features],
        label=data["Label"],
        group_id=data["QueryId"],
    )


def model_params(iterations):
    return dict(
        loss_function="YetiRank",
        eval_metric="NDCG:top=10",
        iterations=iterations,
        learning_rate=0.07,
        depth=6,
        l2_leaf_reg=5.0,
        random_strength=0.5,
        random_seed=SEED,
        thread_count=-1,
        allow_writing_files=False,
        verbose=100,
    )


def train(data_dir, model_file):
    # добавить ранги признаков внутри каждого запроса
    data = add_query_ranks(pd.read_csv(os.path.join(data_dir, "train.csv")))
    features = feature_columns(data)

    # отложить 20% запросов целиком для валидации по nDCG@10
    query_ids = np.random.default_rng(SEED).permutation(data["QueryId"].unique())
    valid_queries = set(query_ids[: len(query_ids) // 5])
    is_valid = data["QueryId"].isin(valid_queries)

    # обучить модель ранжирования и подобрать число итераций
    selector = CatBoostRanker(**model_params(MAX_ITERATIONS))
    selector.fit(
        make_pool(data.loc[~is_valid], features),
        eval_set=make_pool(data.loc[is_valid], features),
        early_stopping_rounds=100,
        use_best_model=True,
    )
    best_iteration = max(selector.get_best_iteration() + 1, 1)
    print(f"selected iterations = {best_iteration}")

    # переобучить модель на всех запросах и сохранить её в model_file
    model = CatBoostRanker(**model_params(best_iteration))
    model.fit(make_pool(data, features))
    model.save_model(model_file)


def predict(data_dir, model_file, submission_file):
    # добавить те же ранги признаков к тестовым данным
    test = add_query_ranks(pd.read_csv(os.path.join(data_dir, "test.csv")))

    # загрузить модель из model_file и посчитать скор для каждой строки
    model = CatBoostRanker()
    model.load_model(model_file)
    scores = model.predict(test[model.feature_names_])

    # сохранить сабмишн в формате ObjectId,Score
    submission = pd.DataFrame({"ObjectId": test["ObjectId"], "Score": scores})
    submission.to_csv(submission_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="Learning-to-rank homework solution")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--submission_file")
    parser.add_argument("data_dir")
    args = parser.parse_args()

    start = timer()
    if args.train:
        train(args.data_dir, args.model_file)
    else:
        if args.submission_file is None:
            parser.error("without --train, --submission_file is required")
        predict(args.data_dir, args.model_file, args.submission_file)
    print(f"finished, elapsed = {timer() - start:.3f}")


if __name__ == "__main__":
    main()
