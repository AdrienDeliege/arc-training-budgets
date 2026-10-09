import json
from pathlib import Path
from utils.eval_utils import get_majority_vote


def _resolve_task_path(task_name, task_type, dataset_root=None, eval_split=None):
    default_path = Path("raw_data") / task_type / "data" / "evaluation" / f"{task_name}.json"
    if default_path.exists():
        return default_path

    if dataset_root is None or eval_split is None:
        return default_path

    split_path = Path(dataset_root) / "data" / eval_split
    candidates = []
    if split_path.is_file():
        candidates.append(split_path)
    candidates.append(split_path.with_suffix(".json"))
    candidates.append(split_path / f"{task_name}.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return default_path


def analyze_data(
    answer_set,
    task_names,
    task_type,
    score_subset="test",
    dataset_root=None,
    eval_split=None,
):
    if score_subset not in {"train", "test"}:
        raise ValueError("score_subset must be 'train' or 'test'.")

    oracle_rank = {}
    ground_truths = {}
    tasks_payload = {}
    for task_name in task_names:
        task_path = _resolve_task_path(
            task_name,
            task_type,
            dataset_root=dataset_root,
            eval_split=eval_split,
        )
        with task_path.open("r") as f:
            cur_data = json.load(f)
        score_data = cur_data[score_subset]
        ground_truths[task_name] = {}
        train_examples = [
            {"input": item.get("input"), "output": item.get("output")}
            for item in cur_data.get("train", [])
        ]
        tasks_payload[task_name] = {"examples": {}, "train_examples": train_examples}
        for idx, item in enumerate(score_data):
            example_id = str(idx)
            ground_truths[task_name][example_id] = item['output']
            tasks_payload[task_name]["examples"][example_id] = {
                "input": item.get("input"),
                "answer": item.get("output"),
                "majority_vote": [],
            }

    all_task_num, correct_num_1, correct_num_2, correct_oracle_num = 0, 0, 0, 0

    for task_name in task_names:
        all_task_num += 1
        for cur_index in answer_set[task_name]:
            majority_vote = get_majority_vote(answer_set[task_name][cur_index])
            cur_index = str(cur_index)
            ground_truth = ground_truths[task_name][cur_index]

            pass_at_1 = (len(majority_vote) > 0 and majority_vote[0]["prediction"] == ground_truth)
            if len(majority_vote) > 1:
                pass_at_2 = (majority_vote[1]["prediction"] == ground_truth or pass_at_1)
            else:
                pass_at_2 = pass_at_1
            cur_score = 1 / len(answer_set[task_name])

            if pass_at_1:
                correct_num_1 += cur_score
                correct_num_2 += cur_score
            elif pass_at_2:
                correct_num_2 += cur_score

            oracle_result = False
            for rank, entry in enumerate(majority_vote):
                if entry['prediction'] == ground_truth:
                    oracle_result = True
                    if rank + 1 not in oracle_rank:
                        oracle_rank[rank + 1] = 0
                    oracle_rank[rank + 1] += cur_score
                    break

            if oracle_result:
                correct_oracle_num += cur_score


    pass_at_1_score = correct_num_1 / all_task_num
    pass_at_2_score = correct_num_2 / all_task_num
    oracle_score = correct_oracle_num / all_task_num
   
    oracle_rank_sorted = dict(sorted(oracle_rank.items()))
    print(f"Scoring subset: {score_subset}")
    sum_correct = 0
    for rank, count in oracle_rank_sorted.items():
        print(f"Oracle rank {rank}: {count}. Cumulative: {(sum_correct + count) / (all_task_num):.4f}")
        sum_correct += count
    
    print(f"Final Oracle Score: {oracle_score:.4f}")
    print(f"Final Pass@1 Score: {pass_at_1_score:.4f}")
    print(f"Final Pass@2 Score: {pass_at_2_score:.4f}")
    if pass_at_1_score or pass_at_2_score:
        print(f"Current task {task_names[0]} is Correct!✅")
    else:
        print(f"Current task {task_names[0]} is Wrong!❌")

    return {
        "oracle": oracle_score,
        "pass_at_1": pass_at_1_score,
        "pass_at_2": pass_at_2_score,
        "oracle_rank": oracle_rank_sorted,
        "score_subset": score_subset,
    }
