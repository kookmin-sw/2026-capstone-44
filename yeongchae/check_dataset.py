from datasets import load_dataset
from collections import Counter

DATASET_PATH = "/data2/huggingface/AerialVG"

ds = load_dataset(DATASET_PATH)


def analyze_split(split_name):
    has_rel = 0
    no_rel = 0
    rel_counter = Counter()

    for sample in ds[split_name]:
        regions = sample["grounding"]["regions"]

        found_relation = False

        for r in regions:
            rel = r.get("realation", None)   
            if rel is not None and str(rel).strip() != "":
                found_relation = True
                rel_counter[str(rel).strip()] += 1

        if found_relation:
            has_rel += 1
        else:
            no_rel += 1

    total = has_rel + no_rel

    print("=" * 60)
    print(f"[{split_name}]")
    print(f"total samples : {total}")
    print(f"has_relation  : {has_rel}")
    print(f"no_relation   : {no_rel}")

    if total > 0:
        print(f"has_rel %     : {has_rel / total * 100:.2f}%")
        print(f"no_rel %      : {no_rel / total * 100:.2f}%")

    print("\nTop relation types")
    if len(rel_counter) == 0:
        print("  No relation types found.")
    else:
        total_rel_instances = sum(rel_counter.values())
        for rel, cnt in rel_counter.most_common():
            print(f"  {rel:<20} {cnt:>6} ({cnt / total_rel_instances * 100:.2f}%)")

    print()
    return {
        "split": split_name,
        "total": total,
        "has_relation": has_rel,
        "no_relation": no_rel,
        "ratio_has": (has_rel / total * 100) if total > 0 else 0,
        "ratio_no": (no_rel / total * 100) if total > 0 else 0,
        "rel_counter": rel_counter,
    }


all_split_counter = Counter()
all_total = 0
all_has = 0
all_no = 0

results = []

for split in ds.keys():
    result = analyze_split(split)
    results.append(result)

    all_total += result["total"]
    all_has += result["has_relation"]
    all_no += result["no_relation"]
    all_split_counter.update(result["rel_counter"])

print("=" * 60)
print("[ALL SPLITS]")
print(f"total samples : {all_total}")
print(f"has_relation  : {all_has}")
print(f"no_relation   : {all_no}")

if all_total > 0:
    print(f"has_rel %     : {all_has / all_total * 100:.2f}%")
    print(f"no_rel %      : {all_no / all_total * 100:.2f}%")

print("\nTop relation types across all splits")
if len(all_split_counter) == 0:
    print("  No relation types found.")
else:
    total_rel_instances = sum(all_split_counter.values())
    for rel, cnt in all_split_counter.most_common():
        print(f"  {rel:<20} {cnt:>6} ({cnt / total_rel_instances * 100:.2f}%)")