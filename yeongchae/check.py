import os
import json

root = "/data2/huggingface/AerialVG"


def check_relations(data):
    total = 0
    has_rel = 0

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            rels = None
            for k in item.keys():
                if "rel" in k.lower() or "trip" in k.lower():
                    rels = item[k]
                    break

            if rels is None:
                rels = []

            total += 1
            if isinstance(rels, list) and len(rels) > 0:
                has_rel += 1

    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                return check_relations(v)

    return total, has_rel


print("JSON 파일 탐색 중...\n")

for dp, dn, fn in os.walk(root):
    for f in fn:
        if f.endswith(".json"):
            path = os.path.join(dp, f)
            print(f"\n파일: {path}")

            try:
                with open(path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                total, has_rel = check_relations(data)

                if total == 0:
                    print("  -> relation 구조 없음 (skip)")
                    continue

                no_rel = total - has_rel

                print(f"  total        : {total}")
                print(f"  has_relation : {has_rel}")
                print(f"  no_relation  : {no_rel}")
                print(f"  has_rel %    : {has_rel / total * 100:.2f}%")
                print(f"  no_rel %     : {no_rel / total * 100:.2f}%")

            except Exception as e:
                print("  ERROR:", e)
