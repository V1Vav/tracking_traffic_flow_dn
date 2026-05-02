import os

CLASS_MAP = {
    3: 0,
    5: 1,
    4: 2,
    0: 3,
    6: 4,
    1: 4,
    7: 4,
    2: 4
}

label_dirs = [
    "train/labels",
    "valid/labels",
    "test/labels"
]

for label_dir in label_dirs:
    for file in os.listdir(label_dir):
        path = os.path.join(label_dir, file)

        new_lines = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                cls_id = int(parts[0])

                if cls_id in CLASS_MAP:
                    parts[0] = str(CLASS_MAP[cls_id])
                    new_lines.append(" ".join(parts))

        with open(path, "w") as f:
            f.write("\n".join(new_lines))