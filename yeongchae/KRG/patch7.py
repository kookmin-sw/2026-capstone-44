with open("visualize_results.py", "r") as f:
    code = f.read()

old = '''    for ax, (method, box, iou, color, label) in zip(axes, [
        ("GroundingDINO (Baseline)", sample["baseline_box"], sample["baseline_iou"], BLUE,   "Baseline"),
        ("KRG (Ours)",              sample["ours_box"],      sample["ours_iou"],     ORANGE, "KRG"),
    ]):
        ax.imshow(img_arr)
        ax.set_facecolor(BG)
        draw_bbox(ax, sample["gt_box"], img_w, img_h, GREEN, "GT", 2.0)
        draw_bbox(ax, box, img_w, img_h, color, label, 2.5)'''

new = '''    for ax, (method, box, iou, color, label) in zip(axes, [
        ("GroundingDINO (Baseline)", sample["baseline_box"], sample["baseline_iou"], BLUE,   "Baseline"),
        ("KRG (Ours)",              sample["ours_box"],      sample["ours_iou"],     ORANGE, "KRG"),
    ]):
        ax.imshow(img_arr)
        ax.set_facecolor(BG)
        draw_bbox(ax, sample["gt_box"], img_w, img_h, GREEN, "GT", 2.0)
        draw_bbox(ax, box, img_w, img_h, color, label, 2.5)
        # GT 박스 항상 양쪽 모두 표시 (이미 위에서 그림)'''

with open("visualize_results.py", "w") as f:
    f.write(code)

print("done")
