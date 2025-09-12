import venn
import json
import os
import time
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


# plt.rcParams['font.size'] = 26

pic_font_size = 25

def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)
    
def write_json(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=3)

def write_text(string_text, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(string_text)


if __name__ == "__main__":
    datasets = ["humaneval", "mbpp-sanitized"]
    models = ["passat10_incoder-1B","passat10_codegen-2B", "passat10_codegen2-1B",  "passat10_santacoder", "passat10_llama3-1b", "passat10_chatgpt"]
    model_to_names = {
        "passat10_incoder-1B": "Incoder",
        "passat10_codegen-2B": "CodeGen",
        "passat10_codegen2-1B": "CodeGen2",
        "passat10_santacoder": "Santacoder",
        "passat10_llama3-1b": "Llama",
        "passat10_chatgpt": "ChatGPT"
    }
    baseline_types = ["base", "comment", "insert_line", "output_mutation", "output_v_mutation"]
    baseline_names = ["Base", "Comment", "Insert Line", "PPM-T", "PPM-V"]
    mutant_types = ["mutant_1_item", "mutant_2_item", "mutant_3_item"]
    colors = ['#5f9d6d', '#cb8863', '#5875a3']
    bar_width = 0.6
    label_font_size = 15
    tick_font_size = 20
    pic_font_size = 20
    title_font_size = 25

    fig = plt.figure(figsize=(18, 12))
    axes = [fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233), fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)]

    # 处理每个模型
    for i, model in enumerate(models):
        ax = axes[i]
        combined_ours = set()
        combined_baselines = {bt: set() for bt in baseline_types}

        for dataset in datasets:
            fdat_result = read_json(f"./fdat/{dataset}_statistics_result.json")
            baselines_result = read_json(f"./baselines/{dataset}_statistics_result.json")

            # ours
            model_result = fdat_result[model]["passed"]
            combined_ours.update(
                model_result["mutant_1_item"] +
                model_result["mutant_2_item"] +
                model_result["mutant_3_item"]
            )

            # baselines
            baseline_result = baselines_result[model]
            for bt in baseline_types:
                combined_baselines[bt].update(baseline_result[bt]["passed"])

        # 计算 overlap
        overlap_counts = []
        for bt in baseline_types:
            ours = combined_ours
            theirs = combined_baselines[bt]

            only_ours = len(ours - theirs)
            only_theirs = len(theirs - ours)
            both = len(ours & theirs)

            overlap_counts.append([only_ours, both, only_theirs])

        overlap_array = np.array(overlap_counts)
        bottoms = np.zeros(len(baseline_types))
        x = np.arange(len(baseline_types))

        for j in range(3):
            bars = ax.bar(
                x,
                overlap_array[:, j],
                bottom=bottoms,
                width=bar_width,
                color=colors[j],
                edgecolor='white',
                linewidth=1.5,
                label=['Only Ours', 'Both', 'Only Baseline'][j] if i == 0 else None
            )

            # 添加数值标注
            for k, bar in enumerate(bars):
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + height / 2,
                        str(int(height)),
                        ha='center',
                        va='center',
                        fontsize=tick_font_size - 5,
                        color='white',
                        fontweight='bold'
                    )

            bottoms += overlap_array[:, j]

        ax.set_xticks(x)
        ax.set_xticklabels(baseline_names, rotation=30, fontsize=tick_font_size)
        ax.tick_params(axis='y', labelsize=tick_font_size)
        ax.set_ylabel("Count", fontsize=tick_font_size)
        ax.set_title(f"Overlap on {model_to_names[model]}", fontsize=title_font_size, y=1.02)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # 添加图例（只画一次）
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',bbox_to_anchor=(0.5, 1), ncol=3, fontsize=label_font_size)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"./results/combined_overlap_distribution.pdf")
    plt.show()
