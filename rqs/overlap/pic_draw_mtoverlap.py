import venn
import json
import os
import time
import matplotlib.pyplot as plt
import matplotlib

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
    mutant_types = ["mutant_1_item", "mutant_2_item", "mutant_3_item"]
    pic_font_size = 30
    title_font_size = 30

    fig = plt.figure(figsize=(18, 12))
    axes = [fig.add_subplot(231), fig.add_subplot(232), fig.add_subplot(233), fig.add_subplot(234), fig.add_subplot(235), fig.add_subplot(236)]

    # 对每个模型，合并两个数据集上的结果后画 Venn 图
    for i, model in enumerate(models):
        mr_aid = set()
        mr_mrd = set()
        mr_sid = set()

        for dataset in datasets:
            fdat_result = read_json(f"./fdat/{dataset}_statistics_result.json")
            model_result = fdat_result[model]["passed"]
            mr_aid.update(model_result["mutant_3_item"])
            mr_mrd.update(model_result["mutant_2_item"])
            mr_sid.update(model_result["mutant_1_item"])

        overlap = venn.get_labels(
            [mr_aid, mr_mrd, mr_sid],
            fill=["number"]
        )

        ax = venn.venn3_ax(
            axes[i], overlap,
            names=["AID", "MRD", "SID"],
            legend=False,
            fontsize=pic_font_size
        )
        ax.set_title(f"Overlap on {model_to_names[model]}", y=-0.07, fontdict={'fontsize': title_font_size})

    plt.tight_layout()
    plt.savefig(f"./results/combined_MR_overlap.pdf")
    plt.close()



