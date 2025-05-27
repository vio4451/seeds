from flask import Flask, request, render_template, jsonify
import torch
import torch.nn.functional as F
from PIL import Image
import io
import json
from torchvision import transforms

app = Flask(__name__)

# 加载 TorchScript 模型
model = torch.jit.load("best_mobilenet_v2.pt")
model.eval()

# 植物学知识库
with open("botanical_knowledge.json", encoding="utf-8") as f:
    knowledge_db = json.load(f)

# 科学级图像预处理流程
scientific_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



# 植物分类学映射
taxonomic_hierarchy = {
    0: {'中文名': '凤凰木', '科': '豆科', '属': '凤凰木属', '拉丁名': 'Delonix regia'},
    1: {'中文名': '刺槐', '科': '豆科', '属': '刺槐属', '拉丁名': 'Robinia pseudoacacia'},
    2: {'中文名': '合欢', '科': '豆科', '属': '合欢属', '拉丁名': 'Albizia julibrissin'},
    3: {'中文名': '含羞草', '科': '豆科', '属': '含羞草属', '拉丁名': 'Mimosa pudica'},
    4: {'中文名': '四棱豆', '科': '豆科', '属': '四棱豆属', '拉丁名': 'Securigera varia'},
    5: {'中文名': '国槐', '科': '豆科', '属': '槐属', '拉丁名': 'Sophora japonica'},
    6: {'中文名': '多头向日葵', '科': '豆科', '属': '向日葵属', '拉丁名': 'Helianthus multiflorus'},
    7: {'中文名': '多花木兰', '科': '木兰科', '属': '木兰属', '拉丁名': 'Magnolia multiflora'},
    8: {'中文名': '大叶相思', '科': '豆科', '属': '相思属', '拉丁名': 'Acacia mangium'},
    9: {'中文名': '大叶莴苣', '科': '菊科', '属': '莴苣属', '拉丁名': 'Lactuca sativa var. angustana'},
    10: {'中文名': '大叶菊苣', '科': '菊科', '属': '菊苣属', '拉丁名': 'Cichorium intybus'},
    11: {'中文名': '大皂角', '科': '豆科', '属': '皂荚属', '拉丁名': 'Gleditsia sinensis'},
    12: {'中文名': '孔雀豆', '科': '豆科', '属': '豆属', '拉丁名': 'Cyamopsis tetragonoloba'},
    13: {'中文名': '小丽花', '科': '菊科', '属': '丽花属', '拉丁名': 'Tagetes patula'},
    14: {'中文名': '小冠花', '科': '豆科', '属': '冠花属', '拉丁名': 'Stylosanthes guianensis'},
    15: {'中文名': '小扁豆', '科': '豆科', '属': '扁豆属', '拉丁名': 'Vigna unguiculata'},
    16: {'中文名': '巨紫荆', '科': '豆科', '属': '紫荆属', '拉丁名': 'Bauhinia gigantea'},
    17: {'中文名': '巴山豆', '科': '豆科', '属': '豆属', '拉丁名': 'Sophora tonkinensis'},
    18: {'中文名': '幸运草', '科': '豆科', '属': '苜蓿属', '拉丁名': 'Trifolium repens'},
    19: {'中文名': '扁豆', '科': '豆科', '属': '扁豆属', '拉丁名': 'Lablab purpureus'},
    20: {'中文名': '拉巴豆', '科': '豆科', '属': '豆属', '拉丁名': 'Lathyrus sativus'},
    21: {'中文名': '斑马豆', '科': '豆科', '属': '豆属', '拉丁名': 'Phaseolus coccineus'},
    22: {'中文名': '木耳菜', '科': '苋科', '属': '木耳菜属', '拉丁名': 'Basella alba'},
    23: {'中文名': '木豆', '科': '豆科', '属': '木豆属', '拉丁名': 'Cajanus cajan'},
    24: {'中文名': '柠条', '科': '豆科', '属': '柠条属', '拉丁名': 'Caragana microphylla'},
    25: {'中文名': '榼藤子', '科': '豆科', '属': '藤属', '拉丁名': 'Vigna vexillata'},
    26: {'中文名': '油麻藤', '科': '豆科', '属': '油麻藤属', '拉丁名': 'Clitoria ternatea'},
    27: {'中文名': '洋金凤', '科': '豆科', '属': '金凤花属', '拉丁名': 'Caesalpinia pulcherrima'},
    28: {'中文名': '海红豆', '科': '豆科', '属': '海红豆属', '拉丁名': 'Abrus precatorius'},
    29: {'中文名': '甘草', '科': '豆科', '属': '甘草属', '拉丁名': 'Glycyrrhiza uralensis'},
    30: {'中文名': '白三叶', '科': '豆科', '属': '苜蓿属', '拉丁名': 'Trifolium repens'},
    31: {'中文名': '白刀豆', '科': '豆科', '属': '刀豆属', '拉丁名': 'Canavalia gladiata'},
    32: {'中文名': '白扁豆', '科': '豆科', '属': '扁豆属', '拉丁名': 'Lablab purpureus'},
    33: {'中文名': '眉豆', '科': '豆科', '属': '眉豆属', '拉丁名': 'Vigna angularis'},
    34: {'中文名': '秋英', '科': '菊科', '属': '秋英属', '拉丁名': 'Aster tataricus'},
    35: {'中文名': '竹豆', '科': '豆科', '属': '豆属', '拉丁名': 'Canavalia ensiformis'},
    36: {'中文名': '紫云英', '科': '豆科', '属': '苜蓿属', '拉丁名': 'Astragalus sinicus'},
    37: {'中文名': '紫穗槐', '科': '豆科', '属': '槐属', '拉丁名': 'Sesbania cannabina'},
    38: {'中文名': '紫色向日葵', '科': '菊科', '属': '向日葵属', '拉丁名': 'Helianthus annuus var. purple'},
    39: {'中文名': '紫藤', '科': '豆科', '属': '紫藤属', '拉丁名': 'Wisteria sinensis'},
    40: {'中文名': '红豆草', '科': '豆科', '属': '红豆草属', '拉丁名': 'Aeschynomene indica'},
    41: {'中文名': '绿苑子', '科': '豆科', '属': '苑子属', '拉丁名': 'Medicago sativa'},
    42: {'中文名': '羽扇豆', '科': '豆科', '属': '羽扇豆属', '拉丁名': 'Lupinus polyphyllus'},
    43: {'中文名': '翠菊', '科': '菊科', '属': '翠菊属', '拉丁名': 'Chrysanthemum coronarium'},
    44: {'中文名': '芦巴子', '科': '豆科', '属': '芦巴属', '拉丁名': 'Trigonella foenum-graecum'},
    45: {'中文名': '花环菊', '科': '菊科', '属': '菊属', '拉丁名': 'Chrysanthemum morifolium'},
    46: {'中文名': '花芸豆', '科': '豆科', '属': '芸豆属', '拉丁名': 'Phaseolus vulgaris'},
    47: {'中文名': '苜蓿', '科': '豆科', '属': '苜蓿属', '拉丁名': 'Medicago sativa'},
    48: {'中文名': '苦豆子', '科': '豆科', '属': '苦豆属', '拉丁名': 'Tephrosia vogelii'},
    49: {'中文名': '草木樨', '科': '豆科', '属': '草木樨属', '拉丁名': 'Lespedeza bicolor'},
    50: {'中文名': '荷包豆', '科': '豆科', '属': '荷包豆属', '拉丁名': 'Erythrina variegata'},
    51: {'中文名': '莴苣', '科': '菊科', '属': '莴苣属', '拉丁名': 'Lactuca sativa'},
    52: {'中文名': '蚕豆', '科': '豆科', '属': '蚕豆属', '拉丁名': 'Vicia faba'},
    53: {'中文名': '补骨脂', '科': '豆科', '属': '补骨脂属', '拉丁名': 'Psoralea corylifolia'},
    54: {'中文名': '豆薯', '科': '豆科', '属': '豆薯属', '拉丁名': 'Pachyrhizus erosus'},
    55: {'中文名': '豇豆', '科': '豆科', '属': '豇豆属', '拉丁名': 'Vigna unguiculata'},
    56: {'中文名': '豌豆', '科': '豆科', '属': '豌豆属', '拉丁名': 'Pisum sativum'},
    57: {'中文名': '鄂西红豆', '科': '豆科', '属': '红豆属', '拉丁名': 'Abrus pulchellus'},
    58: {'中文名': '金丝豆', '科': '豆科', '属': '金丝豆属', '拉丁名': 'Clitoria ternatea'},
    59: {'中文名': '金合欢', '科': '豆科', '属': '相思属', '拉丁名': 'Acacia dealbata'},
    60: {'中文名': '银合欢', '科': '豆科', '属': '合欢属', '拉丁名': 'Leucaena leucocephala'},
    61: {'中文名': '音符豆', '科': '豆科', '属': '豆属', '拉丁名': 'Phaseolus lunatus'},
    62: {'中文名': '骆驼刺', '科': '豆科', '属': '骆驼刺属', '拉丁名': 'Alhagi sparsifolia'},
    63: {'中文名': '鸡冠刺桐', '科': '豆科', '属': '刺桐属', '拉丁名': 'Erythrina crista-galli'},
    64: {'中文名': '麻豌豆', '科': '豆科', '属': '豌豆属', '拉丁名': 'Pisum sativum var. arvense'},
    65: {'中文名': '黄花梨', '科': '豆科', '属': '黄花梨属', '拉丁名': 'Dalbergia odorifera'},
    66: {'中文名': '黄花槐', '科': '豆科', '属': '槐属', '拉丁名': 'Sophora flavescens'},
}


def scientific_analysis(image_bytes):
    """执行科学级图像分析"""
    try:
        tensor = scientific_transform(
            Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)

        probs = F.softmax(logits, dim=1).squeeze()
        top_probs, top_indices = torch.topk(probs, 3)

        return [{
            'prediction': taxonomic_hierarchy.get(idx.item(), {'中文名': f'未知({idx.item()})'}),
            'probability': f"{prob.item():.2%}",
            'confidence': prob.item(),
            'morphology': knowledge_db.get(str(idx.item()), {}).get('morphology'),
            'ecology': knowledge_db.get(str(idx.item()), {}).get('ecology')
        } for prob, idx in zip(top_probs, top_indices)]

    except Exception as e:
        raise RuntimeError(f"科学分析失败: {str(e)}")

@app.route("/", methods=["GET", "POST"])
def research_dashboard():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"status": "error", "message": "未检测到样本"}), 400

        try:
            results = scientific_analysis(request.files["file"].read())
            return jsonify({
                "status": "success",
                "analysis": results,
                "taxonomic_tree": build_taxonomic_tree(results[0]["prediction"]["科"])
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return render_template("index.html")

def build_taxonomic_tree(family):
    """构建分类学树状结构"""
    return {
        "科": family,
        "属": list({v["属"] for v in taxonomic_hierarchy.values() if v["科"] == family}),
        "种": [v["中文名"] for v in taxonomic_hierarchy.values() if v["科"] == family]
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
