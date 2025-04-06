import os
from inference_sdk import InferenceHTTPClient
from dotenv import load_dotenv

# .envを読み込む
load_dotenv()

# APIクライアントを作成
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")  # 安全に取得！
)

# 推論関数
# def predict_image(image_path: str) -> str:
#     result = CLIENT.infer(image_path, model_id="hair-vcdau/5")
#     if result.get("predictions"):
#         pred = result["predictions"][0]
#         return f"{pred['class']}（信頼度: {pred['confidence']:.2f}）"
#     return "分類結果なし"

def predict_image(image_path: str) -> str:
    result = CLIENT.infer(image_path, model_id="hair-vcdau/5")
    return str(result)  # ← これで全部表示するようにする