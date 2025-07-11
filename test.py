import os
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from sklearn.metrics import f1_score
import numpy as np

# ========== 設定 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = '/nas.dbms/ikuto/segment-anything/epoch100_sam_finetuned_vit_b.pth'  # 学習済みモデル
# checkpoint_path = '/nas.dbms/ikuto/segment-anything/sam_vit_b_01ec64.pth'  # 学習済みモデル
image_dir = '/nas.dbms/ikuto/segment-anything/image'
mask_dir = '/nas.dbms/ikuto/segment-anything/mask'
output_mask_dir = './predicted_masks100'
# output_mask_dir = './predicted_masks_base'
os.makedirs(output_mask_dir, exist_ok=True)
result_txt_path = './test_results100.txt'
# result_txt_path = './test_results_base.txt'
batch_size = 1

# ========== データセット ==========
class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_name = self.filenames[idx]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        mask_path = os.path.join(self.mask_dir, image_name)
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize(image.size, resample=Image.NEAREST)
        return image_name, self.to_tensor(image), self.to_tensor(mask)

# ========== モデル準備 ==========
sam_model = sam_model_registry['vit_b'](checkpoint=checkpoint_path).to(device)
sam_model.eval()  # 評価モード

transform = ResizeLongestSide(sam_model.image_encoder.img_size)
dataloader = DataLoader(ImageMaskDataset(image_dir, mask_dir), batch_size=batch_size, shuffle=False)

# ========== 評価関数 ==========
def compute_iou(pred_mask, gt_mask):
    pred = pred_mask > 0.5
    gt = gt_mask > 0.5
    intersection = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    if union == 0:
        return 1.0  # 完全に空マスクの場合はIoU=1とする
    return intersection / union

def compute_f1(pred_mask, gt_mask):
    pred = pred_mask.view(-1) > 0.5
    gt = gt_mask.view(-1) > 0.5
    return f1_score(gt.cpu().numpy(), pred.cpu().numpy())

# ========== テストループ ==========
ious = []
f1s = []
results_per_image = []  # ここに1枚ごとの結果を保存

with torch.no_grad():
    for image_name, image_tensor, gt_mask_tensor in dataloader:
        image_name = image_name[0]
        image = image_tensor[0].numpy().transpose(1, 2, 0)  # [C,H,W] → [H,W,C]
        gt_mask = gt_mask_tensor[0].to(device)

        # 入力画像の前処理
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image.transpose(2, 0, 1)).float().to(device)
        input_image_torch = sam_model.preprocess(input_image_torch)[None, :, :, :]

        original_size = image.shape[:2]
        input_size = tuple(input_image.shape[:2])

        # エンコーダー部分
        image_embedding = sam_model.image_encoder(input_image_torch)
        box_torch = torch.tensor([[0, 0, input_size[1], input_size[0]]], device=device)
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )

        # マスク生成
        low_res_masks, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_size).to(device)

        # 0.5閾値で二値化
        pred_mask = (upscaled_masks > 0.5).float()

        # IoU と F1計算
        iou = compute_iou(pred_mask, gt_mask)
        f1 = compute_f1(pred_mask, gt_mask)

        ious.append(iou)
        f1s.append(f1)

        # 1枚ごとの結果を文字列で保存
        results_per_image.append(f"{image_name}: IoU={iou:.4f}, F1={f1:.4f}")

        # 予測マスク画像保存（PIL画像で保存）
        pred_mask_img = (pred_mask[0, 0].cpu().numpy() * 255).astype('uint8')
        pred_mask_pil = Image.fromarray(pred_mask_img)
        pred_mask_pil.save(os.path.join(output_mask_dir, image_name))

        print(results_per_image[-1])

# ========== 平均スコアと結果保存 ==========
mean_iou = np.mean(ious)
mean_f1 = np.mean(f1s)

with open(result_txt_path, 'w') as f:
    # 1枚ごとの結果をすべて書き込み
    for line in results_per_image:
        f.write(line + "\n")
    # 平均スコアも書き込み
    f.write(f"\nMean IoU: {mean_iou:.4f}\n")
    f.write(f"Mean F1 Score: {mean_f1:.4f}\n")

print(f"Mean IoU: {mean_iou:.4f}")
print(f"Mean F1 Score: {mean_f1:.4f}")
print(f"Results saved to {result_txt_path}")