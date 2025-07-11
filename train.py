import os
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader
# from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator, SamModel
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
from torch.nn.functional import threshold, normalize
from torch.nn.functional import sigmoid
from tqdm import tqdm

# ========== 設定 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = 'sam_vit_b_01ec64.pth'
# image_dir = '/nas.dbms/ikuto/FRINet/code/gradio_model_dataset/TrainDataset_AI_car+real500/image'
# image_dir = '/nas.dbms/ikuto/segment-anything/dataset/segany_TrainDataset_car_kaggle/image'
image_dir = '/nas.dbms/ikuto/segment-anything/dataset/segany_TrainDataset_AI_car/image'

# mask_dir = '/nas.dbms/ikuto/FRINet/code/gradio_model_dataset/TrainDataset_AI_car+real500/mask'
# mask_dir = '/nas.dbms/ikuto/segment-anything/dataset/segany_TrainDataset_car_kaggle/mask'
mask_dir = '/nas.dbms/ikuto/segment-anything/dataset/segany_TrainDataset_AI_car/mask'
batch_size = 4
epochs = 100

# ========== データセットクラス ==========
class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size=(384, 384)):  # サイズを指定
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
        self.size = size  # 統一サイズ
        self.to_tensor = T.ToTensor()
        self.resize_image = T.Resize(self.size, interpolation=T.InterpolationMode.BILINEAR)
        self.resize_mask = T.Resize(self.size, interpolation=T.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_name = self.filenames[idx]
        image = Image.open(os.path.join(self.image_dir, image_name)).convert("RGB")
        mask_path = os.path.join(self.mask_dir, image_name)
        mask = Image.open(mask_path).convert("L")

        # 統一サイズにリサイズ
        image = self.resize_image(image)
        mask = self.resize_mask(mask)

        return self.to_tensor(image), self.to_tensor(mask)

# ========== SAM モデルの読み込み ==========
sam_model = sam_model_registry['vit_b'](checkpoint=checkpoint_path).to(device)
sam_model.train()
sam_model.prompt_encoder.train()

# 最適化対象は mask_decoder のみ
optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

# # ========== トレーニングループ ==========
transform = ResizeLongestSide(sam_model.image_encoder.img_size)
dataloader = DataLoader(ImageMaskDataset(image_dir, mask_dir), batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    epoch_loss = 0.0
    count = 0

    for images, gt_masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        # 入力画像を [B, C, H, W] -> [B, H, W, C] に変換し numpy にする
        image_np = images.permute(0, 2, 3, 1).numpy()

        # 各画像ごとにリサイズを適用し、Tensorに変換
        input_images = []
        input_sizes = []
        original_sizes = []

        for i in range(image_np.shape[0]):
            img = image_np[i]
            transformed = transform.apply_image(img)
            input_images.append(torch.as_tensor(transformed.transpose(2, 0, 1)))
            input_sizes.append(transformed.shape[:2])  # H, W
            original_sizes.append(img.shape[:2])       # H, W

        # バッチとしてまとめてTensor化
        input_images = torch.stack(input_images).float().to(device)
        input_images = sam_model.preprocess(input_images)

        with torch.no_grad():
            image_embeddings = sam_model.image_encoder(input_images)

            # バッチ分のボックスを作る
            box_list = [
                torch.tensor([[0, 0, size[1], size[0]]], device=device) for size in input_sizes
            ]

            sparse_embeddings = []
            dense_embeddings = []

            for box in box_list:
                sparse, dense = sam_model.prompt_encoder(points=None, boxes=box, masks=None)
                sparse_embeddings.append(sparse)
                dense_embeddings.append(dense)

        # マスク生成（バッチ処理）
        pred_masks = []
        for i in range(len(image_embeddings)):
            low_res_mask, _ = sam_model.mask_decoder(
                image_embeddings=image_embeddings[i:i+1],
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings[i],
                dense_prompt_embeddings=dense_embeddings[i],
                multimask_output=False,
            )
            # 元の画像サイズに戻す
            upscaled = sam_model.postprocess_masks(
                low_res_mask, input_sizes[i], original_sizes[i]
            ).to(device)

            pred_masks.append(sigmoid(upscaled))

        pred_masks = torch.cat(pred_masks, dim=0)

        # 損失を計算
        gt_masks = gt_masks.to(device)
        if gt_masks.dim() == 3:
            gt_masks = gt_masks.unsqueeze(1)  # [B, 1, H, W]

        loss = loss_fn(pred_masks, gt_masks)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # 不要なテンソルを削除
        del images, gt_masks, input_images, image_embeddings, box_list, sparse_embeddings, dense_embeddings, pred_masks, loss
        torch.cuda.empty_cache() # VRAMキャッシュをクリア
        
        count += 1

    print(f"[Epoch {epoch+1}] Avg Loss: {epoch_loss / count:.4f}")


# ========== モデル保存 ==========
torch.save(sam_model.state_dict(), "segany_1000AIcar+500AIcar_explanation_epoch100_sam_finetuned_vit_b.pth")


