import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import torch.nn as nn
import torch.nn.functional as F

import os
import pandas as pd

from transformers import AutoTokenizer, CLIPTextModel, AutoModel


tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")


class Discriminator(nn.Module):
    def __init__(self, img_backbone, img_embed_dim, text_encoder, text_embed_dim, num_classes=None):
        super().__init__()
        self.img_backbone = img_backbone
        self.text_encoder = text_encoder
        self.num_classes = num_classes
        self.cross_attn = nn.MultiheadAttention(embed_dim=text_embed_dim, num_heads=4, batch_first=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=text_embed_dim, num_heads=4, batch_first=True)
        self.img_embedding = nn.Linear(img_embed_dim, text_embed_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(text_embed_dim, num_classes)
    
    def forward(self, img, tokenized_caption, caption_attention_mask):
        img_tensor = self.relu(self.img_backbone(img))
        img_tensor_reshaped = img_tensor.view(img_tensor.size(0), img_tensor.size(1), -1)
        img_tensor_reshaped = img_tensor_reshaped.permute(0, 2, 1)
        img_tensor_proj = self.img_embedding(img_tensor_reshaped)

        text_tensor = self.text_encoder(tokenized_caption).last_hidden_state
        text_tensor, _ = self.self_attn(query=text_tensor,
                                        key=text_tensor,
                                        value=text_tensor,
                                        key_padding_mask=caption_attention_mask)
        
        img_txt_tensor, _ = self.cross_attn(query=img_tensor_proj,
                                            key=text_tensor,
                                            value=text_tensor,
                                            key_padding_mask=caption_attention_mask)
        img_tensor_proj = img_tensor_proj + img_txt_tensor
        #avg pooling
        pooled_img_tensor = torch.mean(img_tensor_proj, dim=1)
        pooled_img_tensor = self.relu(pooled_img_tensor)
        logits = self.classifier(pooled_img_tensor)
        return logits


class BinaryClassImageTxtDataset(Dataset):
    def __init__(self, img_df, transform=None):
        self.img_df = img_df
        self.img_path_list = self.img_df['img_path'].tolist()
        self.img_lbl_list = self.img_df['label'].tolist()
        self.caption_list = self.img_df['caption'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        label = self.img_lbl_list[idx]
        caption = self.caption_list[idx]
        img_path = self.img_path_list[idx]
        image = read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        if self.transform:
            image = self.transform(image)
        return image, label, caption
    

def pad_batch(batch):
    batch_imgs = torch.stack([b[0] for b in batch])
    batch_lbls = [b[1] for b in batch]
    batch_txt = [b[2] for b in batch]
    tokenized_batch_txt = tokenizer(batch_txt, padding=True, truncation=True)
    return {
        'image': batch_imgs,
        'labels': torch.tensor(batch_lbls).to(torch.float),
        'text_input_ids': torch.tensor(tokenized_batch_txt['input_ids']),
        'text_attention_mask': torch.tensor(tokenized_batch_txt['attention_mask']).to(torch.float)
    }
    

def train_batch(batch, optimizer, loss_fn, model):
    imgs, lbls, input_ids, attention_masks = batch['image'], batch['labels'], batch['text_input_ids'], batch['text_attention_mask']
    imgs, lbls, input_ids, attention_masks = imgs.to('cuda'), lbls.to('cuda'), input_ids.to('cuda'), attention_masks.to('cuda')
    optimizer.zero_grad()
    preds = model(imgs, input_ids, attention_masks).squeeze()
    loss = loss_fn(preds, lbls)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()  # Disable gradient computation during evaluation
def evaluate(model, loader, loss_fn):
    correct = 0
    total = 0
    total_loss = 0.
    
    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs, lbls, input_ids, attention_masks = batch['image'], batch['labels'], batch['text_input_ids'], batch['text_attention_mask']
            imgs, lbls, input_ids, attention_masks = imgs.to('cuda'), lbls.to('cuda'), input_ids.to('cuda'), attention_masks.to('cuda')
            preds = model(imgs, input_ids, attention_masks).squeeze()
            preds_binary = preds > .0
            total += lbls.shape[0]
            correct += (preds_binary == lbls).sum().item()
            total_loss += loss_fn(preds, lbls.to(torch.float))
            # print(pred_batch_binary == lbl_batch)
    accuracy = correct / total
    loss = (total_loss / (i + 1)).item()
    return accuracy, loss

    
df_path = '/home/aokulikov/proj_dpo_shared/data/datasets/coyo_discr_llava_1.csv'
model_save_path = '/home/aokulikov/proj_dpo_shared/notebooks/aokulikov/model_checkpoints/discriminator_effnetv2s_txt_1_v2/'
batch_size = 128
img_size = 768
lr = 0.001
num_workers = 4


def main():
    model_transforms = torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms()
    transform = transforms.Compose(
        [
            transforms.CenterCrop(img_size),
            model_transforms,
        ]
    )


    df = pd.read_csv(df_path).sample(frac=1)

    # df = df.sample(1000)

    last_train_idx = int(len(df) * 0.9)
    train_df = df.iloc[:last_train_idx, :]
    val_df = df.iloc[last_train_idx:, :]
    train_df.to_csv('/home/aokulikov/proj_dpo_shared/data/datasets/train_coyo_discr_llava_1.csv', index=False)
    val_df.to_csv('/home/aokulikov/proj_dpo_shared/data/datasets/val_coyo_discr_llava_1.csv', index=False)

    train_dataset = BinaryClassImageTxtDataset(train_df, transform=transform)
    val_dataset = BinaryClassImageTxtDataset(val_df, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_batch, num_workers=num_workers)

    # IMG MODEL
    img_model = torchvision.models.efficientnet_v2_s(weights='IMAGENET1K_V1', )
    module_list = [m for m in img_model.named_children()]
    backbone = module_list[0][-1]
    for param in backbone.parameters():
        param.requires_grad = False

    # TXT MODEL
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    for param in text_encoder.parameters():
        param.requires_grad = False


    model = Discriminator(img_backbone=backbone,
                          img_embed_dim=1280,
                          text_encoder=text_encoder,
                          text_embed_dim=512,
                          num_classes=1)
    model = model.to('cuda')


    loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


    show_train_loss_each = 200
    num_epochs = 15
    train_running_loss_list, val_acc_list, val_loss_list = list(), list(), list()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader, 0):
            train_loss = train_batch(batch, optimizer, loss, model)
            running_loss += train_loss
            if i % show_train_loss_each == 0:   
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / show_train_loss_each:.3f}')
                train_running_loss_list.append(running_loss)
                running_loss = 0.0
        val_accuracy, val_loss = evaluate(model, val_dataloader, loss)
        print('val acc: ', val_accuracy, ' val loss: ', val_loss)
        val_acc_list.append(val_accuracy)
        val_loss_list.append(val_loss)
        torch.save(model, model_save_path + 'full_model.pt')
        torch.save(model.state_dict(),  model_save_path + 'state_dict.pt')
        

if __name__ == "__main__":
    main()