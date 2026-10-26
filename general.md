# Phân tích Project R2GenCMN

# 1. Executive Summary

Đây là project PyTorch tạo báo cáo X-quang từ ảnh, hỗ trợ hai dataset: IU X-Ray và MIMIC-CXR. Entry point là `main.py`. Luồng chính khởi tạo tokenizer, train/validation/test DataLoader, `BaseCMNModel`, optimizer, scheduler và `Trainer`, sau đó gọi `trainer.train()`.

Model trích xuất feature ảnh bằng ResNet101, chiếu feature về 512 chiều, truy vấn cross-modal memory, rồi dùng Transformer encoder–decoder tự hồi quy để sinh token báo cáo. Mục tiêu training là masked negative log likelihood; evaluation dùng BLEU-1–4, METEOR và ROUGE-L.

Các vấn đề đáng chú ý tìm thấy trong source:

- Test set được đánh giá ở mỗi epoch; `Trainer._record_best` còn theo dõi kết quả test tốt nhất. Dùng test liên tục trong quá trình phát triển có thể làm sai lệch đánh giá cuối cùng.
- Beam search và chặn trigram có các lệnh `.cuda()` trực tiếp, không theo device mà `Trainer` chọn. Điều này có thể làm inference lỗi khi chạy CPU.
- Scheduler state không được lưu trong checkpoint và không được khôi phục khi resume.
- Các file annotation và ảnh dataset không có trong workspace hiện tại; nội dung split, kích thước thực tế và khả năng chạy end-to-end là **Unknown**.

Phân tích dựa trên source và cấu hình hiện có; không chạy training và không thay đổi file trước khi tạo báo cáo này.

# 2. Project Structure

```text
R2GenCMN/
├── main.py
├── run_iu_xray.sh
├── run_mimic_cxr.sh
├── README.md
├── LICENSE
├── data/
│   └── r2gencmn.md
├── models/
│   └── models.py
├── modules/
│   ├── __init__.py
│   ├── att_model.py
│   ├── base_cmn.py
│   ├── caption_model.py
│   ├── dataloaders.py
│   ├── datasets.py
│   ├── loss.py
│   ├── metrics.py
│   ├── optimizers.py
│   ├── tokenizers.py
│   ├── trainer.py
│   ├── utils.py
│   └── visual_extractor.py
└── .vscode/
    └── settings.json
```

`data/` hiện chỉ có tài liệu `r2gencmn.md`; các annotation và thư mục ảnh mà shell script tham chiếu không có trong danh sách file hiện tại. `.vscode/settings.json` không tham gia luồng runtime được truy vết.

# 3. Code Architecture

## Entry point và thứ tự gọi

Hai script chạy là `run_iu_xray.sh` và `run_mimic_cxr.sh`; mỗi script gọi `python main.py` với tham số tương ứng. Trong Python:

```text
main.py
└── main()
    ├── parse_agrs()
    ├── Tokenizer(args)
    │   └── create_vocabulary()
    ├── R2DataLoader(..., split="train")
    │   └── Dataset(...)
    ├── R2DataLoader(..., split="val")
    │   └── Dataset(...)
    ├── R2DataLoader(..., split="test")
    │   └── Dataset(...)
    ├── BaseCMNModel(args, tokenizer)
    │   ├── VisualExtractor(args)
    │   └── BaseCMN(args, tokenizer)
    ├── build_optimizer(...)
    ├── build_lr_scheduler(...)
    └── Trainer(...).train()
        └── _train_epoch(epoch)
            ├── train forward → loss → backward → optimizer.step()
            ├── validation generation → metrics
            ├── test generation → metrics
            └── lr_scheduler.step()
```

Các import ở đầu `main.py` xác định những module được nối vào luồng này: `models.models`, `dataloaders`, `loss`, `metrics`, `optimizers`, `tokenizers` và `trainer`.

## Module responsibilities

| File | Class/Function | Responsibility | Input | Output | Called by |
|---|---|---|---|---|---|
| `main.py` | `main`, `parse_agrs` | Đọc arguments, tạo toàn bộ pipeline | CLI arguments | Bắt đầu training | Shell scripts gọi `main.py` |
| `modules/tokenizers.py` | `Tokenizer` | Làm sạch report, tạo vocabulary từ train split, mã hóa/giải mã | Report text, annotation | Token IDs hoặc text | `main.py`, dataset, trainer |
| `modules/datasets.py` | `BaseDataset`, hai dataset con | Đọc annotation/ảnh, mã hóa report, tạo sample | Index | ID, ảnh, token IDs, mask, length | `R2DataLoader` |
| `modules/dataloaders.py` | `R2DataLoader`, `collate_fn` | Augmentation, batching và padding trong batch | Samples | Batch ảnh, IDs, masks | `main.py`, `Trainer` |
| `models/models.py` | `BaseCMNModel` | Chọn forward IU hoặc MIMIC; ghép feature ảnh | Ảnh, targets hoặc sampling mode | Log probabilities khi train; tokens và probs khi sample | `main.py`, `Trainer` |
| `modules/visual_extractor.py` | `VisualExtractor` | ResNet feature extraction và spatial flattening | Tensor ảnh | Patch features, pooled features | `BaseCMNModel` |
| `modules/base_cmn.py` | `BaseCMN`, `Transformer`, attention/memory classes | Visual encoder, textual decoder, memory querying và token logits | Visual features, token sequence, memory | Log probabilities cho vocabulary | `BaseCMNModel` |
| `modules/att_model.py` | `AttModel` | Chuẩn bị feature, giải mã và sampling/beam search | Feature, token hiện tại | Log probabilities hoặc chuỗi token | `BaseCMN` |
| `modules/caption_model.py` | `CaptionModel` | Các helper sinh chuỗi, beam search, chọn token | Log probabilities, decoder state | Chuỗi dự đoán | `AttModel` |
| `modules/loss.py` | `LanguageModelCriterion`, `compute_loss` | Masked token-level negative log likelihood | Log probabilities, targets, masks | Scalar loss | `Trainer._train_epoch` |
| `modules/metrics.py` | `compute_scores` | Caption metrics | Ground truth và report sinh ra | BLEU, METEOR, ROUGE-L | `Trainer._train_epoch` |
| `modules/optimizers.py` | `build_optimizer`, `build_lr_scheduler` | Tạo optimizer hai parameter groups và scheduler | Args, model | Optimizer, scheduler | `main.py` |
| `modules/trainer.py` | `Trainer`, `_train_epoch` | Training, validation, test, logging và checkpoint | Model, loaders, criterion, optimizer | Metrics/logs/checkpoints | `main.py` |
| `modules/utils.py` | `repeat_tensors`, `split_tensors`, penalties | Tensor utilities cho decoding | Tensor hoặc cấu hình penalty | Tensor biến đổi hoặc scoring function | `CaptionModel`, `AttModel` |

## Dataset và preprocessing

- Dataset được chọn bằng `args.dataset_name`: `iu_xray` dùng `IuxrayMultiImageDataset`, còn lựa chọn khác dùng `MimiccxrSingleImageDataset` (`modules/dataloaders.py`).
- IU X-Ray đọc hai đường dẫn ảnh cho mỗi example và stack thành một tensor. MIMIC đọc một ảnh. Trong nhánh MIMIC, giá trị `image_id` được thay bằng đường dẫn ảnh trong `__getitem__` (`modules/datasets.py`).
- Train transform: resize cạnh ngắn về 256, random crop 224×224, random horizontal flip, chuyển tensor và ImageNet normalization. Validation/test resize thẳng về 224×224 rồi tensor/normalize (`modules/dataloaders.py`).
- Text được làm sạch theo dataset, tách bằng khoảng trắng, từ có tần suất dưới threshold bị bỏ khỏi vocabulary; token lạ thành `<unk>`. Vocabulary lấy từ `ann['train']` (`modules/tokenizers.py`).
- Tokenizer thêm ID `0` ở đầu và cuối. Dataset cắt chuỗi ở `max_seq_length`; mask được tạo toàn số 1 cho các ID còn lại. Padding đến độ dài lớn nhất trong batch xảy ra trong `collate_fn`, nơi phần padding khởi tạo bằng 0 (`modules/datasets.py`, `modules/dataloaders.py`).

## Các thành phần model thực tế

| Thành phần | Thực hiện trong code |
|---|---|
| Feature extractor/backbone | `torchvision.models.<visual_extractor>` với `pretrained=True` mặc định; bỏ hai layer cuối, lấy feature map ResNet |
| Visual projection | `AttModel.att_embed`: Linear `d_vf → d_model`, ReLU, dropout |
| Cross-modal memory | `BaseCMN.memory_matrix` có shape `[cmm_size, cmm_dim]`; `MultiThreadMemory` chọn top-k response cho visual features và token embeddings |
| Encoder | Transformer encoder gồm `num_layers` layer; self-attention và feed-forward |
| Decoder | Transformer decoder gồm masked self-attention, cross-attention tới visual memory, feed-forward |
| Fusion | Memory response được cộng residual vào visual features và vào textual embeddings |
| Classifier | Không có classifier nhãn riêng. `BaseCMN.logit` là Linear `d_model → vocab_size+1` để dự đoán token tiếp theo |
| Loss | Masked negative log likelihood trên các token target |
| Optimizer | Mặc định Adam với learning rate riêng cho visual extractor và các phần còn lại |
| Evaluation | Beam search mặc định; BLEU-1–4, METEOR và ROUGE-L |

# 4. Module Dependency Map

```text
run_iu_xray.sh ─┐
                ├──> main.py
run_mimic_cxr.sh┘     ├── modules/tokenizers.py
                      ├── modules/dataloaders.py
                      │   └── modules/datasets.py
                      ├── models/models.py
                      │   ├── modules/visual_extractor.py
                      │   └── modules/base_cmn.py
                      │       └── modules/att_model.py
                      │           └── modules/caption_model.py
                      │               └── modules/utils.py
                      ├── modules/loss.py
                      ├── modules/metrics.py
                      ├── modules/optimizers.py
                      └── modules/trainer.py
```

`base_cmn.py` định nghĩa Transformer và memory attention trực tiếp trong cùng file. `att_model.py` cung cấp helper `pack_wrapper`, feature preparation và sampling wrapper. Một số lớp trong `metrics.py` (`compute_mlc`, `MetricWrapper`) có trong source nhưng không được truyền vào training path của `main.py`; `main.py` dùng `compute_scores`.

# 5. Architecture Diagram

```text
IU X-Ray: 2 ảnh                         MIMIC-CXR: 1 ảnh
[B, 2, 3, 224, 224]                    [B, 3, 224, 224]
         │                                        │
         └─────────────┐             ┌───────────┘
                       ↓             ↓
                 ResNet101 feature extractor
                   pretrained mặc định
                       │
       patch map [B, 2048, 7, 7] mỗi ảnh
                       │
        flatten spatial → [B, N, 2048]
                       │
          Linear 2048→512 + ReLU + dropout
                       │
       Visual memory querying (top-k=32)
                       │
       Transformer encoder, 3 layers
                       │
              Visual encoded memory
                       │
  Report prefix → Embedding + positional encoding
                       │
          Text memory querying (top-k=32)
                       │
    Transformer decoder: self-attn + cross-attn
                       │
       Linear 512→vocab_size+1 + log_softmax
                       │
             Next-token log probabilities
                ┌──────┴────────┐
                ↓               ↓
      Train: masked NLL     Sample: beam search
                                │
                      Decoded report text
                                │
                 BLEU / METEOR / ROUGE-L
```

Không có classifier bệnh lý hoặc fusion head riêng. Fusion xảy ra qua residual addition của memory responses tại visual features và textual embeddings trong `base_cmn.py`.

# 6. Data Flow

| Step | File | Function/Class | Input | Operation | Output |
|---|---|---|---|---|---|
| Annotation | `data/.../annotation.json` theo CLI | Được đọc bởi `BaseDataset` và `Tokenizer` | JSON với các split | `Tokenizer` đọc train để lập vocabulary; dataset đọc split được chọn | Examples và từ vựng |
| Text cleaning/tokenization | `modules/tokenizers.py` | `clean_report_*`, `__call__` | Report string | Chuẩn hóa dấu câu/chữ thường, tách token, tra vocabulary, thêm ID 0 đầu/cuối | List token IDs |
| Truncation và mask | `modules/datasets.py` | `BaseDataset.__init__` | Report string | Tokenize, cắt `max_seq_length`, mask toàn 1 | `ids`, `mask`, length |
| Image reading | `modules/datasets.py` | Dataset `__getitem__` | Image paths | Mở PIL, convert RGB | PIL image(s) |
| Image preprocessing | `modules/dataloaders.py` | `R2DataLoader.__init__` | PIL image | Augmentation train hoặc resize eval, tensor hóa, ImageNet normalization | `[3,224,224]` mỗi ảnh |
| IU image grouping | `modules/datasets.py` | `IuxrayMultiImageDataset.__getitem__` | Hai ảnh đã transform | `torch.stack((image_1,image_2),0)` | `[2,3,224,224]` |
| Batch construction | `modules/dataloaders.py` | `collate_fn` | Samples variable length | Stack ảnh, padding IDs/masks tới max length trong batch | IDs `[B,L]`, masks `[B,L]` |
| Model input | `modules/trainer.py` | `Trainer._train_epoch` | Batch tensors | Chuyển ảnh/IDs/masks sang `self.device` | GPU/CPU tensors |
| Visual features | `modules/visual_extractor.py` | `VisualExtractor.forward` | Ảnh | ResNet trunk, average pooling và flatten feature map | `att_feats [B,N,2048]`; `fc_feats [B,2048]` |
| IU feature combination | `models/models.py` | `forward_iu_xray` | Hai feature sets | Gọi extractor riêng cho mỗi ảnh, concat theo feature dimension ở `fc_feats`, theo region ở `att_feats` | `fc_feats [B,4096]`, `att_feats [B,98,2048]` |
| Feature projection/memory | `modules/base_cmn.py`, `modules/att_model.py` | `_prepare_feature_forward` | Attention features | Linear `2048→512`; query memory, cộng response; tạo visual mask | `[B,N,512]` |
| Encoder/decoder | `modules/base_cmn.py` | `BaseCMN._forward`, Transformer | Visual features và `reports_ids[:,:-1]` | Encoder visual; decoder nhận shifted text input và causal mask | Log probabilities `[B,L-1,V+1]` |
| Loss | `modules/loss.py` | `compute_loss` | Model output, shifted targets, shifted masks | Gather log prob của target, masking và chia tổng mask | Scalar loss |
| Evaluation decode | `modules/trainer.py`, `modules/att_model.py` | `Trainer._train_epoch`, `_sample_beam` | Ảnh | Sinh chuỗi bằng beam search; decode token IDs | Report strings |
| Metrics | `modules/metrics.py` | `compute_scores` | Dictionary refs/hypotheses | Tính COCO caption metrics | Metric scores |

Trong `collate_fn`, `seq_lengths_batch` được dùng để xác định độ dài padding tối đa nhưng không được trả về. DataLoader mặc định không bật `pin_memory`; `drop_last` không được truyền nên dùng mặc định PyTorch (`False`). Không có `collate_fn` bổ sung ngoài `R2DataLoader.collate_fn`.

# 7. Tensor Shape Flow

Ký hiệu: `B` là batch size, `L` là độ dài chuỗi lớn nhất trong batch, `V` là số token trong vocabulary, `D=512`, `N` là số vị trí ảnh. Các shape ảnh dựa trên resize/crop trong code và cấu trúc ResNet101 được chọn mặc định. Không có runtime tensor logging trong source để xác nhận trên môi trường cụ thể.

| Tensor/stage | Shape | dtype/device | Ý nghĩa |
|---|---|---|---|
| Raw PIL image | Kích thước nguồn dynamic | PIL, CPU | Ảnh gốc |
| Sau transform | `[3,224,224]` | Float tensor, CPU | RGB, ImageNet-normalized |
| Dataset IU sample | `[2,3,224,224]` | Float, CPU | Hai ảnh cho một example |
| Dataset MIMIC sample | `[3,224,224]` | Float, CPU | Một ảnh |
| Collated IU batch | `[B,2,3,224,224]` | Float, CPU trước Trainer | Batch hai view |
| Collated MIMIC batch | `[B,3,224,224]` | Float, CPU trước Trainer | Batch một view |
| Report IDs | `[B,L]` | `LongTensor`, CPU rồi device của Trainer | Token 0 đầu/cuối và padding 0 |
| Report masks | `[B,L]` | `FloatTensor`, CPU rồi device của Trainer | 1 cho token thật; 0 cho padding |
| ResNet map, mỗi view | `[B,2048,7,7]` | Thường float32; cùng device model | Channel feature và spatial grid |
| ResNet patch features | `[B,49,2048]` mỗi view | Float; cùng device model | 49 vị trí vùng ảnh |
| IU ghép patch features | `[B,98,2048]` | Float; cùng device model | 49 vùng × 2 ảnh |
| MIMIC patch features | `[B,49,2048]` | Float; cùng device model | 49 vùng |
| Pooled features | `[B,2048]` mỗi view | Float; cùng device model | Average pooled; IU ghép thành `[B,4096]` |
| `att_embed` output | `[B,N,512]` | Float; cùng device model | Feature chiếu vào dimension Transformer |
| Memory matrix | `[2048,512]` theo script/default | Parameter float32, cùng device model | 2048 memory slots, dimension 512 |
| Memory response | `[B,N,512]` | Float; cùng device model | Response sau multi-head top-k memory lookup |
| Visual encoder output | `[B,N,512]` | Float; cùng device model | Encoded visual context |
| Decoder input IDs | `[B,L-1]` | Long, cùng device model | `reports_ids[:, :-1]` trong training |
| Decoder embeddings | `[B,L-1,512]` | Float, cùng device model | Embedding cộng positional encoding và memory response |
| Self-attention score | `[B,H,T,T]` | Float, cùng device model | `H=8`; `T` là độ dài sequence liên quan |
| Cross-attention score | `[B,H,T,N]` | Float, cùng device model | Token-to-visual-region attention |
| Model logits/log probabilities | `[B,L-1,V+1]` trong training | Float, cùng device model | Token distribution; `log_softmax` theo vocabulary |
| Shifted target | `[B,L-1]` | Long, cùng device model | `reports_ids[:,1:]` |
| Shifted target mask | `[B,L-1]` | Float, cùng device model | `reports_masks[:,1:]` |
| Loss | Scalar | Float, cùng device model | Mean masked NLL |
| Sampled sequence | `[B,max_seq_length]` | Long, cùng device model | Beam search output, có thể còn padding |
| Decoded text | Một chuỗi/example | Python string | Giải mã dừng ở token ID 0 |
| Metrics | Scalar mỗi metric | Python numeric | BLEU-1–4, METEOR, ROUGE-L |

## Shape transformations và kiểm tra

- `VisualExtractor.forward`: `[B,2048,7,7] → reshape [B,2048,49] → permute [B,49,2048]`. Average pool tạo `[B,2048,1,1]`, rồi `squeeze().reshape(-1,2048)` (`modules/visual_extractor.py`).
- IU forward concat view features theo `dim=1`: pooled vectors thành `[B,4096]`; patch tokens thành `[B,98,2048]`. `fc_feats` được cắt thành `fc_feats[..., :1]` trong `_prepare_feature`, nhưng decoder Transformer chủ yếu dùng `att_feats`; pooled features không đi vào classifier.
- Visual projection là Linear `d_vf→d_model`; cấu hình mặc định/script tương ứng là `2048→512`.
- Multi-head attention reshape `[B,T,512] → [B,8,T,64]`, attention, rồi transpose/reshape lại `[B,T,512]`.
- Memory top-k chọn trên chiều slot; `topk=32` nhỏ hơn kích thước memory `2048` trong shell scripts.
- Training decoder dùng `seq = seq[:, :-1]`; loss target lấy `reports_ids[:,1:]`. Hai phía được shift đúng hướng teacher forcing. Padding bị loại qua target mask và target ID 0.
- `cmm_dim` phải tương thích với dimension mà `MultiThreadMemory` Linear mong đợi (`d_model`); scripts đặt cả hai bằng 512. Code không có projection riêng để chuyển đổi nếu hai tham số khác nhau.
- Các shape phụ thuộc runtime gồm `B`, `L` và device. `N=49`/`98` dựa trên kích thước output ResNet101 ở đầu vào 224×224. Nếu đổi backbone/đầu vào, shape cụ thể cần được xác minh runtime.

# 8. Training Pipeline

## Dataset và DataLoader

- Train, validation và test được tạo từ cùng `ann_path`, với split lần lượt `train`, `val`, `test` tại `main.py`.
- Split strategy được lưu trong annotation JSON; code không tự chia ngẫu nhiên. Tỷ lệ và quy tắc tạo split là **Unknown** từ source hiện có.
- Batch size mặc định và trong hai shell script là 16; `num_workers` mặc định 2.
- Train loader shuffle=True; validation/test shuffle=False.
- `pin_memory` không được bật; `drop_last` không được cấu hình; không có sampler custom.

## Model và trainable parameters

- `VisualExtractor` lấy ResNet với tham số `pretrained=True` mặc định.
- Không có chỗ nào đặt `requires_grad=False`; optimizer đưa toàn bộ visual extractor vào parameter group riêng, do đó backbone được train.
- Transformer, memory matrix, embedding, projection và logit layer được khởi tạo từ code. `make_model` áp dụng Xavier initialization cho parameter có `dim()>1`; memory matrix được khởi tạo normal với std `1/cmm_dim`.
- `logit` layer được tạo bên ngoài `make_model`, nên không nằm trong vòng lặp Xavier của `make_model`; PyTorch Linear dùng initialization mặc định của nó.

## Loss, optimizer và scheduler

- `LanguageModelCriterion.forward` lấy log probability tại target ID, nhân mask, cộng các token rồi chia cho tổng mask.
- Optimizer mặc định là Adam. `build_optimizer` chia tham số thành visual extractor với `lr_ve` và các tham số còn lại với `lr_ed`.
- Defaults: `lr_ve=5e-5`, `lr_ed=7e-4`, `weight_decay=5e-5`, betas `(0.9,0.98)`, epsilon `1e-9`, AMSGrad bật.
- Shell scripts override learning rates thành `1e-4` và `5e-4`.
- Scheduler mặc định `StepLR`, `step_size=50`, `gamma=0.1`; shell scripts dùng `step_size=10` cho IU và `3` cho MIMIC, gamma `0.8`.
- Scheduler được gọi một lần sau cả validation và test, tức mỗi epoch (`modules/trainer.py`).
- `NoamOpt` và plateau optimizer có trong `optimizers.py` nhưng `main.py` không gọi chúng. `noamopt_warmup` và `noamopt_factor` vì vậy không tác động vào pipeline mặc định.

## Training loop

```text
for epoch:
    model.train()
    for batch:
        chuyển ảnh, IDs, masks sang device
        output = model(images, reports_ids, mode="train")
        loss = criterion(output, reports_ids, reports_masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    sinh report trên validation → tính metrics
    sinh report trên test → tính metrics
    lr_scheduler.step()
    ghi log / lưu checkpoint theo điều kiện
```

`targets` được truyền vào model. Teacher forcing xảy ra trong `_prepare_feature_forward` khi decoder input là `seq[:, :-1]`; output được so với `reports_ids[:,1:]`.

## Checkpoint và resume

- Sau mỗi `save_period` epoch, `_save_checkpoint` ghi `current_checkpoint.pth`; nếu metric validation cải thiện thì ghi thêm `model_best.pth`.
- Mặc định monitor là `val_BLEU_4`, mode `max`; early stopping mặc định patience 50 epoch.
- Resume tải `epoch`, `state_dict`, `optimizer` và `monitor_best`. `lr_scheduler` state không được lưu hoặc tải. `best_recorder` cũng được khởi tạo lại, không khôi phục từ checkpoint.
- Việc tạo thư mục `save_dir` xảy ra khi khởi tạo Trainer; không thực thi trong lần phân tích này.

# 9. Evaluation Pipeline

```text
Validation hoặc test Dataset
        ↓
DataLoader, shuffle=False
        ↓
model.eval() + torch.no_grad()
        ↓
model(images, mode="sample")
        ↓
Beam search, beam_size mặc định 3
        ↓
Tokenizer.decode_batch
        ↓
Ghép hypothesis với reference theo index
        ↓
BLEU-1/2/3/4, METEOR, ROUGE-L
```

- `Trainer._train_epoch` gọi evaluation trên cả validation và test sau mỗi epoch.
- `BaseCMNModel.forward_*` chuyển sang `mode='sample'`, gọi decoder sampling.
- `sample_method` mặc định là `beam_search`, `beam_size=3`; `max_seq_length` giới hạn độ dài. `CaptionModel` kết thúc beam khi token EOS được phát ra hoặc đạt max length.
- `block_trigrams` mặc định bật (`1`); `decoding_constraint` mặc định tắt. `output_logsoftmax=1`.
- Tokenizer gán `bos_idx=eos_idx=pad_idx=0` theo defaults. Sampling bắt đầu bằng BOS=0; decode dừng khi gặp ID 0. Token ID 0 vì vậy vừa là marker đầu/cuối/padding theo cách triển khai.
- Không có threshold classification hay top-k classification. Top-k trong `MultiThreadMemory` là chọn memory slots, không phải metric/post-processing.
- `compute_scores` nhận dictionary reference và hypothesis; từng item được bọc thành một reference string. Tính BLEU-1, BLEU-2, BLEU-3, BLEU-4 qua `Bleu(4)`, METEOR và ROUGE-L.
- Các metric multilabel (`compute_mlc`, AUC/F1/recall/precision) có trong file nhưng không được gọi bởi `main.py`/`Trainer`.

# 10. Consistency Check

| Kiểm tra | Kết quả từ source |
|---|---|
| Code ↔ Architecture | Phù hợp với Transformer memory-driven và report generation. Không thấy classifier bệnh lý riêng; architecture cần thể hiện token-generation head. |
| Architecture ↔ Data flow | IU nối hai ảnh thành 98 visual tokens; MIMIC dùng 49. Cả hai đi qua cùng visual projection, memory và Transformer. |
| Data flow ↔ Tensor shapes | Shift decoder input/target tương thích; visual feature projection 2048→512 tương thích defaults/scripts. |
| Training ↔ Evaluation | Training tối ưu masked token NLL; chọn checkpoint bằng validation BLEU-4. Đây là hai mục tiêu khác nhau nhưng là lựa chọn được code thực hiện. |
| Test usage | Test được chạy mỗi epoch và best test được ghi nhận. Đây là nguy cơ làm mất tính độc lập của test metric nếu dùng để báo cáo/chọn quyết định. |
| Config ↔ code | `cmm_size`, `cmm_dim`, `topk`, `d_vf`, `d_model` được dùng trong model; optimizer/scheduler flags mặc định được sử dụng. Noam/plateau-specific settings hiện không được main gọi. |
| Documented behavior ↔ code | README nêu `opencv-python` trong requirements, nhưng source được truy vết không import `cv2`. README/data doc mô tả tải model R2Gen; code hiện tại dựng `BaseCMNModel`/`BaseCMN`. Không đủ bằng chứng để khẳng định đây là sai lệch về artifact được cung cấp, nhưng tên mô hình trong tài liệu không khớp rõ với tên class hiện tại. |
| CPU/device ↔ code | Trainer có CPU fallback, nhưng decoding chứa `.cuda()` trực tiếp trong đường beam/trigram. CPU inference có thể không tương thích. |
| Resume ↔ scheduler | Resume khôi phục optimizer nhưng không scheduler; sau resume lịch LR có thể khác quá trình chạy liên tục. |
| CLI bool ↔ config | `type=bool` được dùng cho `visual_extractor_pretrained`/`amsgrad`. Cách parse boolean từ chuỗi CLI có thể không theo kỳ vọng thông thường; shell hiện không override hai cờ này. |

# 11. Critical Findings

## Finding 1

**Severity:** HIGH  
**File:** `modules/trainer.py`  
**Class/Function:** `BaseTrainer._record_best`, `Trainer._train_epoch`  
**Problem:** Test set được chạy và theo dõi kết quả tốt nhất ở mọi epoch.  
**Evidence:** `_train_epoch` sinh report trên test set và thêm `test_*` metric vào log; `_record_best` cập nhật `best_recorder['test']`.  
**Impact:** Nếu test metric được dùng để lựa chọn mô hình, điều chỉnh tham số hoặc báo cáo kết quả tốt nhất, test set không còn là đánh giá độc lập.  
**Recommended investigation:** Xác minh quy trình nghiên cứu có dùng test metrics để chọn checkpoint/ra quyết định hay chỉ ghi log tham khảo.

## Finding 2

**Severity:** HIGH  
**File:** `modules/att_model.py`, `modules/caption_model.py`  
**Class/Function:** `AttModel._sample`, beam search helpers  
**Problem:** Một số tensor được đưa lên CUDA bằng `.cuda()` trực tiếp, thay vì theo device của model/input.  
**Evidence:** `AttModel._sample` tạo trigram mask bằng `torch.zeros(...).cuda()`; các beam-search helper cũng gọi `it.cuda()`/`.cuda()` cho token và constraint tensor.  
**Impact:** Trainer có fallback sang CPU, nhưng sampling có thể lỗi ở cấu hình CPU hoặc device khác.  
**Recommended investigation:** Xác minh inference trên CPU và multi-GPU/device mapping trước khi dùng các cấu hình đó.

## Finding 3

**Severity:** MEDIUM  
**File:** `modules/trainer.py`, `modules/optimizers.py`  
**Class/Function:** `_save_checkpoint`, `_resume_checkpoint`, `build_lr_scheduler`  
**Problem:** Checkpoint không lưu scheduler state; resume không khôi phục lịch LR.  
**Evidence:** Checkpoint chỉ có `epoch`, `state_dict`, `optimizer`, `monitor_best`; `_resume_checkpoint` chỉ load model và optimizer.  
**Impact:** StepLR tiếp tục từ trạng thái mới sau resume, nên learning rate theo epoch có thể không giống chạy liên tục.  
**Recommended investigation:** Nếu dự kiến resume giữa chừng, kiểm tra LR sau resume có phù hợp với epoch đã lưu không.

## Finding 4

**Severity:** MEDIUM  
**File:** `modules/base_cmn.py`  
**Class/Function:** `BaseCMN.__init__`, `MultiThreadMemory.forward`  
**Problem:** Memory matrix dimension cấu hình độc lập với `d_model`, nhưng memory module dùng Linear layers có input/output `d_model`; không có projection để hòa giải hai dimension.  
**Evidence:** Memory matrix được tạo từ `[cmm_size,cmm_dim]`; scripts đặt `cmm_dim=512` và `d_model` mặc định 512.  
**Impact:** Cấu hình `cmm_dim != d_model` có thể gây lỗi dimension tại memory Linear/matmul. Cấu hình hiện tại trong scripts tương thích.  
**Recommended investigation:** Xác nhận mọi config tương lai giữ `cmm_dim == d_model`, hoặc truy vết hỗ trợ dimension khác trước khi đổi config.

# 12. Technical Debt

- `main.py` gọi `compute_scores`; các hàm MLC trong `metrics.py` không thuộc pipeline được dùng hiện tại.
- `NoamOpt`, `ReduceLROnPlateau`, `build_noamopt_optimizer`, `build_plateau_optimizer` tồn tại nhưng không được gọi từ entry point hiện tại.
- Tham số `fc_feats` được tạo và truyền qua nhiều hàm, nhưng decoder Transformer chính hoạt động trên `att_feats`; pooled features không tham gia trực tiếp vào logits.
- Dataset thay đổi dictionary examples trong `self.ann` bằng cách gắn `ids` và `mask`. Cùng annotation được đọc riêng cho tokenizer và từng dataset instance.
- `collate_fn` dùng NumPy để cấp phát target/mask rồi chuyển sang tensor; độ dài chuỗi là dynamic theo batch.
- `type=bool` cho CLI flags có semantics dễ gây nhầm lẫn khi truyền chuỗi command-line.
- `torchvision` dùng API `pretrained=`; README pin phiên bản cũ, nên tương thích với môi trường hiện tại là **Unknown**.
- README liệt kê OpenCV nhưng source được khảo sát không dùng nó. Các dependency runtime đầy đủ không thể xác nhận chỉ từ file đang có.
- Các shell scripts không khai báo `num_workers`, pretrained flag, monitor metric hay device; chúng dựa vào defaults trong `main.py`.

# 13. Questions / Unknowns

- Annotation JSON và ảnh không có trong workspace được liệt kê, nên số lượng mẫu, split, trường hợp lỗi ảnh, phân bố report length và nội dung nhãn là **Unknown**.
- Không thấy file dependency manifest như `requirements.txt` hoặc `environment.yml`; dependency set đầy đủ và phiên bản đang cài là **Unknown**.
- Không thể xác nhận checkpoint hiện có hoặc kết quả thực nghiệm vì không có checkpoint/log trong file list đã khảo sát.
- Không thể xác nhận shapes runtime cho backbone khác `resnet101`, batch cụ thể, hoặc cấu hình đổi `d_vf`, `d_model`, `cmm_dim`.
- `visual_extractor_pretrained=True` được yêu cầu trong code, nhưng việc tải pretrained weights thành công trong môi trường hiện tại là **Unknown**.
- Nội dung thật của split và liệu test có bị dùng để chọn quyết định nghiên cứu hay không là **Unknown**.
- Khả năng tương thích giữa phiên bản PyTorch/torchvision hiện tại và source cũ là **Unknown**.

# 14. Complete Project Map

```text
                         PROJECT
                            │
          ┌─────────────────┴──────────────────┐
          ↓                                    ↓
         DATA                                 MODEL
          │                                    │
  annotation JSON                    VisualExtractor
          │                                    │
  Tokenizer / vocabulary              ResNet101 trunk
          │                                    │
    Dataset class                   Spatial feature tokens
          │                                    │
  Image transforms                   Linear projection 2048→512
          │                                    │
       DataLoader                    Visual memory response
          │                                    │
          └──────────────┬─────────────────────┘
                         ↓
               BaseCMN Transformer
        encoder + memory + autoregressive decoder
                         │
              token log probabilities
                 ┌───────┴─────────┐
                 ↓                 ↓
          Training loss       Beam generation
          masked NLL          decoded report
                 │                 │
             backward          Metrics
                 │         BLEU/METEOR/ROUGE
             optimizer              │
                 │                  │
             scheduler ─────── validation/test
                 │
             checkpoint
```

- **Data:** `Tokenizer` tạo vocabulary từ train reports. Dataset đọc report/ảnh theo split annotation; transform ảnh phụ thuộc train hay validation/test.
- **Model:** ResNet tạo spatial feature tokens. `att_embed` chiếu về 512 chiều. Memory querying bổ sung response cho visual và textual features. Transformer decoder dự đoán token tiếp theo.
- **Training:** Teacher forcing so sánh phân phối token với report bị shift một vị trí; masked NLL loại padding. Adam tối ưu backbone và phần còn lại với learning rate riêng.
- **Checkpoint:** Mỗi `save_period` lưu current checkpoint; validation metric được cấu hình chọn best.
- **Evaluation:** Sampling sinh report, tokenizer giải mã, caption metrics so sánh hypothesis với reference. Test hiện chạy mỗi epoch.

# 15. Recommended Next Step

Trước khi cải tiến, bước hữu ích nhất là xác nhận bộ annotation/ảnh và cấu hình môi trường thực tế mà project dự kiến dùng. Sau đó có thể đối chiếu các unknown về split, dependency và tính tương thích CPU/device với môi trường chạy, rồi mới chọn phạm vi thay đổi.

**PROJECT ANALYSIS COMPLETE — NO FILES MODIFIED.**
