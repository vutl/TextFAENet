# LFAENet-TGFS: FAENet with Text-Guided Frequency Selection in the Decoder

## 0. Mục tiêu của tài liệu

Tài liệu này mô tả **chi tiết đầy đủ** một kiến trúc dựa trên **FAENet** nhưng được mở rộng cho **language-guided medical image segmentation** bằng cách đưa text vào **decoder** theo **Option A: Text-Guided Frequency Selection (TGFS)**.

Mục tiêu của thiết kế này là:
- giữ lại phần mạnh của **FAENet**: frequency-aware visual encoder-decoder với các khối **FreqA**,
- tránh kiểu fusion text quá thô như `concat(image_feature, text_embedding)` rồi đưa thẳng vào conv,
- làm cho **text trực tiếp điều khiển 4 frequency sub-bands** (`LL/LH/HL/HH`) ở phía decoder.

---

## 1. Đánh giá trước: ý tưởng này có đủ để làm paper không?

### 1.1 Kết luận ngắn
- **Không đủ mạnh** nếu contribution chỉ là:  
  **“lấy FAENet rồi thêm text ở decoder”**.
- **Có thể đủ paper** nếu contribution là:  
  **“text-guided frequency reasoning / text-guided sub-band selection trong decoder-side FreqA”**.

### 1.2 Vì sao bản đơn giản chưa đủ
Có hai dòng paper đã tồn tại:
1. **FAENet** đã chứng minh rằng việc chèn **FreqA** vào encoder-decoder để khai thác DWT, `LL/LH/HL/HH`, `ICCA`, `CCCA`, `SA`, `iDWT` giúp tăng chất lượng segmentation bằng cách jointly model spectral và spatial contexts.
2. **LMIS / text-guided medical segmentation** đã có rất nhiều paper cho thấy **decoder-side fusion** là hợp lý và mạnh. Đặc biệt, **FMISeg** là một **late-fusion model**, trong đó ngôn ngữ tương tác với frequency-domain visual features **ở decoder**.

=> Nếu contribution chỉ là:
> “FAENet + text in decoder”

thì reviewer rất dễ nói:
> đây chỉ là ghép một backbone frequency segmentation với một late-fusion LMIS recipe có sẵn.

### 1.3 Điều làm ý tưởng đủ mạnh hơn
Contribution cần được đổi thành:

> **Text-guided frequency selection in decoder-side frequency reasoning**

tức là:
- text không fuse “sau frequency reasoning”,
- text không chỉ đóng vai trò channel gate chung chung trên spatial feature,
- text **điều khiển trực tiếp** việc nhấn mạnh / làm yếu các sub-band `LL/LH/HL/HH`.

### 1.4 Paper story hợp lý nhất
Paper nên được kể như sau:

- **FAENet encoder** học visual spectral-spatial representation.
- **Decoder** là nơi thực hiện semantic selection và pixel-level refinement.
- **Text** cung cấp semantic prior để chọn **frequency components phù hợp** cho lesion hiện tại:
  - `LL` cho structure/anatomical layout,
  - `LH/HL/HH` cho boundary, direction-specific details, fine texture.

=> Đây là câu chuyện tốt hơn nhiều so với “đưa text vào decoder”.

---

## 2. Tên model đề xuất

Tên làm việc:

**LFAENet-TGFS**  
(Language-guided Frequency Attention-Enhanced Network with Text-Guided Frequency Selection)

Tên khác có thể dùng:
- **TFAENet**
- **TGFS-Net**
- **LaFreqSeg**

Trong tài liệu này sẽ dùng tên **LFAENet-TGFS**.

---

## 3. Nguyên lý thiết kế cốt lõi

### 3.1 Điều không nên làm
Không nên làm:
- cộng trực tiếp text embedding với visual feature map,
- concat text embedding “raw” với feature map spatial rồi ném vào conv.

Ví dụ không tốt:
```python
F_out = Conv(cat(F_visual, text_embedding))
```

Vấn đề:
- `F_visual` có dạng spatial: `B x C x H x W`
- `text_embedding` thường có dạng token-level hoặc pooled vector: `B x L x Ct` hoặc `B x Ct`
- hai loại representation này **khác bản chất**
- conv không phải nơi phù hợp để “hiểu” text token theo kiểu direct fusion

### 3.2 Điều nên làm
Text nên đóng vai trò:
- **gating**
- **attention controller**
- **frequency selector**

Nói cách khác:
> text không trở thành feature map chính  
> mà text **điều khiển cách visual-frequency features được xử lý**

### 3.3 Chọn Option A
Trong thiết kế này, ta dùng:

> **Option A = Text-Guided Frequency Selection (TGFS)**

Cụ thể:
- decoder feature được DWT tách thành `LL/LH/HL/HH`
- text sinh ra 4 nhóm gate
- mỗi nhóm gate điều khiển một sub-band tương ứng
- rồi mới cho sub-bands đi qua interaction và reconstruction

---

## 4. Ký hiệu và input/output

## 4.1 Input ảnh
Giả sử bài toán segmentation 2D:

\[
I \in \mathbb{R}^{B \times C_{in} \times H \times W}
\]

Trong đó:
- `B`: batch size
- `Cin`: số channel đầu vào
- `H, W`: kích thước ảnh

Ví dụ:
- X-ray / CT slice grayscale: `Cin = 1`
- nếu convert qua pseudo-RGB: `Cin = 3`
- `H = W = 224` hoặc `256`

## 4.2 Input text
Text prompt/report được tokenize thành:
\[
T = \{t_1, t_2, \dots, t_L\}
\]

với `L` là số token.

## 4.3 Output
Mask segmentation:
\[
\hat{M} \in \mathbb{R}^{B \times K \times H \times W}
\]

với:
- `K = 1` cho binary lesion segmentation
- `K > 1` cho multi-class segmentation

---

## 5. Text encoder

## 5.1 Chọn text encoder
Có thể dùng:
- **CXR-BERT** nếu task gần report X-ray/CT chest
- **BiomedCLIP text tower** nếu muốn gần hệ VLM đang dùng
- encoder có thể:
  - frozen hoàn toàn,
  - hoặc fine-tune nhẹ phần cuối

## 5.2 Output của text encoder
Ta lấy:
- **token-level embedding**
\[
E_t \in \mathbb{R}^{B \times L \times C_t}
\]
- **pooled text embedding**
\[
e_t \in \mathbb{R}^{B \times C_t}
\]

Trong **Option A**, thành phần chính được dùng là:
\[
e_t
\]

vì nó đủ để sinh ra global semantic control cho các sub-band.

---

## 6. Tổng quan toàn bộ model

Pipeline tổng quát:

```text
Image I
  ↓
Stem
  ↓
Encoder Stage 1 + FreqA → S1
  ↓ down
Encoder Stage 2 + FreqA → S2
  ↓ down
Encoder Stage 3 + FreqA → S3
  ↓ down
Encoder Stage 4 + FreqA → S4
  ↓ down
Bottleneck B
  ↓
Decoder Stage 4: Up + Skip(S4) + TGFSBlock
  ↓
Decoder Stage 3: Up + Skip(S3) + TGFSBlock
  ↓
Decoder Stage 2: Up + Skip(S2) + TGFSBlock
  ↓
Decoder Stage 1: Up + Skip(S1) + TGFSBlock
  ↓
Segmentation Head
  ↓
Mask
```

Text path:

```text
Text T
  ↓
Text Encoder
  ↓
Pooled text vector e_t
  ↓
broadcast to all decoder stages
```

---

## 7. Shape chi tiết toàn model

Giả sử input:
\[
I \in \mathbb{R}^{B \times 1 \times 224 \times 224}
\]

## 7.1 Stem
- `Conv 3x3, stride 1`
- output:
\[
X_0 \in \mathbb{R}^{B \times 64 \times 224 \times 224}
\]

---

## 7.2 Encoder

### Stage E1
- `ConvBlock(64)`
- `FreqA(64)`
- output skip:
\[
S_1 \in \mathbb{R}^{B \times 64 \times 224 \times 224}
\]

Downsample:
\[
X_1 \in \mathbb{R}^{B \times 128 \times 112 \times 112}
\]

### Stage E2
- `ConvBlock(128)`
- `FreqA(128)`
- skip:
\[
S_2 \in \mathbb{R}^{B \times 128 \times 112 \times 112}
\]

Downsample:
\[
X_2 \in \mathbb{R}^{B \times 256 \times 56 \times 56}
\]

### Stage E3
- `ConvBlock(256)`
- `FreqA(256)`
- skip:
\[
S_3 \in \mathbb{R}^{B \times 256 \times 56 \times 56}
\]

Downsample:
\[
X_3 \in \mathbb{R}^{B \times 512 \times 28 \times 28}
\]

### Stage E4
- `ConvBlock(512)`
- `FreqA(512)`
- skip:
\[
S_4 \in \mathbb{R}^{B \times 512 \times 28 \times 28}
\]

Downsample:
\[
X_4 \in \mathbb{R}^{B \times 768 \times 14 \times 14}
\]

### Bottleneck
- `ConvBlock(768)`
- optional `FreqA(768)`
- output:
\[
B \in \mathbb{R}^{B \times 768 \times 14 \times 14}
\]

---

## 7.3 Decoder

### Decoder Stage D4
Upsample bottleneck:
\[
U_4 \in \mathbb{R}^{B \times 768 \times 28 \times 28}
\]

Reduce channels:
\[
\tilde{U}_4 \in \mathbb{R}^{B \times 512 \times 28 \times 28}
\]

Concat với skip `S4`:
\[
\text{Concat}(\tilde{U}_4, S_4) \in \mathbb{R}^{B \times 1024 \times 28 \times 28}
\]

Fuse conv:
\[
G_4 \in \mathbb{R}^{B \times 512 \times 28 \times 28}
\]

TGFSBlock:
\[
D_4 \in \mathbb{R}^{B \times 512 \times 28 \times 28}
\]

### Decoder Stage D3
\[
D_3 \in \mathbb{R}^{B \times 256 \times 56 \times 56}
\]

### Decoder Stage D2
\[
D_2 \in \mathbb{R}^{B \times 128 \times 112 \times 112}
\]

### Decoder Stage D1
\[
D_1 \in \mathbb{R}^{B \times 64 \times 224 \times 224}
\]

### Head
\[
\hat{M} \in \mathbb{R}^{B \times K \times 224 \times 224}
\]

---

## 8. ConvBlock

### 8.1 Vai trò
ConvBlock dùng để:
- học local spatial pattern,
- refine feature trước và sau frequency block,
- giữ backbone theo phong cách CNN/U-Net.

### 8.2 Cấu trúc đề xuất
Một ConvBlock có thể là:
- `Conv 3x3`
- `BN`
- `GELU/ReLU`
- `Conv 3x3`
- `BN`
- `GELU/ReLU`
- residual shortcut (nếu cần)

### 8.3 Shape
Input/output cùng shape:
\[
B \times C \times H \times W
\]

---

## 9. FreqA trong encoder (giữ gần FAENet)

Trong encoder, ta giữ gần như nguyên khối **FreqA** của FAENet.

Cho input:
\[
X \in \mathbb{R}^{B \times C \times H \times W}
\]

## 9.1 Bước 1: DWT
\[
\{X_{LL}, X_{LH}, X_{HL}, X_{HH}\} = \text{DWT}(X)
\]

Mỗi sub-band:
\[
X_k \in \mathbb{R}^{B \times C \times H/2 \times W/2}
\]

Ý nghĩa:
- `LL`: low-frequency / coarse structure
- `LH`: horizontal component
- `HL`: vertical component
- `HH`: diagonal high-frequency component

## 9.2 Bước 2: ICCA
Áp attention nội bộ từng component.

Cho mỗi component \(X_k\):
- GAP theo spatial:
\[
z_k(c) = \frac{1}{HW/4}\sum_{i,j} X_k(i,j,c)
\]

- MLP:
\[
a_k = \sigma(W_2 \delta(W_1 z_k + b_1) + b_2)
\]

- Reweight:
\[
X_k^{ICCA}(i,j,c) = a_k(c)\cdot X_k(i,j,c)
\]

## 9.3 Bước 3: CCCA
Dùng các component sau ICCA để mô hình hóa cross-frequency interaction.

Ký hiệu:
\[
\{X_{LL}^{CCCA}, X_{LH}^{CCCA}, X_{HL}^{CCCA}, X_{HH}^{CCCA}\}
= \text{CCCA}(X_{LL}^{ICCA}, X_{LH}^{ICCA}, X_{HL}^{ICCA}, X_{HH}^{ICCA})
\]

## 9.4 Bước 4: Concat
\[
X_{agg} = \text{Concat}(X_{LL}^{CCCA}, X_{LH}^{CCCA}, X_{HL}^{CCCA}, X_{HH}^{CCCA})
\]

Shape:
\[
X_{agg} \in \mathbb{R}^{B \times 4C \times H/2 \times W/2}
\]

## 9.5 Bước 5: Self-Attention / Mixer
\[
X_{sa} = \text{SA}(X_{agg})
\]

## 9.6 Bước 6: iDWT
\[
Y = \text{iDWT}(X_{sa})
\]

Output:
\[
Y \in \mathbb{R}^{B \times C \times H \times W}
\]

## 9.7 Bước 7: Residual
\[
\text{FreqA}(X) = X + Y
\]

---

## 10. Khối mới: TGFS Decoder Block

Đây là contribution chính.

### 10.1 Input
- Decoder feature:
\[
F_d \in \mathbb{R}^{B \times C \times H \times W}
\]
- Pooled text vector:
\[
e_t \in \mathbb{R}^{B \times C_t}
\]

### 10.2 Output
\[
F_{out} \in \mathbb{R}^{B \times C \times H \times W}
\]

---

## 11. Pipeline chi tiết của TGFSBlock

## 11.1 Step A — Local spatial refinement
Trước khi vào frequency reasoning, refine feature sau skip fusion bằng conv.

\[
F_0 = \text{LocalConv}(F_d)
\]

`LocalConv` gồm:
- `Conv 3x3`
- `BN`
- `GELU`
- `Conv 3x3`
- `BN`
- `GELU`

Shape:
\[
F_0 \in \mathbb{R}^{B \times C \times H \times W}
\]

### Ý nghĩa
- giảm nhiễu sau concat skip
- đồng nhất feature trước khi phân rã bằng wavelet
- để frequency decomposition diễn ra trên feature “sạch” hơn

---

## 11.2 Step B — DWT decomposition
\[
\{F_{LL}, F_{LH}, F_{HL}, F_{HH}\} = \text{DWT}(F_0)
\]

Mỗi nhánh:
\[
F_k \in \mathbb{R}^{B \times C \times H/2 \times W/2}
\]

### Ý nghĩa
- `LL`: coarse anatomical structure
- `LH`: edge ưu tiên theo một hướng
- `HL`: edge theo hướng còn lại
- `HH`: chi tiết chéo / fine detail / cũng dễ nhiễu

---

## 11.3 Step C — ICCA cho từng sub-band
\[
\tilde{F}_k = \text{ICCA}(F_k),\quad k \in \{LL,LH,HL,HH\}
\]

### Ý nghĩa
Mỗi component được refine riêng trước khi text can thiệp.

Đây là bước:
- giữ đúng tinh thần FAENet,
- cho phép text tác động lên sub-band đã được nội bộ tinh lọc trước.

---

## 11.4 Step D — Text-Guided Frequency Selection (Option A)

Đây là bước mới quan trọng nhất.

### 11.4.1 Sinh gate từ text
Chiếu pooled text embedding sang không gian `4C`:

\[
g = W_g^{(2)} \,\phi(W_g^{(1)} e_t + b_g^{(1)}) + b_g^{(2)}
\]

Trong đó:
- `phi`: GELU/ReLU
- output:
\[
g \in \mathbb{R}^{B \times 4C}
\]

Tách:
\[
[g_{LL}, g_{LH}, g_{HL}, g_{HH}] = \text{Split}(g)
\]

với:
\[
g_k \in \mathbb{R}^{B \times C}
\]

Sau đó sigmoid:
\[
\alpha_k = \sigma(g_k), \quad k \in \{LL,LH,HL,HH\}
\]

### 11.4.2 Reshape để broadcast
\[
\alpha_k \rightarrow \mathbb{R}^{B \times C \times 1 \times 1}
\]

### 11.4.3 Apply gate
\[
\hat{F}_k = \tilde{F}_k \odot \alpha_k
\]

Tức là:
\[
\hat{F}_{LL} = \tilde{F}_{LL}\odot \alpha_{LL}
\]
\[
\hat{F}_{LH} = \tilde{F}_{LH}\odot \alpha_{LH}
\]
\[
\hat{F}_{HL} = \tilde{F}_{HL}\odot \alpha_{HL}
\]
\[
\hat{F}_{HH} = \tilde{F}_{HH}\odot \alpha_{HH}
\]

### Ý nghĩa của TGFS
Text quyết định:
- có nên nhấn mạnh structural context không (`LL`)
- có nên tăng trọng cho directional edges không (`LH`, `HL`)
- có nên giảm `HH` nếu nó nhiều noise không

Ví dụ:
- text kiểu “large diffuse opacity”  
  → `LL` mạnh hơn
- text kiểu “small irregular lesion boundary”  
  → `LH/HL/HH` mạnh hơn

Đây chính là:
> **text điều khiển lựa chọn frequency component**

---

## 11.5 Step E — Cross-Component Channel Attention sau TGFS
Sau khi text đã điều khiển từng sub-band, mới cho các sub-band tương tác với nhau:

\[
\{F_{LL}^{*}, F_{LH}^{*}, F_{HL}^{*}, F_{HH}^{*}\}
=
\text{CCCA}(\hat{F}_{LL}, \hat{F}_{LH}, \hat{F}_{HL}, \hat{F}_{HH})
\]

### Ý nghĩa
- text không chỉ gate từng band độc lập,
- text còn **gián tiếp thay đổi cross-frequency interaction**, vì input vào CCCA đã khác.

Nói cách khác:
- FAENet gốc: `ICCA -> CCCA`
- ta sửa thành: `ICCA -> TGFS -> CCCA`

Đây là khác biệt quan trọng.

---

## 11.6 Step F — Aggregate sub-bands
\[
F_{agg} = \text{Concat}(F_{LL}^{*}, F_{LH}^{*}, F_{HL}^{*}, F_{HH}^{*})
\]

Shape:
\[
F_{agg} \in \mathbb{R}^{B \times 4C \times H/2 \times W/2}
\]

### Ý nghĩa
- gom bốn sub-band đã được text-conditioned lại thành một aggregated frequency representation
- chuẩn bị cho mixing và reconstruction

---

## 11.7 Step G — Mixer / Self-Attention
Có hai lựa chọn:

### Lựa chọn nhẹ
- depthwise separable conv
- hoặc 2 conv `3x3`

\[
F_{mix} = \text{Mixer}(F_{agg})
\]

### Lựa chọn nặng hơn
- self-attention trên aggregated feature

\[
F_{mix} = \text{SA}(F_{agg})
\]

### Khuyến nghị
- stage sâu (`D4`, `D3`): có thể dùng SA
- stage nông (`D2`, `D1`): nên dùng mixer nhẹ để giảm compute

---

## 11.8 Step H — iDWT reconstruction
\[
F_{rec} = \text{iDWT}(F_{mix})
\]

Shape:
\[
F_{rec} \in \mathbb{R}^{B \times C \times H \times W}
\]

### Ý nghĩa
- quay từ frequency domain về spatial domain
- tạo refined feature map cùng shape với decoder feature ban đầu

---

## 11.9 Step I — Residual fusion
\[
F_{out} = F_0 + F_{rec}
\]

Có thể thêm conv cuối:
\[
F_{out} = \text{OutConv}(F_0 + F_{rec})
\]

### Ý nghĩa
- giữ thông tin spatial cơ bản của decoder feature
- frequency refinement chỉ đóng vai trò corrective enhancement
- giúp train ổn định hơn

---

## 12. Công thức tổng quát của TGFSBlock

Toàn bộ block có thể viết gọn:

\[
F_0 = \text{LocalConv}(F_d)
\]

\[
\{F_{LL},F_{LH},F_{HL},F_{HH}\}=\text{DWT}(F_0)
\]

\[
\tilde{F}_k = \text{ICCA}(F_k)
\]

\[
[g_{LL},g_{LH},g_{HL},g_{HH}] = \text{Split}(\text{MLP}(e_t))
\]

\[
\alpha_k = \sigma(g_k)
\]

\[
\hat{F}_k = \tilde{F}_k \odot \alpha_k
\]

\[
\{F_{LL}^{*},F_{LH}^{*},F_{HL}^{*},F_{HH}^{*}\}
=
\text{CCCA}(\hat{F}_{LL},\hat{F}_{LH},\hat{F}_{HL},\hat{F}_{HH})
\]

\[
F_{agg} = \text{Concat}(F_{LL}^{*},F_{LH}^{*},F_{HL}^{*},F_{HH}^{*})
\]

\[
F_{mix} = \text{Mixer}(F_{agg})
\]

\[
F_{rec} = \text{iDWT}(F_{mix})
\]

\[
F_{out} = F_0 + F_{rec}
\]

---

## 13. Decoder toàn phần với TGFSBlock

## 13.1 D4
\[
U_4 = \text{Up}(B)
\]
\[
G_4 = \text{FuseConv}(\text{Concat}(U_4,S_4))
\]
\[
D_4 = \text{TGFSBlock}(G_4,e_t)
\]

## 13.2 D3
\[
U_3 = \text{Up}(D_4)
\]
\[
G_3 = \text{FuseConv}(\text{Concat}(U_3,S_3))
\]
\[
D_3 = \text{TGFSBlock}(G_3,e_t)
\]

## 13.3 D2
\[
D_2 = \text{TGFSBlock}(\text{FuseConv}(\text{Concat}(\text{Up}(D_3),S_2)),e_t)
\]

## 13.4 D1
\[
D_1 = \text{TGFSBlock}(\text{FuseConv}(\text{Concat}(\text{Up}(D_2),S_1)),e_t)
\]

## 13.5 Output head
\[
\hat{M} = \text{Conv}_{1\times1}(D_1)
\]

---

## 14. Tại sao text chỉ đưa vào decoder?

### 14.1 Encoder nên tập trung vào visual representation
Encoder cần học:
- edge
- texture
- organ shape
- lesion morphology
- spectral-spatial features

Nếu đưa text quá sớm:
- encoder dễ bị semantic bias sớm
- feature kém generic hơn
- optimization khó hơn

### 14.2 Decoder là nơi semantic selection diễn ra
Decoder chịu trách nhiệm:
- combine skip + high-level semantic
- refine boundary
- quyết định pixel-wise class

Text phù hợp ở đây vì:
- nó giúp “chỉ đúng cái cần segment”
- nó giúp suppress background/irrelevant regions
- nó giúp chọn frequency mode phù hợp ở từng scale

=> Với task này, **decoder-only text fusion** là hợp lý nhất.

---

## 15. Vì sao dùng 4 sub-bands thay vì chỉ HF/LF?

### 15.1 Lợi ích của 4 sub-bands
Nếu chỉ dùng:
- `LF = LL`
- `HF = LH + HL + HH`

thì bạn mất:
- directional information
- khả năng text ưu tiên edge theo hướng cụ thể
- sự phân biệt giữa fine detail hữu ích và noise-like diagonal component

### 15.2 Khi nào 4 sub-bands hợp lý
Trong lesion segmentation:
- boundary có thể anisotropic
- shape irregular
- directional edge cues có giá trị

Do đó dùng:
- `LL`
- `LH`
- `HL`
- `HH`

sẽ phù hợp hơn cho **text-guided frequency selection**

### 15.3 Trade-off
Nhược điểm:
- nặng hơn HF/LF
- nhiều tham số hơn
- cần ablation để chứng minh thực sự có ích

---

## 16. Loss function

## 16.1 Binary segmentation
\[
\mathcal{L}_{seg} = \lambda_{dice}\mathcal{L}_{dice} + \lambda_{bce}\mathcal{L}_{bce}
\]

## 16.2 Multi-class segmentation
\[
\mathcal{L}_{seg} = \lambda_{dice}\mathcal{L}_{dice} + \lambda_{ce}\mathcal{L}_{ce}
\]

## 16.3 Optional gate regularization
Nếu muốn tránh gate luôn bật mạnh mọi band:
\[
\mathcal{L}_{gate} = \frac{1}{4}\sum_{k\in\{LL,LH,HL,HH\}} \|\alpha_k\|_1
\]

Total:
\[
\mathcal{L} = \mathcal{L}_{seg} + \lambda_g \mathcal{L}_{gate}
\]

Ban đầu có thể chưa cần dùng regularization này.

---

## 17. Pseudocode

```python
class TGFSBlock(nn.Module):
    def __init__(self, C, Ct):
        super().__init__()
        self.local_conv = LocalConv(C)
        self.icca_ll = ICCA(C)
        self.icca_lh = ICCA(C)
        self.icca_hl = ICCA(C)
        self.icca_hh = ICCA(C)

        self.text_mlp = nn.Sequential(
            nn.Linear(Ct, Ct),
            nn.GELU(),
            nn.Linear(Ct, 4 * C)
        )

        self.ccca = CCCA(C)
        self.mixer = Mixer(4 * C)
        self.out_conv = nn.Conv2d(C, C, kernel_size=3, padding=1)

    def forward(self, F_d, e_t):
        # F_d: [B, C, H, W]
        # e_t: [B, Ct]

        F0 = self.local_conv(F_d)  # [B, C, H, W]

        F_LL, F_LH, F_HL, F_HH = DWT(F0)  # each [B, C, H/2, W/2]

        F_LL = self.icca_ll(F_LL)
        F_LH = self.icca_lh(F_LH)
        F_HL = self.icca_hl(F_HL)
        F_HH = self.icca_hh(F_HH)

        gates = self.text_mlp(e_t)  # [B, 4C]
        g_LL, g_LH, g_HL, g_HH = torch.chunk(gates, 4, dim=1)

        a_LL = torch.sigmoid(g_LL).unsqueeze(-1).unsqueeze(-1)
        a_LH = torch.sigmoid(g_LH).unsqueeze(-1).unsqueeze(-1)
        a_HL = torch.sigmoid(g_HL).unsqueeze(-1).unsqueeze(-1)
        a_HH = torch.sigmoid(g_HH).unsqueeze(-1).unsqueeze(-1)

        F_LL = F_LL * a_LL
        F_LH = F_LH * a_LH
        F_HL = F_HL * a_HL
        F_HH = F_HH * a_HH

        F_LL, F_LH, F_HL, F_HH = self.ccca(F_LL, F_LH, F_HL, F_HH)

        F_agg = torch.cat([F_LL, F_LH, F_HL, F_HH], dim=1)  # [B, 4C, H/2, W/2]
        F_mix = self.mixer(F_agg)

        F_rec = IDWT(F_mix)  # [B, C, H, W]
        return self.out_conv(F0 + F_rec)
```

---

## 18. Novelty statement nên viết như nào

### 18.1 Nên claim
1. **Một decoder frequency-aware được language-guided**
2. **Text-guided sub-band selection** trên `LL/LH/HL/HH`
3. **Progressive coarse-to-fine multimodal refinement** qua nhiều stage decoder

### 18.2 Không nên claim
Không nên nói:
- first decoder text fusion
- first frequency LMIS
- first multimodal late fusion

Các claim đó dễ bị bắt bẻ.

---

## 19. Ablation bắt buộc

### 19.1 So với baseline
- FAENet visual-only
- FAENet + naive text gate sau decoder conv
- FAENet + text fusion sau FreqA
- **LFAENet-TGFS (ours)**

### 19.2 Số stage có text
- chỉ D4
- D4 + D3
- D4 + D3 + D2
- tất cả D4 + D3 + D2 + D1

### 19.3 Sub-band ablation
- chỉ LL gate
- chỉ HF gate
- HF/LF gate
- 4 sub-band gate đầy đủ

### 19.4 TGFS placement
- trước ICCA
- sau ICCA trước CCCA (**khuyến nghị**)
- sau CCCA

---

## 20. Kết luận cuối

Thiết kế **LFAENet-TGFS** phù hợp để làm paper hơn so với “FAENet + text ở decoder” đơn thuần, vì contribution được đẩy lên thành:

> **text-guided frequency selection inside decoder-side spectral-spatial refinement**

Nó hợp lý về mặt trực giác:
- encoder học visual frequency features
- decoder làm semantic localization + mask refinement
- text đóng vai trò semantic prior để chọn sub-band phù hợp cho lesion được mô tả

Bản thiết kế này:
- rõ ràng,
- có công thức,
- có shape,
- có block,
- có pipeline,
- có chỗ đủ mới để làm contribution.

