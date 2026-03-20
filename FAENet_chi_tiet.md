# FAENet - bản bóc tách cực chi tiết theo layer, block, shape và các điểm paper còn mơ hồ

Tài liệu này bóc tách **FAENet (Frequency Attention-Enhanced Network)** từ paper gốc thành một bản đọc-kỹ-để-code. Mục tiêu là:

1. nói rõ **paper khẳng định chắc chắn điều gì**;
2. tách riêng các phần **suy luận triển khai hợp lý** khi paper không viết đủ;
3. liệt kê **tensor shape**, **thứ tự layer/block**, **dòng chảy dữ liệu** và **các điểm mơ hồ cần quyết định khi code**.

> Nguồn chính: FAENet.pdf mà bạn vừa gửi. FAENet được mô tả là một kiến trúc encoder-decoder đối xứng, dùng **Conv Block + FreqA** ở mọi stage để học đồng thời thông tin không gian và tần số.【98:3†FAENet.pdf†L5-L18】【98:3†FAENet.pdf†L47-L69】

---

## 1) Paper nói chắc chắn những gì?

### 1.1. Khung tổng thể

Paper mô tả FAENet như sau:
- backbone encoder dựa trên **modified ResNet-50**;
- kiến trúc **encoder-decoder đối xứng**;
- mỗi stage gồm **Conv Block** xen kẽ với **FreqA**;
- encoder downsample dần để lấy multiscale feature;
- decoder upsample dần để ra mask phân đoạn cuối cùng.【98:3†FAENet.pdf†L7-L18】【98:3†FAENet.pdf†L52-L69】

### 1.2. Conv Block

Paper viết rõ một Conv Block:
- tương ứng với residual block của ResNet-50;
- gồm **3 lớp convolution kernel 3x3**;
- sau mỗi conv có **BatchNorm + ReLU**;
- có **skip connection bên trong block** để hỗ trợ gradient flow.【98:3†FAENet.pdf†L47-L51】

### 1.3. Decoder / output

Paper ghi bằng công thức:
- encoder: `Fi = ConvBlock(Fi-1)`;
- decoder: `Ui = BilinearUpsample(FN-i)`;
- skip fusion: `Gi = Concat(Ui, Fi)`;
- đầu ra: `O = Softmax(Conv1x1(G0))`.【98:3†FAENet.pdf†L52-L69】【98:9†FAENet.pdf†L21-L36】

### 1.4. FreqA

FreqA gồm các bước lớn:
1. **DWT** tách feature map thành 4 sub-band: `LL, LH, HL, HH`;
2. **FCA** gồm 2 phần: **ICCA** và **CCCA**;
3. **Self-Attention** trên feature đã gộp;
4. **iDWT** đưa feature quay lại spatial domain.【52:0†FAENet.pdf†L9-L22】【52:2†FAENet.pdf†L24-L55】

### 1.5. Shape trong FreqA mà paper ghi rõ

Với input `X ∈ R^{H×W×C}`:
- sau DWT: mỗi band có shape `H/2 × W/2 × C`;
- sau FCA: concat 4 band thành `FFCA ∈ R^{H/2 × W/2 × 4C}`;
- sau SA và iDWT: quay về `Y ∈ R^{H×W×C}`.【52:0†FAENet.pdf†L17-L22】【52:0†FAENet.pdf†L43-L46】【98:0†FAENet.pdf†L59-L74】【52:2†FAENet.pdf†L51-L56】

### 1.6. ICCA và CCCA

- **ICCA**: squeeze-excitation kiểu channel attention cho từng band riêng biệt: GAP -> FC giảm chiều -> ReLU -> FC tăng chiều -> sigmoid -> nhân channel-wise vào band gốc.【56:1†FAENet.pdf†L36-L68】
- **CCCA**: học tương quan kênh tương ứng giữa các band, paper minh họa bằng công thức cộng low-frequency với tổng các tương quan cosine-weighted từ high-frequency bands qua các trọng số học được `αk`.【56:1†FAENet.pdf†L69-L104】【98:0†FAENet.pdf†L26-L48】

### 1.7. Thiết lập huấn luyện

Paper còn ghi:
- framework: PyTorch;
- OS: Linux;
- GPU: NVIDIA A40;
- augmentation: random flipping + cropping;
- learning rate khởi tạo: `0.02`;
- số epoch tối đa: `500`;
- optimizer: SGD, momentum `0.9`;
- lr decay: polynomial với `p = 0.9`.
- ảnh được crop thành patch `256×256` cho cả Potsdam và LoveDA.【98:0†FAENet.pdf†L78-L97】【98:13†FAENet.pdf†L20-L40】【98:12†FAENet.pdf†L1-L12】

---

## 2) Cấu trúc FAENet nhìn theo pipeline end-to-end

Dưới đây là cách đọc rất thẳng theo pipeline:

```text
Input image
  -> Stem Conv 3x3
  -> Encoder Stage 1: Conv Block -> FreqA
  -> Pool 2x2
  -> Encoder Stage 2: Conv Block -> FreqA
  -> Pool 2x2
  -> Encoder Stage 3: Conv Block -> FreqA
  -> Pool 2x2
  -> Encoder Stage 4: Conv Block -> FreqA
  -> Bottleneck / deepest feature
  -> Decoder Stage 4: Bilinear upsample + skip fusion + Conv Block -> FreqA
  -> Decoder Stage 3: Bilinear upsample + skip fusion + Conv Block -> FreqA
  -> Decoder Stage 2: Bilinear upsample + skip fusion + Conv Block -> FreqA
  -> Decoder Stage 1: Bilinear upsample + skip fusion + Conv Block -> FreqA
  -> Conv 1x1
  -> Softmax
```

Hình 1 của paper cho thấy đúng tinh thần này: một stem conv đầu vào, nhiều cặp `Conv Block + FreqA` ở encoder, rồi các bước bilinear upsampling ở decoder và cuối cùng là `Conv1x1 + Softmax`.【98:3†FAENet.pdf†L7-L18】【98:3†FAENet.pdf†L43-L51】

---

## 3) Bóc tách từng khối một

# 3.1. Stem đầu vào

Paper chỉ vẽ một **Conv 3x3** trước encoder, nhưng không ghi chi tiết:
- stride bao nhiêu,
- output channels bao nhiêu,
- có BN/ReLU ngay sau stem hay không.

### Suy luận hợp lý để code

Cách ổn nhất khi re-implement:

```text
Stem:
Conv 3x3, stride=1, padding=1
BN
ReLU
```

với số kênh đầu ra là `C0` (thường chọn 64 nếu muốn giống hệ UNet/ResNet thường gặp).

---

## 3.2. Conv Block

## Paper nói gì?

Conv Block:
- gồm 3 conv 3x3;
- mỗi conv theo sau bởi BN + ReLU;
- có residual/skip bên trong block.【98:3†FAENet.pdf†L47-L51】

## Cách viết block ở mức triển khai

Một Conv Block hợp lý:

```text
input:  B x Cin x H x W
branch:
  conv1 3x3 -> BN -> ReLU
  conv2 3x3 -> BN -> ReLU
  conv3 3x3 -> BN
skip:
  identity nếu Cin = Cout
  hoặc 1x1 proj nếu Cin != Cout
output:
  ReLU(branch + skip)
shape: B x Cout x H x W
```

## Điểm cần chú ý

Paper gọi block này là “modified ResNet-50 residual block”, nhưng lại mô tả bằng **3 conv 3x3** chứ không phải bottleneck `1x1 - 3x3 - 1x1` chuẩn của ResNet-50. Tức là:
- **ý tưởng residual của ResNet-50** còn giữ,
- nhưng **micro-architecture** đã bị sửa.

Nói cách khác: đây **không còn là bottleneck block chuẩn ResNet-50** nữa.

---

## 3.3. Encoder

## Paper nói gì?

Encoder downsample bằng pool 2x2 và sinh ra các feature map đa tỉ lệ `Fi`. Sau mỗi Conv Block sẽ có một FreqA để làm giàu spectral-spatial context.【98:3†FAENet.pdf†L52-L58】

## Dạng hình học chắc chắn

Nếu input là `B x C_in x H x W`, encoder sẽ tạo các mức:

```text
F1: B x C1 x H   x W
F2: B x C2 x H/2 x W/2
F3: B x C3 x H/4 x W/4
F4: B x C4 x H/8 x W/8
F5 (nếu có bottleneck sau pool nữa): B x C5 x H/16 x W/16
```

Tuy nhiên, hình trong paper vẽ **4 khối encoder chính sau stem**, nên cách đọc thực dụng nhất là:
- Stage 1 ở full-res,
- sau đó pool,
- Stage 2 ở 1/2,
- Stage 3 ở 1/4,
- Stage 4 ở 1/8,
- deepest tensor sau pool cuối hoặc ngay sau stage 4 tùy cách code.

## Dòng chảy stage-by-stage (triển khai hợp lý)

Với input 256×256:

```text
Input     : B x 3   x 256 x 256
Stem      : B x C0  x 256 x 256
Enc1 CB   : B x C1  x 256 x 256
Enc1 FreqA: B x C1  x 256 x 256
Pool      : B x C1  x 128 x 128

Enc2 CB   : B x C2  x 128 x 128
Enc2 FreqA: B x C2  x 128 x 128
Pool      : B x C2  x  64 x  64

Enc3 CB   : B x C3  x  64 x  64
Enc3 FreqA: B x C3  x  64 x  64
Pool      : B x C3  x  32 x  32

Enc4 CB   : B x C4  x  32 x  32
Enc4 FreqA: B x C4  x  32 x  32
(deepest)
```

> Lưu ý: paper không in ra bảng kênh `C1..C4`, nên phần này chỉ chắc về **tỉ lệ spatial**, còn **số kênh** phải do ta chọn khi code.【98:3†FAENet.pdf†L52-L69】

---

## 3.4. Decoder

## Paper nói gì?

Decoder đối xứng với encoder:
- bilinear upsample;
- trộn với skip từ encoder cùng scale;
- qua Conv Block để refine; và theo hình thì decoder cũng xen kẽ FreqA.【98:3†FAENet.pdf†L59-L69】【98:3†FAENet.pdf†L7-L18】

## Mâu thuẫn quan trọng cần biết

Có **một chỗ paper không nhất quán**:
- **Equation (3)** viết rõ là `Gi = Concat(Ui, Fi)` -> nghĩa là **concatenation theo channel**.【98:3†FAENet.pdf†L61-L69】
- Nhưng **Figure 1** lại có legend “Element-wise Sum” và hình tròn dấu cộng ở các skip fusion.【98:3†FAENet.pdf†L16-L18】【98:3†FAENet.pdf†L40-L46】

### Kết luận thực dụng

Nếu code lại, nên ưu tiên **Equation (3) = concatenation**, vì:
1. paper viết ra công thức tường minh;
2. decoder kiểu UNet/encoder-decoder đối xứng thường concat hợp lý hơn;
3. sum đòi hỏi số kênh encoder và decoder trùng tuyệt đối.

---

## 3.5. FreqA - khối quan trọng nhất của FAENet

FreqA là nơi FAENet khác các encoder-decoder CNN bình thường.

### 3.5.1. Input/output của FreqA

Input của FreqA:

```text
X ∈ R^(H x W x C)
```

Output của FreqA:

```text
Y ∈ R^(H x W x C)
```

Tức là **FreqA không đổi external shape**, nên có thể cắm sau bất kỳ Conv Block nào mà không phá flow của encoder/decoder.【52:0†FAENet.pdf†L17-L22】【52:2†FAENet.pdf†L51-L56】

### 3.5.2. Bước 1 - DWT

Paper nói DWT tách `X` thành bốn thành phần:
- `FLL`: low-frequency / coarse approximation,
- `FLH`: high-frequency ngang,
- `FHL`: high-frequency dọc,
- `FHH`: high-frequency chéo.【52:0†FAENet.pdf†L17-L22】【98:9†FAENet.pdf†L37-L42】

Shape của từng band:

```text
FLL, FLH, FHL, FHH ∈ R^(H/2 x W/2 x C)
```

Nghĩa là DWT giảm spatial resolution đi 2 lần theo mỗi chiều, nhưng giữ số channel per-band là `C`.

---

## 3.6. FCA bên trong FreqA

FCA = `ICCA + CCCA`.

### 3.6.1. ICCA

ICCA chạy **riêng cho từng band**.

#### Công thức

Với mỗi band `Fk ∈ R^(H/2 x W/2 x C)`:

1. **GAP** theo spatial để lấy vector channel descriptor `zk ∈ R^C`:

```text
zk(c) = average over spatial positions of Fk(:,:,c)
```

2. **FC giảm chiều**:

```text
z_k^FC = ReLU(W1 zk + b1)
W1 ∈ R^(C x C/r)
```

3. **FC tăng chiều + sigmoid** để sinh attention weights `ak ∈ R^C`:

```text
ak = Sigmoid(W2 z_k^FC + b2)
W2 ∈ R^(C/r x C)
```

4. **Reweight theo channel**:

```text
F_k^ICCA(i,j,c) = a_k(c) * F_k(i,j,c)
```

Paper ghi toàn bộ đúng tinh thần trên.【56:1†FAENet.pdf†L36-L68】

#### Shape thật sự trong ICCA

```text
Input band            : B x C x H/2 x W/2
GAP                   : B x C
FC squeeze            : B x (C/r)
FC expand + sigmoid   : B x C
Broadcast multiply    : B x C x H/2 x W/2
```

Đây đúng kiểu SE-attention nhưng áp riêng cho từng wavelet sub-band.

---

### 3.6.2. CCCA

CCCA làm việc **sau ICCA**, mục tiêu là cho các band trao đổi thông tin phổ với nhau.

Paper cho ví dụ với band low-frequency:

```text
a_LL(c) = F_LL^ICCA(c) + sum_k α_k * Corr(F_LL^ICCA(c), F_k^ICCA(c))
```

trong đó:
- `k ∈ {LH, HL, HH}`
- `α_k` là trọng số học được;
- `Corr(·,·)` dùng **cosine similarity** trên các channel tương ứng giữa các band khác nhau.【56:1†FAENet.pdf†L69-L104】【98:0†FAENet.pdf†L26-L48】

#### Ý nghĩa trực giác

- `LL` giữ semantic/coarse context.
- `LH/HL/HH` giữ edge/texture theo các hướng.
- CCCA cho phép từng kênh trong một band biết “band khác đang nói gì”, thay vì mỗi band tự xử lý độc lập.

#### Shape sau CCCA

Paper nói sau khi refine xong từng band rồi concat lại:

```text
FFCA = Concat(F_LL^CCCA, F_LH^CCCA, F_HL^CCCA, F_HH^CCCA)
FFCA ∈ R^(H/2 x W/2 x 4C)
```

【98:0†FAENet.pdf†L59-L74】

---

## 3.7. Self-Attention trong FreqA

Sau FCA, paper đưa `Fagg`/`FFCA` vào self-attention để nắm quan hệ dài hạn hơn trong feature map tần số đã gộp.【52:2†FAENet.pdf†L35-L51】

### Công thức paper

```text
Q = Wq Fagg
K = Wk Fagg
V = Wv Fagg
FSA = Softmax(QK^T / sqrt(dk)) V
```

【52:2†FAENet.pdf†L35-L51】

### Shape triển khai hợp lý

Nếu `Fagg` có shape `B x 4C x H/2 x W/2`, ta flatten spatial:

```text
N = (H/2)*(W/2)
Fagg_flat : B x N x 4C
Q,K,V     : B x N x d
Attn      : B x N x N
FSA_flat  : B x N x 4C   (hoặc B x N x d_model rồi project về 4C)
reshape   : B x 4C x H/2 x W/2
```

Paper không nói multi-head hay single-head, cũng không nói `d_model` cụ thể. Vì vậy khi code cần tự chốt.

---

## 3.8. iDWT trong FreqA

Paper viết:

```text
Y = iDWT(FSA)
```

và đầu ra `Y ∈ R^(H x W x C)`.【52:2†FAENet.pdf†L51-L56】

## Đây là chỗ mơ hồ nhất của paper

Về mặt toán học, **iDWT cần 4 sub-band tách riêng** (`LL, LH, HL, HH`). Nhưng ngay trước đó paper lại:
- concat 4 band -> `Fagg/FFCA` có `4C` channels;
- áp self-attention lên tensor đã concat;
- rồi viết trực tiếp `iDWT(FSA)`.

### Suy luận triển khai hợp lý nhất

Muốn code được, phải hiểu ngầm rằng:

```text
FSA ∈ B x (4C) x H/2 x W/2
split channel-wise thành 4 tensor:
  FSA_LL, FSA_LH, FSA_HL, FSA_HH ∈ B x C x H/2 x W/2
sau đó mới iDWT(FSA_LL, FSA_LH, FSA_HL, FSA_HH)
-> Y ∈ B x C x H x W
```

Nói ngắn gọn: **paper bỏ qua thao tác split lại 4 band trước iDWT**.

---

## 4) Bảng shape đầy đủ cho input 256x256

Vì paper không công bố bảng channel chính thức, dưới đây là **bảng shape chắc chắn về spatial** và **bảng shape gợi ý để triển khai**.

## 4.1. Bảng shape chắc chắn theo paper (symbolic channels)

| Stage | Operation | Output shape |
|---|---|---|
| Input | RGB image | `B x 3 x 256 x 256` |
| Stem | Conv 3x3 | `B x C0 x 256 x 256` |
| Enc1 | Conv Block -> FreqA | `B x C1 x 256 x 256` |
| Pool1 | 2x2 | `B x C1 x 128 x 128` |
| Enc2 | Conv Block -> FreqA | `B x C2 x 128 x 128` |
| Pool2 | 2x2 | `B x C2 x 64 x 64` |
| Enc3 | Conv Block -> FreqA | `B x C3 x 64 x 64` |
| Pool3 | 2x2 | `B x C3 x 32 x 32` |
| Enc4 | Conv Block -> FreqA | `B x C4 x 32 x 32` |
| Dec4 | Upsample + skip fusion + Conv Block -> FreqA | `B x D4 x 64 x 64` |
| Dec3 | Upsample + skip fusion + Conv Block -> FreqA | `B x D3 x 128 x 128` |
| Dec2 | Upsample + skip fusion + Conv Block -> FreqA | `B x D2 x 256 x 256` |
| Head | Conv 1x1 + Softmax | `B x K x 256 x 256` |

> Đây là bảng “an toàn”: đúng tinh thần paper nhưng không bịa số channel khi paper không cho.

---

## 4.2. Bảng shape gợi ý để code lại theo kiểu UNet-Residual hợp lý

Nếu cần một re-implementation gọn, dễ train, đúng tinh thần FAENet:

- `C0 = 64`
- `C1 = 64`
- `C2 = 128`
- `C3 = 256`
- `C4 = 512`
- decoder mirror lại: `512 -> 256 -> 128 -> 64`

thì shape có thể là:

| Stage | Output shape |
|---|---|
| Input | `B x 3 x 256 x 256` |
| Stem | `B x 64 x 256 x 256` |
| Enc1 | `B x 64 x 256 x 256` |
| Pool1 | `B x 64 x 128 x 128` |
| Enc2 | `B x 128 x 128 x 128` |
| Pool2 | `B x 128 x 64 x 64` |
| Enc3 | `B x 256 x 64 x 64` |
| Pool3 | `B x 256 x 32 x 32` |
| Enc4 | `B x 512 x 32 x 32` |
| Up4 | `B x 512 x 64 x 64` |
| Fuse4 | `B x (512+256) x 64 x 64` nếu concat |
| Dec4 | `B x 256 x 64 x 64` |
| Up3 | `B x 256 x 128 x 128` |
| Fuse3 | `B x (256+128) x 128 x 128` |
| Dec3 | `B x 128 x 128 x 128` |
| Up2 | `B x 128 x 256 x 256` |
| Fuse2 | `B x (128+64) x 256 x 256` |
| Dec2 | `B x 64 x 256 x 256` |
| Head | `B x K x 256 x 256` |

### FreqA ở mỗi stage sẽ có shape nội bộ

Ví dụ tại `Enc3`, input FreqA là `B x 256 x 64 x 64`:
- DWT -> 4 band: mỗi band `B x 256 x 32 x 32`
- ICCA từng band: giữ nguyên shape
- CCCA từng band: giữ nguyên shape
- concat -> `B x 1024 x 32 x 32`
- SA -> `B x 1024 x 32 x 32`
- split 4 nhóm -> 4 tensor `B x 256 x 32 x 32`
- iDWT -> `B x 256 x 64 x 64`

Đây là một cách diễn đạt rất quan trọng: **FreqA nội bộ phình channel lên 4 lần, nhưng output ngoài vẫn quay lại đúng shape ban đầu**.

---

## 5) Pseudocode triển khai FAENet

```python
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        ...
    def forward(self, x):
        identity = proj(x) if cin != cout else x
        out = conv1_bn_relu(x)
        out = conv2_bn_relu(out)
        out = conv3_bn(out)
        out = relu(out + identity)
        return out

class ICCA(nn.Module):
    def __init__(self, c, r=16):
        ...
    def forward(self, x):
        # x: B,C,h,w
        z = gap(x)                # B,C
        z = relu(fc1(z))          # B,C/r
        a = sigmoid(fc2(z))       # B,C
        return x * a[:, :, None, None]

class CCCA(nn.Module):
    def __init__(self, c):
        ...
    def forward(self, fll, flh, fhl, fhh):
        # pairwise / low-high correlation across matching channels
        ...
        return fll2, flh2, fhl2, fhh2

class FreqA(nn.Module):
    def __init__(self, c):
        ...
    def forward(self, x):
        fll, flh, fhl, fhh = dwt(x)  # each B,C,h/2,w/2
        fll = icca(fll)
        flh = icca(flh)
        fhl = icca(fhl)
        fhh = icca(fhh)
        fll, flh, fhl, fhh = ccca(fll, flh, fhl, fhh)
        f = torch.cat([fll, flh, fhl, fhh], dim=1)  # B,4C,h/2,w/2
        f = self_attention(f)
        fll, flh, fhl, fhh = torch.chunk(f, 4, dim=1)
        y = idwt(fll, flh, fhl, fhh)               # B,C,H,W
        return y

class FAENet(nn.Module):
    def __init__(self, num_classes):
        ...
    def forward(self, x):
        x0 = stem(x)
        e1 = freqa1(cb1(x0))
        p1 = pool(e1)
        e2 = freqa2(cb2(p1))
        p2 = pool(e2)
        e3 = freqa3(cb3(p2))
        p3 = pool(e3)
        e4 = freqa4(cb4(p3))

        d4 = up(e4)
        d4 = fuse(d4, e3)   # concat ưu tiên theo Eq.(3)
        d4 = freqa5(cb5(d4))

        d3 = up(d4)
        d3 = fuse(d3, e2)
        d3 = freqa6(cb6(d3))

        d2 = up(d3)
        d2 = fuse(d2, e1)
        d2 = freqa7(cb7(d2))

        out = head_1x1(d2)
        out = softmax(out)
        return out
```

---

## 6) Các điểm paper mơ hồ hoặc mâu thuẫn - rất quan trọng nếu bạn muốn code đúng

## 6.1. `Concat` hay `Element-wise Sum` ở skip fusion?

- Equation (3): **Concat**.【98:3†FAENet.pdf†L61-L69】
- Figure legend: có biểu tượng **Element-wise Sum**.【98:3†FAENet.pdf†L40-L46】

### Khuyến nghị
Ưu tiên **Concat** vì đó là công thức tường minh trong phần mô tả toán học.

## 6.2. `G0` là gì?

Paper viết `O = Softmax(Conv1x1(G0))` nhưng không định nghĩa riêng `G0`.【98:9†FAENet.pdf†L21-L24】

### Suy luận hợp lý
`G0` chính là **tensor decoder cuối cùng ở full resolution**, ngay trước segmentation head.

## 6.3. `Fagg` hay `FFCA`?

Paper dùng cả hai cách gọi ở các chỗ khác nhau:
- sau concat 4 band, có lúc gọi `Fagg`,【52:2†FAENet.pdf†L24-L35】
- có lúc gọi `FFCA`.【98:0†FAENet.pdf†L59-L74】

### Suy luận hợp lý
Hai tên này đang chỉ **cùng một tensor**: feature map sau khi concat 4 band sau FCA.

## 6.4. iDWT nhận đầu vào gì?

Paper viết `Y = iDWT(FSA)` nhưng trước đó SA chạy trên tensor đã concat 4C channels. Điều này không đủ chi tiết để code trực tiếp.【52:2†FAENet.pdf†L35-L56】

### Suy luận hợp lý
Phải `chunk/split` `FSA` thành 4 band rồi mới iDWT.

## 6.5. Số stage và số kênh mỗi stage

Paper không công bố bảng channel chi tiết, chỉ cho hình khối. Vì vậy:
- spatial pyramid có thể suy ra khá chắc,
- channel widths thì phải do implementer quyết định.

---

## 7) Vai trò thực sự của từng thành phần

## 7.1. Conv Block

- học feature cục bộ trong spatial domain;
- xây nền feature tương tự CNN encoder-decoder thường gặp.

## 7.2. DWT

- tách feature thành thành phần thô (`LL`) và chi tiết hướng (`LH/HL/HH`);
- giúp block attention làm việc trực tiếp trên miền tần số thay vì spatial thuần túy.【52:0†FAENet.pdf†L17-L22】【52:0†FAENet.pdf†L43-L48】

## 7.3. ICCA

- trả lời câu hỏi: *trong từng band, channel nào đáng tin hơn?*【56:1†FAENet.pdf†L36-L68】

## 7.4. CCCA

- trả lời câu hỏi: *giữa các band, kênh nào nên tham chiếu nhau?*【56:1†FAENet.pdf†L69-L104】

## 7.5. SA

- thêm dependency dài hạn sau khi đã gộp spectral bands, để không chỉ biết “kênh nào quan trọng” mà còn biết “vùng nào tương tác với vùng nào”.【52:2†FAENet.pdf†L35-L51】

## 7.6. iDWT

- đưa feature đã refine quay lại không gian ảnh để tiếp tục đi trong encoder/decoder thường. Đây là điểm giúp FreqA cắm được vào CNN mà không đổi external shape.【52:2†FAENet.pdf†L51-L56】

---

## 8) Tại sao ICCA + CCCA là phần cốt lõi?

Ablation của paper cho thấy:
- bỏ cả ICCA và CCCA thì mIoU thấp hơn rõ rệt;
- chỉ ICCA tốt hơn baseline;
- chỉ CCCA cũng tốt hơn baseline;
- dùng cả hai là tốt nhất trên cả Potsdam và LoveDA.

Cụ thể:
- baseline: Potsdam `73.15 mIoU`, LoveDA `57.93 mIoU`;
- chỉ ICCA: Potsdam `76.97`, LoveDA `61.07`;
- chỉ CCCA: Potsdam `77.72`, LoveDA `61.66`;
- cả hai: Potsdam `83.58`, LoveDA `66.91`.

Điều đó xác nhận rằng **ICCA và CCCA có vai trò bổ sung nhau**: một cái refine trong từng band, một cái nối thông tin giữa các band.【98:2†FAENet.pdf†L1-L32】

---

## 9) Nếu bạn muốn code lại FAENet cho bài của mình, đâu là bản “ít rủi ro nhất”?

Đây là cấu hình mình cho là ít rủi ro nhất khi re-implement:

### 9.1. Kiến trúc ngoài
- stem conv 3x3 + BN + ReLU;
- 4 encoder stages;
- mỗi stage: `ConvBlock -> FreqA -> Pool` (trừ stage cuối không pool);
- decoder 3 hoặc 4 stages đối xứng;
- skip fusion bằng **concat**;
- sau mỗi decoder block cũng đặt `FreqA` như hình paper.

### 9.2. Conv Block
- 3 conv 3x3;
- residual skip nội bộ;
- channel progression kiểu `64, 128, 256, 512`.

### 9.3. FreqA
- DWT Haar;
- ICCA dùng reduction ratio `r = 16` nếu paper không cho cụ thể;
- CCCA dùng cosine similarity theo channel tương ứng;
- SA dùng 1-head hoặc 4-head đều được, miễn output project về `4C`;
- split `4C -> 4 × C` trước iDWT.

### 9.4. Huấn luyện
- patch size 256×256;
- SGD, momentum 0.9;
- lr 0.02, poly decay;
- 500 epochs như paper.【98:0†FAENet.pdf†L78-L97】【98:13†FAENet.pdf†L20-L40】

---

## 10) Kết luận ngắn gọn

Nếu tóm FAENet đúng bản chất, thì nó là:

> **một encoder-decoder CNN residual, trong đó mỗi stage được gắn thêm một khối attention miền tần số (FreqA) có cấu trúc DWT -> ICCA -> CCCA -> Self-Attention -> iDWT.**

Điểm đáng giá nhất của FAENet không nằm ở UNet-like skeleton, mà nằm ở việc:
- tách feature thành **4 wavelet sub-bands**,
- refine **trong từng band** bằng ICCA,
- nối thông tin **giữa các band** bằng CCCA,
- rồi mới dùng SA để gom global dependency.

Nói ngắn: FAENet không chỉ “thêm attention vào CNN”, mà là **đẩy attention vào miền tần số**, rồi đưa nó trở lại spatial domain để tiếp tục đi qua encoder-decoder chuẩn.【52:0†FAENet.pdf†L9-L16】【98:0†FAENet.pdf†L59-L77】

---

## 11) Một câu chốt rất thực dụng cho bạn

Nếu mục tiêu của bạn là **dùng FAENet như xương sống ý tưởng để lai với text-guided LMIS**, thì phần giữ lại đáng giá nhất không phải toàn bộ paper, mà là ba ý sau:

1. `ConvBlock -> FreqA` như một plug-in block;
2. nội bộ FreqA phải giữ đúng logic `DWT -> per-band refine -> cross-band refine -> global SA -> iDWT`;
3. phải sửa các chỗ paper mơ hồ ngay từ đầu: **concat ở skip**, **split trước iDWT**, **chốt channel schedule rõ ràng**.

Khi ba phần này rõ, bạn mới có nền sạch để gắn text vào decoder sau đó.
