import type { CodePracticeProblem } from './code-practice';

export const ARCHITECTURE_CODE_PRACTICE_PROBLEMS = [
  {
    id: 'resnet-from-building-blocks',
    order: 38,
    title: 'Implement ResNet-18 with basic blocks',
    difficulty: 'Hard',
    track: 'architecture',
    summary:
      'Implement a small ResNet-18-style image classifier with two-convolution residual blocks, projected skip connections, and global average pooling.',
    prompt: [
      'Implement a small ResNet-18-style image classifier in PyTorch. Each residual block should follow `y = F(x) + x`, where `F(x)` contains two `3x3` convolutions.',
      'Build `BasicBlock` and `ResNet` for inputs shaped `(B, 3, H, W)`. Use BatchNorm and ReLU, project the skip connection when the shape changes, downsample between stages with `stride=2`, and return logits shaped `(B, num_classes)`.',
    ],
    signature: `class BasicBlock(nn.Module): ...

class ResNet(nn.Module): ...`,
    requirements: [
      'Use `nn.Module` subclasses for the residual block and network.',
      'Each `BasicBlock` must contain two `3x3` convolutions, with BatchNorm and ReLU in the residual path.',
      'Project the skip path with a `1x1` convolution when stride changes or channel counts differ; otherwise use the identity skip.',
      'Build multiple stages, using `stride=2` on the first block of each downsampling stage.',
      'Apply global average pooling before the final linear classifier.',
      'Return logits shaped `(B, num_classes)` for an input shaped `(B, 3, H, W)`.',
    ],
    examples: [
      {
        label: 'Shape check',
        lines: [
          'model = ResNet(num_classes=10)',
          'x.shape = (4, 3, 64, 64)',
        ],
        result: 'model(x).shape == (4, 10)',
      },
    ],
    hint: [
      'A residual addition is valid only when the main and skip paths have the same shape. Use a `1x1` projection when they do not.',
      'Put the requested stride on the first convolution of the residual path and on the projection skip.',
      'Only the first block in a downsampling stage uses `stride=2`; later blocks in that stage keep `stride=1`.',
      'Adaptive average pooling maps `(B, C, Hf, Wf)` to `(B, C, 1, 1)` before flattening.',
    ],
    interview: {
      durationMinutes: 50,
      evaluationCriteria: [
        'Explains the identity-versus-projection decision before coding.',
        'Separates the reusable block, stage assembly, and forward path.',
        'Checks the final tensor shape and names one production concern such as initialization or normalization.',
      ],
      followUps: [
        'How would you generalize this to bottleneck blocks?',
        'What changes would you make for small images such as CIFAR-10?',
      ],
    },
    solutionNotes: [
      'The previous exercise is a small ResNet-18-style model: its four stages use `[2, 2, 2, 2]` `BasicBlock`s. Each block learns a residual correction with two `3x3` convolutions. BatchNorm and ReLU shape the residual branch, then the block adds the skip path and applies a final ReLU:\n`output = ReLU(F(x) + skip(x))`',
      'Addition requires identical tensor shapes. When the channel count or stride changes, the skip uses a `1x1` convolution with the same stride; when the shape already matches, the identity skip (`nn.Identity()`) preserves the input without adding parameters.',
      'The first block of each later stage performs spatial downsampling with `stride=2`. The remaining blocks use stride one, so a four-stage network preserves its feature map within each stage instead of shrinking it at every block.',
      '`make_stage` keeps stage construction consistent: create the first block with the requested stride, update `self.in_channels`, then append stride-one blocks. The forward method can then stack the stages without repeating block logic.',
      'Adaptive average pooling removes the spatial axes for any valid feature-map size:\n`(B, C, Hf, Wf) → (B, C, 1, 1) → (B, C) → (B, num_classes)`',
      'The shape test checks the interview contract directly: a batch of RGB images produces one logit vector per image. It also catches a missing projection, an incorrectly placed stride, or a classifier wired to the wrong channel count.',
    ],
    solutionDiagram: `input (B, 3, H, W)
  -> stem
  -> stage 1
  -> stage 2 / 2
  -> stage 3 / 2
  -> stage 4 / 2
  -> adaptive average pool
  -> linear classifier
  -> logits (B, num_classes)`,
    starterCode: `import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # TODO: define two 3x3 convolutions with BatchNorm and ReLU.
        # TODO: use a 1x1 projection when the skip shape changes.
        raise NotImplementedError("Implement __init__")

    def forward(self, x):
        # TODO: compute the residual branch, add the skip branch, then apply ReLU.
        raise NotImplementedError("Implement forward")


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # TODO: create the stem, four stages, global average pool, and classifier.
        raise NotImplementedError("Implement __init__")

    def make_stage(self, out_channels, num_blocks, stride):
        # TODO: downsample only the first block, then use stride one.
        raise NotImplementedError("Implement make_stage")

    def forward(self, x):
        # TODO: return logits shaped (B, num_classes).
        raise NotImplementedError("Implement forward")


def test_resnet():
    model = ResNet(num_classes=10)
    x = torch.randn(4, 3, 64, 64)
    y = model(x)
    assert y.shape == (4, 10)
    print(y.shape)


if __name__ == "__main__":
    test_resnet()`,
    solutionCode: `import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        # Project the skip connection if its shape changes.
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x = x + residual
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.stage1 = self.make_stage(64, num_blocks=2, stride=1)
        self.stage2 = self.make_stage(128, num_blocks=2, stride=2)
        self.stage3 = self.make_stage(256, num_blocks=2, stride=2)
        self.stage4 = self.make_stage(512, num_blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def make_stage(self, out_channels, num_blocks, stride):
        blocks = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels

        for _ in range(num_blocks - 1):
            blocks.append(BasicBlock(out_channels, out_channels))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        return self.fc(x)


def test_resnet():
    model = ResNet(num_classes=10)

    x = torch.randn(4, 3, 64, 64)
    y = model(x)

    print(y.shape)
    assert y.shape == (4, 10)


test_resnet()`,
    packages: ['torch'],
    tags: ['PyTorch', 'CNNs', 'ResNet', 'Architecture'],
  },
  {
    id: 'resnet-50-bottleneck-blocks',
    order: 39,
    title: 'Implement ResNet-50 with bottleneck blocks',
    difficulty: 'Hard',
    track: 'architecture',
    summary:
      'Implement ResNet-50 with three-convolution bottleneck blocks, fourfold channel expansion, projected skips, and global average pooling.',
    prompt: [
      'Implement a ResNet-50 image classifier in PyTorch. Replace the two-convolution `BasicBlock` with a `Bottleneck` block whose path is `1x1 -> 3x3 -> 1x1`.',
      'Use the bottleneck expansion factor of `4`, the ResNet-50 stage depths `[3, 4, 6, 3]`, and an ImageNet-style `7x7` stem with stride two followed by `3x3` max pooling. Return logits shaped `(B, num_classes)` for inputs shaped `(B, 3, H, W)`.',
    ],
    signature: `class Bottleneck(nn.Module): ...

class ResNet50(nn.Module): ...`,
    requirements: [
      'Implement `Bottleneck` and `ResNet50` as `nn.Module` subclasses.',
      'Use a `1x1`, `3x3`, `1x1` residual path with BatchNorm and ReLU after the first two convolutions.',
      'Expand the final bottleneck channels by `4` and project the skip path when channels or spatial size change.',
      'Build four stages with block counts `[3, 4, 6, 3]`; only the first block of stages two through four uses `stride=2`.',
      'Use global average pooling before a linear classifier and return shape `(B, num_classes)`.',
    ],
    examples: [
      {
        label: 'Shape check',
        lines: [
          'model = ResNet50(num_classes=10)',
          'x.shape = (2, 3, 224, 224)',
        ],
        result: 'model(x).shape == (2, 10)',
      },
    ],
    hint: [
      'The middle `3x3` convolution performs spatial processing; put the stage stride there and on the skip projection.',
      'A bottleneck with base width `C` outputs `4C` channels, so the next block in the same stage receives `4C` input channels.',
      'The stage pattern to memorize is `3, 4, 6, 3`; the first block carries the stage stride and the rest use stride one.',
      'Adaptive average pooling turns the final `(B, 2048, Hf, Wf)` feature map into `(B, 2048, 1, 1)` before flattening.',
    ],
    interview: {
      durationMinutes: 50,
      evaluationCriteria: [
        'Explains why the bottleneck expands channels after the spatial convolution.',
        'Keeps the residual and projection paths shape-compatible at every stage boundary.',
        'Uses the `[3, 4, 6, 3]` depth pattern and verifies the final classifier shape.',
      ],
      followUps: [
        'Why can a bottleneck block be cheaper than three full-width convolutions?',
        'How would you adapt this stem and stage schedule for CIFAR-10?',
      ],
    },
    solutionNotes: [
      'ResNet-50 changes the block, not the residual principle. The residual path is:\n`1x1 -> 3x3 -> 1x1`\nThe first convolution sets the hidden width, the `3x3` processes space, and the last convolution expands the output to `4 * out_channels` before addition.',
      'The skip path must end at the same channel count as the bottleneck output. That means the first block of a stage projects from `in_channels` to `4 * out_channels`; later blocks receive `4 * out_channels` and can use the identity skip when their spatial shape also matches.',
      'The stage depths `[3, 4, 6, 3]` are the ResNet-50 signature. The first block of stages two, three, and four carries `stride=2` in both paths, while the remaining blocks preserve that stage’s spatial resolution.',
      'The channel arithmetic is easier to remember than the full module: base widths are `64, 128, 256, 512`, bottleneck outputs are `256, 512, 1024, 2048`, and the classifier therefore consumes `2048` features.',
      'The diagram summarizes the implementation: remember `1-3-1` for the bottleneck path, `×4` for channel expansion, and `3-4-6-3` for stage depth. Adaptive average pooling then removes the final spatial axes before classification.',
      'The shape test uses a batch of `224x224` RGB images and ten classes. It catches incorrect expansion, missing projection shortcuts, misplaced strides, and a classifier connected to the hidden width instead of the expanded width.',
    ],
    solutionDiagram: `ResNet-50 memory map

input (B, 3, H, W)
  -> 7x7 conv, stride 2 -> 3x3 max pool, stride 2
  -> [1x1 -> 3x3 -> 1x1] x 3, base 64, output 256
  -> [1x1 -> 3x3 -> 1x1] x 4, stride 2, base 128, output 512
  -> [1x1 -> 3x3 -> 1x1] x 6, stride 2, base 256, output 1024
  -> [1x1 -> 3x3 -> 1x1] x 3, stride 2, base 512, output 2048
  -> global average pool -> linear -> logits (B, num_classes)

Remember: 1-3-1 inside each block, x4 at the output, 3-4-6-3 across stages.`,
    starterCode: `import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        # TODO: build the 1x1 -> 3x3 -> 1x1 residual path.
        # TODO: expand the output channels by Bottleneck.expansion.
        # TODO: project the skip when its shape changes.
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: run both paths, add them, and apply the final ReLU.
        raise NotImplementedError("Implement forward")


class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()

        # TODO: create the 7x7 stem, [3, 4, 6, 3] stages, pool, and classifier.
        raise NotImplementedError("Implement __init__")

    def make_stage(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        # TODO: downsample only the first Bottleneck in this stage.
        raise NotImplementedError("Implement make_stage")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return logits shaped (B, num_classes).
        raise NotImplementedError("Implement forward")


def test_resnet50() -> None:
    model = ResNet50(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 10)
    print(y.shape)


test_resnet50()`,
    solutionCode: `import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        hidden_channels = out_channels

        # 1x1: reduce / set channel width
        self.conv1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        # 3x3: spatial processing
        self.conv2 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        # 1x1: expand channels by 4
        self.conv3 = nn.Conv2d(
            hidden_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()

        final_channels = out_channels * self.expansion

        # Match residual shape if spatial size or channels change.
        if stride != 1 or in_channels != final_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    final_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(final_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        x = x + residual
        return self.relu(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()

        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ResNet-50 block counts: [3, 4, 6, 3]
        self.stage1 = self.make_stage(64, num_blocks=3, stride=1)
        self.stage2 = self.make_stage(128, num_blocks=4, stride=2)
        self.stage3 = self.make_stage(256, num_blocks=6, stride=2)
        self.stage4 = self.make_stage(512, num_blocks=3, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def make_stage(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        blocks = [
            Bottleneck(
                self.in_channels,
                out_channels,
                stride=stride,
            )
        ]

        self.in_channels = out_channels * Bottleneck.expansion

        for _ in range(num_blocks - 1):
            blocks.append(
                Bottleneck(
                    self.in_channels,
                    out_channels,
                    stride=1,
                )
            )

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        return self.fc(x)


def test_resnet50():
    model = ResNet50(num_classes=10)

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print(y.shape)
    assert y.shape == (2, 10)


test_resnet50()`,
    packages: ['torch'],
    tags: ['PyTorch', 'CNNs', 'ResNet', 'Bottleneck', 'Architecture'],
  },
  {
    id: 'unet-encoder-decoder',
    order: 40,
    title: 'Implement U-Net with double-convolution blocks',
    difficulty: 'Hard',
    track: 'architecture',
    summary:
      'Implement a small four-level U-Net with DoubleConv blocks, transposed-convolution upsampling, and skip connections for per-pixel prediction.',
    prompt: [
      'A segmentation model should combine deep context with fine spatial detail. Implement `DoubleConv` and `UNet` for inputs shaped `(B, in_channels, H, W)`.',
      'Use four encoder levels and four decoder levels. For this simple interview version, assume `H` and `W` are divisible by 16, and return per-pixel logits shaped `(B, num_classes, H, W)`.',
    ],
    signature: `class DoubleConv(nn.Module): ...
class UNet(nn.Module): ...`,
    requirements: [
      'Implement `DoubleConv` with two `3x3` convolutions and a ReLU after each convolution.',
      'Build encoder levels with channel widths `64, 128, 256, 512`, using `MaxPool2d(2)` between levels.',
      'Use a `1024`-channel bottleneck and four `ConvTranspose2d` upsamplers with `kernel_size=2` and `stride=2`.',
      'Concatenate each upsampled tensor with its matching encoder feature along `dim=1`, then apply a decoder `DoubleConv`.',
      'Use a `1x1` convolutional head and return logits shaped `(B, num_classes, H, W)`.',
      'Assume `H` and `W` are divisible by 16 so the four downsampling and upsampling paths align exactly.',
    ],
    examples: [
      {
        label: 'Segmentation shape check',
        lines: [
          'model = UNet(in_channels=3, num_classes=4)',
          'images.shape = (2, 3, 128, 128)',
        ],
        result: 'model(images).shape == (2, 4, 128, 128)',
      },
    ],
    hint: [
      'Save `x1` through `x4` before pooling; those are the skip features.',
      'After concatenation, the decoder inputs have `1024`, `512`, `256`, and `128` channels.',
      'The four `2x` pools divide each spatial dimension by 16; the four transposed convolutions reverse that.',
      'The `1x1` head changes channels to `num_classes` without changing the image grid.',
    ],
    interview: {
      durationMinutes: 50,
      evaluationCriteria: [
        'Explains what information the skip connections restore.',
        'Keeps the channel bookkeeping correct across concatenation and decoder convolutions.',
        'States the divisible-by-16 input constraint and output-logit contract.',
      ],
      followUps: [
        'When would bilinear upsampling be preferable to transposed convolution?',
        'What changes are needed to support odd input dimensions?',
        'Which loss would you use for multi-class versus multi-label segmentation?',
      ],
    },
    solutionNotes: [
      'Each `DoubleConv` preserves `(H, W)` because both `3x3` convolutions use `padding=1`. The two ReLUs add the nonlinearity after each spatial-processing step.',
      'The encoder saves `x1` through `x4` before pooling. Pooling halves the grid while increasing channels, and the bottleneck operates at `(H/16, W/16)` with `1024` channels.',
      'Every decoder stage follows the same memory pattern:\n`upsample -> concatenate skip -> DoubleConv`\nConcatenation is along `dim=1`, so it doubles the channels at each matching scale.',
      'The four pool/upsample pairs line up exactly when `H` and `W` are divisible by 16. Supporting odd dimensions would require explicit padding, cropping, or interpolation before each concatenation.',
      'The final `1x1` convolution maps 64 features to one logit per class and pixel:\n`(B, 64, H, W) -> (B, num_classes, H, W)`\nKeep these as logits for the chosen segmentation loss.',
      'Memory cue: downsample for context, save the skips, then restore the grid. The decoder reuses the saved high-resolution features instead of asking the bottleneck to recreate every boundary.',
    ],
    solutionDiagram: `U-Net memory map
input (B, C, H, W)
  -> DoubleConv 64 -> pool
  -> DoubleConv 128 -> pool
  -> DoubleConv 256 -> pool
  -> DoubleConv 512 -> pool
  -> bottleneck DoubleConv 1024
  -> up + concat x4 -> DoubleConv 512
  -> up + concat x3 -> DoubleConv 256
  -> up + concat x2 -> DoubleConv 128
  -> up + concat x1 -> DoubleConv 64
  -> 1x1 head -> (B, num_classes, H, W)`,
    starterCode: `from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        # TODO: create two 3x3 convolution + ReLU layers.
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: apply both convolutions and activations.
        raise NotImplementedError("Implement forward")


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        # TODO: assemble the encoder, bottleneck, decoder, and head.
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: save encoder skips, then upsample and concatenate them.
        raise NotImplementedError("Implement forward")


def test_unet() -> None:
    model = UNet(in_channels=3, num_classes=4)
    images = torch.randn(2, 3, 128, 128)
    logits = model(images)
    print(logits.shape)
    assert logits.shape == (2, 4, 128, 128)


if __name__ == "__main__":
    test_unet()`,
    solutionCode: `from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x = self.bottleneck(self.pool(x4))

        x = self.dec4(torch.cat([self.up4(x), x4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), x3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), x2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), x1], dim=1))
        return self.head(x)


def test_unet() -> None:
    model = UNet(in_channels=3, num_classes=4)
    images = torch.randn(2, 3, 128, 128)
    logits = model(images)
    print(logits.shape)
    assert logits.shape == (2, 4, 128, 128)


if __name__ == "__main__":
    test_unet()`,
    packages: ['torch'],
    tags: ['PyTorch', 'Architecture', 'U-Net', 'Segmentation', 'Clean Code'],
  },
  {
    id: 'centernet-style-detector',
    order: 41,
    title: 'Implement CenterNet with three prediction heads',
    difficulty: 'Hard',
    track: 'architecture',
    summary:
      'Implement a compact CenterNet-style detector with a stride-four backbone and separate heatmap, size, and offset prediction heads.',
    prompt: [
      'A CenterNet-style detector predicts object centers on a dense feature grid. Implement `PredictionHead` and `CenterNet` for inputs shaped `(B, 3, H, W)`.',
      'The backbone should downsample by four, then three heads should predict heatmap logits, box size, and center offset. Assume `H` and `W` are divisible by four.',
    ],
    signature: `class PredictionHead(nn.Module): ...
class CenterNet(nn.Module): ...`,
    requirements: [
      'Implement `PredictionHead` as a `3x3` convolution, ReLU, and `1x1` convolution.',
      'Build a backbone with two stride-2 convolutions followed by one stride-1 convolution.',
      'Use separate heads for the class heatmap, box size `(w, h)`, and center offset `(dx, dy)`.',
      'Return heatmap logits shaped `(B, num_classes, H/4, W/4)`.',
      'Return size and offset maps shaped `(B, 2, H/4, W/4)`.',
      'Keep sigmoid, decoding, and detection losses outside `forward`.',
    ],
    examples: [
      {
        label: 'Dense prediction shape check',
        lines: [
          'model = CenterNet(num_classes=6)',
          'images.shape = (2, 3, 64, 80)',
        ],
        result: 'heatmap=(2, 6, 16, 20); size=(2, 2, 16, 20); offset=(2, 2, 16, 20)',
      },
    ],
    hint: [
      'Two stride-2 convolutions turn `(H, W)` into `(H/4, W/4)`.',
      'Reuse the same `PredictionHead` structure for all three tasks; only `out_channels` changes.',
      'The heatmap has `num_classes` channels; size and offset each have two.',
      'Return the raw heatmap values so the training loss can decide how to normalize them.',
    ],
    interview: {
      durationMinutes: 45,
      evaluationCriteria: [
        'Defines the dense output shapes before implementing the modules.',
        'Separates shared feature extraction from task-specific prediction heads.',
        'Tracks the stride-four spatial contract and keeps post-processing outside `forward`.',
      ],
      followUps: [
        'How would you decode these maps into image-space boxes?',
        'Why might you use a sigmoid on the heatmap during training or inference?',
        'Where would a feature pyramid or deformable convolution fit?',
      ],
    },
    solutionNotes: [
      'Two stride-2 convolutions create one shared stride-four feature grid:\n`(B, 3, H, W) -> (B, 128, H/4, W/4)`',
      'Each `PredictionHead` keeps the task-specific mapping small:\n`Conv3x3 -> ReLU -> Conv1x1`\nThe final convolution changes only the number of prediction channels.',
      'The three heads read the same features but produce different meanings:\n`heatmap: (B, classes, H/4, W/4)`\n`size: (B, 2, H/4, W/4)`\n`offset: (B, 2, H/4, W/4)`',
      'At each grid cell, the heatmap scores class centers, size predicts box width and height, and offset corrects the sub-cell center location lost when a continuous point is assigned to a discrete grid.',
      'The four-way spatial reduction is exact for dimensions divisible by four. Decoding peaks, applying sigmoid or thresholds, and converting cell coordinates back to image coordinates belong outside `forward`.',
      'Memory cue: one shared stride-four map, then three heads—heatmap finds centers; size and offset describe the box around each center.',
    ],
    solutionDiagram: `CenterNet memory map
image (B, 3, H, W)
  -> stride-4 backbone (B, 128, H/4, W/4)
     |-> heatmap head -> (B, K, H/4, W/4)
     |-> size head    -> (B, 2, H/4, W/4)
     +-> offset head  -> (B, 2, H/4, W/4)`,
    starterCode: `from __future__ import annotations

import torch
from torch import nn


class PredictionHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        # TODO: build a 3x3 -> ReLU -> 1x1 prediction head.
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return the dense prediction map.
        raise NotImplementedError("Implement forward")


class CenterNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        # TODO: assemble the backbone and three prediction heads.
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: return heatmap, size, and offset maps.
        raise NotImplementedError("Implement forward")


def test_centernet() -> None:
    model = CenterNet(num_classes=6)
    images = torch.randn(2, 3, 64, 80)
    heatmap, size, offset = model(images)
    print(heatmap.shape, size.shape, offset.shape)
    assert heatmap.shape == (2, 6, 16, 20)
    assert size.shape == (2, 2, 16, 20)
    assert offset.shape == (2, 2, 16, 20)


if __name__ == "__main__":
    test_centernet()`,
    solutionCode: `from __future__ import annotations

import torch
from torch import nn


class PredictionHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class CenterNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.heatmap_head = PredictionHead(128, num_classes)
        self.size_head = PredictionHead(128, 2)
        self.offset_head = PredictionHead(128, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        return (
            self.heatmap_head(features),
            self.size_head(features),
            self.offset_head(features),
        )


def test_centernet() -> None:
    model = CenterNet(num_classes=6)
    images = torch.randn(2, 3, 64, 80)
    heatmap, size, offset = model(images)
    print(heatmap.shape, size.shape, offset.shape)
    assert heatmap.shape == (2, 6, 16, 20)
    assert size.shape == (2, 2, 16, 20)
    assert offset.shape == (2, 2, 16, 20)


if __name__ == "__main__":
    test_centernet()`,
    packages: ['torch'],
    tags: ['PyTorch', 'Architecture', 'CenterNet', 'Detection', 'Clean Code'],
  },
] as const satisfies readonly CodePracticeProblem[];
