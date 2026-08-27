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
    title: 'Build a U-Net encoder-decoder',
    difficulty: 'Hard',
    track: 'architecture',
    summary:
      'Implement a configurable U-Net with reusable blocks, skip connections, and robust odd-size handling.',
    prompt: [
      'A segmentation pipeline must preserve fine spatial detail while using deeper context. Implement `DoubleConv`, `UpBlock`, and `UNet` using the supplied `UNetConfig`.',
      'Treat input-size behavior as part of the API. Your decoder must align each upsampled tensor to its skip tensor before concatenation, including when pooling rounded an odd spatial dimension down.',
    ],
    signature: `@dataclass(frozen=True, slots=True)
class UNetConfig: ...

class DoubleConv(nn.Module): ...
class UpBlock(nn.Module): ...
class UNet(nn.Module): ...`,
    requirements: [
      'Use one reusable double-convolution block throughout the encoder and decoder.',
      'Store encoder and decoder blocks in `nn.ModuleList` containers.',
      'Use max pooling to downsample and transposed convolution to upsample.',
      'Resize an upsampled feature map to the exact skip size before concatenation when necessary.',
      'Return logits shaped `(B, out_channels, H, W)` for even and odd input sizes.',
    ],
    examples: [
      {
        label: 'Odd-size acceptance check',
        lines: [
          'config = UNetConfig(in_channels=3, out_channels=4, channels=(32, 64, 128))',
          'images.shape = (2, 3, 127, 131)',
        ],
        result: 'model(images).shape == (2, 4, 127, 131)',
      },
    ],
    hint: [
      'Save the output of each encoder block before pooling it.',
      'The bottleneck has twice as many channels as the deepest encoder block.',
      'Build the decoder by iterating over encoder channels in reverse.',
      'After upsampling, compare `x.shape[-2:]` with `skip.shape[-2:]` before concatenating on the channel axis.',
    ],
    interview: {
      durationMinutes: 50,
      evaluationCriteria: [
        'Explains what information the skip connections restore.',
        'Keeps channel bookkeeping inside reusable blocks instead of the forward method.',
        'Tests an odd spatial size and states the output-logit contract.',
      ],
      followUps: [
        'When would bilinear upsampling be preferable to transposed convolution?',
        'How would you adapt the output and loss for multi-label segmentation?',
      ],
    },
    solutionNotes: [
      'The encoder builds semantics while reducing resolution. Save each block output before pooling; those saved tensors carry the fine boundaries that would otherwise be lost in the bottleneck.',
      'Each decoder stage repeats one shape operation:\n`upsample decoder → align to skip size → concatenate channels → DoubleConv`\nConcatenation happens on channels, so the first decoder convolution receives `decoder_channels + skip_channels`.',
      'The channel ledger is easiest to derive backward from the encoder widths. If the deepest skip has `C` channels, the bottleneck has `2C`; after upsampling to `C`, concatenating the `C`-channel skip produces `2C` input channels for `DoubleConv`.',
      'Odd sizes expose a common hidden assumption: repeated division by two is not exactly reversible. Resize to the actual skip shape:\n`target_size = skip.shape[-2:]`\nThis avoids brittle crop arithmetic and restores the exact input resolution.',
      'The final `1x1` convolution changes channels without changing spatial size:\n`(B, C, H, W) → (B, classes, H, W)`\nReturn logits; sigmoid or softmax belongs with the chosen loss or inference post-processing.',
    ],
    solutionDiagram: `input -> enc 32 -> pool -> enc 64 -> pool -> enc 128 -> bottleneck 256
            |                    |                     |
            +--------------------+---------------------+
                                 decoder + skips -> logits at input size`,
    starterCode: `from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True, slots=True)
class UNetConfig:
    in_channels: int = 3
    out_channels: int = 1
    channels: tuple[int, ...] = (64, 128, 256, 512)

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("input and output channels must be positive")
        if len(self.channels) < 2 or any(channel <= 0 for channel in self.channels):
            raise ValueError("channels must contain at least two positive widths")


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        # TODO: create two Conv-BatchNorm-ReLU layers.
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: apply the double-convolution block.
        raise NotImplementedError("Implement forward")


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        # TODO: create the upsampler and the skip-fusion block.
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # TODO: upsample, align, concatenate, and fuse.
        raise NotImplementedError("Implement forward")


class UNet(nn.Module):
    def __init__(self, config: UNetConfig) -> None:
        # TODO: assemble encoder blocks, bottleneck, decoder blocks, and output head.
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: save skips, run the bottleneck, and decode in reverse order.
        raise NotImplementedError("Implement forward")


def smoke_test() -> None:
    device = torch.device("cpu")
    config = UNetConfig(out_channels=4, channels=(4, 8))
    model = UNet(config).to(device=device, dtype=torch.float32).eval()
    images = torch.randn(1, 3, 17, 19, device=device, dtype=torch.float32)
    with torch.inference_mode():
        logits = model(images)
    assert logits.shape == (1, 4, 17, 19)
    print(tuple(logits.shape))


if __name__ == "__main__":
    smoke_test()`,
    solutionCode: `from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True, slots=True)
class UNetConfig:
    in_channels: int = 3
    out_channels: int = 1
    channels: tuple[int, ...] = (64, 128, 256, 512)

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("input and output channels must be positive")
        if len(self.channels) < 2 or any(channel <= 0 for channel in self.channels):
            raise ValueError("channels must contain at least two positive widths")


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.fuse = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((skip, x), dim=1))


class UNet(nn.Module):
    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        encoder_channels = (config.in_channels, *config.channels[:-1])
        self.encoder = nn.ModuleList(
            DoubleConv(in_channels, out_channels)
            for in_channels, out_channels in zip(encoder_channels, config.channels)
        )
        self.pool = nn.MaxPool2d(2)
        bottleneck_channels = config.channels[-1] * 2
        self.bottleneck = DoubleConv(config.channels[-1], bottleneck_channels)
        decoder = []
        current_channels = bottleneck_channels
        for skip_channels in reversed(config.channels):
            decoder.append(UpBlock(current_channels, skip_channels, skip_channels))
            current_channels = skip_channels
        self.decoder = nn.ModuleList(decoder)
        self.output_head = nn.Conv2d(config.channels[0], config.out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for block, skip in zip(self.decoder, reversed(skips)):
            x = block(x, skip)
        return self.output_head(x)


def smoke_test() -> None:
    device = torch.device("cpu")
    config = UNetConfig(out_channels=4, channels=(4, 8))
    model = UNet(config).to(device=device, dtype=torch.float32).eval()
    images = torch.randn(1, 3, 17, 19, device=device, dtype=torch.float32)
    with torch.inference_mode():
        logits = model(images)
    assert logits.shape == (1, 4, 17, 19)
    print(tuple(logits.shape))


if __name__ == "__main__":
    smoke_test()`,
    packages: ['torch'],
    tags: ['PyTorch', 'Architecture', 'U-Net', 'Segmentation', 'Clean Code'],
  },
  {
    id: 'centernet-style-detector',
    order: 41,
    title: 'Build a CenterNet-style detector',
    difficulty: 'Hard',
    track: 'architecture',
    summary:
      'Implement a compact keypoint detector with typed outputs and separate heatmap, size, and offset heads.',
    prompt: [
      'Design the prediction side of a CenterNet-style detector. The model should turn an image into a stride-four feature map, then predict class-center heatmap logits, box size, and sub-pixel center offset at every location.',
      'Keep decoding and losses out of scope. In the interview, make the output contract explicit and explain why the heatmap head uses a negative prior bias.',
    ],
    signature: `@dataclass(frozen=True, slots=True)
class CenterNetConfig: ...

@dataclass(slots=True)
class CenterNetOutput: ...

class CenterNetDetector(nn.Module): ...`,
    requirements: [
      'Use a frozen config dataclass and a typed output dataclass.',
      'Downsample the input by exactly four before the prediction heads.',
      'Use separate heads for class heatmap logits, width/height, and center offset.',
      'Initialize the final heatmap bias to `-2.19` so initial foreground probabilities are low.',
      'Assume the input height and width are divisible by four.',
      'Return heatmap logits shaped `(B, K, H/4, W/4)` and two regression maps shaped `(B, 2, H/4, W/4)`.',
      'Do not apply sigmoid or decode boxes inside `forward`.',
    ],
    examples: [
      {
        label: 'Acceptance check',
        lines: [
          'config = CenterNetConfig(num_classes=6)',
          'images.shape = (2, 3, 128, 160)',
        ],
        result: 'heatmap=(2, 6, 32, 40); size=(2, 2, 32, 40); offset=(2, 2, 32, 40)',
      },
    ],
    hint: [
      'Two stride-two convolutional blocks produce the required stride-four feature grid.',
      'A small `PredictionHead` class keeps the three task heads structurally consistent.',
      'Initialize only the last convolution in the heatmap head with the negative bias.',
      'Returning logits keeps the model compatible with a numerically stable focal-style loss.',
    ],
    interview: {
      durationMinutes: 45,
      evaluationCriteria: [
        'Defines the dense output shapes before implementing the modules.',
        'Separates shared feature extraction from task-specific prediction heads.',
        'Explains the heatmap prior and keeps post-processing outside `forward`.',
      ],
      followUps: [
        'How would you decode these maps into image-space boxes?',
        'Where would deformable convolutions or a feature pyramid fit?',
      ],
    },
    solutionNotes: [
      'One stride-four feature map feeds three heads:\n`heatmap logits: (B, classes, H/4, W/4)`\n`box size: (B, 2, H/4, W/4)`\n`center offset: (B, 2, H/4, W/4)`',
      'Each output cell corresponds to a stride-four location in the input image. The heatmap asks which class center occupies that cell; size predicts box width and height; offset repairs the sub-cell error caused by mapping a continuous center onto a discrete grid.',
      'The backbone is shared because all three tasks need the same local feature map. Separate small heads let each task learn its own final representation without duplicating the expensive image encoder.',
      'Initialize the final heatmap bias to `-2.19`, so the initial sigmoid probability is about `0.1`. Starting with a low foreground prior prevents the many background locations from producing confident positives before training.',
      'Return raw heatmap logits for a stable focal-style loss. Thresholding, local-maximum suppression, top-k selection, and decoding back to image coordinates are inference steps, so they stay outside `forward`.',
    ],
    solutionDiagram: `image (B, 3, H, W)
  -> stride-4 backbone (B, C, H/4, W/4)
     |-> heatmap logits (B, K, H/4, W/4)
     |-> box size       (B, 2, H/4, W/4)
     +-> center offset  (B, 2, H/4, W/4)`,
    starterCode: `from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class CenterNetConfig:
    in_channels: int = 3
    num_classes: int = 80
    feature_channels: int = 128
    head_channels: int = 64

    def __post_init__(self) -> None:
        if min(self.in_channels, self.num_classes, self.feature_channels, self.head_channels) <= 0:
            raise ValueError("all channel counts and num_classes must be positive")
        if self.feature_channels < 2:
            raise ValueError("feature_channels must be at least two")


@dataclass(slots=True)
class CenterNetOutput:
    heatmap_logits: torch.Tensor
    size: torch.Tensor
    offset: torch.Tensor


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        # TODO: build one stride-aware feature block.
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: apply the feature block.
        raise NotImplementedError("Implement forward")


class PredictionHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        # TODO: build a small task-specific head.
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: return the dense prediction map.
        raise NotImplementedError("Implement forward")


class CenterNetDetector(nn.Module):
    def __init__(self, config: CenterNetConfig) -> None:
        # TODO: assemble the stride-four backbone and three prediction heads.
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> CenterNetOutput:
        # TODO: return typed heatmap, size, and offset maps.
        raise NotImplementedError("Implement forward")


def smoke_test() -> None:
    device = torch.device("cpu")
    config = CenterNetConfig(num_classes=6, feature_channels=8, head_channels=4)
    model = CenterNetDetector(config).to(device=device, dtype=torch.float32).eval()
    images = torch.randn(1, 3, 16, 20, device=device, dtype=torch.float32)
    with torch.inference_mode():
        output = model(images)
    assert output.heatmap_logits.shape == (1, 6, 4, 5)
    assert output.size.shape == output.offset.shape == (1, 2, 4, 5)
    print(tuple(output.heatmap_logits.shape), tuple(output.size.shape), tuple(output.offset.shape))


if __name__ == "__main__":
    smoke_test()`,
    solutionCode: `from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class CenterNetConfig:
    in_channels: int = 3
    num_classes: int = 80
    feature_channels: int = 128
    head_channels: int = 64

    def __post_init__(self) -> None:
        if min(self.in_channels, self.num_classes, self.feature_channels, self.head_channels) <= 0:
            raise ValueError("all channel counts and num_classes must be positive")
        if self.feature_channels < 2:
            raise ValueError("feature_channels must be at least two")


@dataclass(slots=True)
class CenterNetOutput:
    heatmap_logits: torch.Tensor
    size: torch.Tensor
    offset: torch.Tensor


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PredictionHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CenterNetDetector(nn.Module):
    def __init__(self, config: CenterNetConfig) -> None:
        super().__init__()
        mid_channels = config.feature_channels // 2
        self.backbone = nn.Sequential(
            ConvNormAct(config.in_channels, mid_channels, stride=2),
            ConvNormAct(mid_channels, config.feature_channels, stride=2),
            ConvNormAct(config.feature_channels, config.feature_channels),
        )
        self.heatmap_head = PredictionHead(config.feature_channels, config.head_channels, config.num_classes)
        self.size_head = PredictionHead(config.feature_channels, config.head_channels, 2)
        self.offset_head = PredictionHead(config.feature_channels, config.head_channels, 2)
        nn.init.constant_(self.heatmap_head.layers[-1].bias, -2.19)

    def forward(self, x: torch.Tensor) -> CenterNetOutput:
        features = self.backbone(x)
        return CenterNetOutput(
            heatmap_logits=self.heatmap_head(features),
            size=self.size_head(features),
            offset=self.offset_head(features),
        )


def smoke_test() -> None:
    device = torch.device("cpu")
    config = CenterNetConfig(num_classes=6, feature_channels=8, head_channels=4)
    model = CenterNetDetector(config).to(device=device, dtype=torch.float32).eval()
    images = torch.randn(1, 3, 16, 20, device=device, dtype=torch.float32)
    with torch.inference_mode():
        output = model(images)
    assert output.heatmap_logits.shape == (1, 6, 4, 5)
    assert output.size.shape == output.offset.shape == (1, 2, 4, 5)
    print(tuple(output.heatmap_logits.shape), tuple(output.size.shape), tuple(output.offset.shape))


if __name__ == "__main__":
    smoke_test()`,
    packages: ['torch'],
    tags: ['PyTorch', 'Architecture', 'CenterNet', 'Detection', 'Clean Code'],
  },
] as const satisfies readonly CodePracticeProblem[];
