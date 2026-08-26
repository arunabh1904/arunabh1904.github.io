import type { CodePracticeProblem } from './code-practice';

export const ARCHITECTURE_CODE_PRACTICE_PROBLEMS = [
  {
    id: 'resnet-from-building-blocks',
    order: 38,
    title: 'Build a configurable ResNet',
    difficulty: 'Hard',
    track: 'architecture',
    summary:
      'Implement a typed residual block and assemble a configurable ResNet with explicit downsampling rules.',
    prompt: [
      'You are given a classification service that needs a small ResNet family rather than one fixed network. Implement `BasicBlock` and `ResNet` around the supplied `ResNetConfig`.',
      'In the interview, first state when the identity path must be projected. Then build the stages, preserve the batch dimension through global pooling, and finish with a shape smoke test.',
    ],
    signature: `@dataclass(frozen=True, slots=True)
class ResNetConfig: ...

class BasicBlock(nn.Module): ...

class ResNet(nn.Module): ...`,
    requirements: [
      'Use `nn.Module` subclasses for the residual block and network.',
      'Use the supplied frozen dataclass as the architecture contract.',
      'Project the skip path when stride changes or channel counts differ.',
      'Build one stage per `(channel, block_count)` pair in the config.',
      'Return logits shaped `(B, num_classes)` for any valid image height and width.',
      'Keep device and dtype behavior inherited from the input and module parameters.',
    ],
    examples: [
      {
        label: 'Acceptance check',
        lines: [
          'config = ResNetConfig(num_classes=10, blocks_per_stage=(2, 2, 2, 2))',
          'images.shape = (2, 3, 224, 224)',
        ],
        result: 'model(images).shape == (2, 10)',
      },
    ],
    hint: [
      'A residual addition is valid only when the main and skip paths have the same shape.',
      'The first block in each later stage performs spatial downsampling; the remaining blocks use stride one.',
      'Track `self.in_channels` as `_make_stage` appends blocks.',
      'Use adaptive average pooling so the classifier does not depend on a fixed image size.',
    ],
    interview: {
      durationMinutes: 50,
      evaluationCriteria: [
        'Explains the identity-versus-projection decision before coding.',
        'Separates configuration, reusable blocks, stage assembly, and the forward path.',
        'Checks tensor shapes and names one production concern such as initialization or normalization.',
      ],
      followUps: [
        'How would you generalize this to bottleneck blocks without rewriting `ResNet`?',
        'What changes for small images such as CIFAR-10?',
      ],
    },
    solutionNotes: [
      'A residual block learns a correction to the input, then adds the original signal back:\n`output = ReLU(residual_path(x) + skip_path(x))`\nIf both paths already have the same shape, the skip is the identity and adds no parameters.',
      'Addition requires identical shapes. The first block of a new stage usually halves height and width while increasing channels, so its skip path needs a stride-matched `1x1` convolution. Later blocks keep the shape and return to the identity skip.',
      'Track the spatial path before coding:\n`stem: H × W → H/4 × W/4`\n`later stages: /8 → /16 → /32`\nOnly the first block in each later stage downsamples; accidentally applying stride to every block collapses the feature map.',
      '`_make_stage` owns the repetitive construction and updates `self.in_channels` after each block. The config supplies widths and block counts, so changing from `(2, 2, 2, 2)` to another depth does not require rewriting `forward`.',
      'Adaptive average pooling removes the remaining spatial axes regardless of input image size:\n`(B, C, Hf, Wf) → (B, C, 1, 1) → (B, C) → logits (B, K)`',
      'The smoke test checks an odd image size as well as the final class shape. In an interview, that catches fixed-size pooling, incorrect stage strides, and projection shortcuts whose output cannot be added to the residual path.',
    ],
    solutionDiagram: `input
  -> stem / 4
  -> stage 1 / 4
  -> stage 2 / 8
  -> stage 3 / 16
  -> stage 4 / 32
  -> adaptive pool -> linear -> logits`,
    starterCode: `from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ResNetConfig:
    in_channels: int = 3
    num_classes: int = 1000
    stage_channels: tuple[int, ...] = (64, 128, 256, 512)
    blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2)

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.num_classes <= 0:
            raise ValueError("in_channels and num_classes must be positive")
        if len(self.stage_channels) != len(self.blocks_per_stage) or not self.stage_channels:
            raise ValueError("stage_channels and blocks_per_stage must have equal non-zero length")
        if any(value <= 0 for value in (*self.stage_channels, *self.blocks_per_stage)):
            raise ValueError("stage widths and depths must be positive")


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        # TODO: build the residual branch and the conditional projection.
        raise NotImplementedError("Implement __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: combine the residual and skip paths, then apply the final activation.
        raise NotImplementedError("Implement forward")


class ResNet(nn.Module):
    def __init__(self, config: ResNetConfig) -> None:
        # TODO: create the stem, stages, adaptive pool, and classifier.
        raise NotImplementedError("Implement __init__")

    def _make_stage(self, out_channels: int, block_count: int, stride: int) -> nn.Sequential:
        # TODO: downsample once, then append stride-one residual blocks.
        raise NotImplementedError("Implement _make_stage")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: run the classification path and return (B, num_classes) logits.
        raise NotImplementedError("Implement forward")


def smoke_test() -> None:
    device = torch.device("cpu")
    config = ResNetConfig(
        num_classes=10,
        stage_channels=(4, 8),
        blocks_per_stage=(1, 1),
    )
    model = ResNet(config).to(device=device, dtype=torch.float32).eval()
    images = torch.randn(1, 3, 17, 19, device=device, dtype=torch.float32)
    with torch.inference_mode():
        logits = model(images)
    assert logits.shape == (1, 10)
    print(tuple(logits.shape))


if __name__ == "__main__":
    smoke_test()`,
    solutionCode: `from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ResNetConfig:
    in_channels: int = 3
    num_classes: int = 1000
    stage_channels: tuple[int, ...] = (64, 128, 256, 512)
    blocks_per_stage: tuple[int, ...] = (2, 2, 2, 2)

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.num_classes <= 0:
            raise ValueError("in_channels and num_classes must be positive")
        if len(self.stage_channels) != len(self.blocks_per_stage) or not self.stage_channels:
            raise ValueError("stage_channels and blocks_per_stage must have equal non-zero length")
        if any(value <= 0 for value in (*self.stage_channels, *self.blocks_per_stage)):
            raise ValueError("stage widths and depths must be positive")


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.projection = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.residual(x) + self.projection(x))


class ResNet(nn.Module):
    def __init__(self, config: ResNetConfig) -> None:
        super().__init__()
        self.in_channels = config.stage_channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, self.in_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        stages = []
        for index, (channels, blocks) in enumerate(zip(config.stage_channels, config.blocks_per_stage)):
            stages.append(self._make_stage(channels, blocks, stride=1 if index == 0 else 2))
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(config.stage_channels[-1], config.num_classes)

    def _make_stage(self, out_channels: int, block_count: int, stride: int) -> nn.Sequential:
        blocks = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        blocks.extend(BasicBlock(out_channels, out_channels) for _ in range(block_count - 1))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stages(self.stem(x))
        return self.classifier(torch.flatten(self.pool(features), 1))


def smoke_test() -> None:
    device = torch.device("cpu")
    config = ResNetConfig(
        num_classes=10,
        stage_channels=(4, 8),
        blocks_per_stage=(1, 1),
    )
    model = ResNet(config).to(device=device, dtype=torch.float32).eval()
    images = torch.randn(1, 3, 17, 19, device=device, dtype=torch.float32)
    with torch.inference_mode():
        logits = model(images)
    assert logits.shape == (1, 10)
    print(tuple(logits.shape))


if __name__ == "__main__":
    smoke_test()`,
    packages: ['torch'],
    tags: ['PyTorch', 'Architecture', 'ResNet', 'Clean Code'],
  },
  {
    id: 'unet-encoder-decoder',
    order: 39,
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
    order: 40,
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
