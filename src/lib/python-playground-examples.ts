import type { PythonPlaygroundProps } from './python-playground';

export const attentionMechanismsPlayground: PythonPlaygroundProps = {
  title: 'Toy Self-Attention Terminal',
  initialCode: `import math

tokens = ["animal", "street", "tired"]
query = [1.0, 0.2]
keys = {
    "animal": [1.0, 0.3],
    "street": [0.2, 0.9],
    "tired": [0.8, 0.4],
}
values = {
    "animal": [0.9, 0.1],
    "street": [0.1, 0.8],
    "tired": [0.7, 0.6],
}

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

scores = {token: dot(query, keys[token]) for token in tokens}
scale = math.sqrt(len(query))
scaled_scores = {token: round(score / scale, 3) for token, score in scores.items()}
exp_scores = {token: math.exp(scaled_scores[token]) for token in tokens}
normalizer = sum(exp_scores.values())
weights = {token: round(exp_scores[token] / normalizer, 3) for token in tokens}
output = [
    round(sum(weights[token] * values[token][dim] for token in tokens), 3)
    for dim in range(2)
]

print("raw scores:", scores)
print("scaled scores:", scaled_scores)
print("attention weights:", weights)
print("context vector:", output)
`,
  samples: [
    {
      label: 'Pronoun Routing',
      description: 'Trace how a toy query focuses most strongly on "animal" and "tired".',
      code: `import math

tokens = ["animal", "street", "tired"]
query = [1.0, 0.2]
keys = {
    "animal": [1.0, 0.3],
    "street": [0.2, 0.9],
    "tired": [0.8, 0.4],
}
values = {
    "animal": [0.9, 0.1],
    "street": [0.1, 0.8],
    "tired": [0.7, 0.6],
}

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

scores = {token: dot(query, keys[token]) for token in tokens}
scale = math.sqrt(len(query))
scaled_scores = {token: round(score / scale, 3) for token, score in scores.items()}
exp_scores = {token: math.exp(scaled_scores[token]) for token in tokens}
normalizer = sum(exp_scores.values())
weights = {token: round(exp_scores[token] / normalizer, 3) for token in tokens}
output = [
    round(sum(weights[token] * values[token][dim] for token in tokens), 3)
    for dim in range(2)
]

print("raw scores:", scores)
print("scaled scores:", scaled_scores)
print("attention weights:", weights)
print("context vector:", output)
`,
    },
    {
      label: 'Sharper Query',
      description: 'Change the query to over-weight the "tired" token and watch the mix shift.',
      code: `import math

tokens = ["animal", "street", "tired"]
query = [0.4, 1.0]
keys = {
    "animal": [1.0, 0.3],
    "street": [0.2, 0.9],
    "tired": [0.8, 0.4],
}
values = {
    "animal": [0.9, 0.1],
    "street": [0.1, 0.8],
    "tired": [0.7, 0.6],
}

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

scores = {token: dot(query, keys[token]) for token in tokens}
scale = math.sqrt(len(query))
scaled_scores = {token: round(score / scale, 3) for token, score in scores.items()}
exp_scores = {token: math.exp(scaled_scores[token]) for token in tokens}
normalizer = sum(exp_scores.values())
weights = {token: round(exp_scores[token] / normalizer, 3) for token in tokens}
output = [
    round(sum(weights[token] * values[token][dim] for token in tokens), 3)
    for dim in range(2)
]

print("raw scores:", scores)
print("scaled scores:", scaled_scores)
print("attention weights:", weights)
print("context vector:", output)
`,
    },
  ],
  walkthroughSteps: [
    {
      label: 'Compute raw compatibility scores',
      lineHint: 18,
      variables: {
        query: '[1.0, 0.2]',
        scores: "{'animal': 1.06, 'street': 0.38, 'tired': 0.88}",
      },
      output: 'The query best matches `animal`, with `tired` close behind.',
    },
    {
      label: 'Scale before softmax',
      lineHint: 20,
      variables: {
        scale: '1.414',
        scaled_scores: "{'animal': 0.75, 'street': 0.269, 'tired': 0.622}",
      },
      output: 'Dividing by sqrt(d_k) keeps the logits from becoming too sharp too early.',
    },
    {
      label: 'Normalize into attention weights',
      lineHint: 23,
      variables: {
        normalizer: '5.691',
        weights: "{'animal': 0.373, 'street': 0.23, 'tired': 0.397}",
      },
      output: 'Softmax turns the compatibility scores into a probability distribution.',
    },
    {
      label: 'Mix the values into one context vector',
      lineHint: 24,
      variables: {
        output: '[0.659, 0.52]',
      },
      output: 'The final vector is a weighted blend of the returned values, not a hard pointer.',
    },
  ],
  notes:
    'Try editing the query or one of the key vectors, then re-run. Watch how the weight distribution shifts before the context vector changes.',
};

export const attentionIsAllYouNeedPlayground: PythonPlaygroundProps = {
  title: 'Scaled Attention And Positional Encoding Terminal',
  initialCode: `import math

query = [1.0, 0.0]
keys = [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]]
values = [[5.0, 0.0], [2.0, 2.0], [0.0, 4.0]]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

raw_scores = [dot(query, key) for key in keys]
scale = math.sqrt(len(query))
scaled_scores = [round(score / scale, 3) for score in raw_scores]
exp_scores = [math.exp(score) for score in scaled_scores]
normalizer = sum(exp_scores)
weights = [round(score / normalizer, 3) for score in exp_scores]
context = [
    round(sum(weights[i] * values[i][dim] for i in range(len(values))), 3)
    for dim in range(len(values[0]))
]

def positional_encoding(position, d_model):
    row = []
    for i in range(d_model):
        exponent = (2 * (i // 2)) / d_model
        angle = position / (10000 ** exponent)
        row.append(round(math.sin(angle), 4) if i % 2 == 0 else round(math.cos(angle), 4))
    return row

print("scaled scores:", scaled_scores)
print("attention weights:", weights)
print("context:", context)
print("PE(pos=3, d_model=4):", positional_encoding(3, 4))
`,
  samples: [
    {
      label: 'Scaled Attention',
      description: 'Inspect how one query spreads probability mass across three keys.',
      code: `import math

query = [1.0, 0.0]
keys = [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]]
values = [[5.0, 0.0], [2.0, 2.0], [0.0, 4.0]]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

raw_scores = [dot(query, key) for key in keys]
scale = math.sqrt(len(query))
scaled_scores = [round(score / scale, 3) for score in raw_scores]
exp_scores = [math.exp(score) for score in scaled_scores]
normalizer = sum(exp_scores)
weights = [round(score / normalizer, 3) for score in exp_scores]
context = [
    round(sum(weights[i] * values[i][dim] for i in range(len(values))), 3)
    for dim in range(len(values[0]))
]

print("scaled scores:", scaled_scores)
print("attention weights:", weights)
print("context:", context)
`,
    },
    {
      label: 'Positional Encoding',
      description: 'Change the position or model width and see the sinusoidal pattern move.',
      code: `import math

def positional_encoding(position, d_model):
    row = []
    for i in range(d_model):
        exponent = (2 * (i // 2)) / d_model
        angle = position / (10000 ** exponent)
        row.append(round(math.sin(angle), 4) if i % 2 == 0 else round(math.cos(angle), 4))
    return row

for position in range(4):
    print(f"position {position}:", positional_encoding(position, 6))
`,
    },
  ],
  walkthroughSteps: [
    {
      label: 'Score each key against the query',
      lineHint: 9,
      variables: {
        raw_scores: '[1.0, 0.6, 0.0]',
        scale: '1.414',
      },
      output: 'The first key is the best match, but the second still receives some probability mass.',
    },
    {
      label: 'Scale and normalize',
      lineHint: 12,
      variables: {
        scaled_scores: '[0.707, 0.424, 0.0]',
        weights: '[0.444, 0.334, 0.221]',
      },
      output: 'Scaling avoids overconfident logits before softmax turns them into weights.',
    },
    {
      label: 'Form the context vector',
      lineHint: 15,
      variables: {
        context: '[2.888, 1.552]',
      },
      output: 'The context vector is a weighted average of the value vectors.',
    },
    {
      label: 'Inject position information',
      lineHint: 19,
      variables: {
        'PE(pos=3, d_model=4)': '[0.1411, -0.99, 0.03, 0.9996]',
      },
      output: 'The sine/cosine pair changes smoothly with position while staying deterministic.',
    },
  ],
  notes:
    'Run the default sample, then change the query or swap in different keys. For positional encoding, try larger positions and see how the low-frequency dimensions move more slowly.',
};

export const bertPlayground: PythonPlaygroundProps = {
  title: 'Tiny BERT Pretraining Terminal',
  initialCode: `tokens = ["[CLS]", "the", "robot", "learns", "fast", "[SEP]"]
mask_positions = [2, 3]
vocab = ["the", "robot", "learns", "fast", "slow", "agent"]

masked_tokens = tokens[:]
labels = {}

for pos in mask_positions:
    labels[pos] = tokens[pos]
    masked_tokens[pos] = "[MASK]"

prediction_scores = {
    2: {"robot": 0.78, "agent": 0.18, "slow": 0.04},
    3: {"learns": 0.72, "fast": 0.17, "slow": 0.11},
}

predictions = {pos: max(scores, key=scores.get) for pos, scores in prediction_scores.items()}
accuracy = sum(predictions[pos] == labels[pos] for pos in mask_positions) / len(mask_positions)

print("masked input:", masked_tokens)
print("labels:", labels)
print("predictions:", predictions)
print("mlm accuracy:", round(accuracy, 2))
`,
  samples: [
    {
      label: 'Masked LM',
      description: 'Follow how masking creates labels only at the hidden positions.',
      code: `tokens = ["[CLS]", "the", "robot", "learns", "fast", "[SEP]"]
mask_positions = [2, 3]
vocab = ["the", "robot", "learns", "fast", "slow", "agent"]

masked_tokens = tokens[:]
labels = {}

for pos in mask_positions:
    labels[pos] = tokens[pos]
    masked_tokens[pos] = "[MASK]"

prediction_scores = {
    2: {"robot": 0.78, "agent": 0.18, "slow": 0.04},
    3: {"learns": 0.72, "fast": 0.17, "slow": 0.11},
}

predictions = {pos: max(scores, key=scores.get) for pos, scores in prediction_scores.items()}
accuracy = sum(predictions[pos] == labels[pos] for pos in mask_positions) / len(mask_positions)

print("masked input:", masked_tokens)
print("labels:", labels)
print("predictions:", predictions)
print("mlm accuracy:", round(accuracy, 2))
`,
    },
    {
      label: 'Next Sentence Pair',
      description: 'See how an NSP-style label is just a binary coherence target.',
      code: `sentence_a = "The robot picked up the box."
sentence_b = "It placed it on the shelf."
sentence_c = "Bananas are usually yellow."

pairs = [
    (sentence_a, sentence_b, 0),
    (sentence_a, sentence_c, 1),
]

for first, second, label in pairs:
    relation = "is next" if label == 0 else "is random"
    print(f"A: {first}")
    print(f"B: {second}")
    print("label:", relation)
    print("-" * 24)
`,
    },
  ],
  walkthroughSteps: [
    {
      label: 'Choose masked positions',
      lineHint: 1,
      variables: {
        tokens: "['[CLS]', 'the', 'robot', 'learns', 'fast', '[SEP]']",
        mask_positions: '[2, 3]',
      },
      output: 'Only a subset of tokens become prediction targets in MLM.',
    },
    {
      label: 'Build the corrupted input',
      lineHint: 7,
      variables: {
        masked_tokens: "['[CLS]', 'the', '[MASK]', '[MASK]', 'fast', '[SEP]']",
        labels: "{2: 'robot', 3: 'learns'}",
      },
      output: 'BERT sees the masked sequence, but the loss is only computed against the hidden originals.',
    },
    {
      label: 'Pick the top prediction for each mask',
      lineHint: 16,
      variables: {
        predictions: "{2: 'robot', 3: 'learns'}",
        accuracy: '1.0',
      },
      output: 'The decoder head predicts the original tokens independently at each masked slot.',
    },
  ],
  notes:
    'Change the mask positions or the prediction scores and re-run. You will see the labels stay tied only to the masked slots, which is the core trick behind BERT pretraining.',
};

export const ppoPlayground: PythonPlaygroundProps = {
  title: 'PPO Clipping Terminal',
  initialCode: `old_logp = -1.20
new_logp = -0.95
advantage = 1.50
clip_eps = 0.20

ratio = round(pow(2.718281828, new_logp - old_logp), 3)
unclipped = round(ratio * advantage, 3)
clipped_ratio = min(max(ratio, 1 - clip_eps), 1 + clip_eps)
clipped = round(clipped_ratio * advantage, 3)
objective = round(min(unclipped, clipped), 3)
loss = round(-objective, 3)

print("ratio:", ratio)
print("unclipped objective:", unclipped)
print("clipped ratio:", clipped_ratio)
print("clipped objective:", clipped)
print("ppo loss:", loss)
`,
  samples: [
    {
      label: 'Positive Advantage',
      description: 'A beneficial action gets capped before the update becomes too aggressive.',
      code: `old_logp = -1.20
new_logp = -0.95
advantage = 1.50
clip_eps = 0.20

ratio = round(pow(2.718281828, new_logp - old_logp), 3)
unclipped = round(ratio * advantage, 3)
clipped_ratio = min(max(ratio, 1 - clip_eps), 1 + clip_eps)
clipped = round(clipped_ratio * advantage, 3)
objective = round(min(unclipped, clipped), 3)
loss = round(-objective, 3)

print("ratio:", ratio)
print("unclipped objective:", unclipped)
print("clipped ratio:", clipped_ratio)
print("clipped objective:", clipped)
print("ppo loss:", loss)
`,
    },
    {
      label: 'Negative Advantage',
      description: 'Flip the sign of the advantage to see how clipping protects bad actions too.',
      code: `old_logp = -1.20
new_logp = -0.95
advantage = -1.50
clip_eps = 0.20

ratio = round(pow(2.718281828, new_logp - old_logp), 3)
unclipped = round(ratio * advantage, 3)
clipped_ratio = min(max(ratio, 1 - clip_eps), 1 + clip_eps)
clipped = round(clipped_ratio * advantage, 3)
objective = round(max(unclipped, clipped), 3)
loss = round(-objective, 3)

print("ratio:", ratio)
print("unclipped objective:", unclipped)
print("clipped ratio:", clipped_ratio)
print("clipped objective:", clipped)
print("ppo loss:", loss)
`,
    },
  ],
  walkthroughSteps: [
    {
      label: 'Form the likelihood ratio',
      lineHint: 5,
      variables: {
        old_logp: '-1.20',
        new_logp: '-0.95',
        ratio: '1.284',
      },
      output: 'The new policy is assigning more probability to this action than the old policy did.',
    },
    {
      label: 'Compute the unclipped objective',
      lineHint: 6,
      variables: {
        advantage: '1.50',
        unclipped: '1.926',
      },
      output: 'Without clipping, PPO would take the full policy-improvement signal.',
    },
    {
      label: 'Clamp the ratio into the trust region',
      lineHint: 7,
      variables: {
        clip_eps: '0.20',
        clipped_ratio: '1.2',
        clipped: '1.8',
      },
      output: 'Clipping prevents the update from moving too far away from the data-collecting policy.',
    },
    {
      label: 'Take the conservative objective',
      lineHint: 9,
      variables: {
        objective: '1.8',
        loss: '-1.8',
      },
      output: 'PPO uses the smaller improvement so the optimizer stays conservative.',
    },
  ],
  notes:
    'Change `new_logp`, `advantage`, or `clip_eps` and watch when the unclipped update starts getting cut back by the trust-region style clamp.',
};
