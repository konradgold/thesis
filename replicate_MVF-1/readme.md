## Goal
We aim to replicate the winning contribution to the 2025 MV Foul Recognition Challenge [see here](https://arxiv.org/pdf/2508.19182).

Description: "Habel et al. leverage a single TAdaFormer-L/14
model that was pre-trained on the Kinetics 400 and 710
datasets [18]. 16 frames are sampled with a stride of two,
resulting in a context window of 32 frames in total. For
the aggregation of multiple views, max pooling is used be-
fore the classification head. They incorporate view-specific
context by adding a learnable view embedding to the fea-
ture vectors. This embedding is applied before the max
pooling step to distinguish whether the input is a live view
(clip 0) or a replay view (clip 1–3). The backbone gets fine-
tuned in stage one using the 720p videos with an input size
of 280×490 pixels, selecting random two of the up to four
views per foul. For stage two, they extract for all data ten
times the features of the transformer before the classifica-
tion heads using random augmentations and retrain in a sec-
ond step only the classification heads on the extracted fea-
tures. In stage two the model sees all available views, which
can close the gap between training and inference, leading
to a slight improvement in performance. The single model
trained on all given data (train + valid + test) achieves on the
challenge set a combined metric score of 49.52% for stage
one and 52.22% for stage two."

## Todos:
- [ ] Get TAdaFormer-L/14 running as backbone
- [ ] Sampling of 16 frames per view
- [ ] Add view embedding before backbone (where exactly?)
- [ ] Aggregation per view
- [ ] Classification Head
For training:
- [ ] Implement Augmentations
- [ ] Two stages