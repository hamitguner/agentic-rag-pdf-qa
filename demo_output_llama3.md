# Evaluation / Demo Output

Document collection: `llama3_herd`


## Q1: What is the parameter count of the flagship Llama 3 model, and on roughly how many tokens was it pre-trained?

_Probes:_ Fact Extraction - locating headline numerical facts (parameter count, pre-training token budget) stated early in the Introduction / General Overview.

**Expected:** The flagship model has 405B parameters and was pre-trained on a corpus of approximately 15.6 trillion (15T+) multilingual tokens.

**Answer:** The flagship Llama 3 model has **405B parameters** and was pre-trained on **15.6T text tokens** [llama3_eval_p1_c1].

**Grounded:** True · **Confidence:** 100% · **Citations:** llama3_eval_p1_c1


## Q2: What attention mechanism does Llama 3 adopt to improve inference scalability, and how many key-value heads does it use?

_Probes:_ Architectural Detail Extraction - identifying a specific design choice (GQA, 8 KV heads) from the 'Pre-Training > Model Architecture' section.

**Expected:** It uses Grouped Query Attention (GQA) with 8 key-value heads, which speeds up inference and reduces the size of the key-value cache during decoding.

**Answer:** Llama 3 adopts **Grouped Query Attention (GQA)** to improve inference scalability. The model uses **8 key-value heads**.

According to the document, this modification was made to "improve inference speed and to reduce the size of key-value caches during decoding." This design choice applies consistently across all three model sizes (8B, 70B, and 405B), as shown in the model architecture specifications.

**Grounded:** True · **Confidence:** 100% · **Citations:** llama3_eval_p6_c14, llama3_eval_p7_c15


## Q3: Which preference-optimization algorithm does Llama 3 use for alignment during post-training, and does it rely on PPO?

_Probes:_ Section-Based Retrieval & Contrast - finding the alignment method in 'Post-Training > Modeling' and correctly contrasting DPO with PPO.

**Expected:** Post-training aligns the model with supervised finetuning (SFT) followed by Direct Preference Optimization (DPO). The authors deliberately use DPO rather than on-policy RL algorithms such as PPO, finding it more stable and cheaper to run at this scale.

**Answer:** Based on the document, Llama 3 uses **Direct Preference Optimization (DPO)** for alignment during post-training. [llama3_eval_p16_c37]

Regarding PPO: No, Llama 3 does not rely on PPO. The document explicitly states that the team "explored on-policy algorithms such as PPO (Schulman et al., 2017), but found that DPO required less compute for large-scale models and performed better, especially on instruction following benchmarks like IFEval." [llama3_eval_p16_c37]

The post-training strategy involves several rounds where each round includes supervised finetuning (SFT) followed by Direct Preference Optimization (DPO). [llama3_eval_p15_c34]

**Grounded:** True · **Confidence:** 100% · **Citations:** llama3_eval_p16_c37, llama3_eval_p15_c34


## Q4: What GPU hardware, and approximately how many of them, were used to train the Llama 3 405B model?

_Probes:_ Fact Extraction (mid-document) - retrieving infrastructure details from the 'Infrastructure, Scaling, and Efficiency' section partway through a long document.

**Expected:** It was trained on NVIDIA H100 GPUs, using up to 16,000 (16K) H100s on Meta's production clusters.

**Answer:** Based on the document, Llama 3 405B was trained on **up to 16K H100 GPUs** [llama3_eval_p9_c19]. Each GPU had 80GB HBM3 memory and ran at 700W TDP, and they were deployed on Meta's Grand Teton AI server platform with eight GPUs per server connected via NVLink [llama3_eval_p9_c19].

**Grounded:** True · **Confidence:** 100% · **Citations:** llama3_eval_p9_c19


## Q5: What is the approximate final pre-training data mix of Llama 3 across the main content categories?

_Probes:_ Data / Proportion Reading - extracting the percentage breakdown reported in 'Pre-Training Data > Determining the Data Mix'.

**Expected:** Roughly 50% general-knowledge tokens, 25% mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens.

**Answer:** The approximate final pre-training data mix of Llama 3 across the main content categories is:

- **50% general knowledge tokens**
- **25% mathematical and reasoning tokens**
- **17% code tokens**
- **8% multilingual tokens**

**Grounded:** True · **Confidence:** 100% · **Citations:** llama3_eval_p6_c13


## Q6: What maximum context length does Llama 3 support, and how is the model trained to reach it?

_Probes:_ Contextual Inference - combining the supported length with the staged long-context procedure described in 'Training Recipe > Long Context Pre-Training'.

**Expected:** Llama 3 supports a context window of up to 128K tokens. It reaches this through a dedicated long-context pre-training stage that gradually increases the context length in several steps (from the initial 8K up to 128K), applied only after standard short-context pre-training.

**Answer:** Based on the document, Llama 3 supports a maximum context length of **128K tokens** [llama3_eval_p14_c33].

The model is trained to reach this context length through a staged approach during the final pre-training phase:

1. **Gradual context length increase**: Rather than training on long sequences from the start, Llama 3 405B increases the context length gradually in six stages, starting from the original 8K context window and ending in the final 128K context window [llama3_eval_p14_c33].

2. **Why gradual training**: The document explains that long sequences are not trained on earlier in pre-training because "the compute in self-attention layers grows quadratically in the sequence length" [llama3_eval_p14_c32].

3. **Adaptation criteria**: The model increases context length in increments and pre-trains until successful adaptation is confirmed by measuring: (1) whether model performance on short-context evaluations has recovered completely, and (2) whether the model perfectly solves "needle in a haystack" tasks up to that length [llama3_eval_p14_c32].

4. **Training scale**: This long-context pre-training stage was performed using approximately 800B training tokens [llama3_eval_p14_c33].

**Grounded:** True · **Confidence:** 100% · **Citations:** llama3_eval_p14_c33, llama3_eval_p14_c32


## Q7: How are image (vision) capabilities added to Llama 3 architecturally - is it trained as a single multimodal model from scratch?

_Probes:_ Deep-Document Navigation & Nuanced Detail - locating the vision-integration approach in the late 'Vision Experiments' section and correctly characterizing the compositional / cross-attention design.

**Expected:** No. Llama 3 uses a compositional approach: a separately pre-trained image encoder is connected to the language model through cross-attention adapter layers that feed visual representations into the LLM, rather than training one multimodal model jointly from scratch.

**Answer:** Vision capabilities are **not** trained as a single multimodal model from scratch. Instead, Llama 3 uses a **compositional approach** with separate training stages [llama3_eval_p3_c8].

The architectural approach consists of five stages [llama3_eval_p55_c132]:

1. **Language model pre-training** - The base Llama 3 language model is trained first
2. **Multi-modal encoder pre-training** - Separate encoders for images and speech are trained independently
3. **Vision adapter training** - Adapters are trained to connect the vision components to the language model
4. **Model finetuning** - The combined model is fine-tuned
5. **Speech adapter training** - Speech adapters are added

For the image component specifically, the architecture consists of three main components [llama3_eval_p56_c135]:

1. **Image encoder** - A standard Vision Transformer (ViT-H/14) with 630M parameters trained on 2.5B image-text pairs for five epochs [llama3_eval_p57_c137]. The encoder is trained to align images and text using a contrastive learning objective.

2. **Image adapter** - Cross-attention layers that connect visual token representations from the image encoder to the language model's token representations [llama3_eval_p57_c137]. For Llama 3 405B, these cross-attention layers add approximately 100B parameters.

3. **Video adapter** - A separate component for video understanding.

The image encoder and adapter are trained on image-text pairs [llama3_eval_p54_c131], with the image encoder parameters remaining unfrozen during subsequent training stages to improve performance on tasks like text recognition [llama3_eval_p57_c137].

**Grounded:** True · **Confidence:** 95% · **Citations:** llama3_eval_p3_c8, llama3_eval_p56_c135, llama3_eval_p57_c137, llama3_eval_p54_c131


## Q8: What quantization technique does Llama 3 apply to the 405B model for more efficient inference, and what numerical format does it use?

_Probes:_ Fact Extraction (late section) - retrieving the FP8 quantization detail from the 'Inference > FP8 Quantization' section near the end of the document.

**Expected:** It applies FP8 quantization to most matrix multiplications in the model (e.g. the feed-forward layers, while leaving self-attention and the first and last Transformer layers unquantized), using the FP8 8-bit floating-point format with row-wise scaling factors.

**Answer:** Based on the document, Llama 3 applies **FP8 quantization** to the 405B model for more efficient inference, using the **FP8 numerical format** [llama3_eval_p51_c126].

Specifically, the document states: "We investigate two main techniques to make inference with the Llama 3 405B model efficient: (1) pipeline parallelism and (2) FP8 quantization" [llama3_eval_p51_c126].

The implementation details include:

- **Scope**: FP8 quantization is applied to most matrix multiplications inside the model, particularly in the feedforward network layers which account for roughly 50% of the inference compute time. Parameters in the self-attention layers are not quantized [llama3_eval_p52_c128].

- **Granularity**: The technique uses **row-wise quantization**, computing scaling factors across rows for parameter and activation matrices, which works better than tensor-wise quantization [llama3_eval_p53_c129].

- **Dynamic scaling**: The approach leverages dynamic scaling factors for better accuracy, with an upper bound of 1200 on the dynamic scaling factors [llama3_eval_p53_c129].

- **Performance**: FP8 inference leads to throughput improvements of up to 50% during the pre-fill stage and substantially better throughput-latency trade-off during decoding [llama3_eval_p53_c129].

**Grounded:** True · **Confidence:** 95% · **Citations:** llama3_eval_p51_c126, llama3_eval_p52_c128, llama3_eval_p53_c129
