Application Domain: Cybersecurity in IoT and Enterprise Networks

### A concise summary of the research proposal (maximum 1000 words).
The proliferation of Internet of Things (IoT) devices and the expanding attack surfaces of enterprise networks have made them prime targets for sophisticated cyber threats, particularly zero-day worms. These worms propagate autonomously, exploiting unknown vulnerabilities before patches can be deployed. Traditional signature-based detection systems fall short against such novel threats, necessitating advanced behavioral analysis. While deep learning models offer high detection rates, their "black-box" nature hinders their adoption in critical enterprise environments where trust and interpretability are paramount. 

This research proposes an Explainable Liquid Neural Network (LNN) framework tailored for the real-time detection of zero-day worms across IoT and enterprise architectures. LNNs are a class of continuous-time recurrent neural networks that excel at processing sequential data, offering dynamic adaptability to changing network traffic patterns while maintaining a compact parameter footprint suitable for resource-constrained IoT environments. The proposed framework will be rigorously evaluated against the latest "IoT and Ent Worm 2025 dataset," utilizing realistic simulation data from various domains (e.g., air quality monitors, building monitors, and IP cameras). 

The LNN model's performance will be benchmarked against established baselines, including Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM) networks, Gated Recurrent Units (GRU), Autoregressive models, Traffic Transformers, Generative Adversarial Networks (GAN), and Random Forest (RF) algorithms. Performance will be quantified using Precision, Recall, F1-Score, and ROC-AUC metrics. 

To bridge the gap between high accuracy and interpretability, this study will integrate state-of-the-art eXplainable AI (XAI) models, specifically TShap (Time SHAP) and LIME (Local Interpretable Model-agnostic Explanations). These tools will provide human-readable explanations for the LNN's predictions, ensuring that security analysts can comprehend, trust, and swiftly act upon the system's alerts. By synthesizing continuous learning with transparent decision-making, this research aims to deliver a robust, deployable solution for next-generation network security.

### B) A detailed research proposal (about 3000 words).

## 1. Title
Explainable Liquid Neural Network (LNN) Framework for Real-Time Zero-Day Worm Detection in IoT and Enterprise Networks

## 2. Introduction and / or Statement of the Problem
The integration of Internet of Things (IoT) devices within enterprise networks has exponentially increased operational efficiency but concurrently expanded the cyber-attack surface. Zero-day worms represent one of the most severe threats in this landscape, capable of autonomous propagation and widespread disruption before conventional defenses recognize the intrusion. The dynamic, high-volume nature of network traffic further complicates real-time detection. 

**Origin of the research problem**
Current intrusion detection systems (IDS) heavily rely on signature-based methods or conventional machine learning algorithms. While the former cannot detect unknown anomalies, the latter often struggles with the temporal dynamics of network traffic or demands prohibitive computational resources. Furthermore, advanced Deep Learning (DL) models are largely opaque, creating a barrier for security analysts who require understandable alerts to formulate effective incident responses. 

**Interdisciplinary relevance**
This study bridges cybersecurity, artificial intelligence, and network engineering. By integrating Liquid Neural Networks—a concept rooted in biological nervous systems—with network traffic analysis and human-centric explainability, the research addresses both theoretical AI challenges and practical network security needs.

**Review of Research and Development in the Subject:**
- **International and National Status:** Globally, securing IoT infrastructure is a top priority for cybersecurity agencies and research institutions. Recent advances have seen a shift towards anomaly-based detection using models like LSTM and Transformers [1, 2]. However, Liquid Neural Networks (LNNs), which provide continuous-time adaptivity, are only beginning to be explored in network security contexts [3]. 
- **Significance of the study:** The study is significant because it simultaneously addresses three critical requirements for modern IDS: real-time processing of sequential data, detection of unseen (zero-day) threats, and model interpretability via XAI.

## 3. Aims and/or Objectives of the Study
The general aim of this study is to develop, evaluate, and interpret a Liquid Neural Network (LNN) based framework for the real-time detection of zero-day worms in IoT and enterprise networks.

Specific objectives include:
1. To design and train an LNN model capable of processing continuous-time network traffic data to identify anomalous patterns indicative of zero-day worms.
2. To comprehensively benchmark the LNN's performance against standard machine learning and deep learning models (RNN, LSTM, GRU, Autoregressive, Traffic Transformer, GAN, and RF) using the IoT and Ent Worm 2025 dataset.
3. To evaluate the models utilizing standardized metrics: Precision, Recall, and F1-Score.
4. To integrate and evaluate eXplainable AI (XAI) techniques (LIME and TShap) to provide transparent, human-understandable interpretations of the LNN’s real-time detection decisions.

## 4. Conceptual Framework
The proposed conceptual framework operates at the intersection of time-series anomaly detection and interpretable AI. Given the problem of zero-day worm detection, the theoretical perspective posits that network traffic generated by an infiltrating worm disrupts the normal temporal continuity of network flows. 

**Concepts to be used:**
- **Liquid Neural Networks (LNNs):** For modeling the continuous and dynamic nature of network traffic. Their flexible state representations make them robust against time-varying data [3].
- **Explainable AI (LIME and TShap):** To perturb inputs and analyze the local behavior of the LNN, yielding importance scores for network traffic features (e.g., packet sizes, inter-arrival times).

**Innovations (Comparing existing work):**
Unlike existing work that predominantly employs static discrete-time models (like standard RNNs or Transformers) [1, 4], this study leverages continuous-time LNNs which are computationally efficient and highly adaptive. Furthermore, embedding LIME and TShap directly into the real-time inference pipeline transforms a traditional "black-box" detector into a transparent decision-support system.

## 5. Research Question or Hypotheses
**Research Questions:**
1. How does the adaptability of Liquid Neural Networks compare to traditional sequence models (RNN, LSTM, GRU, Transformers) in detecting zero-day worms within the IoT and Ent Worm 2025 dataset?
2. How effectively can LIME and TShap elucidate the decision boundaries of an LNN operating on continuous-time network traffic?

**Hypotheses:**
1. **H1:** The LNN framework will achieve a higher F1-score and Recall in zero-day worm detection compared to standard sequence models (RNN, LSTM, GRU, Autoregressive, Traffic Transformer, GAN, and RF) due to its continuous-time flexibility.
2. **H2:** The integration of TShap and LIME will provide statistically consistent feature importance rankings, allowing security analysts to correctly identify the root cause of the flagged anomaly with high precision.

## 6. Review of Literature with references citation numbers.
The landscape of intrusion detection has evolved rapidly. Signature-based IDS have proven inadequate against zero-day attacks [1]. Consequently, machine learning approaches, specifically Random Forests (RF), were adopted as robust baselines [2]. However, RFs ignore the temporal correlations inherent in network traffic.

To address temporal dependencies, Recurrent Neural Networks (RNNs) and their variants (LSTM, GRU) have been widely deployed for sequence-based anomaly detection [4]. While these models show high Precision and Recall, they often suffer from performance degradation when facing out-of-distribution (zero-day) samples. Recently, Traffic Transformers and GANs have been proposed to model complex network behaviors and generate synthetic anomalies for better training [5, 6]. 

Liquid Neural Networks (LNNs), introduced by Hasani et al. [3], represent a paradigm shift by utilizing ordinary differential equations (ODEs) to continuously adapt to incoming data streams. While highly promising for robotics, their application in cybersecurity is nascent. Moreover, a major inadequacy in current DL-based IDS is their lack of interpretability. The application of Local Interpretable Model-agnostic Explanations (LIME) and SHapley Additive exPlanations (SHAP), specifically Time SHAP (TShap), has become crucial for operationalizing AI in security [7, 8]. This proposal bridges the gap by synthesizing LNNs with TShap and LIME for the IoT and Ent Worm 2025 dataset.

## 7. Scope and Methodology with block diagram
**Scope and Coverage:** 
The study focuses on network-layer detection of zero-day worms using the comprehensive "IoT and Ent Worm 2025" dataset (including processed subsets like iotsim-air-quality, building-monitor, and IP camera datasets). 

**Approach and Methodology:**
1. **Data Preprocessing:** Network traffic captures will be scaled using Standard Scalers and segmented into continuous time-series windows. A strict anti-leakage pipeline will ensure rigorous train-test separation.
2. **Model Implementation:** 
   - Baseline Models: RF, RNN, LSTM, GRU, Autoregressive, GAN, and Traffic Transformer.
   - Proposed Model: LNN utilizing ODE-based continuous time-stepping. The core continuous-time ODE integration is implemented as follows:
     ```python
     class LNN(nn.Module):
         def __init__(self, input_dim: int, hidden_dim: int, num_steps: int = 6, dt: float = 0.1):
             super(LNN, self).__init__()
             # Network definitions omitted for brevity
             self.num_steps = num_steps
             self.dt = dt
             
         def forward(self, x: torch.Tensor) -> torch.Tensor:
             batch_size = x.size(0)
             u = self.input_layer(x)
             h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
             # ODE integration via explicit time-stepping
             for _ in range(self.num_steps):
                 a_val = self.A_net(h)
                 b_val = self.B_net(h)
                 dh = -a_val * h + b_val * u
                 h = h + self.dt * dh
             return self.output_layer(h)
     ```
3. **Training and Evaluation:** All models will be trained on benign and known attack data, and tested on unseen zero-day worm signatures. Performance will be captured via Precision, Recall, F1-Score, and ROC-AUC.
4. **Explainability Pipeline:** Post-hoc analysis using LIME (for feature-level perturbations) and TShap (for temporal sequence attribution) to interpret the LNN's predictions.

> **[Figure 1 Suggestion]** *Title: Overview of the proposed research framework*
> Content: A block diagram showing the flow from the IoT and Ent Worm 2025 dataset, through the Data Preprocessing module, into parallel training paths for the LNN and baseline models (RNN, LSTM, GRU, GAN, Transformer, RF). The output of the LNN feeds into the XAI module (LIME/TShap), resulting in the final Explainable Detection Output.
> Recommended style: Clean vector graphics with a unified color scheme.

## 8. Relevance, Anticipated Outcomes and Proposed Outputs from the Research
**Relevance:**
This research directly addresses the critical vulnerability of IoT and enterprise networks to rapidly propagating zero-day worms. As networks grow in complexity, the need for automated, real-time, and transparent detection systems is paramount.

**Anticipated Outcomes:**
- An empirically validated LNN architecture optimized for real-time network traffic analysis.
- A comprehensive benchmark of LNN against 7 distinct baseline models on the novel IoT and Ent Worm 2025 dataset.
- A functional explainability module utilizing TShap and LIME, tailored for continuous-time sequence models.

**Contributions:**
- **Theoretical:** Advancing the application of Liquid Neural Networks into the cybersecurity domain.
- **Methodological:** Providing a framework for evaluating XAI outputs on continuous-time network sequence data.
- **Practical:** Delivering a transparent IDS solution that security Operations Centers (SOCs) can trust.

## 9. Result and Conclusion
This proposal outlines the development of a state-of-the-art anomaly detection system. By addressing both the need for high-accuracy continuous-time analysis (via LNNs) and the imperative for interpretability (via LIME and TShap), the research will conclude with a deployable, resilient framework capable of defending next-generation IoT and enterprise networks against zero-day worm threats.

## 10. Time Frame (One Year)
- **Months 1-2:** Literature review, data acquisition (IoT and Ent Worm 2025 dataset), and preprocessing pipeline setup.
- **Months 3-4:** Implementation and hyperparameter tuning of baseline models (RF, RNN, LSTM, GRU, Autoregressive, GAN, Transformer).
- **Months 5-6:** Design, implementation, and training of the Liquid Neural Network (LNN) framework.
- **Months 7-8:** Comparative performance evaluation (Precision, Recall, F1, ROC-AUC) across all models.
- **Months 9-10:** Integration and testing of TShap and LIME for the LNN model.
- **Months 11-12:** Final data analysis, synthesis of results, and writing of the final research manuscript/thesis.

## 11. Bibliography
[1] A. Khraisat, I. Gondal, P. Vamplew, and J. Kamruzzaman, "Survey of intrusion detection systems: techniques, datasets and challenges," *Cybersecurity*, vol. 2, no. 1, p. 20, 2019.
[2] N. Moustafa and J. Slay, "The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 dataset and the comparison with the KDD99 dataset," *Information Security Journal: A Global Perspective*, vol. 25, no. 1-3, pp. 18-31, 2016.
[3] R. Hasani, M. Lechner, A. Amini, D. Rus, and R. Grosu, "Liquid time-constant networks," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 35, no. 9, pp. 7657-7666, 2021.
[4] R. Vinayakumar, M. Alazab, K. P. Soman, P. Poornachandran, A. Al-Nemrat, and S. Venkatraman, "Deep learning approach for intelligent intrusion detection system," *IEEE Access*, vol. 7, pp. 41525-41550, 2019.
[5] M. Ring, S. Wunderlich, D. Scheuring, D. Landes, and A. Hotho, "A survey of network-based intrusion detection data sets," *Computers & Security*, vol. 86, pp. 147-167, 2019.
[6] J. Gao, H. Chai, Y. Qiu, et al., "Generative adversarial networks for intrusion detection: A review," *IEEE Access*, vol. 8, pp. 122241-122256, 2020.
[7] M. T. Ribeiro, S. Singh, and C. Guestrin, ""Why should I trust you?" Explaining the predictions of any classifier," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 1135-1144.
[8] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.
