# Edge Case Handling & Future Roadmap

Transitioning this pipeline from a localized prototype to a production-ready, offline-first application introduces significant system architecture constraints. Below is our roadmap for handling these edge cases, mitigating hardware limitations, and evolving the Decision Engine.

## 1. Feature Extraction & Stateful Persistence

**The Challenge:** The current model relies on highly specific inputs like `face_emotion_hint` and `reflection_quality`. Relying on manual user input for these features introduces severe UX friction and subjective bias. Furthermore, features like `previous_day_mood` require the system to remember past states.

**The Mitigation:**:

- **Upstream ML Pipelines:** We must integrate lightweight, auxiliary models prior to the main XGBoost inference. This includes a small computer vision model (e.g., MobileNet) to passively extract `face_emotion_hint` via the front-facing camera, and a localized NLP heuristic to grade `reflection_quality`.

- **State Management:** To support `previous_day_mood`, the application must transition from a stateless API to a stateful architecture. We will implement a local, encrypted database (e.g., SQLite or Realm) to securely persist longitudinal user data on-device.

## 2. Edge Compute Constraints (Offline-First Limitations)

**The Challenge:** The strict requirement for offline functionality restricts our compute budget. We cannot deploy massive, resource-heavy neural networks due to mobile VRAM limitations, thermal throttling, and battery drain.

**The Mitigation:**

- **Model Distillation & Data Collection:** While our current XGBoost model is highly performant and edge-friendly, its accuracy is constrained by the initial dataset size. To improve performance without increasing model size, we need to establish an opt-in telemetry pipeline to collect real-world, anonymized usage data. This data flywheel will allow us to train heavy cloud models, and then use knowledge distillation to transfer those insights back into our lightweight edge models.

## 3. LLM Integration for Supportive Messaging

**The Challenge:** To increase user retention, we want to generate dynamic, contextual "supportive messages." This requires Large Language Models (LLMs). However, small, edge-deployable LLMs (1B-3B parameters) are notoriously "dumb"—they struggle with strict prompt adherence, often ignoring injected context or hallucinating. Generating truly empathetic, context-aware advice currently demands heavier models that cannot run locally.

**The Mitigation:**

- **Hybrid Architecture:** We will adopt a tiered response system. When offline, the system will fall back to highly curated, deterministic text templates based on the Decision Engine's output. When online, the system can ping a secure, cloud-based LLM API (e.g., GPT-4o-mini or Gemini Flash) for nuanced, context-injected supportive messaging.
- **Future Local fine-tuning:** As edge-LLM quantization improves, we plan to use LoRA (Low-Rank Adaptation) to fine-tune a small local model _strictly_ for empathetic support, bypassing its lack of general reasoning.

## 4. Personalization & Decision Engine Robustness

**The Challenge:** The current Decision Engine is robust but static. Mental health interventions are highly user-specific; an intervention like "box breathing" might instantly calm User A, but severely agitate User B. A one-size-fits-all approach will eventually lead to user churn.

**The Mitigation:**

- **Personality-Dependent Routing:** The Decision Engine must evolve to incorporate a "User Personality Matrix." We will introduce implicit feedback loops (tracking which interventions the user completes vs. immediately dismisses).
- **Multi-Armed Bandit Integration:** By treating interventions as a contextual multi-armed bandit problem, the Decision Engine will dynamically adjust its weights. If a user consistently ignores "journaling" but engages with "sound therapy," the engine will adapt, creating a highly tailored feedback loop that maximizes long-term user engagement.
